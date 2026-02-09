from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import GradientClipping, EarlyStopping
from basicts.runners.callback import AddAuxiliaryLoss
# from basicts.runners.callback import NoBP
from basicts.models.MyModel_gpt import MyModel, MyModelConfig


def main():

    model_config = MyModelConfig(
        # ---- required ----
        input_len=12,
        output_len=12,
        num_features=883,
        num_timestamps=2,
        timestamp_sizes=(288, 7),  # 仅 meta，不参与 embedding（你的时间戳已归一到[0,1]）

        # ---- preprocessing ----
        last_value_centering=True,

        # ---- backbone (best-like) ----
        backbone_type="gru",
        backbone_hidden_size=384,
        backbone_layers=3,
        backbone_dropout=0.15,
        backbone_tap_layer=-1,  # 你实验2更优：取最后层作为并行分支输入 H
        use_input_timestamps=False,  # 你实验更优：不把 ts_in 喂给 backbone

        # ---- identity embeddings ----
        node_emb_dim=64,
        step_emb_dim=32,
        dropout=0.1,

        # ============================================================
        # Innovation #1: Spatial (low-rank, avoid N×N)
        # ============================================================
        enable_spatial=True,
        spatial_rank=96,
        spatial_alpha=0.2,
        spatial_scale_hidden=320,
        spatial_scale_dropout=0.1,
        reg_spatial_orth=1e-4,
        spatial_use_output_timestamps=True,
        spatial_basis_normalize=True,

        # ============================================================
        # Innovation #2: Time (Spectral-Token Attention)
        # ============================================================
        enable_time=True,
        time_tod_harmonics=6,
        time_dow_harmonics=3,
        time_attn_dim=96,
        time_alpha=1.2,
        time_gate_bound=1.0,
        time_attn_dropout=0.1,
        time_token_dropout=0.1,
        time_attn_temperature=1.1,

        # ---- convex fusion ----
        fusion_learnable=True,
        fusion_raw_spatial_init=-0.8,
        fusion_raw_time_init=-0.8,
        fusion_mode="adaptive",
        fusion_hidden=128,
        fusion_dropout=0.1,
        fusion_use_step_embedding=True,

        # ============================================================
        # Innovation #3: Distribution fitting (Student-t)
        # ============================================================
        enable_distribution=True,  # ✅ 分布拟合消融开关
        dist_trunk_hidden=320,
        dist_trunk_layers=2,

        min_scale=0.01,  # sigma 下界（数值稳定很关键）
        studentt_df_mode="learned_from_features",  # df 由网络学习（b/o/n 可变）
        studentt_df_init=10.0,
        studentt_df_min=2.1,
        studentt_df_max=60.0,

        # ---- decoder conditioning ----
        decoder_use_output_timestamps=True,
        enable_linear_skip=True,

        # ---- loss (inside forward) ----
        point_loss="huber",
        huber_delta=1.5,
        lambda_point=1.0,
        lambda_nll=0.05,  # 你实验2更优：NLL 权重小
        nll_warmup_steps=1000,

        compute_loss_in_forward=True,

        # ---- outputs ----
        return_interpretation=False,
        return_components=False,

    )


    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=MyModel,
        input_len=12,
        output_len=12,
        model_config=model_config,
        dataset_name="PEMS07",
        # loss="MAE",
        num_epochs=300,
        callbacks=[GradientClipping(1.0), EarlyStopping(30)],
        # callbacks=[AddAuxiliaryLoss(["aux_loss"])], # DUTE
        # callbacks = [NoBP()], # HI
        gpus="0",
        tf32=True,
        train_data_prefetch=True,
        train_data_num_workers=2,
        train_data_pin_memory=True,
        val_data_prefetch=True,
        val_data_num_workers=2,
        val_data_pin_memory=True,
        test_data_prefetch=True,
        test_data_num_workers=2,
        test_data_pin_memory=True,
    ))


if __name__ == "__main__":
    main()
