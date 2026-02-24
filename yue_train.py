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
        backbone_type="gru", # gru transformer
        backbone_hidden_size=256, # 256
        backbone_layers=3, # 3
        backbone_dropout=0.0,
        backbone_tap_layer=-1,

        use_input_timestamps=False,  

        # ---- identity embeddings ----
        node_emb_dim=128, # 64
        step_emb_dim=128, # 64

        # ============================================================
        # Innovation #1: Spatial (low-rank, avoid N×N)
        # ============================================================
        enable_spatial=True,
        spatial_rank=96,
        spatial_alpha=0.1,
        spatial_scale_hidden=256, # 256
        spatial_scale_dropout=0.0,
        reg_spatial_orth=1e-4,
        spatial_use_output_timestamps=True,
        spatial_basis_normalize=True,

        # ============================================================
        # Innovation #2: Time (Spectral-Token Attention)
        # ============================================================
        enable_time=True,
        time_tod_harmonics=6,
        time_dow_harmonics=3,
        time_attn_dim=128, # 96
        time_alpha=1.0,
        time_gate_bound=1.0,
        time_attn_dropout=0.0,
        time_token_dropout=0.0,
        time_attn_temperature=1.0,

        # ---- convex fusion ----
        fusion_learnable=True,
        fusion_raw_spatial_init=-1.0,
        fusion_raw_time_init=-1.0,
        fusion_mode="adaptive",
        fusion_hidden=512, # 256
        fusion_dropout=0.0,
        fusion_use_step_embedding=True,

        # ============================================================
        # Innovation #3: Distribution fitting (Student-t)
        # ============================================================
        enable_distribution=True,  # ✅ 分布拟合消融开关
        dist_trunk_hidden=512,
        dist_trunk_layers=3,

        min_scale=0.01,  # sigma 下界（数值稳定很关键）
        studentt_df_init=5.0,
        studentt_df_min=2.1,
        studentt_df_max=30.0,

        # ---- decoder conditioning ----
        decoder_use_output_timestamps=True,
        enable_linear_skip=True,

        # ---- loss (inside forward) ----
        point_loss="mae",
        huber_delta=1.0,
        lambda_point=1.0,
        lambda_nll=1.0,
        nll_total_epochs=300,

        compute_loss_in_forward=True,

        # ---- outputs ----
        return_interpretation=True,
        return_components=True,

    )


    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=MyModel,
        input_len=12,
        output_len=12,
        model_config=model_config,
        dataset_name="PEMS07",
        # A100 40G profile: higher throughput + stable convergence.
        model_dtype="bfloat16",
        compile_model=True,
        train_batch_size=128,
        val_batch_size=128,
        test_batch_size=128,
        optimizer_params={"lr": 3e-4, "weight_decay": 1e-4},
        num_epochs=500,
        callbacks=[GradientClipping(1.0), EarlyStopping(30)],
        # callbacks=[AddAuxiliaryLoss(["aux_loss"])], # DUTE
        # callbacks = [NoBP()], # HI
        gpus="0",
        tf32=True,
        train_data_prefetch=True,
        train_data_num_workers=8,
        train_data_pin_memory=True,
        val_data_prefetch=True,
        val_data_num_workers=4,
        val_data_pin_memory=True,
        test_data_prefetch=True,
        test_data_num_workers=4,
        test_data_pin_memory=True,
        save_results=True,
    ))


if __name__ == "__main__":
    main()
