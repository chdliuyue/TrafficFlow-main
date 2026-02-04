from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import GradientClipping, EarlyStopping
from basicts.runners.callback import AddAuxiliaryLoss
# from basicts.runners.callback import NoBP
from basicts.models.MyModel_gpt import MyModel, MyModelConfig


def main():

    model_config = MyModelConfig(
        # required
        input_len=12,
        output_len=12,
        num_features=883,
        num_timestamps=2,

        last_value_centering=True,

        # backbone: 下一阶段建议更强（先跑这一条主线）
        backbone_type="gru",
        backbone_hidden_size=320,
        backbone_layers=3,
        backbone_dropout=0.1,
        backbone_tap_layer=1,  # 3层时取中间层作为并行分支入口（符合“中间层并行”）

        use_input_timestamps=False,  # 你此前已经证明 ts_out 更关键，这里建议先关掉做对照

        # embeddings（你已经证实是强增益项）
        node_emb_dim=64,
        step_emb_dim=32,
        dropout=0.1,

        # spatial branch（低秩降维，避免 N×N）
        enable_spatial=True,
        spatial_rank=64,
        spatial_alpha=0.1,
        spatial_scale_hidden=256,
        spatial_scale_dropout=0.1,
        reg_spatial_orth=1e-4,
        spatial_use_output_timestamps=True,

        # time branch（创新：谱token注意力）
        enable_time=True,
        time_tod_harmonics=4,
        time_dow_harmonics=2,
        time_attn_dim=64,
        time_alpha=1.0,
        time_gate_bound=1.0,

        # convex fusion（默认学习）
        fusion_learnable=True,
        fusion_raw_spatial_init=-1.0,
        fusion_raw_time_init=-1.0,

        # decoder
        decoder_mlp_hidden=256,
        decoder_use_output_timestamps=True,
        enable_linear_skip=True,

        # distribution + loss（Part C先温和一点）
        likelihood="studentt",
        min_scale=0.01,
        studentt_df_mode="learned_global",
        studentt_df_init=10.0,
        studentt_df_min=2.1,

        point_loss="mae",
        lambda_point=1.0,
        lambda_nll=0.05,  # 建议从 0.05 起跑；如果点指标下降，就降到 0.0~0.02
        compute_loss_in_forward=True,

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
        callbacks=[EarlyStopping(30)],
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
