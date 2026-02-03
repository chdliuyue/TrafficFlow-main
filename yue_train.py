from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import GradientClipping, EarlyStopping
from basicts.runners.callback import AddAuxiliaryLoss
# from basicts.runners.callback import NoBP
from basicts.models.MyModel_gpt import MyModel, MyModelConfig


def main():

    model_config = MyModelConfig(
        input_len=12,
        output_len=12,
        num_features=883,

        backbone_type="gru",
        backbone_layers=2,
        backbone_hidden_size=256,
        backbone_dropout=0.0,

        use_input_timestamps=True,
        use_output_timestamps=True,

        node_emb_dim=64,
        step_emb_dim=32,
        node_emb_dropout=0.1,
        node_bias=False,

        enable_dynamic_graph=True,
        graph_nonnegative_basis=False,
        graph_normalize=False,
        graph_rank=64,
        graph_alpha=0.1,
        graph_scale_hidden_size=256,
        graph_scale_dropout=0.1,

        reg_graph_orth=1e-4,
        reg_graph_l1=0.0,
        reg_graph_scale_smooth=0.0,

        fusion_learnable=True,
        fusion_mode="per_horizon",
        fusion_raw_init=-1.0,
        reg_fusion_l1=1e-4,

        enable_time_effect=True,
        time_tod_harmonics=4,
        time_dow_harmonics=2,
        time_coef_hidden=128,
        time_coef_dropout=0.1,

        likelihood="none",

        lambda_nll=0.0,
        dropout=0.0,

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
