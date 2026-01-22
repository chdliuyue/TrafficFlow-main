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
        num_features=358,
        use_input_timestamps=True,
        use_output_timestamps=True,
        likelihood="gaussian",
        # timestamp_sizes=[288, 7]
    )


    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=MyModel,
        input_len=12,
        output_len=12,
        model_config=model_config,
        dataset_name="PEMS03",
        # loss="MAE",
        num_epochs=300,
        callbacks=[GradientClipping(1.0), EarlyStopping(20)],
        # callbacks=[AddAuxiliaryLoss(["aux_loss"])], # DUTE
        # callbacks = [NoBP()], # HI
        gpus="0",
        tf32=True,
        # train_data_prefetch=True,
        # train_data_num_workers=2,
        # train_data_pin_memory=True,
        # val_data_prefetch=True,
        # val_data_num_workers=2,
        # val_data_pin_memory=True,
        # test_data_prefetch=True,
        # test_data_num_workers=2,
        # test_data_pin_memory=True,
    ))


if __name__ == "__main__":
    main()
