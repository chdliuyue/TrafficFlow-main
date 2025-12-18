from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.SOFTS import SOFTS, SOFTSConfig


def main():

    model_config = SOFTSConfig(
        input_len=12,
        output_len=12,
        # label_len=6,
        num_features=358,
        # seg_len=4,
        # use_timestamp=True,
        # timestamp_sizes=[288, 7]
    )


    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=SOFTS,
        input_len=12,
        output_len=12,
        model_config=model_config,
        dataset_name="PEMS03",
        # loss="MAE",
        # callbacks=[AddAuxiliaryLoss(["aux_loss"])],
        gpus="0",
        # compile_model=True,
        # train_data_prefetch=True,
        # train_data_num_workers=4,
        # train_data_pin_memory=True,
        # val_data_prefetch=True,
        # val_data_num_workers=4,
        # val_data_pin_memory=True,
        # test_data_prefetch=True,
        # test_data_num_workers=4,
        # test_data_pin_memory=True,
    ))


if __name__ == "__main__":
    main()
