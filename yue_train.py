from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DUET import DUET, DUETConfig
from basicts.runners.callback import AddAuxiliaryLoss


def main():

    model_config = DUETConfig(
        input_len=12,
        output_len=12,
        num_features=358,
        # use_timestamp=True,
        # timestamp_sizes=[288, 7]
    )


    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=DUET,
        input_len=12,
        output_len=12,
        model_config=model_config,
        dataset_name="PEMS03",
        metrics=["MAE", "MSE", "RMSE", "MAPE", "WAPE", "SMAPE", "R2", "CORR", "HUBER"],
        loss="MAE",
        callbacks=[AddAuxiliaryLoss(["aux_loss"])],
        gpus="0"
    ))


if __name__ == "__main__":
    main()
