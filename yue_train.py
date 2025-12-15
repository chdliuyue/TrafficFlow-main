from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.GDGCN import GDGCN, GDGCNConfig


def main():

    model_config = GDGCNConfig(
        input_len=12,
        output_len=12,
        num_features=358,
    )


    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=GDGCN,
        input_len=12,
        output_len=12,
        model_config=model_config,
        dataset_name="PEMS03",
        metrics=["MAE", "MSE", "RMSE", "MAPE", "WAPE", "SMAPE", "R2", "CORR", "HUBER"],
        loss="MAE",
        gpus="0"
    ))


if __name__ == "__main__":
    main()
