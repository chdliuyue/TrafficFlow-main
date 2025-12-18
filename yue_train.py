from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.NonstationaryTransformer import NonstationaryTransformerConfig, NonstationaryTransformerForForecasting


def main():

    model_config = NonstationaryTransformerConfig(
        input_len=12,
        output_len=12,
        label_len=6,
        num_features=358,
        use_timestamp=True,
        timestamp_sizes=[288, 7]
    )


    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=NonstationaryTransformerForForecasting,
        input_len=12,
        output_len=12,
        model_config=model_config,
        dataset_name="PEMS03",
        metrics=["MAE", "MSE", "RMSE", "MAPE", "WAPE", "SMAPE", "R2", "CORR", "HUBER"],
        loss="MAE",
        # callbacks=[AddAuxiliaryLoss(["aux_loss"])],
        gpus="0"
    ))


if __name__ == "__main__":
    main()
