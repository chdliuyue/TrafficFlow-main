from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.DLinear import DLinear, DLinearConfig


def main():

    model_config = DLinearConfig(input_len=12, output_len=12)

    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name="PEMS03",
        gpus="0"
    ))


if __name__ == "__main__":
    main()
