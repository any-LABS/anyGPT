import argparse
from anyGPT.config.settings import AnyGPTSettings
from anyGPT.config.util import get_settings
from anyGPT.training.trainer import AnyGPTTrainer


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='anyGPT trainer',
        description='Trains anyGPT.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('trainer_config_path', action='store', help="The path to the training config file.")
    return parser


def train(settings: AnyGPTSettings):
    trainer = AnyGPTTrainer(settings)
    trainer.fit()


def main():
    parser = _create_parser()
    args = parser.parse_args()
    settings = get_settings(args.trainer_config_path)
    train(settings)


if __name__ == "__main__":
    main()
