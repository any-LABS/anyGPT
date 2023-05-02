from anyGPT.config.settings import AnyGPTSettings
from anyGPT.config.util import anyfig
from anyGPT.training.trainer import AnyGPTTrainer


def train(settings: AnyGPTSettings):
    trainer = AnyGPTTrainer(settings)
    trainer.fit()


@anyfig(AnyGPTSettings)
def main(settings):
    train(settings)


if __name__ == "__main__":
    main()
