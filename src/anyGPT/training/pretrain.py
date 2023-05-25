from anyGPT.config.settings import AnyGPTSettings
from anyGPT.config.util import anyfig
from anyGPT.training.trainers import AnyGPTPreTrainer


def pretrain(settings: AnyGPTSettings) -> None:
    trainer = AnyGPTPreTrainer(settings)
    trainer.fit()


@anyfig(AnyGPTSettings)
def main(settings):
    pretrain(settings)


if __name__ == "__main__":
    main()
