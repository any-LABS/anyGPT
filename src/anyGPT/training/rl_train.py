from anyGPT.config.settings import AnyGPTSettings
from anyGPT.config.util import anyfig
from anyGPT.training.trainers import AnyGPTPPOTrainer


def rl_train(settings: AnyGPTSettings) -> None:
    trainer = AnyGPTPPOTrainer(settings)
    trainer.fit()


@anyfig(AnyGPTSettings)
def main(settings):
    rl_train(settings)


if __name__ == "__main__":
    main()
