import argparse

from anyGPT.inference.runner import AnyGPTRunner


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anyGPT inference",
        description="Loads an anyGPT model and runs inference given an input.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model", action="store", help="Path t0 the trained model checkpoint to load"
    )
    parser.add_argument("input", action="store", help="The input string")

    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    runner = AnyGPTRunner(args.model)
    print(runner.sample(args.input))


if __name__ == "__main__":
    main()
