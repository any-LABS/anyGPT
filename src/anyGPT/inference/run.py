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

    parser.add_argument(
        "-m", "--max_new_tokens", action="store", default=500, help="The temperature"
    )
    parser.add_argument(
        "-t", "--temperature", action="store", default=0.8, help="The temperature"
    )
    parser.add_argument(
        "-k",
        "--top_k",
        action="store",
        default=200,
        help="The top k most likely tokens to retain",
    )

    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    runner = AnyGPTRunner(args.model)
    kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
    }
    print(runner.sample(args.input, **kwargs))


if __name__ == "__main__":
    main()
