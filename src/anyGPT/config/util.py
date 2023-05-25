import argparse
import functools
from typing import Any

import yaml

from anyGPT.config.settings import AnyGPTSettings


def read_config(filename):
    with open(filename, "r") as f:
        return f.read()


def parse_config(yaml_string: str) -> Any:
    try:
        return yaml.safe_load(yaml_string)
    except yaml.YAMLError:
        print("Error reading YAML config file.")


def config_to_settings(config: dict) -> AnyGPTSettings:
    settings = None
    try:
        settings = AnyGPTSettings(**config)
    except TypeError:
        print("Error converting YAML to config.")

    return settings


def get_settings(config_file: str) -> AnyGPTSettings:
    config_file = read_config(config_file)
    parsed_config = parse_config(config_file)
    any_gpt_settings = config_to_settings(parsed_config)
    return any_gpt_settings


def _create_parser(config_cls: object) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anyGPT trainer",
        description="Trains anyGPT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "trainer_config_path",
        action="store",
        help="The path to the training config file.",
    )

    for _ in config_cls.__annotations__.keys():
        # TODO add automatic arguments based on 1-level nested config dataclasses
        pass

    return parser


def anyfig(config_cls):
    parser = _create_parser(config_cls)
    arguments = parser.parse_args()
    settings = get_settings(arguments.trainer_config_path)

    def anyfig_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = func(settings, *args, **kwargs)
            return value

        return wrapper

    return anyfig_decorator
