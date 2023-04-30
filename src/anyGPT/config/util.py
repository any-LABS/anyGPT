import yaml

from anyGPT.config.settings import AnyGPTSettings


def read_config(filename):
    with open(filename, 'r') as f:
        return f.read()


def parse_config(yaml_string: str):
    try:
        return yaml.safe_load(yaml_string)
    except yaml.YAMLError as e:
        print("Error reading YAML config file.")


def config_to_settings(config):
    settings = None
    try:
        settings = AnyGPTSettings(**config)
    except TypeError as e:
        print("Error converting YAML to config.")

    return settings


def get_settings(config_file: str) -> AnyGPTSettings:
    config_file = read_config(config_file)
    parsed_config = parse_config(config_file)
    any_gpt_settings = config_to_settings(parsed_config)
    return any_gpt_settings
