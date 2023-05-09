import argparse
import os
import re
from configparser import ConfigParser

parser = argparse.ArgumentParser(
    prog="gen_req",
    description="Generates requirement.txt from setup.cfg",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "package_path",
    action="store",
    default=".",
    nargs="?",
    help="The path and install options.",
)

parser.add_argument(
    "-o",
    "--output",
    action="store",
    help="The output file pathname. If none specified will write to stdout.",
)


def main():
    args = parser.parse_args()
    package_path = args.package_path.split("[")[0]
    extras_keys = re.findall("\[([^)]+)\]", args.package_path)
    config_path = os.path.join(os.path.abspath(package_path), "setup.cfg")
    if not os.path.exists(config_path):
        raise FileNotFoundError
    cf = ConfigParser()
    cf.read(config_path)
    req_str = cf["options"]["install_requires"]
    for key in extras_keys:
        try:
            req_str += cf["options.extras_require"][key]
        except KeyError as e:
            print(f"Extra requires key {key} not found.")
    if args.output is not None:
        with open(args.output, "w") as f:
            f.write(req_str)
    else:
        print(req_str)


if __name__ == "__main__":
    main()
