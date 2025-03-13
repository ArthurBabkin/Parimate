import sys
from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from internal.app import App


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config_path", required=True, type=str)
    args = arg_parser.parse_args()

    # load config
    print("load config")
    cfg = OmegaConf.load(args.config_path)

    # create app
    print("create app")
    app = App(cfg)

    # run app
    print("run app")
    app.run()


if __name__ == "__main__":
    main()
