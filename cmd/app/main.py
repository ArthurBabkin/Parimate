import asyncio
import sys
from pathlib import Path

from omegaconf import OmegaConf

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from internal.app.app import App

CONFIG_PATH = "config/config.yaml"


def main():
    # load config
    print("load config")
    cfg = OmegaConf.load(CONFIG_PATH)

    # create app
    print("create app")
    app = App(cfg)

    # run app
    print("run app")
    app.run()


if __name__ == "__main__":
    main()
