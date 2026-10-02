import os

from ldr.core.trainer import Trainer
from ldr.common.utils.params_manager import ParamsManager


def main():
    config_path = os.path.abspath(os.path.join(__file__, "../../../configs/user_config.yaml"))
    config_basedir_path = os.path.dirname(config_path)
    # change relative path to absolute
    config_file = ParamsManager.load(config_path)
    config_file["agent"] = os.path.join(config_basedir_path, config_file["agent"])
    config_file["environment"] = os.path.join(config_basedir_path, config_file["environment"])

    trainer = Trainer(config_file=config_file)
    trainer.train()


if __name__ == "__main__":
    main()
