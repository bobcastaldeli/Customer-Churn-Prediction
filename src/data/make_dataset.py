"""Get data from kaggle."""


import configparser
import logging
import subprocess
from kaggle.api import KaggleApi


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def download_data():
    """Authenticate and downloading train and test data from Kaggle Titanic
    competition."""
    config = configparser.ConfigParser()
    config.read("configs.ini")
    name = config["datasets"]["name"]
    output_dir = config["datasets"]["raw_folder"]

    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading train data")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            f"{name}",
            "--p",
            f"{output_dir}",
            "--force",
        ],
        check=True,
    )


if __name__ == "__main__":
    download_data()
