from ultralytics import YOLO

from . import config


def train():
    model = YOLO(config.BASE_WEIGHTS)
    return model.train(**config.TRAIN_HYPERPARAMS)


if __name__ == "__main__":
    train()
