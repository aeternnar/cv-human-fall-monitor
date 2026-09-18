from ultralytics import YOLO

from . import config


def predict(source, weights=config.BEST_WEIGHTS):
    model = YOLO(weights)
    return model.predict(source=source, **config.PREDICT_PARAMS)


if __name__ == "__main__":
    predict(source="dataset/final_test_vid.mp4")
