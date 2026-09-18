import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from . import config


def validate(weights=config.BEST_WEIGHTS):
    best_model = YOLO(weights)
    return best_model.val(**config.VAL_PARAMS)


def load_results(results_csv=config.RESULTS_CSV):
    return pd.read_csv(results_csv)


def plot_precision_recall(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')

    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Precision and Recall Dynamics')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()


def plot_map(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')

    plt.xlabel('Epoch')
    plt.ylabel('mAP Value')
    plt.title('mAP50 and mAP50-95 Dynamics')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()


if __name__ == "__main__":
    validate()
    results_df = load_results()
    plot_precision_recall(results_df)
    plot_map(results_df)
    plt.show()
