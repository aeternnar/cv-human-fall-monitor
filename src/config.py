SEED = 42

DATASET_DIR = "./dataset"
DATASET_YAML = f"{DATASET_DIR}/data.yaml"

TRAIN_IMAGES_DIR = f"{DATASET_DIR}/train/images"
TRAIN_LABELS_DIR = f"{DATASET_DIR}/train/labels"
VAL_IMAGES_DIR = f"{DATASET_DIR}/val/images"
TEST_IMAGES_DIR = f"{DATASET_DIR}/test/images"

CLASS_NAMES = ["falling"]

BASE_WEIGHTS = "./models/yolo11s.pt"
TRAIN_RUN_DIR = "runs/detect/train5"
BEST_WEIGHTS = f"{TRAIN_RUN_DIR}/weights/best.pt"
RESULTS_CSV = f"{TRAIN_RUN_DIR}/results.csv"

TRAIN_HYPERPARAMS = dict(
    data=DATASET_YAML,
    epochs=200,
    patience=30,
    imgsz=768,
    batch=16,
    device="cuda",
    verbose=False,
    pretrained=True,
    val=True,
    save_period=10,
    plots=True,
    augment=True,
    mosaic=0.25,
    mixup=0.0,
    degrees=5.0,
    translate=0.20,
    hsv_h=0.015,
    hsv_s=0.3,
    hsv_v=0.4,
    shear=2.0,
    box=4.0,
    cls=0.8,
)

VAL_PARAMS = dict(
    data=DATASET_YAML,
    split="test",
    batch=16,
    conf=0.25,
    iou=0.5,
    device="cuda",
    save_json=True,
    save_conf=True,
    plots=True,
)

PREDICT_PARAMS = dict(
    save=True,
    show=True,
    device="cuda",
    conf=0.4,
    iou=0.5,
    verbose=False,
)
