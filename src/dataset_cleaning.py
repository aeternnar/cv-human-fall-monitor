import os
import random

import cv2
import imagehash
from PIL import Image
from tqdm import tqdm

from . import config


def find_duplicates_across_splits(train_dir, test_dir):
    train_hashes = {}
    for img_path in tqdm(os.listdir(train_dir)):
        if img_path.endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(train_dir, img_path)
            try:
                with Image.open(full_path) as img:
                    img_hash = str(imagehash.average_hash(img))
                    train_hashes[img_hash] = full_path
            except Exception as e:
                print(f"Ошибка обработки {full_path}: {e}")

    duplicates = []
    for img_path in tqdm(os.listdir(test_dir)):
        if img_path.endswith(('.jpg', '.jpeg', '.png')):
            full_path = os.path.join(test_dir, img_path)
            try:
                with Image.open(full_path) as img:
                    test_hash = str(imagehash.average_hash(img))
                    if test_hash in train_hashes:
                        duplicates.append((full_path, train_hashes[test_hash]))
            except Exception as e:
                print(f"Ошибка обработки {full_path}: {e}")

    return duplicates


def show_duplicates(test_path, train_path):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    test_img = Image.open(test_path)
    ax1.imshow(test_img)
    ax1.set_title(f'SPLIT1: {test_path.split("/")[-1]}', size=10)
    ax1.axis('off')

    train_img = Image.open(train_path)
    ax2.imshow(train_img)
    ax2.set_title(f'SPLIT2: {train_path.split("/")[-1]}', size=10)
    ax2.axis('off')

    plt.tight_layout()


def show_duplicates_for_split(duplicates, n):
    for split1_img, split2_img in duplicates[:n]:
        show_duplicates(split1_img, split2_img)


def clean_dataset(duplicates):
    removed_files = []

    for file_to_remove, file_to_keep in duplicates:
        if not os.path.exists(file_to_remove):
            print(f"Файл не существует, пропускаем: {file_to_remove}")
            continue

        os.remove(file_to_remove)

        txt_to_remove = file_to_remove.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if os.path.exists(txt_to_remove):
            os.remove(txt_to_remove)

        removed_files.append((file_to_remove, file_to_keep))

    print(f'Удалено {len(removed_files)} файлов')
    return removed_files


def find_and_remove_duplicates():
    train_val_duplicates = find_duplicates_across_splits(config.TRAIN_IMAGES_DIR, config.VAL_IMAGES_DIR)
    train_test_duplicates = find_duplicates_across_splits(config.TRAIN_IMAGES_DIR, config.TEST_IMAGES_DIR)
    val_test_duplicates = find_duplicates_across_splits(config.VAL_IMAGES_DIR, config.TEST_IMAGES_DIR)

    for name, duplicates in [
        ("train/val", train_val_duplicates),
        ("train/test", train_test_duplicates),
        ("val/test", val_test_duplicates),
    ]:
        if duplicates:
            print(f"Найдено {len(duplicates)} повторяющихся изображений между {name}")

    return (
        clean_dataset(train_val_duplicates),
        clean_dataset(train_test_duplicates),
        clean_dataset(val_test_duplicates),
    )


def visualize_annotations(image_dir=config.TRAIN_IMAGES_DIR, label_dir=config.TRAIN_LABELS_DIR,
                           class_names=config.CLASS_NAMES):
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in random.sample(images, 10):
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                data = line.strip().split()
                if len(data) == 5:
                    class_id, x_center, y_center, width, height = map(float, data)

                    h, w = image.shape[:2]
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, class_names[int(class_id)], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Annotation Check', image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_and_remove_duplicates()
