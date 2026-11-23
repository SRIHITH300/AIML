import os
import pickle
import tarfile
import urllib.request
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

ARCHIVE_PATH = os.path.join(
    DATA_DIR, "cifar-10-python.tar.gz"
)

EXTRACT_DIR = os.path.join(
    DATA_DIR, "cifar-10-batches-py"
)

MODEL_PATH = os.path.join(
    MODEL_DIR, "best_cifar10_model.keras"
)


# 3. DOWNLOAD AND EXTRACT DATASET


def download_dataset():

    if not os.path.exists(EXTRACT_DIR):

        if not os.path.exists(ARCHIVE_PATH):

            print("Downloading CIFAR-10 dataset...")

            urllib.request.urlretrieve(
                DATA_URL,
                ARCHIVE_PATH
            )

            print("Download completed!")

        print("Extracting dataset...")

        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

        print("Dataset extracted successfully!")



# 4. LOAD CIFAR-10 PICKLE FILES


def load_batch(file_path):

    with open(file_path, "rb") as file:

        batch = pickle.load(
            file,
            encoding="bytes"
        )

    images = batch[b"data"]
    labels = batch[b"labels"]

    images = images.reshape(
        -1, 3, 32, 32
    )

    images = images.transpose(
        0, 2, 3, 1
    )

    return images, np.array(labels)


def load_dataset():

    download_dataset()

    train_images = []
    train_labels = []

    for i in range(1, 6):

        file_path = os.path.join(
            EXTRACT_DIR,
            f"data_batch_{i}"
        )

        images, labels = load_batch(file_path)

        train_images.append(images)
        train_labels.append(labels)

    X_train = np.concatenate(train_images)
    y_train = np.concatenate(train_labels)

    test_path = os.path.join(
        EXTRACT_DIR,
        "test_batch"
    )

    X_test, y_test = load_batch(test_path)

    return X_train, y_train, X_test, y_test


print("\nLoading CIFAR-10 dataset...")

X_train, y_train, X_test, y_test = load_dataset()

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)