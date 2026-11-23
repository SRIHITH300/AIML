
import os
import pickle
import tarfile
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


# ==========================================
# 2. CIFAR-10 CLASS NAMES
# ==========================================

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ==========================================
# 3. DOWNLOAD AND EXTRACT DATASET
# ==========================================

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


# ==========================================
# 4. LOAD CIFAR-10 PICKLE FILES
# ==========================================

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


# ==========================================
# 5. DISPLAY RANDOM IMAGES
# ==========================================

plt.figure(figsize=(12, 6))

for i in range(10):

    index = np.random.randint(len(X_train))

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_train[index])

    plt.title(classes[y_train[index]])

    plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "sample_images.png")
)

plt.show()


# ==========================================
# 6. NORMALIZATION
# ==========================================

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print("\nNormalization completed!")


# ==========================================
# 7. TRAIN AND VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ==========================================
# 8. ONE-HOT ENCODING
# ==========================================

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)


# ==========================================
# 9. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.0417),

    layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    )

], name="data_augmentation")


# ==========================================
# 10. CNN MODEL ARCHITECTURE
# ==========================================

model = models.Sequential([

    layers.Input(shape=(32, 32, 3)),

    data_augmentation,

    # BLOCK 1
    layers.Conv2D(
        32, (3, 3), padding="same"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(
        32, (3, 3), padding="same"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),


    # BLOCK 2
    layers.Conv2D(
        64, (3, 3), padding="same"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(
        64, (3, 3), padding="same"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),


    # BLOCK 3
    layers.Conv2D(
        128, (3, 3), padding="same"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.Conv2D(
        128, (3, 3), padding="same"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),


    # BLOCK 4
    layers.Conv2D(
        128, (3, 3), padding="same"
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),

    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),


    # CLASSIFICATION LAYERS
    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        10,
        activation="softmax"
    )

])


# ==========================================
# 11. COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

model.summary()


# ==========================================
# 12. SAVE BEST MODEL CHECKPOINT
# ==========================================

checkpoint = ModelCheckpoint(

    filepath=MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)


# ==========================================
# 13. TRAIN MODEL
# ==========================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=25,

    batch_size=64,

    callbacks=[checkpoint]

)


# ==========================================
# 14. PLOT ACCURACY
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.legend()
plt.grid()

plt.savefig(
    os.path.join(OUTPUT_DIR, "accuracy.png")
)

plt.show()


# ==========================================
# 15. PLOT LOSS
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.legend()
plt.grid()

plt.savefig(
    os.path.join(OUTPUT_DIR, "loss.png")
)

plt.show()


# ==========================================
# 16. LOAD BEST MODEL
# ==========================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ==========================================
# 17. TEST MODEL
# ==========================================

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)


# ==========================================
# 18. PREDICTIONS
# ==========================================

y_pred = best_model.predict(X_test)

y_pred_classes = np.argmax(
    y_pred,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)


# ==========================================
# 19. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred_classes,
        target_names=classes
    )
)


# ==========================================
# 20. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred_classes
)

plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation="nearest")

plt.title("CIFAR-10 Confusion Matrix")

plt.colorbar()

plt.xticks(
    range(10),
    classes,
    rotation=45
)

plt.yticks(
    range(10),
    classes
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png")
)

plt.show()


print("\nTraining and evaluation completed!")

print("Best model saved at:", MODEL_PATH)
