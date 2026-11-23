import os
import numpy as np
import tensorflow as tf

from dataset import load_dataset

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

os.makedirs("models", exist_ok=True)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.0417),
    layers.RandomTranslation(
        height_factor=0.1,
        width_factor=0.1
    )
])

X_train, y_train, X_test, y_test = load_dataset()

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(32, 32, 3))

# Augmentation
x = data_augmentation(inputs)

# Resize CIFAR-10 images from 32x32 to 96x96
x = layers.Resizing(96, 96)(x)

# Convert normalized [0,1] pixels to [-1,1]
x = layers.Rescaling(
    scale=2.0,
    offset=-1.0
)(x)

# Extract pretrained features
x = base_model(x, training=False)

# New classification head
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    10,
    activation="softmax"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs
)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_stage1 = ModelCheckpoint(
    "models/mobilenetv2_stage1.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history1 = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=64,
    callbacks=[
        checkpoint_stage1,
        early_stop
    ]
)

base_model.trainable = True

# Freeze all layers except the last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_stage2 = ModelCheckpoint(
    "models/mobilenetv2_finetuned.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

history2 = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=64,
    callbacks=[
        checkpoint_stage2,
        early_stop
    ]
)

best_model = tf.keras.models.load_model(
    "models/mobilenetv2_finetuned.keras"
)

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")


from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

predictions = best_model.predict(X_test)

# Predicted class IDs
y_pred = np.argmax(predictions, axis=1)

# Actual class IDs (already integers)
y_true = y_test.reshape(-1)

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=classes
))

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

cm = confusion_matrix(y_true, y_pred)
print(cm)