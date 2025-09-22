import os
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# -------------------------
#  Categories
# -------------------------
categories = [
    "apple", "airplane", "banana", "cat", "dog",
    "car", "chair", "clock", "house", "tree"
]

# -------------------------
#  Download data if missing
# -------------------------
def download_if_missing(cat):
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{cat}.npy"
    filename = f"{cat}.npy"
    if not os.path.exists(filename):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, filename)
    else:
        print(f"{filename} already exists, skipping download.")

for cat in categories:
    download_if_missing(cat)

# -------------------------
#  Load and preprocess data
# -------------------------
def load_data(categories, samples_per_class=20000):  # larger subset
    X, y = [], []
    for idx, cat in enumerate(categories):
        data = np.load(f"{cat}.npy")
        data = data[:samples_per_class]
        X.append(data)
        y.append(np.full(samples_per_class, idx))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

X, y = load_data(categories)
X = X.reshape(-1,28,28,1).astype("float32") / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=len(categories))

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

# -------------------------
# Data augmentation (train only)
# -------------------------
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)
train_gen = datagen.flow(X_train, y_train, batch_size=64)

# -------------------------
#  Build CNN
# -------------------------
model = models.Sequential([
    layers.Input(shape=(28,28,1)),

    layers.Conv2D(32,(3,3),activation="relu",padding="same"),
    layers.Conv2D(32,(3,3),activation="relu",padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    layers.Conv2D(64,(3,3),activation="relu",padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256,activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(len(categories),activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------
# Train
# -------------------------
epochs = 20
model.fit(
    train_gen,
    validation_data=(X_val, y_val),
    epochs=epochs
)

# -------------------------
# Evaluate
# -------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc:.2f}")

# -------------------------
# Save model & categories
# -------------------------
model.save("draw_model_advanced.keras")
with open("categories.txt","w") as f:
    f.write("\n".join(categories))
