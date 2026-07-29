import kagglehub

# Download latest version
path = kagglehub.dataset_download("mmohaiminulislam/vehicles-image-dataset")

print("Path to dataset files:", path)
import os

# Show files
for root, dirs, files in os.walk(path):
    print(root)
    print(dirs[:5])
    break
import splitfolders
import os
input_folder = os.path.join(path, "vehicle_data")
output_folder = os.path.join(path, "split_dataset")

# Show files in vehicle_data
print(f"Files in vehicle_data: {os.listdir(input_folder)[:10]}")

splitfolders.ratio(
    input=input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.8, 0.1, 0.1),
    group_prefix=None,
    move=False
)
from PIL import Image
import os

dataset_dir = output_folder

bad = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
            p = os.path.join(root, file)

            try:
                img = Image.open(p)
                if img.mode != "RGB":
                    bad.append((p, img.mode))
            except Exception as e:
                bad.append((p, str(e)))

print("Bad Images:", len(bad))

for b in bad[:50]:
    print(b)
from PIL import Image
import os

dataset_dir = os.path.join(path, "split_dataset")

converted = 0
failed = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            img_path = os.path.join(root, file)

            try:
                img = Image.open(img_path)

                # Convert every non-RGB image
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    img.save(img_path)
                    converted += 1

            except Exception as e:
                failed.append((img_path, str(e)))

print(f"Converted: {converted}")
print(f"Failed: {len(failed)}")

if failed:
    print(failed[:10])
for folder in ["train", "val", "test"]:
    print(f"\n{folder.upper()}")
    print(os.listdir(os.path.join(output_folder, folder))[:5])
train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")
test_dir = os.path.join(output_folder, "test")
import os
import tensorflow as tf

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

print("Number of Classes:", len(class_names))
print("Classes:", class_names)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Dropout, Rescaling,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast,
    RandomTranslation
)
from tensorflow.keras.regularizers import l2

num_classes = len(class_names)


model = Sequential([
    Input(shape=(128, 128, 3)),

    # Augmentation
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),

    Rescaling(1./255),

    # Block 1
    Conv2D(32, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),

    # Block 2
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),

    # Block 4
    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.35),

    GlobalAveragePooling2D(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("\n🚀 Training started...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Evaluate on test set
loss, accuracy = model.evaluate(test_ds)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
model.save("cars_cnn.keras")
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

def predict_image(img_path, model, class_names, img_size=128):
    # Load and resize image
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    # Show image with prediction
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.show()

    return predicted_class, confidence


# Usage: Dynamically find an image from the 'bicycle' class in the test dataset
bicycle_test_dir = os.path.join(test_dir, 'bicycle')

# Get a list of image files in the 'bicycle' directory
bicycle_images = [f for f in os.listdir(bicycle_test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

if bicycle_images:
    # Use the first image found as a sample
    sample_bike_path = os.path.join(bicycle_test_dir, bicycle_images[0])
    predicted_class, confidence = predict_image(sample_bike_path, model, class_names)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}%")
else:
    print(f"No image files found in {bicycle_test_dir}. Please ensure the directory contains images or specify a valid path.")
