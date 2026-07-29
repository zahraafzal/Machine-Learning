import os
import kagglehub
import splitfolders
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    GlobalAveragePooling2D, Dense, Dropout, Rescaling,
    RandomFlip, RandomRotation, RandomZoom
)
from tensorflow.keras.preprocessing import image

# Configuration
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

print("=" * 60)
print("VEHICLE CLASSIFICATION - CNN TRAINING")
print("=" * 60)

# Step 1: Download Dataset
print("\n[1/7] Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("mmohaiminulislam/vehicles-image-dataset")
print(f"✓ Dataset downloaded: {path}")

# Step 2: Dataset Structure
print("\n[2/7] Checking dataset structure...")
for root, dirs, files in os.walk(path):
    print(f"Root: {root}")
    print(f"Subdirectories: {dirs[:5]}")
    break

# Step 3: Split Dataset
print("\n[3/7] Splitting dataset into train/val/test...")
input_folder = os.path.join(path, "vehicle_data")
output_folder = os.path.join(path, "split_dataset")

print(f"Categories: {os.listdir(input_folder)[:10]}")

splitfolders.ratio(
    input=input_folder,
    output=output_folder,
    seed=42,
    ratio=(0.8, 0.1, 0.1),
    group_prefix=None,
    move=False
)
print("✓ Dataset split complete")

# Step 4: Check and Convert Images
print("\n[4/7] Checking image formats...")
dataset_dir = output_folder
bad_images = []
converted = 0
failed = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):
            img_path = os.path.join(root, file)
            try:
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    img.save(img_path)
                    converted += 1
            except Exception as e:
                failed.append((img_path, str(e)))

print(f"✓ Converted {converted} images to RGB")
print(f"✓ Failed: {len(failed)}")

# Step 5: Load Dataset
print("\n[5/7] Loading datasets...")
train_dir = os.path.join(output_folder, "train")
val_dir = os.path.join(output_folder, "val")
test_dir = os.path.join(output_folder, "test")

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

print(f"✓ Number of Classes: {len(class_names)}")
print(f"✓ Classes: {class_names}")

# Step 6: Build and Train Model
print("\n[6/7] Building CNN model...")
num_classes = len(class_names)

model = Sequential([
    Input(shape=(128, 128, 3)),
    
    # Data Augmentation
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    Rescaling(1./255),
    
    # Convolutional Blocks
    Conv2D(32, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.2),
    
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),
    
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.3),
    
    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.35),
    
    # Dense Layers
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

print("\n🚀 Training started...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Step 7: Evaluate and Save
print("\n[7/7] Evaluating model...")
loss, accuracy = model.evaluate(test_ds)
print(f"✓ Test Loss: {loss:.4f}")
print(f"✓ Test Accuracy: {accuracy:.4f}")

model.save("cars_cnn.keras")
print("\n✓ Model saved as 'cars_cnn.keras'")

# Test Prediction
print("\n" + "=" * 60)
print("TESTING PREDICTION")
print("=" * 60)

def predict_image(img_path, model, class_names, img_size=128):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.show()
    
    return predicted_class, confidence

# Test with bicycle image
bicycle_test_dir = os.path.join(test_dir, 'bicycle')
bicycle_images = [f for f in os.listdir(bicycle_test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if bicycle_images:
    sample_bike_path = os.path.join(bicycle_test_dir, bicycle_images[0])
    predicted_class, confidence = predict_image(sample_bike_path, model, class_names)
    print(f"✓ Predicted: {predicted_class}, Confidence: {confidence:.2f}%")
else:
    print("✗ No test images found")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

