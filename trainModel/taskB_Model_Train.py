import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ✅ GPU configuration
print("\n🔍 Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Using GPU(s): {[gpu.name for gpu in gpus]}")
else:
    print("⚠️ No GPU detected. Using CPU.")

# ✅ Paths and parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../giveDataSet/Task_B/train")
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

# ✅ Load dataset with validation split
train_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.15,
    subset="training",
    seed=42,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.15,
    subset="validation",
    seed=42,
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# ✅ Save class names
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"📦 Classes: {class_names}")

os.makedirs("models", exist_ok=True)
class_names_path = os.path.join("models", "class_names.json")
with open(class_names_path, "w") as f:
    json.dump(class_names, f)
print(f"✅ Class names saved to {class_names_path}")

# ✅ Performance optimization
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ✅ Augmentation
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomRotation(0.1),
    layers.GaussianNoise(0.1),
])

# ✅ Model architecture (ResNet50)
def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = augmentation(inputs)
    x = tf.keras.applications.resnet.preprocess_input(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    base_model.trainable = True

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# ✅ Build and compile model
model = build_model((*IMG_SIZE, 3), num_classes)
model.compile(
    optimizer=optimizers.Adam(1e-4),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()

# ✅ Callbacks
os.makedirs("logs", exist_ok=True)
callback_list = [
    # callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint("models/best_face_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1),
    callbacks.CSVLogger("logs/face_training_log.csv")
]

# ✅ Train the model
print("\n🚻 Training Face Recognition Model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callback_list
)

# ✅ Save final model
model.save("TaskB_Model.h5")
print("✅ Final model saved as models/final_face_model.h5")
