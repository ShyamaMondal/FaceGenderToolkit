# taskA_train_boosted.py
import os, random, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (Input, Dense, Dropout, Concatenate,
                                     GlobalAveragePooling2D, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        CSVLogger, ModelCheckpoint)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow_addons.losses import SigmoidFocalCrossEntropy  # pip install tensorflow-addons

# ╔══ 0. REPRODUCIBILITY ══════════════════════════════════════════════╗
SEED = 42
tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ╔══ 1. OPTIONAL MIXED PRECISION ═════════════════════════════════════╗
if tf.config.list_physical_devices('GPU'):
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("🎛️  Mixed precision enabled.")
    except:
        print("⚠️  Mixed precision not available, continuing without it.")

# ╔══ 2. PATHS & CONSTANTS ════════════════════════════════════════════╗
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, "../giveDataSet/Task_A/train")
val_dir   = os.path.join(BASE_DIR, "../giveDataSet/Task_A/val")
os.makedirs("models", exist_ok=True); os.makedirs("logs", exist_ok=True)

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32
EPOCHS     = 30

# ╔══ 3. DATA GENERATORS ══════════════════════════════════════════════╗
train_aug = ImageDataGenerator(
    rescale=1/255., rotation_range=25,
    width_shift_range=0.15, height_shift_range=0.15,
    zoom_range=0.25, shear_range=12, brightness_range=[0.4,1.4],
    horizontal_flip=True
)
val_aug   = ImageDataGenerator(rescale=1/255.)

train_ds = train_aug.flow_from_directory(
    train_dir, target_size=IMG_SIZE, color_mode="grayscale",
    class_mode="categorical", batch_size=BATCH_SIZE, shuffle=True, seed=SEED
)
val_ds = val_aug.flow_from_directory(
    val_dir, target_size=IMG_SIZE, color_mode="grayscale",
    class_mode="categorical", batch_size=BATCH_SIZE, shuffle=False
)

print("Classes:", train_ds.class_indices)

# ╔══ 4. CLASS WEIGHTS ════════════════════════════════════════════════╗
y = train_ds.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))
print("Class weights:", class_weight_dict)

# ╔══ 5. MODEL DEFINITION  (EfficientNet‑B0) ══════════════════════════╗
def build_model():
    inp = Input(shape=(128,128,1))
    x3  = Concatenate()([inp, inp, inp])          # replicate gray ➜ RGB
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=x3
    )
    base.trainable = True                         # end‑to‑end fine‑tune

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    out = Dense(2, activation="softmax", dtype="float32")(x)  # ensure float32 output
    return Model(inp, out)

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0),
    loss=SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0),
    metrics=["accuracy",
             Precision(name="precision"),
             Recall(name="recall"),
             AUC(name="balanced_auc", multi_label=True)]
)

# ╔══ 6. CALLBACKS ════════════════════════════════════════════════════╗
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2,
                      min_lr=1e-6, verbose=1),
    # EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True,
    #               verbose=1),
    ModelCheckpoint("models/best_gender_model.h5", monitor="val_accuracy",
                    save_best_only=True, mode="max", verbose=1),
    CSVLogger("logs/training_log.csv")
]

# ╔══ 7. TRAIN ═══════════════════════════════════════════════════════╗
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=callbacks, class_weight=class_weight_dict
)

# ╔══ 8. SAVE FINAL MODEL ════════════════════════════════════════════╗
model.save("TaskA_Model.h5")
print("Saved final model to models/final_gender_model.h5")

# ╔══ 9. PLOTS ═══════════════════════════════════════════════════════╗
def plot(metric):
    plt.figure()
    plt.plot(history.history[metric], label=f"Train {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"Val {metric}")
    plt.title(metric); plt.xlabel("Epoch"); plt.ylabel(metric)
    plt.legend()

for m in ["accuracy", "precision", "recall", "balanced_auc", "loss"]:
    plot(m)
plt.show()
