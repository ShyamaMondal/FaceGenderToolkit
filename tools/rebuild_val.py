import os
import shutil
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, "../giveDataSet/Task_B/train")
val_dir = os.path.join(BASE_DIR, "../giveDataSet/Task_B/val")


max_images_per_class = 2  # You can change this if you want more validation images per class

# Step 1: Delete existing val directory
if os.path.exists(val_dir):
    print("🧹 Deleting existing val/ folder...")
    shutil.rmtree(val_dir)

# Step 2: Recreate val directory and copy structure from train
print("🔄 Rebuilding val/ from train/...")
os.makedirs(val_dir, exist_ok=True)

for person_folder in tqdm(os.listdir(train_dir), desc="Processing persons"):
    person_train_path = os.path.join(train_dir, person_folder)
    person_val_path = os.path.join(val_dir, person_folder)

    if not os.path.isdir(person_train_path):
        continue

    os.makedirs(person_val_path, exist_ok=True)

    images = [img for img in os.listdir(person_train_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Copy a few images to val folder
    for img in images[:max_images_per_class]:
        src = os.path.join(person_train_path, img)
        dst = os.path.join(person_val_path, img)
        shutil.copy(src, dst)

print("✅ Rebuilt val/ directory with matching structure and limited images.")
