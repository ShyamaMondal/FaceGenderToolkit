import os
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, "../giveDataSet/Task_B/train")
val_dir = os.path.join(BASE_DIR, "../giveDataSet/Task_B/val")

train_people = set(os.listdir(train_dir))
val_people = set(os.listdir(val_dir))

extra = val_people - train_people
print(f"🔍 Found {len(extra)} extra folders in val/ not present in train/. Removing them...")

for person in extra:
    folder = os.path.join(val_dir, person)
    if os.path.isdir(folder):
        shutil.rmtree(folder)

print("✅ Extra folders removed successfully.")
