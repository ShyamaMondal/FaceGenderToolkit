import os
import shutil


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, "../giveDataSet/Task_B/train")
val_dir = os.path.join(BASE_DIR, "../giveDataSet/Task_B/val")

# Create missing folders in val and copy one image from train
for person in os.listdir(train_dir):
    person_train_path = os.path.join(train_dir, person)
    person_val_path = os.path.join(val_dir, person)

    if not os.path.exists(person_val_path):
        os.makedirs(person_val_path)
        # Copy one image from train to val
        for file in os.listdir(person_train_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(os.path.join(person_train_path, file), person_val_path)
                break  # Copy only 1 image
