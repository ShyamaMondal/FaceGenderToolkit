# utils.py
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img

def preprocess_image(path_or_array, img_size=(128, 128)):
    if isinstance(path_or_array, str):  # filepath
        img = load_img(path_or_array, color_mode="grayscale")
        img = img.resize(img_size, Image.Resampling.LANCZOS)
        img = np.array(img)
    else:  # numpy array (from webcam)
        img = path_or_array
        img = Image.fromarray(img).resize(img_size, Image.Resampling.LANCZOS)
        img = np.array(img)
    img = img.astype("float32") / 255.0
    return img.reshape(*img_size, 1)
