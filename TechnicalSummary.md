Here is a one-page technical summary of your **Face & Gender Detection Toolkit**, outlining the approach, innovations, and model architecture:

---

## 🧠 Technical Summary – Face & Gender Detection Toolkit

### 📌 Objective

The project aims to perform **real-time gender classification** and **identity recognition** from live webcam input using deep learning. It is split into two subtasks:

* **Task A**: Gender classification (binary classification)
* **Task B**: Face recognition (multi-class classification)

Both tasks are implemented using **fine-tuned CNN architectures** trained on custom datasets and integrated into a live prediction pipeline.

---

### 🔬 Approach

#### 1. **Task A – Gender Detection**

* **Input**: 128×128 grayscale face image
* **Model**: Fine-tuned **EfficientNetB0**
* **Loss Function**: `SigmoidFocalCrossEntropy` from TensorFlow Addons
* **Output**: Softmax probabilities for `Female` and `Male`
* **Augmentation**: Rotation, brightness, zoom, flip, shear
* **Class imbalance**: Addressed using `compute_class_weight`

#### 2. **Task B – Face Recognition**

* **Input**: 128×128 RGB face image
* **Model**: Fine-tuned **ResNet50**
* **Loss Function**: `SparseCategoricalCrossentropy`
* **Output**: Softmax over person identities
* **Augmentation**: Zoom, flip, rotation, Gaussian noise, contrast

---

### ⚙️ Architecture & Pipeline

#### **Model Architecture Overview**

| Component        | Task A (Gender)                          | Task B (Face ID)                         |
| ---------------- | ---------------------------------------- | ---------------------------------------- |
| Base Model       | EfficientNetB0 (ImageNet weights)        | ResNet50 (ImageNet weights)              |
| Input Preprocess | Grayscale → RGB (3-channel merge)        | Standard RGB with preprocessing          |
| Head Layers      | GAP → BN → Dropout → Dense(256) → Output | GAP → BN → Dense(256) → Dropout → Output |

#### **Webcam Inference Pipeline**

1. **Face Detection**: `MTCNN` to localize faces in live frames.
2. **Preprocessing**: Convert to grayscale (for Task A) and resize.
3. **Gender Prediction**: Use gender model to predict class with confidence.
4. **Face Recognition**: Use face model to predict most probable identity.
5. **Rendering**: Annotate real-time video with bounding boxes and predictions.

---

### 🌟 Key Innovations

* **Dual-model architecture**: Seamless integration of two specialized models.
* **Mixed-precision training**: Automatic GPU float16 support improves performance.
* **Robust augmentation pipeline**: Improves generalization under variable lighting and poses.
* **Lightweight and portable**: Designed to run on laptops with or without GPU.
* **Folder structure automation**: Utility scripts help synchronize `train/` and `val/` datasets efficiently.

---

### 🧪 Results & Evaluation

* Accuracy and AUC monitored during training with TensorBoard-like plots.
* Best models saved via `ModelCheckpoint`; training logs exported to CSV.
* Real-world performance evaluated in noisy webcam environments with fast inference times.

---

