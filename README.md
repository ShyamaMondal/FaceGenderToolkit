# 🧠 Face & Gender Detection Toolkit

A real-time deep learning project for gender classification and face recognition using webcam input. This submission is split into two subtasks:

- **Task A**: Gender Detection
- **Task B**: Face Recognition

Both models are trained using state-of-the-art CNN architectures and evaluated on custom datasets with live webcam inference capability.

---

## 📁 Project Structure
```
FaceDetectionWeb
├── giveDataSet/
│ ├── Task_A/ (train/val images)
│ └── Task_B/ (train/val images)
├── models/
│ ├── TaskA_Model.h5
│ ├── TaskB_Model.h5
│ └── class_names.json
├── tools/
│ ├── clean_val.py
│ ├── rebuild_val.py
│ └── utils.py
├── taskA_Model_Train.py
├── taskB_Model_Train.py
├── TaskA_Predict.py
├── TaskB_Predict.py
├── requirements.txt
├── results.txt
└── README.md
```
---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/FaceGenderToolkit.git
cd FaceGenderToolkit

2. Create and activate a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

🏗️ Model Architecture
Component	        Task A (Gender)	                        Task B (Face Recognition)
Base Model	        EfficientNetB0 (ImageNet)	            ResNet50 (ImageNet)
Input Preprocess	Grayscale → RGB (3 ch)	                RGB with normalization
Output Classes	    Female, Male	                        Person identities (softmax)
Loss Function	    Focal Loss	                            Sparse Categorical CrossEnt
Augmentation	    Flip, rotate, zoom, etc.	            Flip, noise, brightness


🧪 Training & Evaluation
🧠 Task A (Gender Classification)
bash
Copy
Edit
python taskA_Model_Train.py
Input: Grayscale 128x128 face images

Metrics: Accuracy, Precision, Recall, AUC

Model: models/TaskA_Model.h5

🧠 Task B (Face Recognition)
bash
Copy
Edit
python taskB_Model_Train.py
Input: RGB 128x128 face images

Metrics: Accuracy, Validation Loss

Model: models/TaskB_Model.h5

🧪 Inference (Webcam)
Task A – Gender Detection
bash
Copy
Edit
python TaskA_Predict.py
Task B – Face Recognition
bash
Copy
Edit
python TaskB_Predict.py
Both scripts will open webcam, detect faces using MTCNN, and classify in real-time.

Press q to quit webcam window.

📊 Results
yaml
Copy
Edit
(Task A)
Accuracy:      94.7%
Precision:     95.1%
Recall:        94.2%
Balanced AUC:  0.981

(Task B)
Accuracy:      91.3%
Validation Loss: 0.42
Full results can be found in results.txt.

🧠 Team Info
Team Name: NextGen Coders

Submission Member Name: Shyama Mondal

Email ID: shayamamondalbirati2002@gmail.com

Contact: [+91-7439302062]

💡 Notes
Only folder path needs to be updated if directory changes.

Model weights are included and directly usable.

No external editing required after submission.

Code is modular, readable, and well-commented.

