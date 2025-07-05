

import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

# ─── Load the trained model ────────────────────────────────────────────────────
MODEL_PATH = "models/TaskA_Model.h5"
model = load_model(
    MODEL_PATH,
    custom_objects={"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy},
)
print("✅ Model loaded.")

# Class labels
class_names = ["Female", "Male"]

# ─── Utilities ────────────────────────────────────────────────────────────────
def preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Resize → grayscale → blur → [0‑1] scale → add batch‑axis"""
    face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
    pil = Image.fromarray(face_img).convert("L").resize((128, 128), Image.LANCZOS)
    arr = img_to_array(pil) / 255.0
    return np.expand_dims(arr, axis=0)  # shape (1, 128, 128, 1)

# MTCNN detector (kept outside the loop for speed)
detector = MTCNN()

# ─── Open the webcam ──────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("📡 Live gender detection started — press 'q' to quit.")

# ─── Main loop ────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Unable to read from camera.")
        break

    # Detect faces (convert to RGB for MTCNN)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    for det in detections:
        if det["confidence"] < 0.90:
            continue  # skip weak detections

        x, y, w, h = det["box"]
        x, y = max(0, x), max(0, y)  # keep inside frame
        face = rgb_frame[y : y + h, x : x + w]

        try:
            # Pre‑process and predict
            gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            tensor = preprocess_face(gray_face)
            pred = model.predict(tensor, verbose=0)[0]

            prob_male = float(pred[1])
            label = "Male" if prob_male >= 0.4 else "Female"
            confidence = prob_male if label == "Male" else 1 - prob_male
            label_text = f"{label}  {confidence*100:.1f}%"

            # Draw bounding‑box & label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception as e:
            print("⚠️  Face processing error:", e)

    # Show the annotated frame
    cv2.imshow("Live Gender Detection", frame)

    # Exit if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ─── Cleanup ──────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print("👋 Exited.")
