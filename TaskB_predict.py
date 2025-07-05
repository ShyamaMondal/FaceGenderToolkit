import os, json, cv2, numpy as np, tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model

# ──────────────────────────────────
# ✅ Paths & constants
# ──────────────────────────────────
MODEL_PATH       = "models/TaskB_Model.h5"     
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE         = (128, 128)

# ──────────────────────────────────
# ✅ Load model & class names
# ──────────────────────────────────
model = load_model(MODEL_PATH)                      # no custom_objects needed
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"✅ Model & {num_classes} class names loaded.")

# ──────────────────────────────────
# ✅ Face detector
# ──────────────────────────────────
detector = MTCNN()

def preprocess_face(face_rgb):
    """Resize + ResNet preprocess; returns (1,128,128,3) float32 tensor."""
    face = cv2.resize(face_rgb, IMG_SIZE, interpolation=cv2.INTER_LANCZOS4)
    face_arr = img_to_array(face)
    face_arr = np.expand_dims(face_arr, axis=0)
    return preprocess_input(face_arr)

# ──────────────────────────────────
# ✅ Live webcam loop (auto‑predict)
# ──────────────────────────────────
cap = cv2.VideoCapture(0)          # change index if you have multiple cams
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("📸 Live prediction started | Press  'q'  to quit")

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("❌ Could not read from camera.")
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(frame_rgb)

    # draw & predict for each detected face
    for det in detections:
        if det["confidence"] < 0.90:      # skip weak detections
            continue

        x, y, w, h = det["box"]
        x, y = max(0, x), max(0, y)       # clamp to image bounds
        face_rgb = frame_rgb[y:y+h, x:x+w]

        try:
            input_tensor = preprocess_face(face_rgb)
            preds        = model.predict(input_tensor, verbose=0)[0]
            top_idx      = int(np.argmax(preds))
            confidence   = float(preds[top_idx])
            name         = class_names[top_idx]
            label_txt    = f"{name} ({confidence*100:.1f}%)"

            # Draw bounding box & label
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_bgr, label_txt, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            print("⚠️ Error during prediction:", e)

    # show frame with overlays
    cv2.imshow("Live Face Recognition", frame_bgr)

    # quit if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
