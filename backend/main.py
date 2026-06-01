import io, base64, time
from contextlib import asynccontextmanager
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras

# Bypassing the quantization_config Keras deserialization bug
_orig_keras_dense = keras.layers.Dense.__init__
keras.layers.Dense.__init__ = lambda self, *args, **kwargs: _orig_keras_dense(self, *args, **{k: v for k, v in kwargs.items() if k != 'quantization_config'})

_orig_tf_dense = tf.keras.layers.Dense.__init__
tf.keras.layers.Dense.__init__ = lambda self, *args, **kwargs: _orig_tf_dense(self, *args, **{k: v for k, v in kwargs.items() if k != 'quantization_config'})

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import onnxruntime as ort

# ── Config ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH    = str(_PROJECT_ROOT / "EfficientNetB2_brain_tumor_model.keras")
YOLO_MODEL_PATH = str(_PROJECT_ROOT / "mri_tumor_run" / "weights" / "best.onnx")
CLASS_NAMES   = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE      = (224, 224) # Matches your trained model weights

app = FastAPI(title="🧠 Brain Tumor API (Fixed & Synchronized)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

tokenizer = model = None
predict_fn = None
gradcam_model = None
yolo_session = None

def load_assets():
    global model, predict_fn, gradcam_model, yolo_session
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if Path(YOLO_MODEL_PATH).exists():
        yolo_session = ort.InferenceSession(YOLO_MODEL_PATH, providers=['CPUExecutionProvider'])
    
    try:
        effnet = model.get_layer("efficientnetb2")
        top_conv = effnet.get_layer("top_conv")
        effnet_extractor = tf.keras.Model(inputs=effnet.input, outputs=[top_conv.output, effnet.output])
        
        inp = tf.keras.Input(shape=(224, 224, 3))
        c_out, e_out = effnet_extractor(inp)
        
        x = e_out
        for layer in model.layers[1:]: x = layer(x)
        gradcam_model = tf.keras.Model(inputs=inp, outputs=[c_out, x])
        print("✅ Grad-CAM architecture locked to 224x224")
    except Exception as e:
        print(f"⚠️ Grad-CAM Init Failed: {e}")

    @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    def _infer(x): return model(x, training=False)
    
    predict_fn = _infer
    predict_fn(np.zeros((1, 224, 224, 3), np.float32)) # Warm-up
    print("✅ Inference engine ready.")

@asynccontextmanager
async def lifespan(_: FastAPI):
    load_assets()
    yield

app.router.lifespan_context = lifespan

def _preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def _gradcam_b64(pil_img: Image.Image, pred_idx: int) -> str:
    if gradcam_model is None:
        buf = io.BytesIO(); pil_img.resize(IMG_SIZE).save(buf, format="JPEG"); return base64.b64encode(buf.getvalue()).decode()
    
    tensor = tf.constant(_preprocess(pil_img))
    with tf.GradientTape() as tape:
        c_out, preds = gradcam_model(tensor, training=False)
        tape.watch(c_out)
        loss = preds[:, pred_idx]
    
    grads = tape.gradient(loss, c_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = (c_out[0] @ pooled[..., tf.newaxis]).numpy().squeeze()
    heatmap = np.maximum(heatmap, 0) / (heatmap.max() + 1e-10)
    
    orig_size = pil_img.size # (width, height)
    h = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, orig_size)), cv2.COLORMAP_JET)
    orig = np.array(pil_img.convert("RGB"))
    overlay = cv2.addWeighted(orig, 0.6, h, 0.4, 0)
    _, buf = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buf).decode()

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form("EfficientNetB2")):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    t0 = time.perf_counter()
    
    if model_type == "YOLOv8 Segmentation" and yolo_session is not None:
        img_arr = np.array(img)
        original_h, original_w = img_arr.shape[:2]
        
        # Preprocess for YOLO
        img_resized = cv2.resize(img_arr, (640, 640))
        input_tensor = img_resized.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        input_name = yolo_session.get_inputs()[0].name
        outputs = yolo_session.run(None, {input_name: input_tensor})
        
        preds, protos = None, None
        for out in outputs:
            if len(out.shape) == 4: protos = out[0]
            elif len(out.shape) == 3: preds = out[0].T
            
        conf_thresh = 0.25
        valid = preds[:, 4] > conf_thresh
        filtered = preds[valid]
        
        if len(filtered) > 0:
            best_idx = np.argmax(filtered[:, 4])
            best = filtered[best_idx]
            
            cx, cy, w, h = best[:4]
            conf = float(best[4])
            mask_coeffs = best[5:]
            
            mask = mask_coeffs @ protos.reshape(32, -1)
            mask = 1 / (1 + np.exp(-mask))
            mask = mask.reshape(160, 160)
            mask_640 = cv2.resize(mask, (640, 640))
            
            x1, y1 = max(0, int(cx - w/2)), max(0, int(cy - h/2))
            x2, y2 = min(640, int(cx + w/2)), min(640, int(cy + h/2))
            
            crop_mask = np.zeros((640, 640), dtype=np.float32)
            crop_mask[y1:y2, x1:x2] = mask_640[y1:y2, x1:x2]
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            crop_mask = cv2.resize(crop_mask, (original_w, original_h))
            
            overlay = img_arr.copy()
            overlay[crop_mask == 1] = [255, 50, 50]
            annotated = cv2.addWeighted(img_arr, 0.6, overlay, 0.4, 0)
            
            rx, ry = original_w / 640, original_h / 640
            bx1, by1 = int(x1 * rx), int(y1 * ry)
            bx2, by2 = int(x2 * rx), int(y2 * ry)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 50, 50), 3)
            
            annotated_img = Image.fromarray(annotated)
            label = "Tumor"
            scores = {"Tumor": conf, "No Tumor": 1 - conf, "Glioma": 0.0, "Meningioma": 0.0}
        else:
            annotated_img = img
            label = "No Tumor"
            conf = 1.0
            scores = {"No Tumor": 1.0, "Tumor": 0.0, "Glioma": 0.0, "Meningioma": 0.0}
            
        buf = io.BytesIO()
        annotated_img.save(buf, format="JPEG")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode()
            
        lat = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "label": label,
            "confidence": conf,
            "scores": scores,
            "latency_ms": lat,
            "gradcam_b64": gradcam_b64
        }

    # EfficientNetB2 Code
    arr = _preprocess(img)
    
    # TTA Inference
    p1 = predict_fn(arr).numpy()[0]
    p2 = predict_fn(tf.image.flip_left_right(arr)).numpy()[0]
    avg_probs = (p1 + p2) / 2
    lat = round((time.perf_counter() - t0) * 1000, 2)
    
    idx = int(np.argmax(avg_probs))
    return {
        "label": CLASS_NAMES[idx],
        "confidence": float(avg_probs[idx]),
        "scores": {CLASS_NAMES[i]: float(avg_probs[i]) for i in range(4)},
        "latency_ms": lat,
        "gradcam_b64": _gradcam_b64(img, idx)
    }