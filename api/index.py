import gzip
import os
import io
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
import tensorflow as tf
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# CONFIGURATION
CLINICAL_THRESHOLD = 0.6875
WEIGHTS_PATH = '/tmp/asteri_phase8_attention_pet.keras'

# 1. MODEL ARCHITECTURE
def spatial_attention_3d(x):
    avg_pool = tf.keras.layers.Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(x)
    max_pool = tf.keras.layers.Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(x)
    concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = tf.keras.layers.Conv3D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return tf.keras.layers.Multiply()([x, attention])

def resnet_attention_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv3D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv3D(filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv3D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = spatial_attention_3d(x)
    x = tf.keras.layers.Add()([x, shortcut])
    return tf.keras.layers.Activation('relu')(x)

def build_asteri_brain():
    inputs = tf.keras.Input(shape=(80, 80, 80, 1))
    x = tf.keras.layers.Conv3D(32, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=3, strides=2, padding='same')(x)
    x = resnet_attention_block(x, 32)
    x = resnet_attention_block(x, 64, stride=2)
    x = resnet_attention_block(x, 128, stride=2)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

# 2. LIFESPAN
clinical_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clinical_model
    
    model_url = "https://github.com/Wasup1st/Asteri-NeuroAI/releases/download/v1.0/asteri_phase8_attention_pet.keras"
    
    if not os.path.exists(WEIGHTS_PATH):
        print("Downloading model from cloud storage...")
        response = requests.get(model_url)
        with open(WEIGHTS_PATH, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
        
    clinical_model = build_asteri_brain()
    clinical_model.load_weights(WEIGHTS_PATH)
    print("Diagnostic Core Online.")
    yield
    clinical_model = None

app = FastAPI(title="Asteri Neuro API", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 3. INFERENCE LOGIC
def process_raw_bytes(file_bytes):
    # Detect the gzip magic number and decompress if it is a .nii.gz file
    if file_bytes.startswith(b'\x1f\x8b'):
        file_bytes = gzip.decompress(file_bytes)
        
    file_like = io.BytesIO(file_bytes)
    fh = nib.FileHolder(fileobj=file_like)
    img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
    
    tensor = img.get_fdata()
    max_val = np.max(tensor)
    if max_val > 0: 
        tensor = tensor / max_val
        
    mask = tensor > 0.05
    coords = np.array(np.nonzero(mask))
    if coords.size > 0:
        min_c, max_c = coords.min(axis=1), coords.max(axis=1)
        tensor = tensor[min_c[0]:max_c[0], min_c[1]:max_c[1], min_c[2]:max_c[2]]
        
    zoom_factors = [t/c for t, c in zip((80, 80, 80), tensor.shape)]
    tensor = ndimage.zoom(tensor, zoom_factors, order=1)
    
    return np.expand_dims(np.expand_dims(tensor, axis=-1), axis=0)

def compute_masked_saliency(model, input_tensor):
    input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        prediction = model(input_tensor)
        output = prediction[0, 0]
    grads = tape.gradient(output, input_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy().squeeze()
    saliency = ndimage.gaussian_filter(saliency, sigma=1.5)
    vmax = np.percentile(saliency, 99)
    saliency = np.clip((saliency - np.min(saliency)) / (vmax - np.min(saliency) + 1e-8), 0, 1)
    raw_volume = input_tensor.numpy().squeeze()
    brain_mask = (raw_volume > 0.05).astype(float)
    saliency = saliency * brain_mask
    return saliency, float(prediction.numpy()[0][0])

# 4. ENDPOINT
@app.post("/")
@app.post("/predict")
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        processed_tensor = process_raw_bytes(file_bytes)
        heatmap, confidence = compute_masked_saliency(clinical_model, processed_tensor)
        return {
            "status": "success",
            "metrics": {
                "diagnosis": "ALZHEIMER'S DISEASE DETECTED" if confidence >= CLINICAL_THRESHOLD else "HEALTHY",
                "confidence_score": confidence
            },
            "visualization": {
                "heatmap_slice": heatmap[:, :, 40].tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
