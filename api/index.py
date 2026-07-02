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

# --- CONFIGURATION ---
CLINICAL_THRESHOLD = 0.6875
# This is where the model will be saved temporarily in the server's memory
WEIGHTS_PATH = '/tmp/asteri_phase8_attention_pet.keras'

# --- 1. MODEL ARCHITECTURE ---
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

# --- 2. LIFESPAN: Load Model on Startup ---
clinical_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clinical_model
    
    # YOUR DIRECT DOWNLOAD URL FROM GITHUB RELEASES GOES HERE
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. INFERENCE LOGIC ---
def process_raw_bytes(file_bytes):
    file_like = io.BytesIO(file_bytes)
    fh = nib.FileHolder(fileobj=file_like)
    img = nib.Nifti1Image.from_file_map({'header': fh, 'image':
