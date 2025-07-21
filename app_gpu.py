# app_gpu.py - GPU-enabled version of PalmPay backend
# Uses CUDA (GPU) for ResNet if available, otherwise falls back to CPU

from flask import Flask, render_template, request, jsonify, redirect, url_for
import sqlite3
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import base64
import os
import numpy as np
import cv2
import time
import mediapipe as mp
from werkzeug.security import generate_password_hash, check_password_hash
from typing import Optional, Any

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'palm_images'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Set device for PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ResNet50 model for better accuracy
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(  # type: ignore
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256)
)
model = model.to(DEVICE)
model.eval()

# Enhanced image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Larger input size
    transforms.CenterCrop(224),     # Center crop
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add some color augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def enhance_palm_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def extract_palm_roi(image):
    img_rgb = np.array(image.convert('RGB'))
    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        ) as hands:
            results = hands.process(img_rgb)
            if not results.multi_hand_landmarks:
                return None
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = img_rgb.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            min_x = int(min(x_coords) * w)
            max_x = int(max(x_coords) * w)
            min_y = int(min(y_coords) * h)
            max_y = int(max(y_coords) * h)
            margin_x = int(0.12 * w)
            margin_y = int(0.12 * h)
            min_x = max(min_x - margin_x, 0)
            max_x = min(max_x + margin_x, w)
            min_y = max(min_y - margin_y, 0)
            max_y = min(max_y + margin_y, h)
            palm_roi = img_rgb[min_y:max_y, min_x:max_x]
            if palm_roi.size == 0 or palm_roi.shape[0] < 30 or palm_roi.shape[1] < 30:
                return None
            return Image.fromarray(palm_roi)
    except Exception as e:
        print(f"MediaPipe error: {e}")
        return None

def init_db():
    conn = sqlite3.connect('palmpayment.db')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS transactions')
    c.execute('DROP TABLE IF EXISTS users')
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            phone TEXT NOT NULL UNIQUE,
            palm_image BLOB NOT NULL,
            wallet_balance INTEGER DEFAULT 1000
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            amount INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def get_palm_features(palm_image):
    enhanced_image = enhance_palm_image(palm_image)
    img_tensor = transform(enhanced_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model(img_tensor)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.squeeze().cpu().numpy()

# The rest of the code (Flask routes, etc.) is identical to app_new.py
# You can copy-paste the rest of your app_new.py code here without changes. 