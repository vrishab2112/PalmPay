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

# Load pre-trained ResNet50 model for better accuracy
model = models.resnet50(pretrained=True)
# Remove the last layer and add new layers for better feature extraction
# Type ignore to handle the linter warning about Sequential vs Linear
model.fc = nn.Sequential(  # type: ignore
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 256)
)
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
    """Apply additional image enhancements"""
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Convert back to PIL Image
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def extract_palm_roi(image):
    # Convert PIL Image to RGB numpy array
    img_rgb = np.array(image.convert('RGB'))
    
    # Initialize MediaPipe Hands with improved settings
    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.5,  # Lowered for better detection
            min_tracking_confidence=0.5,
            model_complexity=0  # Use simpler model for faster detection
        ) as hands:
            results = hands.process(img_rgb)
            
            if not results.multi_hand_landmarks:
                return None
                
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = img_rgb.shape
            
            # Get all landmark coordinates
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            # Calculate bounding box with margin
            min_x = int(min(x_coords) * w)
            max_x = int(max(x_coords) * w)
            min_y = int(min(y_coords) * h)
            max_y = int(max(y_coords) * h)
            
            # Add margin for better palm capture
            margin_x = int(0.12 * w)  # Increased margin
            margin_y = int(0.12 * h)
            
            # Ensure bounds are within image
            min_x = max(min_x - margin_x, 0)
            max_x = min(max_x + margin_x, w)
            min_y = max(min_y - margin_y, 0)
            max_y = min(max_y + margin_y, h)
            
            # Extract palm region
            palm_roi = img_rgb[min_y:max_y, min_x:max_x]
            
            # Check if ROI is valid (more lenient)
            if palm_roi.size == 0 or palm_roi.shape[0] < 30 or palm_roi.shape[1] < 30:
                return None
                
            # Convert back to PIL Image
            return Image.fromarray(palm_roi)
    except Exception as e:
        print(f"MediaPipe error: {e}")
        return None

def init_db():
    conn = sqlite3.connect('palmpayment.db')
    c = conn.cursor()
    
    # Drop existing tables if they exist
    c.execute('DROP TABLE IF EXISTS transactions')
    c.execute('DROP TABLE IF EXISTS users')
    
    # Create users table with palm_image column
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
    
    # Create transactions table
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
    """Extract features from palm image using ResNet"""
    # Apply image enhancement
    enhanced_image = enhance_palm_image(palm_image)
    
    # Convert to tensor and get features
    img_tensor = transform(enhanced_image).unsqueeze(0)  # Now properly a tensor
    with torch.no_grad():
        features = model(img_tensor)
    
    # Normalize the feature vector
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.squeeze().numpy()

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/payment')
def payment():
    return render_template('payment.html')

@app.route('/recharge')
def recharge():
    return render_template('recharge.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    try:
        data = request.get_json(force=True)
        image_data = data.get('image')
        name = data.get('name')
        password = data.get('password')
        phone = data.get('phone')
        if not image_data or not name or not password or not phone:
            return jsonify({'error': 'Missing required fields'}), 400
        try:
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'error': 'Invalid image format'}), 400
        try:
            palm_roi = extract_palm_roi(image)
            if palm_roi is None:
                return jsonify({'error': 'No palm detected'}), 400
        except Exception as e:
            return jsonify({'error': 'Failed to process palm image'}), 500
        try:
            img_buffer = io.BytesIO()
            palm_roi.save(img_buffer, format='JPEG', quality=95)
            img_bytes = img_buffer.getvalue()
        except Exception as e:
            return jsonify({'error': 'Failed to save palm image'}), 500
        try:
            conn = sqlite3.connect('palmpayment.db')
            c = conn.cursor()
            # Check for duplicate phone
            c.execute('SELECT id FROM users WHERE phone = ?', (phone,))
            if c.fetchone():
                conn.close()
                return jsonify({'error': 'Phone number already registered'}), 409
            # Check for duplicate palm
            current_features = get_palm_features(palm_roi)
            c.execute('SELECT id, name, palm_image FROM users')
            existing_users = c.fetchall()
            for user_id, existing_name, existing_palm_bytes in existing_users:
                try:
                    existing_palm = Image.open(io.BytesIO(existing_palm_bytes))
                    existing_features = get_palm_features(existing_palm)
                    similarity = np.dot(current_features, existing_features) / (np.linalg.norm(current_features) * np.linalg.norm(existing_features))
                    if similarity > 0.95:
                        conn.close()
                        return jsonify({'error': f'This palm is already registered under name: {existing_name}'}), 409
                except Exception:
                    continue
            # Store password as plain text for testing
            c.execute('INSERT INTO users (name, password, phone, palm_image, wallet_balance) VALUES (?, ?, ?, ?, ?)',
                      (name, password, phone, img_bytes, 1000))
            conn.commit()
            conn.close()
        except Exception as e:
            return jsonify({'error': 'Database error: ' + str(e)}), 500
        return jsonify({'success': True, 'message': 'User registered successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json(force=True)
        name = data.get('name')
        password = data.get('password')
        if not name or not password:
            return jsonify({'error': 'Missing name or password'}), 400
        conn = sqlite3.connect('palmpayment.db')
        c = conn.cursor()
        c.execute('SELECT password FROM users WHERE name = ?', (name,))
        row = c.fetchone()
        if not row:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
        stored_pw = row[0]
        if stored_pw != password:
            conn.close()
            return jsonify({'error': 'Incorrect password'}), 401
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-payment', methods=['POST'])
def api_verify_payment():
    try:
        print("Starting payment verification process...")  # Debug log
        
        # Get the base64 image and amount from the request
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        image_data = request_data.get('image')
        amount = request_data.get('amount')
        
        if image_data is None or amount is None:
            print(f"Missing data - image: {bool(image_data)}, amount: {bool(amount)}")  # Debug log
            return jsonify({'error': 'Missing image or amount'}), 400
        
        try:
            # Convert base64 to image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            print("Successfully decoded image")  # Debug log
        except Exception as e:
            print(f"Error decoding image: {str(e)}")  # Debug log
            return jsonify({'error': 'Invalid image format'}), 400
        
        try:
            # Extract palm ROI
            palm_roi = extract_palm_roi(image)
            if palm_roi is None:
                print("No palm detected in image")  # Debug log
                return jsonify({'error': 'No palm detected'}), 400
            print("Successfully extracted palm ROI")  # Debug log
        except Exception as e:
            print(f"Error extracting palm ROI: {str(e)}")  # Debug log
            return jsonify({'error': 'Failed to process palm image'}), 500
        
        try:
            # Get features from the current palm image
            current_features = get_palm_features(palm_roi)
            print("Successfully extracted features")  # Debug log
        except Exception as e:
            print(f"Error extracting features: {str(e)}")  # Debug log
            return jsonify({'error': 'Failed to process palm features'}), 500
        
        try:
            # Compare with database
            conn = sqlite3.connect('palmpayment.db')
            c = conn.cursor()
            c.execute('SELECT id, name, palm_image, wallet_balance FROM users')
            users = c.fetchall()
            print(f"Found {len(users)} users in database")  # Debug log
            
            if not users:
                print("No users found in database")  # Debug log
                return jsonify({'error': 'No registered users found'}), 404
            
            best_match = None
            best_similarity = float('-inf')
            
            for user_id, name, stored_image_bytes, wallet_balance in users:
                try:
                    # Convert stored image bytes back to PIL Image
                    stored_image = Image.open(io.BytesIO(stored_image_bytes))
                    # Get features from stored image
                    stored_features = get_palm_features(stored_image)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(current_features, stored_features) / (np.linalg.norm(current_features) * np.linalg.norm(stored_features))
                    print(f"Similarity with user {name}: {similarity}")  # Debug log
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = (user_id, name, wallet_balance)
                except Exception as e:
                    print(f"Error processing user {name}: {str(e)}")  # Debug log
                    continue
            
            if best_similarity < 0.91:
                print(f"Best similarity {best_similarity} below threshold")  # Debug log
                return jsonify({'error': 'Palm verification failed'}), 401
            
            if best_match is None:
                print("No valid matches found")  # Debug log
                return jsonify({'error': 'No matching user found'}), 404
            
            user_id, name, wallet_balance = best_match
            print(f"Found matching user: {name}")  # Debug log
            
            # Check wallet balance (only if amount > 0)
            if amount > 0 and wallet_balance < amount:
                print(f"Insufficient balance: {wallet_balance} < {amount}")  # Debug log
                return jsonify({'error': 'Insufficient balance'}), 402
            
            # Process payment (only if amount > 0)
            new_balance = wallet_balance
            if amount > 0:
                new_balance = wallet_balance - amount
                c.execute('UPDATE users SET wallet_balance = ? WHERE id = ?', (new_balance, user_id))
                c.execute('INSERT INTO transactions (user_id, amount) VALUES (?, ?)', (user_id, -amount))
                conn.commit()
                print(f"Payment processed successfully. New balance: {new_balance}")  # Debug log
            else:
                print(f"Palm verification successful for user: {name}")  # Debug log
            
            conn.close()
            
            return jsonify({
                'success': True,
                'message': 'Payment successful' if amount > 0 else 'Palm verification successful',
                'name': name,
                'new_balance': new_balance
            })
            
        except Exception as e:
            print(f"Database error: {str(e)}")  # Debug log
            return jsonify({'error': 'Database error: ' + str(e)}), 500
    
    except Exception as e:
        print(f"Unexpected error during payment: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

@app.route('/api/recharge', methods=['POST'])
def api_recharge():
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        amount = request_data.get('amount')
        upi_id = request_data.get('upi_id')

        if not amount or not upi_id:
            return jsonify({'error': 'Missing amount or UPI ID'}), 400

        # Simulate payment processing delay
        time.sleep(1)

        # Get user from palm verification (simulated for recharge)
        conn = sqlite3.connect('palmpayment.db')
        c = conn.cursor()
        
        # For demo, update the first user's balance
        c.execute('SELECT id, wallet_balance FROM users LIMIT 1')
        user = c.fetchone()
        
        if not user:
            return jsonify({'error': 'No user found'}), 404
            
        user_id, current_balance = user
        new_balance = current_balance + amount
        
        # Update balance and record transaction
        c.execute('UPDATE users SET wallet_balance = ? WHERE id = ?', (new_balance, user_id))
        c.execute('INSERT INTO transactions (user_id, amount) VALUES (?, ?)', (user_id, amount))
        
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'message': 'Recharge successful',
            'new_balance': new_balance
        })

    except Exception as e:
        print(f"Error during recharge: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/user-balance')
def api_user_balance():
    username = request.args.get('username')
    if not username:
        return jsonify({'success': False, 'error': 'Missing username'}), 400
    try:
        conn = sqlite3.connect('palmpayment.db')
        c = conn.cursor()
        c.execute('SELECT wallet_balance FROM users WHERE name = ?', (username,))
        row = c.fetchone()
        conn.close()
        if not row:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        return jsonify({'success': True, 'balance': int(row[0])})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin')
def admin():
    conn = sqlite3.connect('palmpayment.db')
    c = conn.cursor()
    # Get all users
    c.execute('SELECT id, name, wallet_balance FROM users')
    users = c.fetchall()
    # Get all transactions with user names
    c.execute('''
        SELECT t.id, u.name, t.amount, datetime(t.timestamp, 'localtime') 
        FROM transactions t 
        JOIN users u ON t.user_id = u.id 
        ORDER BY t.timestamp DESC
    ''')
    transactions = c.fetchall()
    conn.close()
    return render_template('admin.html', users=users, transactions=transactions)

@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    try:
        conn = sqlite3.connect('palmpayment.db')
        c = conn.cursor()
        # Delete user's transactions first (due to foreign key constraint)
        c.execute('DELETE FROM transactions WHERE user_id = ?', (user_id,))
        # Then delete the user
        c.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        return redirect(url_for('admin'))
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/user/update/<int:user_id>', methods=['POST'])
def admin_update_user(user_id):
    try:
        name = request.form.get('name')
        balance = request.form.get('balance')
        
        if not name or not balance:
            return redirect(url_for('admin'))
            
        conn = sqlite3.connect('palmpayment.db')
        c = conn.cursor()
        c.execute('UPDATE users SET name = ?, wallet_balance = ? WHERE id = ?', 
                 (name, balance, user_id))
        conn.commit()
        conn.close()
        
        return redirect(url_for('admin'))
    except Exception as e:
        print(f"Error updating user: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/transaction/delete/<int:transaction_id>', methods=['POST'])
def admin_delete_transaction(transaction_id):
    try:
        conn = sqlite3.connect('palmpayment.db')
        c = conn.cursor()
        
        # Get transaction details before deleting
        c.execute('SELECT user_id, amount FROM transactions WHERE id = ?', (transaction_id,))
        transaction = c.fetchone()
        
        if transaction:
            user_id, amount = transaction
            # Reverse the transaction amount in user's wallet
            c.execute('UPDATE users SET wallet_balance = wallet_balance - ? WHERE id = ?', (amount, user_id))
            # Delete the transaction
            c.execute('DELETE FROM transactions WHERE id = ?', (transaction_id,))
            conn.commit()
        
        conn.close()
        return redirect(url_for('admin'))
    except Exception as e:
        print(f"Error deleting transaction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/admin/user/add', methods=['POST'])
def admin_add_user():
    try:
        name = request.form.get('name')
        balance = request.form.get('balance', 1000)
        
        if not name:
            return redirect(url_for('admin'))
            
        conn = sqlite3.connect('palmpayment.db')
        c = conn.cursor()
        # Store empty bytes for palm_image as it will be updated later
        c.execute('INSERT INTO users (name, palm_image, wallet_balance) VALUES (?, ?, ?)', 
                 (name, b'', balance))
        conn.commit()
        conn.close()
        
        return redirect(url_for('admin'))
    except Exception as e:
        print(f"Error adding user: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.debug = True  # Enable debug mode
    app.secret_key = 'your-secret-key-here'  # Required for session management
    app.run(host='0.0.0.0', port=5000, debug=True)  # Run with debug mode and on all interfaces 