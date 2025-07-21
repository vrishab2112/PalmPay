# PalmPay

PalmPay is a modern, secure, and user-friendly palm-based payment web application. It uses biometric palm recognition for authentication and payments, with a token system represented by a professional gold coin 'W' icon.

## Features
- **Palm-based Registration & Login**: Users register and log in using their palm image and credentials.
- **Biometric Payment**: Make payments by verifying your palm.
- **Token System**: All balances and transactions use tokens (1 token = 1 rupee), displayed with a gold coin 'W' icon.
- **Recharge Wallet**: Users can top up their wallet via UPI.
- **Admin Dashboard**: Manage users, view transactions, and edit balances.
- **Modern UI/UX**: Responsive, professional design with modals, icons, and clear flows.

## Tech Stack
- **Backend**: Python, Flask, SQLite
- **Biometrics**: MediaPipe, PyTorch, ResNet50
- **Frontend**: HTML, CSS, JavaScript (vanilla), Bootstrap, SVG icons

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd facepayment (1)/facepayment
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize the Database
The database is initialized automatically when you run the app for the first time. To reset, delete `palmpayment.db` and restart the app.

### 5. Run the Application
```bash
python app_new.py
```
The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage
- **Register**: Capture your palm, enter your details, and sign up.
- **Login**: Enter your username and password, or use palm verification for payments.
- **Make Payment**: Add items to your cart, verify your palm, and pay with tokens.
- **Recharge**: Top up your wallet using UPI.
- **Admin**: Log in as admin to manage users and transactions.

## Gold Coin 'W' Icon
All token/currency displays use a custom SVG gold coin with a 'W' in the center for a professional, unified look.

## Credits
- Palm recognition: [MediaPipe](https://mediapipe.dev/), [PyTorch](https://pytorch.org/)
- UI: [Bootstrap](https://getbootstrap.com/), [FontAwesome](https://fontawesome.com/)
- Project by [Your Name]

---
For any issues or contributions, please open an issue or pull request. 