let videoElement = null;
let stream = null;
let verifiedUser = null;
let verifiedPalmImage = null;
let cartItems = [
    { id: 1, name: 'Premium Cotton T-Shirt', price: 25, quantity: 1, description: 'Comfortable 100% cotton t-shirt', size: 'L', color: 'Blue' },
    { id: 2, name: 'Running Shoes', price: 90, quantity: 1, description: 'Lightweight athletic running shoes', size: '10', color: 'Black' },
    { id: 3, name: 'Wireless Headphones', price: 120, quantity: 1, description: 'High-quality Bluetooth headphones', size: 'Pro', color: 'White' },
    { id: 4, name: 'Phone Case', price: 50, quantity: 1, description: 'Durable protective phone case', size: 'Universal', color: 'Clear' }
];

// Cart calculations
const TAX_RATE = 0.085;
const SHIPPING_COST = 10;

const COIN_W_SVG = `<svg class="coin-w" width="18" height="18" viewBox="0 0 32 32" style="vertical-align:middle;"><defs><radialGradient id="gold" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#fffbe6"/><stop offset="60%" stop-color="#ffe066"/><stop offset="100%" stop-color="#ffd700"/></radialGradient></defs><circle cx="16" cy="16" r="15" fill="url(#gold)" stroke="#bfa100" stroke-width="2"/><text x="16" y="21" text-anchor="middle" font-size="16" font-family="Arial Black,Arial,sans-serif" fill="#bfa100" font-weight="bold">W</text></svg>`;

async function initCamera() {
    try {
        videoElement = document.getElementById('videoElement');
        const statusIndicator = document.getElementById('statusIndicator');
        
        // Update status
        statusIndicator.innerHTML = '<i class="fas fa-circle"></i><span>Initializing camera...</span>';
        
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment'
            } 
        });
        
        videoElement.srcObject = stream;
        
        // Wait for video to be ready
        videoElement.onloadedmetadata = () => {
            statusIndicator.innerHTML = '<i class="fas fa-circle"></i><span>Camera Ready</span>';
            statusIndicator.querySelector('i').style.color = '#4CAF50';
        };
        
    } catch (err) {
        console.error('Error accessing camera:', err);
        showError('Camera access denied. Please ensure camera permissions are granted and try again.');
        updateStatus('Camera Error', 'error');
    }
}

async function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
}

function updateStatus(message, type = 'info') {
    const statusIndicator = document.getElementById('statusIndicator');
    const icon = statusIndicator.querySelector('i');
    
    statusIndicator.querySelector('span').textContent = message;
    
    switch(type) {
        case 'success':
            icon.style.color = '#4CAF50';
            break;
        case 'error':
            icon.style.color = '#f44336';
            break;
        case 'warning':
            icon.style.color = '#ff9800';
            break;
        default:
            icon.style.color = '#2196F3';
    }
}

function showLoading() {
    document.getElementById('loadingModal').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingModal').style.display = 'none';
}

function showSuccess() {
    document.getElementById('successModal').style.display = 'flex';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorModal').style.display = 'flex';
}

function closeErrorModal() {
    document.getElementById('errorModal').style.display = 'none';
}

function updateQuantity(itemId, change) {
    const item = cartItems.find(item => item.id === itemId);
    if (item) {
        item.quantity = Math.max(1, item.quantity + change);
        updateCartDisplay();
        calculateTotals();
    }
}

function updateCartDisplay() {
    // Update quantity displays
    cartItems.forEach(item => {
        const quantityElement = document.querySelector(`[onclick=\"updateQuantity(${item.id}, -1)\"]`).nextElementSibling;
        quantityElement.textContent = item.quantity;
    });
}

function calculateTotals() {
    const subtotal = cartItems.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    const tax = Math.round(subtotal * TAX_RATE);
    const total = subtotal + tax + SHIPPING_COST;
    
    document.getElementById('subtotal').innerHTML = `${COIN_W_SVG}${subtotal}`;
    document.getElementById('tax').innerHTML = `${COIN_W_SVG}${tax}`;
    document.getElementById('shipping').innerHTML = `${COIN_W_SVG}${SHIPPING_COST}`;
    document.getElementById('total').innerHTML = `${COIN_W_SVG}${total}`;
    document.getElementById('payAmount').textContent = total;
    
    return total;
}

function showUserInfo(userName, currentBalance) {
    const userInfo = document.getElementById('userInfo');
    const userNameElement = document.getElementById('userName');
    const currentBalanceElement = document.getElementById('currentBalance');
    
    userNameElement.textContent = userName;
    currentBalanceElement.textContent = currentBalance;
    userInfo.style.display = 'flex';
    
    // Show pay button
    document.getElementById('payButton').style.display = 'flex';
    document.getElementById('verifyButton').style.display = 'none';
}

function hideUserInfo() {
    document.getElementById('userInfo').style.display = 'none';
    document.getElementById('payButton').style.display = 'none';
    document.getElementById('verifyButton').style.display = 'flex';
}

async function verifyPalm() {
    if (!videoElement || !stream) {
        showError('Camera is not ready. Please refresh the page and try again.');
        return;
    }
    
    try {
        updateStatus('Verifying palm...', 'info');
        
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        canvas.getContext('2d').drawImage(videoElement, 0, 0);
        
        verifiedPalmImage = canvas.toDataURL('image/jpeg', 0.9);
        
        showLoading();
        
        const response = await fetch('/api/verify-payment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                amount: 0, // Just for verification, not actual payment
                image: verifiedPalmImage
            })
        });

        const result = await response.json();
        
        hideLoading();
        
        if (response.ok) {
            verifiedUser = {
                name: result.name,
                currentBalance: result.new_balance // No verification charge
            };
            
            updateStatus('Palm verified successfully', 'success');
            showUserInfo(verifiedUser.name, verifiedUser.currentBalance);
        } else {
            updateStatus('Palm verification failed', 'error');
            showError(result.error || 'Palm verification failed. Please try again.');
        }
    } catch (err) {
        console.error('Error during palm verification:', err);
        hideLoading();
        updateStatus('Verification error', 'error');
        showError('Network error. Please check your connection and try again.');
    }
}

async function processPayment() {
    if (!verifiedUser || !verifiedPalmImage) {
        showError('Please verify your palm first.');
        return;
    }
    
    const totalAmount = calculateTotals();
    console.log('Processing payment with amount:', totalAmount, 'and image:', !!verifiedPalmImage);
    if (!totalAmount || isNaN(totalAmount) || totalAmount <= 0) {
        showError('Cart total is invalid or empty. Please add items to your cart and try again.');
        return;
    }
    
    if (verifiedUser.currentBalance < totalAmount) {
        showError(`Insufficient balance. You have ₹${verifiedUser.currentBalance} but need ₹${totalAmount}`);
        return;
    }
    
    try {
        updateStatus('Processing payment...', 'info');
        
        showLoading();
        
        const response = await fetch('/api/verify-payment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                amount: totalAmount, // Send as integer rupees
                image: verifiedPalmImage
            })
        });

        const result = await response.json();
        
        hideLoading();
        
        if (response.ok) {
            updateStatus('Payment successful!', 'success');
            
            // Show success modal with details
            document.getElementById('successCustomer').textContent = result.name;
            document.getElementById('successAmount').textContent = totalAmount;
            document.getElementById('successBalance').textContent = result.new_balance;
            document.getElementById('orderId').textContent = generateOrderId();
            
            showSuccess();
        } else {
            updateStatus('Payment failed', 'error');
            showError(result.error || 'Payment failed. Please try again.');
        }
    } catch (err) {
        console.error('Error during payment:', err);
        hideLoading();
        updateStatus('Payment error', 'error');
        showError('Network error. Please check your connection and try again.');
    }
}

function generateOrderId() {
    const timestamp = Date.now();
    const random = Math.floor(Math.random() * 1000);
    return `ORD-${timestamp}-${random}`;
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    initCamera();
    calculateTotals();
    
    // Add click outside modal to close
    document.getElementById('errorModal').addEventListener('click', (e) => {
        if (e.target.id === 'errorModal') {
            closeErrorModal();
        }
    });
    
    // Add click outside modal to close success modal
    document.getElementById('successModal').addEventListener('click', (e) => {
        if (e.target.id === 'successModal') {
            document.getElementById('successModal').style.display = 'none';
        }
    });
});

// Clean up when page is unloaded
window.addEventListener('beforeunload', stopCamera); 