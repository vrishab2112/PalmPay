let videoElement = null;
let stream = null;
let capturedImage = null;

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

function showPreview(imageData, name) {
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const previewName = document.getElementById('previewName');
    const previewStatus = document.getElementById('previewStatus');
    
    previewImage.src = imageData;
    previewName.textContent = name;
    previewStatus.textContent = 'Ready to register';
    previewStatus.style.color = '#4CAF50';
    
    previewContainer.style.display = 'block';
    
    // Show retake and register buttons
    document.getElementById('retakeButton').style.display = 'flex';
    document.getElementById('registerButton').style.display = 'flex';
    document.getElementById('captureButton').style.display = 'none';
}

function hidePreview() {
    document.getElementById('previewContainer').style.display = 'none';
    document.getElementById('retakeButton').style.display = 'none';
    document.getElementById('registerButton').style.display = 'none';
    document.getElementById('captureButton').style.display = 'flex';
}

async function captureImage() {
    const name = document.getElementById('nameInput').value.trim();
    
    if (!name) {
        showError('Please enter your full name');
        return;
    }
    
    if (name.length < 2) {
        showError('Please enter a valid name (at least 2 characters)');
        return;
    }
    
    if (!videoElement || !stream) {
        showError('Camera is not ready. Please refresh the page and try again.');
        return;
    }
    
    try {
        updateStatus('Capturing image...', 'info');
        
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        canvas.getContext('2d').drawImage(videoElement, 0, 0);
        
        capturedImage = canvas.toDataURL('image/jpeg', 0.9);
        
        // Show preview first
        showPreview(capturedImage, name);
        updateStatus('Image captured successfully', 'success');
        
    } catch (err) {
        console.error('Error capturing image:', err);
        showError('Error capturing image. Please try again.');
        updateStatus('Capture failed', 'error');
    }
}

async function retakeImage() {
    hidePreview();
    updateStatus('Camera Ready', 'success');
}

async function registerUser() {
    if (!capturedImage) {
        showError('No image captured. Please capture your palm first.');
        return;
    }
    const name = document.getElementById('nameInput').value.trim();
    const phone = document.getElementById('phoneInput').value.trim();
    const password = document.getElementById('passwordInput').value;
    const confirmPassword = document.getElementById('confirmPasswordInput').value;

    if (!name || !phone || !password || !confirmPassword) {
        showError('Please fill in all fields.');
        return;
    }
    if (!/^\d{10}$/.test(phone)) {
        showError('Please enter a valid 10-digit phone number.');
        return;
    }
    if (password.length < 6) {
        showError('Password must be at least 6 characters.');
        return;
    }
    if (password !== confirmPassword) {
        showError('Passwords do not match.');
        return;
    }

    showLoading();
    updateStatus('Processing registration...', 'info');

    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                phone: phone,
                password: password,
                image: capturedImage
            })
        });

        const result = await response.json();
        hideLoading();
        if (response.ok) {
            updateStatus('Registration successful!', 'success');
            showSuccess();
        } else {
            updateStatus('Registration failed', 'error');
            showError(result.error || result.message || 'Registration failed. Please try again.');
        }
    } catch (err) {
        console.error('Error during registration:', err);
        hideLoading();
        updateStatus('Network error', 'error');
        showError('Network error. Please check your connection and try again.');
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    initCamera();
    
    // Add enter key support for name input
    document.getElementById('nameInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            captureImage();
        }
    });
    
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

// Prevent form submission on enter key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.id === 'nameInput') {
        e.preventDefault();
    }
}); 