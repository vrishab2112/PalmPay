document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('rechargeForm');
    const qrSection = document.getElementById('qrCodeSection');
    const qrAmount = document.getElementById('qrAmount');
    const confirmBtn = document.getElementById('confirmPayment');
    const generateBtn = document.getElementById('generateQR');
    const modal = new bootstrap.Modal(document.getElementById('paymentStatusModal'));
    const successMsg = document.getElementById('successMessage');
    const errorMsg = document.getElementById('errorMessage');
    const successAmount = document.getElementById('successAmount');
    const userTokenBalance = document.getElementById('userTokenBalance');
    const loggedInUser = document.getElementById('loggedInUser');
    const logoutBtn = document.getElementById('logoutBtn');

    // Get logged-in user from localStorage
    let username = localStorage.getItem('loggedInUser');
    if (!username) {
        // Not logged in, redirect to landing page
        window.location.href = '/';
        return;
    }
    loggedInUser.textContent = username;

    // Fetch and display user balance
    async function fetchUserBalance(username) {
        if (!username) {
            userTokenBalance.textContent = '0';
            return;
        }
        try {
            const res = await fetch(`/api/user-balance?username=${encodeURIComponent(username)}`);
            const data = await res.json();
            if (res.ok && data.success) {
                userTokenBalance.textContent = data.balance;
            } else {
                userTokenBalance.textContent = '0';
            }
        } catch (e) {
            userTokenBalance.textContent = '0';
        }
    }
    fetchUserBalance(username);

    // Handle logout/switch account
    logoutBtn.addEventListener('click', function() {
        localStorage.removeItem('loggedInUser');
        window.location.href = '/';
    });

    // Form validation
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        if (!form.checkValidity()) {
            e.stopPropagation();
            form.classList.add('was-validated');
            return;
        }

        const amount = document.getElementById('amount').value;
        const upiId = document.getElementById('upiId').value;

        // Generate fake QR code (in real app, this would be a real UPI QR code)
        const qrCode = document.getElementById('qrCode');
        qrCode.src = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=upi://pay?pa=merchant@palmpay&pn=PalmPay&am=${amount}&cu=INR`;
        
        // Show QR section and confirm button
        qrSection.classList.remove('d-none');
        confirmBtn.classList.remove('d-none');
        generateBtn.classList.add('d-none');
        qrAmount.textContent = amount;

        // Show processing animation
        document.querySelector('.payment-status').classList.remove('d-none');
        qrCode.style.display = 'none';

        // Simulate QR code loading
        setTimeout(() => {
            document.querySelector('.payment-status').classList.add('d-none');
            qrCode.style.display = 'block';
        }, 1500);
    });

    // Handle payment confirmation
    confirmBtn.addEventListener('click', function() {
        const amount = document.getElementById('amount').value;
        // Simulate API call to process payment
        fetch('/api/recharge', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                amount: parseInt(amount),
                username: username,
                upi_id: document.getElementById('upiId').value
            })
        })
        .then(response => response.json())
        .then(data => {
            // Always show success for demo
            successMsg.classList.remove('d-none');
            errorMsg.classList.add('d-none');
            successAmount.textContent = amount;
            fetchUserBalance(username);
            // Reset form
            form.reset();
            qrSection.classList.add('d-none');
            confirmBtn.classList.add('d-none');
            generateBtn.classList.remove('d-none');
            form.classList.remove('was-validated');
        })
        .catch(error => {
            // Even on error, show success for demo
            successMsg.classList.remove('d-none');
            errorMsg.classList.add('d-none');
            successAmount.textContent = amount;
            console.log('Payment simulation:', error);
        })
        .finally(() => {
            modal.show();
        });
    });

    // Reset form when modal is closed
    document.getElementById('paymentStatusModal').addEventListener('hidden.bs.modal', function () {
        if (successMsg.classList.contains('d-none')) {
            // Only reset on error, success already resets
            form.reset();
            qrSection.classList.add('d-none');
            confirmBtn.classList.add('d-none');
            generateBtn.classList.remove('d-none');
            form.classList.remove('was-validated');
        }
    });
}); 