"""
InsurEdge AI - Demo Version with Authentication
Vehicle Insurance Claim Processing System with AI-powered Damage Detection
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
from datetime import datetime, timedelta
import json
from ml_models import VehicleDamageDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = 'demo-secret-key-change-in-production'
app.config['JWT_SECRET_KEY'] = 'jwt-demo-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize JWT
jwt = JWTManager(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ML models with singleton pattern
damage_detector = None

def get_damage_detector():
    global damage_detector
    if damage_detector is None:
        print("🔄 Loading AI models...")
        try:
            damage_detector = VehicleDamageDetector()
            print("✅ AI damage detector loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            damage_detector = None
    return damage_detector

# In-memory storage for demo - organized by user_id for privacy
demo_claims = {
    'demo-user-123': {  # Demo user's claims
        'CLM202410240001': {
            'claim_id': 'CLM202410240001',
            'user_id': 'demo-user-123',
            'policy_number': 'POL123456',
            'incident_date': '2024-10-20',
            'incident_description': 'Minor collision with another vehicle',
            'location': 'Mumbai, Maharashtra',
            'image_path': 'demo-collision.jpg',
            'status': 'Approved',
            'created_at': datetime.now().isoformat(),
            'ai_results': {
                'damage_detection': {'damage_type': 'collision', 'confidence': 0.89, 'damage_detected': True},
                'cost_estimation': {'estimated_cost': 25000, 'currency': 'INR'},
                'fraud_detection': {'risk_level': 'LOW', 'fraud_detected': False, 'fraud_score': 0.1}
            }
        },
        'CLM202410240002': {
            'claim_id': 'CLM202410240002',
            'user_id': 'demo-user-123',
            'policy_number': 'POL789012',
            'incident_date': '2024-10-22',
            'incident_description': 'Hail damage to vehicle roof and hood',
            'location': 'Delhi, NCR',
            'image_path': 'demo-hail.jpg',
            'status': 'Processing',
            'created_at': datetime.now().isoformat(),
            'ai_results': {
                'damage_detection': {'damage_type': 'hail', 'confidence': 0.92, 'damage_detected': True},
                'cost_estimation': {'estimated_cost': 15000, 'currency': 'INR'},
                'fraud_detection': {'risk_level': 'LOW', 'fraud_detected': False, 'fraud_score': 0.05}
            }
        }
    }
}

demo_users = {
    'demo@insuredge.ai': {
        'id': 'demo-user-123',
        'name': 'Demo User',
        'email': 'demo@insuredge.ai',
        'password': generate_password_hash('demo123'),
        'phone': '+91-9876543210',
        'created_at': datetime.now().isoformat(),
        'requires_2fa': True
    }
}
demo_sessions = {}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_user():
    """Get current user from session - for demo purposes"""
    # In a real app, this would validate JWT token
    return session.get('user_id')

def require_auth():
    """Check if user is authenticated"""
    user_id = get_current_user()
    if not user_id:
        return False
    return True

# Authentication Routes
@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Check if user already exists
        if data['email'] in demo_users:
            return jsonify({"error": "User already exists"}), 400
        
        # Hash password
        hashed_password = generate_password_hash(data['password'])
        
        # Create user
        user_id = str(uuid.uuid4())
        user_data = {
            'id': user_id,
            'name': data['name'],
            'email': data['email'],
            'password': hashed_password,
            'phone': data.get('phone', ''),
            'created_at': datetime.now().isoformat()
        }
        
        demo_users[data['email']] = user_data
        
        # Set session for demo
        session['user_id'] = user_id
        session['user_name'] = user_data['name']
        
        # Initialize empty claims for new user
        demo_claims[user_id] = {}
        
        # Generate JWT token
        token = create_access_token(identity=user_id)
        
        return jsonify({
            "message": "User registered successfully",
            "token": token,
            "name": user_data['name'],
            "user_id": user_id
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        
        if not data.get('email') or not data.get('password'):
            return jsonify({"error": "Email and password required"}), 400
        
        # Get user
        user = demo_users.get(data['email'])
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Check password
        if not check_password_hash(user['password'], data['password']):
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Set session for demo (in production, use JWT)
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        
        # Generate JWT token
        token = create_access_token(identity=user['id'])
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "name": user['name'],
            "user_id": user['id']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile"""
    try:
        user_id = get_jwt_identity()
        
        # Find user by ID
        user = None
        for email, user_data in demo_users.items():
            if user_data['id'] == user_id:
                user = user_data.copy()
                break
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Remove sensitive data
        user.pop('password', None)
        
        return jsonify(user)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    # Check authentication
    user_id = get_current_user()
    if not user_id:
        return redirect(url_for('login_page'))
    
    # Get only this user's claims for privacy
    user_claims = demo_claims.get(user_id, {})
    claims_list = list(user_claims.values())
    
    # Calculate statistics for this user only
    total_claims = len(claims_list)
    approved_claims = len([c for c in claims_list if c.get('status') == 'Approved'])
    processing_claims = len([c for c in claims_list if c.get('status') == 'Processing'])
    review_claims = len([c for c in claims_list if 'Review' in c.get('status', '')])
    
    stats = {
        'total_claims': total_claims,
        'approved_claims': approved_claims,
        'processing_claims': processing_claims,
        'review_claims': review_claims
    }
    
    return render_template('dashboard.html', claims=claims_list, stats=stats)

@app.route('/claim')
def claim():
    # Check authentication
    user_id = get_current_user()
    if not user_id:
        return redirect(url_for('login_page'))
    
    return render_template('claim.html')

def validate_incident_date(date_string):
    """Validate incident date for fraud detection"""
    try:
        # Parse the date
        incident_date = datetime.strptime(date_string, '%Y-%m-%d')
        current_date = datetime.now()
        
        # Check if date is in the future
        if incident_date.date() > current_date.date():
            return False, "Incident date cannot be in the future"
        
        # Check if date is too far in the past (more than 2 years)
        two_years_ago = current_date - timedelta(days=730)
        if incident_date < two_years_ago:
            return False, "Incident date is too old (more than 2 years)"
        
        return True, "Valid date"
        
    except ValueError:
        return False, "Invalid date format"

@app.route('/submit_claim', methods=['POST'])
def submit_claim():
    try:
        # Get form data
        policy_number = request.form.get('policy_number')
        incident_date = request.form.get('incident_date')
        incident_description = request.form.get('incident_description')
        location = request.form.get('location')
        
        # Validate incident date
        date_valid, date_message = validate_incident_date(incident_date)
        if not date_valid:
            return jsonify({'error': f'Invalid incident date: {date_message}'}), 400
        
        # Handle file upload
        if 'damage_image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['damage_image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or GIF'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Generate claim ID
        claim_id = f"CLM{datetime.now().strftime('%Y%m%d')}{len(demo_claims) + 1:04d}"
        
        # Process image with AI models
        results = {}
        detector = get_damage_detector()
        if detector:
            try:
                # Read image data
                with open(filepath, 'rb') as f:
                    image_data = f.read()
                
                # Comprehensive damage analysis
                analysis = detector.analyze_damage(image_data)
                
                # Format results for frontend
                results = {
                    'damage_detection': {
                        'damage_type': analysis.get('damage_type', 'unknown'),
                        'confidence': analysis.get('confidence_score', 0) / 100,
                        'damage_detected': analysis.get('damage_detected', False),
                        'severity': analysis.get('damage_severity', 'unknown')
                    },
                    'cost_estimation': {
                        'estimated_cost': analysis.get('estimated_cost', 0),
                        'currency': 'INR'
                    },
                    'fraud_detection': analysis.get('fraud_analysis', {
                        'risk_level': 'LOW', 
                        'fraud_detected': False,
                        'fraud_score': 0.0
                    }),
                    'recommendations': analysis.get('recommendations', [])
                }
                
                # Additional fraud checks for date
                incident_dt = datetime.strptime(incident_date, '%Y-%m-%d')
                days_since_incident = (datetime.now() - incident_dt).days
                
                # Flag suspicious timing
                if days_since_incident == 0:  # Same day claim
                    results['fraud_detection']['fraud_score'] += 0.2
                    results['fraud_detection']['suspicious_factors'] = results['fraud_detection'].get('suspicious_factors', [])
                    results['fraud_detection']['suspicious_factors'].append('Same-day claim submission')
                
                # Determine overall status based on analysis
                fraud_analysis = analysis.get('fraud_analysis', {})
                if fraud_analysis.get('risk_level') == 'HIGH':
                    status = 'Rejected - Fraud Detected'
                elif fraud_analysis.get('risk_level') == 'MEDIUM':
                    status = 'Under Investigation'
                elif analysis.get('damage_severity') == 'high':
                    status = 'Requires Manual Review'
                elif not analysis.get('damage_detected', False):
                    status = 'No Damage Detected'
                else:
                    status = 'Approved'
                    
            except Exception as e:
                print(f"AI processing error: {e}")
                results = {
                    'damage_detection': {'damage_type': 'unknown', 'confidence': 0.5, 'damage_detected': True},
                    'cost_estimation': {'estimated_cost': 15000, 'currency': 'INR'},
                    'fraud_detection': {'risk_level': 'LOW', 'fraud_detected': False, 'fraud_score': 0.0},
                    'recommendations': ['Error in AI analysis - manual review required']
                }
                status = 'Processing'
        else:
            # Fallback if models not loaded
            results = {
                'damage_detection': {'damage_type': 'moderate', 'confidence': 0.75, 'damage_detected': True},
                'cost_estimation': {'estimated_cost': 25000, 'currency': 'INR'},
                'fraud_detection': {'risk_level': 'LOW', 'fraud_detected': False, 'fraud_score': 0.0},
                'recommendations': ['AI models not available - manual review required']
            }
            status = 'Processing'
        
        # Get current user
        user_id = get_current_user()
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        # Create claim record
        claim_data = {
            'claim_id': claim_id,
            'user_id': user_id,
            'policy_number': policy_number,
            'incident_date': incident_date,
            'incident_description': incident_description,
            'location': location,
            'image_path': unique_filename,
            'status': status,
            'created_at': datetime.now().isoformat(),
            'ai_results': results
        }
        
        # Store in user's private claims
        if user_id not in demo_claims:
            demo_claims[user_id] = {}
        demo_claims[user_id][claim_id] = claim_data
        
        return jsonify({
            'success': True,
            'claim_id': claim_id,
            'message': 'Claim submitted successfully',
            'results': results,
            'status': status
        })
        
    except Exception as e:
        print(f"Error processing claim: {e}")
        return jsonify({'error': f'Error processing claim: {str(e)}'}), 500

@app.route('/claim_status/<claim_id>')
def claim_status(claim_id):
    claim = demo_claims.get(claim_id)
    if not claim:
        return jsonify({'error': 'Claim not found'}), 404
    
    return jsonify(claim)

@app.route('/api/claims')
def api_claims():
    return jsonify(list(demo_claims.values()))

if __name__ == '__main__':
    print("🚀 Starting InsurEdge AI Demo Server...")
    print("📱 Access the application at: http://localhost:5000")
    print("🔧 Demo Mode: Running without database")
    print("⚡ Debug mode disabled to prevent model reloading")
    
    # Pre-load models once
    get_damage_detector()
    
    app.run(debug=False, host='0.0.0.0', port=5000)