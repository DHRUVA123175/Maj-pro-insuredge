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

# Auto-approval configuration
AUTO_APPROVAL_CONFIG = {
    'enabled': True,
    'low_risk_delay_hours': 24,      # Low risk claims auto-approve after 24 hours
    'medium_risk_delay_hours': 72,   # Medium risk claims auto-approve after 72 hours (3 days)
    'high_risk_never': True          # High risk claims NEVER auto-approve
}

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
        'requires_2fa': True,
        'role': 'user'
    },
    'admin@insuredge.ai': {
        'id': 'admin-001',
        'name': 'Admin User',
        'email': 'admin@insuredge.ai',
        'password': generate_password_hash('admin123'),
        'phone': '+91-9999999999',
        'created_at': datetime.now().isoformat(),
        'requires_2fa': False,
        'role': 'admin'
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

def is_admin():
    """Check if current user is admin"""
    user_id = get_current_user()
    if not user_id:
        return False
    
    for email, user_data in demo_users.items():
        if user_data['id'] == user_id and user_data.get('role') == 'admin':
            return True
    return False

def check_auto_approval(claim):
    """Check if a claim should be auto-approved based on time and risk level"""
    if not AUTO_APPROVAL_CONFIG['enabled']:
        return False, claim['status']
    
    # Don't auto-approve already processed claims
    if claim['status'] in ['Approved', 'Rejected', 'Rejected - Fraud Detected']:
        return False, claim['status']
    
    # Get fraud risk level
    fraud_risk = claim.get('ai_results', {}).get('fraud_detection', {}).get('risk_level', 'MEDIUM')
    
    # HIGH risk claims never auto-approve
    if fraud_risk == 'HIGH' and AUTO_APPROVAL_CONFIG['high_risk_never']:
        return False, claim['status']
    
    # Calculate time since claim creation
    created_at = datetime.fromisoformat(claim['created_at'])
    hours_elapsed = (datetime.now() - created_at).total_seconds() / 3600
    
    # Check if enough time has passed based on risk level
    if fraud_risk == 'LOW':
        if hours_elapsed >= AUTO_APPROVAL_CONFIG['low_risk_delay_hours']:
            return True, 'Auto-Approved (Low Risk)'
    elif fraud_risk == 'MEDIUM':
        if hours_elapsed >= AUTO_APPROVAL_CONFIG['medium_risk_delay_hours']:
            return True, 'Auto-Approved (Medium Risk - Reviewed)'
    
    # Calculate remaining time
    if fraud_risk == 'LOW':
        remaining_hours = AUTO_APPROVAL_CONFIG['low_risk_delay_hours'] - hours_elapsed
    elif fraud_risk == 'MEDIUM':
        remaining_hours = AUTO_APPROVAL_CONFIG['medium_risk_delay_hours'] - hours_elapsed
    else:
        remaining_hours = None
    
    # Update status with time remaining
    if remaining_hours and remaining_hours > 0:
        if fraud_risk == 'LOW':
            new_status = f"Processing (Auto-approve in {int(remaining_hours)}h)"
        else:
            new_status = f"Under Investigation (Auto-approve in {int(remaining_hours)}h)"
        return False, new_status
    
    return False, claim['status']

def process_auto_approvals():
    """Process all pending claims for auto-approval"""
    approved_count = 0
    for user_id, user_claims in demo_claims.items():
        for claim_id, claim in user_claims.items():
            should_approve, new_status = check_auto_approval(claim)
            if should_approve:
                demo_claims[user_id][claim_id]['status'] = new_status
                demo_claims[user_id][claim_id]['auto_approved_at'] = datetime.now().isoformat()
                approved_count += 1
            elif new_status != claim['status']:
                demo_claims[user_id][claim_id]['status'] = new_status
    
    return approved_count

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
            'created_at': datetime.now().isoformat(),
            'role': 'user'  # Regular user role
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
        session['user_role'] = user.get('role', 'user')
        
        # Generate JWT token
        token = create_access_token(identity=user['id'])
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "name": user['name'],
            "user_id": user['id'],
            "role": user.get('role', 'user')
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
    
    # Process auto-approvals before showing dashboard
    process_auto_approvals()
    
    # Get only this user's claims for privacy
    user_claims = demo_claims.get(user_id, {})
    claims_list = list(user_claims.values())
    
    # Calculate statistics for this user only
    total_claims = len(claims_list)
    approved_claims = len([c for c in claims_list if 'Approved' in c.get('status', '')])
    processing_claims = len([c for c in claims_list if 'Processing' in c.get('status', '')])
    review_claims = len([c for c in claims_list if 'Review' in c.get('status', '') or 'Investigation' in c.get('status', '')])
    
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
                
                # Comprehensive damage analysis with description
                analysis = detector.analyze_damage(image_data, incident_description, location)
                
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

# Admin Routes
@app.route('/admin')
def admin_dashboard():
    """Admin dashboard page"""
    if not is_admin():
        return redirect(url_for('login_page'))
    
    return render_template('admin.html')

@app.route('/api/admin/stats')
def admin_stats():
    """Get comprehensive admin statistics"""
    if not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Process auto-approvals first
    auto_approved = process_auto_approvals()
    
    # Collect all claims from all users
    all_claims = []
    for user_id, user_claims in demo_claims.items():
        all_claims.extend(user_claims.values())
    
    # Calculate statistics
    total_users = len([u for u in demo_users.values() if u.get('role') != 'admin'])
    total_claims = len(all_claims)
    approved_claims = len([c for c in all_claims if c.get('status') == 'Approved'])
    processing_claims = len([c for c in all_claims if c.get('status') == 'Processing'])
    rejected_claims = len([c for c in all_claims if 'Rejected' in c.get('status', '')])
    fraud_detected = len([c for c in all_claims if c.get('ai_results', {}).get('fraud_detection', {}).get('fraud_detected', False)])
    
    # Calculate total claim amounts
    total_amount = sum([c.get('ai_results', {}).get('cost_estimation', {}).get('estimated_cost', 0) for c in all_claims])
    approved_amount = sum([c.get('ai_results', {}).get('cost_estimation', {}).get('estimated_cost', 0) 
                          for c in all_claims if c.get('status') == 'Approved'])
    
    # Recent activity
    recent_claims = sorted(all_claims, key=lambda x: x.get('created_at', ''), reverse=True)[:10]
    
    return jsonify({
        'total_users': total_users,
        'total_claims': total_claims,
        'approved_claims': approved_claims,
        'processing_claims': processing_claims,
        'rejected_claims': rejected_claims,
        'fraud_detected': fraud_detected,
        'total_amount': total_amount,
        'approved_amount': approved_amount,
        'recent_claims': recent_claims
    })

@app.route('/api/admin/users')
def admin_users():
    """Get all users with their claim statistics"""
    if not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    users_list = []
    for email, user_data in demo_users.items():
        if user_data.get('role') == 'admin':
            continue
        
        user_id = user_data['id']
        user_claims = demo_claims.get(user_id, {})
        
        # Calculate user statistics
        total_claims = len(user_claims)
        approved = len([c for c in user_claims.values() if c.get('status') == 'Approved'])
        total_claimed = sum([c.get('ai_results', {}).get('cost_estimation', {}).get('estimated_cost', 0) 
                            for c in user_claims.values()])
        
        users_list.append({
            'id': user_id,
            'name': user_data['name'],
            'email': email,
            'phone': user_data.get('phone', 'N/A'),
            'created_at': user_data['created_at'],
            'total_claims': total_claims,
            'approved_claims': approved,
            'total_claimed_amount': total_claimed
        })
    
    return jsonify(users_list)

@app.route('/api/admin/claims')
def admin_all_claims():
    """Get all claims from all users"""
    if not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    all_claims = []
    for user_id, user_claims in demo_claims.items():
        for claim in user_claims.values():
            # Add user info to claim
            user_info = None
            for email, user_data in demo_users.items():
                if user_data['id'] == user_id:
                    user_info = {
                        'name': user_data['name'],
                        'email': email,
                        'phone': user_data.get('phone', 'N/A')
                    }
                    break
            
            claim_with_user = claim.copy()
            claim_with_user['user_info'] = user_info
            all_claims.append(claim_with_user)
    
    # Sort by created date
    all_claims.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return jsonify(all_claims)

@app.route('/api/admin/claim/<claim_id>/update', methods=['POST'])
def admin_update_claim(claim_id):
    """Update claim status"""
    if not is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    new_status = data.get('status')
    
    if not new_status:
        return jsonify({'error': 'Status required'}), 400
    
    # Find and update claim
    for user_id, user_claims in demo_claims.items():
        if claim_id in user_claims:
            demo_claims[user_id][claim_id]['status'] = new_status
            return jsonify({'success': True, 'message': 'Claim updated successfully'})
    
    return jsonify({'error': 'Claim not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting InsurEdge AI Demo Server...")
    print("📱 Access the application at: http://localhost:5000")
    print("🔧 Demo Mode: Running without database")
    print("⚡ Debug mode disabled to prevent model reloading")
    
    # Pre-load models once
    get_damage_detector()
    
    app.run(debug=False, host='0.0.0.0', port=5000)