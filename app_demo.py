#!/usr/bin/env python3
"""
InsurEdge AI Demo Version - Runs without MongoDB for presentation
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import datetime
import json
from PIL import Image
import io

# Import ML models
from ml_models import damage_detector

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JWT_SECRET_KEY'] = 'demo-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

jwt = JWTManager(app)

# In-memory storage for demo (replaces MongoDB)
demo_users = {}
demo_claims = {}
demo_policies = {}
claim_counter = 1

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_claim_id():
    """Generate unique claim ID"""
    global claim_counter
    year = datetime.datetime.now().year
    claim_id = f"VC{year}{str(claim_counter).zfill(4)}"
    claim_counter += 1
    return claim_id

# HTML Routes
@app.route('/')
@app.route('/index.html')
def index():
    """Serve homepage"""
    return send_from_directory('.', 'index.html')

@app.route('/login.html')
def login_page():
    """Serve login page"""
    return send_from_directory('.', 'login.html')

@app.route('/register.html')
def register_page():
    """Serve register page"""
    return send_from_directory('.', 'register.html')

@app.route('/dashboard.html')
def dashboard_page():
    """Serve dashboard page"""
    return send_from_directory('.', 'dashboard.html')

@app.route('/claim.html')
def claim_page():
    """Serve claim page"""
    return send_from_directory('.', 'claim.html')

@app.route('/styles.css')
def styles():
    """Serve CSS file"""
    return send_from_directory('.', 'styles.css')

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
        user_id = f"user_{len(demo_users) + 1}"
        demo_users[data['email']] = {
            '_id': user_id,
            'name': data['name'],
            'email': data['email'],
            'password': hashed_password,
            'phone': data.get('phone', ''),
            'address': data.get('address', ''),
            'active_policies': 3,
            'claims_processed': 0,
            'ai_accuracy': 95.0,
            'pending_renewal': 1,
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        
        # Generate JWT token
        token = create_access_token(identity=user_id)
        
        return jsonify({
            "message": "User registered successfully",
            "token": token,
            "name": data['name'],
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
        
        # Generate JWT token
        token = create_access_token(identity=user['_id'])
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "name": user['name'],
            "user_id": user['_id']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Dashboard Routes
@app.route('/api/dashboard', methods=['GET'])
@jwt_required()
def dashboard():
    """Get user dashboard data"""
    try:
        user_id = get_jwt_identity()
        
        # Find user
        user = None
        for email, u in demo_users.items():
            if u['_id'] == user_id:
                user = u
                break
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Get user claims
        user_claims = [claim for claim in demo_claims.values() if claim['user_id'] == user_id]
        user_claims.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Calculate stats
        total_amount = sum(claim.get('estimated_cost', 0) for claim in user_claims)
        
        stats = {
            "name": user['name'],
            "active_policies": user.get('active_policies', 3),
            "claims_processed": len(user_claims),
            "ai_accuracy": user.get('ai_accuracy', 95.0),
            "pending_renewal": user.get('pending_renewal', 1),
            "total_claims_amount": total_amount,
            "recent_claims": user_claims[:5]
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Claims Routes
@app.route('/api/claims', methods=['POST'])
@jwt_required()
def submit_claim():
    """Submit a new claim with AI analysis"""
    try:
        user_id = get_jwt_identity()
        
        # Get form data
        policy_id = request.form.get('policy_id')
        vehicle_info = request.form.get('vehicle_info')
        claim_type = request.form.get('claim_type')
        description = request.form.get('description')
        
        # Validate required fields
        if not all([policy_id, vehicle_info, claim_type, description]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Check if files were uploaded
        if 'evidence' not in request.files:
            return jsonify({"error": "Evidence files required"}), 400
        
        files = request.files.getlist('evidence')
        if not files or files[0].filename == '':
            return jsonify({"error": "No files selected"}), 400
        
        # Process uploaded images
        processed_files = []
        all_analyses = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Read image data
                    image_data = file.read()
                    
                    # Analyze damage using ML model
                    analysis = damage_detector.analyze_damage(image_data)
                    all_analyses.append(analysis)
                    
                    # Save file
                    filename = secure_filename(f"claim-{datetime.datetime.utcnow().timestamp()}-{file.filename}")
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    with open(file_path, 'wb') as f:
                        f.write(image_data)
                    
                    processed_files.append(filename)
                    
                except Exception as e:
                    print(f"Error processing image {file.filename}: {e}")
        
        if not all_analyses:
            return jsonify({"error": "Failed to analyze uploaded images"}), 400
        
        # Use the best analysis (highest confidence)
        best_analysis = max(all_analyses, key=lambda x: x.get('confidence_score', 0))
        
        # Generate claim ID
        claim_id = generate_claim_id()
        
        # Create claim
        claim = {
            'claim_id': claim_id,
            'user_id': user_id,
            'policy_id': policy_id,
            'vehicle_info': vehicle_info,
            'claim_type': claim_type,
            'description': description,
            'evidence_files': processed_files,
            'ai_assessment': best_analysis,
            'status': best_analysis.get('claim_status', 'processing'),
            'estimated_cost': best_analysis.get('estimated_cost', 0),
            'created_at': datetime.datetime.utcnow().isoformat()
        }
        
        # Store claim
        demo_claims[claim_id] = claim
        
        return jsonify({
            "message": "Claim submitted successfully",
            "claim_id": claim_id,
            "status": f"Claim {claim_id} under AI analysis",
            "ai_assessment": best_analysis
        }), 201
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/claims', methods=['GET'])
@jwt_required()
def get_claims():
    """Get user's claims"""
    try:
        user_id = get_jwt_identity()
        limit = request.args.get('limit', 10, type=int)
        
        # Get user claims
        user_claims = [claim for claim in demo_claims.values() if claim['user_id'] == user_id]
        user_claims.sort(key=lambda x: x['created_at'], reverse=True)
        
        return jsonify({
            "claims": user_claims[:limit],
            "total": len(user_claims)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# AI Analysis Routes
@app.route('/api/analyze-damage', methods=['POST'])
@jwt_required()
def analyze_damage():
    """Analyze vehicle damage from uploaded images"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Image file required"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400
        
        # Read image data
        image_data = file.read()
        
        # Analyze damage
        analysis = damage_detector.analyze_damage(image_data)
        
        return jsonify({
            "analysis": analysis,
            "filename": file.filename
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# File Routes
@app.route('/uploads/<filename>')
def get_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "InsurEdge AI Demo Backend",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "database": "in-memory (demo mode)",
        "ml_models": "loaded"
    })

# API Root endpoint
@app.route('/api')
def api_home():
    """API Root endpoint"""
    return jsonify({
        "message": "Welcome to InsurEdge AI Demo Backend",
        "version": "2.0.0-demo",
        "mode": "Demo Mode (No MongoDB Required)",
        "endpoints": {
            "auth": "/api/register, /api/login",
            "dashboard": "/api/dashboard",
            "claims": "/api/claims",
            "ai_analysis": "/api/analyze-damage",
            "health": "/api/health"
        }
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting InsurEdge AI Demo Backend...")
    print("📊 ML Models: Loading...")
    print("🗄️  Database: In-Memory Demo Mode")
    print("🌐 Server: Starting on http://localhost:8000")
    print("✅ No MongoDB required - Perfect for demo!")
    
    app.run(debug=True, host='0.0.0.0', port=8000)