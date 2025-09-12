# InsurEdge AI Enhanced Backend
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

# Import our custom modules
from ml_models import damage_detector
from database import db

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

jwt = JWTManager(app)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_images(files):
    """Process uploaded images and return analysis results"""
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Read image data
                image_data = file.read()
                
                # Analyze damage using ML model
                analysis = damage_detector.analyze_damage(image_data)
                
                # Save file
                filename = secure_filename(f"claim-{datetime.datetime.utcnow().timestamp()}-{file.filename}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(file_path, 'wb') as f:
                    f.write(image_data)
                
                results.append({
                    'filename': filename,
                    'original_name': file.filename,
                    'analysis': analysis
                })
                
            except Exception as e:
                print(f"Error processing image {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'analysis': None
                })
    
    return results

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
        
        # Hash password
        hashed_password = generate_password_hash(data['password'])
        
        # Create user data
        user_data = {
            'name': data['name'],
            'email': data['email'],
            'password': hashed_password,
            'phone': data.get('phone', ''),
            'address': data.get('address', '')
        }
        
        # Create user in database
        user = db.create_user(user_data)
        
        # Generate JWT token
        token = create_access_token(identity=user['_id'])
        
        return jsonify({
            "message": "User registered successfully",
            "token": token,
            "name": user['name'],
            "user_id": user['_id']
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
        
        # Get user from database
        user = db.get_user_by_email(data['email'])
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
        
        # Get dashboard stats from database
        stats = db.get_dashboard_stats(user_id)
        
        if not stats:
            return jsonify({"error": "User not found"}), 404
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile"""
    try:
        user_id = get_jwt_identity()
        user = db.get_user_by_id(user_id)
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Remove sensitive data
        user.pop('password', None)
        
        return jsonify(user)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/user/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Remove sensitive fields that shouldn't be updated via this endpoint
        data.pop('password', None)
        data.pop('email', None)
        data.pop('_id', None)
        
        success = db.update_user(user_id, data)
        
        if success:
            return jsonify({"message": "Profile updated successfully"})
        else:
            return jsonify({"error": "Failed to update profile"}), 400
            
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
        processed_files = process_uploaded_images(files)
        
        # Aggregate AI analysis results
        all_analyses = [f['analysis'] for f in processed_files if f['analysis']]
        
        if not all_analyses:
            return jsonify({"error": "Failed to analyze uploaded images"}), 400
        
        # Use the best analysis (highest confidence)
        best_analysis = max(all_analyses, key=lambda x: x.get('confidence_score', 0))
        
        # Prepare claim data
        claim_data = {
            'user_id': user_id,
            'policy_id': policy_id,
            'vehicle_info': vehicle_info,
            'claim_type': claim_type,
            'description': description,
            'evidence_files': [f['filename'] for f in processed_files if f.get('filename')],
            'ai_assessment': best_analysis
        }
        
        # Create claim in database
        claim = db.create_claim(claim_data)
        
        # Log analytics
        db.log_analytics({
            'type': 'claim_submitted',
            'user_id': user_id,
            'data': {
                'claim_id': claim['claim_id'],
                'claim_type': claim_type,
                'ai_confidence': best_analysis.get('confidence_score', 0)
            }
        })
        
        return jsonify({
            "message": "Claim submitted successfully",
            "claim_id": claim['claim_id'],
            "status": f"Claim {claim['claim_id']} under AI analysis",
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
        
        claims = db.get_user_claims(user_id, limit=limit)
        
        return jsonify({
            "claims": claims,
            "total": len(claims)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/claims/<claim_id>', methods=['GET'])
@jwt_required()
def get_claim(claim_id):
    """Get specific claim details"""
    try:
        user_id = get_jwt_identity()
        claim = db.get_claim_by_id(claim_id)
        
        if not claim:
            return jsonify({"error": "Claim not found"}), 404
        
        # Check if user owns this claim
        if claim['user_id'] != user_id:
            return jsonify({"error": "Unauthorized"}), 403
        
        return jsonify(claim)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Policies Routes
@app.route('/api/policies', methods=['GET'])
@jwt_required()
def get_policies():
    """Get user's policies"""
    try:
        user_id = get_jwt_identity()
        policies = db.get_user_policies(user_id)
        
        return jsonify({
            "policies": policies,
            "total": len(policies)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/policies', methods=['POST'])
@jwt_required()
def create_policy():
    """Create a new policy"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Add user_id to data
        data['user_id'] = user_id
        
        # Create policy
        policy = db.create_policy(data)
        
        return jsonify({
            "message": "Policy created successfully",
            "policy": policy
        }), 201
        
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

# Analytics Routes
@app.route('/api/analytics/claim-stats', methods=['GET'])
@jwt_required()
def get_claim_stats():
    """Get claim statistics for analytics"""
    try:
        user_id = get_jwt_identity()
        
        # Get user claims
        claims = db.get_user_claims(user_id, limit=100)
        
        # Calculate statistics
        total_claims = len(claims)
        approved_claims = len([c for c in claims if c.get('status') == 'approved'])
        total_amount = sum(c.get('approved_amount', 0) for c in claims)
        
        # Claims by type
        claims_by_type = {}
        for claim in claims:
            claim_type = claim.get('claim_type', 'other')
            claims_by_type[claim_type] = claims_by_type.get(claim_type, 0) + 1
        
        stats = {
            "total_claims": total_claims,
            "approved_claims": approved_claims,
            "approval_rate": (approved_claims / total_claims * 100) if total_claims > 0 else 0,
            "total_amount": total_amount,
            "claims_by_type": claims_by_type
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health Check
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db.client.admin.command('ping')
        
        return jsonify({
            "status": "healthy",
            "service": "InsurEdge AI Backend",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "database": "connected",
            "ml_models": "loaded"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

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

# API Root endpoint
@app.route('/api')
def api_home():
    """API Root endpoint"""
    return jsonify({
        "message": "Welcome to InsurEdge AI Backend",
        "version": "2.0.0",
        "endpoints": {
            "auth": "/api/register, /api/login",
            "dashboard": "/api/dashboard",
            "claims": "/api/claims",
            "policies": "/api/policies",
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
    print("🚀 Starting InsurEdge AI Backend...")
    print("📊 ML Models: Loading...")
    print("🗄️  Database: Connecting...")
    print("🌐 Server: Starting on http://localhost:8000")
    
    app.run(debug=True, host='0.0.0.0', port=8000) 