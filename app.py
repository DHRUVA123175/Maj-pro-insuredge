# InsurEdge AI Backend using Flask

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson import ObjectId
import os
import datetime
import random

app = Flask(__name__)
CORS(app)

# Config
app.config['JWT_SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

jwt = JWTManager(app)

# DB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['insuredge_ai']
users = db['users']
claims = db['claims']

# Helper Functions
def generate_claim_id():
    year = datetime.datetime.now().year
    rand = str(random.randint(0, 999)).zfill(3)
    return f"VC{year}{rand}"

def simulate_ai_assessment():
    severity = random.choice(['low', 'medium', 'high'])
    cost = random.randint(10000, 110000)
    confidence = random.randint(80, 100)
    return {
        "damage_detected": True,
        "damage_severity": severity,
        "estimated_cost": cost,
        "confidence_score": confidence
    }

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if users.find_one({"email": data['email']}):
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = generate_password_hash(data['password'])
    user = {
        "name": data['name'],
        "email": data['email'],
        "password": hashed_pw,
        "active_policies": 3,
        "claims_processed": 0,
        "ai_accuracy": 95,
        "pending_renewal": 1,
        "created_at": datetime.datetime.utcnow()
    }
    result = users.insert_one(user)
    token = create_access_token(identity=str(result.inserted_id))
    return jsonify({"message": "User registered", "token": token, "name": user['name']}), 201

@app.route('/api/vehicle-login', methods=['POST'])
def login():
    data = request.get_json()
    user = users.find_one({"email": data['email']})
    if not user or not check_password_hash(user['password'], data['password']):
        return jsonify({"error": "Invalid credentials"}), 400

    token = create_access_token(identity=str(user['_id']))
    return jsonify({"message": "Login successful", "token": token, "name": user['name']})

@app.route('/api/vehicle-dashboard')
@jwt_required()
def dashboard():
    uid = get_jwt_identity()
    user = users.find_one({"_id": ObjectId(uid)})
    user_claims = list(claims.find({"user_id": uid}).sort("created_at", -1))

    recent = [{
        "claim_id": c['claim_id'],
        "vehicle_info": c['vehicle_info'],
        "status": c['status'],
        "created_at": c['created_at'],
        "ai_assessment": c['ai_assessment']
    } for c in user_claims[:3]]

    return jsonify({
        "name": user['name'],
        "active_policies": user['active_policies'],
        "claims_processed": len(user_claims),
        "ai_accuracy": user['ai_accuracy'],
        "pending_renewal": user['pending_renewal'],
        "recent_claims": recent
    })

@app.route('/api/submit-vehicle-claim', methods=['POST'])
@jwt_required()
def submit_claim():
    uid = get_jwt_identity()
    policy_id = request.form.get('policy_id')
    vehicle_info = request.form.get('vehicle_info')
    claim_type = request.form.get('claim_type')
    description = request.form.get('description')

    if 'evidence' not in request.files:
        return jsonify({"error": "Evidence files required"}), 400

    files = request.files.getlist('evidence')
    saved_files = []
    for f in files:
        filename = secure_filename(f"claim-{datetime.datetime.utcnow().timestamp()}-{f.filename}")
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        saved_files.append(filename)

    assessment = simulate_ai_assessment()
    claim = {
        "user_id": uid,
        "policy_id": policy_id,
        "vehicle_info": vehicle_info,
        "claim_type": claim_type,
        "description": description,
        "evidence_files": saved_files,
        "status": "processing",
        "claim_id": generate_claim_id(),
        "ai_assessment": assessment,
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow()
    }

    claims.insert_one(claim)
    users.update_one({"_id": ObjectId(uid)}, {"$inc": {"claims_processed": 1}})

    return jsonify({
        "message": "Claim submitted",
        "status": f"Claim {claim['claim_id']} under AI analysis",
        "claim_id": claim['claim_id'],
        "ai_assessment": assessment
    })

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/health')
def health():
    return jsonify({"status": "OK", "service": "InsurEdge Flask Backend"})

@app.route('/')
def home():
    return "<h2>Welcome to InsurEdge AI Backend 🎉</h2>"    

if __name__ == '__main__':
    app.run(debug=True, port=8000)
