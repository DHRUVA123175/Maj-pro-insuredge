from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from bson import ObjectId
import datetime
import json
import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InsurEdgeDatabase:
    def __init__(self, connection_string: str = None):
        """Initialize database connection"""
        self.connection_string = connection_string or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.client = None
        self.db = None
        self.users = None
        self.claims = None
        self.policies = None
        self.analytics = None
        
        self._connect()
        self._setup_collections()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command('ping')
            print("✅ Connected to MongoDB successfully")
            
            self.db = self.client['insuredge_ai']
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def _setup_collections(self):
        """Setup database collections with indexes"""
        try:
            # Users collection
            self.users = self.db['users']
            self.users.create_index("email", unique=True)
            self.users.create_index("phone", sparse=True)
            
            # Claims collection
            self.claims = self.db['claims']
            self.claims.create_index("claim_id", unique=True)
            self.claims.create_index("user_id")
            self.claims.create_index("status")
            self.claims.create_index("created_at")
            
            # Policies collection
            self.policies = self.db['policies']
            self.policies.create_index("policy_id", unique=True)
            self.policies.create_index("user_id")
            self.policies.create_index("vehicle_number")
            
            # Analytics collection
            self.analytics = self.db['analytics']
            self.analytics.create_index("date")
            self.analytics.create_index("type")
            
            print("✅ Database collections and indexes setup complete")
        except Exception as e:
            print(f"❌ Error setting up collections: {e}")
            raise
    
    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user"""
        try:
            # Validate required fields
            required_fields = ['name', 'email', 'password']
            for field in required_fields:
                if field not in user_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Check if user already exists
            existing_user = self.users.find_one({"email": user_data['email']})
            if existing_user:
                raise DuplicateKeyError("User with this email already exists")
            
            # Prepare user document
            user_doc = {
                "name": user_data['name'],
                "email": user_data['email'],
                "password": user_data['password'],  # Should be hashed
                "phone": user_data.get('phone', ''),
                "address": user_data.get('address', ''),
                "active_policies": 0,
                "claims_processed": 0,
                "ai_accuracy": 95.0,
                "pending_renewal": 0,
                "total_claims_amount": 0,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow(),
                "status": "active"
            }
            
            result = self.users.insert_one(user_doc)
            user_doc['_id'] = str(result.inserted_id)
            
            print(f"✅ User created successfully: {user_doc['email']}")
            return user_doc
            
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            raise
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            user = self.users.find_one({"email": email})
            if user:
                user['_id'] = str(user['_id'])
            return user
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            user = self.users.find_one({"_id": ObjectId(user_id)})
            if user:
                user['_id'] = str(user['_id'])
            return user
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user information"""
        try:
            update_data['updated_at'] = datetime.datetime.utcnow()
            result = self.users.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Error updating user: {e}")
            return False
    
    def create_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new claim"""
        try:
            # Validate required fields
            required_fields = ['user_id', 'policy_id', 'vehicle_info', 'claim_type', 'description']
            for field in required_fields:
                if field not in claim_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Generate claim ID
            claim_id = self._generate_claim_id()
            
            # Prepare claim document
            claim_doc = {
                "claim_id": claim_id,
                "user_id": claim_data['user_id'],
                "policy_id": claim_data['policy_id'],
                "vehicle_info": claim_data['vehicle_info'],
                "claim_type": claim_data['claim_type'],
                "description": claim_data['description'],
                "evidence_files": claim_data.get('evidence_files', []),
                "ai_assessment": claim_data.get('ai_assessment', {}),
                "status": "processing",
                "estimated_cost": 0,
                "approved_amount": 0,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow(),
                "processed_at": None,
                "notes": []
            }
            
            result = self.claims.insert_one(claim_doc)
            claim_doc['_id'] = str(result.inserted_id)
            
            # Update user stats
            self.users.update_one(
                {"_id": ObjectId(claim_data['user_id'])},
                {"$inc": {"claims_processed": 1}}
            )
            
            print(f"✅ Claim created successfully: {claim_id}")
            return claim_doc
            
        except Exception as e:
            print(f"❌ Error creating claim: {e}")
            raise
    
    def get_user_claims(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get claims for a specific user"""
        try:
            claims = list(self.claims.find(
                {"user_id": user_id}
            ).sort("created_at", -1).limit(limit))
            
            # Convert ObjectId to string
            for claim in claims:
                claim['_id'] = str(claim['_id'])
            
            return claims
        except Exception as e:
            print(f"❌ Error getting user claims: {e}")
            return []
    
    def get_claim_by_id(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Get claim by ID"""
        try:
            claim = self.claims.find_one({"claim_id": claim_id})
            if claim:
                claim['_id'] = str(claim['_id'])
            return claim
        except Exception as e:
            print(f"❌ Error getting claim: {e}")
            return None
    
    def update_claim(self, claim_id: str, update_data: Dict[str, Any]) -> bool:
        """Update claim information"""
        try:
            update_data['updated_at'] = datetime.datetime.utcnow()
            result = self.claims.update_one(
                {"claim_id": claim_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"❌ Error updating claim: {e}")
            return False
    
    def create_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new insurance policy"""
        try:
            # Validate required fields
            required_fields = ['user_id', 'vehicle_info', 'policy_type', 'premium_amount']
            for field in required_fields:
                if field not in policy_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Generate policy ID
            policy_id = self._generate_policy_id()
            
            # Prepare policy document
            policy_doc = {
                "policy_id": policy_id,
                "user_id": policy_data['user_id'],
                "vehicle_info": policy_data['vehicle_info'],
                "policy_type": policy_data['policy_type'],
                "premium_amount": policy_data['premium_amount'],
                "coverage_amount": policy_data.get('coverage_amount', 0),
                "start_date": policy_data.get('start_date', datetime.datetime.utcnow()),
                "end_date": policy_data.get('end_date'),
                "status": "active",
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
            
            result = self.policies.insert_one(policy_doc)
            policy_doc['_id'] = str(result.inserted_id)
            
            # Update user stats
            self.users.update_one(
                {"_id": ObjectId(policy_data['user_id'])},
                {"$inc": {"active_policies": 1}}
            )
            
            print(f"✅ Policy created successfully: {policy_id}")
            return policy_doc
            
        except Exception as e:
            print(f"❌ Error creating policy: {e}")
            raise
    
    def get_user_policies(self, user_id: str) -> List[Dict[str, Any]]:
        """Get policies for a specific user"""
        try:
            policies = list(self.policies.find(
                {"user_id": user_id, "status": "active"}
            ).sort("created_at", -1))
            
            # Convert ObjectId to string
            for policy in policies:
                policy['_id'] = str(policy['_id'])
            
            return policies
        except Exception as e:
            print(f"❌ Error getting user policies: {e}")
            return []
    
    def log_analytics(self, analytics_data: Dict[str, Any]) -> bool:
        """Log analytics data"""
        try:
            analytics_doc = {
                "type": analytics_data.get('type', 'general'),
                "data": analytics_data.get('data', {}),
                "date": datetime.datetime.utcnow(),
                "user_id": analytics_data.get('user_id'),
                "session_id": analytics_data.get('session_id')
            }
            
            self.analytics.insert_one(analytics_doc)
            return True
        except Exception as e:
            print(f"❌ Error logging analytics: {e}")
            return False
    
    def get_dashboard_stats(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard statistics for a user"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return {}
            
            # Get user claims
            user_claims = self.get_user_claims(user_id, limit=5)
            
            # Calculate total claims amount
            total_amount = sum(claim.get('approved_amount', 0) for claim in user_claims)
            
            # Get active policies count
            active_policies = len(self.get_user_policies(user_id))
            
            stats = {
                "name": user['name'],
                "active_policies": active_policies,
                "claims_processed": user.get('claims_processed', 0),
                "ai_accuracy": user.get('ai_accuracy', 95.0),
                "pending_renewal": user.get('pending_renewal', 0),
                "total_claims_amount": total_amount,
                "recent_claims": user_claims
            }
            
            return stats
        except Exception as e:
            print(f"❌ Error getting dashboard stats: {e}")
            return {}
    
    def _generate_claim_id(self) -> str:
        """Generate unique claim ID"""
        year = datetime.datetime.now().year
        # Get count of claims this year
        count = self.claims.count_documents({
            "created_at": {
                "$gte": datetime.datetime(year, 1, 1),
                "$lt": datetime.datetime(year + 1, 1, 1)
            }
        })
        return f"VC{year}{str(count + 1).zfill(4)}"
    
    def _generate_policy_id(self) -> str:
        """Generate unique policy ID"""
        year = datetime.datetime.now().year
        # Get count of policies this year
        count = self.policies.count_documents({
            "created_at": {
                "$gte": datetime.datetime(year, 1, 1),
                "$lt": datetime.datetime(year + 1, 1, 1)
            }
        })
        return f"VP{year}{str(count + 1).zfill(4)}"
    
    def close_connection(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            print("✅ Database connection closed")

# Global database instance
db = InsurEdgeDatabase() 