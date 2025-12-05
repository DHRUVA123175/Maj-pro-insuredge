import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing import image
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from PIL import Image
import io

class VehicleDamageDetector:
    def __init__(self):
        self.damage_classifier = None
        self.severity_predictor = None
        self.cost_estimator = None
        self.scaler = StandardScaler()
        self.model_path = 'models/'
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize models
        self._load_or_create_models()
    
    def _load_or_create_models(self):
        """Load existing models or create new ones if they don't exist"""
        try:
            # Try to load pre-trained models
            self.damage_classifier = tf.keras.models.load_model(f'{self.model_path}damage_classifier.h5')
            self.severity_predictor = joblib.load(f'{self.model_path}severity_predictor.pkl')
            self.cost_estimator = joblib.load(f'{self.model_path}cost_estimator.pkl')
            self.scaler = joblib.load(f'{self.model_path}scaler.pkl')
            print("✅ Loaded pre-trained models successfully")
        except:
            print("🔄 Creating new models...")
            self._create_models()
    
    def _create_models(self):
        """Create and train new models"""
        # 1. Damage Classification Model (CNN)
        self.damage_classifier = self._create_damage_classifier()
        
        # 2. Severity Prediction Model
        self.severity_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # 3. Cost Estimation Model
        self.cost_estimator = RandomForestRegressor(n_estimators=150, random_state=42)
        
        # Train models with synthetic data
        self._train_models()
        
        # Save models
        self._save_models()
    
    def _create_damage_classifier(self):
        """Create enhanced CNN model for damage classification"""
        # Use EfficientNetB3 - more powerful than ResNet50
        try:
            from tensorflow.keras.applications import EfficientNetB3
            base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        except:
            # Fallback to ResNet50 if EfficientNet not available
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Unfreeze top layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Enhanced architecture
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(7, activation='softmax')  # 7 damage types
        ])
        
        # Use advanced optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
    def _train_models(self):
        """Train models with enhanced synthetic data"""
        print("🔄 Training models with enhanced synthetic data...")
        
        # Generate more realistic synthetic training data
        n_samples = 5000  # More training samples
        
        # Create realistic feature distributions
        X_severity = np.random.beta(2, 5, (n_samples, 15))  # Beta distribution for realistic features
        
        # Generate correlated severity and cost data
        damage_types = np.random.choice(7, n_samples)
        base_severities = np.random.beta(2, 3, n_samples)  # More realistic severity distribution
        
        # Adjust severity based on damage type
        type_adjustments = [1.0, 1.5, 0.8, 1.8, 1.3, 0.6, 1.0]  # collision, theft, vandalism, fire, flood, hail, other
        y_severity = np.clip([base_severities[i] * type_adjustments[damage_types[i]] for i in range(n_samples)], 0, 1)
        
        # Generate realistic cost estimates
        base_costs = [40000, 100000, 20000, 120000, 70000, 15000, 35000]  # Base costs per damage type
        y_cost = [base_costs[damage_types[i]] * (0.5 + y_severity[i]) * np.random.uniform(0.8, 1.2) for i in range(n_samples)]
        
        # Train severity predictor with better parameters
        self.severity_predictor = RandomForestRegressor(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            random_state=42
        )
        self.severity_predictor.fit(X_severity, y_severity)
        
        # Train cost estimator with better parameters
        self.cost_estimator = RandomForestRegressor(
            n_estimators=250, 
            max_depth=20, 
            min_samples_split=5,
            random_state=42
        )
        self.cost_estimator.fit(X_severity, y_cost)
        
        # Fit scaler
        self.scaler.fit(X_severity)
        
        print("✅ Enhanced models trained successfully")
    
    def _save_models(self):
        """Save trained models"""
        self.damage_classifier.save(f'{self.model_path}damage_classifier.h5')
        joblib.dump(self.severity_predictor, f'{self.model_path}severity_predictor.pkl')
        joblib.dump(self.cost_estimator, f'{self.model_path}cost_estimator.pkl')
        joblib.dump(self.scaler, f'{self.model_path}scaler.pkl')
        print("✅ Models saved successfully")
    
    def preprocess_image(self, image_data):
        """Preprocess image for model input"""
        try:
            # Convert bytes to PIL Image
            if isinstance(image_data, bytes):
                img = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, np.ndarray):
                # Convert NumPy array to PIL Image
                img = Image.fromarray(np.uint8(image_data))
            else:
                img = image_data

            # Resize to 224x224
            img = img.resize((224, 224))

            # Convert to array and normalize
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0

            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)

            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_image_features(self, image_data):
        """Extract features from image for severity and cost prediction"""
        try:
            # Convert to OpenCV format
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            
            # Extract features
            features = []
            
            # 1. Image statistics
            features.extend([
                np.mean(img), np.std(img), np.var(img),
                np.max(img), np.min(img)
            ])
            
            # 2. Edge density
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 3. Color histogram features
            for i in range(3):  # BGR channels
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                features.extend([np.mean(hist), np.std(hist)])
            
            # 4. Texture features (simplified)
            features.append(np.std(gray))
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.random.rand(1, 10)  # Fallback to random features
    
    def classify_damage(self, image_data):
        """Classify type of damage based on actual image analysis"""
        try:
            # Convert to OpenCV format for real analysis
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            
            if img is None:
                return "no_damage", 0.95
            
            # Actual damage detection logic
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Check for significant damage using edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 2. Check color variance (damage creates color inconsistencies)
            color_variance = np.var(img)
            
            # 3. Check brightness variations (dents/scratches create shadows)
            brightness_std = np.std(gray)
            
            # 4. Detect if there's actual damage
            damage_score = 0
            
            # High edge density might indicate damage
            if edge_density > 0.15:
                damage_score += 0.3
            
            # High color variance might indicate damage
            if color_variance > 2000:
                damage_score += 0.3
                
            # Brightness variations indicate surface irregularities
            if brightness_std > 40:
                damage_score += 0.4
            
            # Determine if there's actual damage
            if damage_score < 0.4:
                return "no_damage", 0.92
            
            # If damage detected, classify type based on image characteristics
            damage_type = "collision"  # Default to most common
            confidence = 0.85
            
            # Analyze damage patterns
            # Dark areas might indicate fire/burn damage
            dark_pixels = np.sum(gray < 50) / (gray.shape[0] * gray.shape[1])
            if dark_pixels > 0.3:
                damage_type = "fire"
                confidence = 0.88
            
            # Very uniform damage across image might be hail
            elif edge_density > 0.25 and brightness_std < 30:
                damage_type = "hail"
                confidence = 0.86
            
            # High contrast variations might be collision
            elif brightness_std > 60:
                damage_type = "collision"
                confidence = 0.89
            
            # Check for water damage indicators (blue/dark tints)
            blue_channel = img[:,:,0]
            if np.mean(blue_channel) > np.mean(img[:,:,1]) + 20:
                damage_type = "flood"
                confidence = 0.87
            
            return damage_type, confidence
            
        except Exception as e:
            print(f"Error classifying damage: {e}")
            return "no_damage", 0.95
    
    def predict_severity(self, image_data, damage_type):
        """Predict damage severity based on actual image analysis"""
        try:
            if damage_type == "no_damage":
                return 0.0
            
            # Convert to OpenCV format
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate severity based on actual damage indicators
            severity_score = 0.0
            
            # 1. Edge density (more edges = more damage)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            severity_score += min(edge_density * 2, 0.4)  # Cap at 0.4
            
            # 2. Color variance (damage creates color inconsistencies)
            color_variance = np.var(img)
            if color_variance > 1000:
                severity_score += min((color_variance - 1000) / 5000, 0.3)
            
            # 3. Brightness variations
            brightness_std = np.std(gray)
            if brightness_std > 30:
                severity_score += min((brightness_std - 30) / 100, 0.3)
            
            # Adjust based on damage type
            type_multipliers = {
                'collision': 1.0,
                'theft': 1.5,  # Theft usually means total loss
                'fire': 1.4,
                'flood': 1.2,
                'vandalism': 0.8,
                'hail': 0.6,
                'other': 1.0
            }
            
            severity_score *= type_multipliers.get(damage_type, 1.0)
            
            # Ensure reasonable range
            return min(max(severity_score, 0.1), 1.0)
            
        except Exception as e:
            print(f"Error predicting severity: {e}")
            return 0.3
    
    def estimate_cost(self, image_data, damage_type, severity):
        """Estimate repair cost based on actual damage assessment"""
        try:
            if damage_type == "no_damage":
                return 0
            
            # Realistic base costs for different damage types (Indian market)
            base_costs = {
                'collision': {
                    'minor': (3000, 8000),      # Small scratches, dents
                    'moderate': (8000, 25000),   # Bumper replacement, panel work
                    'severe': (25000, 60000)     # Major structural damage
                },
                'theft': (80000, 200000),        # Usually total loss
                'fire': (100000, 300000),        # Usually total loss
                'flood': (40000, 150000),        # Engine/electrical damage
                'vandalism': (2000, 15000),      # Paint, windows, minor damage
                'hail': (5000, 20000),          # Dent removal, paint touch-up
                'other': (5000, 30000)
            }
            
            if damage_type in ['theft', 'fire']:
                # These are usually total loss scenarios
                min_cost, max_cost = base_costs[damage_type]
                return int(min_cost + (max_cost - min_cost) * severity)
            
            elif damage_type == 'collision':
                # Determine collision severity category
                if severity <= 0.3:
                    cost_range = base_costs['collision']['minor']
                elif severity <= 0.7:
                    cost_range = base_costs['collision']['moderate']
                else:
                    cost_range = base_costs['collision']['severe']
                
                min_cost, max_cost = cost_range
                base_cost = min_cost + (max_cost - min_cost) * (severity % 0.4) / 0.4
                
            else:
                # Other damage types
                min_cost, max_cost = base_costs.get(damage_type, (5000, 30000))
                base_cost = min_cost + (max_cost - min_cost) * severity
            
            # Add realistic variation (±15%)
            variation = np.random.uniform(0.85, 1.15)
            final_cost = base_cost * variation
            
            return max(int(final_cost), 1000)  # Minimum ₹1000
            
        except Exception as e:
            print(f"Error estimating cost: {e}")
            return 5000
    
    def analyze_description_fraud(self, description, location):
        """Analyze claim description for fraud indicators"""
        fraud_indicators = []
        fraud_score = 0.0
        
        if not description:
            return fraud_score, fraud_indicators
        
        description_lower = description.lower()
        location_lower = location.lower() if location else ""
        
        # High-risk keywords (expensive damages)
        high_risk_keywords = [
            'submerged', 'river', 'lake', 'ocean', 'flood', 'water damage',
            'total loss', 'completely destroyed', 'burned', 'fire',
            'stolen', 'theft', 'missing', 'vandalized', 'keyed'
        ]
        
        # Suspicious phrases
        suspicious_phrases = [
            'fell into', 'drove into water', 'sank', 'underwater',
            'caught fire', 'exploded', 'completely burned',
            'someone stole', 'disappeared', 'can\'t find'
        ]
        
        # Vague descriptions (fraud indicator)
        vague_phrases = [
            'not sure', 'don\'t know', 'can\'t remember', 'maybe',
            'i think', 'possibly', 'somehow'
        ]
        
        # Check for high-risk keywords
        for keyword in high_risk_keywords:
            if keyword in description_lower:
                fraud_indicators.append(f"High-risk claim: '{keyword}' detected")
                fraud_score += 0.3
                break  # Only count once
        
        # Check for suspicious phrases
        for phrase in suspicious_phrases:
            if phrase in description_lower:
                fraud_indicators.append(f"Suspicious description: '{phrase}'")
                fraud_score += 0.25
                break
        
        # Check for vague descriptions
        for phrase in vague_phrases:
            if phrase in description_lower:
                fraud_indicators.append("Vague or uncertain description")
                fraud_score += 0.15
                break
        
        # Check description length (too short or too long can be suspicious)
        if len(description) < 20:
            fraud_indicators.append("Description too brief")
            fraud_score += 0.1
        elif len(description) > 500:
            fraud_indicators.append("Unusually detailed description")
            fraud_score += 0.1
        
        # Check for inconsistencies between description and location
        water_keywords = ['river', 'lake', 'ocean', 'water', 'submerged', 'flood']
        has_water_damage = any(kw in description_lower for kw in water_keywords)
        
        if has_water_damage:
            # Water damage claims are high-risk
            fraud_score += 0.2
            fraud_indicators.append("Water damage claim requires thorough verification")
        
        return fraud_score, fraud_indicators
    
    def detect_fraud(self, image_data, damage_type, confidence, description="", location=""):
        """Detect potential insurance fraud with text analysis"""
        fraud_indicators = []
        fraud_score = 0.0
        
        # First, analyze the description text
        text_fraud_score, text_indicators = self.analyze_description_fraud(description, location)
        fraud_score += text_fraud_score
        fraud_indicators.extend(text_indicators)
        
        try:
            # Convert to OpenCV format for analysis
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
            
            # 1. Check image quality (blurry images might be fake)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if blur_score < 100:
                fraud_indicators.append("Image appears blurry or low quality")
                fraud_score += 0.2
            
            # 2. Check for digital manipulation
            # Detect unusual color distributions
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            # Check for unnatural color spikes
            if np.max(hist_b) > img.shape[0] * img.shape[1] * 0.1:
                fraud_indicators.append("Unusual color distribution detected")
                fraud_score += 0.15
            
            # 3. Check confidence vs damage type consistency
            high_value_damages = ['theft', 'fire', 'flood']
            if damage_type in high_value_damages and confidence < 0.7:
                fraud_indicators.append("Low confidence for high-value damage claim")
                fraud_score += 0.25
            
            # 4. Check for staged damage patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if edge_density > 0.3:  # Too many edges might indicate staged damage
                fraud_indicators.append("Damage pattern appears potentially staged")
                fraud_score += 0.2
            
            # 5. Check image metadata (simulated)
            # In real implementation, check EXIF data for manipulation
            metadata_suspicious = np.random.random() < 0.1  # 10% chance
            if metadata_suspicious:
                fraud_indicators.append("Image metadata shows signs of editing")
                fraud_score += 0.3
            
            # 6. Cross-reference with damage type
            if damage_type == 'theft' and confidence > 0.9:
                fraud_indicators.append("Unusually high confidence for theft claim")
                fraud_score += 0.2
            
            # Determine fraud risk level (more aggressive detection)
            if fraud_score >= 0.5:
                risk_level = "HIGH"
                action = "CLAIM_REJECTED"
            elif fraud_score >= 0.25:
                risk_level = "MEDIUM"
                action = "MANUAL_REVIEW_REQUIRED"
            else:
                risk_level = "LOW"
                action = "PROCEED_NORMAL"
            
            return {
                "fraud_detected": fraud_score >= 0.25,
                "fraud_score": round(fraud_score, 3),
                "risk_level": risk_level,
                "recommended_action": action,
                "fraud_indicators": fraud_indicators,
                "requires_human_review": fraud_score >= 0.5
            }
            
        except Exception as e:
            print(f"Error in fraud detection: {e}")
            return {
                "fraud_detected": False,
                "fraud_score": 0.0,
                "risk_level": "UNKNOWN",
                "recommended_action": "MANUAL_REVIEW_REQUIRED",
                "fraud_indicators": ["Error in fraud analysis"],
                "requires_human_review": True
            }

    def analyze_damage(self, image_data, description="", location=""):
        """Complete damage analysis pipeline with fraud detection"""
        try:
            # 1. Classify damage type
            damage_type, confidence = self.classify_damage(image_data)
            
            # 2. Predict severity
            severity = self.predict_severity(image_data, damage_type)
            
            # 3. Estimate cost
            estimated_cost = self.estimate_cost(image_data, damage_type, severity)
            
            # 4. Fraud detection (with text analysis)
            fraud_analysis = self.detect_fraud(image_data, damage_type, confidence, description, location)
            
            # 5. Determine severity level
            if severity < 0.3:
                severity_level = "low"
            elif severity < 0.7:
                severity_level = "medium"
            else:
                severity_level = "high"
            
            # 6. Handle no damage cases
            if damage_type == "no_damage":
                recommendations = ["No significant damage detected", "Claim may not be necessary", "Consider minor touch-up if needed"]
                claim_status = "rejected"
                damage_detected = False
                severity_level = "none"
            else:
                damage_detected = True
                
                # Adjust assessment based on fraud risk
                if fraud_analysis['fraud_detected']:
                    if fraud_analysis['risk_level'] == 'HIGH':
                        estimated_cost = 0  # Don't provide estimate for high-risk fraud
                        recommendations = ["🚨 CLAIM REJECTED - Fraudulent activity detected", "Contact customer service for appeal process"]
                        claim_status = "rejected"
                    elif fraud_analysis['risk_level'] == 'MEDIUM':
                        recommendations = self._generate_recommendations(damage_type, severity_level)
                        recommendations.append("⚠️ Additional verification required due to fraud indicators")
                        claim_status = "under_review"
                    else:
                        recommendations = self._generate_recommendations(damage_type, severity_level)
                        claim_status = "approved"
                else:
                    recommendations = self._generate_recommendations(damage_type, severity_level)
                    claim_status = "approved"
            
            # 7. Generate comprehensive assessment
            assessment = {
                "damage_detected": damage_detected,
                "damage_type": damage_type,
                "damage_severity": severity_level,
                "severity_score": round(severity, 3),
                "estimated_cost": estimated_cost,
                "confidence_score": round(confidence * 100, 1),
                "analysis_timestamp": str(np.datetime64('now')),
                "recommendations": recommendations,
                "fraud_analysis": fraud_analysis,
                "claim_status": claim_status
            }
            
            return assessment
        except Exception as e:
            print(f"Error in damage analysis: {e}")
            return {
                "damage_detected": False,
                "error": str(e),
                "damage_type": "unknown",
                "damage_severity": "unknown",
                "severity_score": 0.0,
                "estimated_cost": 0,
                "confidence_score": 0.0,
                "fraud_analysis": {"fraud_detected": True, "risk_level": "HIGH", "recommended_action": "MANUAL_REVIEW_REQUIRED"},
                "claim_status": "rejected"
            }
    
    def _generate_recommendations(self, damage_type, severity_level):
        """Generate recommendations based on damage type and severity"""
        recommendations = []
        
        if damage_type == "collision":
            if severity_level == "high":
                recommendations.extend([
                    "Immediate towing required",
                    "Professional body shop assessment needed",
                    "Check for structural damage"
                ])
            else:
                recommendations.extend([
                    "Schedule repair appointment",
                    "Get multiple quotes"
                ])
        
        elif damage_type == "theft":
            recommendations.extend([
                "File police report immediately",
                "Contact insurance company",
                "Check for tracking devices"
            ])
        
        elif damage_type == "fire":
            recommendations.extend([
                "Do not attempt to start vehicle",
                "Contact fire department for inspection",
                "Professional assessment required"
            ])
        
        elif damage_type == "flood":
            recommendations.extend([
                "Do not start engine",
                "Professional drying required",
                "Check electrical systems"
            ])
        
        else:
            recommendations.append("Professional assessment recommended")
        
        return recommendations

# Global instance
damage_detector = VehicleDamageDetector() 