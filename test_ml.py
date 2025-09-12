#!/usr/bin/env python3
"""
Test script for InsurEdge AI ML Models
Tests damage detection and analysis functionality
"""

import numpy as np
from PIL import Image
import io
import time

def create_test_image(width=224, height=224):
    """Create a test image for damage simulation"""
    # Create a simple test image with some "damage" patterns
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some "damage" patterns (darker areas)
    damage_areas = [
        (50, 50, 100, 100),   # Top-left damage
        (150, 150, 200, 200), # Bottom-right damage
    ]
    
    for x1, y1, x2, y2 in damage_areas:
        img_array[y1:y2, x1:x2] = img_array[y1:y2, x1:x2] // 3  # Darker area
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_ml_models():
    """Test the ML models functionality"""
    print("🧪 Testing InsurEdge AI ML Models")
    print("=" * 50)
    
    try:
        # Import the damage detector
        from ml_models import damage_detector
        print("✅ ML models imported successfully")
        
        # Create test image
        print("\n📸 Creating test image...")
        test_image = create_test_image()
        print("✅ Test image created")
        
        # Test damage analysis
        print("\n🔍 Testing damage analysis...")
        start_time = time.time()
        
        analysis = damage_detector.analyze_damage(test_image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Analysis completed in {processing_time:.2f} seconds")
        
        # Display results
        print("\n📊 Analysis Results:")
        print("-" * 30)
        print(f"Damage Detected: {analysis.get('damage_detected', 'Unknown')}")
        print(f"Damage Type: {analysis.get('damage_type', 'Unknown')}")
        print(f"Severity Level: {analysis.get('damage_severity', 'Unknown')}")
        print(f"Severity Score: {analysis.get('severity_score', 'Unknown')}")
        print(f"Estimated Cost: ₹{analysis.get('estimated_cost', 'Unknown'):,}")
        print(f"Confidence Score: {analysis.get('confidence_score', 'Unknown')}%")
        
        if 'recommendations' in analysis:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Test individual components
        print("\n🔧 Testing individual components...")
        
        # Test damage classification
        damage_type, confidence = damage_detector.classify_damage(test_image)
        print(f"✅ Damage Classification: {damage_type} (confidence: {confidence:.2f})")
        
        # Test severity prediction
        severity = damage_detector.predict_severity(test_image, damage_type)
        print(f"✅ Severity Prediction: {severity:.3f}")
        
        # Test cost estimation
        cost = damage_detector.estimate_cost(test_image, damage_type, severity)
        print(f"✅ Cost Estimation: ₹{cost:,}")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import ML models: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_database_connection():
    """Test database connection"""
    print("\n🗄️  Testing Database Connection")
    print("=" * 50)
    
    try:
        from database import db
        print("✅ Database module imported successfully")
        
        # Test connection
        db.client.admin.command('ping')
        print("✅ Database connection successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 InsurEdge AI - ML Models Test")
    print("=" * 50)
    
    # Test ML models
    ml_success = test_ml_models()
    
    # Test database
    db_success = test_database_connection()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    print(f"ML Models: {'✅ PASS' if ml_success else '❌ FAIL'}")
    print(f"Database: {'✅ PASS' if db_success else '❌ FAIL'}")
    
    if ml_success and db_success:
        print("\n🎉 All systems are ready!")
        print("🚀 You can now run: python start.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        if not ml_success:
            print("💡 Try: pip install -r requirements.txt")
        if not db_success:
            print("💡 Make sure MongoDB is running")

if __name__ == "__main__":
    main() 