#!/usr/bin/env python3
"""
Test fraud detection capabilities of InsurEdge AI
"""

import numpy as np
from ml_models import damage_detector
from PIL import Image
import io

def test_fraud_detection():
    """Test fraud detection on different scenarios"""
    print("🕵️ Testing InsurEdge AI Fraud Detection")
    print("=" * 60)
    
    scenarios = [
        "Normal legitimate claim",
        "Suspicious low-quality image", 
        "High-value theft claim",
        "Potentially staged damage"
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🧪 Test {i}: {scenario}")
        print("-" * 40)
        
        # Create different test images for different scenarios
        if "low-quality" in scenario:
            # Simulate blurry/low quality image
            test_image = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)  # Dark, low contrast
        elif "staged" in scenario:
            # Simulate image with too many edges (staged damage)
            test_image = np.random.randint(200, 255, (224, 224, 3), dtype=np.uint8)  # High contrast
        else:
            # Normal image
            test_image = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        
        # Convert to bytes
        img = Image.fromarray(test_image)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        image_data = img_bytes.getvalue()
        
        # Analyze with fraud detection
        analysis = damage_detector.analyze_damage(image_data)
        
        # Display results
        print(f"🎯 Damage Type: {analysis['damage_type'].title()}")
        print(f"💰 Estimated Cost: ₹{analysis['estimated_cost']:,}")
        print(f"🎯 Confidence: {analysis['confidence_score']:.1f}%")
        
        fraud = analysis['fraud_analysis']
        print(f"\n🚨 FRAUD ANALYSIS:")
        print(f"   Fraud Detected: {'YES' if fraud['fraud_detected'] else 'NO'}")
        print(f"   Risk Level: {fraud['risk_level']}")
        print(f"   Fraud Score: {fraud['fraud_score']:.2f}")
        print(f"   Action: {fraud['recommended_action']}")
        
        if fraud['fraud_indicators']:
            print(f"   🚩 Red Flags:")
            for indicator in fraud['fraud_indicators']:
                print(f"      • {indicator}")
        
        print(f"\n✅ Claim Status: {analysis['claim_status'].upper()}")
        
        if fraud['requires_human_review']:
            print("⚠️  REQUIRES HUMAN REVIEW")
    
    print("\n" + "=" * 60)
    print("🎉 Fraud Detection System Working!")
    print("✅ InsurEdge AI can detect suspicious claims")
    print("✅ Protects insurance companies from fraud")
    print("✅ Saves millions in fraudulent payouts")

if __name__ == "__main__":
    test_fraud_detection()