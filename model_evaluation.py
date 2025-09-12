import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from ml_models import damage_detector
import os
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_damage_classifier(self, test_images, test_labels):
        """Evaluate damage classification model"""
        predictions = []
        confidences = []
        
        for image in test_images:
            damage_type, confidence = damage_detector.classify_damage(image)
            predictions.append(damage_type)
            confidences.append(confidence)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        f1 = f1_score(test_labels, predictions, average='weighted')
        
        self.results['classification'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': np.mean(confidences)
        }
        
        return self.results['classification']
    
    def evaluate_severity_prediction(self, test_images, true_severities):
        """Evaluate severity prediction model"""
        predicted_severities = []
        
        for image in test_images:
            severity = damage_detector.predict_severity(image, 'collision')
            predicted_severities.append(severity)
        
        # Calculate regression metrics
        mae = mean_absolute_error(true_severities, predicted_severities)
        mse = mean_squared_error(true_severities, predicted_severities)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_severities, predicted_severities)
        
        self.results['severity'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2
        }
        
        return self.results['severity']
    
    def evaluate_cost_estimation(self, test_images, true_costs):
        """Evaluate cost estimation model"""
        predicted_costs = []
        
        for image in test_images:
            cost = damage_detector.estimate_cost(image, 'collision', 0.5)
            predicted_costs.append(cost)
        
        # Calculate regression metrics
        mae = mean_absolute_error(true_costs, predicted_costs)
        mse = mean_squared_error(true_costs, predicted_costs)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_costs, predicted_costs)
        
        # Calculate percentage error
        percentage_errors = np.abs((np.array(true_costs) - np.array(predicted_costs)) / np.array(true_costs)) * 100
        mean_percentage_error = np.mean(percentage_errors)
        
        self.results['cost_estimation'] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2_score': r2,
            'mean_percentage_error': mean_percentage_error
        }
        
        return self.results['cost_estimation']
    
    def plot_confusion_matrix(self, test_labels, predictions, damage_types):
        """Plot confusion matrix for damage classification"""
        cm = confusion_matrix(test_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=damage_types, yticklabels=damage_types)
        plt.title('Damage Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_severity_comparison(self, true_severities, predicted_severities):
        """Plot severity prediction comparison"""
        plt.figure(figsize=(10, 6))
        plt.scatter(true_severities, predicted_severities, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.xlabel('True Severity')
        plt.ylabel('Predicted Severity')
        plt.title('Severity Prediction: True vs Predicted')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('severity_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cost_comparison(self, true_costs, predicted_costs):
        """Plot cost estimation comparison"""
        plt.figure(figsize=(10, 6))
        plt.scatter(true_costs, predicted_costs, alpha=0.6)
        plt.plot([min(true_costs), max(true_costs)], [min(true_costs), max(true_costs)], 'r--', lw=2)
        plt.xlabel('True Cost (₹)')
        plt.ylabel('Predicted Cost (₹)')
        plt.title('Cost Estimation: True vs Predicted')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cost_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'model_performance': self.results,
            'summary': {
                'classification_accuracy': self.results.get('classification', {}).get('accuracy', 0),
                'severity_r2_score': self.results.get('severity', {}).get('r2_score', 0),
                'cost_estimation_accuracy': 100 - self.results.get('cost_estimation', {}).get('mean_percentage_error', 100)
            }
        }
        
        # Save report
        with open('model_evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("📊 Model Evaluation Report")
        print("=" * 50)
        
        if 'classification' in self.results:
            print(f"🎯 Damage Classification Accuracy: {self.results['classification']['accuracy']:.2%}")
            print(f"🎯 Average Confidence Score: {self.results['classification']['avg_confidence']:.2%}")
        
        if 'severity' in self.results:
            print(f"📈 Severity Prediction R² Score: {self.results['severity']['r2_score']:.3f}")
            print(f"📈 Severity RMSE: {self.results['severity']['rmse']:.3f}")
        
        if 'cost_estimation' in self.results:
            print(f"💰 Cost Estimation Accuracy: {100 - self.results['cost_estimation']['mean_percentage_error']:.1f}%")
            print(f"💰 Average Error: ₹{self.results['cost_estimation']['mae']:.0f}")
        
        print("=" * 50)
        return report

def run_evaluation():
    """Run complete model evaluation with realistic simulation"""
    evaluator = ModelEvaluator()
    
    # Generate more realistic test data
    print("🔄 Generating realistic test data...")
    test_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(100)]
    
    # Create realistic test labels with proper distribution
    damage_distribution = ['collision'] * 40 + ['vandalism'] * 15 + ['hail'] * 15 + ['theft'] * 10 + ['flood'] * 10 + ['fire'] * 5 + ['other'] * 5
    test_labels = np.random.choice(damage_distribution, 100)
    
    # Generate correlated severity and cost data
    true_severities = []
    true_costs = []
    
    for label in test_labels:
        if label == 'collision':
            severity = np.random.beta(2, 3)  # Moderate severity
            cost = np.random.normal(45000, 15000)
        elif label == 'theft':
            severity = np.random.beta(4, 2)  # High severity
            cost = np.random.normal(80000, 20000)
        elif label == 'fire':
            severity = np.random.beta(5, 2)  # Very high severity
            cost = np.random.normal(120000, 30000)
        elif label == 'flood':
            severity = np.random.beta(3, 2)  # High severity
            cost = np.random.normal(70000, 20000)
        elif label == 'vandalism':
            severity = np.random.beta(2, 4)  # Low-moderate severity
            cost = np.random.normal(25000, 10000)
        elif label == 'hail':
            severity = np.random.beta(2, 5)  # Low severity
            cost = np.random.normal(18000, 8000)
        else:  # other
            severity = np.random.beta(2, 3)
            cost = np.random.normal(35000, 15000)
        
        true_severities.append(np.clip(severity, 0, 1))
        true_costs.append(max(5000, cost))
    
    print("🧪 Evaluating damage classification...")
    classification_results = evaluator.evaluate_damage_classifier(test_images, test_labels)
    
    print("🧪 Evaluating severity prediction...")
    severity_results = evaluator.evaluate_severity_prediction(test_images, true_severities)
    
    print("🧪 Evaluating cost estimation...")
    cost_results = evaluator.evaluate_cost_estimation(test_images, true_costs)
    
    print("📊 Generating evaluation report...")
    report = evaluator.generate_report()
    
    return evaluator, report

if __name__ == "__main__":
    evaluator, report = run_evaluation()