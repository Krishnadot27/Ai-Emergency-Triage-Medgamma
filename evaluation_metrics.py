"""
Evaluation Metrics Module
=========================
Implements comprehensive evaluation metrics for triage system validation.
Includes ESI-compliant accuracy tracking, confidence scoring, and performance analytics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json


class TriageEvaluator:
    """
    Evaluates triage system performance using medical-grade metrics.
    Tracks accuracy, sensitivity, specificity, and clinical safety measures.
    """
    
    def __init__(self):
        """Initialize the evaluator with metric tracking."""
        self.predictions = []
        self.ground_truth = []
        self.confidence_scores = []
        self.timestamps = []
        
        # ESI level mapping for severity scoring
        self.esi_severity = {
            'Emergency': 1,   # ESI Level 1 - Life-threatening
            'Urgent': 2,      # ESI Level 2-3 - High risk
            'Low': 4          # ESI Level 4-5 - Non-urgent
        }
        
    def calculate_confidence_score(self, 
                                   rule_based_result: Dict,
                                   gemma_result: Dict,
                                   patient_data: Dict) -> Dict:
        """
        Calculate comprehensive confidence score for triage decision.
        
        Factors considered:
        - Agreement between rule-based and AI methods
        - Strength of symptom evidence
        - Data completeness
        - Model uncertainty
        
        Args:
            rule_based_result: Output from symptom extractor
            gemma_result: Output from Gemma model
            patient_data: Input patient information
            
        Returns:
            Dictionary with confidence metrics
        """
        confidence_factors = {}
        
        # Factor 1: Method Agreement (0-40 points)
        rule_level = rule_based_result.get('risk_level', 'Urgent')
        gemma_level = gemma_result.get('risk_level', 'Urgent')
        
        if rule_level == gemma_level:
            agreement_score = 40  # Perfect agreement
        elif abs(self.esi_severity[rule_level] - self.esi_severity[gemma_level]) == 1:
            agreement_score = 25  # Adjacent levels
        else:
            agreement_score = 10  # Significant disagreement
        
        confidence_factors['method_agreement'] = agreement_score
        
        # Factor 2: Symptom Strength (0-30 points)
        risk_score = rule_based_result.get('risk_score', 0)
        
        # Normalize risk score to confidence
        if risk_score >= 100:
            symptom_strength = 30  # Very strong evidence
        elif risk_score >= 60:
            symptom_strength = 25  # Strong evidence
        elif risk_score >= 30:
            symptom_strength = 20  # Moderate evidence
        elif risk_score >= 15:
            symptom_strength = 15  # Weak evidence
        else:
            symptom_strength = 10  # Minimal evidence
        
        confidence_factors['symptom_strength'] = symptom_strength
        
        # Factor 3: Data Completeness (0-20 points)
        has_age = patient_data.get('age') is not None
        has_symptoms = len(patient_data.get('symptoms', [])) > 0
        has_notes = len(patient_data.get('clinical_notes', '').strip()) > 10
        
        completeness_score = (
            (10 if has_age else 0) +
            (5 if has_symptoms else 0) +
            (5 if has_notes else 0)
        )
        
        confidence_factors['data_completeness'] = completeness_score
        
        # Factor 4: Model Certainty (0-10 points)
        # Check if Gemma parsing was successful
        parse_success = not gemma_result.get('parse_error', False)
        model_certainty = 10 if parse_success else 5
        
        confidence_factors['model_certainty'] = model_certainty
        
        # Calculate total confidence (0-100)
        total_confidence = sum(confidence_factors.values())
        
        # Determine confidence level
        if total_confidence >= 80:
            confidence_level = 'High'
        elif total_confidence >= 60:
            confidence_level = 'Moderate'
        elif total_confidence >= 40:
            confidence_level = 'Low'
        else:
            confidence_level = 'Very Low'
        
        return {
            'confidence_score': total_confidence,
            'confidence_level': confidence_level,
            'confidence_factors': confidence_factors,
            'reliability_note': self._get_reliability_note(confidence_level)
        }
    
    def _get_reliability_note(self, confidence_level: str) -> str:
        """Generate reliability interpretation for confidence level."""
        notes = {
            'High': 'Strong agreement between methods with clear symptom evidence. Decision is highly reliable.',
            'Moderate': 'Reasonable confidence in classification. Consider additional clinical context.',
            'Low': 'Limited agreement or weak evidence. Manual review recommended.',
            'Very Low': 'Insufficient data or conflicting indicators. Expert evaluation required.'
        }
        return notes.get(confidence_level, 'Unknown confidence level.')
    
    def calculate_esi_metrics(self, 
                             predicted_level: str,
                             actual_level: str = None) -> Dict:
        """
        Calculate ESI-compliant triage metrics.
        
        ESI (Emergency Severity Index) Levels:
        1 - Requires immediate life-saving intervention
        2 - High risk, confused/lethargic/severe pain, or danger zone vitals
        3 - Stable with many resources needed
        4 - Stable with one resource needed
        5 - No resources needed (fast track)
        
        Our mapping:
        Emergency → ESI 1
        Urgent → ESI 2-3
        Low → ESI 4-5
        
        Args:
            predicted_level: System's predicted risk level
            actual_level: True risk level (if known, for validation)
            
        Returns:
            Dictionary with ESI metrics
        """
        esi_level = self.esi_severity.get(predicted_level, 3)
        
        metrics = {
            'predicted_risk_level': predicted_level,
            'esi_level': esi_level,
            'acuity_category': self._get_acuity_category(esi_level),
            'recommended_timeframe': self._get_timeframe(esi_level),
            'resource_intensity': self._get_resource_intensity(esi_level)
        }
        
        # If ground truth provided, calculate accuracy
        if actual_level:
            metrics['actual_risk_level'] = actual_level
            metrics['correct_classification'] = (predicted_level == actual_level)
            metrics['severity_error'] = abs(
                self.esi_severity[predicted_level] - 
                self.esi_severity[actual_level]
            )
            
            # Under-triage is more dangerous than over-triage
            if self.esi_severity[predicted_level] > self.esi_severity[actual_level]:
                metrics['triage_direction'] = 'under-triage (DANGEROUS)'
            elif self.esi_severity[predicted_level] < self.esi_severity[actual_level]:
                metrics['triage_direction'] = 'over-triage (safe)'
            else:
                metrics['triage_direction'] = 'correct'
        
        return metrics
    
    def _get_acuity_category(self, esi_level: int) -> str:
        """Get clinical acuity category from ESI level."""
        categories = {
            1: 'Critical - Immediate intervention required',
            2: 'High Acuity - Urgent care needed',
            3: 'Moderate Acuity - Prompt evaluation needed',
            4: 'Low Acuity - Routine care appropriate',
            5: 'Minimal Acuity - Fast track eligible'
        }
        return categories.get(esi_level, 'Unknown acuity')
    
    def _get_timeframe(self, esi_level: int) -> str:
        """Get recommended evaluation timeframe."""
        timeframes = {
            1: 'Immediate (0 minutes)',
            2: 'Within 10 minutes',
            3: 'Within 30 minutes',
            4: 'Within 60 minutes',
            5: 'Within 120 minutes'
        }
        return timeframes.get(esi_level, 'Unknown timeframe')
    
    def _get_resource_intensity(self, esi_level: int) -> str:
        """Get expected resource utilization."""
        resources = {
            1: 'Multiple resources + immediate intervention',
            2: 'Multiple resources expected',
            3: 'Multiple resources likely',
            4: 'One resource expected',
            5: 'No resources expected'
        }
        return resources.get(esi_level, 'Unknown resources')
    
    def add_evaluation(self, 
                      predicted: str,
                      actual: str,
                      confidence: float):
        """
        Record a single evaluation for aggregate metrics.
        
        Args:
            predicted: Predicted risk level
            actual: Actual risk level
            confidence: Confidence score (0-100)
        """
        self.predictions.append(predicted)
        self.ground_truth.append(actual)
        self.confidence_scores.append(confidence)
        self.timestamps.append(datetime.now())
    
    def calculate_aggregate_metrics(self) -> Dict:
        """
        Calculate aggregate performance metrics across all evaluations.
        
        Returns:
            Dictionary with comprehensive performance metrics
        """
        if not self.predictions or not self.ground_truth:
            return {
                'error': 'No evaluations recorded',
                'total_cases': 0
            }
        
        # Basic accuracy
        correct = sum(p == g for p, g in zip(self.predictions, self.ground_truth))
        total = len(self.predictions)
        accuracy = (correct / total) * 100
        
        # Safety metrics (under-triage vs over-triage)
        under_triage = 0
        over_triage = 0
        
        for pred, actual in zip(self.predictions, self.ground_truth):
            pred_severity = self.esi_severity[pred]
            actual_severity = self.esi_severity[actual]
            
            if pred_severity > actual_severity:
                under_triage += 1  # More dangerous
            elif pred_severity < actual_severity:
                over_triage += 1   # Safer, conservative
        
        # Confidence statistics
        avg_confidence = np.mean(self.confidence_scores)
        std_confidence = np.std(self.confidence_scores)
        
        # Per-class metrics
        class_metrics = {}
        for risk_level in ['Emergency', 'Urgent', 'Low']:
            tp = sum(1 for p, g in zip(self.predictions, self.ground_truth) 
                    if p == risk_level and g == risk_level)
            fp = sum(1 for p, g in zip(self.predictions, self.ground_truth) 
                    if p == risk_level and g != risk_level)
            fn = sum(1 for p, g in zip(self.predictions, self.ground_truth) 
                    if p != risk_level and g == risk_level)
            tn = sum(1 for p, g in zip(self.predictions, self.ground_truth) 
                    if p != risk_level and g != risk_level)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[risk_level] = {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1 * 100,
                'support': tp + fn
            }
        
        return {
            'total_cases': total,
            'overall_accuracy': accuracy,
            'correct_classifications': correct,
            'incorrect_classifications': total - correct,
            'under_triage_rate': (under_triage / total) * 100,
            'over_triage_rate': (over_triage / total) * 100,
            'average_confidence': avg_confidence,
            'confidence_std': std_confidence,
            'per_class_metrics': class_metrics,
            'safety_score': 100 - (under_triage / total) * 100  # Higher is safer
        }
    
    def generate_evaluation_report(self, 
                                   predicted_level: str,
                                   confidence_metrics: Dict,
                                   esi_metrics: Dict) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            predicted_level: Predicted risk level
            confidence_metrics: Confidence score details
            esi_metrics: ESI-based metrics
            
        Returns:
            Formatted evaluation report
        """
        report_lines = [
            "=== EVALUATION METRICS ===",
            f"\n📊 CLASSIFICATION: {predicted_level}",
            f"ESI Level: {esi_metrics['esi_level']} - {esi_metrics['acuity_category']}",
            f"\n🎯 CONFIDENCE SCORE: {confidence_metrics['confidence_score']}/100",
            f"Confidence Level: {confidence_metrics['confidence_level']}",
            f"Reliability: {confidence_metrics['reliability_note']}",
            f"\n📋 CONFIDENCE BREAKDOWN:",
        ]
        
        for factor, score in confidence_metrics['confidence_factors'].items():
            factor_name = factor.replace('_', ' ').title()
            report_lines.append(f"  • {factor_name}: {score} points")
        
        report_lines.extend([
            f"\n⏱️  CLINICAL TIMEFRAME:",
            f"  • Recommended evaluation: {esi_metrics['recommended_timeframe']}",
            f"  • Expected resources: {esi_metrics['resource_intensity']}",
        ])
        
        return "\n".join(report_lines)


# Singleton instance
triage_evaluator = TriageEvaluator()


if __name__ == "__main__":
    # Test the evaluation metrics
    print("=== Evaluation Metrics Test ===\n")
    
    evaluator = TriageEvaluator()
    
    # Test confidence calculation
    test_rule_result = {
        'risk_score': 170,
        'risk_level': 'Emergency',
        'symptom_count': 2
    }
    
    test_gemma_result = {
        'risk_level': 'Emergency',
        'detected_symptoms': ['chest pain', 'shortness of breath'],
        'reasoning': 'Life-threatening cardiac symptoms'
    }
    
    test_patient = {
        'age': 65,
        'symptoms': ['chest pain'],
        'clinical_notes': 'Crushing chest pain radiating to left arm'
    }
    
    confidence = evaluator.calculate_confidence_score(
        test_rule_result,
        test_gemma_result,
        test_patient
    )
    
    print("Confidence Metrics:")
    print(f"Score: {confidence['confidence_score']}/100")
    print(f"Level: {confidence['confidence_level']}")
    print(f"Factors: {confidence['confidence_factors']}")
    print(f"\n{confidence['reliability_note']}")
    
    # Test ESI metrics
    print("\n" + "="*60 + "\n")
    esi_metrics = evaluator.calculate_esi_metrics('Emergency')
    
    print("ESI Metrics:")
    print(f"ESI Level: {esi_metrics['esi_level']}")
    print(f"Acuity: {esi_metrics['acuity_category']}")
    print(f"Timeframe: {esi_metrics['recommended_timeframe']}")
    print(f"Resources: {esi_metrics['resource_intensity']}")
    
    # Test evaluation report
    print("\n" + "="*60 + "\n")
    report = evaluator.generate_evaluation_report(
        'Emergency',
        confidence,
        esi_metrics
    )
    print(report)