"""
Clinical Explanations Module
============================
Generates plain-language medical explanations for triage decisions.
Suitable for healthcare providers, judges, and demonstration purposes.
"""

from typing import Dict, List


class ClinicalExplainer:
    """
    Provides evidence-based medical explanations for triage decisions.
    Emphasizes safety, clarity, and clinical relevance.
    """
    
    def __init__(self):
        """Initialize explanation templates and medical knowledge base."""
        # Risk level descriptions
        self.risk_descriptions = {
            'Emergency': {
                'severity': 'LIFE-THREATENING',
                'action': 'Requires immediate medical intervention',
                'timeframe': 'Minutes to hours',
                'recommendation': 'Call 911 or go to the Emergency Department immediately'
            },
            'Urgent': {
                'severity': 'SERIOUS',
                'action': 'Requires prompt medical evaluation',
                'timeframe': 'Within 2-6 hours',
                'recommendation': 'Seek urgent care or emergency department evaluation today'
            },
            'Low': {
                'severity': 'NON-URGENT',
                'action': 'Can be managed with routine care',
                'timeframe': 'Within 24-48 hours',
                'recommendation': 'Schedule appointment with primary care provider or visit walk-in clinic'
            }
        }
        
        # Symptom-specific explanations
        self.symptom_explanations = {
            'chest pain': 'Chest pain can indicate cardiac emergencies (heart attack, angina), pulmonary issues (pulmonary embolism), or other serious conditions requiring immediate evaluation.',
            'shortness of breath': 'Difficulty breathing may indicate respiratory failure, cardiac issues, or pulmonary conditions requiring urgent assessment.',
            'altered consciousness': 'Changes in mental status suggest potential neurological emergency, metabolic crisis, or severe infection.',
            'severe bleeding': 'Uncontrolled bleeding is a life-threatening emergency requiring immediate hemorrhage control and resuscitation.',
            'stroke symptoms': 'Signs of stroke (facial drooping, arm weakness, speech difficulty) require immediate intervention as "time is brain."',
            'seizure': 'Active seizures or recent seizure activity require emergency evaluation to prevent complications and identify underlying causes.',
            'high fever': 'Elevated body temperature, especially with other symptoms, may indicate serious infection requiring medical evaluation.',
            'severe pain': 'High-intensity pain suggests significant tissue injury, inflammation, or organ dysfunction requiring assessment.',
            'low blood pressure': 'Hypotension may indicate shock, severe dehydration, cardiac issues, or internal bleeding.',
            'vomiting blood': 'Hematemesis indicates upper gastrointestinal bleeding, a potentially life-threatening emergency.',
            'severe abdominal pain': 'Acute abdominal pain may represent surgical emergencies such as appendicitis, obstruction, or perforation.',
        }
        
    def generate_explanation(self, 
                            risk_level: str,
                            symptoms: List[Dict],
                            rule_based_score: int,
                            gemma_reasoning: str = None,
                            age: int = None) -> str:
        """
        Generate comprehensive clinical explanation.
        
        Args:
            risk_level: Emergency/Urgent/Low
            symptoms: List of detected symptom dictionaries
            rule_based_score: Numerical risk score
            gemma_reasoning: AI-generated reasoning
            age: Patient age
            
        Returns:
            Human-readable explanation
        """
        # Start with risk level summary
        risk_info = self.risk_descriptions.get(risk_level, self.risk_descriptions['Urgent'])
        
        explanation_parts = []
        
        # Header
        explanation_parts.append(f"=== TRIAGE ASSESSMENT: {risk_info['severity']} ===\n")
        
        # Primary determination
        explanation_parts.append(f"**Risk Classification:** {risk_level}")
        explanation_parts.append(f"**Risk Score:** {rule_based_score}/400")
        explanation_parts.append(f"**Recommended Action:** {risk_info['recommendation']}\n")
        
        # Age considerations
        if age is not None:
            age_note = self._get_age_considerations(age, risk_level)
            if age_note:
                explanation_parts.append(f"**Age Consideration:** {age_note}\n")
        
        # Symptom analysis
        if symptoms:
            explanation_parts.append("**Clinical Findings:**")
            
            # Group by severity
            critical = [s for s in symptoms if s['weight'] >= 90]
            high = [s for s in symptoms if 50 <= s['weight'] < 90]
            moderate = [s for s in symptoms if 20 <= s['weight'] < 50]
            
            if critical:
                explanation_parts.append("\n🚨 CRITICAL Symptoms Detected:")
                for symptom in critical[:3]:
                    explanation_parts.append(
                        f"  • {symptom['symptom'].title()} (Weight: {symptom['weight']})"
                    )
                    # Add clinical explanation if available
                    explanation = self.symptom_explanations.get(symptom['symptom'])
                    if explanation:
                        explanation_parts.append(f"    → {explanation}")
            
            if high:
                explanation_parts.append("\n⚠️  HIGH RISK Symptoms:")
                for symptom in high[:3]:
                    explanation_parts.append(
                        f"  • {symptom['symptom'].title()} (Weight: {symptom['weight']})"
                    )
            
            if moderate and not critical and not high:
                explanation_parts.append("\nModerate Symptoms:")
                for symptom in moderate[:3]:
                    explanation_parts.append(
                        f"  • {symptom['symptom'].title()} (Weight: {symptom['weight']})"
                    )
        
        # AI reasoning
        if gemma_reasoning:
            explanation_parts.append(f"\n**AI Analysis:**")
            explanation_parts.append(f"{gemma_reasoning}")
        
        # Medical decision rationale
        explanation_parts.append(f"\n**Clinical Rationale:**")
        rationale = self._generate_rationale(risk_level, symptoms, rule_based_score)
        explanation_parts.append(rationale)
        
        # Important disclaimers
        explanation_parts.append("\n" + "="*50)
        explanation_parts.append(self._get_disclaimer(risk_level))
        
        return "\n".join(explanation_parts)
    
    def _get_age_considerations(self, age: int, risk_level: str) -> str:
        """Generate age-specific clinical considerations."""
        if age < 2:
            return "Infants have limited physiological reserve and can decompensate rapidly. Increased clinical vigilance warranted."
        elif age < 18:
            return "Pediatric patients may have age-specific vital sign ranges and different presentation patterns."
        elif age > 65:
            return "Elderly patients may have atypical presentations and are at higher risk for complications. Consider underlying comorbidities."
        elif age > 80:
            return "Advanced age increases risk for adverse outcomes. Even seemingly minor symptoms may indicate serious underlying conditions."
        return None
    
    def _generate_rationale(self, risk_level: str, symptoms: List[Dict], score: int) -> str:
        """Generate decision-making rationale."""
        if risk_level == 'Emergency':
            if score >= 100:
                return ("Multiple high-severity symptoms present. Emergency triage indicated based on "
                       "potential for rapid clinical deterioration and need for immediate intervention. "
                       "This assessment prioritizes patient safety through conservative risk stratification.")
            else:
                return ("Critical symptoms detected that require emergency evaluation regardless of overall "
                       "score. Clinical guidelines prioritize early intervention for these presentations.")
        
        elif risk_level == 'Urgent':
            return ("Significant symptoms present requiring timely medical evaluation. While not immediately "
                   "life-threatening, delayed treatment could lead to complications or clinical deterioration. "
                   "Prompt assessment recommended within several hours.")
        
        else:  # Low
            return ("Symptoms suggest non-urgent condition appropriate for routine medical care. "
                   "No immediate life-threatening concerns identified. However, if symptoms worsen or "
                   "new concerning symptoms develop, re-evaluation is indicated.")
    
    def _get_disclaimer(self, risk_level: str) -> str:
        """Generate appropriate medical disclaimer."""
        base_disclaimer = (
            "⚕️  IMPORTANT MEDICAL DISCLAIMER:\n"
            "This AI system provides decision support only and is NOT a substitute for professional "
            "medical judgment. All triage decisions should be reviewed by qualified healthcare providers. "
        )
        
        if risk_level == 'Emergency':
            specific = (
                "For life-threatening emergencies: CALL 911 IMMEDIATELY. "
                "Do not delay emergency care while consulting this system."
            )
        elif risk_level == 'Urgent':
            specific = (
                "Seek in-person medical evaluation promptly. Do not rely solely on AI assessment "
                "for urgent medical conditions."
            )
        else:
            specific = (
                "If symptoms worsen, change significantly, or you have concerns, "
                "seek medical attention regardless of this assessment."
            )
        
        return base_disclaimer + specific
    
    def generate_short_summary(self, risk_level: str, top_symptoms: List[str]) -> str:
        """Generate brief summary for UI display."""
        risk_info = self.risk_descriptions[risk_level]
        
        if top_symptoms:
            symptom_text = ", ".join(top_symptoms[:3])
            return (f"{risk_level} triage indicated due to: {symptom_text}. "
                   f"{risk_info['recommendation']}")
        else:
            return f"{risk_level} risk classification. {risk_info['recommendation']}"


# Singleton instance
clinical_explainer = ClinicalExplainer()


if __name__ == "__main__":
    # Test explanation generation
    print("=== Clinical Explainer Test ===\n")
    
    explainer = ClinicalExplainer()
    
    # Test case: Emergency scenario
    test_symptoms = [
        {'symptom': 'chest pain', 'weight': 100, 'source': 'text'},
        {'symptom': 'shortness of breath', 'weight': 70, 'source': 'selected'},
    ]
    
    explanation = explainer.generate_explanation(
        risk_level='Emergency',
        symptoms=test_symptoms,
        rule_based_score=170,
        gemma_reasoning='Patient presents with classic cardiac symptoms requiring immediate evaluation.',
        age=65
    )
    
    print(explanation)
    print("\n" + "="*60 + "\n")
    
    # Test case: Low risk
    test_symptoms_low = [
        {'symptom': 'mild headache', 'weight': 10, 'source': 'text'},
        {'symptom': 'runny nose', 'weight': 5, 'source': 'text'},
    ]
    
    explanation_low = explainer.generate_explanation(
        risk_level='Low',
        symptoms=test_symptoms_low,
        rule_based_score=15,
        gemma_reasoning='Symptoms consistent with common upper respiratory infection.',
        age=30
    )
    
    print(explanation_low)