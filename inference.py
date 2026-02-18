"""
Gemma Inference Module
=====================
Handles Google Gemma 2B model loading and structured medical reasoning.
Optimized for CPU-only inference with minimal memory footprint.
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class GemmaInference:
    """
    Manages Gemma 2B-IT model for medical triage reasoning.
    Enforces structured JSON output for reliable parsing.
    """
    
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        """
        Initialize Gemma model on CPU.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.max_length = 512  # Limit for CPU efficiency
        
    def load_model(self):
        """Load Gemma model and tokenizer with CPU optimizations."""
        if self.model is not None:
            print("Model already loaded.")
            return
        
        print(f"Loading {self.model_name} on CPU...")
        print("This may take 1-2 minutes on first run...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with CPU optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"✓ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_triage_prompt(self, patient_data: Dict) -> str:
        """
        Create medically-optimized structured prompt for emergency triage.
        Enhanced with ESI guidelines and medical safety protocols.
        
        Args:
            patient_data: Dictionary with age, gender, symptoms, clinical_notes
            
        Returns:
            Formatted prompt string with safety-focused instructions
        """
        age = patient_data.get('age', 'unknown')
        gender = patient_data.get('gender', 'unknown')
        symptoms = patient_data.get('symptoms', [])
        notes = patient_data.get('clinical_notes', '')
        
        symptoms_text = ', '.join(symptoms) if symptoms else 'none reported'
        
        # Enhanced medical safety prompt
        prompt = f"""You are an AI medical triage assistant using Emergency Severity Index (ESI) guidelines. Your role is to identify life-threatening conditions and recommend appropriate care urgency.

PATIENT PRESENTATION:
Age: {age} years old
Gender: {gender}
Chief Complaints: {symptoms_text}
Clinical Notes: {notes}

TRIAGE CLASSIFICATION (ESI-Based):

Emergency (ESI Level 1):
- Requires IMMEDIATE life-saving intervention
- Examples: Cardiac arrest, severe respiratory distress, uncontrolled bleeding, stroke symptoms (facial droop/arm weakness/speech), altered mental status, seizures, severe chest pain, unconsciousness, major trauma
- Time: 0 minutes (call 911)

Urgent (ESI Level 2-3):
- High risk situation OR multiple resources needed
- Examples: High fever with severe symptoms, severe pain (8-10/10), moderate respiratory distress, signs of infection with vital sign abnormalities, severe dehydration, persistent vomiting, pregnancy complications
- Time: Within 30-60 minutes

Low (ESI Level 4-5):
- Stable vital signs, one or no resources needed
- Examples: Minor injuries, mild symptoms, common cold, minor rash, mild pain (1-4/10), chronic stable conditions
- Time: Can wait 1-2 hours or schedule appointment

MEDICAL SAFETY PRINCIPLES:
1. When in doubt, classify HIGHER risk (conservative approach)
2. Age extremes (infants, elderly) increase risk
3. Multiple symptoms suggest higher acuity
4. Vague or concerning combinations warrant urgent evaluation
5. NEVER downplay chest pain, breathing difficulty, or neurological symptoms

CRITICAL RED FLAGS (Always Emergency):
- Chest pain/pressure (cardiac concern)
- Difficulty breathing/shortness of breath
- Altered consciousness/confusion
- Severe bleeding or trauma
- Signs of stroke
- Severe allergic reaction
- Suicidal ideation/overdose

OUTPUT REQUIREMENTS:
Respond with ONLY valid JSON. No markdown, no code blocks, no extra text.

REQUIRED JSON FORMAT:
{{
  "risk_level": "Emergency" or "Urgent" or "Low",
  "detected_symptoms": ["symptom1", "symptom2", "symptom3"],
  "reasoning": "Brief clinical justification based on ESI criteria (max 100 words)",
  "confidence": "high" or "moderate" or "low"
}}

Analyze the patient and respond with ONLY the JSON object:"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate model response with temperature control.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Raw model output
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )
        
        # Generate with controlled parameters
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,  # Limit output length
                temperature=0.1,     # Low temperature for consistency
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response after the prompt
        response = response[len(prompt):].strip()
        
        return response
    
    def parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Parse JSON from model response with robust error handling.
        Now includes confidence parsing from Gemma output.
        
        Args:
            response: Raw model output
            
        Returns:
            Parsed dictionary or None if parsing fails
        """
        # Remove markdown code blocks if present
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # Try to find JSON object
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response
        
        try:
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['risk_level', 'detected_symptoms', 'reasoning']
            if all(field in parsed for field in required_fields):
                # Normalize risk level
                risk = parsed['risk_level'].strip().title()
                if risk not in ['Emergency', 'Urgent', 'Low']:
                    # Try to map common variations
                    if 'emerg' in risk.lower():
                        risk = 'Emergency'
                    elif 'urg' in risk.lower():
                        risk = 'Urgent'
                    else:
                        risk = 'Low'
                parsed['risk_level'] = risk
                
                # Parse confidence if present (optional field)
                if 'confidence' in parsed:
                    conf = parsed['confidence'].lower().strip()
                    if conf not in ['high', 'moderate', 'low']:
                        parsed['confidence'] = 'moderate'  # Default
                else:
                    # Infer confidence from reasoning length and clarity
                    reasoning_len = len(parsed.get('reasoning', ''))
                    if reasoning_len > 50:
                        parsed['confidence'] = 'moderate'
                    else:
                        parsed['confidence'] = 'low'
                
                return parsed
            else:
                print(f"Missing required fields in response: {parsed}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {response[:200]}")
            return None
    
    def analyze_patient(self, patient_data: Dict) -> Dict:
        """
        Complete pipeline: create prompt, generate, and parse response.
        
        Args:
            patient_data: Patient information dictionary
            
        Returns:
            Structured analysis result with fallback
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Create prompt
        prompt = self.create_triage_prompt(patient_data)
        
        # Generate response
        raw_response = self.generate_response(prompt)
        
        # Parse JSON
        parsed = self.parse_json_response(raw_response)
        
        # Fallback if parsing fails
        if parsed is None:
            print("⚠ Gemma parsing failed, using fallback response")
            parsed = {
                'risk_level': 'Urgent',
                'detected_symptoms': patient_data.get('symptoms', [])[:3],
                'reasoning': 'Unable to parse model output. Manual review recommended.',
                'parse_error': True
            }
        
        # Add raw response for debugging
        parsed['raw_response'] = raw_response
        
        return parsed


# Singleton instance
gemma_model = GemmaInference()


if __name__ == "__main__":
    # Test the inference pipeline
    print("=== Gemma Inference Test ===\n")
    
    # Initialize model
    gemma = GemmaInference()
    gemma.load_model()
    
    # Test case
    test_patient = {
        'age': 65,
        'gender': 'Male',
        'symptoms': ['chest pain', 'shortness of breath'],
        'clinical_notes': 'Patient reports crushing chest pain radiating to left arm for past 30 minutes. Sweating and nauseous.'
    }
    
    print("Analyzing test patient...")
    result = gemma.analyze_patient(test_patient)
    
    print("\n=== Results ===")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Detected Symptoms: {result['detected_symptoms']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"\nRaw Response:\n{result['raw_response'][:300]}...")