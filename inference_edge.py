"""
Edge-Optimized Inference Module
================================
Lightweight version for mobile, tablets, edge servers, and offline deployment.
Optimized for 4GB RAM, slower CPUs, and battery-powered devices.
"""

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional
import warnings
import os

warnings.filterwarnings('ignore')


class GemmaInferenceEdge:
    """
    Edge-optimized version of Gemma inference.
    
    Key optimizations:
    - Reduced context length (512 → 256)
    - Faster generation (150 → 100 tokens)
    - Greedy decoding (no sampling)
    - Lower memory footprint
    - Optional quantization
    """
    
    def __init__(self, model_name: str = "google/gemma-2b-it"):
        """Initialize edge-optimized Gemma."""
        self.model_name = model_name
        self.device = "cpu"
        self.model = None
        self.tokenizer = None
        self.max_length = 256  # Reduced from 512
        self.edge_mode = True
        
    def load_model(self, use_half_precision=False):
        """
        Load model with edge optimizations.
        
        Args:
            use_half_precision: Use float16 (smaller, faster, slight accuracy loss)
        """
        print(f"Loading {self.model_name} in EDGE mode...")
        print("Edge optimizations: Reduced memory, faster inference")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with edge settings
            dtype = torch.float16 if use_half_precision else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Optimize for edge
            if hasattr(torch, 'set_num_threads'):
                torch.set_num_threads(2)  # Limit CPU threads for battery
            
            print(f"✓ Model loaded in edge mode (dtype: {dtype})")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_triage_prompt(self, patient_data: Dict) -> str:
        """
        Create COMPACT medical triage prompt.
        Shorter than standard version for faster processing.
        """
        age = patient_data.get('age', 'unknown')
        gender = patient_data.get('gender', 'unknown')
        symptoms = patient_data.get('symptoms', [])
        notes = patient_data.get('clinical_notes', '')
        
        symptoms_text = ', '.join(symptoms[:5]) if symptoms else 'none'  # Limit to 5
        
        # Compact prompt (shorter = faster)
        prompt = f"""Medical triage for {age}yo {gender}
Symptoms: {symptoms_text}
Notes: {notes[:200]}

Classify as Emergency (life-threat), Urgent (serious), or Low (stable).
Respond JSON only:
{{"risk_level": "Emergency|Urgent|Low", "detected_symptoms": [...], "reasoning": "brief justification"}}

Response:"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Fast generation optimized for edge devices.
        
        Changes from standard:
        - Fewer tokens (100 vs 150)
        - Greedy decoding (no sampling)
        - No beam search
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize with reduced length
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length  # Shorter context
        )
        
        # Fast generation settings
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100,      # Reduced from 150
                do_sample=False,         # Greedy = faster
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response
    
    def parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response (same as standard version)."""
        # Remove markdown
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()
        
        # Find JSON
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response
        
        try:
            parsed = json.loads(json_str)
            
            # Validate
            required_fields = ['risk_level', 'detected_symptoms', 'reasoning']
            if all(field in parsed for field in required_fields):
                # Normalize risk level
                risk = parsed['risk_level'].strip().title()
                if risk not in ['Emergency', 'Urgent', 'Low']:
                    if 'emerg' in risk.lower():
                        risk = 'Emergency'
                    elif 'urg' in risk.lower():
                        risk = 'Urgent'
                    else:
                        risk = 'Low'
                parsed['risk_level'] = risk
                
                return parsed
            else:
                return None
                
        except json.JSONDecodeError:
            return None
    
    def analyze_patient(self, patient_data: Dict) -> Dict:
        """
        Complete edge analysis pipeline.
        Faster and lighter than standard version.
        """
        if self.model is None:
            self.load_model()
        
        # Create compact prompt
        prompt = self.create_triage_prompt(patient_data)
        
        # Fast generation
        raw_response = self.generate_response(prompt)
        
        # Parse
        parsed = self.parse_json_response(raw_response)
        
        # Fallback
        if parsed is None:
            parsed = {
                'risk_level': 'Urgent',
                'detected_symptoms': patient_data.get('symptoms', [])[:3],
                'reasoning': 'Edge mode: Manual review recommended.',
                'parse_error': True
            }
        
        parsed['raw_response'] = raw_response
        parsed['edge_mode'] = True
        
        return parsed
    
    def get_edge_stats(self) -> Dict:
        """Get edge deployment statistics."""
        import psutil
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            'memory_usage_mb': round(memory_mb, 2),
            'cpu_threads': torch.get_num_threads() if hasattr(torch, 'get_num_threads') else 'N/A',
            'model_dtype': str(self.model.dtype) if self.model else 'Not loaded',
            'max_context_length': self.max_length,
            'edge_optimizations': [
                'Reduced context (256 tokens)',
                'Fast generation (100 tokens)',
                'Greedy decoding',
                'CPU thread limit',
                'Low memory mode'
            ]
        }


# Singleton instance
gemma_edge = GemmaInferenceEdge()


if __name__ == "__main__":
    # Test edge inference
    print("="*70)
    print("EDGE INFERENCE TEST")
    print("="*70)
    
    # Load model
    edge = GemmaInferenceEdge()
    edge.load_model(use_half_precision=False)
    
    # Test case
    test_patient = {
        'age': 65,
        'gender': 'Male',
        'symptoms': ['chest pain', 'shortness of breath'],
        'clinical_notes': 'Crushing chest pain for 30 minutes.'
    }
    
    import time
    start = time.time()
    
    print("\nAnalyzing test patient...")
    result = edge.analyze_patient(test_patient)
    
    duration = time.time() - start
    
    print(f"\n✓ Analysis complete in {duration:.2f} seconds")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Detected Symptoms: {result['detected_symptoms']}")
    print(f"Reasoning: {result['reasoning']}")
    
    # Show edge stats
    print("\n" + "="*70)
    print("EDGE DEPLOYMENT STATS")
    print("="*70)
    stats = edge.get_edge_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")