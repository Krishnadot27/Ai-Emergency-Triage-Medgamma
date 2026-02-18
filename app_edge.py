"""
Edge Deployment App
===================
Mobile and low-resource optimized version.
Designed for: tablets, phones, edge servers, ambulances, rural clinics.
"""

import gradio as gr
from inference_edge import gemma_edge
from symptom_extractor import symptom_extractor
from safety_override import safety_override

# Edge triage system
class EdgeTriageSystem:
    """Lightweight triage system for edge deployment."""
    
    def __init__(self):
        self.gemma = gemma_edge
        self.symptom_detector = symptom_extractor
        self.safety = safety_override
        self.model_loaded = False
        
    def analyze_patient_edge(self, age: int, symptoms_text: str) -> str:
        """
        Simplified edge analysis.
        
        Args:
            age: Patient age
            symptoms_text: Comma-separated symptoms
            
        Returns:
            Formatted result string
        """
        # Parse symptoms
        symptoms = [s.strip() for s in symptoms_text.split(',') if s.strip()]
        
        # Rule-based analysis
        rule_result = self.symptom_detector.extract_symptoms(
            text=symptoms_text,
            age=age,
            selected_symptoms=symptoms
        )
        
        # Prepare patient data
        patient_data = {
            'age': age,
            'symptoms': symptoms,
            'clinical_notes': symptoms_text
        }
        
        # Edge AI analysis
        if not self.model_loaded:
            self.gemma.load_model()
            self.model_loaded = True
        
        ai_result = self.gemma.analyze_patient(patient_data)
        
        # Safety override
        final_risk, final_conf, safety_info = self.safety.apply_safety_override(
            symptoms_text,
            age,
            rule_result['risk_level'],
            rule_result['risk_score']
        )
        
        # Format compact output for mobile
        output = f"""
🏥 TRIAGE RESULT
{'='*40}

🚨 CLASSIFICATION: {final_risk}
🎯 CONFIDENCE: {rule_result['risk_score']}/400

{'🛡️ SAFETY OVERRIDE ACTIVE' if safety_info['override_active'] else ''}

DETECTED SYMPTOMS:
{', '.join([s['symptom'] for s in rule_result['contributing_symptoms'][:3]])}

RECOMMENDATION:
{self._get_action(final_risk)}

{'='*40}
⚕️ AI-Powered Emergency Triage
📱 Edge Mode: Offline Capable
        """
        
        return output.strip()
    
    def _get_action(self, risk_level: str) -> str:
        """Get action recommendation."""
        actions = {
            'Emergency': '🚨 CALL 911 IMMEDIATELY',
            'Urgent': '⚠️ Seek ER within 1 hour',
            'Low': '✅ Schedule doctor visit'
        }
        return actions.get(risk_level, 'Seek medical attention')


# Create edge system
edge_system = EdgeTriageSystem()


def create_edge_ui():
    """Create mobile-optimized interface."""
    
    with gr.Blocks(
        title="Emergency Triage - Edge",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 600px !important}
        .output-text {font-size: 14px !important}
        """
    ) as demo:
        
        gr.Markdown("""
        # 🏥 Emergency Triage (Edge)
        **Mobile-Optimized | Offline Capable | Low Resource**
        """)
        
        with gr.Row():
            age_input = gr.Number(
                label="Age",
                value=40,
                minimum=0,
                maximum=120
            )
        
        symptoms_input = gr.Textbox(
            label="Symptoms (comma-separated)",
            placeholder="chest pain, shortness of breath, nausea",
            lines=3
        )
        
        analyze_btn = gr.Button(
            "🔍 Analyze",
            variant="primary",
            size="lg"
        )
        
        result_output = gr.Textbox(
            label="Triage Result",
            lines=12,
            max_lines=15
        )
        
        # Quick examples
        with gr.Row():
            ex1 = gr.Button("Example: Emergency", size="sm")
            ex2 = gr.Button("Example: Urgent", size="sm")
            ex3 = gr.Button("Example: Low", size="sm")
        
        # Info section
        gr.Markdown("""
        ---
        ### 📱 Edge Deployment Features
        - ✅ **Offline Capable** - Works without internet
        - ✅ **Low Memory** - Runs on 4GB RAM
        - ✅ **Fast** - Optimized for mobile CPUs
        - ✅ **Battery Friendly** - Reduced computation
        - ✅ **Mobile Ready** - Touch-optimized interface
        
        **Use Cases:** Ambulances | Rural Clinics | Disaster Response | Mobile Units
        """)
        
        # Connect buttons
        analyze_btn.click(
            fn=edge_system.analyze_patient_edge,
            inputs=[age_input, symptoms_input],
            outputs=result_output
        )
        
        # Example loaders
        ex1.click(
            lambda: (65, "crushing chest pain, shortness of breath, nausea"),
            outputs=[age_input, symptoms_input]
        )
        
        ex2.click(
            lambda: (8, "high fever, severe abdominal pain, vomiting"),
            outputs=[age_input, symptoms_input]
        )
        
        ex3.click(
            lambda: (30, "mild headache, runny nose, sore throat"),
            outputs=[age_input, symptoms_input]
        )
    
    return demo


if __name__ == "__main__":
    print("="*70)
    print("EDGE DEPLOYMENT APP - STARTING")
    print("="*70)
    print("\n📱 Mobile-optimized interface")
    print("🔌 Offline capable")
    print("⚡ Low resource usage")
    print("\nStarting on port 7861...")
    print("="*70 + "\n")
    
    demo = create_edge_ui()
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,  # Different port from main app
        share=False,
        show_error=True,
        inbrowser=True
    )