"""
AI Emergency Triage Support System
==================================
An explainable AI emergency triage system combining clinical rules 
and Google Gemma LLM to detect life-threatening medical red flags.

Main application with Gradio UI.
"""

import gradio as gr
import json
from typing import Tuple
import sys

# Import custom modules
from symptom_extractor import symptom_extractor
from inference import gemma_model
from clinical_explanations import clinical_explainer
from evaluation_metrics import triage_evaluator
from safety_override import safety_override


class TriageSystem:
    """
    Main orchestrator combining rule-based and AI-based triage.
    """
    
    def __init__(self):
        """Initialize the triage system."""
        self.gemma = gemma_model
        self.symptom_detector = symptom_extractor
        self.explainer = clinical_explainer
        self.evaluator = triage_evaluator
        self.safety = safety_override
        self.model_loaded = False
        
    def initialize_model(self):
        """Load Gemma model (called on first use)."""
        if not self.model_loaded:
            print("Initializing Gemma model...")
            try:
                self.gemma.load_model()
                self.model_loaded = True
                return "✓ Model loaded successfully"
            except Exception as e:
                return f"⚠ Model loading failed: {str(e)}"
        return "Model already loaded"
    
    def analyze_patient(self,
                       age: int,
                       gender: str,
                       selected_symptoms: list,
                       clinical_notes: str) -> Tuple[str, str, str, str, str]:
        """
        Complete triage analysis combining rule-based and AI methods.
        Now includes confidence scoring and ESI-compliant evaluation metrics.
        
        Args:
            age: Patient age
            gender: Patient gender
            selected_symptoms: List of selected symptoms from checkboxes
            clinical_notes: Free-text clinical description
            
        Returns:
            Tuple of (triage_result, detailed_explanation, evaluation_metrics, gemma_output, status_message)
        """
        # Validate inputs
        if age < 0 or age > 120:
            return ("Invalid input", "Please enter a valid age (0-120)", "", "", "❌ Invalid age")
        
        if not selected_symptoms and not clinical_notes.strip():
            return (
                "Insufficient Information",
                "Please provide either selected symptoms or clinical notes.",
                "",
                "",
                "❌ No symptoms provided"
            )
        
        # Initialize model if needed
        if not self.model_loaded:
            status = self.initialize_model()
            if "failed" in status:
                return ("Error", status, "", "", status)
        
        try:
            # === STEP 1: Rule-based symptom extraction ===
            rule_based_result = self.symptom_detector.extract_symptoms(
                text=clinical_notes,
                age=age,
                selected_symptoms=selected_symptoms
            )
            
            # === STEP 2: Gemma AI analysis ===
            patient_data = {
                'age': age,
                'gender': gender,
                'symptoms': selected_symptoms if selected_symptoms else [],
                'clinical_notes': clinical_notes
            }
            
            gemma_result = self.gemma.analyze_patient(patient_data)
            
            # === STEP 4: Calculate confidence score ===
            confidence_metrics = self.evaluator.calculate_confidence_score(
                rule_based_result,
                gemma_result,
                patient_data
            )
            
            # === STEP 5: Combine results (rule-based takes priority for safety) ===
            # Use the more severe risk level between rule-based and AI
            risk_levels_severity = {'Emergency': 3, 'Urgent': 2, 'Low': 1}
            
            rule_severity = risk_levels_severity.get(rule_based_result['risk_level'], 2)
            gemma_severity = risk_levels_severity.get(gemma_result['risk_level'], 2)
            
            # Conservative approach: take the higher risk
            if rule_severity >= gemma_severity:
                preliminary_risk_level = rule_based_result['risk_level']
                primary_method = "rule-based (conservative)"
            else:
                preliminary_risk_level = gemma_result['risk_level']
                primary_method = "AI-detected"
            
            # === STEP 6: SAFETY OVERRIDE LAYER (Critical) ===
            final_risk_level, final_confidence, safety_info = self.safety.apply_safety_override(
                clinical_notes,
                age,
                preliminary_risk_level,
                confidence_metrics['confidence_score']
            )
            
            # Update confidence if overridden
            if safety_info['override_active']:
                confidence_metrics['confidence_score'] = final_confidence
                confidence_metrics['confidence_level'] = 'High' if final_confidence >= 80 else 'Moderate'
                primary_method = "SAFETY OVERRIDE"
            
            # === STEP 7: Calculate ESI metrics ===
            esi_metrics = self.evaluator.calculate_esi_metrics(final_risk_level)
            
            # === STEP 8: Generate comprehensive explanation ===
            explanation = self.explainer.generate_explanation(
                risk_level=final_risk_level,
                symptoms=rule_based_result['contributing_symptoms'],
                rule_based_score=rule_based_result['risk_score'],
                gemma_reasoning=gemma_result.get('reasoning', 'N/A'),
                age=age
            )
            
            # Add safety override message if applicable
            if safety_info.get('safety_message'):
                explanation = safety_info['safety_message'] + "\n\n" + "="*70 + "\n\n" + explanation
            
            # === STEP 9: Format triage result ===
            triage_summary = self._format_triage_summary(
                final_risk_level,
                rule_based_result,
                gemma_result,
                primary_method,
                confidence_metrics,
                safety_info
            )
            
            # === STEP 10: Generate evaluation report ===
            evaluation_report = self.evaluator.generate_evaluation_report(
                final_risk_level,
                confidence_metrics,
                esi_metrics
            )
            
            # Add safety override info to evaluation if applicable
            if safety_info['override_active']:
                evaluation_report += f"\n\n{'='*50}\n🛡️  SAFETY OVERRIDE INFORMATION\n"
                evaluation_report += f"Override Type: {safety_info['override_type']}\n"
                evaluation_report += f"Original Classification: {safety_info['original_classification']}\n"
                evaluation_report += f"Overridden To: {final_risk_level}\n"
                if safety_info.get('triggered_conditions'):
                    evaluation_report += f"\nTriggered Conditions:\n"
                    for cond in safety_info['triggered_conditions']:
                        evaluation_report += f"  • {cond['condition']}: {cond['reason']}\n"
            
            # === STEP 11: Format Gemma output for display ===
            gemma_display = self._format_gemma_output(gemma_result)
            
            status_msg = f"✓ Analysis complete using {primary_method} classification | Confidence: {confidence_metrics['confidence_level']}"
            
            return (triage_summary, explanation, evaluation_report, gemma_display, status_msg)
            
        except Exception as e:
            error_msg = f"Analysis error: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return (
                "System Error",
                f"An error occurred during analysis: {str(e)}",
                "",
                "",
                f"❌ {error_msg}"
            )
    
    def _format_triage_summary(self, risk_level: str, rule_result: dict, 
                               gemma_result: dict, method: str, confidence_metrics: dict,
                               safety_info: dict = None) -> str:
        """Format the triage result summary with confidence and safety override information."""
        # Determine emoji based on risk level
        emoji_map = {
            'Emergency': '🚨',
            'Urgent': '⚠️',
            'Low': '✅'
        }
        emoji = emoji_map.get(risk_level, '❓')
        
        # Confidence indicator
        conf_emoji_map = {
            'High': '🎯',
            'Moderate': '📊',
            'Low': '⚡',
            'Very Low': '❓'
        }
        conf_emoji = conf_emoji_map.get(confidence_metrics['confidence_level'], '📊')
        
        # Safety override indicator
        safety_indicator = ""
        if safety_info and safety_info.get('override_active'):
            safety_indicator = "\n🛡️  SAFETY OVERRIDE ACTIVE - Critical condition detected"
        
        summary_parts = [
            f"{emoji} TRIAGE CLASSIFICATION: {risk_level.upper()}{safety_indicator}",
            f"\n{conf_emoji} CONFIDENCE: {confidence_metrics['confidence_score']}/100 ({confidence_metrics['confidence_level']})",
            f"\nPrimary Method: {method}",
            f"\n{'='*50}",
            f"\nRULE-BASED ANALYSIS:",
            f"  • Risk Score: {rule_result['risk_score']}/400",
            f"  • Risk Level: {rule_result['risk_level']}",
            f"  • Symptoms Detected: {rule_result['symptom_count']}",
        ]
        
        if rule_result['contributing_symptoms']:
            top_symptoms = [s['symptom'] for s in rule_result['contributing_symptoms'][:5]]
            summary_parts.append(f"  • Top Symptoms: {', '.join(top_symptoms)}")
        
        # Add Gemma confidence if available
        gemma_conf = gemma_result.get('confidence', 'moderate')
        
        summary_parts.extend([
            f"\nAI (GEMMA) ANALYSIS:",
            f"  • Risk Level: {gemma_result['risk_level']}",
            f"  • AI Confidence: {gemma_conf.title()}",
            f"  • Detected Symptoms: {len(gemma_result.get('detected_symptoms', []))}",
        ])
        
        # Add confidence breakdown
        summary_parts.append(f"\nCONFIDENCE BREAKDOWN:")
        for factor, score in confidence_metrics['confidence_factors'].items():
            factor_name = factor.replace('_', ' ').title()
            summary_parts.append(f"  • {factor_name}: {score}/40" if 'agreement' in factor 
                               else f"  • {factor_name}: {score}")
        
        # Add safety override info if applicable
        if safety_info:
            if safety_info.get('override_active'):
                summary_parts.append(f"\n🛡️  SAFETY OVERRIDE:")
                summary_parts.append(f"  • Override Type: {safety_info['override_type']}")
                summary_parts.append(f"  • Original Classification: {safety_info['original_classification']}")
                if safety_info.get('triggered_conditions'):
                    summary_parts.append(f"  • Critical Conditions: {len(safety_info['triggered_conditions'])}")
            elif safety_info.get('high_risk_flags'):
                summary_parts.append(f"\n⚠️  HIGH RISK FLAGS: {len(safety_info['high_risk_flags'])} detected")
        
        return "\n".join(summary_parts)
    
    def _format_gemma_output(self, gemma_result: dict) -> str:
        """Format Gemma output for display."""
        display_parts = ["=== GEMMA MODEL OUTPUT ===\n"]
        
        # Structured output
        display_parts.append("Structured Response:")
        display_parts.append(f"  Risk Level: {gemma_result['risk_level']}")
        display_parts.append(f"  Detected Symptoms: {', '.join(gemma_result.get('detected_symptoms', []))}")
        display_parts.append(f"  Reasoning: {gemma_result.get('reasoning', 'N/A')}")
        
        # Raw output for transparency
        if 'raw_response' in gemma_result:
            display_parts.append(f"\nRaw Model Output:")
            display_parts.append(f"{gemma_result['raw_response'][:500]}...")
        
        # Parse error warning
        if gemma_result.get('parse_error'):
            display_parts.insert(1, "\n⚠️ WARNING: JSON parsing failed, using fallback values\n")
        
        return "\n".join(display_parts)


# Initialize the system
triage_system = TriageSystem()


def create_ui():
    """Create and configure the Gradio interface."""
    
    # Common symptoms for quick selection
    common_symptoms = [
        "Chest pain",
        "Shortness of breath",
        "Severe pain",
        "High fever",
        "Altered consciousness",
        "Severe bleeding",
        "Vomiting blood",
        "Severe headache",
        "Abdominal pain",
        "Seizure",
        "Low blood pressure",
        "Confusion",
        "Difficulty breathing",
        "Rapid heart rate",
        "Dizziness",
    ]
    
    with gr.Blocks(title="AI Emergency Triage System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🏥 AI Emergency Triage Support System
            
            **An explainable AI emergency triage system combining clinical rules and Google Gemma LLM 
            to detect life-threatening medical red flags.**
            
            ### How to Use:
            1. Enter patient age and gender
            2. Select relevant symptoms from the checklist
            3. Add detailed clinical notes (optional but recommended)
            4. Click "Analyze Patient" to get triage classification
            
            ### ⚕️ Medical Disclaimer:
            This system is for **educational and research purposes only**. It is NOT a substitute for 
            professional medical judgment. Always consult qualified healthcare providers for medical decisions.
            
            ---
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Patient Information")
                
                age_input = gr.Number(
                    label="Age (years)",
                    value=40,
                    minimum=0,
                    maximum=120,
                    info="Patient age affects risk calculation"
                )
                
                gender_input = gr.Radio(
                    label="Gender",
                    choices=["Male", "Female", "Other"],
                    value="Male"
                )
                
                symptoms_input = gr.CheckboxGroup(
                    label="Select Symptoms (check all that apply)",
                    choices=common_symptoms,
                    info="Quick symptom selection"
                )
                
                clinical_notes_input = gr.Textbox(
                    label="Clinical Notes / Patient Description",
                    placeholder="Enter detailed patient presentation, vital signs, and clinical findings...\n\nExample: 'Patient complains of crushing chest pain radiating to left arm for 30 minutes. Diaphoretic, BP 90/60, HR 110.'",
                    lines=6,
                    info="Free-text clinical description (analyzed by AI)"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Patient", variant="primary", size="lg")
                
                status_output = gr.Textbox(
                    label="System Status",
                    value="Ready to analyze. Click 'Analyze Patient' to begin.",
                    interactive=False,
                    lines=1
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Triage Results")
                
                triage_output = gr.Textbox(
                    label="Triage Classification Summary",
                    lines=14,
                    interactive=False,
                    info="Combined rule-based and AI analysis with confidence scoring"
                )
                
                explanation_output = gr.Textbox(
                    label="Detailed Clinical Explanation",
                    lines=16,
                    interactive=False,
                    info="Evidence-based medical reasoning"
                )
                
                evaluation_output = gr.Textbox(
                    label="📈 Evaluation Metrics & ESI Classification",
                    lines=12,
                    interactive=False,
                    info="Confidence score, ESI level, and quality metrics"
                )
                
                with gr.Accordion("🤖 Gemma AI Model Output (Technical Details)", open=False):
                    gemma_output = gr.Textbox(
                        label="Raw Gemma Analysis",
                        lines=10,
                        interactive=False,
                        info="Structured output from Gemma 2B model"
                    )
        
        # Example cases
        with gr.Accordion("💡 Example Cases (Click to Load)", open=False):
            gr.Markdown("""
            **Emergency Example:** Chest pain, 65-year-old  
            **Urgent Example:** High fever with severe abdominal pain, 8-year-old  
            **Low Risk Example:** Mild cold symptoms, 30-year-old
            """)
            
            with gr.Row():
                emergency_btn = gr.Button("Load Emergency Case")
                urgent_btn = gr.Button("Load Urgent Case")
                low_btn = gr.Button("Load Low Risk Case")
            
            def load_emergency():
                return (
                    65, "Male",
                    ["Chest pain", "Shortness of breath"],
                    "Patient reports crushing chest pain radiating to left arm for past 30 minutes. Diaphoretic, nauseous. BP 90/60, HR 110. Patient appears in distress."
                )
            
            def load_urgent():
                return (
                    8, "Female",
                    ["High fever", "Abdominal pain", "Vomiting"],
                    "Pediatric patient with fever 104°F for 6 hours. Severe right lower quadrant abdominal pain. Vomiting 3 times. Parent reports decreased oral intake."
                )
            
            def load_low():
                return (
                    30, "Female",
                    ["Mild headache", "Runny nose"],
                    "Patient has mild headache and runny nose for 3 days. No fever. Able to eat and drink normally. Otherwise feels okay."
                )
            
            emergency_btn.click(
                fn=load_emergency,
                outputs=[age_input, gender_input, symptoms_input, clinical_notes_input]
            )
            
            urgent_btn.click(
                fn=load_urgent,
                outputs=[age_input, gender_input, symptoms_input, clinical_notes_input]
            )
            
            low_btn.click(
                fn=load_low,
                outputs=[age_input, gender_input, symptoms_input, clinical_notes_input]
            )
        
        # Connect analyze button
        analyze_btn.click(
            fn=triage_system.analyze_patient,
            inputs=[age_input, gender_input, symptoms_input, clinical_notes_input],
            outputs=[triage_output, explanation_output, evaluation_output, gemma_output, status_output]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### 📖 About This System
            
            **Architecture:**
            - **Rule-based Engine:** Evidence-based symptom detection with weighted scoring
            - **AI Reasoning:** Google Gemma 2B-IT for natural language understanding
            - **Hybrid Approach:** Conservative risk classification (takes higher of rule-based vs AI)
            - **Confidence Scoring:** Multi-factor confidence assessment (0-100 scale)
            - **ESI Compliance:** Emergency Severity Index aligned classification
            - **Explainable:** Full transparency into decision-making process
            
            **Technical Details:**
            - Model: Google Gemma 2B Instruction-Tuned
            - Platform: CPU-only inference (8GB RAM compatible)
            - Framework: Transformers + Gradio
            - Classification: Emergency / Urgent / Low risk (ESI Level 1 / 2-3 / 4-5)
            - Evaluation: Confidence scoring with 4 independent factors
            
            **Safety Features:**
            - Conservative risk stratification (prioritizes patient safety)
            - Age-adjusted risk scoring
            - Clinical evidence-based symptom weights
            - Multiple validation layers
            - Confidence scoring for decision quality
            - ESI-compliant triage levels
            
            **New Features:**
            ✅ **Confidence Scoring** - 4-factor confidence assessment (0-100)  
            ✅ **ESI Classification** - Emergency Severity Index compliant levels  
            ✅ **Evaluation Metrics** - Quality metrics and reliability indicators  
            ✅ **Enhanced Gemma Prompt** - Medical safety optimized prompting  
            
            **Kaggle Competition Compliance:**
            ✅ Uses Google Gemma (2B-IT)  
            ✅ Explainable AI with full transparency  
            ✅ CPU-only, lightweight architecture  
            ✅ Production-ready modular design  
            ✅ Medical safety prioritized  
            ✅ Comprehensive evaluation metrics  
            
            ---
            *Developed for educational and research purposes. Not for clinical use.*
            """
        )
    
    return demo


if __name__ == "__main__":
    print("="*60)
    print("AI EMERGENCY TRIAGE SUPPORT SYSTEM")
    print("="*60)
    print("\nInitializing system...")
    print("Note: Gemma model will load on first analysis (1-2 minutes)")
    print("\nStarting Gradio interface...")
    print("="*60 + "\n")
    
    # Create and launch UI
    demo = create_ui()
    
    # Launch the application
    # Note: Use 127.0.0.1 (localhost) for local access
    # Set share=True if you want a public URL
    demo.launch(
        server_name="127.0.0.1",  # Localhost access
        server_port=7860,
        share=False,  # Set to True to create public Gradio link
        show_error=True,
        inbrowser=True  # Automatically open browser
    )