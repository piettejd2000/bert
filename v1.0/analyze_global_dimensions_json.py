#!/usr/bin/env python3
# Copyright (c) 2025 John D. Piette
# Licensed under the "JD Piette Non-Commercial License".
# For commercial licensing, contact: piettejd2000@yahoo.com.
# See LICENSE.txt for full details.
"""
Analyze MI transcripts for global dimensions using trained global model
Outputs JSON format for GUI display
"""

import torch
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from huggingface_hub import hf_hub_download
import torch.nn as nn
import json

# --- Model hosting on HuggingFace Hub (works locally and on the cloud) ---
BEHAVIOR_MODEL_REPO = "piettejd2000/mi-behavior-roberta"
GLOBAL_MODEL_REPO = "piettejd2000/mi-global-roberta"
import os
import sys
import re
from collections import Counter
from datetime import datetime

# MI behavior labels - standardized codes
MI_BEHAVIORS = {
    "AF": "Affirm",
    "AS": "Emphasizing Autonomy",
    "CO": "Confront",
    "CQ": "Closed Question", 
    "CR": "Complex Reflection",
    "GI": "Giving Information",
    "OQ": "Open Question",
    "PE": "Persuade",
    "PWP": "Persuade with Permission",
    "SC": "Seeking Collaboration",
    "SR": "Simple Reflection"
}

# Positive MI behaviors to highlight
POSITIVE_BEHAVIORS = ["CR", "SC", "AF", "AS", "OQ"]

# Anti-MI behaviors (areas for improvement)
ANTI_MI_BEHAVIORS = ["CO", "PE"]

# Global dimensions
GLOBAL_DIMENSIONS = [
    "Partnership",
    "Empathy", 
    "Cultivating Change Talk",
    "Softening Sustain Talk"
]

# MITI score interpretations (1-5 scale)
MITI_INTERPRETATIONS = {
    "Partnership": {
        1: "Clinician explicitly assumes the expert role, giving little or no opportunity for client input",
        2: "Clinician structures the interaction with minimal collaboration",
        3: "Clinician incorporates client input while maintaining control of the interaction",
        4: "Clinician fosters collaboration with attention to client's ideas",
        5: "Clinician actively fosters power sharing throughout the interaction"
    },
    "Empathy": {
        1: "Clinician gives little or no attention to client's perspective",
        2: "Clinician makes sporadic efforts to explore client's perspective",
        3: "Clinician actively tries to understand client's perspective with modest success",
        4: "Clinician shows evidence of accurate understanding of client's worldview",
        5: "Clinician shows deep understanding of client's perspective throughout"
    },
    "Cultivating Change Talk": {
        1: "Clinician shows no explicit attention to or preference for change language",
        2: "Clinician sporadically attends to change talk with minimal development",
        3: "Clinician attends to change talk but misses some opportunities",
        4: "Clinician consistently attends to change talk and works to elicit it",
        5: "Clinician consistently cultivates change talk with sophisticated methods"
    },
    "Softening Sustain Talk": {
        1: "Clinician responds to sustain talk in ways that increase its frequency or depth",
        2: "Clinician responds to sustain talk without particular strategy",
        3: "Clinician makes some efforts to diminish sustain talk",
        4: "Clinician consistently avoids emphasizing sustain talk",
        5: "Clinician skillfully navigates sustain talk to minimize its impact"
    }
}

class RobertaForGlobalDimensions(nn.Module):
    """Custom model for global dimension scoring"""
    def __init__(self, base_model_name='roberta-base', num_outputs=4):
        super().__init__()
        # Always load with pretrained weights to avoid meta tensors
        # Do not use try/except that falls back to config-only initialization
        self.roberta = RobertaModel.from_pretrained(base_model_name)
        
        self.dropout = nn.Dropout(0.1)
        # Multi-layer regressor matching the saved model
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_outputs)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        scores = self.regressor(pooled)
        return scores

def extract_utterances(transcript_path=None, transcript_text=None):
    """Extract all utterances from transcript with context"""
    if transcript_path:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = transcript_text
    
    # Pattern to capture all utterances with speaker labels
    pattern = r'(Client|Clinician):\s*(.*?)(?=(?:Client|Clinician):|$)'
    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
    
    utterances = []
    last_client_utterance = None
    
    for speaker, text in matches:
        text = text.strip()
        if text:
            if speaker.lower() == 'client':
                last_client_utterance = text
                utterances.append({
                    'speaker': 'client',
                    'text': text,
                    'client_context': None
                })
            else:  # clinician
                utterances.append({
                    'speaker': 'clinician',
                    'text': text,
                    'client_context': last_client_utterance
                })
    
    return utterances

# Models are loaded once and kept in memory. On Streamlit (local or cloud) this
# module persists across reruns, so every analysis after the first reuses the
# loaded models instead of re-reading ~950 MB from disk each time.
_BEHAVIOR_MODEL_CACHE = None
_GLOBAL_MODEL_CACHE = None

def load_behavior_model():
    """Return the cached behavior model, loading it once on first use."""
    global _BEHAVIOR_MODEL_CACHE
    if _BEHAVIOR_MODEL_CACHE is None or _BEHAVIOR_MODEL_CACHE[0] is None:
        _BEHAVIOR_MODEL_CACHE = _load_behavior_model_uncached()
    return _BEHAVIOR_MODEL_CACHE

def _load_behavior_model_uncached():
    """Load the behavior classification model from HuggingFace Hub"""
    model_path = BEHAVIOR_MODEL_REPO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load model from safetensors format
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Use Auto classes to handle safetensors format
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=False
        )
        
        # Move to device
        model = model.to(device)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading behavior model: {e}", file=sys.stderr)
        # Try fallback method
        try:
            model = RobertaForSequenceClassification.from_pretrained(model_path)
            model = model.to(device) 
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
            return model, tokenizer, device
        except Exception as e2:
            print(f"Fallback also failed: {e2}", file=sys.stderr)
            return None, None, None

def load_global_model(device):
    """Return the cached global-dimensions model, loading it once on first use."""
    global _GLOBAL_MODEL_CACHE
    if _GLOBAL_MODEL_CACHE is None:
        global_ckpt_path = hf_hub_download(GLOBAL_MODEL_REPO, "global_model.pt")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForGlobalDimensions('roberta-base')
        model_state = torch.load(global_ckpt_path, map_location='cpu', weights_only=True)
        model.load_state_dict(model_state, strict=False)
        model = model.to(device)
        model.eval()
        _GLOBAL_MODEL_CACHE = (model, tokenizer)
    return _GLOBAL_MODEL_CACHE

def classify_behaviors(utterances, model, tokenizer, device):
    """Classify MI behaviors for clinician utterances with full text and context"""
    behavior_labels = list(MI_BEHAVIORS.keys())
    behaviors = []
    
    for utt in utterances:
        if utt['speaker'] == 'clinician':
            # Check for closing statements first to prevent misclassification
            text_lower = utt['text'].lower()
            closing_statement_indicators = [
                "see you soon", "see you next time", "have a great", "have a good",
                "have an excellent", "hasta la vista", "until next time", "take care",
                "goodbye", "good bye", "farewell", "best wishes", "wishing you",
                "hope you have", "enjoy the rest", "have a wonderful", "have a nice",
                "talk to you soon", "speak soon", "catch you later", "until we meet",
                "looking forward to", "see you at our next", "until then",
                "pronto", "hasta pronto", "que tengas", "buen día", "excelente día",
                "resto de tu día", "resto del día", "que pases", "que disfrutes"
            ]
            
            # Skip BERT classification for closing statements - label as Simple Reflection (neutral)
            if any(indicator in text_lower for indicator in closing_statement_indicators):
                behaviors.append({
                    'behavior': 'SR',  # Simple Reflection - neutral, non-confrontational
                    'confidence': 0.95,  # High confidence this is correct
                    'text': utt['text'],
                    'client_context': utt.get('client_context'),
                    'note': 'Closing statement detected - protected from misclassification'
                })
                continue
            
            # Use full text for classification (truncate only for model if needed)
            text_for_model = utt['text'][:1000] if len(utt['text']) > 1000 else utt['text']
            
            inputs = tokenizer(text_for_model, return_tensors="pt", truncation=True, max_length=256).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            behavior = behavior_labels[predicted_class]
            behaviors.append({
                'behavior': behavior,
                'confidence': confidence,
                'text': utt['text'],  # Store full text
                'client_context': utt.get('client_context')  # Include client context
            })
    
    return behaviors

def score_global_dimensions(transcript_text, model, tokenizer, device):
    """Score the 4 global dimensions for the transcript"""
    # Truncate if too long
    if len(transcript_text) > 2000:
        transcript_text = transcript_text[:2000]
    
    inputs = tokenizer(
        transcript_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        scores = model(inputs['input_ids'], inputs['attention_mask'])
        # Convert to 1-5 scale (assuming model outputs 0-1 or similar range)
        # Apply sigmoid and scale to 1-5
        scores = torch.sigmoid(scores) * 4 + 1
        scores = scores.cpu().numpy()[0]
    
    return scores

def get_score_interpretation(dimension, score):
    """Get MITI interpretation for a dimension score"""
    rounded_score = int(round(score))
    rounded_score = max(1, min(5, rounded_score))
    return MITI_INTERPRETATIONS[dimension][rounded_score]

def analyze_transcript(transcript_path=None, transcript_text=None, output_json_path=None):
    """Main analysis function - returns JSON-compatible dict"""
    
    # Extract utterances
    utterances = extract_utterances(transcript_path, transcript_text)
    clinician_count = sum(1 for u in utterances if u['speaker'] == 'clinician')
    
    # Create full transcript text
    full_transcript_text = " ".join([u['text'] for u in utterances])
    
    # Load global model (cached in memory after first load)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_global_model(device)
    
    # Score global dimensions
    scores = score_global_dimensions(full_transcript_text, model, tokenizer, device)
    
    # Load behavior model and analyze behaviors first (needed for excellence detection)
    behavior_model, behavior_tokenizer, device = load_behavior_model()
    behaviors = []
    if behavior_model:
        behaviors = classify_behaviors(utterances, behavior_model, behavior_tokenizer, device)
    
    # Apply MI Excellence Recognition and Score Adjustment
    from mi_excellence_booster import detect_mi_excellence_patterns, apply_excellence_adjustments
    
    # Create dimension scores dict for excellence analysis
    original_dimension_scores = {
        dimension: float(scores[i]) for i, dimension in enumerate(GLOBAL_DIMENSIONS)
    }
    
    # Detect excellence patterns and apply adjustments
    excellence_analysis = detect_mi_excellence_patterns(behaviors, utterances)
    excellence_results = apply_excellence_adjustments(
        original_dimension_scores, 
        excellence_analysis, 
        min_boost_threshold=0.4  # Require 40% confidence to boost
    )
    
    # Use adjusted scores if available
    final_scores = excellence_results['adjusted_scores']
    
    # Initialize results
    results = {
        "transcript_id": os.path.basename(transcript_path) if transcript_path else "direct_input",
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": "RoBERTa-base fine-tuned"
        },
        "global_dimensions": final_scores,
        "original_scores": original_dimension_scores,
        "excellence_analysis": excellence_results,
        "interpretations": {
            dimension: get_score_interpretation(dimension, final_scores[dimension]) 
            for dimension in GLOBAL_DIMENSIONS
        },
        "total_utterances": len(utterances),
        "clinician_utterances": clinician_count,
        "turns": []  # Will be populated with labeled utterances
    }
    
    # Continue with behavior analysis (already loaded above for excellence detection)
    if behavior_model:
        
        # Count behaviors - using proper codes for GUI compatibility
        behavior_counts = Counter([b['behavior'] for b in behaviors])
        
        # Convert to GUI-compatible format - INCLUDING ALL BEHAVIORS
        gui_behavior_counts = {}
        gui_behavior_counts["GI"] = behavior_counts.get("GI", 0)
        gui_behavior_counts["Q_Open"] = behavior_counts.get("OQ", 0)
        gui_behavior_counts["Q_Closed"] = behavior_counts.get("CQ", 0)
        gui_behavior_counts["CR"] = behavior_counts.get("CR", 0)
        gui_behavior_counts["SR"] = behavior_counts.get("SR", 0)
        gui_behavior_counts["AF"] = behavior_counts.get("AF", 0)
        gui_behavior_counts["CO"] = behavior_counts.get("CO", 0)
        gui_behavior_counts["PE"] = behavior_counts.get("PE", 0)
        gui_behavior_counts["SC"] = behavior_counts.get("SC", 0)
        gui_behavior_counts["AS"] = behavior_counts.get("AS", 0)
        gui_behavior_counts["PWP"] = behavior_counts.get("PWP", 0)
        
        results["behavior_counts"] = gui_behavior_counts
        results["behavior_distribution"] = dict(behavior_counts)
        
        # Find positive behaviors with examples
        positive_found = []
        for behavior in POSITIVE_BEHAVIORS:
            if behavior in behavior_counts:
                count = behavior_counts[behavior]
                name = MI_BEHAVIORS[behavior]
                # Only include examples that have client context
                examples = [b for b in behaviors if b['behavior'] == behavior and b.get('client_context')]
                if examples:
                    best_example = max(examples, key=lambda x: x['confidence'])
                    positive_found.append({
                        "code": behavior,
                        "name": name,
                        "count": count,
                        "example": {
                            "clinician_text": best_example['text'],
                            "client_context": best_example.get('client_context'),
                            "confidence": best_example['confidence']
                        }
                    })
        
        results["positive_behaviors"] = sorted(positive_found, key=lambda x: -x['count'])[:3]
        
        # Find anti-MI behaviors
        anti_found = []
        for behavior in ANTI_MI_BEHAVIORS:
            if behavior in behavior_counts:
                count = behavior_counts[behavior]
                name = MI_BEHAVIORS[behavior]
                # Only include examples that have client context
                examples = [b for b in behaviors if b['behavior'] == behavior and b.get('client_context')]
                if examples:
                    example = examples[0]
                    anti_found.append({
                        "code": behavior,
                        "name": name,
                        "count": count,
                        "example": {
                            "clinician_text": example['text'],
                            "client_context": example.get('client_context'),
                            "confidence": example['confidence']
                        }
                    })
        
        results["areas_for_improvement"] = sorted(anti_found, key=lambda x: -x['count'])[:3]
        
        # Create turns list for GUI display
        behavior_idx = 0
        for utt in utterances:
            turn = {
                "speaker": utt['speaker'].title(),
                "text": utt['text']
            }
            
            if utt['speaker'] == 'clinician' and behavior_idx < len(behaviors):
                behavior_info = behaviors[behavior_idx]
                # Map behavior code to GUI format
                gui_label = behavior_info['behavior']
                if gui_label == "OQ":
                    gui_label = "Q_Open"
                elif gui_label == "CQ":
                    gui_label = "Q_Closed"
                
                turn["labels"] = [gui_label]
                behavior_idx += 1
            else:
                turn["labels"] = []
            
            results["turns"].append(turn)
        
        # Add top excerpts for positive behaviors and anti-MI behaviors
        top_excerpts = []
        
        # Add positive behavior examples
        for pos in results["positive_behaviors"][:2]:
            if pos.get("example"):
                excerpt = {
                    "speaker": "Clinician",
                    "text": pos["example"]["clinician_text"],
                    "labels": [pos["code"] if pos["code"] not in ["OQ", "CQ"] else 
                              ("Q_Open" if pos["code"] == "OQ" else "Q_Closed")],
                    "note": f"Strong {pos['name']} example",
                    "score": pos["example"]["confidence"]
                }
                if pos["example"].get("client_context"):
                    # Add client context as separate excerpt
                    top_excerpts.append({
                        "speaker": "Client",
                        "text": pos["example"]["client_context"],
                        "labels": [],
                        "note": "Prior client statement"
                    })
                top_excerpts.append(excerpt)
        
        # Add anti-MI behavior examples
        for anti in results["areas_for_improvement"][:1]:
            if anti.get("example"):
                excerpt = {
                    "speaker": "Clinician",
                    "text": anti["example"]["clinician_text"],
                    "labels": [anti["code"]],
                    "note": f"Area for improvement: {anti['name']}",
                    "score": anti["example"]["confidence"]
                }
                if anti["example"].get("client_context"):
                    # Add client context as separate excerpt
                    top_excerpts.append({
                        "speaker": "Client",
                        "text": anti["example"]["client_context"],
                        "labels": [],
                        "note": "Prior client statement"
                    })
                top_excerpts.append(excerpt)
        
        results["top_excerpts"] = top_excerpts
    
    # Save to JSON if path provided
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_json_path}")
    
    return results

def analyze(text):
    """Simple interface for GUI integration - accepts text directly"""
    return analyze_transcript(transcript_text=text)

def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze MI transcript and output JSON')
    parser.add_argument('--in', dest='input_file', help='Input transcript file path')
    parser.add_argument('--out', dest='output_file', help='Output JSON file path')
    parser.add_argument('transcript', nargs='?', help='Transcript file (alternative to --in)')
    
    args = parser.parse_args()
    
    # Determine input file
    input_file = args.input_file or args.transcript
    
    if not input_file:
        # Default to a transcript
        input_file = "/Users/johnpiette/healthcare_rl/mi_prototype/testing/test_transcripts/generated_transcripts/3_Trans_Claude_4.txt"
    
    # Build full path if needed
    if not os.path.isabs(input_file):
        # Check if it's in transcripts folder
        transcript_path = f"/Users/johnpiette/healthcare_rl/mi_prototype/transcripts/{input_file}"
        if os.path.exists(transcript_path):
            input_file = transcript_path
        else:
            # Check in test transcripts
            test_path = f"/Users/johnpiette/healthcare_rl/mi_prototype/testing/test_transcripts/generated_transcripts/{input_file}"
            if os.path.exists(test_path):
                input_file = test_path
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Transcript not found at {input_file}")
        sys.exit(1)
    
    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        # Default output location
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"/Users/johnpiette/healthcare_rl/mi_prototype/trained_models/bert/jsons/{base_name}_analysis.json"
    
    print("=" * 80)
    print("🌟 MI Global Dimensions & Behavior Analysis (JSON Output)")
    print("=" * 80)
    print(f"📄 Input: {input_file}")
    print(f"💾 Output: {output_file}")
    print("=" * 80)
    
    # Analyze the transcript
    results = analyze_transcript(transcript_path=input_file, output_json_path=output_file)
    
    # Print summary
    print("\n📊 Analysis Complete!")
    print(f"Global Dimensions:")
    for dim, score in results["global_dimensions"].items():
        print(f"  - {dim}: {score:.2f}/5.0")
    
    if "behavior_counts" in results:
        print(f"\nBehavior Summary:")
        total_behaviors = sum(results["behavior_counts"].values())
        print(f"  Total clinician behaviors analyzed: {total_behaviors}")
        if results.get("positive_behaviors"):
            print(f"  Top positive behaviors: {', '.join([b['name'] for b in results['positive_behaviors'][:3]])}")
        if results.get("areas_for_improvement"):
            print(f"  Areas for improvement: {', '.join([b['name'] for b in results['areas_for_improvement']])}")

if __name__ == "__main__":
    main()