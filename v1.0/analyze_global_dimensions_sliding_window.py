#!/usr/bin/env python3
# Copyright (c) 2025 John D. Piette
# Licensed under the "JD Piette Non-Commercial License".
# For commercial licensing, contact: piettejd2000@yahoo.com.
# See LICENSE.txt for full details.
"""
Analyze MI transcripts for global dimensions using sliding window approach
Enhanced version that can handle conversations up to 45+ minutes
Outputs JSON format for GUI display
"""

import torch
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch.nn as nn
import json

# --- Model hosting on HuggingFace Hub (works locally and on the cloud) ---
BEHAVIOR_MODEL_REPO = "piettejd2000/mi-behavior-roberta"
GLOBAL_MODEL_REPO = "piettejd2000/mi-global-roberta"
import os
import sys
import re
import numpy as np
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import argparse
import io
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

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

# Full behavior names for display
BEHAVIOR_FULL_NAMES = {
    "GI": "Giving Information",
    "Q_Open": "Open Question",
    "Q_Closed": "Closed Question",
    "CR": "Complex Reflection",
    "SR": "Simple Reflection",
    "AF": "Affirm",
    "AS": "Emphasizing Autonomy",
    "SC": "Seeking Collaboration",
    "CO": "Confront",
    "PE": "Persuade",
    "OQ": "Open Question",
    "CQ": "Closed Question",
    "PWP": "Persuade with Permission"
}

def create_radar_chart(scores_dict):
    """Create radar chart for global dimensions"""
    labels = GLOBAL_DIMENSIONS
    values = [scores_dict.get(dim, 0) for dim in labels]
    
    # Number of dimensions
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]  # Complete the circle
    angles += angles[:1]
    
    # Create plot with equal aspect ratio
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=14)
    
    # Set y-axis (radial) limits and labels
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=12)
    ax.set_rlabel_position(0)
    
    # Plot data
    ax.plot(angles, values_plot, 'o-', linewidth=2, color='#2E86AB')
    ax.fill(angles, values_plot, alpha=0.25, color='#2E86AB')
    
    # Add grid
    ax.grid(True)
    
    # Add title
    plt.title("Global Dimensions Profile", size=18, y=1.08)
    
    # Ensure the plot is circular
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return fig

def generate_pdf_report(analysis_results, patient_id="", output_path=None):
    """Generate a PDF report from analysis results"""
    if output_path:
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
    else:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
    
    # Container for PDF elements
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        textColor=HexColor('#1e3a8a')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        textColor=HexColor('#3730a3')
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY
    )
    
    # Add header function for each page
    def add_header_footer(canvas, doc):
        canvas.saveState()
        # Header with patient ID if provided
        if patient_id:
            canvas.setFont('Helvetica', 9)
            canvas.drawString(72, letter[1] - 40, f"Patient ID: {patient_id}")
            canvas.drawString(letter[0] - 180, letter[1] - 40, f"Page {doc.page}")
        else:
            canvas.drawString(letter[0] - 100, letter[1] - 40, f"Page {doc.page}")
        
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.drawString(72, 40, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        canvas.drawString(letter[0] - 250, 40, "MI Transcript Analysis Report")
        canvas.restoreState()
    
    # Title
    story.append(Paragraph("MI Transcript Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient ID if provided
    if patient_id:
        story.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", normal_style))
        story.append(Spacer(1, 10))
    
    # Transcript info
    story.append(Paragraph(f"<b>Transcript ID:</b> {analysis_results.get('transcript_id', 'Unknown')}", normal_style))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    story.append(Paragraph(f"<b>Total Utterances:</b> {analysis_results.get('total_utterances', 0)}", normal_style))
    story.append(Paragraph(f"<b>Clinician Utterances:</b> {analysis_results.get('clinician_utterances', 0)}", normal_style))
    
    # Add analysis method info for sliding window
    if analysis_results.get('meta', {}).get('analysis_method') == 'sliding_window':
        window_config = analysis_results.get('meta', {}).get('window_config', {})
        story.append(Paragraph(f"<b>Analysis Method:</b> Enhanced Sliding Window", normal_style))
        story.append(Paragraph(f"<b>Windows Analyzed:</b> {window_config.get('total_windows', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Total Tokens:</b> {window_config.get('total_tokens_estimated', 'N/A')}", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Global Dimensions Section
    story.append(Paragraph("Global Dimensions Scores", heading_style))
    story.append(Spacer(1, 10))
    
    global_scores = analysis_results.get('global_dimensions', {})
    
    # Create table for global dimensions
    dim_data = [['Dimension', 'Score', 'Level']]
    for dimension in GLOBAL_DIMENSIONS:
        score = global_scores.get(dimension, 0)
        interpretation = 'High' if score >= 4.0 else 'Medium' if score >= 3.0 else 'Low'
        dim_data.append([dimension, f"{score:.2f}/5.0", interpretation])
    
    dim_table = Table(dim_data, colWidths=[3*inch, 1*inch, 0.8*inch])
    dim_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e0e7ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1e3a8a')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#000000')),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(dim_table)
    story.append(Spacer(1, 20))
    
    # MITI Interpretations
    if analysis_results.get('interpretations'):
        story.append(Paragraph("MITI Interpretations", subheading_style))
        for dimension in GLOBAL_DIMENSIONS:
            if dimension in analysis_results['interpretations']:
                story.append(Paragraph(f"<b>{dimension}:</b>", normal_style))
                story.append(Paragraph(analysis_results['interpretations'][dimension], normal_style))
                story.append(Spacer(1, 8))
    
    # Generate and add radar chart
    try:
        fig = create_radar_chart(global_scores)
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        # Keep aspect ratio by not forcing both dimensions
        img = Image(img_buffer, width=4*inch, height=4*inch, kind='proportional')
        story.append(img)
        plt.close(fig)
    except Exception as e:
        story.append(Paragraph(f"Chart generation error: {str(e)}", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Behavior Analysis Section
    story.append(Paragraph("Behavior Analysis", heading_style))
    story.append(Spacer(1, 10))
    
    # Match UI exactly: 2-column layout with Behavior Distribution and Strengths/Areas
    behavior_counts = analysis_results.get('behavior_counts', {})
    positive_behaviors = analysis_results.get('positive_behaviors', [])
    areas_for_improvement = analysis_results.get('areas_for_improvement', [])
    
    # LEFT COLUMN: Behavior Distribution
    behavior_dist_content = []
    behavior_dist_content.append(Paragraph("<b>Behavior Distribution</b>", subheading_style))
    
    # Include ALL MI behaviors from the model (remove duplicates)
    MI_BEHAVIOR_LABELS = ["GI", "OQ", "CQ", "CR", "SR", "AF", "AS", "SC", "PWP", "CO", "PE"]
    
    # Create behavior table
    behavior_data = [['MI Behavior Label', 'Count']]
    has_behaviors = False
    
    for label in MI_BEHAVIOR_LABELS:
        count = behavior_counts.get(label, 0)
        if count > 0:
            has_behaviors = True
            full_name = BEHAVIOR_FULL_NAMES.get(label, label)
            behavior_data.append([full_name, str(count)])
    
    if has_behaviors and len(behavior_data) > 1:
        behavior_table = Table(behavior_data, colWidths=[2.2*inch, 0.7*inch])
        behavior_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#e0e7ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#1e3a8a')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#808080')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ]))
        behavior_dist_content.append(behavior_table)
    else:
        behavior_dist_content.append(Paragraph("No behavior data available", normal_style))
    
    # RIGHT COLUMN: Strengths and Areas for Improvement
    right_column_content = []
    
    # Strengths section
    right_column_content.append(Paragraph("<b>Strengths</b>", subheading_style))
    if positive_behaviors:
        for behavior in positive_behaviors[:3]:
            right_column_content.append(Paragraph(
                f"• <b>{behavior['name']}</b> ({behavior['code']}): {behavior['count']} instances",
                normal_style
            ))
    else:
        right_column_content.append(Paragraph("No positive behaviors identified", normal_style))
    
    right_column_content.append(Spacer(1, 8))
    
    # Areas for Improvement section  
    right_column_content.append(Paragraph("<b>Areas for Improvement</b>", subheading_style))
    if areas_for_improvement:
        for behavior in areas_for_improvement:
            right_column_content.append(Paragraph(
                f"• <b>{behavior['name']}</b> ({behavior['code']}): {behavior['count']} instances",
                normal_style
            ))
    else:
        right_column_content.append(Paragraph("No anti-MI behaviors detected!", normal_style))
    
    # Create 2-column layout matching UI
    layout_data = [[behavior_dist_content, right_column_content]]
    
    layout_table = Table(layout_data, colWidths=[3*inch, 3.2*inch])
    layout_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (0, 0), 0),
        ('LEFTPADDING', (1, 0), (1, 0), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    story.append(layout_table)
    
    story.append(Spacer(1, 5))
    # Page break before key examples
    story.append(PageBreak())
    
    # Key Examples Section
    story.append(Paragraph("Key Examples", heading_style))
    story.append(Spacer(1, 10))
    
    top_excerpts = analysis_results.get('top_excerpts', [])
    if top_excerpts:
        for excerpt in top_excerpts:
            speaker = excerpt.get('speaker', 'Unknown')
            text = excerpt.get('text', '')
            labels = excerpt.get('labels', [])
            note = excerpt.get('note', '')
            
            # Format text - escape XML special characters
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Format speaker text using same style as Full Transcript
            speaker_style = ParagraphStyle(
                'ExampleSpeaker',
                parent=normal_style,
                fontSize=10,
                leftIndent=0 if speaker == 'Clinician' else 20
            )
            
            story.append(Paragraph(f"<b>{speaker}:</b> {text}", speaker_style))
            
            # Add behavior labels with full names for clinician utterances
            if labels and speaker == 'Clinician':
                full_names = [BEHAVIOR_FULL_NAMES.get(label, label) for label in labels]
                story.append(Paragraph(
                    f"<font color='blue'><i>[{', '.join(full_names)}]</i></font>",
                    speaker_style
                ))
            
            # Add note if present
            if note:
                note_style = ParagraphStyle(
                    'Note',
                    parent=normal_style,
                    fontSize=9,
                    leftIndent=10,
                    textColor=HexColor('#666666')
                )
                story.append(Paragraph(f"<i>Note: {note}</i>", note_style))
            
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("No key examples available for this transcript.", normal_style))
        story.append(Spacer(1, 10))
    
    # Page break before full transcript
    story.append(PageBreak())
    
    # Full Transcript Section (labeled)
    story.append(Paragraph("Full Transcript with Behavior Labels", heading_style))
    story.append(Spacer(1, 10))
    
    turns = analysis_results.get('turns', [])
    if turns:
        for turn in turns[:100]:  # Limit to first 100 turns for PDF size
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            labels = turn.get('labels', [])
            
            # Format text - escape XML special characters
            text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Format speaker text
            speaker_style = ParagraphStyle(
                'Speaker',
                parent=normal_style,
                fontSize=10,
                leftIndent=0 if speaker == 'Clinician' else 20
            )
            
            story.append(Paragraph(f"<b>{speaker}:</b> {text}", speaker_style))
            
            if labels and speaker == 'Clinician':
                full_names = [BEHAVIOR_FULL_NAMES.get(label, label) for label in labels]
                story.append(Paragraph(
                    f"<font color='blue'><i>[{', '.join(full_names)}]</i></font>",
                    speaker_style
                ))
            
            story.append(Spacer(1, 6))
        
        if len(turns) > 100:
            story.append(Paragraph(
                f"<i>Note: Transcript truncated. Showing first 100 of {len(turns)} turns.</i>",
                normal_style
            ))
    else:
        story.append(Paragraph("No transcript data available.", normal_style))
    
    # Build PDF
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    
    # Return PDF data if using buffer
    if not output_path:
        buffer.seek(0)
        return buffer.getvalue()
    else:
        return None

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
        pooled_output = self.dropout(outputs.pooler_output)
        return self.regressor(pooled_output)

def get_score_interpretation(dimension: str, score: float) -> str:
    """Get text interpretation for MITI score"""
    score_rounded = max(1, min(5, round(score)))
    return MITI_INTERPRETATIONS[dimension].get(score_rounded, f"Score: {score:.2f}")

def parse_transcript(text: str) -> List[Dict]:
    """Parse transcript text into structured utterances with client context"""
    lines = text.strip().split('\n')
    utterances = []
    last_client_utterance = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match speaker patterns
        if line.lower().startswith('client:') or line.lower().startswith('patient:'):
            text = line.split(':', 1)[1].strip()
            last_client_utterance = text
            utterances.append({
                'speaker': 'client',
                'text': text,
                'client_context': None
            })
        elif line.lower().startswith('clinician:') or line.lower().startswith('therapist:') or line.lower().startswith('counselor:'):
            text = line.split(':', 1)[1].strip()
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

def classify_behaviors(utterances: List[Dict], model, tokenizer, device) -> List[Dict]:
    """Classify MI behaviors for clinician utterances with full text and context"""
    behavior_labels = list(MI_BEHAVIORS.keys())
    behaviors = []
    
    for utt in utterances:
        if utt['speaker'] == 'clinician':
            # Use full text for classification (truncate only for model if needed)
            text_for_model = utt['text'][:1000] if len(utt['text']) > 1000 else utt['text']
            
            inputs = tokenizer(text_for_model, return_tensors="pt", truncation=True, max_length=256).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                # Get the most likely behavior
                predicted_idx = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_idx].item()
                behavior = behavior_labels[predicted_idx]
                
            behaviors.append({
                'behavior': behavior,
                'confidence': confidence,
                'text': utt['text'],  # Store full text
                'client_context': utt.get('client_context')  # Include client context
            })
    
    return behaviors

def create_sliding_windows(text: str, tokenizer, max_length: int = 512, overlap_ratio: float = 0.2) -> List[str]:
    """
    Create overlapping text windows that fit within token limits
    
    Args:
        text: Full transcript text
        tokenizer: RoBERTa tokenizer
        max_length: Maximum tokens per window
        overlap_ratio: Fraction of overlap between windows (0.0 to 0.5)
    
    Returns:
        List of text windows
    """
    # Tokenize full text to understand total length
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_length:
        # Text fits in single window
        return [text]
    
    # Calculate step size with overlap
    overlap_tokens = int(max_length * overlap_ratio)
    step_size = max_length - overlap_tokens
    
    windows = []
    start_idx = 0
    
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_length, len(tokens))
        window_tokens = tokens[start_idx:end_idx]
        
        # Decode back to text
        window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)
        windows.append(window_text)
        
        # Move to next window
        start_idx += step_size
        
        # Stop if we've covered all tokens
        if end_idx >= len(tokens):
            break
    
    return windows

def score_global_dimensions_sliding_window(text: str, model, tokenizer, device, 
                                         max_length: int = 512, overlap_ratio: float = 0.25) -> np.ndarray:
    """
    Score global dimensions using sliding window approach
    
    Args:
        text: Full transcript text
        model: Global dimensions model
        tokenizer: RoBERTa tokenizer
        device: PyTorch device
        max_length: Maximum tokens per window
        overlap_ratio: Overlap between windows
    
    Returns:
        Averaged scores for global dimensions
    """
    windows = create_sliding_windows(text, tokenizer, max_length, overlap_ratio)
    
    all_scores = []
    
    for window in windows:
        # Tokenize and score each window
        inputs = tokenizer(
            window,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            # Apply sigmoid to get scores in 0-1 range, then scale to 1-5
            scores = torch.sigmoid(outputs).cpu().numpy()[0]
            scores = scores * 4 + 1  # Scale from [0,1] to [1,5]
            all_scores.append(scores)
    
    # Average scores across all windows
    if len(all_scores) == 1:
        final_scores = all_scores[0]
    else:
        # Weighted average - give more weight to middle windows for longer conversations
        weights = np.ones(len(all_scores))
        if len(all_scores) >= 3:
            # Reduce weight of first and last windows slightly
            weights[0] = 0.8
            weights[-1] = 0.8
        
        weights = weights / weights.sum()  # Normalize
        final_scores = np.average(all_scores, axis=0, weights=weights)
    
    return final_scores

def analyze_transcript_sliding_window(transcript_path: Optional[str] = None, 
                                    transcript_text: Optional[str] = None,
                                    output_json_path: Optional[str] = None,
                                    max_window_tokens: int = 512,
                                    window_overlap: float = 0.25) -> Optional[Dict]:
    """
    Analyze transcript with sliding window approach for global dimensions
    
    Args:
        transcript_path: Path to transcript file
        transcript_text: Direct transcript text
        output_json_path: Where to save JSON results
        max_window_tokens: Maximum tokens per analysis window
        window_overlap: Overlap ratio between windows (0.0 to 0.5)
    
    Returns:
        Analysis results dictionary
    """
    
    # Load transcript
    if transcript_text:
        text = transcript_text
    elif transcript_path and os.path.exists(transcript_path):
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("No transcript provided", file=sys.stderr)
        return None
    
    # Parse into utterances
    utterances = parse_transcript(text)
    
    if not utterances:
        print("No utterances found in transcript", file=sys.stderr)
        return None
    
    # Load behavior model and classify behaviors
    behavior_model, behavior_tokenizer, device = load_behavior_model()
    
    behaviors = []
    if behavior_model:
        behaviors = classify_behaviors(utterances, behavior_model, behavior_tokenizer, device)
    
    # Create full transcript text for global analysis
    full_transcript_text = " ".join([u['text'] for u in utterances])
    
    # Load global model (cached in memory after first load)
    model, tokenizer = load_global_model(device)
    
    # Score global dimensions using sliding window approach
    scores = score_global_dimensions_sliding_window(
        full_transcript_text, model, tokenizer, device,
        max_length=max_window_tokens,
        overlap_ratio=window_overlap
    )
    
    # Calculate metrics
    total_utterances = len(utterances)
    clinician_utterances = sum(1 for u in utterances if u['speaker'] == 'clinician')
    
    # Count behaviors
    behavior_counts = Counter(b['behavior'] for b in behaviors)
    
    # Initialize results with sliding window metadata
    results = {
        "transcript_id": os.path.basename(transcript_path) if transcript_path else "direct_input",
        "meta": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": "RoBERTa-base fine-tuned with sliding window analysis",
            "analysis_method": "sliding_window",
            "window_config": {
                "max_tokens_per_window": max_window_tokens,
                "overlap_ratio": window_overlap,
                "total_windows": len(create_sliding_windows(full_transcript_text, tokenizer, max_window_tokens, window_overlap)),
                "total_tokens_estimated": len(tokenizer.encode(full_transcript_text))
            }
        },
        "global_dimensions": {
            dimension: float(scores[i]) for i, dimension in enumerate(GLOBAL_DIMENSIONS)
        },
        "interpretations": {
            dimension: get_score_interpretation(dimension, scores[i]) 
            for i, dimension in enumerate(GLOBAL_DIMENSIONS)
        },
        "total_utterances": total_utterances,
        "clinician_utterances": clinician_utterances,
        "turns": []
    }
    
    # Add behavior analysis if available
    if behaviors:
        # Map behavior codes to GUI format - include ALL behaviors
        gui_behavior_counts = {"GI": 0, "OQ": 0, "Q_Open": 0, "CQ": 0, "Q_Closed": 0, "CR": 0, "SR": 0, "AF": 0, "SC": 0, "AS": 0, "PWP": 0, "CO": 0, "PE": 0}
        gui_behavior_counts["GI"] = behavior_counts.get("GI", 0)
        gui_behavior_counts["OQ"] = behavior_counts.get("OQ", 0)  # Direct OQ mapping
        gui_behavior_counts["Q_Open"] = behavior_counts.get("OQ", 0)  # Also map to Q_Open for compatibility
        gui_behavior_counts["CQ"] = behavior_counts.get("CQ", 0)  # Direct CQ mapping
        gui_behavior_counts["Q_Closed"] = behavior_counts.get("CQ", 0)  # Also map to Q_Closed for compatibility
        gui_behavior_counts["CR"] = behavior_counts.get("CR", 0)
        gui_behavior_counts["SR"] = behavior_counts.get("SR", 0)
        gui_behavior_counts["AF"] = behavior_counts.get("AF", 0)
        gui_behavior_counts["SC"] = behavior_counts.get("SC", 0)  # Added SC
        gui_behavior_counts["AS"] = behavior_counts.get("AS", 0)  # Added AS
        gui_behavior_counts["PWP"] = behavior_counts.get("PWP", 0)  # Added PWP
        gui_behavior_counts["CO"] = behavior_counts.get("CO", 0)
        gui_behavior_counts["PE"] = behavior_counts.get("PE", 0)
        
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
        print(f"Analysis saved to: {output_json_path}")
    
    return results

def main():
    """Main function for command line usage"""
    
    parser = argparse.ArgumentParser(description="Analyze MI transcript with sliding window approach")
    parser.add_argument("--in", dest="input_path", help="Input transcript file path")
    parser.add_argument("--out", dest="output_path", help="Output JSON file path")
    parser.add_argument("--pdf", dest="pdf_path", help="Output PDF file path (optional)")
    parser.add_argument("--patient-id", dest="patient_id", default="", 
                       help="Patient ID for PDF header (optional)")
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=512,
                       help="Maximum tokens per window (default: 512)")
    parser.add_argument("--overlap", dest="overlap", type=float, default=0.25,
                       help="Window overlap ratio (default: 0.25)")
    
    args = parser.parse_args()
    
    if not args.input_path:
        print("Error: Input transcript path required (--in)", file=sys.stderr)
        sys.exit(1)
    
    if not args.output_path:
        print("Error: Output JSON path required (--out)", file=sys.stderr)
        sys.exit(1)
    
    results = analyze_transcript_sliding_window(
        transcript_path=args.input_path,
        output_json_path=args.output_path,
        max_window_tokens=args.max_tokens,
        window_overlap=args.overlap
    )
    
    if results:
        print("✅ Analysis completed successfully!")
        windows_used = results["meta"]["window_config"]["total_windows"]
        total_tokens = results["meta"]["window_config"]["total_tokens_estimated"]
        print(f"   Processed {total_tokens} tokens using {windows_used} overlapping windows")
        
        for dimension, score in results["global_dimensions"].items():
            print(f"   {dimension}: {score:.2f}")
        
        # Generate PDF if requested
        if args.pdf_path:
            print("Generating PDF report...")
            try:
                generate_pdf_report(results, args.patient_id, args.pdf_path)
                print(f"✅ PDF report saved to: {args.pdf_path}")
            except Exception as e:
                print(f"❌ PDF generation failed: {str(e)}")
                sys.exit(1)
    else:
        print("❌ Analysis failed")
        sys.exit(1)

if __name__ == "__main__":
    main()