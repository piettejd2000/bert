#!/usr/bin/env python3
# Copyright (c) 2025 John D. Piette
# Licensed under the "JD Piette Non-Commercial License".
# For commercial licensing, contact: piettejd2000@yahoo.com.
# See LICENSE.txt for full details.
"""
MI Results Dashboard - Streamlit GUI for visualizing MI transcript analysis
Integrates with analyze_global_dimensions_json.py for automatic analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
import io
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

# Add this script's own directory to the path so the analyzer modules import
# whether run locally or on the cloud (they sit in the same folder).
APP_DIR = Path(__file__).resolve().parent
sys.path.append(str(APP_DIR))

st.set_page_config(
    page_title="MI Transcript Analysis Dashboard",
    page_icon="🌟",
    layout="wide"
)

# Custom CSS for font sizes
DEBUG_MODE = False  # Set to True to see element labels

if DEBUG_MODE:
    # Debug CSS - shows borders and labels for each element type
    st.markdown("""
    <style>
        /* Debug mode - shows element types with colored borders */
        .stMarkdown h1 {
            border: 3px solid red !important;
            position: relative !important;
        }
        .stMarkdown h1::before {
            content: "H1: " !important;
            color: red !important;
            font-size: 12px !important;
            position: absolute !important;
            top: -15px !important;
            left: 0 !important;
        }
        
        .stMarkdown h2 {
            border: 3px solid orange !important;
            position: relative !important;
        }
        .stMarkdown h2::before {
            content: "H2: " !important;
            color: orange !important;
            font-size: 12px !important;
        }
        
        .stMarkdown h3 {
            border: 3px solid yellow !important;
            position: relative !important;
        }
        .stMarkdown h3::before {
            content: "H3: " !important;
            color: goldenrod !important;
            font-size: 12px !important;
        }
        
        .stMarkdown p {
            border: 1px solid blue !important;
        }
        
        [data-testid="metric-container"] {
            border: 2px solid purple !important;
            position: relative !important;
        }
        
        .stButton > button {
            border: 3px solid green !important;
        }
        
        [data-testid="caption"] {
            border: 2px solid pink !important;
        }
        
        .stAlert {
            border: 3px solid brown !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Main CSS for font sizes
st.markdown("""
<style>
    /* Main content area font sizes */
    .stMarkdown p, .stMarkdown {
        font-size: 18px !important;
    }
    
    /* Headers */
    .stMarkdown h1 {
        font-size: 42px !important;
    }
    .stMarkdown h2 {
        font-size: 36px !important;
    }
    .stMarkdown h3 {
        font-size: 28px !important;
    }
    
    /* Metrics (the big numbers) */
    [data-testid="metric-container"] {
        font-size: 20px !important;
    }
    [data-testid="metric-container"] > div:nth-child(1) {
        font-size: 20px !important;
    }
    [data-testid="metric-container"] > div:nth-child(2) {
        font-size: 32px !important;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        font-size: 16px !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-size: 18px !important;
        padding: 10px 20px !important;
    }
    
    /* Tab labels */
    .stTabs [data-baseweb="tab-list"] {
        font-size: 18px !important;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        font-size: 18px !important;
    }
    
    /* Info/warning/error messages */
    .stAlert {
        font-size: 18px !important;
    }
    
    /* Caption text (smaller text) */
    .caption, [data-testid="caption"] {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# MI behavior mappings for display
# Include ALL MI behaviors from the model - using actual labels from JSON
MI_BEHAVIOR_LABELS = ["GI", "Q_Open", "Q_Closed", "CR", "SR", "AF", "AS", "SC", "PWP", "CO", "PE"]
GLOBAL_DIMENSIONS = ["Partnership", "Empathy", "Cultivating Change Talk", "Softening Sustain Talk"]

# Full behavior names for display
BEHAVIOR_FULL_NAMES = {
    "GI": "Giving Information",
    "Q_Open": "Open Question",
    "Q_Closed": "Closed Question",
    "CR": "Complex Reflection",
    "SR": "Simple Reflection",
    "AF": "Affirm",
    "CO": "Confront",
    "PE": "Persuade",
    "OQ": "Open Question",
    "CQ": "Closed Question",
    "AS": "Emphasizing Autonomy",
    "SC": "Seeking Collaboration",
    "PWP": "Persuade with Permission"
}

# Color palette for behaviors
BEHAVIOR_COLORS = {
    "GI": "#e0f7fa",      # Light cyan
    "Q_Open": "#e8f5e9",   # Light green
    "Q_Closed": "#fff3e0", # Light orange
    "CR": "#ede7f6",       # Light purple
    "SR": "#f3e5f5",       # Light pink
    "AF": "#e8eaf6",       # Light indigo
    "CO": "#ffebee",       # Light red (anti-MI)
    "PE": "#ffcdd2",       # Light red (anti-MI)
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
    ax.set_xticklabels(labels, size=14)  # Increased from 10
    
    # Set y-axis (radial) limits and labels
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], size=12)  # Increased from 8
    ax.set_rlabel_position(0)
    
    # Plot data
    ax.plot(angles, values_plot, 'o-', linewidth=2, color='#2E86AB')
    ax.fill(angles, values_plot, alpha=0.25, color='#2E86AB')
    
    # Add grid
    ax.grid(True)
    
    # Add title
    plt.title("Global Dimensions Profile", size=18, y=1.08)  # Increased from 14
    
    # Ensure the plot is circular
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return fig

def get_score_color(score):
    """Get color based on score value"""
    if score >= 4.0:
        return "#4CAF50"  # Green
    elif score >= 3.0:
        return "#FFA726"  # Orange
    else:
        return "#EF5350"  # Red

def generate_pdf_report(analysis_results, patient_id=""):
    """Generate a PDF report from analysis results"""
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

    label_style = ParagraphStyle(
        'CustomLabel',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        spaceAfter=6,
        textColor=HexColor('#6b7280')
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
    behavior_dist_content.append(Paragraph("<i>Includes both primary and secondary behaviors</i>", label_style))
    
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
    story.append(Spacer(1, 10))
    
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
            
            # Format speaker text - escape XML special characters
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
    
    # Get PDF data
    buffer.seek(0)
    return buffer.getvalue()

def run_analyzer(transcript_text, transcript_filename="temp_transcript.txt", use_sliding_window=True):
    """Run the analysis script on transcript text with optional sliding window approach"""
    try:
        # First try to import directly
        try:
            if use_sliding_window:
                from analyze_global_dimensions_sliding_window import analyze_transcript_sliding_window as analyze_func
            else:
                from analyze_global_dimensions_json import analyze_transcript as analyze_func
            
            # Create output path for JSON (temp dir; works locally and on the cloud)
            output_dir = os.path.join(tempfile.gettempdir(), "mi_analysis_output")
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(transcript_filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_analysis.json")
            
            # Run analysis
            if use_sliding_window:
                results = analyze_func(
                    transcript_text=transcript_text, 
                    output_json_path=output_path,
                    max_window_tokens=512,
                    window_overlap=0.25
                )
            else:
                results = analyze_func(transcript_text=transcript_text, output_json_path=output_path)
            
            return results, None
            
        except ImportError as import_error:
            # If import fails, try running as subprocess with explicit python3
            # (subprocess and tempfile are already imported at module level)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_input:
                tmp_input.write(transcript_text)
                tmp_input_path = tmp_input.name
            
            try:
                # Create output path for JSON (temp dir)
                output_dir = os.path.join(tempfile.gettempdir(), "mi_analysis_output")
                os.makedirs(output_dir, exist_ok=True)
                
                base_name = os.path.splitext(transcript_filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}_analysis.json")
                
                # Try different Python executables
                # Check if we have a saved Python config
                python_config_path = str(APP_DIR / ".python_config")
                python_executables = []
                
                if os.path.exists(python_config_path):
                    with open(python_config_path, 'r') as f:
                        preferred_python = f.read().strip()
                        if preferred_python:
                            python_executables.append(preferred_python)
                
                # Add other candidates
                python_executables.extend([
                    "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",  # Known working Python
                    sys.executable,  # Current Python interpreter
                    "python3",       # System python3
                    "/usr/bin/python3",  # Explicit system python3
                    "/usr/local/bin/python3",  # Homebrew python3
                ])
                
                last_error = None
                for python_exe in python_executables:
                    try:
                        cmd = [
                            python_exe,
                            str(APP_DIR / "analyze_global_dimensions_json.py"),
                            "--in", tmp_input_path,
                            "--out", output_path
                        ]
                        
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=60,
                            cwd=str(APP_DIR)  # Set working directory
                        )
                        
                        if result.returncode == 0:
                            break  # Success, exit the loop
                        else:
                            last_error = f"{python_exe}: {result.stderr}"
                            
                    except (FileNotFoundError, OSError) as e:
                        last_error = f"{python_exe}: {str(e)}"
                        continue
                
                if result.returncode != 0:
                    raise RuntimeError(f"Analysis failed with all Python executables. Last error: {last_error}")
                
                # Load the generated JSON
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    return results, None
                else:
                    raise FileNotFoundError("Output JSON not created")
                    
            finally:
                # Clean up temp file
                if os.path.exists(tmp_input_path):
                    os.unlink(tmp_input_path)
        
    except Exception as e:
        return None, str(e)

# Sidebar configuration
with st.sidebar:
    st.title("📁 Data Input")
    
    input_method = st.radio(
        "Select input method:",
        ["Upload Transcript (.txt)", "Upload Analysis Results (.json)", "Select Example Transcript"]
    )
    
    uploaded_file = None
    selected_example = None
    
    if input_method == "Upload Transcript (.txt)":
        uploaded_file = st.file_uploader(
            "Choose a transcript file",
            type=['txt'],
            help="Upload a transcript with 'Client:' and 'Clinician:' speaker labels"
        )
        
    elif input_method == "Upload Analysis Results (.json)":
        uploaded_file = st.file_uploader(
            "Choose a JSON results file",
            type=['json'],
            help="Upload a pre-analyzed JSON file from the analyzer"
        )
        
    else:  # Select Example Transcript
        transcript_dir = str(APP_DIR / "examples")
        if os.path.exists(transcript_dir):
            transcript_files = [f for f in os.listdir(transcript_dir) if f.endswith('.txt')]
            if transcript_files:
                selected_example = st.selectbox(
                    "Select an example transcript:",
                    transcript_files[:20]  # Show first 20 files
                )
    
    st.divider()
    
    # Analysis settings
    st.subheader("⚙️ Analysis Settings")
    auto_analyze = st.checkbox(
        "Auto-analyze transcripts",
        value=True,
        help="Automatically run analysis when a transcript is uploaded"
    )
    
    show_interpretations = st.checkbox(
        "Show MITI interpretations",
        value=True,
        help="Display detailed MITI score interpretations"
    )
    
    use_sliding_window = st.checkbox(
        "Enhanced analysis for long conversations",
        value=True,
        help="Use sliding window approach for conversations longer than ~3 minutes. Recommended for 30+ minute conversations."
    )
    
    st.divider()
    
    # PDF Export settings
    st.subheader("📄 PDF Export")
    export_pdf = st.checkbox(
        "Export to PDF",
        value=False,
        help="Generate a PDF report of the analysis results"
    )
    
    patient_id = ""
    if export_pdf:
        patient_id = st.text_input(
            "Patient ID",
            placeholder="Enter patient ID (optional)",
            help="This will be displayed at the top of each page"
        )
    
    st.divider()
    
    # Info section
    st.info(
        "💡 **Tips:**\n"
        "- Upload a transcript to automatically analyze it\n"
        "- Upload a JSON file to view pre-analyzed results\n"
        "- Use example transcripts for testing"
    )

# Main content area
st.title("🌟 MI Transcript Analysis Dashboard")
st.caption("*v2.0 - with secondary behavior detection*")
st.markdown("Comprehensive analysis of Motivational Interviewing transcripts using MITI 4.2.1 criteria")

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# Process input
analysis_results = None
error_message = None

if uploaded_file is not None:
    if input_method == "Upload Analysis Results (.json)":
        # Load JSON directly
        try:
            content = uploaded_file.read()
            analysis_results = json.loads(content)
            st.session_state.analysis_results = analysis_results
            st.session_state.error_message = None
        except Exception as e:
            error_message = f"Error loading JSON: {str(e)}"
            st.session_state.error_message = error_message
            
    elif input_method == "Upload Transcript (.txt)" and auto_analyze:
        # Read transcript and analyze
        try:
            transcript_text = uploaded_file.read().decode('utf-8')
            
            with st.spinner("🔄 Analyzing transcript... This may take a moment."):
                analysis_results, error = run_analyzer(
                    transcript_text,
                    uploaded_file.name,
                    use_sliding_window
                )
                
            if error:
                error_message = f"Analysis error: {error}"
                st.session_state.error_message = error_message
            else:
                st.session_state.analysis_results = analysis_results
                st.session_state.error_message = None
                
        except Exception as e:
            error_message = f"Error processing transcript: {str(e)}"
            st.session_state.error_message = error_message

elif selected_example is not None:
    # Load and analyze example transcript
    transcript_path = os.path.join(
        str(APP_DIR / "examples"),
        selected_example
    )
    
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            with st.spinner("🔄 Analyzing example transcript..."):
                analysis_results, error = run_analyzer(
                    transcript_text,
                    selected_example,
                    use_sliding_window
                )
                
            if error:
                error_message = f"Analysis error: {error}"
                st.session_state.error_message = error_message
            else:
                st.session_state.analysis_results = analysis_results
                st.session_state.error_message = None
                
        except Exception as e:
            error_message = f"Error loading example: {str(e)}"
            st.session_state.error_message = error_message

# Use session state if no new input
if analysis_results is None:
    analysis_results = st.session_state.analysis_results
if error_message is None:
    error_message = st.session_state.error_message

# Display error if any
if error_message:
    st.error(error_message)

# Display results if available
if analysis_results:
    # Header with metadata
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📄 {analysis_results.get('transcript_id', 'Unknown Transcript')}")
    
    with col2:
        st.metric("Total Utterances", analysis_results.get('total_utterances', 0))
    
    with col3:
        st.metric("Clinician Utterances", analysis_results.get('clinician_utterances', 0))
    
    st.divider()
    
    # Global Dimensions Section
    st.header("📊 Global Dimensions Scores")
    
    # Display scores as metrics
    global_scores = analysis_results.get('global_dimensions', {})
    cols = st.columns(4)
    
    for idx, dimension in enumerate(GLOBAL_DIMENSIONS):
        score = global_scores.get(dimension, 0)
        with cols[idx]:
            st.metric(
                dimension,
                f"{score:.2f}/5.0",
                delta=None,
                help=analysis_results.get('interpretations', {}).get(dimension, "")
            )
            
            # Add color indicator
            color = get_score_color(score)
            st.markdown(
                f'<div style="background-color: {color}; height: 10px; border-radius: 5px;"></div>',
                unsafe_allow_html=True
            )
    
    # Radar chart
    if global_scores:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Global Dimensions Profile")
            fig = create_radar_chart(global_scores)
            st.pyplot(fig)
        
        with col2:
            if show_interpretations:
                st.subheader("MITI Interpretations")
                interpretations = analysis_results.get('interpretations', {})
                for dimension in GLOBAL_DIMENSIONS:
                    if dimension in interpretations:
                        with st.expander(f"{dimension} ({global_scores.get(dimension, 0):.2f}/5.0)"):
                            st.write(interpretations[dimension])
    
    st.divider()
    
    # Behavior Analysis Section
    st.header("🎯 Behavior Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Behavior Distribution")
        st.caption("*Includes both primary and secondary behaviors*")
        behavior_counts = analysis_results.get('behavior_counts', {})
        
        if behavior_counts:
            # Create bar chart
            behavior_df = pd.DataFrame(
                [(label, behavior_counts.get(label, 0)) for label in MI_BEHAVIOR_LABELS],
                columns=['Behavior', 'Count']
            )
            behavior_df = behavior_df[behavior_df['Count'] > 0]
            
            if not behavior_df.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = [BEHAVIOR_COLORS.get(b, '#888888') for b in behavior_df['Behavior']]
                bars = ax.bar(behavior_df['Behavior'], behavior_df['Count'], color=colors)
                ax.set_xlabel('Behavior Code', fontsize=14)  # Reduced from 16
                ax.set_ylabel('Count', fontsize=14)  # Reduced from 16
                ax.set_title('MI Behavior Distribution', fontsize=16)  # Reduced from 18
                ax.tick_params(axis='both', labelsize=12)  # Reduced from 14
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=12)  # Reduced from 14
                
                st.pyplot(fig)
            else:
                st.info("No behavior data available")
    
    with col2:
        # Positive behaviors and areas for improvement
        st.subheader("✨ Strengths")
        positive_behaviors = analysis_results.get('positive_behaviors', [])
        
        if positive_behaviors:
            for behavior in positive_behaviors[:3]:
                st.success(f"**{behavior['name']}** ({behavior['code']}): {behavior['count']} instances")
        else:
            st.info("No positive behaviors identified")
        
        st.subheader("⚠️ Areas for Improvement")
        areas_for_improvement = analysis_results.get('areas_for_improvement', [])
        
        if areas_for_improvement:
            for behavior in areas_for_improvement:
                st.warning(f"**{behavior['name']}** ({behavior['code']}): {behavior['count']} instances")
        else:
            st.success("No anti-MI behaviors detected!")
    
    st.divider()
    
    # Key Examples Section
    st.header("💬 Key Examples")
    
    tab1, tab2 = st.tabs(["Top Excerpts", "Full Transcript"])
    
    with tab1:
        top_excerpts = analysis_results.get('top_excerpts', [])
        
        if top_excerpts:
            for excerpt in top_excerpts:
                speaker = excerpt.get('speaker', 'Unknown')
                text = excerpt.get('text', '')
                labels = excerpt.get('labels', [])
                note = excerpt.get('note', '')
                
                # Create colored container based on speaker
                container_color = "#f0f8ff" if speaker == "Clinician" else "#fff8f0"
                
                with st.container():
                    st.markdown(
                        f'<div style="background-color: {container_color}; padding: 15px; border-radius: 10px; margin-bottom: 10px;">'
                        f'<strong>{speaker}:</strong> {text}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    if labels or note:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if labels:
                                # Check for secondary behaviors
                                secondary = excerpt.get('secondary_behaviors', [])
                                if secondary:
                                    secondary_text = ", ".join([s['behavior'] for s in secondary])
                                    st.caption(f"**Primary:** {', '.join(labels)}")
                                    st.caption(f"**Secondary:** {secondary_text}")
                                else:
                                    st.caption(f"**Behavior:** {', '.join(labels)}")
                        with col2:
                            if note:
                                st.caption(f"**Note:** {note}")
        else:
            st.info("No key excerpts available")
    
    with tab2:
        turns = analysis_results.get('turns', [])
        
        if turns:
            st.subheader("Full Transcript with Behavior Labels")
            
            for turn in turns:
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                labels = turn.get('labels', [])
                
                # Apply color based on speaker and behaviors
                if speaker == "Clinician" and labels:
                    # Get color for first label
                    label_color = BEHAVIOR_COLORS.get(labels[0], "#f0f0f0")
                    # Convert abbreviations to full names
                    full_names = [BEHAVIOR_FULL_NAMES.get(label, label) for label in labels]
                    
                    # Get secondary behaviors if present
                    secondary = turn.get('secondary_behaviors', [])
                    secondary_text = ""
                    if secondary:
                        secondary_names = [BEHAVIOR_FULL_NAMES.get(s['behavior'], s['behavior']) 
                                         for s in secondary]
                        secondary_text = f'&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Secondary: {", ".join(secondary_names)}'
                    
                    st.markdown(
                        f'<div style="background-color: {label_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;">'
                        f'<strong>{speaker}:</strong> {text}'
                        f'<br><small>Primary: {", ".join(full_names)}{secondary_text}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # Default styling for client or unlabeled utterances
                    bg_color = "#fff8f0" if speaker == "Client" else "#f0f8ff"
                    st.markdown(
                        f'<div style="background-color: {bg_color}; padding: 10px; border-radius: 8px; margin-bottom: 10px;">'
                        f'<strong>{speaker}:</strong> {text}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("No transcript data available")
    
    st.divider()
    
    # Export Section
    st.header("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download JSON
        json_str = json.dumps(analysis_results, indent=2)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"{analysis_results.get('transcript_id', 'analysis')}.json",
            mime="application/json"
        )
    
    with col2:
        # Generate and download report
        report_lines = [
            f"# MI Transcript Analysis Report",
            f"## {analysis_results.get('transcript_id', 'Unknown Transcript')}",
            f"",
            f"**Date:** {analysis_results.get('meta', {}).get('date', 'N/A')}",
            f"**Total Utterances:** {analysis_results.get('total_utterances', 0)}",
            f"**Clinician Utterances:** {analysis_results.get('clinician_utterances', 0)}",
            f"",
            f"## Global Dimensions Scores",
            f""
        ]
        
        for dimension in GLOBAL_DIMENSIONS:
            score = global_scores.get(dimension, 0)
            interpretation = analysis_results.get('interpretations', {}).get(dimension, '')
            report_lines.append(f"### {dimension}: {score:.2f}/5.0")
            report_lines.append(f"{interpretation}")
            report_lines.append("")
        
        report_lines.extend([
            f"## Behavior Analysis",
            f"",
            f"### Behavior Distribution"
        ])
        
        if behavior_counts:
            for behavior, count in behavior_counts.items():
                if count > 0:
                    report_lines.append(f"- {behavior}: {count}")
        
        report_lines.append("")
        
        if positive_behaviors:
            report_lines.append("### Strengths")
            for behavior in positive_behaviors[:3]:
                report_lines.append(f"- **{behavior['name']}** ({behavior['code']}): {behavior['count']} instances")
        
        if areas_for_improvement:
            report_lines.append("")
            report_lines.append("### Areas for Improvement")
            for behavior in areas_for_improvement:
                report_lines.append(f"- **{behavior['name']}** ({behavior['code']}): {behavior['count']} instances")
        
        report_text = "\n".join(report_lines)
        
        st.download_button(
            label="⬇️ Download Report (Markdown)",
            data=report_text,
            file_name=f"{analysis_results.get('transcript_id', 'analysis')}_report.md",
            mime="text/markdown"
        )
    
    with col3:
        # Generate CSV for behavior counts
        if behavior_counts:
            behavior_df = pd.DataFrame(
                [(k, v) for k, v in behavior_counts.items()],
                columns=['Behavior', 'Count']
            )
            csv = behavior_df.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Download Behavior Data (CSV)",
                data=csv,
                file_name=f"{analysis_results.get('transcript_id', 'analysis')}_behaviors.csv",
                mime="text/csv"
            )
    
    # Add PDF export button if checkbox is selected
    if export_pdf:
        st.divider()
        st.subheader("📄 PDF Report Generation")
        
        # Custom CSS for dark blue button
        st.markdown("""
        <style>
        div[data-testid="column"]:first-child div.stButton > button {
            background-color: #1e3a8a;
            color: white;
            border: none;
            font-weight: 500;
        }
        div[data-testid="column"]:first-child div.stButton > button:hover {
            background-color: #1e40af;
            color: white;
            border: none;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_data = generate_pdf_report(analysis_results, patient_id)
                        
                        # Prepare filename
                        transcript_id = analysis_results.get('transcript_id', 'analysis')
                        if patient_id:
                            pdf_filename = f"{transcript_id}_patient_{patient_id}_report.pdf"
                        else:
                            pdf_filename = f"{transcript_id}_report.pdf"
                        
                        # Display download button for PDF
                        st.success("✅ PDF report generated successfully!")
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_data,
                            file_name=pdf_filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
        
        with col2:
            st.info(
                "📋 **PDF Contents:**\n"
                "• Summary statistics\n"
                "• Global dimensions scores\n"
                "• Behavior analysis\n"
                "• Key examples\n"
                "• Full transcript (first 100 turns)"
            )

else:
    # No results to display
    st.info(
        "👈 Please upload a transcript or JSON file to begin analysis.\n\n"
        "The dashboard will automatically analyze transcripts and display:\n"
        "- Global dimension scores (Partnership, Empathy, etc.)\n"
        "- Behavior distribution and patterns\n"
        "- Key examples of positive MI and areas for improvement\n"
        "- Full transcript with behavior labels"
    )

# Footer
st.divider()
st.caption(
    "Built with Streamlit | MI Analysis powered by fine-tuned RoBERTa models | "
    "MITI 4.2.1 Coding Manual"
)
