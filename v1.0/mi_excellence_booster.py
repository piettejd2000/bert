#!/usr/bin/env python3
"""
MI Excellence Recognition and Score Adjustment System

Based on MITI 4.2.1 guidelines for recognizing Level 5 (Excellent) MI quality.
Provides post-BERT score adjustments when pattern-based evidence demonstrates
excellent MI technique that may be under-scored by the neural model.
"""

from collections import Counter
import re

def detect_mi_excellence_patterns(behaviors, utterances):
    """
    Detect patterns that indicate excellent MI quality based on MITI guidelines.
    
    Returns dict with:
    - excellence_indicators: specific patterns found
    - dimension_adjustments: recommended score boosts for each dimension
    - confidence: how confident we are in the excellence assessment
    """
    
    # Count MI behaviors
    behavior_counts = Counter(b['behavior'] for b in behaviors)
    total_clinician_utterances = len([b for b in behaviors])
    
    # Get anti-MI behaviors
    anti_mi_count = behavior_counts.get('CO', 0) + behavior_counts.get('PE', 0)
    
    excellence_indicators = {}
    dimension_adjustments = {
        'Partnership': 0,
        'Empathy': 0, 
        'Cultivating Change Talk': 0,
        'Softening Sustain Talk': 0
    }
    
    # === PARTNERSHIP EXCELLENCE INDICATORS ===
    partnership_score = 0
    
    # 1. High ratio of collaborative behaviors
    collaborative_behaviors = behavior_counts.get('SC', 0) + behavior_counts.get('AS', 0)
    if total_clinician_utterances > 0:
        collab_ratio = collaborative_behaviors / total_clinician_utterances
        if collab_ratio >= 0.3:  # 30%+ collaborative
            partnership_score += 0.4
            excellence_indicators['high_collaboration_ratio'] = f"{collab_ratio:.1%}"
    
    # 2. Multiple affirmations showing respect for client autonomy
    affirmations = behavior_counts.get('AF', 0)
    if affirmations >= 2:
        partnership_score += 0.3
        excellence_indicators['multiple_affirmations'] = affirmations
    
    # 3. Open questions that invite client input
    open_questions = behavior_counts.get('OQ', 0)
    if open_questions >= 3:
        partnership_score += 0.3
        excellence_indicators['ample_open_questions'] = open_questions
    
    # 4. Zero directive/confrontational language
    if anti_mi_count == 0:
        partnership_score += 0.5
        excellence_indicators['zero_anti_mi'] = "No confrontation or persuasion"
        
    dimension_adjustments['Partnership'] = min(partnership_score, 1.0)
    
    
    # === EMPATHY EXCELLENCE INDICATORS ===
    empathy_score = 0
    
    # 1. Multiple complex reflections showing deep understanding
    complex_reflections = behavior_counts.get('CR', 0)
    if complex_reflections >= 3:
        empathy_score += 0.5
        excellence_indicators['rich_complex_reflections'] = complex_reflections
    elif complex_reflections >= 2:
        empathy_score += 0.3
        excellence_indicators['good_complex_reflections'] = complex_reflections
    
    # 2. Simple reflections also contribute to empathy
    simple_reflections = behavior_counts.get('SR', 0)
    total_reflections = complex_reflections + simple_reflections
    if total_reflections >= 4:
        empathy_score += 0.3
        excellence_indicators['total_reflections'] = total_reflections
        
    # 3. Check for empathic language patterns in text
    empathic_patterns = count_empathic_language_patterns(behaviors)
    if empathic_patterns >= 3:
        empathy_score += 0.4
        excellence_indicators['empathic_language_patterns'] = empathic_patterns
    
    dimension_adjustments['Empathy'] = min(empathy_score, 1.0)
    
    
    # === CULTIVATING CHANGE TALK EXCELLENCE ===
    change_talk_score = 0
    
    # 1. Strategic open questions that evoke change talk
    evocative_patterns = count_evocative_question_patterns(behaviors)
    if evocative_patterns >= 3:
        change_talk_score += 0.5
        excellence_indicators['evocative_questions'] = evocative_patterns
    elif evocative_patterns >= 2:
        change_talk_score += 0.3
        
    # 2. Multiple complex reflections (often amplify change talk)
    if complex_reflections >= 3:
        change_talk_score += 0.3
        
    # 3. Strategic affirmations that reinforce change talk
    if affirmations >= 2:
        change_talk_score += 0.2
        
    # 4. Values exploration patterns
    values_patterns = count_values_exploration_patterns(behaviors)
    if values_patterns >= 2:
        change_talk_score += 0.4
        excellence_indicators['values_exploration'] = values_patterns
    
    dimension_adjustments['Cultivating Change Talk'] = min(change_talk_score, 1.0)
    
    
    # === SOFTENING SUSTAIN TALK EXCELLENCE ===
    sustain_talk_score = 0
    
    # 1. Zero confrontational responses to resistance
    if anti_mi_count == 0:
        sustain_talk_score += 0.6
        
    # 2. Complex reflections that reframe resistance
    if complex_reflections >= 2:
        sustain_talk_score += 0.3
        
    # 3. Rolling with resistance patterns
    rolling_patterns = count_rolling_with_resistance_patterns(behaviors)
    if rolling_patterns >= 1:
        sustain_talk_score += 0.4
        excellence_indicators['rolling_with_resistance'] = rolling_patterns
    
    dimension_adjustments['Softening Sustain Talk'] = min(sustain_talk_score, 1.0)
    
    
    # === OVERALL EXCELLENCE CONFIDENCE ===
    total_excellence_score = sum(dimension_adjustments.values())
    confidence = min(total_excellence_score / 2.5, 1.0)  # Normalize to 0-1
    
    return {
        'excellence_indicators': excellence_indicators,
        'dimension_adjustments': dimension_adjustments,
        'confidence': confidence,
        'behavior_summary': {
            'total_utterances': total_clinician_utterances,
            'complex_reflections': complex_reflections,
            'open_questions': open_questions,
            'affirmations': affirmations,
            'seeking_collaboration': behavior_counts.get('SC', 0),
            'autonomy_support': behavior_counts.get('AS', 0),
            'anti_mi_behaviors': anti_mi_count
        }
    }

def count_empathic_language_patterns(behaviors):
    """Count patterns that indicate empathic understanding"""
    empathy_patterns = [
        "it sounds like", "i hear", "you're feeling", "that must be",
        "i can see", "it seems like", "you're experiencing", "that brings up",
        "part of you", "on one hand", "at the same time", "you're torn",
        "mixed feelings", "ambivalent", "conflicted", "struggling with"
    ]
    
    count = 0
    for behavior in behaviors:
        text_lower = behavior['text'].lower()
        for pattern in empathy_patterns:
            if pattern in text_lower:
                count += 1
                break  # Only count once per utterance
    return count

def count_evocative_question_patterns(behaviors):
    """Count questions that specifically evoke change talk"""
    evocative_patterns = [
        "what would be different", "how would", "what concerns you about",
        "what worries you", "what are the good things about", "what do you like about",
        "on a scale", "how important", "what would have to change",
        "what might be some reasons", "what else", "what makes this important",
        "how confident", "what would it take", "what's the next step",
        "what are your thoughts about", "how do you see yourself"
    ]
    
    count = 0
    for behavior in behaviors:
        if behavior['behavior'] == 'OQ':  # Only count open questions
            text_lower = behavior['text'].lower()
            for pattern in evocative_patterns:
                if pattern in text_lower:
                    count += 1
                    break
    return count

def count_values_exploration_patterns(behaviors):
    """Count statements/questions that explore client values"""
    values_patterns = [
        "important to you", "what matters to you", "what do you value",
        "what's meaningful", "what you care about", "your priorities",
        "what drives you", "what motivates you", "your goals", "your dreams",
        "what you hope for", "vision for", "what you want"
    ]
    
    count = 0
    for behavior in behaviors:
        text_lower = behavior['text'].lower()
        for pattern in values_patterns:
            if pattern in text_lower:
                count += 1
                break
    return count

def count_rolling_with_resistance_patterns(behaviors):
    """Count instances of rolling with resistance rather than opposing it"""
    rolling_patterns = [
        "you're right", "that makes sense", "i can understand that",
        "that's a fair point", "you know yourself best", "you're the expert",
        "that's one way to look at it", "you have good reasons", "that's understandable",
        "many people feel that way", "that's a common concern"
    ]
    
    count = 0
    for behavior in behaviors:
        text_lower = behavior['text'].lower()
        for pattern in rolling_patterns:
            if pattern in text_lower:
                count += 1
                break
    return count

def apply_excellence_adjustments(original_scores, excellence_analysis, min_boost_threshold=0.4):
    """
    Apply excellence-based score adjustments to global dimension scores.
    
    Args:
        original_scores: dict with dimension names -> scores (1-5 scale)
        excellence_analysis: output from detect_mi_excellence_patterns()
        min_boost_threshold: minimum confidence required to apply boosts
    
    Returns:
        dict with adjusted scores and explanation
    """
    
    adjusted_scores = original_scores.copy()
    adjustments_applied = {}
    
    if excellence_analysis['confidence'] < min_boost_threshold:
        return {
            'adjusted_scores': adjusted_scores,
            'adjustments_applied': {},
            'reason': f"Confidence {excellence_analysis['confidence']:.2f} below threshold {min_boost_threshold}"
        }
    
    for dimension, boost in excellence_analysis['dimension_adjustments'].items():
        if boost > 0:
            original_score = original_scores[dimension]
            # Apply boost but cap at 5.0
            new_score = min(original_score + boost, 5.0)
            
            # Only apply if it makes a meaningful difference
            if new_score > original_score + 0.1:
                adjusted_scores[dimension] = new_score
                adjustments_applied[dimension] = {
                    'original': original_score,
                    'boost': boost,
                    'final': new_score,
                    'reason': f"Excellence patterns detected with confidence {excellence_analysis['confidence']:.2f}"
                }
    
    return {
        'adjusted_scores': adjusted_scores,
        'adjustments_applied': adjustments_applied,
        'excellence_indicators': excellence_analysis['excellence_indicators'],
        'behavior_summary': excellence_analysis['behavior_summary'],
        'confidence': excellence_analysis['confidence']
    }