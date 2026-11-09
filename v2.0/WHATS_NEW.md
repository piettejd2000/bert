# What's New in v2.0

## Overview
Version 2.0 of the MI Transcript Analyzer introduces two major enhancements to improve the accuracy and depth of Motivational Interviewing (MI) behavior analysis:

1. **Multi-behavior Detection**: Captures secondary behaviors when clinician utterances exhibit multiple MI techniques simultaneously
2. **Improved Valediction Handling**: Distinguishes between pure closings and MI-rich concluding statements

## 1. Secondary Behavior Detection

### The Challenge
In real therapeutic conversations, clinicians often employ multiple MI techniques within a single utterance. For example:

> "You're feeling overwhelmed by the situation, and at the same time you're wondering what steps might help you move forward?"

This utterance combines:
- **Complex Reflection (CR)**: Reflecting the client's emotional state
- **Open Question (OQ)**: Exploring potential next steps

Version 1.0 would only capture the dominant behavior, missing this nuanced therapeutic approach.

### The Solution: Conservative Multi-label Classification

The v2.0 analyzer now detects up to 2 secondary behaviors alongside the primary behavior using conservative thresholds:

#### Technical Implementation
```python
# Conservative thresholds for secondary behaviors
SECONDARY_ABS_THRESHOLD = 0.20  # Minimum absolute probability (20%)
SECONDARY_REL_THRESHOLD = 0.35  # Must be at least 35% of primary probability
MAX_SECONDARY = 2               # Maximum number of secondary behaviors
```

#### How It Works
1. **Primary behavior selection** remains unchanged (argmax) - ensuring backward compatibility
2. **Secondary behaviors** are identified when they meet BOTH criteria:
   - Absolute probability ≥ 0.20 (meaningful confidence)
   - Relative probability ≥ 0.35 × primary probability (substantial relative to dominant behavior)
3. **Conservative approach** prevents over-detection while capturing genuine multi-behavior utterances

#### Example Classifications
- **High confidence single behavior** (primary: 0.92)
  - Secondary behaviors unlikely to meet thresholds
  - Result: Single label (as in v1.0)

- **Multi-behavior utterance** (primary: 0.45, secondary: 0.28)
  - Secondary meets both thresholds (0.28 > 0.20 AND 0.28 > 0.45×0.35)
  - Result: Primary + Secondary labels displayed

### Current Approach vs. Sentence-Level Alternative

#### Current Approach: Full-Turn Classification
The v2.0 analyzer classifies entire clinician turns as single units, which has important implications:

**Advantages:**
- **Preserves context**: MI behaviors often require surrounding context for accurate classification
- **Maintains therapeutic flow**: Captures the overall therapeutic intent of the clinician's response
- **Computational efficiency**: Single BERT inference per clinician turn
- **Training alignment**: Model was trained on complete turns, not sentence fragments

**Limitation with High-Confidence Classifications:**
When a clinician turn contains multiple distinct MI behaviors across sentences, the softmax constraint (all probabilities sum to 1.0) can prevent secondary detection:

Example:
> "That takes real courage to share something so personal. [AF behavior]  
> How would you like to move forward from here? [SC behavior]"

If the model assigns 0.97 confidence to AF, only 0.03 remains for all other behaviors combined. The SC component cannot meet the 0.20 absolute threshold, despite being clearly present.

#### Alternative: Sentence-Level Classification
Analyzing each sentence independently would allow multiple behaviors with high confidence but introduces challenges:

**Potential Benefits:**
- Each sentence could independently score high for its behavior
- No mathematical constraint preventing multiple high-confidence behaviors
- More granular behavior detection

**Significant Downsides:**
1. **Context loss**: "So you're feeling torn" alone might be Simple Reflection, but in context with following sentences, it's Complex Reflection
2. **Fragmented behaviors**: Affirmations often span multiple sentences that work together
3. **Computational cost**: 5x processing time for a 5-sentence turn
4. **Boundary ambiguity**: Conversational speech lacks clear sentence boundaries
5. **Training mismatch**: Model trained on full turns may perform poorly on fragments
6. **Aggregation complexity**: How to meaningfully combine 5 different sentence-level labels?
7. **False positives**: "Right." or "Okay." might trigger classifications inappropriately

**Decision Rationale:**
The current full-turn approach with conservative secondary thresholds balances accuracy with usability. While it may miss some secondary behaviors in highly confident classifications, it maintains the therapeutic context and intent that are crucial for meaningful MI analysis.

### Dashboard Display Updates

The Streamlit dashboard now shows:
- **Top Excerpts Section**: 
  ```
  Primary: Seeking Collaboration (SC)
  Secondary: Open Question, Affirm
  ```
- **Full Transcript Section**: Both primary and secondary labels on the same line
- **Behavior Distribution**: Includes both primary and secondary behaviors in counts
- **Header**: Updated to "MI Analysis Dashboard v2.0"

## 2. Improved Valediction Handling

### The Challenge
Version 1.0 had two problems with closing statements (valedictions):

1. **Over-aggressive detection**: Any utterance containing phrases like "You're welcome" or "Thank you" was automatically labeled as Simple Reflection (SR), even when it contained rich MI content:
   > "You're very welcome. Your courage in reflecting and choosing your path is meaningful. Keep trusting your inner wisdom—it's guiding you forward. Take care."
   
   This should be Affirm (AF), not SR.

2. **Inappropriate MI labeling**: Pure closings with no therapeutic content received MI behavior labels:
   > "Bye!"
   
   This has no MI content and shouldn't receive an MI label.

### The Solution: Context-Aware Valediction Detection

Version 2.0 implements a nuanced approach:

#### Pure Valedictions (No MI Content)
Simple closings are identified by:
- Very short length (≤8 words)
- Contains only closing phrases like: "bye", "goodbye", "see you next time", "take care"
- No therapeutic content

**Result**: Labeled as **"No significant MI content"** instead of forcing to SR

Examples:
- "Bye!" → "No significant MI content"
- "See you next time." → "No significant MI content"
- "Take care, Jennifer." → "No significant MI content"

#### MI-Rich Valedictions
Longer closing statements with therapeutic content:
- Processed through BERT classifier
- Receive appropriate MI behavior labels based on content
- Not forced to any particular label

Examples:
- "Your courage in reflecting is meaningful..." → AF (Affirm)
- "Thank you for sharing. What would you like to explore next session?" → SC (Seeking Collaboration)

#### Technical Implementation
```python
# Check for PURE valedictions (short, simple closings without MI content)
word_count = len(text_lower.split())
simple_valedictions = [
    "see you next week", "see you next time", "take care", 
    "goodbye", "thanks for coming", "have a good day",
    "have a nice day", "have a great week", "see you soon",
    "talk soon", "until next time", "bye", "farewell"
]

# Only skip BERT for very short utterances with valediction phrases
if word_count <= 8:
    for phrase in simple_valedictions:
        if phrase in text_lower:
            # Label as non-MI content, not SR
            behavior = 'No significant MI content'
            note = 'Pure valediction - no MI therapeutic content'
            break
```

## Benefits of v2.0

1. **More Complete Analysis**: Captures the full therapeutic technique repertoire in multi-faceted utterances
2. **Accurate Valediction Handling**: No longer mislabels pure closings or MI-rich conclusions
3. **Backward Compatible**: Primary behavior selection unchanged - existing workflows remain valid
4. **Conservative Approach**: Avoids over-detection of secondary behaviors through careful thresholds
5. **Enhanced Clinical Insights**: Better represents the complexity of real therapeutic interactions

## Usage

Navigate to the v2.0 folder and run:
```bash
# For JSON output
python3 analyze_global_dimensions_json.py <transcript_file> --out <output.json>

# For sliding window analysis
python3 analyze_global_dimensions_sliding_window.py <transcript_file> --out <output.json>

# For interactive dashboard
streamlit run mi_results_dashboard.py <json_results_file>
```

## Technical Notes

- Both analyzer versions (JSON and sliding window) include these enhancements
- Secondary behavior detection uses the same BERT model - no retraining required
- Thresholds are configurable constants for easy adjustment if needed
- "No significant MI content" label appears in behavior counts but is excluded from MI-specific metrics