import re
import pandas as pd
import numpy as np

# ==========================================================================
# HYBRID UNFOCUSED STEM LOGIC (Rule-Based + ML)
# Negative phrasing is decoupled from unfocused detection.
# ==========================================================================

def check_unfocused_combined(text):
    """
    Checks for 'Unfocused' phrasing patterns using Regex.
    Returns 'Unfocused' if a rule matches, otherwise returns 'Focused'.
    """
    if not isinstance(text, str):
        return 'Focused'

    # --- PART A: VAGUE OPENERS ---
    if re.search(r'(?i)Which of the following\s+(statements?(\(s\))?\s+)?(is|are|is/are)\s+(true|correct)', text): return 'Unfocused'
    if re.search(r'(?i)Which of the following (is|are|is/are) an examples?', text): return 'Unfocused'
    if re.search(r'(?i)Which of the following correctly\s+\w+', text): return 'Unfocused'
    if re.search(r'(?i)(Choose|Select)\s+(a|the)\s+(true|correct)\s+statement', text): return 'Unfocused'
    if re.search(r'(?i)Which of the following statements?\s+correctly\s+\w+', text): return 'Unfocused'
    if re.search(r'(?i)Which of the following options\s+(is|are)', text): return 'Unfocused'
    if re.search(r'(?i)Which of the following\s+(stroke|diabetes|diabetic|cancer|hypertension|hypertensive|heart failure|asthma|COPD|obesity|obese|epilepsy|epileptic|HIV|hepatitis|renal|CKD|pregnant|pediatric|geriatric|surgical|ICU|trauma|sepsis|pneumonia)\s+patients?', text): return 'Unfocused'
    if re.search(r'(?i)Which of the following statements?\s+(about|regarding|concerning|related to|involving)\b', text): return 'Unfocused'
    if re.search(r'(?i)Which of the (statements|above|previous)\s+(is|are)\s+(true|correct)', text): return 'Unfocused'
    if re.search(r'(?i)(Select|Choose)\s+the\s+best\s+statement', text): return 'Unfocused'
    if re.search(r'(?i)select the statement that (correctly|accurately)\s+\w+', text): return 'Unfocused'
    if re.search(r'(?i)Which of the following statements?\s+best\s+describes', text): return 'Unfocused'
    if re.search(r'(?i)Select the best (CORRECT|correct)\s+statement', text): return 'Unfocused'

    # --- PART B: BARE "WHICH/WHAT IS TRUE/CORRECT" (no "of the following") ---
    if re.search(r'(?i)\b(What|Which)\s+(is|are)\s+(true|correct)\b', text): return 'Unfocused'
    if re.search(r'(?i)\bWhat\s+(is|are)\s+(the\s+)?correct\s+statements?', text): return 'Unfocused'
    if re.search(r'(?i)\bWhich\s+statements?\s+.{0,120}\s+(is|are|is/are)\s+(correct|true)\b', text): return 'Unfocused'
    if re.search(r'(?i)Which\s+statements?\s+best\s+(explains?|describes?|represents?|illustrates?|characterizes?)', text): return 'Unfocused'
    if re.search(r'(?i)\bWhich\s+is\s+(a\s+)?(FALSE|TRUE|false|true)\s+statement', text): return 'Unfocused'
    if re.search(r'(?i)\bWhi[a-z]{1,3}h?\s+is\s+(true|correct)\b', text): return 'Unfocused'

    # --- PART C: END OF STEM TRIGGERS ---
    if re.search(r'(?i)\b(is|are|is/are)\s+true[?:]?\s*$', text): return 'Unfocused'
    if re.search(r'(?i)\b(is|are|is/are)\s+correct[?:]?\s*$', text): return 'Unfocused'
    if re.search(r'(?i)statements?\s+.{0,150}\s+(is|are|is/are)\s+(CORRECT|correct|TRUE|true)[?:]?\s*$', text): return 'Unfocused'
    if re.search(r'(?i)What (is|are)\s+(the\s+)?(correct|true) statements?', text): return 'Unfocused'

    return 'Focused'

def rule_based_label(text):
    """
    Checks ONLY the unfocused regex patterns. Negative phrasing is now handled
    independently and no longer triggers an unfocused label.
    Returns 'Unfocused' if the unfocused regex matches, otherwise None.
    """
    if check_unfocused_combined(text) == 'Unfocused':
        return 'Unfocused'
    return None

def apply_hybrid_unfocused_detection(df, model, vectorizer, label_encoder, stem_col='Item Text', threshold=0.53):
    """
    Applies Rules First -> Then ML Model using probability threshold.
    Returns a tuple of (final_labels, confidence_scores).
    """
    final_labels = []
    confidence_scores = []

    for text in df[stem_col]:
        rule_result = rule_based_label(text)
        if rule_result == 'Unfocused':
            final_labels.append('Unfocused')
            confidence_scores.append(100)
        else:
            if model and vectorizer and label_encoder:
                try:
                    vec_text = vectorizer.transform([text])
                    proba = model.predict_proba(vec_text)[0]
                    if proba[0] >= threshold:
                        final_labels.append('Unfocused')
                        confidence_scores.append(int(round(proba[0] * 100)))
                    else:
                        final_labels.append('Focused')
                        confidence_scores.append(int(round(proba[1] * 100)))
                except Exception as e:
                    final_labels.append('Focused')
                    confidence_scores.append(50)
            else:
                final_labels.append('Focused')
                confidence_scores.append(50)

    return final_labels, confidence_scores

# ==========================================================================
# NEGATIVE PHRASING CHECK
# ==========================================================================

def check_negative_phrasing_regex(text):
    """
    Checks for specific negative phrasing patterns using Regex.
    Returns 'Negative' if a rule matches, otherwise returns 'None'.
    """
    if not isinstance(text, str):
        return 'None'

    # --- 1. Strongest Indicators (Capitalized Emphasis) ---
    if re.search(r'\b(are|is) FALSE\b', text): return 'Negative'
    if re.search(r'\b(are|is) INCORRECT\b', text): return 'Negative'
    if re.search(r'\bINCORRECTLY\b', text): return 'Negative'

    # --- 1a. All-caps NOT, INCORRECT, FALSE ---
    if re.search(r'\bNOT\b', text): return 'Negative'
    if re.search(r'\bINCORRECT\b', text): return 'Negative'
    if re.search(r'\bFALSE\b', text): return 'Negative'

    # --- 2. "Incorrect" Variations ---
    if re.search(r'(?i)incorrect[:?]\s*$', text): return 'Negative'
    if re.search(r'(?i)Which of the following statements?\s+(is|are)\s+incorrect', text): return 'Negative'
    if re.search(r'(?i)Which of the following is (an?|the) incorrect\b', text): return 'Negative'
    if re.search(r'(?i)(Choose|Select)\s+the\s+incorrect\s+(statement|answer)', text): return 'Negative'
    if re.search(r'(?i)Which one is incorrect\?', text): return 'Negative'
    if re.search(r'(?i)Select the incorrect\s+\w+', text): return 'Negative'

    # --- 3. "False" Variations ---
    if re.search(r'(?i)(is|are)\s+false\?\s*$', text): return 'Negative'
    if re.search(r'(?i)Which of the following is an?\s+false statement', text): return 'Negative'
    if re.search(r'(?i)Which of the following\s+(statements?\s+)?(is|are)\s+false', text): return 'Negative'
    if re.search(r'(?i)(choose|select|indicate)\s+(the|a|an)?\s*FALSE\s+statement', text): return 'Negative'
    if re.search(r'(?i)(choose|select|indicate)\s+(the\s+)?best\s+FALSE\s+statement', text): return 'Negative'
    if re.search(r'(?i)(choose|select|indicate)\s+FALSE', text): return 'Negative'
    if re.search(r'(?i)indicate\s+(the\s+)?false\s+statement', text): return 'Negative'
    if re.search(r'(?i)\bWhich\s+is\s+(a\s+)?FALSE\s+statement', text): return 'Negative'

    # --- 4. "Not True/Correct" Variations ---
    if re.search(r'(?i)(is|are)\s+not\s+true[:?]', text): return 'Negative'
    if re.search(r'(?i)(is|are)\s+not\s+correct[:?]', text): return 'Negative'
    if re.search(r'(?i)which of the following is not true', text): return 'Negative'
    if re.search(r'(?i)Which one is not correct\?', text): return 'Negative'

    # --- 5. "Does/Do/Was/Were Not" Variations ---
    if re.search(r'(?i)Which of the following statements?\s+(does|do)\s+not', text): return 'Negative'
    if re.search(r'(?i)Which of the following\s+(does|do)\s+not', text): return 'Negative'
    if re.search(r'(?i)Which of the following\s+(was|were)\s+not', text): return 'Negative'
    if re.search(r'(?i)Which of these are not', text): return 'Negative'
    if re.search(r'(?i)Which one of the following is not', text): return 'Negative'
    if re.search(r'(?i)which of the following\s+(is|are)\s+not', text): return 'Negative'

    # --- 6. "Except" Variations ---
    if re.search(r'(?i)except\?', text): return 'Negative'
    if re.search(r'(?i)are correct,\s*except:', text): return 'Negative'
    if re.search(r'(?i)\bexcept[:?]?\s*$', text): return 'Negative'
    if re.search(r'(?i)\bexcept\s*[_]+', text): return 'Negative'

    # --- 7. Targeted NOT in question phrase ---
    if re.search(r'(?i)\bwhich\b.{0,40}\bNOT\b', text): return 'Negative'
    if re.search(r'(?i)\bwhat\b.{0,40}\bNOT\b', text): return 'Negative'
    if re.search(r'\bNOT\b.{0,20}[?:]\s*$', text): return 'Negative'

    # --- 8. Expanded NOT patterns ---
    if re.search(r'(?i)Which of the following\s+.*?\s+is\s+NOT\s+', text): return 'Negative'
    if re.search(r'(?i)Which of the following\s+.*?\s+(does|do)\s+NOT\s+', text): return 'Negative'
    if re.search(r'(?i)Which\s+.*?\s+is\s+NOT\s+a\s+', text): return 'Negative'
    if re.search(r'(?i)Identify the statement that (does|is) NOT', text): return 'Negative'
    if re.search(r'(?i)Which one .+ has NOT been', text): return 'Negative'

    return 'None'

# ==========================================================================
# REVISED BLANK PLACEMENT CHECK (v3 Logic)
# ==========================================================================
BLANK_REGEX_PATTERN = r'_+'
blank_regex = re.compile(BLANK_REGEX_PATTERN)

def check_blank_placement_regex(text):
    if not isinstance(text, str):
        return "Invalid Input"

    clean_text = text.strip()
    matches = list(blank_regex.finditer(clean_text))

    if len(matches) == 0:
        return "Good Placement"
    if len(matches) > 1:
        return "Bad Placement"

    match = matches[0]
    trailing_text = clean_text[match.end():]

    if re.fullmatch(r'[\s\W]*', trailing_text):
        return "Good Placement"
    else:
        return "Bad Placement"

# ==========================================================================
# REVISED K-TYPE CHECK (v4 Multi-Label Logic)
# ==========================================================================

ALL_NONE_PATTERN_STR = r"""
(
    \b(?:all|none)\s+(?:of\s+)?(?:the\s+)?(?:above|these|them|choices?|answers?(?:\s+choices?)?|options?|statements?)\b |
    \b(?:all|none)\s+(?:are|is|will)\b |
    \b(?:all|none)\s+(?:are|is)\s+(?:correct|true|accurate)\b |
    \b(?:all|none)\s+of\s+the\s+(?:above|these|them)\s+(?:are|is)\s+(?:correct|true|accurate)\b |
    \beach\s+(?:one\s+)?of\s+the\s+(?:above|these|them)\b |
    \bevery\s+(?:one\s+)?of\s+the\s+(?:above|these|them)\b |
    \bnot\s+any\s+of\s+the\s+(?:above|these|them)\b |
    \ball\s+(?:are|is|will|statement|option|listed|choice-based)\b |
    \bnone\s+(?:are|is|will|statement|option)\b
)
"""

LETTER_COMBO_PATTERN_STR = r"""
(
    \b[A-G]\.?\s*(?:&|\+)\s*[A-G]\.?(?:\s*only)?\b |
    (?<!Part\s)\b[A-G]\.?(?:\s*(?:,|/|and)\s*[A-G]\.?){1,4}\s*(?:only)?\b |
    \b(?:both|neither)\s+[A-G]\.?\s*(?:and|&|nor)\s*[A-G]\.?\b
)
"""

ROMAN_COMBO_PATTERN_STR = r"""
(
    \b(?:I|II|III|IV|V|VI)(?:\s*(?:,|/|and|&)\s*(?:I|II|III|IV|V|VI)){1,4}\s*(?:only)?\b |
    \b(?:I|II|III|IV|V|VI)\s*(?:,?\s+and\s+|\s+and\s+)(?:I|II|III|IV|V|VI)\s+(?:are|is)\s+(?:true|false|correct)\b
)
"""

STEM_MULTI_PATTERN_STR = r"""
(
    \b(?:select|choose|pick|identify)\s+(?:two|three|more\s+than\s+one|multiple)\b |
    \b(?:select|choose|identify)\s+all\s+that\s+apply\b |
    \bwhich\s+(?:two|three)\s+of\s+the\s+following\b |
    \bwhat\s+(?:two|three)\s+\(\d\)\s+strategies\b |
    \bwhich\s+of\s+the\s+(?:two|three)\s+following\b
)
"""

rx_all_none = re.compile(ALL_NONE_PATTERN_STR, re.IGNORECASE | re.VERBOSE)
rx_letter_combo = re.compile(LETTER_COMBO_PATTERN_STR, re.IGNORECASE | re.VERBOSE)
rx_roman_combo = re.compile(ROMAN_COMBO_PATTERN_STR, re.IGNORECASE | re.VERBOSE)
rx_stem_multi = re.compile(STEM_MULTI_PATTERN_STR, re.IGNORECASE | re.VERBOSE)

def check_k_type_v8(row, stem_col='Item Text', answer_cols=['A - Text', 'B - Text', 'C - Text', 'D - Text', 'E - Text', 'F - Text', 'G - Text']):
    """
    Checks if an item is a K-Type question.
    SCANS ALL OPTIONS and returns ALL flaws found (comma-separated).
    """
    found_flaws = set()

    stem_text = row.get(stem_col, '')
    if isinstance(stem_text, str) and rx_stem_multi.search(stem_text):
        found_flaws.add("Multiple Correct Answers")

    for col in answer_cols:
        if col not in row.index or pd.isna(row.get(col)):
            continue

        answer_text = str(row.get(col)).strip()

        if rx_all_none.search(answer_text):
            found_flaws.add("All/None of the Above")
        if rx_letter_combo.search(answer_text):
            found_flaws.add("Multiple Correct Answers")
        if rx_roman_combo.search(answer_text):
            found_flaws.add("K Type")

    if not found_flaws:
        return "Not K-Type"

    return ", ".join(sorted(list(found_flaws)))

# ==========================================================================
# NON-FUNCTIONING DISTRACTOR & PARALLEL CHOICES
# ==========================================================================

def check_non_functioning_distractors(row, percent_cols=['A - % Selected', 'B - % Selected', 'C - % Selected', 'D - % Selected', 'E - % Selected', 'F - % Selected', 'G - % Selected'], threshold=5.0, min_count=2):
    count_below_threshold = 0
    valid_cols_checked = 0

    for col in percent_cols:
        if col in row.index and pd.notna(row[col]):
            valid_cols_checked += 1
            try:
                percent_val = float(str(row[col]).replace('%', '').strip())
                if percent_val < threshold:
                    count_below_threshold += 1
            except (ValueError, TypeError):
                continue

    if valid_cols_checked >= 3 and count_below_threshold >= min_count:
        return "Flagged"
    return "OK"

def check_parallel_choices(row, answer_cols=['A - Text', 'B - Text', 'C - Text', 'D - Text', 'E - Text', 'F - Text', 'G - Text'], max_word_diff=10):
    word_counts = []
    for col in answer_cols:
        if col in row.index and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text:
                word_counts.append(len(text.split()))

    valid_options_count = len(word_counts)
    if valid_options_count < 2:
        return "Not Applicable"

    diff = max(word_counts) - min(word_counts)
    return "Flagged" if diff > max_word_diff else "OK"

# ==========================================================================
# FILE PROCESSING & VISUALIZATION (Restored)
# ==========================================================================
import matplotlib.pyplot as plt

ANSWER_TEXT_COLS = ['A - Text', 'B - Text', 'C - Text', 'D - Text', 'E - Text', 'F - Text', 'G - Text']

def process_uploaded_file(file):
    """Reads the uploaded ExamSoft CSV, skipping metadata rows."""
    try:
        # Skip the first 3 rows of ExamSoft metadata to get to the real headers
        df = pd.read_csv(file, skiprows=3, on_bad_lines='skip', encoding='utf-8')
        
        # Deduplicate any duplicate columns to prevent pandas errors
        seen = {}
        new_cols = []
        for col in df.columns:
            col_str = str(col)
            if col_str not in seen:
                seen[col_str] = 1
                new_cols.append(col_str)
            else:
                seen[col_str] += 1
                new_cols.append(f"{col_str}.{seen[col_str]}")
        df.columns = new_cols
        
        return df
    except Exception as e:
        return None