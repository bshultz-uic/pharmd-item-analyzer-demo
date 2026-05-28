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
    """Reads the uploaded ExamSoft CSV, filtering for MC items, and returns (df, error_message)."""
    try:
        # Rewind file pointer for Streamlit
        file.seek(0)
        
        # Handle ExamSoft encoding
        try:
            df = pd.read_csv(file, skiprows=3, on_bad_lines='skip', encoding='utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, skiprows=3, on_bad_lines='skip', encoding='cp1252')
            
        if df.empty:
            return None, "File is empty after skipping the first 3 rows."

        # Strip whitespace and deduplicate (matches Colab logic)
        seen = {}
        new_cols = []
        for col in df.columns:
            col_str = str(col).strip() # Strip added to prevent ExamSoft spacing bugs
            if col_str not in seen:
                seen[col_str] = 1
                new_cols.append(col_str)
            else:
                seen[col_str] += 1
                new_cols.append(f"{col_str}.{seen[col_str]}")
        df.columns = new_cols
        
        # Filter for Multiple Choice (MC)
        if 'Question Type #' in df.columns:
            df['Question Type #'] = df['Question Type #'].astype(str).str.strip()
            df = df[df['Question Type #'].str.upper() == 'MC']
            if df.empty:
                return None, "No Multiple Choice (MC) questions found after filtering."
                
        # Check for core columns needed for the visual dashboard
        required = ['Item Text', 'Diff(p)', 'Point Biserial']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Missing columns: {', '.join(missing)}. (Are there exactly 3 metadata rows at the top of this file?)"
            
        return df, "Success"
    except Exception as e:
        return None, f"Python Error: {str(e)}"


def generate_analysis_pie_chart(good_count, bad_count, good_label, bad_label, title=""):
    """Generates a pie chart for a single flaw category. Fixed figsize ensures uniform rendering."""
    fig, ax = plt.subplots(figsize=(3, 3))
    colors = ['#4CAF50', '#F44336']
    ax.pie(
        [good_count, bad_count],
        labels=[good_label, bad_label],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white'},
    )
    ax.axis('equal')
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    return fig


def categorize_item_psychometrics(difficulty, point_biserial):
    """Assigns each item to a psychometric difficulty band matching the pipeline report."""
    try:
        diff = float(difficulty)
        pb = float(point_biserial)
    except (ValueError, TypeError):
        return "Uncategorized"
    if diff >= 0.939:
        return "Very Easy"
    elif diff >= 0.869:
        return "Moderately Easy" if pb >= 0.145 else "Uncategorized"
    elif diff >= 0.769:
        return "Target Zone" if pb >= 0.195 else "Uncategorized"
    elif diff >= 0.639:
        return "Challenging" if pb >= 0.145 else "Uncategorized"
    else:
        return "Most Challenging"


def generate_difficulty_distribution_chart(df, course_label="Your Course"):
    """
    Horizontal stacked-bar chart matching the pipeline report's psychometric category display.
    Top bar = course distribution, bottom bar = fixed benchmark (5/20/50/20/5).
    Legend below shows category name, difficulty range, and PBS threshold.
    """
    from matplotlib.patches import Rectangle

    categories = ["Most Challenging", "Challenging", "Target Zone", "Moderately Easy", "Very Easy"]
    color_map = {
        "Most Challenging": '#FF800E',
        "Challenging":      '#FFBC79',
        "Target Zone":      '#006BA4',
        "Moderately Easy":  '#5F9ED1',
        "Very Easy":        '#A2C8EC',
        "Uncategorized":    '#C8D0D9',
    }
    benchmark = [5, 20, 50, 20, 5]

    psych = df.apply(
        lambda r: categorize_item_psychometrics(r.get('Diff(p)'), r.get('Point Biserial')),
        axis=1,
    )
    counts = psych.value_counts()
    n = max(len(df), 1)

    unc_pct  = counts.get("Uncategorized", 0) / n * 100
    raw      = [counts.get(c, 0) / n * 100 for c in categories]
    total_all = sum(raw) + unc_pct
    norm     = 100.0 / total_all if total_all > 0 else 1
    course_vals = [v * norm for v in raw]
    unc_display = unc_pct * norm

    fig, ax = plt.subplots(figsize=(10, 4.5))
    plt.subplots_adjust(top=0.88, bottom=0.34, left=0.13, right=0.97)

    y_pos = [1, 0]
    bh = 0.5
    colors_list = [color_map[c] for c in categories]

    # Course bar
    left = 0
    if unc_display > 0:
        ax.barh(y_pos[0], unc_display, left=left, height=bh,
                color=color_map["Uncategorized"], edgecolor='white', linewidth=1)
        if unc_display >= 3:
            ax.text(left + unc_display / 2, y_pos[0], f"{unc_display:.0f}%",
                    ha='center', va='center', fontsize=11, weight='bold', color='black')
        left += unc_display
    for cat, val in zip(categories, course_vals):
        if val > 0:
            ax.barh(y_pos[0], val, left=left, height=bh,
                    color=color_map[cat], edgecolor='white', linewidth=1)
            if val >= 3:
                tc = 'white' if cat in ("Target Zone", "Very Easy") else 'black'
                ax.text(left + val / 2, y_pos[0], f"{val:.0f}%",
                        ha='center', va='center', fontsize=11, weight='bold', color=tc)
            left += val

    # Benchmark bar
    left = 0
    for cat, val in zip(categories, benchmark):
        ax.barh(y_pos[1], val, left=left, height=bh,
                color=color_map[cat], edgecolor='white', linewidth=1)
        tc = 'white' if cat in ("Target Zone", "Very Easy") else 'black'
        ax.text(left + val / 2, y_pos[1], f"{val}%",
                ha='center', va='center', fontsize=11, weight='bold', color=tc)
        left += val

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 1.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([course_label, 'Benchmark'], fontsize=13, weight='bold')
    ax.set_xlabel('Percentage of Items (%)', fontsize=12, weight='bold')
    ax.set_xticks(range(0, 101, 10))
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.text(0.55, 0.97, "Items by Psychometric Category",
             ha='center', fontsize=15, weight='bold')
    fig.text(0.55, 0.92,
             "Item difficulty categories calibrated to program-level quintiles.",
             ha='center', fontsize=10, style='italic')

    legend_cats = ["Uncategorized", "Most Challenging", "Challenging",
                   "Target Zone", "Moderately Easy", "Very Easy"]
    legend_details = {
        "Uncategorized":    ("Uncategorized",   "Did not meet criteria", ""),
        "Most Challenging": ("Most Challenging", "Diff: <0.64",          "PBS: Any"),
        "Challenging":      ("Challenging",      "Diff: 0.64–0.76", "PBS: >0.15"),
        "Target Zone":      ("Target Zone",      "Diff: 0.77–0.86", "PBS: >0.20"),
        "Moderately Easy":  ("Moderately Easy",  "Diff: 0.87–0.93", "PBS: >0.15"),
        "Very Easy":        ("Very Easy",        "Diff: >0.94",          "PBS: Any"),
    }
    x0, bw = 0.05, 0.155
    ly = 0.09
    for i, cat in enumerate(legend_cats):
        x = x0 + i * bw
        rect = Rectangle((x, ly + 0.045), bw * 0.88, 0.025,
                          facecolor=color_map[cat], edgecolor='white',
                          linewidth=1, transform=fig.transFigure)
        fig.add_artist(rect)
        name, line2, line3 = legend_details[cat]
        fig.text(x + bw * 0.44, ly + 0.025, name,
                 ha='center', va='top', fontsize=9, weight='bold',
                 transform=fig.transFigure)
        fig.text(x + bw * 0.44, ly - 0.005, line2,
                 ha='center', va='top', fontsize=8,
                 transform=fig.transFigure)
        if line3:
            fig.text(x + bw * 0.44, ly - 0.035, line3,
                     ha='center', va='top', fontsize=8,
                     transform=fig.transFigure)
    return fig