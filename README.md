# PharmD Item Analyzer - Faculty Development Tool

## Project Overview
This tool automates the detection of psychometric flaws in multiple-choice exam items. It analyzes both **Individual Items** (text input) and **Full Exam Reports** (CSV uploads from ExamSoft).

## Architecture
The project is built in Python using Streamlit for the interface.
* **`utils.py`**: The "Core Engine." Contains all business logic, Regex patterns, psychometric rules, and chart generation functions. **Integrate this logic into your backend.**
* **`app.py`**: The "Reference UI." A functional prototype built in Streamlit. Use this to understand the intended user flow, session state management, and visualization needs.
* **Models (`.joblib`)**: Pre-trained Machine Learning models required for the "Unfocused Stem" detection.

## Dependencies
Ensure these libraries are installed via `pip install -r requirements.txt`:
* `streamlit`
* `pandas`
* `numpy`
* `scikit-learn`
* `joblib`
* `matplotlib` (New: Required for generating benchmark and flaw charts)

## Key Features & Logic
1.  **Single Item Check:**
    * Accepts Question Stem & Distractors.
    * Runs Regex checks: Negative Phrasing, Blank Placement.
    * Runs ML checks: Unfocused Stem (Hybrid Rule + Model approach).
    * Runs Distractor checks: Parallelism (Length), All/None of the Above, K-Type.

2.  **Full Exam Analysis:**
    * Parses ExamSoft CSVs.
    * **Psychometric Benchmarking:** Compares exam difficulty distribution against a fixed program benchmark (Visualized via Bar Chart).
    * **Flaw Detection:** automatically flags items for:
        * Unfocused Stems
        * Negative Phrasing
        * Bad Blank Placement
        * Non-Functioning Distractors (<5% selection)
        * K-Type / Multiple Answer Cues (in Stem OR Options)
    * **Dynamic Visuals:** Generates Pie Charts only for flaws present in the uploaded data.

## Setup Instructions (Local Testing)
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the prototype: `streamlit run app.py`

## Security & Privacy Notes
* The current prototype processes data in memory using Streamlit Session State.
* For institutional deployment, ensure uploaded ExamSoft files (which may contain student performance data) are handled according to FERPA guidelines.