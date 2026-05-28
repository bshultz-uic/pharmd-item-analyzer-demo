import streamlit as st
import pandas as pd
import joblib
from utils import (
    check_blank_placement_regex, 
    check_negative_phrasing_regex, 
    check_parallel_choices,
    apply_hybrid_unfocused_detection,
    check_non_functioning_distractors,
    check_k_type_v8,
    process_uploaded_file,
    generate_psychometric_chart,
    generate_analysis_pie_chart,
    ANSWER_TEXT_COLS
)

st.set_page_config(page_title="PharmD Item Analyzer", layout="wide")
st.title("PharmD Item Analysis Portal")
st.markdown("Use the tabs below to check a single question or analyze a full exam.")

# --- Session State Initialization ---
if 'exam_data' not in st.session_state:
    st.session_state['exam_data'] = None

# --- Load Models ---
@st.cache_resource
def load_models():
    try:
        rf = joblib.load('unfocused_rf_model.joblib')
        vec = joblib.load('unfocused_vectorizer.joblib')
        enc = joblib.load('unfocused_label_encoder.joblib')
        return rf, vec, enc
    except FileNotFoundError:
        return None, None, None

rf_model, vectorizer, label_encoder = load_models()

tab1, tab2 = st.tabs(["Single Item Check", "Full Exam Upload"])

# ==========================================
# MODE 1: Single Item Check
# ==========================================
with tab1:
    st.header("Draft a New Question")
    
    col1, col2 = st.columns(2)
    with col1:
        stem_input = st.text_area("Question Stem", height=150, placeholder="Type your question here...")
        distractors_raw = st.text_area("Answer Choices (One per line)", height=150, placeholder="Option A\nOption B\n...")
        analyze_btn = st.button("Check for Flaws", key="single_check_btn")

    with col2:
        if analyze_btn and stem_input:
            st.subheader("Results")
            flaws_found = False 

            # 1. Unfocused Stem
            single_df = pd.DataFrame([{'Item Text': stem_input}])
            labels, _ = apply_hybrid_unfocused_detection(single_df, rf_model, vectorizer, label_encoder)
            final_unfocused = labels[0]
            
            if final_unfocused == "Unfocused":
                st.error("❌ Stem Focus: Unfocused")
                flaws_found = True
            else:
                st.success("✅ Stem Focus: Focused")

            # 2. Negative Phrasing
            neg_res = check_negative_phrasing_regex(stem_input)
            if neg_res == "Negative":
                st.error("❌ Negative Phrasing: Detected")
                flaws_found = True
            else:
                st.success("✅ Negative Phrasing: None Detected")

            # 3. Blank Placement
            blank_res = check_blank_placement_regex(stem_input)
            if blank_res == "Bad Placement":
                st.error(f"❌ Blank Placement: Bad Placement")
                flaws_found = True
            elif blank_res == "Good Placement":
                st.success("✅ Blank Placement: Good")
            else:
                st.info("ℹ️ Blank Placement: N/A (Direct question)")

            # 4. Distractors
            distractors_list = [d for d in distractors_raw.split('\n') if d.strip()]
            
            if len(distractors_list) > 1:
                # Create a mock row to utilize the new pandas-based utils functions
                mock_row = {'Item Text': stem_input}
                for i, opt in enumerate(distractors_list):
                    mock_row[f"{chr(65+i)} - Text"] = opt
                mock_series = pd.Series(mock_row)

                # K-Type / AOTA / Multi Check
                k_res = check_k_type_v8(mock_series)
                
                if "All/None of the Above" in k_res:
                    st.error("❌ Distractors: 'All/None of the Above' detected")
                    flaws_found = True
                else:
                    st.success("✅ Distractors: No 'All/None of the Above' cues")

                if "Multiple Correct Answers" in k_res:
                    st.error("❌ Distractors: Multiple Correct Answer cues detected")
                    flaws_found = True
                else:
                    st.success("✅ Distractors: No Multiple Correct cues")

                if "K Type" in k_res:
                    st.error("❌ Distractors: K-Type (Roman Numerals) detected")
                    flaws_found = True
                else:
                    st.success("✅ Distractors: No K-Type Formatting")

                # Parallel Check
                parallel_res = check_parallel_choices(mock_series)
                if parallel_res == "Flagged":
                    st.error("❌ Distractors: Not Parallel (Length)")
                    flaws_found = True
                else:
                    st.success("✅ Distractors: Parallel (Length)")

            else:
                st.info("ℹ️ Enter at least 2 answer choices to check distractors for All/None, Multi-Correct, and K-Type.")

            if not flaws_found:
                st.balloons()
                st.success("🎉 No writing flaws detected! Great job.")

# ==========================================
# MODE 2: Full Exam Analysis
# ==========================================
with tab2:
    st.header("Analyze ExamSoft Report")
    
    # File Uploader
    uploaded_files = st.file_uploader("Upload CSV Reports", accept_multiple_files=True, type=['csv'])
    
    # PROCESS FILES ON BUTTON CLICK
    if uploaded_files and st.button("Run Analysis", key="full_exam_btn"):
        all_data = []
        for file in uploaded_files:
            df = process_uploaded_file(file)
            if df is not None:
                df['Source_File'] = file.name
                all_data.append(df)
            else:
                st.error(f"❌ Failed to process '{file.name}'. Ensure it is an ExamSoft CSV with Multiple Choice (MC) questions.")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # --- RUN PIPELINE ---
            
            # 1. Unfocused Stem (New Hybrid Pipeline)
            labels, scores = apply_hybrid_unfocused_detection(combined_df, rf_model, vectorizer, label_encoder)
            combined_df['Unfocused_Check'] = labels

            # 2. Other Checks
            combined_df['Negative_Check'] = combined_df['Item Text'].fillna('').apply(check_negative_phrasing_regex)
            combined_df['Blank_Check'] = combined_df['Item Text'].fillna('').apply(check_blank_placement_regex)
            
            # 3. K-Type Logic (New v8 Combined Function)
            k_results = combined_df.apply(check_k_type_v8, axis=1)
            combined_df['AOTA_Check'] = k_results.apply(lambda x: 'Flagged' if 'All/None of the Above' in x else 'OK')
            combined_df['Multi_Correct_Check'] = k_results.apply(lambda x: 'Flagged' if 'Multiple Correct Answers' in x else 'OK')
            combined_df['K_Type_Check'] = k_results.apply(lambda x: 'Flagged' if 'K Type' in x else 'OK')

            # 4. Distractors and Parallelism
            combined_df['NFD_Check'] = combined_df.apply(check_non_functioning_distractors, axis=1)
            combined_df['Parallel_Check'] = combined_df.apply(check_parallel_choices, axis=1)

            # SAVE TO SESSION STATE
            st.session_state['exam_data'] = combined_df

    # --- DISPLAY RESULTS (FROM STATE) ---
    if st.session_state['exam_data'] is not None:
        combined_df = st.session_state['exam_data']
        total_items = len(combined_df)
        
        st.success(f"Loaded {total_items} items.")

        # 1. Psychometric Chart
        st.divider()
        st.subheader("1. Psychometric Distribution")
        fig_bar = generate_psychometric_chart(combined_df)
        if fig_bar: st.pyplot(fig_bar)

        # 2. Dynamic Pie Charts
        st.subheader("2. Flaw Analysis Breakdown")
        
        charts_to_show = []
        
        # Unfocused
        u_fail = len(combined_df[combined_df['Unfocused_Check'] == 'Unfocused'])
        if u_fail > 0:
            charts_to_show.append(generate_analysis_pie_chart(total_items - u_fail, u_fail, "Focused", "Unfocused"))
            
        # Negative
        n_fail = len(combined_df[combined_df['Negative_Check'] == 'Negative'])
        if n_fail > 0:
            charts_to_show.append(generate_analysis_pie_chart(total_items - n_fail, n_fail, "Positive", "Negative"))
            
        # K-Type (Combined Logic for Visuals)
        k_fail = len(combined_df[
            (combined_df['K_Type_Check'] == 'Flagged') | 
            (combined_df['AOTA_Check'] == 'Flagged') | 
            (combined_df['Multi_Correct_Check'] == 'Flagged')
        ])
        if k_fail > 0:
            charts_to_show.append(generate_analysis_pie_chart(total_items - k_fail, k_fail, "Standard", "K-Type/Multi"))

        # NFDs
        d_fail = len(combined_df[combined_df['NFD_Check'] == 'Flagged'])
        if d_fail > 0:
            charts_to_show.append(generate_analysis_pie_chart(total_items - d_fail, d_fail, "Good", "NFDs"))

        # Parallelism
        p_fail = len(combined_df[combined_df['Parallel_Check'] == 'Flagged'])
        if p_fail > 0:
            charts_to_show.append(generate_analysis_pie_chart(total_items - p_fail, p_fail, "Parallel", "Non-Parallel"))

        # Display charts dynamically
        if charts_to_show:
            cols = st.columns(len(charts_to_show))
            for i, fig in enumerate(charts_to_show):
                if fig: cols[i].pyplot(fig)
        else:
            st.info("No major flaws detected across the exam!")

        # 3. Metrics & Table
        st.divider()
        st.subheader("3. Detailed Flag Counts")
        
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1: st.metric("Unfocused", u_fail, delta_color="inverse")
        with c2: st.metric("Negative", n_fail, delta_color="inverse")
        with c3: st.metric("Bad Blanks", len(combined_df[combined_df['Blank_Check'] == 'Bad Placement']), delta_color="inverse")
        with c4: st.metric("All/None", len(combined_df[combined_df['AOTA_Check'] == 'Flagged']), delta_color="inverse")
        with c5: st.metric("Multi-Ans", len(combined_df[combined_df['Multi_Correct_Check'] == 'Flagged']), delta_color="inverse")
        with c6: st.metric("K-Type", len(combined_df[combined_df['K_Type_Check'] == 'Flagged']), delta_color="inverse")
        with c7: st.metric("Distractors", d_fail, delta_color="inverse")

        st.divider()
        st.subheader("Review Flagged Items")
        
        filter_option = st.selectbox("Show items flagged for:", 
            ["All Flaws", "Unfocused Stem", "Negative Phrasing", "Bad Blank Placement", 
             "All/None of Above", "Multiple Correct Answers", "K-Type (Roman)",
             "Poor Distractors", "Non-Parallel Choices"])
        
        view_df = combined_df.copy()
        if filter_option == "Unfocused Stem": view_df = view_df[view_df['Unfocused_Check'] == 'Unfocused']
        elif filter_option == "Negative Phrasing": view_df = view_df[view_df['Negative_Check'] == 'Negative']
        elif filter_option == "Bad Blank Placement": view_df = view_df[view_df['Blank_Check'] == 'Bad Placement']
        elif filter_option == "All/None of Above": view_df = view_df[view_df['AOTA_Check'] == 'Flagged']
        elif filter_option == "Multiple Correct Answers": view_df = view_df[view_df['Multi_Correct_Check'] == 'Flagged']
        elif filter_option == "K-Type (Roman)": view_df = view_df[view_df['K_Type_Check'] == 'Flagged']
        elif filter_option == "Poor Distractors": view_df = view_df[view_df['NFD_Check'] == 'Flagged']
        elif filter_option == "Non-Parallel Choices": view_df = view_df[view_df['Parallel_Check'] == 'Flagged']
        elif filter_option == "All Flaws":
            view_df = view_df[
                (view_df['Unfocused_Check'] == 'Unfocused') | (view_df['Negative_Check'] == 'Negative') |
                (view_df['Blank_Check'] == 'Bad Placement') | (view_df['AOTA_Check'] == 'Flagged') |
                (view_df['Multi_Correct_Check'] == 'Flagged') | (view_df['K_Type_Check'] == 'Flagged') |
                (view_df['NFD_Check'] == 'Flagged') | (view_df['Parallel_Check'] == 'Flagged')
            ]
        
        if view_df.empty:
            st.info("No items found matching this filter!")
        else:
            display_cols = ['Item Text', 'Unfocused_Check', 'Negative_Check', 'Blank_Check', 
                           'AOTA_Check', 'Multi_Correct_Check', 'K_Type_Check', 
                           'NFD_Check', 'Parallel_Check', 'Diff(p)', 'Point Biserial']
            final_cols = [c for c in display_cols if c in view_df.columns]
            st.dataframe(view_df[final_cols], use_container_width=True)