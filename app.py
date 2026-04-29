import streamlit as st
import pandas as pd
from main import run_evaluation, get_task, get_dataset

st.set_page_config(page_title="Multimodal Eval Engine", layout="wide")

st.title(" Multimodal Evaluation Engine")
st.markdown("Professional benchmarking of AI pipelines across Text, OCR, and ASR.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Experiment Settings")
    task_name = st.selectbox("Task", ["summarization", "question_answering", "ocr_extraction", "speech_transcription"])
    mode = st.selectbox("Mode", ["benchmark", "production"])
    dataset_name = st.selectbox("Dataset", ["text_prod", "ocr_prod", "asr_prod"])
    
    run_btn = st.button("Run Evaluation", type="primary")

# --- Main Content ---
if run_btn:
    with st.spinner(f"Running {task_name} in {mode} mode..."):
        final_state = run_evaluation(task_name, mode, dataset_name)
        
        if final_state:
            summary = final_state.get("summary", {})
            best_pipeline = final_state.get("best_pipeline")
            decision_reason = final_state.get("decision_reason")
            analysis = final_state.get("analysis", {})

            st.success(f"Benchmark Complete! Best Pipeline: **{best_pipeline}**")

            # Display Results Table
            st.subheader("📊 Aggregated Results")
            df_data = []
            for p_name, m in summary.items():
                df_data.append({
                    "Pipeline": p_name,
                    "Type": m.get("type", "unknown"),
                    "Accuracy": f"{m['avg_accuracy']:.2f}",
                    "Latency": f"{m['avg_latency']:.2f}s",
                    "Cost": f"${m['total_cost']:.4f}",
                    "Failures": m['failure_count'],
                    "Score": f"{m['score']:.2f}"
                })
            
            df = pd.DataFrame(df_data)
            st.table(df)

            # Decision Reason
            st.subheader("💡 Recommendation")
            st.info(decision_reason)

            # Critic Analysis
            with st.expander("📝 Detailed Critic Analysis"):
                for p_name, report in analysis.items():
                    st.markdown(f"**{p_name}**")
                    st.code(report)
        else:
            st.error("Evaluation failed to return results.")
else:
    st.info("Select settings in the sidebar and click 'Run Evaluation' to start.")
