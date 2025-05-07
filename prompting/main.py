from sys import path
path.append('.')

import streamlit as st
from config import LLM_CONFIG as llm_config
from optimizers.pipeline import OptimizationPipeline

# Streamlit page config
st.set_page_config(page_title="Code Optimizer Chatbot", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "current_code_input" not in st.session_state:
    st.session_state.current_code_input = ""

# Function to run optimization
def run_optimization(code):
    pipeline = OptimizationPipeline(llm_config)
    try:
        results = pipeline.optimize(code)
        return results
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        return None

# Title and description
st.title("Code Optimizer Chatbot")
st.markdown("Paste your code, and I'll optimize it for you! Get optimized code, performance stats, and insights.")

# Input form
with st.form(key="code_input_form"):
    code_input = st.text_area("Enter your code here:", height=200, placeholder="e.g., def calculate_pairs(nums): ...")
    submit_button = st.form_submit_button("Optimize Code")

    if submit_button and code_input:
        st.session_state.messages.append({"role": "user", "content": f"```python\n{code_input}\n```"})
        st.session_state.is_processing = True
        st.session_state.current_code_input = code_input

# Run optimization outside form
if st.session_state.is_processing and st.session_state.current_code_input:
    code_to_process = st.session_state.current_code_input
    results = run_optimization(code_to_process)

    if results:
        optimized_code = f"### Optimized Code\n```python\n{results['optimized_code']}\n```"
        benchmark = results['benchmark']
        benchmark_text = (
            f"### Benchmark Improvements\n"
            f"- **Time Improvement**: {benchmark['time_improvement']}%\n"
            f"- **Memory Change**: {benchmark['memory_change']}%\n"
            f"- **Original Time**: {benchmark['original_time']}s\n"
            f"- **Optimized Time**: {benchmark['optimized_time']}s\n"
            f"- **Original Memory**: {benchmark['original_memory']}KB\n"
            f"- **Optimized Memory**: {benchmark['optimized_memory']}KB"
        )
        insights = f"### Insights\n{results['insights']}"

        st.session_state.messages.append({"role": "assistant", "content": optimized_code})
        st.session_state.messages.append({"role": "assistant", "content": benchmark_text})
        st.session_state.messages.append({"role": "assistant", "content": insights})

    # Reset flags
    st.session_state.is_processing = False
    st.session_state.current_code_input = ""

# Display chat history below the input form
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.is_processing:
        with st.chat_message("assistant"):
            st.markdown("⏳ Processing your code, please wait...")
