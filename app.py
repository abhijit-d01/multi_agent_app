import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph import create_workflow

# Load environment variables from .env file
load_dotenv()

# Page Config
st.set_page_config(page_title="GenAI Data Analyst", page_icon="📊", layout="wide")

# Custom CSS for "Interesting" UI
st.markdown("""
<style>
    .stChatMessage {border-radius: 10px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    
    /* Metric Card Styling */
    .stMetric {
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #ff4b4b;
    }
    
    /* Force text colors for metrics to ensure visibility on light background (fixes white-on-white in Dark Mode) */
    [data-testid="stMetricLabel"] { color: #000000 !important; }
    [data-testid="stMetricValue"] { color: #000000 !important; }
    [data-testid="stMetricDelta"] { color: #333333 !important; }
    
    div[data-testid="stExpander"] {border: 1px solid #e0e0e0; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Talk to Your Data")
st.markdown("**Multi-Agent System:** Analyst (Python Logic) ➝ Writer (Concise Reporting)")

# Helper to handle weird CSV formats
def load_csv_robust(file):
    errors = []
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=encoding, low_memory=False)
            
            # Clean date columns if they exist
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            return df
        except Exception as e:
            errors.append(f"{encoding}: {str(e)}")
    
    raise ValueError(f"Failed to load CSV with multiple encodings. Errors: {errors}")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Strictly load API Key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY is missing in your .env file.")
        st.info("Please create a .env file with: GOOGLE_API_KEY=your_key_here")
        st.stop()
    
    st.success("✅ API Key loaded")
    
    st.divider()
    st.subheader("📂 Data Sources")
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    loaded_dfs = {}
    selected_dfs = {}

    if uploaded_files:
        st.caption("Select files to analyze:")
        for file in uploaded_files:
            try:
                df = load_csv_robust(file)
                filename = file.name
                loaded_dfs[filename] = df
                
                # Smart Selection
                if st.checkbox(f"📄 {filename}", value=True):
                    selected_dfs[filename] = df
                    
            except Exception as e:
                st.error(f"❌ {file.name}: {e}")

# --- Main Layout ---
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📋 Data Snapshot")
    if selected_dfs:
        tab_list = list(selected_dfs.keys())
        tabs = st.tabs([name[:15]+"..." for name in tab_list]) # Shorten names for tabs
        
        for i, tab in enumerate(tabs):
            name = tab_list[i]
            df = selected_dfs[name]
            with tab:
                st.metric("Rows", df.shape[0], delta=f"{df.shape[1]} Cols")
                st.dataframe(df.head(5), use_container_width=True)
    else:
        st.info("Upload & select files.")

with col1:
    # Initialize Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

    # Input Area
    if prompt := st.chat_input("Ask a question about your data..."):
        if not api_key:
            st.warning("Please configure your API Key in the .env file.")
            st.stop()
            
        if not selected_dfs:
            st.warning("Please upload a file first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="⚙️"):
            status = st.status("Agents are working...", expanded=True)
            
            try:
                app = create_workflow(api_key, selected_dfs)
                inputs = {"messages": [HumanMessage(content=prompt)]}
                final_answer = ""
                
                for output in app.stream(inputs):
                    for key, value in output.items():
                        content = value["messages"][-1].content
                        
                        if key == "data_analyst":
                            status.write("✅ Analysis complete")
                            with st.expander("Show Technical details"):
                                st.code(content)
                        
                        if key == "writer":
                            final_answer = content
                            status.write("✅ Report generated")

                status.update(label="Complete", state="complete", expanded=False)
                
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                st.error(f"Error: {e}")