import streamlit as st
import os
import subprocess
from pathlib import Path
from atlas_engine import process_paper # Import your existing logic
from db import init_db

st.set_page_config(page_title="Ingestion", layout="wide")
st.title("📂 Document Ingestion & Preprocessing")

# Create directories if they don't exist
output_dir = Path("output")
(output_dir / "json").mkdir(parents=True, exist_ok=True)
(output_dir / "figures").mkdir(parents=True, exist_ok=True)

uploaded_files = st.file_uploader("Upload Sepsis Research Papers (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Start Processing Pipeline"):
        conn = init_db()
        
        for uploaded_file in uploaded_files:
            with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                # 1. Save File
                pdf_path = Path(uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write("✅ File saved.")

                # 2. Run pdffigures2
                st.write("🔍 Extracting visual assets (pdffigures2)...")
                json_out = output_dir / "json" / f"metadata.json{pdf_path.stem}.json"
                subprocess.run([
                    "java", "-jar", "pdffigures2.jar",
                    str(pdf_path),
                    "-g", str(json_out),
                    "-m", str(output_dir / "figures") + "/"
                ], check=True)
                st.write("✅ Figures extracted.")

                # 3. Run Ingestion (Markdown + LLM Extraction)
                st.write("🧠 Extracting clinical variables (GPT-4o Vision)...")
                process_paper(pdf_path, json_out, output_dir / "figures")
                
                status.update(label=f"Finished {uploaded_file.name}!", state="complete")
        
        st.success("All papers ingested! You can now use the Chatbot.")
        # Clear storage directory to force LlamaIndex to re-index new data
        if os.path.exists("./storage"):
            import shutil
            shutil.rmtree("./storage")
            st.info("LlamaIndex storage refreshed.")