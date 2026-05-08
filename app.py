import streamlit as st

st.set_page_config(page_title="Sepsis Atlas AI", layout="wide")

st.title("🧬 Sepsis Atlas: Clinical Evidence Engine")
st.markdown("""
### Transforming unstructured research into analysis-ready knowledge.

**Challenge:** Clinical data is trapped in PDFs. Manual meta-analysis is slow. 
**Solution:** An AI pipeline that uses Vision to parse tables and LlamaIndex to query them.

---
#### 🛠️ How to use:
1. **Preprocessing:** Go to the sidebar and upload your clinical papers. The system will extract figures, chunk the text, and build the evidence database.
2. **Sepsis Assistant:** Ask questions about biomarkers, mortality rates, or study inclusion criteria.
""")