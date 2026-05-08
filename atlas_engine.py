import os
import json
import base64
import pandas as pd
import pymupdf4llm
from openai import OpenAI
from pathlib import Path
from models import SepsisAtlas

from dotenv import load_dotenv
load_dotenv()

from db import init_db, save_paper_metadata, save_text_chunk, save_observation, save_figure_metadata
conn = init_db()

from helper_functions import calculate_pdf_hash

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

def encode_image(image_path):
    """Encodes a local image to base64 for the Vision API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_paper(pdf_path, json_metadata_path, figures_dir):
    # This is now a permanent, unchangeable ID
    paper_id = calculate_pdf_hash(pdf_path) 

    # 1. Get high-fidelity Markdown text using pymupdf4llm
    # This preserves the structure (Headers, Tables) better than raw text.
    print(f"Extracting Markdown structure from {pdf_path.name}...")
    md_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    save_paper_metadata(conn, paper_id, pdf_path.name)
    for i, page in enumerate(md_pages):
        save_text_chunk(conn, paper_id, i + 1, page["text"])

    # Reconstruct full text for the LLM prompt
    md_text = "\n".join([p["text"] for p in md_pages])

    # 2. Use pdffigures2 JSON to map visual assets
    with open(json_metadata_path, 'r') as f:
        fig_data = json.load(f)

    # Filter for figures
    relevant_items = []
    for fig in fig_data.get('figures', []):
        if fig.get('figType', '') != "Figure":
            continue
        relevant_items.append(fig)

    if not relevant_items:
        print(f"No figures found via pdffigures2 in {pdf_path.name}")
        # We still return the MD text for a text-only backup if needed
        return []

    all_extracted_rows = []

    # 3. Process each relevant figure with Vision + Markdown Context
    for item in relevant_items:
        img_url = item.get('renderURL')
        if not img_url: continue
        
        local_img_path = Path(figures_dir) / Path(img_url).name
        if not local_img_path.exists(): continue
        
        print(f"Analyzing Figure: {item.get('caption')}...")
        base64_image = encode_image(local_img_path)

        system_prompt = """
        You are a Senior Clinical Data Scientist. You will be provided with a clinical paper in Markdown and a specific image of a figure.
        Your task is to extract evidence for 'Counterfactual Mortality Estimation'.
        
        You MUST return a JSON object with a key 'evidence_base' containing a list of objects.
        
        TARGET CATEGORIES PER ROW:
        - study: Lead Author and Year (find this in the Markdown text).
        - population: Location, sepsis type, and age group.
        - sample_size: A SINGLE STRING (e.g., 'Total N=550; Died N=68; Survived N=482').
        - predictor: The clinical variable (e.g. 'Lactate', 'SOFA score', 'NEWS').
        - outcome: e.g. '28-day mortality' or 'In-hospital death'.
        - measurement_timing: State exactly when the clinical variable was measured (e.g., 'At ED triage', 'Upon ICU admission', 'Within 24 hours'). If it is not in the table, you MUST look at the Methods section or the Title in the Markdown text to find it. Do not default to 'Not Reported' if the setting implies 'on admission'.
        - method: e.g. 'Multivariable Logistic Regression', 'ROC analysis'.
        - effect_size: OR, HR, or Cutoff (e.g. 'OR 1.25').
        - performance: AUC with 95% CI and p-values (e.g. '0.81 (0.76-0.86), p<0.05').
        - notes: List adjustment variables or specific model info.

        STRICT RULES:
        1. ALL FIELDS MUST BE STRINGS. DO NOT return nested JSON objects or arrays inside the fields.
        2. If the figure lists multiple predictors, create a separate JSON object for each one.
        3. Normalize all Lactate to mmol/L (mg/dL / 9.0).
        4. Use 'Not Reported' ONLY if the information is truly nowhere in the table, methods, or title.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"--- STUDY MARKDOWN CONTEXT ---\n{md_text}"},
                    {"type": "text", "text": f"--- TARGET IMAGE CAPTION ---\n{item.get('caption')}"},
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:image/png;base64,{base64_image}" }
                    },
                    {"type": "text", "text": "Extract all clinical predictors from this figure image into the evidence_base list based on the context."}
                ]
            }
        ]

        fig_db_id = save_figure_metadata(conn, paper_id, item.get('name'), item.get('caption'), local_img_path, item.get('page')+1)
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            res_json = json.loads(response.choices[0].message.content)

            # Validate the ENTIRE JSON response against the parent model
            atlas_data = SepsisAtlas(**res_json)
            
            for obs in atlas_data.evidence_base:
                try:
                    # obs is now a fully validated Pydantic object
                    save_observation(conn, paper_id, fig_db_id, item.get('page')+1, obs)
                except Exception as db_err:
                    print(f"   ⚠️ Database save failed for row: {db_err}")

        except Exception as e:
            print(f"   ❌ API or Validation Error: {e}")

    return all_extracted_rows

def run_atlas_pipeline(pdf_dir, json_dir, figures_dir):
    processed_count = 0

    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        json_file = Path(json_dir) / f"{pdf_file.stem}.json"
        
        if not json_file.exists():
            print(f"⏩ Skipping {pdf_file.name}: No pdffigures2 JSON found.")
            continue

        print(f"🔬 Processing {pdf_file.name}...")
        
        process_paper(pdf_file, json_file, figures_dir)
        
        processed_count += 1
    
    print(f"\n✅ PIPELINE COMPLETE: {processed_count} papers processed.")
    print(f"📂 All clinical evidence and narrative text are now in 'sepsis_atlas.db'.")

if __name__ == "__main__":
    run_atlas_pipeline(
        pdf_dir="./", 
        json_dir="./output/json", 
        figures_dir="./output/figures"
    )