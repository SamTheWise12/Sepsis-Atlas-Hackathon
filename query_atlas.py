import os
import sqlite3
from sqlalchemy import create_engine

from helper_functions import get_paper_name_from_id

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import SQLDatabase, Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core import PromptTemplate

from llama_index.core import StorageContext, load_index_from_storage
PERSIST_DIR = "./storage"

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

from llama_index.llms.openai import OpenAI

# Configure OpenRouter LLM & Standard HuggingFace Embeddings
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    api_base="https://openrouter.ai/api/v1",
    temperature=0.0
)


embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_folder="./hf_cache"
)

Settings.llm = llm
Settings.embed_model = embed_model

def setup_sql_engine():
    """Sets up the SQL engine for structured table data."""
    engine = create_engine("sqlite:///sepsis_atlas.db")
    sql_database = SQLDatabase(engine, include_tables=["observations", "papers"])
    
    sql_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["observations", "papers"],
        llm=llm,
        synthesize_response=True
    )
    
    # --- STRICT SQL SYNTHESIS PROMPT ---
    sql_synthesis_tmpl_str = (
        "You are an AI Clinical Evidence Assistant. Use the following SQL results to build a table.\n"
        "Query: {query_str}\n"
        "SQL Response: {context_str}\n"
        "RULES:\n"
        "1. Format the results EXCLUSIVELY as a Markdown table.\n"
        "2. Create a 'Source' column. For every row, look at the 'study' and 'page_number' values "
        "and combine them into a string like: 'Suttapanit et al. 2022, Page 6'.\n"
        "3. If 'page_number' is missing, just use the 'study' name.\n"
        "4. If no data was found in the SQL Response, output: 'There is no evidence for this in the provided sources.'\n"
        "Response: "
    )
    sql_synthesis_prompt = PromptTemplate(sql_synthesis_tmpl_str)
    sql_engine.update_prompts({"response_synthesis_prompt": sql_synthesis_prompt})
    
    return sql_engine

def setup_vector_engine():
    """Sets up the Vector RAG engine for unstructured Markdown text."""
    from helper_functions import get_paper_name_from_id 

    qa_prompt_tmpl_str = (
        "You are an expert Clinical Evidence Assistant. Use the provided context to answer the query.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "RULES:\n"
        "1. Answer ONLY using the context provided. Do NOT use outside knowledge.\n"
        "2. You MUST cite the source at the end of every sentence using: [Source: {paper_name}, Page: {page}].\n"
        "3. If the answer is NOT in the context, state: 'There is no evidence for this in the provided sources.'\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt = PromptTemplate(qa_prompt_tmpl_str)

    if not os.path.exists(PERSIST_DIR):
        print("⚡ Index not found, generating new embeddings...")
        conn = sqlite3.connect("sepsis_atlas.db")
        df_text = pd.read_sql_query("SELECT paper_id, page_number, chunk_content FROM text_chunks", conn)
        conn.close()

        documents = []
        for _, row in df_text.iterrows():
            readable_name = get_paper_name_from_id("sepsis_atlas.db", row["paper_id"])
            
            doc = Document(
                text=row["chunk_content"],
                metadata={
                    "paper_name": readable_name, 
                    "page": str(row["page_number"])
                }
            )

            doc.excluded_llm_metadata_keys = [] 
            documents.append(doc)

        parser = MarkdownNodeParser(chunk_size=1024, chunk_overlap=128) 
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes, settings=Settings)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        print("📂 Loading existing index from disk...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
 
    return index.as_query_engine(
        llm=llm, 
        text_qa_template=qa_prompt,
        similarity_top_k=10 
    )

def build_hybrid_router():
    """Combines SQL and Vector engines into one smart Agent."""
    sql_engine = setup_sql_engine()
    vector_engine = setup_vector_engine()

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_engine,
        name="sql_structured_data_tool",
        description=(
            "Useful for answering questions about numbers, metrics, mortality rates, "
            "AUC scores, sample sizes, and generating structured tables comparing clinical predictors."
        )
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        name="vector_narrative_text_tool",
        description=(
            "Useful for answering qualitative questions about the paper's methodology, "
            "inclusion/exclusion criteria, study design, definitions, limitations, and discussions."
        )
    )

    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[sql_tool, vector_tool],
    )
    
    return router_query_engine

if __name__ == "__main__":
    router = build_hybrid_router()

    print("\n--- Test 1: Valid Structured Request ---")
    print(router.query("Create a markdown table comparing the AUC performance of NEWS vs qSOFA. Include the source."))
    # print(safe_query(router, "Create a markdown table comparing the AUC performance of NEWS vs qSOFA. Include the source."))

    print("\n--- Test 2: Valid Unstructured Request ---")
    print(router.query("What were the exclusion criteria for the Suttapanit study?"))
    
    print("\n--- Test 3: Hallucination Trap (Testing the Fallback) ---")
    # Ask about a predictor or paper that you purposefully DID NOT feed into the database
    print(router.query("What is the AUC performance of the Apache IV score in the study by Dr. Simeon Gerov and Dr. Samuil Gerov in 2018?"))