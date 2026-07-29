# pip install -U langchain langchain-mistralai langchain-community
# faiss-cpu pypdf python-dotenv langsmith

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------------------------------------------
# LangSmith's traceable decorator.
#
# It is used to create CUSTOM RUNS for normal Python functions.
#
# By default, LangSmith automatically traces only LangChain components
# such as:
#
# • PromptTemplate
# • Chat Models
# • Retrievers
# • Chains
# • Output Parsers
#
# It DOES NOT trace ordinary Python functions.
#
# Using @traceable allows us to observe every important preprocessing
# step such as:
#
# • PDF Loading
# • Chunking
# • Building Index
# • Loading Cached Index
# • Entire Setup Pipeline
#
# making debugging and performance analysis much easier.
# -------------------------------------------------------------------
from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import (
    MistralAIEmbeddings,
    ChatMistralAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


# -------------------------------------------------------------------
# Load all API keys.
#
# Required in .env
#
# LANGCHAIN_API_KEY
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=RAG Chatbot
# MISTRAL_API_KEY
#
# Every execution will automatically appear inside LangSmith.
# -------------------------------------------------------------------
load_dotenv()


PDF_PATH = "islr.pdf"

# -------------------------------------------------------------------
# Folder where all FAISS indexes are stored.
#
# Instead of rebuilding embeddings every time,
# the index will be saved here.
#
# Example
#
# .indices/
#      a83729ad8f7....
#      92af721cc8...
#
# Every folder represents one unique PDF configuration.
# -------------------------------------------------------------------
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)



# ===================================================================
#               PDF LOADING (CUSTOM TRACE)
# ===================================================================

@traceable(name="load_pdf")
def load_pdf(path: str):

    # This function loads the PDF.

    # Since it is decorated with @traceable,
    # LangSmith creates a separate child run named
    #
    # load_pdf
    #
    # showing:
    #
    # • execution time
    # • inputs
    # • outputs
    # • errors
    #
    return PyPDFLoader(path).load()



# ===================================================================
#                 DOCUMENT CHUNKING (CUSTOM TRACE)
# ===================================================================

@traceable(name="split_documents")
def split_documents(
    docs,
    chunk_size=1000,
    chunk_overlap=150,
):

    # Splits long documents into smaller chunks.

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # LangSmith now records

    # split_documents
    #
    # as a separate run.

    return splitter.split_documents(docs)



# ===================================================================
#            BUILD VECTOR DATABASE (CUSTOM TRACE)
# ===================================================================

@traceable(name="build_vectorstore")
def build_vectorstore(
    splits,
    embed_model_name: str,
):

    # Create embedding model

    emb = MistralAIEmbeddings(
        model=embed_model_name
    )

    # Convert chunks into embeddings.

    # LangSmith automatically records
    #
    # • embedding calls
    # • latency
    # • tokens (if applicable)

    return FAISS.from_documents(
        splits,
        emb
    )



# ===================================================================
#              CREATE A UNIQUE PDF FINGERPRINT
# ===================================================================

def _file_fingerprint(path: str):

    # Instead of checking only the filename,
    # we calculate a SHA256 hash.

    # If even one byte changes inside the PDF,
    # the hash changes.

    p = Path(path)

    h = hashlib.sha256()

    with p.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return {

        "sha256": h.hexdigest(),

        "size": p.stat().st_size,

        "mtime": int(
            p.stat().st_mtime
        ),
    }



# ===================================================================
#          GENERATE UNIQUE CACHE KEY
# ===================================================================

def _index_key(
    pdf_path,
    chunk_size,
    chunk_overlap,
    embed_model_name,
):

    # Every index depends on

    # • PDF
    # • Embedding model
    # • Chunk size
    # • Chunk overlap

    # If ANY of these changes,
    # a completely new cache key is created.

    meta = {

        "pdf_fingerprint":
            _file_fingerprint(pdf_path),

        "chunk_size":
            chunk_size,

        "chunk_overlap":
            chunk_overlap,

        "embedding_model":
            embed_model_name,

        "format":
            "v1",
    }

    return hashlib.sha256(

        json.dumps(
            meta,
            sort_keys=True
        ).encode("utf-8")

    ).hexdigest()



# ===================================================================
#            LOAD EXISTING INDEX (CUSTOM TRACE)
# ===================================================================

@traceable(
    name="load_index",
    tags=["index"]
)
def load_index_run(
    index_dir,
    embed_model_name,
):

    # If the index already exists,
    # DO NOT create embeddings again.

    emb = MistralAIEmbeddings(
        model=embed_model_name
    )

    # Simply load the FAISS index.

    # LangSmith shows

    # load_index

    # instead of

    # build_index

    return FAISS.load_local(

        str(index_dir),

        emb,

        allow_dangerous_deserialization=True,
    )



# ===================================================================
#            BUILD NEW INDEX (CUSTOM TRACE)
# ===================================================================

@traceable(
    name="build_index",
    tags=["index"]
)
def build_index_run(

    pdf_path,

    index_dir,

    chunk_size,

    chunk_overlap,

    embed_model_name,
):

    # Parent Run

    # build_index

    # Child Runs

    # ├── load_pdf
    # ├── split_documents
    # └── build_vectorstore

    docs = load_pdf(pdf_path)

    splits = split_documents(

        docs,

        chunk_size,

        chunk_overlap,

    )

    vs = build_vectorstore(

        splits,

        embed_model_name,

    )

    index_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    # Save FAISS permanently.

    vs.save_local(str(index_dir))

    # Save metadata.

    (index_dir / "meta.json").write_text(

        json.dumps({

            "pdf_path":
                os.path.abspath(pdf_path),

            "chunk_size":
                chunk_size,

            "chunk_overlap":
                chunk_overlap,

            "embedding_model":
                embed_model_name,

        }, indent=2)

    )

    return vs



# ===================================================================
#       LOAD CACHE OR BUILD NEW INDEX
# ===================================================================

def load_or_build_index(

    pdf_path,

    chunk_size=1000,

    chunk_overlap=150,

    embed_model_name="mistral-embed",

    force_rebuild=False,

):

    # Create cache key.

    key = _index_key(

        pdf_path,

        chunk_size,

        chunk_overlap,

        embed_model_name,

    )

    index_dir = INDEX_ROOT / key

    # If cached index exists,
    # load it.

    cache_hit = (

        index_dir.exists()

        and

        not force_rebuild

    )

    if cache_hit:

        return load_index_run(

            index_dir,

            embed_model_name,

        )

    # Otherwise create new index.

    return build_index_run(

        pdf_path,

        index_dir,

        chunk_size,

        chunk_overlap,

        embed_model_name,

    )



# ===================================================================
#                  LLM
# ===================================================================

llm = ChatMistralAI(

    model="mistral-small-2506",

    temperature=0,

)



# ===================================================================
#                    PROMPT
# ===================================================================

prompt = ChatPromptTemplate.from_messages([

    (

        "system",

        "Answer ONLY from the provided context."

        " If not found, say you don't know.",

    ),

    (

        "human",

        "Question: {question}\n\nContext:\n{context}",

    ),

])



def format_docs(docs):

    return "\n\n".join(

        doc.page_content

        for doc in docs

    )



# ===================================================================
#               COMPLETE SETUP (CUSTOM TRACE)
# ===================================================================

@traceable(
    name="setup_pipeline",
    tags=["setup"]
)
def setup_pipeline(

    pdf_path,

    chunk_size=1000,

    chunk_overlap=150,

    embed_model_name="mistral-embed",

    force_rebuild=False,

):

    # LangSmith creates

    # setup_pipeline

    # which internally calls either

    # load_index

    # OR

    # build_index

    return load_or_build_index(

        pdf_path,

        chunk_size,

        chunk_overlap,

        embed_model_name,

        force_rebuild,

    )



# ===================================================================
#            COMPLETE RAG EXECUTION
# ===================================================================

@traceable(name="pdf_rag_full_run")
def setup_pipeline_and_query(

    pdf_path,

    question,

    chunk_size=1000,

    chunk_overlap=150,

    embed_model_name="mistral-embed",

    force_rebuild=False,

):

    vectorstore = setup_pipeline(

        pdf_path,

        chunk_size,

        chunk_overlap,

        embed_model_name,

        force_rebuild,

    )

    retriever = vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={"k": 4},

    )

    parallel = RunnableParallel({

        "context":retriever | RunnableLambda(format_docs),

        "question": RunnablePassthrough(),

    })

    chain = ( parallel | prompt | llm | StrOutputParser())

    return chain.invoke(

        question,

        config={

            "run_name": "pdf_rag_query",

            "tags": ["qa"],

            "metadata": {"k": 4},

        }

    )



# ===================================================================
#                     CLI
# ===================================================================

if __name__ == "__main__":

    print("PDF RAG ready.")

    q = input("\nQ: ").strip()

    ans = setup_pipeline_and_query(

        PDF_PATH,

        q,

    )

    print("\nA:", ans)