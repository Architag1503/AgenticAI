# pip install -U langchain langchain-mistralai langchain-community faiss-cpu pypdf python-dotenv

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------------
# Load environment variables.
#
# Make sure your .env file contains:
# LANGCHAIN_API_KEY=<your_langsmith_api_key>
# LANGCHAIN_TRACING_V2=true
# MISTRAL_API_KEY=<your_mistral_api_key>
#
# These enable automatic tracing in LangSmith.
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# All traces generated from this application will be grouped under
# the "RAG Chatbot" project inside LangSmith.
# -------------------------------------------------------------------
os.environ["LANGCHAIN_PROJECT"] = "RAG Chatbot"

PDF_PATH = "islr.pdf"  # PDF to build the knowledge base from


# ===================================================================
# 1) Load PDF
# ===================================================================

# Reads the PDF and converts each page into a LangChain Document.
#
# LangSmith will record this loader execution as part of the chain
# whenever it is wrapped inside a traced workflow.
loader = PyPDFLoader(PDF_PATH)

# One Document object per PDF page.
docs = loader.load()


# ===================================================================
# 2) Split into Chunks
# ===================================================================

# Splits long pages into smaller overlapping chunks.
#
# Chunking improves retrieval quality because embeddings work better
# on smaller pieces of text.
#
# LangSmith allows you to inspect retrieved chunks later during
# debugging if they are passed through the chain.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

splits = splitter.split_documents(docs)


# ===================================================================
# 3) Create Embeddings & Vector Store
# ===================================================================

# Converts every chunk into a vector embedding.
#
# LangSmith records this embedding model whenever embedding calls
# occur during traced executions.
emb = MistralAIEmbeddings(model="mistral-embed")

# Stores embeddings in a FAISS vector database.
vs = FAISS.from_documents(splits, emb)

# Retriever fetches the Top-4 most relevant chunks for a query.
#
# During LangSmith tracing you can inspect:
# • Retrieved documents
# • Similarity search
# • Retrieved context
retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)


# ===================================================================
# 4) Prompt Template
# ===================================================================

# Defines the prompt sent to the LLM.
#
# LangSmith records:
# • Prompt template
# • Filled prompt
# • Retrieved context
# • User question
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If not found, say you don't know.",
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}",
    ),
])


# ===================================================================
# 5) Create LLM
# ===================================================================

# Mistral model used for answer generation.
#
# LangSmith records:
# • Model name
# • Prompt
# • Response
# • Token usage
# • Latency
llm = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0,
)


# ===================================================================
# Helper Function
# ===================================================================

# Combines retrieved document chunks into a single string.
#
# LangSmith displays the formatted context passed to the prompt,
# making it easy to verify whether retrieval worked correctly.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ===================================================================
# 6) Parallel Runnable
# ===================================================================

# RunnableParallel executes both branches simultaneously.
#
# Branch 1:
#     User Question
#          ↓
#      Retriever
#          ↓
#    Retrieved Documents
#          ↓
#    format_docs()
#
# Branch 2:
#     Original Question
#
# LangSmith visualizes these as parallel child runs,
# allowing you to inspect each independently.
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
})


# ===================================================================
# 7) Complete RAG Chain
# ===================================================================

# Overall Flow:
#
# User Question
#        │
#        ▼
# RunnableParallel
#   ├── Retriever
#   └── Question
#        │
#        ▼
# Prompt Template
#        │
#        ▼
# Mistral LLM
#        │
#        ▼
# StrOutputParser
#
# LangSmith displays this complete pipeline as a hierarchical trace,
# making it easy to debug every stage of retrieval and generation.
chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)


# ===================================================================
# 8) User Query
# ===================================================================

print("PDF RAG ready. Ask a question (Ctrl+C to exit).")

q = input("\nQ: ")


# ===================================================================
# 9) LangSmith Run Configuration
# ===================================================================

# This configuration is attached to the trace.
#
# It helps organize runs and compare experiments.
config = {

    # Custom run name shown in LangSmith.
    "run_name": "PDF RAG Query",

    # Tags for filtering and searching traces.
    "tags": [
        "rag",
        "pdf chatbot",
        "faiss",
        "mistral",
    ],

    # Extra information stored with every trace.
    # Useful for experiment comparison.
    "metadata": {
        "embedding_model": "mistral-embed",
        "llm": "mistral-small-2506",
        "retriever": "FAISS",
        "search_type": "similarity",
        "top_k": 4,
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "parser": "StrOutputParser",
    },
}


# ===================================================================
# 10) Execute Chain
# ===================================================================

# LangSmith automatically records:
#
# ✓ User question
# ✓ Retrieved documents
# ✓ Retrieved context
# ✓ Prompt
# ✓ LLM request
# ✓ LLM response
# ✓ Output parser
# ✓ Final answer
# ✓ Token usage
# ✓ Latency
# ✓ Metadata
# ✓ Tags
# ✓ Parent-child execution graph
ans = chain.invoke(
    q.strip(),
    config=config,
)

print("\nA:", ans)

# Above application still not traced properly like the steps are:
# Loading of PDF
# Creation of Embeddings

# Another problem is that in every run pdf is load and chunk although it had done previously & becuase of it it becomes more time consuming