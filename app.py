import os
import fitz  # PyMuPDF
import gradio as gr
from typing import Annotated, TypedDict, List, Dict, Any
from typing_extensions import Literal

# LangChain & LangGraph Imports
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults 
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_community.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

# --- 1. STATE DEFINITION ---
class AgentState(TypedDict):
    domain: str
    basic_problem: str
    messages: List[BaseMessage]
    raw_papers: List[Dict[str, str]]
    parsed_docs: List[Document]
    retrieved_chunks: List[Document]
    comparison_matrix: str
    research_gaps: List[str]
    novel_method: str
    paper_draft: str
    citations: str
    next_step: str

# --- 2. GLOBAL MODELS & TOOLS ---
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
arxiv_tool = ArxivAPIWrapper(top_k_results=3)

# --- 3. NODE DEFINITIONS ---

def supervisor_node(state: AgentState):
    """Orchestrates initial routing."""
    return {"next_step": "search"}

def search_node(state: AgentState, groq_key: str, tavily_key: str):
    """Searches Tavily and arXiv for relevant paper titles and summaries."""
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["TAVILY_API_KEY"] = tavily_key
    
    tavily_tool = TavilySearchResults(max_results=3)
    query = f"{state['domain']} {state['basic_problem']}"
    
    # 1. arXiv Search
    arxiv_res = arxiv_tool.run(query)
    
    # 2. Tavily Search
    tavily_res = tavily_tool.invoke(query)
    
    raw_papers = [
        {"source": "arXiv", "content": arxiv_res},
        {"source": "Tavily", "content": str(tavily_res)}
    ]
    
    return {"raw_papers": raw_papers, "next_step": "parser"}

def parser_node(state: AgentState):
    """Parses gathered data and converts it into structured documents."""
    docs = []
    for paper in state["raw_papers"]:
        doc = Document(
            page_content=paper["content"],
            metadata={"source": paper["source"]}
        )
        docs.append(doc)
    return {"parsed_docs": docs, "next_step": "rag"}

def rag_node(state: AgentState):
    """Chunking, Embedding, FAISS + BM25 Hybrid Retrieval, and Reranking."""
    # Chunking
    chunks = text_splitter.split_documents(state["parsed_docs"])
    
    # Vector DB (Dense Retrieval)
    faiss_db = FAISS.from_documents(chunks, embeddings)
    dense_retriever = faiss_db.as_retriever(search_kwargs={"k": 3})
    
    # BM25 (Sparse Retrieval)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3
    
    # Hybrid Retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6]
    )
    
    query = f"Methodologies and architectures for {state['basic_problem']}"
    retrieved_chunks = ensemble_retriever.invoke(query)
    
    return {"retrieved_chunks": retrieved_chunks, "next_step": "reviewer"}

def reviewer_node(state: AgentState, llm: ChatGroq):
    """Compares papers and extracts research gaps."""
    context = "\n\n".join([c.page_content for c in state["retrieved_chunks"]])
    prompt = f"""You are a Senior Literature Reviewer.
    Domain: {state['domain']}
    Problem: {state['basic_problem']}
    Context: {context}

    Task:
    1. Provide a brief Comparison Matrix of the current approaches.
    2. Identify 3 specific 'Research Gaps'.
    Format output with 'GAP:' prefix for each gap."""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content
    
    gaps = [line.strip() for line in response.split('\n') if 'GAP:' in line or 'gap' in line.lower()][:3]
    if not gaps:
        gaps = [response]

    return {
        "comparison_matrix": response,
        "research_gaps": gaps,
        "next_step": "planner"
    }

def planner_node(state: AgentState, llm: ChatGroq):
    """Proposes novel research directions to address identified gaps."""
    prompt = f"""You are an Academic Research Planner.
    Gaps: {state['research_gaps']}
    
    Task: Formulate 1 highly novel research methodology/architecture to address these gaps."""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"novel_method": response, "next_step": "writer"}

def writer_node(state: AgentState, llm: ChatGroq):
    """Drafts formal IEEE-formatted research proposal."""
    prompt = f"""You are an Academic Paper Architect.
    Domain: {state['domain']}
    Problem: {state['basic_problem']}
    Literature Gaps: {state['research_gaps']}
    Proposed Novel Method: {state['novel_method']}

    Task: Draft an IEEE-style Research Proposal containing:
    1. Abstract
    2. Introduction
    3. Related Work
    4. Proposed Methodology & Architecture
    5. Expected Results & Conclusion"""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"paper_draft": response, "next_step": "citation"}

def citation_node(state: AgentState, llm: ChatGroq):
    """Generates standard IEEE/APA reference citations."""
    prompt = f"""Generate full IEEE format citations for the following research background context:
    {state['comparison_matrix']}"""
    
    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"citations": response, "next_step": "end"}

# --- 4. GRAPH COMPOSITION ---

def run_workflow(domain: str, problem: str, groq_key: str, tavily_key: str):
    if not groq_key or not tavily_key:
        yield "❌ Error: Please provide both Groq and Tavily API Keys.", "", "", ""
        return

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=groq_key
    )

    # Instantiate LangGraph StateGraph
    workflow = StateGraph(AgentState)

    # Bind parameters to nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("search", lambda s: search_node(s, groq_key, tavily_key))
    workflow.add_node("parser", parser_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("reviewer", lambda s: reviewer_node(s, llm))
    workflow.add_node("planner", lambda s: planner_node(s, llm))
    workflow.add_node("writer", lambda s: writer_node(s, llm))
    workflow.add_node("citation", lambda s: citation_node(s, llm))

    # Define flow
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "search")
    workflow.add_edge("search", "parser")
    workflow.add_edge("parser", "rag")
    workflow.add_edge("rag", "reviewer")
    workflow.add_edge("reviewer", "planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "citation")
    workflow.add_edge("citation", END)

    app = workflow.compile()

    initial_state = {
        "domain": domain,
        "basic_problem": problem,
        "messages": [],
        "raw_papers": [],
        "parsed_docs": [],
        "retrieved_chunks": [],
        "comparison_matrix": "",
        "research_gaps": [],
        "novel_method": "",
        "paper_draft": "",
        "citations": "",
        "next_step": "supervisor"
    }

    status_log = "⚡ Initiating PRAGYAN AI Research Pipeline...\n"
    yield status_log, "", "", ""

    comp_matrix = ""
    gaps_and_plan = ""
    full_paper = ""

    for event in app.stream(initial_state):
        for node_name, node_state in event.items():
            status_log += f"➡️ Completed Node: [{node_name.upper()}]\n"
            
            if "comparison_matrix" in node_state:
                comp_matrix = node_state["comparison_matrix"]
            if "novel_method" in node_state:
                gaps_and_plan = f"### Identified Gaps:\n" + "\n".join(node_state.get("research_gaps", [])) + f"\n\n### Novel Method:\n" + node_state["novel_method"]
            if "paper_draft" in node_state:
                full_paper = node_state["paper_draft"]
            if "citations" in node_state:
                full_paper += "\n\n## References & Citations\n" + node_state["citations"]

            yield status_log, comp_matrix, gaps_and_plan, full_paper

# --- 5. GRADIO USER INTERFACE ---

with gr.Blocks(title="PRAGYAN AI - Multi-Agent Academic Research Platform", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 PRAGYAN AI: Multi-Agent Academic Research Platform")
    gr.Markdown("Automated end-to-end research paper exploration, literature matrix generation, research gap identification, and IEEE draft generation[cite: 1].")

    with gr.Sidebar():
        gr.Markdown("### 🔑 API Configuration")
        groq_input = gr.Textbox(label="Groq API Key", type="password")
        tavily_input = gr.Textbox(label="Tavily API Key", type="password")

    with gr.Row():
        with gr.Column(scale=1):
            domain_input = gr.Textbox(label="Research Domain", value="Agentic & Generative AI in EdTech")
            problem_input = gr.Textbox(label="Research Problem / Objective", value="Personalized autonomous AI tutoring systems with context retention", lines=3)
            submit_btn = gr.Button("🚀 Run Multi-Agent System", variant="primary")
            
            status_output = gr.Textbox(label="Agent Execution Status Log", lines=10, interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📄 IEEE Paper Proposal"):
                    paper_output = gr.Markdown()
                with gr.TabItem("📊 Literature Review & Comparison"):
                    matrix_output = gr.Markdown()
                with gr.TabItem("💡 Research Gaps & Novel Method"):
                    gaps_output = gr.Markdown()

    submit_btn.click(
        fn=run_workflow,
        inputs=[domain_input, problem_input, groq_input, tavily_input],
        outputs=[status_output, matrix_output, gaps_output, paper_output]
    )

if __name__ == "__main__":
    demo.queue().launch()
