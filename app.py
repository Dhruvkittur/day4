import os
import streamlit as st
from typing import Annotated, TypedDict, List, Dict
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Academic Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Groq-Powered AI Research Assistant")
st.caption("Automated Literature Search, Gap Identification & Draft Generation")

# --- SIDEBAR: API KEYS & CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check st.secrets or prompt user for inputs
    groq_api_key = st.text_input(
        "Groq API Key", 
        value=st.secrets.get("GROQ_API_KEY", ""), 
        type="password",
        help="Get your key at https://console.groq.com/"
    )
    
    tavily_api_key = st.text_input(
        "Tavily API Key", 
        value=st.secrets.get("TAVILY_API_KEY", ""), 
        type="password",
        help="Get your key at https://tavily.com/"
    )
    
    model_choice = st.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192"],
        index=0
    )

# Set environment variables if key inputs exist
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key

# --- LANGGRAPH SETUP ---

class AgentState(TypedDict):
    """The state of the research assistant."""
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    domain: str
    basic_problem: str
    research_gaps: List[str]
    paper_draft: Dict[str, str]
    next_step: str

def create_agent(llm, tools, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

def build_graph(llm, search_tool):
    """Builds and compiles the StateGraph workflow."""
    
    def explorer_node(state: AgentState):
        explorer_prompt = f"""You are the Search Specialist using GROQ speed.
        Domain: {state['domain']}
        Current Problem: {state['basic_problem']}

        Task: Find the 3 most relevant current papers.
        Focus on specific methodologies and architectural trends."""

        agent = create_agent(llm, [search_tool], explorer_prompt)
        response = agent.invoke(state)
        return {"messages": [response], "next_step": "reviewer"}

    def reviewer_node(state: AgentState):
        reviewer_prompt = f"""You are the Literature Reviewer.
        Analyze gathered info and identify 3 'Research Gaps'.
        What is missing in {state['domain']} regarding {state['basic_problem']}?"""

        agent = create_agent(llm, None, reviewer_prompt)
        response = agent.invoke(state)
        
        # Extract gaps from response text
        gaps = [line for line in response.content.split('\n') if 'gap' in line.lower()][:3]
        if not gaps:
            gaps = [response.content]

        return {
            "messages": [response],
            "research_gaps": gaps,
            "next_step": "writer"
        }

    def writer_node(state: AgentState):
        writer_prompt = f"""You are the Academic Paper Architect.
        Gaps: {state['research_gaps']}
        Draft: Abstract, Intro, Methodology, Proposed Arch, and Conclusion.
        Tone: Formal Academic."""

        agent = create_agent(llm, None, writer_prompt)
        response = agent.invoke(state)

        return {
            "messages": [response],
            "paper_draft": {"full_report": response.content},
            "next_step": "end"
        }

    def router(state: AgentState):
        return state["next_step"]

    workflow = StateGraph(AgentState)
    workflow.add_node("explorer", explorer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("writer", writer_node)

    workflow.add_edge(START, "explorer")
    workflow.add_conditional_edges("explorer", router, {"reviewer": "reviewer", "end": END})
    workflow.add_conditional_edges("reviewer", router, {"writer": "writer", "end": END})
    workflow.add_edge("writer", END)

    return workflow.compile()


# --- INPUT FORM ---
with st.form("research_form"):
    domain_input = st.text_input(
        "Research Domain", 
        value="Sustainable Energy AI"
    )
    problem_input = st.text_area(
        "Research Problem / Objective", 
        value="Predicting micro-grid stability with high intermittent renewable penetration"
    )
    submit_button = st.form_submit_button("🚀 Start Agent Research Workflow")

# --- EXECUTION LOGIC ---
if submit_button:
    if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
        st.error("Please provide both Groq and Tavily API Keys in the sidebar to proceed.")
    else:
        try:
            llm = ChatGroq(model=model_choice, temperature=0.1, max_retries=2)
            search_tool = TavilySearchResults(max_results=3)
            app = build_graph(llm, search_tool)

            initial_input = {
                "domain": domain_input,
                "basic_problem": problem_input,
                "messages": [HumanMessage(content=f"Analyze {domain_input} research on {problem_input}.")],
                "research_gaps": [],
                "paper_draft": {},
                "next_step": ""
            }

            st.write("---")
            st.subheader("⚡ Agent Workflow Progress")
            
            # Create a container to stream status updates
            status_container = st.container()

            with status_container:
                for output in app.stream(initial_input):
                    for key, value in output.items():
                        if key == "explorer":
                            with st.status("🔍 **Explorer Node**: Searching literature...", expanded=False):
                                st.markdown(value['messages'][-1].content)
                        elif key == "reviewer":
                            with st.status("🧐 **Reviewer Node**: Identifying research gaps...", expanded=False):
                                st.markdown(value['messages'][-1].content)
                        elif key == "writer":
                            with st.status("✍️ **Writer Node**: Drafting paper proposal...", expanded=False):
                                st.markdown(value['messages'][-1].content)
                                draft_content = value.get("paper_draft", {}).get("full_report", "")

            st.success("Workflow Complete!")

            # Display final draft report
            if 'draft_content' in locals() and draft_content:
                st.write("---")
                st.header("📄 Generated Research Paper Draft")
                st.markdown(draft_content)
                
                st.download_button(
                    label="📥 Download Paper Draft (.md)",
                    data=draft_content,
                    file_name="research_paper_draft.md",
                    mime="text/markdown"
                )

        except Exception as e:
            st.error(f"An error occurred during execution: {e}")
