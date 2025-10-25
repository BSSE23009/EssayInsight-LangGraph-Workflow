import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
import operator

# Load environment variables
load_dotenv()

# Initialize LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define structured output
class InputSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay.")
    score: int = Field(description="Score out of 10.", ge=0, le=10)

structured_model = model.with_structured_output(InputSchema)

# Define state
class EssayInsightState(TypedDict):
    essay_text: str
    language_feedback: str
    grammar_feedback: str
    structured_feedback: str
    overall_feedback: str
    scores: Annotated[list[int], operator.add]
    average_score: float

# Workflow functions
def input_essay(state: EssayInsightState) -> EssayInsightState:
    return {'essay_text': state['essay_text']}

def language_check(state: EssayInsightState) -> EssayInsightState:
    response = structured_model.invoke(f"Evaluate language quality and give a score: {state['essay_text']}")
    return {'language_feedback': response.feedback, 'scores': [response.score]}

def grammar_check(state: EssayInsightState) -> EssayInsightState:
    response = structured_model.invoke(f"Evaluate grammar and give a score: {state['essay_text']}")
    return {'grammar_feedback': response.feedback, 'scores': [response.score]}

def structure_check(state: EssayInsightState) -> EssayInsightState:
    response = structured_model.invoke(f"Evaluate structure and give a score: {state['essay_text']}")
    return {'structured_feedback': response.feedback, 'scores': [response.score]}

def overall_check(state: EssayInsightState) -> EssayInsightState:
    response = structured_model.invoke(
        f"Give short overall feedback based on:\n"
        f"Language: {state['language_feedback']}\n"
        f"Grammar: {state['grammar_feedback']}\n"
        f"Structure: {state['structured_feedback']}"
    )
    avg = sum(state['scores']) / len(state['scores'])
    return {'overall_feedback': response.feedback, 'average_score': avg}

def display_results(state: EssayInsightState) -> None:
    st.markdown("### 📊 Results")
    st.write(f"**Overall Feedback:** {state['overall_feedback']}")
    st.write(f"**Average Score:** {state['average_score']:.2f}")

# Iteration logic (conceptual)
def condition(state: EssayInsightState) -> str:
    """Represents iterative decision logic"""
    return "Approved" if state['average_score'] >= 7 else "Needs Improvement"

# Build graph
graph = StateGraph(EssayInsightState)
graph.add_node("Input Essay", input_essay)
graph.add_node("Language Check", language_check)
graph.add_node("Grammar Check", grammar_check)
graph.add_node("Structure Check", structure_check)
graph.add_node("Overall Check", overall_check)
graph.add_node("Display", display_results)

graph.add_edge(START, "Input Essay")
graph.add_edge("Input Essay", "Language Check")
graph.add_edge("Input Essay", "Grammar Check")
graph.add_edge("Input Essay", "Structure Check")
graph.add_edge("Language Check", "Overall Check")
graph.add_edge("Grammar Check", "Overall Check")
graph.add_edge("Structure Check", "Overall Check")

# Demonstrate conditional iteration logic
graph.add_conditional_edges("Overall Check", condition, {
    "Approved": "Display",
    "Needs Improvement": "Display"  # Instead of looping to Input Essay
})

graph.add_edge("Display", END)
workflow = graph.compile()




st.markdown("## 🧩 Workflow Visualization")





# Streamlit app
st.title("🧠 EssayInsight – Essay Evaluation Tool")
st.write("Evaluates your essay and demonstrates iterative workflow logic using LangGraph.")

essay_input = st.text_area("✍️ Paste your essay here", height=250)

if st.button("Evaluate Essay"):
    if not essay_input.strip():
        st.warning("Please enter an essay first.")
    else:
        result = workflow.invoke({'essay_text': essay_input})

        

        if result['average_score'] < 7:
            st.error("🌀 The system would normally re-run (iterate) until improvement — shown conceptually here.")
        else:
            st.success("✅ Essay approved.")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#ffffff; font-weight:600;'>Made by <b>Manan Ch</b></p>", unsafe_allow_html=True)
