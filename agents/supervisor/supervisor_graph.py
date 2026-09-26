from typing import TypedDict, Optional, Literal
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import os
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
import mlflow

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "semantic_layer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "text2sql"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rca"))
from text2sql_graph import agent as text2sql_agent
from rca_graph import agent as rca_agent


mlflow.langchain.autolog()

class IntentOutput(BaseModel):
    route: Literal["conversational", "text2sql", "rca"]


llm = ChatGroq(model="openai/gpt-oss-20b")
structured_llm_intent = llm.with_structured_output(IntentOutput, method="json_mode")


class SupervisorState(TypedDict):
    question: str
    conversation_id: str
    route: Optional[str]
    final_answer: Optional[str]


def classify_intent(state: SupervisorState):
    prompt = f"""You are classifying a user's message to decide how to handle it.

    Message: {state['question']}
    
    STRICT RULES:
    - Classify as "conversational" if the message is a greeting, small talk, a question about what this system can do, a question about what data or datasets are available, or anything that does NOT require querying actual data.
    - Classify as "text2sql" if the message asks a direct factual question answerable with a single query — a specific number, total, count, or comparison (e.g. "what is total revenue", "how many customers are there").
    - Classify as "rca" if the message asks an investigative "why" question that requires multiple steps of analysis to answer (e.g. "why is revenue lower", "what's causing machine failures to increase").
    - If uncertain between text2sql and rca, prefer "text2sql" — it is simpler and faster; only choose "rca" when the question clearly requires investigation rather than a single lookup.
    
    Respond in JSON format with exactly one key: "route" (must be exactly the string "conversational", "text2sql", or "rca")."""

    result = structured_llm_intent.invoke(prompt)
    return {"route": result.route}


def respond_conversational(state: SupervisorState):
    prompt = f"""You are a helpful assistant for a data analytics system. The system can answer questions about bakery sales/transactions data and industrial machine maintenance data using SQL queries, and can investigate more complex "why" questions.

    Respond naturally and briefly to this message: {state['question']}
    
    If the message asks what you can do, mention you can answer questions about sales/revenue/customers/franchises (bakehouse data) and machine failures/maintenance (maintenance data)."""

    response = llm.invoke(prompt)
    return {"final_answer": str(response.content).strip()}


def call_text2sql(state: SupervisorState):
    sub_state = {
        "question": state["question"],
        "conversation_id": state["conversation_id"],
        "space_name": None,
        "space_confidence": None,
        "awaiting_clarification": False,
        "space_config": None,
        "grounded_context": None,
        "sql_query": None,
        "validation_error": None,
        "repair_attempts": 0,
        "llm_call_count": 0,
        "sql_result": None,
        "final_answer": None,
        "clarification_attempts": 0,
    }

    thread_id = f"{state['conversation_id']}-text2sql"
    result = text2sql_agent.invoke(sub_state, config={"configurable": {"thread_id": thread_id}})

    return {"final_answer": result["final_answer"]}


def call_rca(state: SupervisorState):
    sub_state = {
        "question": state["question"],
        "conversation_id": state["conversation_id"],
        "space_name": None,
        "space_config": None,
        "plan": None,
        "sub_questions": [],
        "findings": [],
        "iteration_count": 0,
        "investigation_complete": False,
        "final_answer": None,
        "clarification_attempts": 0,
    }

    thread_id = f"{state['conversation_id']}-rca"
    result = rca_agent.invoke(sub_state, config={"configurable": {"thread_id": thread_id}})

    return {"final_answer": result["final_answer"]}


def route_from_intent(state: SupervisorState):
    return state["route"]


DB_URI = os.environ["POSTGRES_CONNECTION_STRING"]
pool = ConnectionPool(conninfo=DB_URI, max_size=10, kwargs={"autocommit": True})
checkpointer = PostgresSaver(pool)
checkpointer.setup()

graph = StateGraph(SupervisorState)

graph.add_node("classify_intent", classify_intent)
graph.add_node("respond_conversational", respond_conversational)
graph.add_node("call_text2sql", call_text2sql)
graph.add_node("call_rca", call_rca)


graph.add_edge(START, "classify_intent")
graph.add_conditional_edges("classify_intent", route_from_intent, {"conversational": "respond_conversational", "text2sql": "call_text2sql", "rca": "call_rca"})
graph.add_edge("respond_conversational", END)
graph.add_edge("call_text2sql", END)
graph.add_edge("call_rca", END)

agent = graph.compile(checkpointer=checkpointer)