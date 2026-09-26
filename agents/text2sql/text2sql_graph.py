from typing import TypedDict, Optional, Any
import sys
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langgraph.types import interrupt
import mlflow
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "semantic_layer"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

from models import SpaceConfig
from retriever import detect_space
from registry import get_space
from tools import execute_sql
from langchain_groq import ChatGroq


mlflow.langchain.autolog()


MAX_CLARIFICATION_ATTEMPTS = 2

DB_URI = os.environ["POSTGRES_CONNECTION_STRING"]
pool = ConnectionPool(conninfo=DB_URI, max_size=10, kwargs={"autocommit": True})
checkpointer = PostgresSaver(pool)
checkpointer.setup()

llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.environ["GROQ_API_KEY"])

class Text2SQLState(TypedDict):
    question: str
    conversation_id: str
    space_name: Optional[str]
    space_confidence: Optional[float]
    awaiting_clarification: bool
    space_config: Optional[SpaceConfig]
    grounded_context: Optional[str]
    sql_query: Optional[str]
    validation_error: Optional[str]
    repair_attempts: int
    llm_call_count: int
    sql_result: Optional[Any]
    final_answer: Optional[str]
    clarification_attempts: int


def route_from_start(state: Text2SQLState):
    if state["space_name"] is not None:
        return "skip_classification"
    else:
        return "classify"
    

def classify_space(state: Text2SQLState):
    space_name, space_confidence = detect_space(state['question'])

    if space_name is None:
        awaiting_clarification = True
    else:
        awaiting_clarification = False

    return {
        "space_name": space_name,
        "space_confidence": space_confidence,
        "awaiting_clarification": awaiting_clarification
    }


def route_after_classification(state: Text2SQLState):
    if state['awaiting_clarification']:
        return "clarify"
    else:
        return "proceed"


def retrieve_schema(state: Text2SQLState):

    space_config = get_space(state['space_name'])
    return {"space_config": space_config}


def clarification_node(state: Text2SQLState):
    space_names = ["maintenance", "bakehouse"] 

    answer = interrupt(
        f"I'm not sure which dataset your question relates to — "
        f"did you mean {' or '.join(space_names)}? Could you clarify?"
    )

    combined_question = f"{state['question']} (clarification: {answer})"
    space_name, space_confidence = detect_space(combined_question)

    awaiting_clarification = space_name is None

    return {
        "space_name": space_name,
        "space_confidence": space_confidence,
        "awaiting_clarification": awaiting_clarification,
        "clarification_attempts": state["clarification_attempts"] + 1
    }


def route_after_clarification(state: Text2SQLState):
    if not state["awaiting_clarification"]:
        return "confident"
    elif state["clarification_attempts"] < MAX_CLARIFICATION_ATTEMPTS:
        return "retry"
    else:
        return "give_up"


def give_up_node(state: Text2SQLState):
    return {
        "final_answer": "I wasn't able to determine which dataset you meant — could you try rephrasing your question?"
    }


def ground_data(state: Text2SQLState):
    table = state["space_config"].tables[0]  # start with just the first table for now
    sample_query = f"SELECT * FROM {table.full_name} LIMIT 3"
    sample_rows = execute_sql(sample_query)

    return {"grounded_context": str(sample_rows)}


def generate_sql(state: Text2SQLState):
    space_config = state["space_config"]
    schema_text = "\n".join(
        f"Table {t.full_name}: " + ", ".join(
            f"{c.name} ({c.data_type}) - {c.description}" for c in t.columns
        )
        for t in space_config.tables
    )

    prompt = f"""You are a SQL expert. Write a single Databricks SQL query to answer the question.

        Schema: {schema_text}
        
        Sample data: {state['grounded_context']}
        
        Question: {state['question']}
        
        STRICT RULES:
        - ALWAYS aggregate (SUM, COUNT, AVG, MIN, MAX) with GROUP BY where the question involves comparing, trending, or summarizing data across more than a handful of items.
        - NEVER return raw row-level data (e.g. one row per transaction, per hour, per timestamp) unless the question explicitly asks to "list" or "show individual" records.
        - ALWAYS include a LIMIT clause (LIMIT 20 or fewer) on any query that could return more than 20 rows.
        - If the question is broad or exploratory (e.g. "investigate X", "look into Y"), interpret it as asking for a small number of summary statistics, NOT a full data dump.
        
        Return ONLY the SQL query. No explanation. No markdown formatting. No code fences."""


    response = llm.invoke(prompt)
    sql_query = str(response.content).strip()

    return {
        "sql_query": sql_query,
        "llm_call_count": state["llm_call_count"] + 1
    }


def validate_and_execute_sql(state: Text2SQLState):
    try:
        result = execute_sql(state["sql_query"])
        plain_result = [row.asDict() for row in result]
        return {"sql_result": plain_result, "validation_error": None}
    except Exception as e:
        return {"validation_error": str(e)}
    

def route_after_execution(state: Text2SQLState):
    if state["validation_error"] is None:
        return "success"
    elif state["repair_attempts"] < 2:
        return "repair"
    else:
        return "give_up"
    

def repair_sql(state: Text2SQLState):
    prompt = f"""This SQL query failed:
{state['sql_query']}

Error: {state['validation_error']}

Fix it. Return ONLY the corrected SQL, no explanation."""

    response = llm.invoke(prompt)
    fixed_sql = str(response.content).strip()

    return {
        "sql_query": fixed_sql,
        "repair_attempts": state["repair_attempts"] + 1,
        "llm_call_count": state["llm_call_count"] + 1
    }


def format_answer(state: Text2SQLState):
    if state["validation_error"] is not None:
        return {"final_answer": "I couldn't successfully generate a working query for this question."}

    prompt = f"""Question: {state['question']}
    SQL query used: {state['sql_query']}
    Result: {state['sql_result']}
    
    Write one concise, natural-language sentence answering the question directly, using the actual number(s) from the result. Do not mention SQL or the query itself."""

    response = llm.invoke(prompt)
    summary = str(response.content).strip()

    return {"final_answer": summary}


graph = StateGraph(Text2SQLState)

graph.add_node("classify_space",classify_space)
graph.add_node("retrieve_schema",retrieve_schema)
graph.add_node("clarification_node", clarification_node)
graph.add_node("give_up_node", give_up_node)
graph.add_node("ground_data", ground_data)
graph.add_node("generate_sql", generate_sql)
graph.add_node("validate_and_execute_sql", validate_and_execute_sql)
graph.add_node("repair_sql", repair_sql)
graph.add_node("format_answer", format_answer)

graph.add_conditional_edges(START, route_from_start, {"classify": "classify_space", "skip_classification": "retrieve_schema"})
graph.add_conditional_edges("classify_space",route_after_classification,{"clarify":"clarification_node","proceed":"retrieve_schema"})
graph.add_conditional_edges("clarification_node", route_after_clarification, {"confident": "retrieve_schema", "retry": "clarification_node", "give_up": "give_up_node"})
graph.add_edge("give_up_node", END)
graph.add_edge("retrieve_schema", "ground_data")
graph.add_edge("ground_data", "generate_sql")
graph.add_edge("generate_sql", "validate_and_execute_sql")
graph.add_conditional_edges("validate_and_execute_sql", route_after_execution, {"success": "format_answer", "repair": "repair_sql", "give_up": "format_answer"})
graph.add_edge("repair_sql", "validate_and_execute_sql")
graph.add_edge("format_answer", END)

agent = graph.compile(checkpointer=checkpointer)