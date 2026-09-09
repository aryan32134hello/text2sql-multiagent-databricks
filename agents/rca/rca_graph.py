from typing import TypedDict, Optional, Any, List, Dict
import sys
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
import mlflow
from langchain_groq import ChatGroq
import json
from pydantic import BaseModel


sys.path.append("/Workspace/Users/aryan32134@gmail.com/text2sql_prototype/semantic_layer")
sys.path.append("/Workspace/Users/aryan32134@gmail.com/text2sql_prototype/agents/text2sql")


from models import SpaceConfig
from retriever import detect_space
from registry import get_space
from text2sql_graph import agent as text2sql_agent


mlflow.langchain.autolog()

MAX_CLARIFICATION_ATTEMPTS = 2
MAX_ITERATIONS = 3


class PlanOutput(BaseModel):
    plan: str
    sub_questions: List[str]

class ReflectOutput(BaseModel):
    investigation_complete: bool
    reasoning: str
    next_sub_questions: List[str]

class SynthesisOutput(BaseModel):
    final_answer: str
    
llm = ChatGroq(model="openai/gpt-oss-20b")

structured_llm_plan = llm.with_structured_output(PlanOutput, method = "json_mode")
structured_llm_reflect = llm.with_structured_output(ReflectOutput, method="json_mode")
structured_llm_synthesize = llm.with_structured_output(SynthesisOutput, method="json_mode")


class RCAState(TypedDict):
    question: str
    conversation_id: str
    space_name: Optional[str]
    plan: Optional[str]
    sub_questions: List[str]
    findings: List[Dict[str, str]]
    iteration_count: int
    investigation_complete: bool
    final_answer: Optional[str]
    clarification_attempts: int
    space_config: Optional[SpaceConfig]


def detect_spaces(state: RCAState):
    space_name, space_confidence = detect_space(state["question"])
    return {"space_name": space_name}


def route_after_detection(state: RCAState):
    if state["space_name"] is None:
        return "clarify"
    else:
        return "proceed"


def clarification_node(state: RCAState):
    space_names = ["maintenance", "bakehouse"]

    answer = interrupt(
        f"I'm not sure which dataset your investigation relates to — "
        f"did you mean {' or '.join(space_names)}? Could you clarify?"
    )

    combined_question = f"{state['question']} (clarification: {answer})"
    space_name, space_confidence = detect_space(combined_question)

    return {
        "space_name": space_name,
        "clarification_attempts": state["clarification_attempts"] + 1
    }


def route_after_clarification(state: RCAState):
    if state["space_name"] is not None:
        return "confident"
    elif state["clarification_attempts"] < MAX_CLARIFICATION_ATTEMPTS:
        return "retry"
    else:
        return "give_up"


def give_up_node(state: RCAState):
    return {"final_answer": "I couldn't determine which dataset this investigation relates to."}


def retrieve_schema(state: RCAState):
    space_config = get_space(state["space_name"])
    return {"space_config": space_config}


def plan(state: RCAState):
    space_config = state["space_config"]
    schema_text = "\n".join(
        f"Table {t.full_name}: " + ", ".join(
            f"{c.name} ({c.data_type}) - {c.description}" for c in t.columns
        )
        for t in space_config.tables
    )

    prompt = f"""You are investigating a question that may require multiple steps to answer.

    Schema:
    {schema_text}
    
    Question: {state['question']}
    
    STRICT RULES:
    - The plan must be 1-2 sentences stating a specific, testable hypothesis — not a vague restatement of the question.
    - Generate EXACTLY 2 sub-questions, no more, no fewer.
    - Each sub-question MUST be plain English, phrased as a question a person would ask — NOT SQL, NOT a description of a query, NOT column names or table names. It will be passed to a separate system that converts natural-language questions into SQL itself, so writing SQL here is wrong.
    - Each sub-question must be specific enough to answer with a single aggregated query (e.g. asking for totals, averages, or comparisons) — avoid broad or open-ended     sub-questions that would require dumping raw row-level data.
    - Each sub-question must use only concepts and columns that actually exist in the schema above. Do not reference data that isn't described in the schema.
    - The two sub-questions must investigate DIFFERENT aspects of the problem, not the same aspect worded differently. For example, "what is the total revenue" and "how much did customers pay in total" are THE SAME question with different wording — this is NOT allowed. Instead, the second sub-question should investigate a different dimension entirely, such as breaking the first finding down by category, time, location, or comparing it against a related metric.

    
    Respond in JSON format with exactly these two keys: "plan" (a string) and "sub_questions" (a list of exactly 2 plain-English question strings)."""

    result = structured_llm_plan.invoke(prompt)

    return {
        "plan": result.plan,
        "sub_questions": result.sub_questions
    }


def investigate(state: RCAState):
    new_findings = []

    for i, sub_question in enumerate(state["sub_questions"]):
        sub_state = {
            "question": sub_question,
            "conversation_id": state["conversation_id"],
            "space_name": state["space_name"],
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

        sub_thread_id = f"{state['conversation_id']}-sub-{state['iteration_count']}-{i}"
        result = text2sql_agent.invoke(sub_state, config={"configurable": {"thread_id": sub_thread_id}})

        new_findings.append({
            "sub_question": sub_question,
            "answer": result["final_answer"]
        })

    return {
        "findings": state["findings"] + new_findings,
        "iteration_count": state["iteration_count"] + 1
    }


def reflect(state: RCAState):
    findings_text = "\n\n".join(
        f"Q: {f['sub_question']}\nA: {f['answer']}" for f in state["findings"]
    )

    prompt = f"""You are reviewing an investigation into this question: {state['question']}

    Investigation plan: {state['plan']}
    
    Findings so far:
    {findings_text}
    
    STRICT RULES:
    - Set investigation_complete to true ONLY if the findings directly and specifically answer the original question with concrete numbers or facts. Do NOT mark it complete just because sub-questions were answered — the answers must actually resolve the original question.
    - If any finding looks incomplete, empty, contradictory, or doesn't match what its sub-question asked for, treat the investigation as NOT complete and generate a sub-question to clarify or fix it.
    - If you mark investigation_complete as false, generate exactly 1-2 new plain-English sub-questions (NOT SQL queries) that address the specific gap you identified — do not repeat a sub-question that was already asked.
    - If investigation_complete is true, next_sub_questions MUST be an empty list.
    - Base your decision only on the findings shown above. Do not assume data exists that isn't shown.
    
    Respond in JSON format with exactly these three keys: "investigation_complete" (true or false — boolean, not a string), "reasoning" (a string explaining specifically what evidence supports or undermines completeness), "next_sub_questions" (a list of plain-English question strings, empty if complete)."""

    result = structured_llm_reflect.invoke(prompt)

    return {
        "investigation_complete": result.investigation_complete,
        "sub_questions": result.next_sub_questions
    }


def route_after_reflect(state: RCAState):
    if state["investigation_complete"]:
        return "synthesize"
    elif state["iteration_count"] < MAX_ITERATIONS:
        return "continue"
    else:
        return "synthesize"
    

def synthesize(state: RCAState):
    findings_text = "\n\n".join(
        f"Q: {f['sub_question']}\nA: {f['answer']}" for f in state["findings"]
    )

    confidence_note = (
        "The investigation reached a confident conclusion."
        if state["investigation_complete"]
        else "The investigation reached its iteration limit before full confidence was reached — answer with the best available evidence, and note any remaining uncertainty."
    )

    prompt = f"""You are writing the final answer to this investigation.

    Original question: {state['question']}
    
    Investigation plan: {state['plan']}
    
    Findings:
    {findings_text}
    
    {confidence_note}
    
    STRICT RULES:
    - Write a direct, concrete answer to the original question — lead with the conclusion, not a summary of the process.
    - Cite specific numbers and facts from the findings above. Do not state anything not supported by the findings.
    - If the findings are incomplete or uncertain, say so plainly rather than overstating confidence.
    - Keep the answer concise — a few sentences, not a report.
    - Respond in JSON format with exactly one key: "final_answer" (a string)."""

    result = structured_llm_synthesize.invoke(prompt)

    return {"final_answer": result.final_answer}


checkpointer = MemorySaver()
graph = StateGraph(RCAState)

graph.add_node("detect_spaces", detect_spaces)
graph.add_node("clarification_node", clarification_node)
graph.add_node("give_up_node", give_up_node)
graph.add_node("retrieve_schema", retrieve_schema)
graph.add_node("plan", plan)
graph.add_node("investigate", investigate)
graph.add_node("reflect", reflect)
graph.add_node("synthesize", synthesize)


graph.add_edge(START, "detect_spaces")
graph.add_conditional_edges("detect_spaces", route_after_detection, {"clarify": "clarification_node", "proceed": "retrieve_schema"})
graph.add_conditional_edges("clarification_node", route_after_clarification, {"confident": "retrieve_schema", "retry": "clarification_node", "give_up": "give_up_node"})
graph.add_edge("give_up_node", END)
graph.add_edge("retrieve_schema", "plan")
graph.add_edge("plan", "investigate")
graph.add_edge("investigate", "reflect")
graph.add_conditional_edges("reflect", route_after_reflect, {"continue": "investigate", "synthesize": "synthesize"})
graph.add_edge("synthesize", END)

agent = graph.compile(checkpointer=checkpointer)