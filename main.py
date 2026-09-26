import os
from dotenv import load_dotenv

# Load .env for local runs — on Render, real environment variables are already
# set in the dashboard, so this call just does nothing there (no .env file exists).
load_dotenv()

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "agents", "supervisor")))

import mlflow
mlflow.set_tracking_uri("databricks")  # requires DATABRICKS_HOST and DATABRICKS_TOKEN in env

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from psycopg_pool import ConnectionPool

from supervisor_graph import agent as supervisor_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://text2sql-multiagent-databricks.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Separate connection pool, just for the conversation log (distinct from the graphs' checkpointer pool) ---
log_pool = ConnectionPool(conninfo=os.environ["POSTGRES_CONNECTION_STRING"], max_size=5, kwargs={"autocommit": True})

def init_conversation_log():
    with log_pool.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_log (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                route TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            )
        """)

init_conversation_log()


class AskRequest(BaseModel):
    question: str
    conversation_id: str


@app.post("/ask")
def ask(request: AskRequest):
    result = supervisor_agent.invoke(...)

    with log_pool.connection() as conn:
        conn.execute(
            "INSERT INTO conversation_log (conversation_id, question, answer, route) VALUES (%s, %s, %s, %s)",
            (request.conversation_id, request.question, result["final_answer"], result["route"])
        )

    return {
        "answer": result["final_answer"],
        "route": result["route"],
        "sql_query": result.get("sql_query"),
        "sql_result": result.get("sql_result"),
    }


@app.get("/conversations")
def list_conversations():
    with log_pool.connection() as conn:
        rows = conn.execute("""
            SELECT conversation_id, MIN(question) as title, MIN(created_at) as started
            FROM conversation_log
            GROUP BY conversation_id
            ORDER BY started DESC
        """).fetchall()
    return [{"conversation_id": r[0], "title": r[1][:50], "started": r[2].isoformat()} for r in rows]


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    with log_pool.connection() as conn:
        rows = conn.execute(
            "SELECT question, answer, route FROM conversation_log WHERE conversation_id = %s ORDER BY created_at",
            (conversation_id,)
        ).fetchall()
    return [{"question": r[0], "answer": r[1], "route": r[2]} for r in rows]


@app.get("/health")
def health():
    return {"status": "ok"}