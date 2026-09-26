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

from supervisor_graph import agent as supervisor_agent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://text2sql-multiagent-databricks.vercel.app"],  # tighten to your real Vercel domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    conversation_id: str

@app.post("/ask")
def ask(request: AskRequest):
    result = supervisor_agent.invoke(
        {
            "question": request.question,
            "conversation_id": request.conversation_id,
            "route": None,
            "final_answer": None,
        },
        config={"configurable": {"thread_id": request.conversation_id}}
    )
    return {"answer": result["final_answer"], "route": result["route"]}

@app.get("/health")
def health():
    return {"status": "ok"}