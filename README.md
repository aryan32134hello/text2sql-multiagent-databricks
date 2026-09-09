\# Text2SQL Multi-Agent System (Databricks)



A config-driven, multi-agent natural-language-to-SQL system built on Databricks Free Edition using LangGraph. Includes a Text2SQL engine with self-correcting SQL generation, and a Root-Cause-Analysis (RCA) agent that decomposes complex questions into sub-questions and investigates by calling the Text2SQL engine as a tool.



\## Architecture



\- \*\*Semantic layer\*\* (`semantic\_layer/`) — typed schema definitions (`models.py`), a config registry loading per-domain YAML files (`registry.py`), and embedding-based space classification using Voyage AI with a margin-based confidence threshold (`retriever.py`).

\- \*\*Text2SQL engine\*\* (`agents/text2sql/`) — a LangGraph state machine: classify space → retrieve schema → ground with sample data → generate SQL → execute → self-repair on failure (max 2 retries) → format answer. Includes a human-in-the-loop clarification flow for ambiguous questions.

\- \*\*RCA engine\*\* (`agents/rca/`) — a second LangGraph state machine: classify space → plan (initial hypothesis + sub-questions) → investigate (calls the Text2SQL engine as a tool for each sub-question) ↔ reflect (bounded loop, max 3 iterations, decides if more investigation is needed) → synthesize (final grounded answer).

\- \*\*Shared tools\*\* (`tools/`) — SQL execution against a Databricks SQL warehouse.



\## Data



\- `bakehouse` space: Databricks' built-in sample retail dataset (`samples.bakehouse`) — sales transactions, customers, franchises.

\- `maintenance` space: \[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), loaded from Kaggle into Unity Catalog.



\## Key design decisions



\- \*\*Embedding-based space routing\*\* (Voyage AI `voyage-3`) rather than keyword matching, with a margin-based threshold (not just an absolute cutoff) to distinguish genuinely ambiguous questions from confident-but-low-scoring ones.

\- \*\*LLM inference via Groq\*\* (`langchain-groq`), chosen after Databricks' pay-per-token Foundation Model APIs proved unavailable on this Free Edition workspace.

\- \*\*RCA reuses the Text2SQL engine as a tool\*\* rather than duplicating SQL-generation logic — a single sub-question is answered by invoking the already-compiled Text2SQL graph directly.

\- \*\*Checkpointed execution\*\* (LangGraph `MemorySaver`) and \*\*MLflow autolog tracing\*\* throughout both graphs.

\- \*\*Guardrails\*\*: bounded repair attempts, bounded clarification retries, bounded RCA investigation iterations.



\## Status



Both the Text2SQL and RCA engines are complete and tested end-to-end. Not yet built: a top-level router/Conversational Agent (for greetings, glossary, and routing between engines) and a Dynamic Agent for parallel multi-space investigations, per the original architecture design.



\## Stack



Databricks Free Edition (Unity Catalog, SQL Warehouse), LangGraph, LangChain, Groq (LLM inference), Voyage AI (embeddings), MLflow (tracing).

