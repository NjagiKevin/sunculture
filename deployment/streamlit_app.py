import sys
import sqlite3
import sqlparse
from pathlib import Path
from datetime import datetime
from typing import Literal, TypedDict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data import DataLoader, DataPreprocessor

load_dotenv(find_dotenv())
llm = ChatOpenAI(model="gpt-4o", temperature=0)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Senior_Data_Scientist_Assessment_Data.xlsx"
DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DATA_DB = str(DB_DIR / "data.db")
HISTORY_DB = str(DB_DIR / "chat_history.db")

HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    sql TEXT,
    answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

SQL_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a SQLite expert. Given a schema and question, return only the SQL query. No markdown, no explanation, no commentary. Even if the question seems unanswerable with the given schema, write your best SQL query — never return anything other than a SQL query."),
    ("human", "Schema:\n{schema}\n\nQuestion: {question}\n\n{error_context}"),
])

INTERPRET_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a business analyst. Given a question and query result, answer in plain English with a business insight."),
    ("human", "Question: {question}\n\nQuery result:\n{result}\n\nAnswer:"),
])


class AgentState(TypedDict):
    question: str
    schema: str
    sql: str
    sql_valid: bool
    error: str
    result: Any
    answer: str
    retries: int


def generate_sql(state: AgentState) -> dict:
    if state.get("retries", 0) >= 2:
        return {"sql": "", "sql_valid": False, "error": "Max retries exceeded"}
    error_ctx = state.get("error", "")
    context = f"Previous attempt failed: {error_ctx}. Fix the SQL query." if error_ctx else ""
    prompt = SQL_GEN_PROMPT.invoke({
        "schema": state["schema"],
        "question": state["question"],
        "error_context": context,
    })
    response = llm.invoke(prompt)
    sql = response.content.strip().removeprefix("```sql").removesuffix("```").strip()
    return {"sql": sql, "retries": state.get("retries", 0) + 1}


def validate_sql(state: AgentState) -> dict:
    sql = state["sql"].strip().lower()
    forbidden = ["drop", "alter", "create", "insert", "update", "delete", "truncate", "pragma"]
    if any(kw in sql for kw in forbidden):
        return {"sql_valid": False, "error": "Destructive SQL detected"}
    stripped = state["sql"].strip().lower()
    if not stripped.startswith("select") and not stripped.startswith("with"):
        return {"sql_valid": False, "error": "Query must start with SELECT or WITH"}
    parsed = sqlparse.parse(state["sql"])
    if not parsed or not parsed[0].tokens:
        return {"sql_valid": False, "error": "Empty or unparseable SQL"}
    return {"sql_valid": True, "error": ""}


def route_validation(state: AgentState) -> Literal["execute_sql", "generate_sql"]:
    if state["sql_valid"]:
        return "execute_sql"
    if state.get("retries", 0) >= 2:
        return "execute_sql"
    return "generate_sql"


def execute_sql(state: AgentState) -> dict:
    if state.get("error"):
        return {"result": None}
    try:
        result = pd.read_sql(state["sql"], _DB_CONN)
        return {"result": result, "error": ""}
    except Exception as e:
        return {"result": None, "error": str(e)}


def interpret_result(state: AgentState) -> dict:
    if state.get("error"):
        return {"answer": f"I encountered an error: {state['error']}"}
    result = state["result"]
    if isinstance(result, pd.DataFrame):
        if len(result) > 50:
            result_text = result.head(50).to_string(index=False)
            result_text += f"\n... and {len(result) - 50} more rows (truncated)"
        else:
            result_text = result.to_string(index=False)
    else:
        result_text = str(result)
    prompt = INTERPRET_PROMPT.invoke({"question": state["question"], "result": result_text})
    response = llm.invoke(prompt)
    return {"answer": response.content.strip()}


_DB_CONN: sqlite3.Connection | None = None


def build_graph(conn: sqlite3.Connection) -> Any:
    global _DB_CONN
    _DB_CONN = conn
    builder = StateGraph(AgentState)
    builder.add_node("generate_sql", generate_sql)
    builder.add_node("validate_sql", validate_sql)
    builder.add_node("execute_sql", execute_sql)
    builder.add_node("interpret_result", interpret_result)
    builder.add_edge(START, "generate_sql")
    builder.add_edge("generate_sql", "validate_sql")
    builder.add_conditional_edges("validate_sql", route_validation, {
        "execute_sql": "execute_sql",
        "generate_sql": "generate_sql",
    })
    builder.add_edge("execute_sql", "interpret_result")
    builder.add_edge("interpret_result", END)
    return builder.compile()


@st.cache_resource
def init_db():
    conn = sqlite3.connect(str(DATA_DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    if len(tables) == 0:
        loader = DataLoader(DATA_PATH)
        raw = loader.load_all()
        preprocessor = DataPreprocessor(raw)
        cleaned = preprocessor.clean_all()
        for name, df in cleaned.items():
            df.to_sql(name.lower(), conn, if_exists="replace", index=False)

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    schema = ""
    for (tname,) in tables:
        tname = tname.lower()
        df = pd.read_sql(f"SELECT * FROM \"{tname}\" LIMIT 1", conn)
        cols = ", ".join([f"{c} ({dtype})" for c, dtype in df.dtypes.items()])
        schema += f"Table: {tname} | Columns: {cols}\n"
        for col in df.select_dtypes(include="object").columns[:3]:
            vals = pd.read_sql(f"SELECT DISTINCT \"{col}\" FROM \"{tname}\" WHERE \"{col}\" IS NOT NULL LIMIT 5", conn)
            schema += f"  Sample {col} values: {vals[col].tolist()}\n"

    hist = sqlite3.connect(str(HISTORY_DB), check_same_thread=False)
    hist.row_factory = sqlite3.Row
    hist.execute(HISTORY_DDL)
    hist.commit()
    return conn, schema, build_graph(conn), hist


def load_history(hist: sqlite3.Connection) -> list[dict]:
    rows = hist.execute(
        "SELECT question, sql, answer FROM conversations ORDER BY id DESC LIMIT 50"
    ).fetchall()
    return [
        {"question": r["question"], "sql": r["sql"], "answer": r["answer"], "result": None}
        for r in reversed(rows)
    ]


def save_to_history(hist: sqlite3.Connection, question: str, sql: str, answer: str):
    hist.execute(
        "INSERT INTO conversations (question, sql, answer) VALUES (?, ?, ?)",
        (question, sql, answer),
    )
    hist.commit()


def suggest_questions(context: str) -> list[str]:
    prompt = f"""Based on this conversation about a solar energy company's data:\n{context}\n
Suggest 3 follow-up questions the user could ask next. Return as a numbered list, no explanation."""
    response = llm.invoke(prompt)
    content = response.content.strip()
    questions = [
        line.split(". ", 1)[1] if ". " in line else line
        for line in content.split("\n")
        if line.strip() and line[0].isdigit()
    ]
    return questions[:3]


def should_plot(result: pd.DataFrame) -> bool:
    if not isinstance(result, pd.DataFrame) or result.empty:
        return False
    num_cols = result.select_dtypes(include="number").columns
    return len(num_cols) >= 1 and len(result) > 1 and len(result) <= 30


def ask_suggestion(q: str):
    st.session_state.suggestion = q


st.set_page_config(page_title="SunCulture Data Assistant", layout="wide")
st.title("SunCulture Data Assistant")
st.markdown("Ask questions about your customer data in plain English.")

conn, schema, graph, hist = init_db()

if "history" not in st.session_state:
    st.session_state.history = load_history(hist)
if "suggestion" not in st.session_state:
    st.session_state.suggestion = ""

question = st.text_input(
    "What would you like to know?",
    value=st.session_state.suggestion,
    placeholder="e.g. Which region has the most customers with accounts in arrears?",
)

if st.session_state.suggestion:
    question = st.session_state.suggestion

col1, col2 = st.columns([5, 1])
with col2:
    if st.button("Clear history"):
        hist.execute("DELETE FROM conversations")
        hist.commit()
        st.session_state.history = []
        st.rerun()

if question:
    with st.spinner("Thinking..."):
        result = graph.invoke({
            "question": question,
            "schema": schema,
            "sql": "",
            "sql_valid": False,
            "error": "",
            "result": None,
            "answer": "",
            "retries": 0,
            "conn": conn,
        })

    save_to_history(hist, result["question"], result["sql"], result["answer"])
    st.session_state.history.append({
        "question": result["question"],
        "sql": result["sql"],
        "result": result["result"],
        "answer": result["answer"],
    })
    st.session_state.suggestion = ""

for i, entry in enumerate(reversed(st.session_state.history)):
    with st.expander(f"Q: {entry['question']}", expanded=(i == 0)):
        col_a, col_m = st.columns([3, 1])
        with col_a:
            st.markdown(f"**Answer:** {entry['answer']}")
        with col_m:
            if isinstance(entry["result"], pd.DataFrame) and not entry["result"].empty:
                csv = entry["result"].to_csv(index=False)
                st.download_button("CSV", csv, f"result_{i}.csv", "text/csv")

        with st.expander("Show SQL & raw result"):
            st.code(entry["sql"], language="sql")
            if isinstance(entry["result"], pd.DataFrame) and not entry["result"].empty:
                st.dataframe(entry["result"], use_container_width=True)

        if isinstance(entry["result"], pd.DataFrame) and should_plot(entry["result"]):
            num_cols = entry["result"].select_dtypes(include="number").columns
            cat_cols = entry["result"].select_dtypes(exclude="number").columns
            if len(num_cols) == 1:
                idx = cat_cols[0] if len(cat_cols) > 0 else entry["result"].index
                st.bar_chart(entry["result"].set_index(idx)[num_cols[0]])
            else:
                st.bar_chart(entry["result"].select_dtypes(include="number"))

        if i == 0:
            context = "\n".join([
                f"Q: {e['question']}\nA: {e['answer']}"
                for e in st.session_state.history[-3:]
            ])
            suggestions = suggest_questions(context)
            if suggestions:
                st.markdown("**Suggested next questions:**")
                for sg in suggestions:
                    st.button(sg, on_click=ask_suggestion, args=(sg,))
