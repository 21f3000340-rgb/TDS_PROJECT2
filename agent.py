from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time

from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool, transcribe_audio,
    encode_image_to_base64
)

from typing import TypedDict, Annotated, List
from langchain_core.messages import trim_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000


# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [
    run_code, get_rendered_html, download_file, post_request,
    add_dependencies, ocr_image_tool, transcribe_audio, encode_image_to_base64
]


# ---------------------------------------------------------
# LLM INIT
# ---------------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,
    check_every_n_seconds=1,
    max_bucket_size=4
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY via the correct endpoint.
5. Follow new URLs until done, then output END.

Rules:
- For base64 image encoding: ALWAYS use encode_image_to_base64 (tool).
- Never hallucinate URLs or fields.
- Never shorten endpoints.
- Always inspect server output.
- Never stop early.
- Use tools for HTML, files, OCR, code, or audio as needed.

Always include:
email = {EMAIL}
secret = {SECRET}
"""


# ---------------------------------------------------------
# HANDLE MALFORMED JSON
# ---------------------------------------------------------
def handle_malformed_node(state: AgentState):
    print("⚠️ MALFORMED JSON DETECTED — Asking LLM to retry")
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "SYSTEM ERROR: The last tool call had INVALID JSON.\n"
                    "Rewrite the tool call with correct JSON (escape quotes, newlines).\n"
                    "Try again now."
                )
            }
        ]
    }


# ---------------------------------------------------------
# MAIN AGENT NODE
# ---------------------------------------------------------
def agent_node(state: AgentState):

    now = time.time()
    cur_url = os.getenv("url")
    prev_time = url_time.get(cur_url)
    offset = os.getenv("offset", "0")

    # Timeout logic
    if prev_time:
        prev_time = float(prev_time)
        diff = now - prev_time

        if diff >= 180 or (offset != "0" and (now - float(offset)) > 90):
            print(f"⏳ TIMEOUT ({diff}s) — Forcing wrong submission")

            timeout_msg = HumanMessage(
                content=(
                    "You exceeded the allowed time (180 seconds). "
                    "Immediately submit a WRONG answer using `post_request`."
                )
            )

            result = llm.invoke(state["messages"] + [timeout_msg])
            return {"messages": [result]}

    # Trim message history
    trimmed = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm
    )

    # Ensure HumanMessage exists
    if not any(m.type == "human" for m in trimmed):
        reminder = HumanMessage(content=f"Continue solving the quiz at {cur_url}")
        trimmed.append(reminder)

    print(f"🤖 LLM INVOKE — {len(trimmed)} messages")
    result = llm.invoke(trimmed)

    return {"messages": [result]}


# ---------------------------------------------------------
# ROUTING LOGIC
# ---------------------------------------------------------
def route(state):
    last = state["messages"][-1]

    # Malformed tool call → retry
    meta = getattr(last, "response_metadata", {})
    if meta and meta.get("finish_reason") == "MALFORMED_FUNCTION_CALL":
        return "handle_malformed"

    # Valid tool call
    if getattr(last, "tool_calls", None):
        print("🔧 ROUTE → tools")
        return "tools"

    # END check
    content = getattr(last, "content", "")

    if isinstance(content, str) and content.strip() == "END":
        return END

    if isinstance(content, list) and len(content) and isinstance(content[0], dict):
        if content[0].get("text", "").strip() == "END":
            return END

    print("🔁 ROUTE → agent")
    return "agent"


# ---------------------------------------------------------
# GRAPH DEFINITION
# ---------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")

graph.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed",
        END: END
    }
)

app = graph.compile()


# ---------------------------------------------------------
# RUN AGENT (FINAL FIXED VERSION)
# ---------------------------------------------------------
def run_agent(url: str):
    """
    Called by FastAPI in a background task.
    FastAPI passes only the URL string.
    """

    if not url:
        print("❌ run_agent() received empty URL")
        return

    # Prepare runtime environment
    os.environ["url"] = url
    os.environ["offset"] = "0"
    url_time[url] = time.time()

    print(f"🚀 Agent starting for: {url}")

    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]

    try:
        app.invoke(
            {"messages": initial_messages},
            config={"recursion_limit": RECURSION_LIMIT}
        )

        print("🎉 Agent completed all tasks!")

    except Exception as e:
        print("💥 AGENT CRASHED:", e)
