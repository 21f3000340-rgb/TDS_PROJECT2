from langgraph.graph import StateGraph, END, START
from shared_store import url_time
import time

from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool,
    transcribe_audio, encode_image_to_base64
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


# ----------------------------------------
# STATE
# ----------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [
    run_code, get_rendered_html, download_file, post_request,
    add_dependencies, ocr_image_tool, transcribe_audio,
    encode_image_to_base64
]


# ----------------------------------------
# LLM INIT
# ----------------------------------------
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


# ----------------------------------------
# SYSTEM PROMPT
# ----------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job:
1. Load the quiz page from a given URL.
2. Extract instructions and submit endpoint.
3. Solve tasks EXACTLY.
4. Submit ONLY via the correct endpoint.
5. Follow new URLs until none remain.

Rules:
- ALWAYS use encode_image_to_base64 for base64 encoding.
- NEVER hallucinate URLs or fields.
- NEVER shorten endpoints.
- ALWAYS inspect server output carefully.
- NEVER stop early.
- ALWAYS include:
    email = {EMAIL}
    secret = {SECRET}
"""


# ----------------------------------------
# MALFORMED JSON NODE
# ----------------------------------------
def handle_malformed_node(state: AgentState):
    print("⚠️ Malformed JSON detected — requesting retry.")
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "SYSTEM ERROR: Your last tool call had INVALID JSON.\n"
                    "Rewrite the JSON correctly. Escape quotes/newlines.\n"
                    "Try again now."
                )
            }
        ]
    }


# ----------------------------------------
# MAIN AGENT NODE
# ----------------------------------------
def agent_node(state: AgentState):

    now = time.time()
    cur_url = os.getenv("url")
    prev_time = url_time.get(cur_url)
    offset = os.getenv("offset", "0")

    # Timeout logic
    if prev_time:
        diff = now - float(prev_time)

        if diff >= 180 or (offset != "0" and now - float(offset) > 90):
            print(f"⏳ Timeout ({diff}s) — sending forced wrong submission")

            timeout_msg = HumanMessage(
                content="You exceeded time. Submit a wrong answer via `post_request`."
            )
            result = llm.invoke(state["messages"] + [timeout_msg])
            return {"messages": [result]}

    # Trim messages
    trimmed = trim_messages(
        state["messages"],
        max_tokens=MAX_TOKENS,
        include_system=True,
        start_on="human",
        strategy="last",
        token_counter=llm
    )

    if not any(m.type == "human" for m in trimmed):
        trimmed.append(HumanMessage(content=f"Continue solving URL: {cur_url}"))

    print(f"🤖 LLM INVOKE — {len(trimmed)} messages")
    result = llm.invoke(trimmed)

    return {"messages": [result]}


# ----------------------------------------
# ROUTER
# ----------------------------------------
def route(state):
    last = state["messages"][-1]

    meta = getattr(last, "response_metadata", {})
    if meta.get("finish_reason") == "MALFORMED_FUNCTION_CALL":
        return "handle_malformed"

    if getattr(last, "tool_calls", None):
        print("🔧 Route → tools")
        return "tools"

    content = getattr(last, "content", "")
    if isinstance(content, str) and content.strip() == "END":
        return END

    if isinstance(content, list) and content and isinstance(content[0], dict):
        if content[0].get("text", "").strip() == "END":
            return END

    print("🔁 Route → agent")
    return "agent"


# ----------------------------------------
# GRAPH SETUP
# ----------------------------------------
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


# ----------------------------------------
# FINAL FIXED run_agent()
# ----------------------------------------
def run_agent(data: dict):
    """
    FastAPI passes the FULL dict: {email, secret, url}.
    """
    if not isinstance(data, dict):
        print("❌ run_agent() expected dict, got:", data)
        return

    url = data.get("url")
    if not url:
        print("❌ URL missing in run_agent payload:", data)
        return

    os.environ["url"] = str(url)
    os.environ["offset"] = "0"
    url_time[str(url)] = time.time()

    print(f"🚀 Agent starting for: {url}")

    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(url)}
    ]

    try:
        app.invoke(
            {"messages": initial_messages},
            config={"recursion_limit": RECURSION_LIMIT}
        )
        print("🎉 Agent completed all tasks!")

    except Exception as e:
        print("💥 Agent crashed:", e)
