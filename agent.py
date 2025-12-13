import os
import time
from typing import TypedDict, Annotated, List

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model

from tools import (
    get_rendered_html, download_file, post_request,
    run_code, add_dependencies, ocr_image_tool,
    transcribe_audio, encode_image_to_base64
)

from shared_store import url_time

# Load env values
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

MAX_TOKENS = 60000
RECURSION_LIMIT = 5000

# -------------------------------------
# STATE
# -------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
]

# -------------------------------------
# LLM INIT
# -------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4/60,  # slow & safe
    check_every_n_seconds=1,
    max_bucket_size=4
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)

# -------------------------------------
# SYSTEM PROMPT
# -------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your responsibilities:
1. Load each quiz page via URL.
2. Extract instructions and submit endpoint.
3. Solve tasks EXACTLY.
4. Submit answers ONLY through the correct endpoint.
5. Move to next URL until no more remain, then say END.

Rules:
- For base64 image encoding ALWAYS use encode_image_to_base64 tool.
- Never hallucinate endpoints or URLs.
- Never shorten submit URLs.
- Inspect server JSON carefully.
- Use tools for HTML rendering, file downloads, OCR, audio, or code execution.
- Never stop early.

Always include:
email = {EMAIL}
secret = {SECRET}
"""

# -------------------------------------
# MALFORMED JSON NODE
# -------------------------------------
def handle_malformed_node(state: AgentState):
    print("⚠️ MALFORMED JSON detected — asking LLM to fix.")
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Your last tool call used invalid JSON. "
                    "Rewrite the JSON correctly. Escape quotes and newlines. Try again."
                )
            }
        ]
    }

# -------------------------------------
# MAIN AGENT NODE
# -------------------------------------
def agent_node(state: AgentState):

    current_url = os.getenv("url")
    now = time.time()
    prev = url_time.get(current_url)
    offset = os.getenv("offset", "0")

    # --- TIMEOUT LOGIC ---
    if prev:
        diff = now - prev
        if diff >= 180 or (offset != "0" and (now - float(offset)) > 90):
            print(f"⏳ Timeout reached ({diff}s). Sending WRONG answer fallback.")

            fail_msg = HumanMessage(content=(
                "Timeout exceeded 180 seconds. "
                "Immediately call post_request with a WRONG answer."
            ))

            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}

    trimmed = trim_messages(
        messages=state["messages"],
        max_tokens=MAX_TOKENS,
        strategy="last",
        include_system=True,
        start_on="human",
        token_counter=llm
    )

    if not any(m.type == "human" for m in trimmed):
        print("⚠️ No human message found — injecting URL reminder.")
        reminder = HumanMessage(content=f"Continue solving quiz at {current_url}")
        trimmed.append(reminder)

    print(f"🤖 LLM INVOKE — {len(trimmed)} messages")
    result = llm.invoke(trimmed)
    return {"messages": [result]}

# -------------------------------------
# ROUTING LOGIC
# -------------------------------------
def route(state: AgentState):
    last = state["messages"][-1]

    # malformed function call
    meta = getattr(last, "response_metadata", {})
    if meta.get("finish_reason") == "MALFORMED_FUNCTION_CALL":
        return "handle_malformed"

    # valid tool call
    if getattr(last, "tool_calls", None):
        print("🔧 Routing → tools")
        return "tools"

    # END check
    content = last.content
    if isinstance(content, str) and content.strip() == "END":
        print("🏁 END detected.")
        return END

    return "agent"

# -------------------------------------
# BUILD GRAPH
# -------------------------------------
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

# -------------------------------------
# RUN AGENT ENTRYPOINT
# -------------------------------------
def run_agent(data: dict):
    """
    data is the full JSON received:
    {
        "email": "...",
        "secret": "...",
        "url": "..."
    }
    """

    url = data.get("url", "")
    email = data.get("email", "")
    secret = data.get("secret", "")

    # Store in environment for entire agent
    os.environ["url"] = url
    os.environ["EMAIL"] = email
    os.environ["SECRET"] = secret
    os.environ["offset"] = "0"

    # Reset timers
    url_time.clear()
    url_time[url] = time.time()

    print(f"🚀 Starting agent for {url}")

    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url}
    ]

    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT}
    )

    print("🎉 Agent finished all tasks.")
