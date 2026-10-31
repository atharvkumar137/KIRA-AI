from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from werkzeug.utils import secure_filename

import os
import ast
import operator
import tempfile
import logging


# ==========================================
# KIRA AI - STATELESS BACKEND
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

client = (
    Groq(
        api_key=GROQ_API_KEY,
        timeout=45,
        max_retries=0
    )
    if GROQ_API_KEY else None
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)

ALLOWED_EXTENSIONS = {
    "txt", "md", "csv", "json", "pdf", "docx"
}


# ==========================================
# KIRA PERSONALITY
# ==========================================

SYSTEM_INSTRUCTION = """
You are KIRA-18, a general-purpose personal AI assistant.

Personality:
- Smart, friendly and practical.
- Speak naturally in English or Hinglish.
- Match the user's language.
- Explain difficult things simply.
- Give step-by-step instructions when useful.
- Be concise but helpful.

IMPORTANT:
- Never invent facts, memories, tools or actions.
- Never claim to have performed an action you did not perform.
- Never claim to be GPT-4 or another unrelated model.
- If asked about your identity, say you are KIRA-18,
  powered by the Groq model configured by the developer.
- You cannot access the user's computer, files, camera,
  microphone or live web unless the application provides
  relevant content or tool results.
- Personal memory is provided by the browser when relevant.
  Use it only to answer the current user's request.
"""


# ==========================================
# WEBSITE
# ==========================================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


# ==========================================
# STATUS
# ==========================================

@app.route("/api/status")
def status():
    return jsonify({
        "configured": client is not None,
        "provider": "Groq",
        "model": MODEL_NAME,
        "storage": "browser-only",
        "features": {
            "memory": True,
            "chat_history": True,
            "voice_browser": True,
            "calculator": True,
            "documents": True,
            "web_search": True,
            "image_analysis": False
        }
    })


# ==========================================
# NEW CHAT ID
# No chat is saved on the server.
# ==========================================

@app.route("/new-chat", methods=["GET", "POST"])
def new_chat():
    import uuid

    return jsonify({
        "chat_id": str(uuid.uuid4()),
        "title": "New Chat"
    })


# ==========================================
# SAFE CALCULATOR
# ==========================================

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg
}


def safe_calculate(expression):
    tree = ast.parse(expression, mode="eval")

    def visit(node, depth=0):
        if depth > 20:
            raise ValueError("Expression is too complex")

        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)

        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
        ):
            return node.value

        if (
            isinstance(node, ast.BinOp)
            and type(node.op) in _BIN_OPS
        ):
            left = visit(node.left, depth + 1)
            right = visit(node.right, depth + 1)

            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent too large")

            result = _BIN_OPS[type(node.op)](left, right)

            if not isinstance(result, (int, float)):
                raise ValueError("Invalid result")

            if abs(result) > 1e100:
                raise ValueError("Result too large")

            return result

        if (
            isinstance(node, ast.UnaryOp)
            and type(node.op) in _UNARY_OPS
        ):
            return _UNARY_OPS[type(node.op)](
                visit(node.operand, depth + 1)
            )

        raise ValueError("Only basic arithmetic is supported")

    return visit(tree)


@app.route("/api/calculate", methods=["POST"])
def calculate():
    data = request.get_json(silent=True) or {}

    try:
        expression = str(data.get("expression", ""))[:200]
        result = safe_calculate(expression)

        return jsonify({"result": result})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ==========================================
# DOCUMENT READER
# Temporary files are deleted after extraction.
# ==========================================

def extract_document(path):
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md", ".csv", ".json"):
        return path.read_text(
            encoding="utf-8",
            errors="replace"
        )[:30000]

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )[:30000]

    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))

        return "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
        )[:30000]

    raise ValueError("Unsupported document type")


@app.route("/api/upload", methods=["POST"])
def upload():
    uploaded = request.files.get("file")

    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Choose a file first"}), 400

    filename = secure_filename(uploaded.filename)

    if (
        "." not in filename
        or filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS
    ):
        return jsonify({
            "error": "Supported formats: TXT, MD, CSV, JSON, PDF and DOCX"
        }), 400

    suffix = Path(filename).suffix.lower()
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            uploaded.save(temporary)

        content = extract_document(temporary_path)

        if not content.strip():
            return jsonify({
                "error": "No readable text found in this document"
            }), 400

        return jsonify({
            "filename": filename,
            "text": content,
            "message": "Document read successfully."
        })

    except ImportError:
        return jsonify({
            "error": "A document reader is missing. Install requirements.txt."
        }), 500

    except Exception:
        app.logger.exception("Document extraction failed")
        return jsonify({
            "error": "Could not read this document."
        }), 400

    finally:
        if temporary_path:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                app.logger.warning("Temporary upload cleanup failed")


# ==========================================
# WEB SEARCH
# ==========================================

@app.route("/api/search", methods=["POST"])
def web_search():
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"error": "Enter a search query"}), 400

    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query[:300]},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for item in soup.select(".result")[:6]:
            link = item.select_one("a.result__a")
            snippet = item.select_one(".result__snippet")

            if not link:
                continue

            url = link.get("href", "")

            if not url.startswith(("https://", "http://")):
                continue

            results.append({
                "title": link.get_text(" ", strip=True),
                "url": url,
                "snippet": (
                    snippet.get_text(" ", strip=True)
                    if snippet else ""
                )
            })

        return jsonify({"results": results})

    except Exception:
        app.logger.exception("Web search failed")
        return jsonify({
            "error": "Web search is temporarily unavailable."
        }), 502


# ==========================================
# GROQ AI
# Server does NOT load or save chat history.
# Browser sends only the conversation needed
# for the current response.
# ==========================================

@app.route("/chat", methods=["POST"])
def chat_route():
    if client is None:
        return jsonify({
            "error": "Groq is not configured. Check the private .env file."
        }), 503

    data = request.get_json(silent=True) or {}

    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    # Memory is stored in the visitor's browser.
    # The browser sends only relevant memory text.
    memory_context = str(
        data.get("memory_context", "")
    )[:8000]

    # Recent conversation is also sent by the browser.
    conversation = data.get("messages", [])

    if not isinstance(conversation, list):
        conversation = []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION
        }
    ]

    if memory_context:
        messages.append({
            "role": "system",
            "content": (
                "Personal memory supplied by this browser. "
                "Use only when relevant:\n" + memory_context
            )
        })

    # Accept only user/assistant messages.
    # Never accept system/developer roles from the browser.
    for item in conversation[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        messages.append({
            "role": role,
            "content": content[:12000]
        })

    # Avoid duplicating the current message if the frontend
    # already included it in the messages array.
    if (
        not messages
        or messages[-1].get("role") != "user"
        or messages[-1].get("content") != message
    ):
        messages.append({
            "role": "user",
            "content": message[:12000]
        })

    tool_context = str(
        data.get("tool_context", "")
    )[:30000]

    if tool_context:
        messages.insert(1, {
            "role": "system",
            "content": "Document/tool context:\n" + tool_context
        })

    try:
        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        answer = result.choices[0].message.content

        if answer and answer.strip():
            return jsonify({"reply": answer.strip()})

        return jsonify({
            "reply": "I couldn't generate a response. Please try again."
        })

    except Exception:
        app.logger.exception("Groq request failed")

        return jsonify({
            "error": "My AI service couldn't complete that request. Please try again."
        }), 502


# ==========================================
# OLD SERVER-SIDE STORAGE ROUTES
# These deliberately do not store personal data.
# New frontend uses browser IndexedDB instead.
# ==========================================

@app.route("/chats")
def old_chats_route():
    return jsonify({"chats": []})


@app.route("/history/<chat_id>")
def old_history_route(chat_id):
    return jsonify({
        "error": "Chat history is stored in your browser."
    }), 404


@app.route("/api/memory", methods=["GET", "POST", "DELETE"])
def old_memory_route():
    return jsonify({
        "error": "Memory is stored in your browser, not on this server."
    }), 410


# ==========================================
# START KIRA
# ==========================================

if __name__ == "__main__":
    print("\n========== KIRA AI ONLINE ==========")
    print("Provider: Groq")
    print("Model:", MODEL_NAME)
    print("API configured:", bool(client))
    print("Personal data storage: Browser only")
    print("Server chat/memory persistence: DISABLED")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )