import os
import json
import time
from flask import Flask, render_template, request, Response, stream_with_context
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("VENICE_API_KEY"), base_url="https://api.venice.ai/api/v1")

app = Flask(__name__)

# In-memory session data
session_data = {"file_content": None, "filename": None, "task_id": None, "results": {}, "selected_model": None}

# Ensure logs/sunshine directory exists
LOG_DIR = "logs/sunshine"
os.makedirs(LOG_DIR, exist_ok=True)

# Load models from models.json
def load_models():
    with open("models.json", "r", encoding="utf-8") as f:
        return json.load(f)["models"]

# Get model details by ID
def get_model_info(model_id, models):
    for model in models:
        if model["id"] == model_id:
            return model
    return None

def estimate_tokens(text):
    # Rough estimate: 1 token ≈ 4 characters
    return len(text) // 4

def process_chunk(chunk, prompt, model_id, chunk_num, total_chunks, task_id, max_context_tokens):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    request_data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Potato {chunk_num}/{total_chunks}:\n{chunk}"}
        ],
        "temperature": 0.3,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timestamp": datetime.now().isoformat()  # For logging only
    }
    request_file = os.path.join(LOG_DIR, f"{timestamp}_venice_request.json")
    with open(request_file, "w", encoding="utf-8") as f:
        json.dump(request_data, f, indent=2)
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=request_data["messages"],
            temperature=0.3,
            stream=True,
            stream_options={"include_usage": True}
        )
        full_response = ""
        usage = None
        first_chunk = True
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                if first_chunk:
                    yield f"Potato {chunk_num}: Growing...\n{content}"
                    first_chunk = False
                else:
                    yield content
            if hasattr(chunk, "usage") and chunk.usage:  # Final chunk with usage
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens
                }
        yield f"\nPotato {chunk_num}: Done\n\n"
        
        response_data = {
            "id": chunk.id if hasattr(chunk, "id") else "unknown",
            "object": chunk.object if hasattr(chunk, "object") else "chat.completion.chunk",
            "created": chunk.created if hasattr(chunk, "created") else int(time.time()),
            "model": model_id,
            "choices": [{"message": {"role": "assistant", "content": full_response}}],
            "usage": usage or {"prompt_tokens": "unknown", "completion_tokens": "unknown", "total_tokens": "unknown"},
            "timestamp": datetime.now().isoformat()
        }
        response_file = os.path.join(LOG_DIR, f"{timestamp}_venice_response.json")
        with open(response_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2)
    except Exception as e:
        error_msg = f"Potato {chunk_num}: Error - {str(e)}\n\n"
        error_data = {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        error_file = os.path.join(LOG_DIR, f"{timestamp}_venice_error.json")
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2)
        yield error_msg

@app.route("/", methods=["GET"])
def index():
    models = [m for m in load_models() if m.get("model_extra", {}).get("type") == "text"]
    model_ids = [m["id"] for m in models]
    default_model = "mistral-31-24b" if "mistral-31-24b" in model_ids else "llama-3.2-3b"
    filename = session_data["filename"] if session_data["filename"] else "No file chosen"
    return render_template("index.html", models=model_ids, default_model=default_model, 
                         estimate=None, filename=filename)

@app.route("/estimate", methods=["POST"])
def estimate():
    models = [m for m in load_models() if m.get("model_extra", {}).get("type") == "text"]
    model_ids = [m["id"] for m in models]
    default_model = "mistral-31-24b" if "mistral-31-24b" in model_ids else "llama-3.2-3b"
    
    file = request.files.get("file")
    prompt = request.form.get("prompt", "")
    model_id = request.form.get("model", default_model)
    session_data["selected_model"] = model_id  # Store selected model
    
    if file:
        session_data["file_content"] = file.read().decode("utf-8")
        session_data["filename"] = file.filename
    
    chat_text = session_data["file_content"]
    if not chat_text:
        return render_template("index.html", models=model_ids, default_model=default_model, 
                             estimate="No file uploaded yet", filename="No file chosen")
    
    model_info = get_model_info(model_id, models)
    if not model_info:
        return render_template("index.html", models=model_ids, default_model=default_model, 
                             estimate="Invalid model", filename=session_data["filename"])
    
    file_size = len(chat_text)
    prompt_tokens = estimate_tokens(prompt)
    max_context_tokens = model_info["context_tokens"]
    chunk_size = max_context_tokens - (prompt_tokens + 150)  # Account for prompt and buffer
    num_chunks = (file_size + chunk_size - 1) // chunk_size
    
    total_tokens = (file_size // 4) + (num_chunks * (prompt_tokens + 150))  # Rough estimate including prompt
    cost_per_token = 0.0001  # Placeholder
    estimated_cost = total_tokens * cost_per_token
    estimate = f"We’ll plant {num_chunks} potato{'es' if num_chunks != 1 else ''} costing ~${estimated_cost:.2f}"
    
    return render_template("index.html", models=model_ids, default_model=model_id, 
                         estimate=estimate, filename=session_data["filename"], prompt=prompt)

@app.route("/process", methods=["POST"])
def process():
    models = [m for m in load_models() if m.get("model_extra", {}).get("type") == "text"]
    model_ids = [m["id"] for m in models]
    
    file = request.files.get("file")
    prompt = request.form.get("prompt", "")
    model_id = request.form.get("model", session_data.get("selected_model", "mistral-31-24b"))
    
    if file:
        session_data["file_content"] = file.read().decode("utf-8")
        session_data["filename"] = file.filename
    
    chat_text = session_data["file_content"]
    if not chat_text:
        return "No file uploaded yet", 400
    
    model_info = get_model_info(model_id, models)
    if not model_info:
        return "Invalid model", 400
    
    file_size = len(chat_text)
    prompt_tokens = estimate_tokens(prompt)
    max_context_tokens = model_info["context_tokens"]
    chunk_size = max_context_tokens - (prompt_tokens + 150)  # Adjust for prompt and buffer
    num_chunks = (file_size + chunk_size - 1) // chunk_size
    
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data["task_id"] = task_id
    session_data["results"][task_id] = {"status": "running", "output": ""}
    
    def generate():
        output = f"Starting to plant {num_chunks} potato{'es' if num_chunks != 1 else ''}...\n\n"
        yield output
        session_data["results"][task_id]["output"] += output
        chunks = [chat_text[i:i+chunk_size] for i in range(0, len(chat_text), chunk_size)]
        for i, chunk in enumerate(chunks, 1):
            output = f"Planting potato {i} of {num_chunks}...\n"
            yield output
            session_data["results"][task_id]["output"] += output
            for streamed_output in process_chunk(chunk, prompt, model_id, i, num_chunks, task_id, max_context_tokens):
                yield streamed_output
                session_data["results"][task_id]["output"] += streamed_output
            time.sleep(0.1)  # Small delay for streaming effect
        output = f"\nMashing complete! Task ID: {task_id}\n"
        yield output
        session_data["results"][task_id]["output"] += output
        session_data["results"][task_id]["status"] = "complete"
    
    return Response(stream_with_context(generate()), mimetype="text/plain")

@app.route("/download")
def download():
    task_id = session_data.get("task_id")
    if not task_id or task_id not in session_data["results"] or session_data["results"][task_id]["status"] != "complete":
        return "No completed result available", 404
    result = session_data["results"][task_id]["output"]
    timestamp = task_id
    return Response(result, mimetype="text/plain", 
                   headers={"Content-Disposition": f"attachment;filename=Mashed_Venetian_Potato_{timestamp}.txt"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)