import os
import json
import time
from flask import Flask, render_template, request, Response, stream_with_context
from openai import OpenAI, APIConnectionError
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
env_api_key = os.getenv("VENICE_API_KEY")
if not env_api_key:
    logger.error("VENICE_API_KEY not found in environment variables")
print(f"Loaded VENICE_API_KEY: {'[hidden]' if env_api_key else 'Not found'}")

app = Flask(__name__)

# Initialize OpenAI client with retries
client = OpenAI(api_key=env_api_key, base_url="https://api.venice.ai/api/v1")

session_data = {
    "file_content": None,
    "filename": None,
    "task_id": None,
    "results": {},
    "selected_model": None,
    "running": False,
    "prompt": "",
    "max_output_tokens": 3337,
    "api_key": env_api_key or ""
}

LOG_DIR = "logs/sunshine"
if not os.getenv("VERCEL"):
    os.makedirs(LOG_DIR, exist_ok=True)

PRICING = {
    "llama-3.2-3b": {"input": 0.75 / 1000000, "output": 3.00 / 1000000},
    "qwen-2.5-coder-32b": {"input": 2.50 / 1000000, "output": 10.00 / 1000000},
    "qwen-2.5-qwq-32b": {"input": 2.50 / 1000000, "output": 10.00 / 1000000},
    "mistral-31-24b": {"input": 2.50 / 1000000, "output": 10.00 / 1000000},
    "llama-3.3-70b": {"input": 3.50 / 1000000, "output": 14.00 / 1000000},
    "dolphin-2.9.2-qwen2-72b": {"input": 3.50 / 1000000, "output": 14.00 / 1000000},
    "deepseek-r1-70b": {"input": 3.50 / 1000000, "output": 14.00 / 1000000},
    "qwen-2.5-vl-72b": {"input": 3.50 / 1000000, "output": 14.00 / 1000000},
    "llama-3.1-405b": {"input": 7.50 / 1000000, "output": 30.00 / 1000000},
    "deepseek-r1-671b": {"input": 17.50 / 1000000, "output": 70.00 / 1000000}
}

def load_models():
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            return [m for m in json.load(f)["models"] if m.get("model_extra", {}).get("type") == "text"]
    except Exception as e:
        logger.error(f"Error loading models.json: {e}")
        return []

def get_model_info(model_id, models):
    for model in models:
        if model["id"] == model_id:
            return model
    return None

def estimate_tokens(text):
    # Conservative estimate: 1 token ≈ 3 characters
    return len(text) // 3

def split_into_chunks(text, max_context_tokens, prompt_tokens, message_overhead, max_output_tokens):
    safety_factor = 0.7
    max_allowed_total = int(max_context_tokens * safety_factor)
    
    lines = text.splitlines()
    header = lines[0] + "\n" if lines else ""
    data_lines = lines[1:] if len(lines) > 1 else []
    
    available_tokens = max_allowed_total - (prompt_tokens + message_overhead + max_output_tokens)
    if available_tokens <= 0:
        raise ValueError(f"Context too small: {max_allowed_total} < {prompt_tokens + message_overhead + max_output_tokens}")
    
    header_tokens = estimate_tokens(header)
    data_tokens = estimate_tokens("\n".join(data_lines))
    chunk_size_tokens = available_tokens - header_tokens
    if chunk_size_tokens <= 0:
        raise ValueError(f"Header too large: {header_tokens} tokens exceed available {available_tokens}")
    
    num_chunks = max(1, (data_tokens + chunk_size_tokens - 1) // chunk_size_tokens)
    chunk_size_lines = len(data_lines) // num_chunks + (1 if len(data_lines) % num_chunks else 0)
    
    chunks = []
    for i in range(0, len(data_lines), chunk_size_lines):
        chunk_data = data_lines[i:i + chunk_size_lines]
        chunk_text = header + "\n".join(chunk_data)
        chunk_tokens = estimate_tokens(chunk_text)
        total_chunk_tokens = prompt_tokens + message_overhead + chunk_tokens + max_output_tokens
        if total_chunk_tokens > max_allowed_total:
            smaller_chunk_size_lines = chunk_size_lines // 2
            if smaller_chunk_size_lines < 1:
                raise ValueError(f"Chunk {i//chunk_size_lines + 1} too large: {total_chunk_tokens} > {max_allowed_total}")
            for j in range(i, min(i + chunk_size_lines, len(data_lines)), smaller_chunk_size_lines):
                smaller_chunk_data = data_lines[j:j + smaller_chunk_size_lines]
                smaller_chunk_text = header + "\n".join(smaller_chunk_data)
                smaller_chunk_tokens = estimate_tokens(smaller_chunk_text)
                total_smaller_chunk_tokens = prompt_tokens + message_overhead + smaller_chunk_tokens + max_output_tokens
                if total_smaller_chunk_tokens > max_allowed_total:
                    raise ValueError(f"Smaller chunk at line {j} exceeds context: {total_smaller_chunk_tokens} > {max_allowed_total}")
                chunks.append(smaller_chunk_text)
                logger.info(f"Chunk {(j-i)//smaller_chunk_size_lines + i//chunk_size_lines + 1}: estimated tokens = {total_smaller_chunk_tokens}")
        else:
            chunks.append(chunk_text)
            logger.info(f"Chunk {i//chunk_size_lines + 1}: estimated tokens = {total_chunk_tokens}")
    
    logger.info(f"DEBUG: Split into {len(chunks)} chunks")
    return chunks, len(chunks)

def process_chunk(chunk, prompt, model_id, chunk_num, total_chunks, task_id, max_context_tokens, max_output_tokens, api_key):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    system_prompt = f"{prompt} Please ensure output does not exceed {max_output_tokens} tokens."
    user_content = f"Potato {chunk_num}/{total_chunks} (size: {estimate_tokens(chunk)} tokens):\n{chunk}"
    request_data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.3,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_completion_tokens": max_output_tokens
    }
    request_file = os.path.join(LOG_DIR, f"{timestamp}_venice_request.json")
    with open(request_file, "w", encoding="utf-8") as f:
        json.dump(request_data, f, indent=2)
    
    full_response = ""
    usage = None
    first_chunk = True
    
    # Retry logic for API connection issues
    for attempt in range(3):
        try:
            response = client.chat.completions.create(**request_data)
            for chunk in response:
                if not session_data["running"]:
                    response_data = {
                        "id": chunk.id if hasattr(chunk, "id") else "unknown",
                        "model": model_id,
                        "choices": [{"message": {"role": "assistant", "content": full_response}}],
                        "usage": usage or {"prompt_tokens": "unknown", "completion_tokens": "unknown", "total_tokens": "unknown"},
                        "timestamp": datetime.now().isoformat(),
                        "status": "stopped"
                    }
                    response_file = os.path.join(LOG_DIR, f"{timestamp}_venice_response.json")
                    with open(response_file, "w", encoding="utf-8") as f:
                        json.dump(response_data, f, indent=2)
                    yield "Process stopped by user.\n"
                    return
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    if first_chunk:
                        yield f"Potato {chunk_num}: Growing...\n{content}"
                        first_chunk = False
                    else:
                        yield content
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens
                    }
            break  # Exit retry loop on success
        except APIConnectionError as e:
            error_msg = f"Potato {chunk_num}: Connection attempt {attempt + 1} failed - {str(e)}"
            logger.error(error_msg)
            if attempt < 2:  # Retry up to 3 times
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            else:
                yield f"Potato {chunk_num}: Failed after 3 attempts - {str(e)}\n\n"
                error_data = {"error": str(e), "timestamp": datetime.now().isoformat()}
                error_file = os.path.join(LOG_DIR, f"{timestamp}_venice_error.json")
                with open(error_file, "w", encoding="utf-8") as f:
                    json.dump(error_data, f, indent=2)
                return
        except Exception as e:
            yield f"Potato {chunk_num}: Error - {type(e).__name__}: {str(e)}\n\n"
            error_data = {"error": str(e), "timestamp": datetime.now().isoformat()}
            error_file = os.path.join(LOG_DIR, f"{timestamp}_venice_error.json")
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_data, f, indent=2)
            return
    
    if session_data["running"]:
        yield f"\nPotato {chunk_num}: Done\n\n"
        response_data = {
            "id": "unknown" if not locals().get("chunk") else chunk.id,
            "model": model_id,
            "choices": [{"message": {"role": "assistant", "content": full_response}}],
            "usage": usage or {"prompt_tokens": "unknown", "completion_tokens": "unknown", "total_tokens": "unknown"},
            "timestamp": datetime.now().isoformat()
        }
        response_file = os.path.join(LOG_DIR, f"{timestamp}_venice_response.json")
        with open(response_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2)

@app.route("/", methods=["GET"])
def index():
    models = load_models()
    model_ids = [m["id"] for m in models]
    default_model = session_data["selected_model"] or ("mistral-31-24b" if "mistral-31-24b" in model_ids else "llama-3.2-3b")
    filename = session_data["filename"] or "No file chosen"
    return render_template("index.html", models=model_ids, default_model=default_model, 
                         estimate=None, filename=filename, prompt=session_data["prompt"], 
                         max_output_tokens=session_data["max_output_tokens"], api_key=session_data["api_key"])

@app.route("/estimate", methods=["GET", "POST"])
def estimate():
    models = load_models()
    model_ids = [m["id"] for m in models]
    default_model = session_data["selected_model"] or ("mistral-31-24b" if "mistral-31-24b" in model_ids else "llama-3.2-3b")
    
    if request.method == "GET":
        filename = session_data["filename"] or "No file chosen"
        return render_template("index.html", models=model_ids, default_model=default_model, 
                             estimate=None, filename=filename, prompt=session_data["prompt"], 
                             max_output_tokens=session_data["max_output_tokens"], api_key=session_data["api_key"])
    
    file = request.files.get("file")
    prompt = request.form.get("prompt", "")
    model_id = request.form.get("model", default_model)
    max_output_tokens = int(request.form.get("max_output_tokens", 3337))
    api_key_form = request.form.get("api_key", "").strip()
    api_key = api_key_form if api_key_form else (session_data["api_key"] or env_api_key or "")
    if not api_key:
        return render_template("index.html", models=model_ids, default_model=default_model, 
                             estimate="Please provide an API key", filename="No file chosen", 
                             prompt=prompt, max_output_tokens=max_output_tokens, api_key="")
    
    if api_key_form:
        session_data["api_key"] = api_key
    session_data["prompt"] = prompt
    session_data["max_output_tokens"] = max_output_tokens
    session_data["selected_model"] = model_id
    
    if file:
        session_data["file_content"] = file.read().decode("utf-8")
        session_data["filename"] = file.filename
    
    chat_text = session_data["file_content"]
    if not chat_text:
        return render_template("index.html", models=model_ids, default_model=default_model, 
                             estimate="No file uploaded yet", filename="No file chosen", 
                             prompt=prompt, max_output_tokens=max_output_tokens, api_key=api_key)
    
    model_info = get_model_info(model_id, models)
    if not model_info:
        return render_template("index.html", models=model_ids, default_model=default_model, 
                             estimate="Invalid model", filename=session_data["filename"], 
                             prompt=prompt, max_output_tokens=max_output_tokens, api_key=api_key)
    
    system_prompt = f"{prompt} Please ensure output does not exceed {max_output_tokens} tokens."
    prompt_tokens = estimate_tokens(system_prompt)
    max_context_tokens = model_info.get("context_tokens", 131072)
    file_tokens = estimate_tokens(chat_text)
    sample_message = f"Potato 1/X (size: {estimate_tokens(chat_text[:1000])} tokens):\n{chat_text[:1000]}"
    message_overhead = estimate_tokens(sample_message) - estimate_tokens(chat_text[:1000])
    
    try:
        chunks, num_chunks = split_into_chunks(chat_text, max_context_tokens, prompt_tokens, message_overhead, max_output_tokens)
    except ValueError as e:
        return render_template("index.html", models=model_ids, default_model=default_model, 
                             estimate=str(e), filename=session_data["filename"], 
                             prompt=prompt, max_output_tokens=max_output_tokens, api_key=api_key)
    
    input_tokens = file_tokens + (num_chunks * (prompt_tokens + message_overhead))
    pricing = PRICING.get(model_id, PRICING["mistral-31-24b"])
    estimated_cost = (input_tokens * pricing["input"]) + (max_output_tokens * num_chunks * pricing["output"])
    estimate = f"We’ll plant {num_chunks} potato{'es' if num_chunks != 1 else ''} costing ~${estimated_cost:.2f}"
    
    return render_template("index.html", models=model_ids, default_model=model_id, 
                         estimate=estimate, filename=session_data["filename"], prompt=prompt, 
                         max_output_tokens=max_output_tokens, max_context_tokens=max_context_tokens, api_key=api_key)

@app.route("/process", methods=["POST"])
def process():
    models = load_models()
    model_ids = [m["id"] for m in models]
    
    file = request.files.get("file")
    prompt = request.form.get("prompt", "")
    model_id = request.form.get("model", session_data.get("selected_model", "mistral-31-24b"))
    max_output_tokens = int(request.form.get("max_output_tokens", 3337))
    api_key_form = request.form.get("api_key", "").strip()
    api_key_header = request.headers.get("X-API-Key", "").strip()
    api_key = api_key_header or api_key_form or session_data["api_key"] or env_api_key or ""
    if not api_key:
        return "No API key provided", 400
    
    if api_key_form or api_key_header:
        session_data["api_key"] = api_key
    session_data["prompt"] = prompt
    session_data["max_output_tokens"] = max_output_tokens
    session_data["selected_model"] = model_id
    
    if file:
        session_data["file_content"] = file.read().decode("utf-8")
        session_data["filename"] = file.filename
    
    chat_text = session_data["file_content"]
    if not chat_text:
        return "No file uploaded yet", 400
    
    model_info = get_model_info(model_id, models)
    if not model_info:
        return "Invalid model", 400
    
    system_prompt = f"{prompt} Please ensure output does not exceed {max_output_tokens} tokens."
    prompt_tokens = estimate_tokens(system_prompt)
    max_context_tokens = model_info.get("context_tokens", 131072)
    file_tokens = estimate_tokens(chat_text)
    sample_message = f"Potato 1/X (size: {estimate_tokens(chat_text[:1000])} tokens):\n{chat_text[:1000]}"
    message_overhead = estimate_tokens(sample_message) - estimate_tokens(chat_text[:1000])
    
    try:
        chunks, num_chunks = split_into_chunks(chat_text, max_context_tokens, prompt_tokens, message_overhead, max_output_tokens)
    except ValueError as e:
        return str(e), 400
    
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data["task_id"] = task_id
    session_data["results"][task_id] = {"status": "running", "output": ""}
    session_data["running"] = True
    
    def generate():
        logger.info("Starting generate function")
        output = f"Starting to plant {num_chunks} potato{'es' if num_chunks != 1 else ''}...\n\n"
        yield output
        session_data["results"][task_id]["output"] += output
        for i, chunk in enumerate(chunks, 1):
            if not session_data["running"]:
                yield "Process stopped by user.\n"
                break
            output = f"Planting potato {i} of {num_chunks}...\n"
            yield output
            session_data["results"][task_id]["output"] += output
            logger.info(f"Processing chunk {i}/{num_chunks}")
            for streamed_output in process_chunk(chunk, prompt, model_id, i, num_chunks, task_id, max_context_tokens, max_output_tokens, api_key):
                if not session_data["running"]:
                    yield "Process stopped by user.\n"
                    break
                yield streamed_output
                session_data["results"][task_id]["output"] += streamed_output
            time.sleep(0.1)
        if session_data["running"]:
            output = f"\nMashing complete! Task ID: {task_id}\n"
            yield output
            session_data["results"][task_id]["output"] += output
            session_data["results"][task_id]["status"] = "complete"
        session_data["running"] = False
        logger.info("Generate function completed")
    
    return Response(stream_with_context(generate()), mimetype="text/plain")

@app.route("/stop", methods=["POST"])
def stop():
    if session_data["running"]:
        session_data["running"] = False
        return "Stopping process...", 200
    return "No process running", 400

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