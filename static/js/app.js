import os
import json
from flask import Flask, render_template, request, send_file, Response
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("VENICE_API_KEY"), base_url="https://api.venice.ai/api/v1")

app = Flask(__name__)

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

@app.route("/")
def index():
    models = load_models()
    model_ids = [m["id"] for m in models]
    default_model = "mistral-31-24b" if "mistral-31-24b" in model_ids else "llama-3.2-3b"
    return render_template("index.html", models=model_ids, default_model=default_model)

@app.route("/estimate", methods=["POST"])
def estimate():
    file = request.files["file"]
    model_id = request.form["model"]
    models = load_models()
    model_info = get_model_info(model_id, models)
    
    if not file or not model_info:
        return "Missing file or invalid model", 400
    
    file_content = file.read().decode("utf-8")
    file_size = len(file_content)  # Size in characters
    context_tokens = model_info["context_tokens"]
    chunk_size = int((context_tokens * 0.8 - 150) * 4)  # 150 tokens for prompt, 80% context
    num_chunks = (file_size + chunk_size - 1) // chunk_size  # Ceiling division
    
    # Rough token estimate (1 token ≈ 4 chars)
    total_tokens = file_size // 4 + num_chunks * 150
    # Hypothetical cost (since API doesn’t provide pricing)
    cost_per_token = 0.0001  # Placeholder until Venice.ai provides this
    estimated_cost = total_tokens * cost_per_token
    
    return f"Estimated runs: {num_chunks}, Estimated cost: ${estimated_cost:.2f}"

@app.route("/process", methods=["POST"])
def process():
    file = request.files["file"]
    prompt = request.form["prompt"]
    model_id = request.form["model"]
    
    if not file or not prompt or not model_id:
        return "Missing inputs", 400
    
    if file.filename.split(".")[-1].lower() not in ["csv", "txt"]:
        return "Invalid file type", 400
    
    models = load_models()
    model_info = get_model_info(model_id, models)
    if not model_info:
        return "Invalid model", 400
    
    chat_text = file.read().decode("utf-8")
    context_tokens = model_info["context_tokens"]
    chunk_size = int((context_tokens * 0.8 - 150) * 4)  # 150 tokens for prompt
    chunks = [chat_text[i:i+chunk_size] for i in range(0, len(chat_text), chunk_size)]
    
    result = "(Abridged) Prompt History for Venetian Potato Sower\n\n"
    for i, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Chunk {i+1}/{len(chunks)}:\n{chunk}"}
            ],
            temperature=0.3
        )
        result += f"Chunk {i+1}:\n{response.choices[0].message.content}\n\n"
    
    # Store result for download
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result)
    
    return Response(result, mimetype="text/plain")

@app.route("/download")
def download():
    return send_file("output.txt", as_attachment=True, download_name="prompt_history.txt")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))