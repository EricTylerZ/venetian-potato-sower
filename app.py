import os
import json
from flask import Flask, render_template, request, send_file, Response

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("VENICE_API_KEY"), base_url="https://api.venice.ai/api/v1")

app = Flask(__name__)

# Store result temporarily (in-memory for Vercel)
latest_result = {"content": None}

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

@app.route("/", methods=["GET", "POST"])
def index():
    models = [m for m in load_models() if m.get("model_extra", {}).get("type") == "text"]
    model_ids = [m["id"] for m in models]
    default_model = "mistral-31-24b" if "mistral-31-24b" in model_ids else "llama-3.2-3b"
    
    estimate = None
    result = None
    
    if request.method == "POST":
        action = request.form.get("action")
        file = request.files.get("file")
        prompt = request.form.get("prompt", "")
        model_id = request.form.get("model", default_model)
        
        if not file or file.filename.split(".")[-1].lower() not in ["csv", "txt"]:
            return render_template("index.html", models=model_ids, default_model=default_model, 
                                 estimate="Invalid file type", result=None)
        
        chat_text = file.read().decode("utf-8")
        model_info = get_model_info(model_id, models)
        if not model_info:
            return render_template("index.html", models=model_ids, default_model=default_model, 
                                 estimate="Invalid model", result=None)
        
        file_size = len(chat_text)
        context_tokens = model_info["context_tokens"]
        chunk_size = int((context_tokens * 0.8 - 150) * 4)  # 150 tokens for prompt
        num_chunks = (file_size + chunk_size - 1) // chunk_size
        
        if action == "estimate":
            total_tokens = file_size // 4 + num_chunks * 150
            cost_per_token = 0.0001  # Placeholder
            estimated_cost = total_tokens * cost_per_token
            estimate = f"Estimated runs: {num_chunks}, Estimated cost: ${estimated_cost:.2f}"
        
        elif action == "process":
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
            latest_result["content"] = result
    
    return render_template("index.html", models=model_ids, default_model=default_model, 
                         estimate=estimate, result=result)

@app.route("/download")
def download():
    if not latest_result["content"]:
        return "No result available", 404
    return Response(latest_result["content"], mimetype="text/plain", 
                   headers={"Content-Disposition": "attachment;filename=prompt_history.txt"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))