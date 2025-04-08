from flask import Flask, render_template, request
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("VENICE_API_KEY"), base_url="https://api.venice.ai/api/v1")

app = Flask(__name__)

# Load model information
with open("models.json", "r") as f:
    models = json.load(f)["models"]

def get_model_info(model_id):
    """Retrieve model details by ID."""
    for model in models:
        if model["id"] == model_id:
            return model
    return None

@app.route("/")
def index():
    """Render the main page with upload form."""
    return render_template("index.html", models=[m["id"] for m in models])

@app.route("/process", methods=["POST"])
def process():
    """Handle file upload, chunking, and API processing."""
    file = request.files["file"]
    prompt = request.form["prompt"]
    model_id = request.form["model"]

    # Validate inputs
    if not file or not prompt or not model_id:
        return "Missing inputs", 400
    if file.filename.split(".")[-1].lower() not in ["csv", "txt"]:
        return "Invalid file type", 400

    # Read and process the file
    chat_text = file.read().decode("utf-8")
    model_info = get_model_info(model_id)
    if not model_info:
        return "Invalid model", 400

    context_tokens = model_info["context_tokens"]
    # Chunk size in characters, leaving room for prompt (150 tokens) and buffer
    chunk_size = int((context_tokens * 0.8 - 150) * 4)
    chunks = [chat_text[i:i+chunk_size] for i in range(0, len(chat_text), chunk_size)]
    
    result = "(Abridged) Prompt History for Venetian Potatoes Sower\n\n"
    for i, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": chunk}
            ],
            temperature=0.3
        )
        result += f"Chunk {i+1}:\n" + response.choices[0].message.content + "\n\n"

    return result

if __name__ == "__main__":
    app.run(debug=True)