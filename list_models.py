import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("VENICE_API_KEY"), base_url="https://api.venice.ai/api/v1")

def is_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def fetch_and_save_models(output_file="models.json"):
    models = client.models.list()
    model_data = []
    for model in models.data:
        model_dict = {
            "id": model.id,
            "created": model.created,
            "owned_by": model.owned_by,
            "context_tokens": getattr(model, "model_spec", {}).get("availableContextTokens", 65536)
        }
        # Include all available attributes dynamically
        for attr in dir(model):
            if not attr.startswith("_") and attr not in model_dict:
                try:
                    value = getattr(model, attr)
                    if is_serializable(value):
                        model_dict[attr] = value
                except AttributeError:
                    pass
        model_data.append(model_dict)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"models": model_data}, f, indent=2)
    print(f"Saved {len(model_data)} models to {output_file}")

if __name__ == "__main__":
    fetch_and_save_models()