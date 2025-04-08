const models = {
    "llama-3.2-3b": { context_tokens: 131072, cost_per_input_token: 0.0001 },
    "llama-3.3-70b": { context_tokens: 131072, cost_per_input_token: 0.001 }
    // Add more models as needed from Venice.ai
};

function estimateCost() {
    const fileInput = document.getElementById('file');
    const modelSelect = document.getElementById('model');
    const estimateP = document.getElementById('estimate');
    const confirmBtn = document.getElementById('confirm');

    if (!fileInput.files[0]) {
        estimateP.textContent = "Please select a file.";
        return;
    }

    const fileSize = fileInput.files[0].size;  // Size in bytes
    const modelId = modelSelect.value;
    const model = models[modelId];
    if (!model) {
        estimateP.textContent = "Invalid model selected.";
        return;
    }

    const context_tokens = model.context_tokens;
    const cost_per_input_token = model.cost_per_input_token;
    const tokens_in_prompt = 150;  // Approximate token count for the prompt
    const chunk_size = Math.floor((context_tokens * 0.8 - tokens_in_prompt) * 4);  // Chunk size in characters
    const number_of_chunks = Math.ceil(fileSize / chunk_size);
    const total_tokens_in_chat_text = fileSize / 4;  // Approximate tokens (1 token ≈ 4 chars)
    const total_input_tokens = number_of_chunks * tokens_in_prompt + total_tokens_in_chat_text;
    const estimated_cost = total_input_tokens * cost_per_input_token;

    estimateP.textContent = `Estimated cost: $${estimated_cost.toFixed(2)}`;
    confirmBtn.style.display = 'inline';
}