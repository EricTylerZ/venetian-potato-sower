# Venetian Potato Sower

A web-based tool to analyze large files (presently .txt and csv) across multiple LLM requests and then recombine into a single output. Using the Venice.ai API, with cost estimation before processing. 

## Why "Venetian Potato Sower"?
- **Venetian:** Powered by the uncensored Venice.ai API.
- **Potato** Large files are chunked into pieces, like cutting potatoes and planting
- **Sower:** Each chunk is processed ("sown") to yield valuable results.
- **Harvest & Mash** We'll then take each processed plant in the field and mash them together.

## Features
- Upload CSV or .txt chat logs.
- Input a custom prompt for analysis.
- Select from multiple Venice.ai models (e.g., 'mistral-31-24b', `llama-3.2-3b`).
- Estimate processing cost based on file size and model pricing.
- Process files with chunking for large logs, returning a combined set of mashed potatoes.