## Lightweight wrapper around ChatGPT using Facebook AI Similarity Search (Faiss) as a vector database

## GPTWrapper
#### - Wrapper around GPT 3.5 turbo 
#### - Retains user input and model output to allow for use of OpenAI's Chat Completions API
#
## VectorDB
#### Wrapper around Facebook AI Similarity Search (Faiss)
#### Tracks supplied metadata and embedded vectors
#
## SGPTWrapper
#### - Niklas Muennighoff's SGPT embedder that can be found here: https://github.com/Muennighoff/sgpt
#
### Current Example (main.py)
#### A simple example that:
####   - Reads and embeds a supplied PDF.
####   - Loads the embedded pdf substrings it into the vector database.
####   - Asks the user for a topic to retrieve relevant embedded vectors as context.
####   - Feed the formatted prompt into the LLM, print the response, and save the output in GPTWrapper.
