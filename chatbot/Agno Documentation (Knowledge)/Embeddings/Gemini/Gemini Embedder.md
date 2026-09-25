# Gemini Embedder

The `GeminiEmbedder` class is used to embed text data into vectors using the Gemini API. You can get one from [here](https://ai.google.dev/aistudio).

## Usage

```python gemini_embedder.py theme={null}
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector
from agno.knowledge.embedder.google import GeminiEmbedder

# Embed sentence in database
embeddings = GeminiEmbedder().get_embedding("The quick brown fox jumps over the lazy dog.")

# Print the embeddings and their dimensions
print(f"Embeddings: {embeddings[:5]}")
print(f"Dimensions: {len(embeddings)}")

# Use an embedder in a knowledge base
knowledge = Knowledge(
    vector_db=PgVector(
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        table_name="gemini_embeddings",
        embedder=GeminiEmbedder(),
    ),
    max_results=2,
)
```

## Params

| Parameter        | Type                       | Default                     | Description                                                       |
| ---------------- | -------------------------- | --------------------------- | ----------------------------------------------------------------- |
| `dimensions`     | `int`                      | `768`                       | The dimensionality of the generated embeddings                    |
| `model`          | `str`                      | `models/text-embedding-004` | The name of the Gemini model to use                               |
| `task_type`      | `str`                      | -                           | The type of task for which embeddings are being generated         |
| `title`          | `str`                      | -                           | Optional title for the embedding task                             |
| `api_key`        | `str`                      | -                           | The API key used for authenticating requests.                     |
| `request_params` | `Optional[Dict[str, Any]]` | -                           | Optional dictionary of parameters for the embedding request       |
| `client_params`  | `Optional[Dict[str, Any]]` | -                           | Optional dictionary of parameters for the Gemini client           |
| `gemini_client`  | `Optional[Client]`         | -                           | Optional pre-configured Gemini client instance                    |
| `enable_batch`   | `bool`                     | `False`                     | Enable batch processing to reduce API calls and avoid rate limits |
| `batch_size`     | `int`                      | `100`                       | Number of texts to process in each API call for batch operations. |

## Developer Resources

* View [Cookbook](https://github.com/agno-agi/agno/tree/main/cookbook/08_knowledge/embedders/gemini_embedder.py)


---

# Gemini Embedder

## Code

```python  theme={null}
from agno.knowledge.embedder.google import GeminiEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector import PgVector

embeddings = GeminiEmbedder().get_embedding(
    "The quick brown fox jumps over the lazy dog."
)

# Print the embeddings and their dimensions
print(f"Embeddings: {embeddings[:5]}")
print(f"Dimensions: {len(embeddings)}")

# Example usage:
knowledge = Knowledge(
    vector_db=PgVector(
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        table_name="gemini_embeddings",
        embedder=GeminiEmbedder(),
    ),
    max_results=2,
)
```

## Usage

<Steps>
  <Snippet file="create-venv-step.mdx" />

  <Step title="Set your API key">
    ```bash  theme={null}
    export GOOGLE_API_KEY=xxx
    ```
  </Step>

  <Step title="Install libraries">
    ```bash  theme={null}
    pip install -U sqlalchemy psycopg pgvector google-generativeai agno
    ```
  </Step>

  <Snippet file="run-pgvector-step.mdx" />

  <Step title="Run Agent">
    <CodeGroup>
      ```bash Mac theme={null}
      python cookbook/08_knowledge/embedders/gemini_embedder.py
      ```

      ```bash Windows theme={null}
      python cookbook/08_knowledge/embedders/gemini_embedder.py
      ```
    </CodeGroup>
  </Step>
</Steps>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.agno.com/llms.txt