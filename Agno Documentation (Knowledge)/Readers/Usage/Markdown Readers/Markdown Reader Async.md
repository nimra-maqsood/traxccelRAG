# Markdown Reader Async

The **Markdown Reader** with asynchronous processing allows you to handle Markdown files efficiently and integrate them with knowledge bases.

## Code

```python examples/basics/knowledge/readers/markdown_reader_async.py theme={null}
import asyncio
from pathlib import Path

from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.pgvector.pgvector import PgVector

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge = Knowledge(
    vector_db=PgVector(
        table_name="markdown_documents",
        db_url=db_url,
    ),
    max_results=5,  # Number of results to return on search
)

agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,
)

if __name__ == "__main__":
    asyncio.run(
        knowledge.add_content_async(
            path=Path("README.md"),
        )
    )

    asyncio.run(
        agent.aprint_response(
            "What can you tell me about Agno?",
            markdown=True,
        )
    )
```

## Usage

<Steps>
  <Snippet file="create-venv-step.mdx" />

  <Step title="Install libraries">
    ```bash  theme={null}
    pip install -U markdown sqlalchemy psycopg pgvector agno openai
    ```
  </Step>

  <Step title="Set environment variables">
    ```bash  theme={null}
    export OPENAI_API_KEY=xxx
    ```
  </Step>

  <Snippet file="run-pgvector-step.mdx" />

  <Step title="Run Agent">
    <CodeGroup>
      ```bash Mac theme={null}
      python examples/basics/knowledge/readers/markdown_reader_async.py
      ```

      ```bash Windows theme={null}
      python examples/basics/knowledge/readers/markdown_reader_async.py
      ```
    </CodeGroup>
  </Step>
</Steps>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.agno.com/llms.txt