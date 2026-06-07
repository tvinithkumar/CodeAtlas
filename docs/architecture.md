# CodeAtlas MVP Architecture

The MVP uses a local-first pipeline:

```text
RepositoryLoader
  -> PythonParser
  -> SymbolExtractor
  -> ASTChunker
  -> SymbolProfiler
  -> optional LLMEnricher
  -> LocalEmbeddingGenerator
  -> SQLiteStore + QdrantVectorStore
  -> RipgrepRetriever + FTSSearch + VectorSearch
  -> HybridSearch + GraphSearch
```

SQLite is the source of truth for files, symbols, chunks, and FTS5 keyword search.
It also stores `symbol_edges` for the first graph search implementation. Qdrant
stores chunk vectors for semantic retrieval. If Qdrant is unavailable, indexing
still succeeds and SQLite search remains usable.

Ripgrep is the primary lexical retriever for exact code search. It uses
`rg --json` and returns normalized retrieval chunks with file path, line range,
content, source, and retrieval method.

Indexing units are split into:

```text
symbol profile
function chunk
class chunk
file summary
```

The current graph search supports:

```text
explain_symbol
find_usage
impact_analysis
architecture_search
natural_language_search via HybridSearch
```

Hybrid search uses a query analyzer:

```text
CamelCase, snake_case, error codes, metric names, routes, file paths
  -> 70% ripgrep, 20% FTS, 10% vector

natural-language queries
  -> 25% ripgrep, 35% FTS, 40% vector
```

The ranked lists are merged with Reciprocal Rank Fusion before returning top K.

## Evaluation

The eval harness compares retrieval modes against YAML benchmark cases:

```text
evals/benchmark_cases.yaml
evals/run_retrieval_eval.py
evals/metrics.py
```

Core metrics:

```text
Recall@5
Recall@10
MRR
Hit@1
Context Compression Ratio
```

`Context Compression Ratio` is computed as:

```text
raw repository context tokens / retrieved context tokens
```

This approximates how much source code an agent avoids reading when using
CodeAtlas retrieval.

## Code Windows

`CodeWindowFetcher` supports bounded file reads for future agent/MCP tools:

```bash
codeatlas window --repo /path/to/repo --file app/config.py --line 42 --radius 20
```

Postgres and a dedicated graph database are intentionally deferred until after the MVP
pipeline proves useful locally.

## Optional LLM Layer

The LLM layer is deliberately outside the query path:

```text
symbol/function code
  -> LiteLLM client
  -> description, tags, inputs, outputs, failure modes
  -> SQLite + FTS + Qdrant
```

Provider defaults:

```text
provider: ollama
model: qwen2.5-coder:7b
temperature: 0.1
```

The minimal interface is:

```python
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...
```
