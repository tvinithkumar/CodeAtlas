# CodeAtlas

CodeAtlas transforms source code into searchable symbol-level knowledge using AST
analysis, symbol extraction, code chunking, generated descriptions, ripgrep,
SQLite FTS, and Qdrant vector search.

The MVP is intentionally local-first:

```text
RepositoryIndexer
  -> RepositoryLoader
  -> PythonParser
  -> SymbolExtractor
  -> ASTChunker
  -> SymbolProfiler
  -> HashEmbeddingGenerator
  -> SQLiteStore + QdrantVectorStore
  -> Ripgrep + FTS + Vector
  -> HybridSearch
```

## MVP Storage

CodeAtlas uses:

```text
SQLite
  -> files, symbols, chunks, generated profiles, indexing metadata

SQLite FTS5
  -> exact keyword search over symbol names, paths, profiles, and code chunks

Qdrant
  -> vector search over generated profiles and code chunks

ripgrep
  -> zero-embedding exact retrieval over the working tree
```

Postgres is deliberately deferred until the local MVP pipeline is useful.

The graph layer is stored in SQLite for now:

```sql
symbol_edges(
  src_symbol_id,
  dst_symbol_id,
  edge_type,
  file_path,
  confidence
)
```

Current edge types include:

```text
calls
imports
inherits
reads_config
emits_metric
logs
queries_table
```

## Quick Start

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

Install optional LLM support:

```bash
pip install -e ".[llm]"
```

Index a repository with SQLite only:

```bash
codeatlas index /path/to/repo --no-qdrant
```

Run offline LLM enrichment with Ollama and LiteLLM:

```bash
ollama pull qwen2.5-coder:7b
codeatlas index /path/to/repo --no-qdrant --with-llm
```

Or use a YAML config:

```yaml
llm:
  enabled: true
  provider: ollama
  model: qwen2.5-coder:7b
  temperature: 0.1
```

```bash
codeatlas --config codeatlas.yml index /path/to/repo --no-qdrant
```

Index with Qdrant enabled after starting a local Qdrant server:

```bash
docker run -p 6333:6333 qdrant/qdrant
codeatlas index /path/to/repo
```

Search and inspect graph relationships:

```bash
codeatlas search "where is retry logic configured" --no-vectors
codeatlas search "retryBackoffMs" --repo /path/to/repo --no-vectors
codeatlas explain retry_delay
codeatlas find-usage load_user
codeatlas impact REQUEST_TIMEOUT_MS
codeatlas window --repo /path/to/repo --file app/config.py --line 42 --radius 20
```

By default, the SQLite database is written to:

```text
.codeatlas/codeatlas.db
```

## Current Scope

The first implementation supports Python source files. It extracts classes,
functions, async functions, symbol relationships, symbol chunks, and file
summaries. It stores metadata in SQLite, indexes lexical content with FTS5,
and can upsert vectors to Qdrant.

Embeddings default to the local FastEmbed model:

```text
BAAI/bge-small-en-v1.5
```

The tests inject a deterministic hash embedder so CI does not need to download
model files.

## Optional LLM Enrichment

The LLM layer is provider-pluggable and uses LiteLLM:

```text
LiteLLM -> Ollama, OpenAI, Anthropic, Gemini, or another LiteLLM provider
```

The first supported local default is:

```text
ollama/qwen2.5-coder:7b
```

LM Studio is supported through its OpenAI-compatible local server:

```yaml
llm:
  enabled: true
  provider: lmstudio
  model: qwen3-coder-30b-a3b-instruct
  temperature: 0.1
  max_tokens: 512
  base_url: http://localhost:1234/v1
  api_key: lm-studio
```

Check the server with:

```bash
curl http://localhost:1234/v1/models
```

Index with LM Studio directly:

```bash
codeatlas index examples/benchmark_repo \
  --no-qdrant \
  --llm-enabled \
  --llm-provider lmstudio \
  --llm-model qwen3-coder-30b-a3b-instruct \
  --llm-base-url http://localhost:1234/v1 \
  --llm-max-tokens 512
```

LLM enrichment runs only during indexing. Search does not call an LLM yet.

Current LLM tasks:

```text
describe_symbol
summarize_function
generate_search_tags
```

The generated description, tags, inputs, outputs, and failure modes are stored
in SQLite and folded into FTS/Qdrant indexing text.

## Retrieval Modes

CodeAtlas has three retrieval modes:

```text
Ripgrep retrieval
  -> exact code search for error messages, metric names, config keys, env vars,
     API routes, class names, function names, and SQL table names

Symbol/FTS retrieval
  -> indexed search over symbol profiles, generated tags, file summaries, and chunks

Semantic retrieval
  -> Qdrant vector search over enriched code profiles and chunks
```

Hybrid search routes exact-looking queries toward ripgrep, then fuses ripgrep,
FTS, and vector results with Reciprocal Rank Fusion. Exact code-like queries
such as `retryBackoffMs`, `alert_grouping_latency`, `ConnectionRefusedError`,
or `/api/v2/tickets` get a heavier ripgrep weight.

## Evaluation

CodeAtlas includes a small retrieval evaluation harness:

```bash
codeatlas-eval --repo examples/benchmark_repo --cases evals/benchmark_cases.yaml
```

It reports per-mode and per-case metrics for:

```text
Recall@5
Recall@10
MRR
Hit@1
Context Compression Ratio
```

The default benchmark compares:

```text
ripgrep
SQLite FTS
hybrid CodeAtlas retrieval
```

## MCP Server

Install optional MCP support:

```bash
pip install -e ".[mcp]"
```

Run the server after indexing a repository:

```bash
codeatlas index /path/to/repo --no-qdrant
codeatlas-mcp --repo /path/to/repo --sqlite-path .codeatlas/codeatlas.db
```

Exposed tools:

```text
search_code(query, top_k=5)
get_code_window(file, line, radius=20)
explain_symbol(symbol)
find_usages(symbol)
related_symbols(symbol)
```

Default token-safety limits:

```text
max_results = 5
max_lines_per_result = 40
max_total_chars = 12000
```

Planned next steps:

```text
1. Add Tree-sitter parsers for JavaScript, TypeScript, Go, and Java.
2. Expand Python relationship extraction for attribute resolution and cross-file imports.
3. Add incremental indexing.
4. Add file summaries from a real local or provider LLM.
5. Introduce Postgres after the MVP data model stabilizes.
```
