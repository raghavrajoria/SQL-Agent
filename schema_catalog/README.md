# Schema Catalog

This directory contains the schema-understanding and schema-retrieval layer.

## Purpose

The SQL generator should not receive the entire database schema for every question.

Instead, this layer:

1. extracts schema metadata;
2. stores table/column/FK information;
3. generates table-level semantic descriptions;
4. embeds those descriptions;
5. retrieves relevant tables for a question;
6. expands the retrieved tables through known foreign-key relationships;
7. ranks the resulting context before SQL generation.

## Current retrieval flow

```text
User question
    |
    v
SentenceTransformer embedding
    |
    v
Relevant table retrieval
    |
    v
FK graph expansion
    |
    v
Ranked schema context
    |
    v
SQL generator
```

## Important modules

### `extract_schema.py`

Extracts database metadata such as:

- table names;
- column names;
- column types;
- primary keys;
- foreign keys.

### `generate_descriptions.py`

Generates table-level natural-language descriptions using the LLM.

These descriptions are intended for retrieval rather than direct SQL generation.

### `embed_schema.py`

Embeds table descriptions using:

`sentence-transformers/all-MiniLM-L6-v2`

### `retrieve_schema.py`

Embeds a user question and retrieves the most relevant schema entries.

### `expand_schema.py`

Provides:

- schema catalog loading;
- FK graph construction;
- relationship expansion.

FK information is important because a question may retrieve one table directly while the correct SQL requires related tables.

### `rank_schema_context.py`

Combines retrieval results with FK expansion and produces the ranked context consumed by SQL generation.

## Current model

Embeddings use:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model runs locally. A Hugging Face account/token is not required for the current local development workflow.

## Current status

**Implemented and integrated with the agent.**

Retrieval and FK expansion have been tested with Sakila questions such as:

- customers with the most rentals;
- actors appearing in the most movies;
- films generating rental revenue.

## Known limitation

Retrieval quality is the major concern when moving toward a much larger production schema. Cryptic or legacy table/column names can reduce the quality of automatically generated descriptions and therefore retrieval accuracy.
