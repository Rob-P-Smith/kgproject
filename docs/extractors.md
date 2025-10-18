---
layout: default
title: Extractors Module
---

# Extractors Module Documentation

## Module Overview

The extractors module implements ML-powered knowledge extraction from unstructured text. It consists of two specialized extractors: EntityExtractor for named entity recognition using GLiNER, and RelationshipExtractor for semantic relationship identification using vLLM.

**Files**:
- `extractors/entity_extractor.py`: GLiNER-based entity recognition
- `extractors/relation_extractor.py`: vLLM-based relationship extraction
- `extractors/__init__.py`: Module exports

## Entity Extractor

### Purpose

Extract named entities from text with hierarchical type classification, supporting 300+ entity types across multiple domains (technology, business, AI, data, etc.). Handles documents of arbitrary length by automatic chunking for GLiNER's token limits.

### Architecture

**Class**: `EntityExtractor`

**Dependencies**:
- GLiNER model (loaded from HuggingFace)
- Entity taxonomy (YAML file defining hierarchical types)
- Configuration settings (threshold, batch size, etc.)

**Singleton Pattern**: Global instance via `get_entity_extractor()` ensures model loaded once

### Initialization

```
Constructor Parameters:
- model_name: str (default from config.GLINER_MODEL)
- taxonomy_path: str (default "taxonomy/entities.yaml")
- threshold: float (default from config.GLINER_THRESHOLD)

Initialization Steps:
1. Load GLiNER model from HuggingFace (urchade/gliner_large-v2.1)
   - Downloads ~4GB model weights on first run
   - Loads into memory (requires ~4GB RAM)
2. Load entity taxonomy from YAML file
   - Parses hierarchical entity types (e.g., "Framework::Backend::Python")
   - Flattens into list of 300+ type strings
3. Initialize type cache (empty dict for parsed hierarchies)

Model Load Time: ~30 seconds on first run, instant on subsequent (cached)
```

### Entity Taxonomy

**File**: `taxonomy/entities.yaml`

**Structure**:
```yaml
entity_categories:
  technology:
    - Technology::Software
    - Technology::Framework
    - Programming::Language
    - Web::Framework
  data:
    - Database::Relational
    - Database::Graph
  ai:
    - AI::LLM
    - MachineLearning::Framework
```

**Hierarchy Levels**:
- Level 1 (Primary): Framework, Database, AI, Organization, etc.
- Level 2 (Sub1): Backend, Graph, LLM, etc.
- Level 3 (Sub2): Python, NoSQL, etc.
- Level 4 (Sub3): Optional fourth level for fine-grained classification

**Total Types**: ~300 types across 10 categories

### Extraction Process

**Method**: `extract(text, custom_types=None, threshold=None) -> List[Dict]`

**Input**:
- `text`: Full document text (any length)
- `custom_types`: Optional entity types (defaults to taxonomy)
- `threshold`: Confidence threshold (default 0.4)

**Output**: List of entity dictionaries with:
```python
{
  "text": "FastAPI",              # Original entity text
  "normalized": "fastapi",        # Lowercase normalized form
  "start": 342,                   # Character start position in document
  "end": 349,                     # Character end position
  "type_full": "Framework::Backend::Python",  # Complete hierarchical type
  "type_primary": "Framework",    # Top-level type
  "type_sub1": "Backend",         # Second-level type
  "type_sub2": "Python",          # Third-level type
  "type_sub3": None,              # Fourth-level (if exists)
  "confidence": 0.95,             # GLiNER confidence score
  "context_before": "modern web ",  # 50 chars before entity
  "context_after": " for building", # 50 chars after entity
  "sentence": "FastAPI is a modern web framework for building APIs.",
  "extraction_method": "gliner"
}
```

**Processing Steps**:

1. **Text Validation**:
   - Check for empty/None text → return empty list
   - Log warning if empty

2. **Text Chunking** (for long documents):
   ```
   If len(text) > 1500 chars:
     Split into GLiNER-compatible chunks (~1000 chars each)
     Chunk on word boundaries (not mid-word)
     Track char_start/char_end for each chunk
     Process each chunk separately
     Adjust entity positions to document coordinates
     Merge predictions from all chunks
   Else:
     Process entire text in single GLiNER call
   ```

3. **GLiNER Inference**:
   ```
   Call: model.predict_entities(text, entity_types, threshold)

   GLiNER Processing:
   - Tokenizes text (max 384 tokens, ~1500 chars)
   - Runs transformer model for entity span detection
   - Classifies each span into entity_types
   - Returns predictions above threshold

   Output: List of {text, label, score, start, end}
   ```

4. **Type Hierarchy Parsing**:
   ```
   For each prediction:
     Parse label "Framework::Backend::Python" into:
       type_primary: "Framework"
       type_sub1: "Backend"
       type_sub2: "Python"
       type_sub3: None
     Cache parsed hierarchy for reuse
   ```

5. **Context Extraction**:
   ```
   For each entity:
     Extract 50-char window before/after entity
     Find sentence boundaries (periods before/after)
     Extract full sentence containing entity
     Store context_before, context_after, sentence
   ```

6. **Deduplication**:
   ```
   If config.ENTITY_DEDUPLICATION enabled:
     Create mention_key = f"{text.lower()}:{start}:{end}"
     Skip if mention_key already seen
     Add to seen_mentions set
   ```

7. **Normalization**:
   ```
   For each entity:
     normalized = text.lower().strip()
     Remove extra whitespace
   ```

**Performance**:
- Short text (<1500 chars): ~500ms per document
- Long text (>1500 chars): ~2-3 seconds per document
- Scales linearly with number of chunks

### Batch Processing

**Method**: `extract_batch(texts, batch_size=None) -> List[List[Dict]]`

Processes multiple documents in batches for improved throughput.

**Input**:
- `texts`: List of text strings
- `batch_size`: Batch size (default from config.GLINER_BATCH_SIZE = 8)

**Output**: List of entity lists (one per input text)

**Implementation**:
```
For i in range(0, len(texts), batch_size):
  batch = texts[i:i+batch_size]
  For text in batch:
    results.append(extract(text))
```

Note: Current implementation processes serially within batches. Future optimization: parallel processing.

### Type Utilities

**Method**: `get_entity_types(category=None) -> List[str]`

Returns available entity types, optionally filtered by category.

**Example**:
```python
extractor.get_entity_types()  # All 300 types
extractor.get_entity_types("Framework")  # Only Framework::* types
```

**Method**: `get_type_hierarchy_tree() -> Dict`

Builds nested dictionary representing type hierarchy.

**Output**:
```python
{
  "Framework": {
    "Backend": {
      "Python": {"_full_type": "Framework::Backend::Python"},
      "JavaScript": {"_full_type": "Framework::Backend::JavaScript"}
    }
  }
}
```

### Configuration

**Settings** (from `config.py`):
- `GLINER_MODEL`: Model name (urchade/gliner_large-v2.1)
- `GLINER_THRESHOLD`: Confidence threshold (0.4)
- `GLINER_BATCH_SIZE`: Batch processing size (8)
- `GLINER_MAX_LENGTH`: Token limit (384)
- `ENTITY_TAXONOMY_PATH`: Taxonomy file path
- `ENTITY_MIN_CONFIDENCE`: Minimum confidence (0.4)
- `ENTITY_DEDUPLICATION`: Enable dedup (True)

### Error Handling

**Model Load Failure**:
- Logs error with full traceback
- Raises exception (service cannot start)

**Empty Text**:
- Logs warning
- Returns empty list

**Inference Error**:
- Logs error with exception details
- Raises exception to caller

## Relationship Extractor

### Purpose

Extract semantic relationships between entities using large language models (LLMs) via vLLM. Identifies relationship types (uses, implements, extends, etc.) and validates against extracted entities.

### Architecture

**Class**: `RelationshipExtractor`

**Dependencies**:
- vLLM client (HTTP connection to inference server)
- Configuration settings (temperature, max tokens, etc.)

**Singleton Pattern**: Global instance via `get_relation_extractor()`

### Initialization

```
Constructor Parameters:
- None (uses global config)

Initialization:
- Sets vllm_client to None (lazy initialization)
- Loads relationship type definitions (8 categories, ~60 types)
- Configures extraction parameters:
  - max_entity_distance: 3 sentences (unused in current impl)
  - min_confidence: 0.45
  - context_window: 200 characters
```

### Relationship Types

**Categories**:

1. **Technical** (13 types):
   - uses, implements, extends, depends_on, requires
   - provides, supports, integrates_with, based_on
   - built_with, powered_by, runs_on, compatible_with

2. **Comparison** (7 types):
   - similar_to, alternative_to, competes_with, differs_from
   - replaces, supersedes, evolved_from

3. **Hierarchical** (8 types):
   - part_of, contains, includes, composed_of
   - category_of, type_of, instance_of, subclass_of

4. **Functional** (8 types):
   - processes, generates, transforms, analyzes
   - validates, handles, manages, controls

5. **Development** (6 types):
   - developed_by, maintained_by, created_by, designed_by
   - contributed_to, sponsored_by

6. **Documentation** (5 types):
   - documented_in, described_in, defined_in
   - referenced_in, mentioned_in

7. **Configuration** (5 types):
   - configured_with, settings_for, parameter_of
   - option_for, enabled_by

8. **Performance** (5 types):
   - optimizes, improves, accelerates, scales_with
   - benchmarked_against

**Total**: ~60 predefined relationship types, extensible by LLM

### Extraction Process

**Method**: `extract_relationships(text, entities) -> List[Dict]`

**Input**:
- `text`: Full document text
- `entities`: List of entity dicts from EntityExtractor

**Output**: List of relationship dictionaries:
```python
{
  "subject_text": "FastAPI",
  "subject_normalized": "fastapi",
  "subject_type": "Framework::Backend::Python",
  "predicate": "uses",
  "object_text": "Pydantic",
  "object_normalized": "pydantic",
  "object_type": "Programming::Library",
  "confidence": 0.88,
  "context": "FastAPI uses Pydantic for data validation",
  "subject_start": 120,
  "subject_end": 127,
  "object_start": 145,
  "object_end": 153
}
```

**Processing Steps**:

1. **Entity Validation**:
   ```
   If entities empty:
     Log info message
     Return empty list
   ```

2. **Entity Conversion**:
   ```
   For each entity dict:
     Find position in text (use provided start/end or text.find())
     Create EntityMention dataclass:
       - text, normalized, type_full, type_primary
       - start_pos, end_pos
       - confidence, sentence, context
   ```

3. **Document Chunking** (for long documents):
   ```
   If len(text) > 30000 chars:
     Split into sections of 30000 chars
     Add 3000 char overlap between sections
     Filter entities per section by position
     Extract relationships from each section
     Deduplicate across sections
   Else:
     Process entire document
   ```

4. **vLLM Client Initialization**:
   ```
   If vllm_client is None:
     Call get_vllm_client() (singleton)
     Call ensure_model() to discover model name
     Log model name
   ```

5. **Prompt Construction**:
   ```
   Build prompt with:
     - Full document text
     - Numbered list of entities with types
     - Relationship type categories
     - Output format specification (JSON array)
     - Example relationships
     - Strict rules:
       * Only explicit relationships in text
       * Entities must match list exactly
       * snake_case predicates
       * Confidence reflects clarity
       * Context quote from text
       * Empty array if no relationships
   ```

6. **vLLM Inference**:
   ```
   Call vllm_client.complete():
     - prompt: Built prompt
     - max_tokens: 65536 (large for guided JSON)
     - temperature: 0.1 (low for consistency)

   LLM Processing:
     - Reads document and entities
     - Identifies relationships mentioned in text
     - Formats as JSON array
     - Returns JSON string

   Response: Raw text (may contain markdown, extra text)
   ```

7. **Response Parsing**:
   ```
   Clean response:
     - Remove markdown code fences (```json, ```)
     - Find all JSON arrays in response
     - Handle multiple arrays (example vs actual)
     - Select longest array (actual data)

   Parse JSON:
     - json.loads() on extracted array
     - Return empty list if parse fails
   ```

8. **Relationship Validation**:
   ```
   For each relationship in parsed array:
     Verify required fields: subject, predicate, object, confidence
     Build entity lookup: normalized → EntityMention
     Find subject entity by normalized text
     Find object entity by normalized text
     Skip if either entity not found
     Skip if subject == object (self-relationship)
     Check confidence >= min_confidence (0.45)
     Add entity type information
     Normalize predicate (lowercase, snake_case)
     Truncate context to 500 chars
   ```

9. **Deduplication**:
   ```
   Group by (subject_normalized, predicate, object_normalized)
   For duplicates, keep highest confidence
   ```

**Performance**:
- vLLM latency: ~5-10 seconds per document (depends on LLM and document length)
- Bottleneck: LLM inference (network + compute)

### Prompt Engineering

**Prompt Structure**:

```
You are an expert at extracting semantic relationships between entities in technical documentation.

DO NOT RETURN A SUMMARY.
DO NOT EXPLAIN YOUR CHOICES.
DO NOT ADD ANYTHING OUTSIDE OF THE JSON

**Text:**
[Full document text]

**Entities:**
1. **FastAPI** (Framework)
2. **Pydantic** (Library)
...

**Task:**
Identify meaningful relationships between the entities above. Focus on explicit relationships mentioned in the text.

**Relationship Types (organized by category):**
- **Technical**: uses, implements, extends, ...
- **Comparison**: similar_to, alternative_to, ...
...

Use the most appropriate relationship type from the categories above, or create similar snake_case predicates if needed.

**Output Format:**
Return a JSON array of relationships. Each relationship should have:
- subject: The entity name (must match one from the list above)
- predicate: The relationship type (use snake_case)
- object: The target entity name (must match one from the list above)
- confidence: Float between 0 and 1
- context: Brief supporting text from the document

**Example:**
[JSON example with 2 relationships]

**Important Rules:**
1. Only extract relationships explicitly stated in the text
2. Subject and object MUST be entity names from the list above
3. Use lowercase snake_case for predicates
4. Confidence should reflect clarity (lower = weaker relationship)
5. Context should be 50-100 words quote
6. Return empty array [] if no clear relationships
7. Focus on meaningful relationships, not trivial mentions

Return ONLY the JSON array, no additional text.
DO NOT RETURN A SUMMARY.
DO NOT EXPLAIN YOUR CHOICES.
DO NOT ADD ANYTHING OUTSIDE OF THE JSON
```

**Prompt Design Principles**:
- Clear task definition
- Explicit output format with example
- Strict constraints to prevent hallucination
- Emphasis on text-based evidence
- Entity list prevents fabrication
- Relationship type suggestions guide LLM
- Multiple reminders to return only JSON

### Error Handling

**vLLM Unavailable**:
- `ensure_model()` returns False
- Logs warning
- Returns empty relationship list (graceful degradation)

**vLLM Request Failure**:
- Catches httpx.HTTPError
- Logs error with status code and response preview
- Resets model state (triggers reconnect on next request)
- Raises ModelUnavailableError

**JSON Parse Failure**:
- Logs warning with response preview
- Returns empty list

**Entity Not Found**:
- Logs debug message
- Skips relationship (doesn't fail entire extraction)

**Invalid Relationship**:
- Logs warning
- Skips relationship

### Configuration

**Settings** (from `config.py`):
- `VLLM_BASE_URL`: vLLM server URL (http://localhost:8078)
- `VLLM_TIMEOUT`: Request timeout (600 seconds)
- `VLLM_MAX_TOKENS`: Max generated tokens (65536)
- `VLLM_TEMPERATURE`: Sampling temperature (0.1)
- `VLLM_RETRY_INTERVAL`: Retry interval (30 seconds)
- `VLLM_MAX_RETRIES`: Max retry attempts (3)
- `RELATION_MIN_CONFIDENCE`: Min confidence (0.45)
- `RELATION_MAX_DISTANCE`: Max sentence distance (3, unused)
- `RELATION_CONTEXT_WINDOW`: Context window (200 chars)

## Integration

### Pipeline Integration

**Entity Extraction Flow**:
```
KGProcessor.process_document()
  → EntityExtractor.extract(markdown)
  → Returns entities[]
  → Pass to RelationshipExtractor
```

**Relationship Extraction Flow**:
```
KGProcessor.process_document()
  → RelationshipExtractor.extract_relationships(markdown, entities)
  → Returns relationships[]
  → Pass to ChunkMapper
```

### Data Flow

```
Input: Full document markdown
  ↓
EntityExtractor
  ↓
Entities with positions
  ↓
RelationshipExtractor
  ↓
Relationships with entity pairs
  ↓
Output: Knowledge tuples (entities + relationships)
```

## Performance Optimization

### Entity Extraction

**Optimization Techniques**:
1. **Automatic Chunking**: Split long documents for GLiNER token limit
2. **Type Caching**: Cache parsed hierarchies to avoid repeated string splits
3. **Early Exit**: Return empty list for empty input
4. **Deduplication**: Set-based tracking for O(1) lookups

**Bottlenecks**:
- GLiNER model inference (GPU-accelerated if available)
- Model loading on first run (~30s, one-time cost)

**Scaling**:
- CPU-bound: Benefits from multi-core CPUs
- Memory: Requires ~4GB for model weights
- Parallelization: Process multiple documents concurrently (separate instances)

### Relationship Extraction

**Optimization Techniques**:
1. **Lazy vLLM Initialization**: Only connect when first needed
2. **Connection Pooling**: Reuse HTTP connections via httpx client
3. **Timeout Configuration**: Prevent hanging requests
4. **Deduplication**: Reduce redundant relationships

**Bottlenecks**:
- vLLM inference latency (5-10s per document)
- Network round-trip time
- LLM model size (larger = slower)

**Scaling**:
- Network-bound: Limited by vLLM server throughput
- Parallelization: Multiple vLLM servers with load balancing
- Model optimization: Quantized models for faster inference

## Testing

### Unit Testing

**EntityExtractor Tests**:
```python
# Test basic extraction
entities = extractor.extract("FastAPI is a Python framework")
assert len(entities) > 0
assert any(e["text"] == "FastAPI" for e in entities)

# Test empty input
assert extractor.extract("") == []

# Test long document chunking
long_text = "word " * 1000
entities = extractor.extract(long_text)

# Test hierarchy parsing
entity = entities[0]
assert entity["type_primary"] is not None
assert entity["type_full"].startswith(entity["type_primary"])
```

**RelationshipExtractor Tests**:
```python
# Mock entities
entities = [
  {"text": "FastAPI", "normalized": "fastapi", "type_full": "Framework"},
  {"text": "Pydantic", "normalized": "pydantic", "type_full": "Library"}
]

# Test extraction
text = "FastAPI uses Pydantic for validation"
relationships = await extractor.extract_relationships(text, entities)
assert len(relationships) > 0
assert relationships[0]["predicate"] in ["uses", "integrates_with"]

# Test empty entities
assert await extractor.extract_relationships(text, []) == []
```

### Integration Testing

See `tests/test_relationship_extractor.py` for real-world tests with vLLM server.

---

[Next: Pipeline Module Documentation](pipeline.md)
