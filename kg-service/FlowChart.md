NEW Flow (Graph-Enhanced RAG):
User query: "FastAPI best practices"
   ↓
1. 🔍 Neo4j Graph Exploration (FIRST!)
   │
   ├─ Parse query for entities: "FastAPI"
   │
   ├─ Query Neo4j:
   │   MATCH (e:Entity {text: "FastAPI"})
   │   MATCH (e)-[r:RELATED_TO*1..N]-(related:Entity)  // N = exploration depth
   │   MATCH (related)<-[:CONTAINS]-(chunk:Chunk)
   │   RETURN chunk.vector_rowid, chunk.content_id, related.text
   │
   ├─ Results:
   │   - chunk_rowids: [45001, 45003, 45123, 45201, ...]
   │   - related_entities: ["Pydantic", "Uvicorn", "async", "pytest", ...]
   │   - content_ids: [123, 456, 789, ...]
   │
   ↓
2. 🎯 Vector Search (SECOND - but now graph-informed!)
   │
   ├─ Expand query with graph entities:
   │   Original: "FastAPI best practices"
   │   Expanded: "FastAPI Pydantic Uvicorn async pytest best practices"
   │
   ├─ Embed expanded query → Vector search
   │
   ├─ Optional: Boost chunks found by graph:
   │   • Chunks in graph results get +0.2 similarity score boost
   │   • Prioritize graph-connected content
   │
   ↓
3. 📊 Combine & Rank Results
   │
   ├─ Merge:
   │   - Graph-based chunks (authoritative - mentions FastAPI)
   │   - Vector search chunks (semantic - best practices)
   │
   ├─ De-duplicate by chunk_rowid
   │
   ├─ Re-rank by:
   │   - Combined score = (vector_similarity * 0.6) + (graph_relevance * 0.4)
   │   - graph_relevance = relationship_count + entity_match_count
   │
   ↓
4. 🔄 Fetch Full Content from SQLite
   │
   ├─ For each top chunk_rowid:
   │   SELECT cc.url, cc.title, cc.markdown, ce.entity_text
   │   FROM content_vectors cv
   │   JOIN crawled_content cc ON cv.content_id = cc.id
   │   LEFT JOIN chunk_entities ce ON cv.rowid = ce.chunk_rowid
   │   WHERE cv.rowid IN (45001, 45003, ...)
   │
   ↓
5. 📝 Enrich Results with Graph Context
   │
   ├─ For each result, add:
   │   {
   │     "url": "...",
   │     "title": "...",
   │     "content": "chunk text...",
   │     "similarity_score": 0.89,
   │     "graph_context": {
   │       "entities_in_chunk": ["FastAPI", "Pydantic"],
   │       "relationships": [
   │         {"FastAPI": "uses" → "Pydantic"},
   │         {"FastAPI": "competes_with" → "Django"}
   │       ],
   │       "related_entities_nearby": ["Uvicorn", "async", "pytest"],
   │       "cross_document_connections": 3  // appears with same entities in 3 other docs
   │     }
   │   }
   │
   ↓
6. ✅ Return Enhanced Results to User