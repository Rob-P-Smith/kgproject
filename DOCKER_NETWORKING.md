# Docker Container-to-Container Communication

This document explains how Neo4j and mcpragcrawl4ai communicate within Docker networks.

## Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network: crawler_default           │
│                                                               │
│  ┌──────────────────────┐         ┌──────────────────────┐  │
│  │   neo4j-kg           │         │ crawl4ai-rag-server  │  │
│  │                      │         │                      │  │
│  │  - Port 7474 (HTTP)  │◄────────┤  - Port 8080 (API)   │  │
│  │  - Port 7687 (Bolt)  │         │  - Port 3000 (MCP)   │  │
│  │                      │         │                      │  │
│  │  Container name:     │         │  Container name:     │  │
│  │  neo4j-kg            │         │  crawl4ai-rag-server │  │
│  └──────────────────────┘         └──────────────────────┘  │
│           │                                    │              │
└───────────┼────────────────────────────────────┼─────────────┘
            │                                    │
            │ Exposed to host                    │ Exposed to host
            ▼                                    ▼
    localhost:7474                      localhost:8080
    localhost:7687                      localhost:3000
```

## Connection URLs

### From Host Machine (Your Computer)

When connecting from Python scripts or Neo4j Browser on your local machine:

```bash
# Neo4j Browser
http://localhost:7474

# Python neo4j driver
bolt://localhost:7687

# mcpragcrawl4ai API
http://localhost:8080
```

### From Container to Container (Inside Docker Network)

When containers communicate with each other, use container names:

#### From mcpragcrawl4ai to Neo4j:
```python
# Use container name 'neo4j-kg'
NEO4J_URI = "bolt://neo4j-kg:7687"
NEO4J_HTTP = "http://neo4j-kg:7474"
```

#### From Neo4j to mcpragcrawl4ai:
```python
# Use container name 'crawl4ai-rag-server'
MCPRAG_API = "http://crawl4ai-rag-server:8080"
```

## Setup Instructions

### 1. Ensure crawler_default Network Exists

First, make sure the crawler_default network is created:

```bash
# Check if network exists
docker network ls | grep crawler_default

# If it doesn't exist, create it
docker network create crawler_default
```

### 2. Start mcpragcrawl4ai Service

```bash
cd /home/robiloo/Documents/raid/mcpragcrawl4ai/deployments/server
docker compose up -d
```

This creates/uses the `crawler_default` network.

### 3. Start Neo4j Service

```bash
cd /home/robiloo/Documents/KG-project
docker compose up -d
```

Neo4j will join the existing `crawler_default` network.

### 4. Verify Network Connectivity

```bash
# Check both containers are on the same network
docker network inspect crawler_default

# Should show both:
# - neo4j-kg
# - crawl4ai-rag-server
```

## Testing Container Communication

### Test 1: Neo4j to mcpragcrawl4ai

```bash
# Execute inside Neo4j container
docker exec -it neo4j-kg bash

# Inside container - test connectivity
curl http://crawl4ai-rag-server:8080/health

# Should return health check response
```

### Test 2: mcpragcrawl4ai to Neo4j

```bash
# Execute inside mcpragcrawl4ai container
docker exec -it crawl4ai-rag-server bash

# Inside container - test connectivity
curl http://neo4j-kg:7474

# Should return Neo4j browser HTML
```

### Test 3: Python Connection from mcpragcrawl4ai

Create a test script in your mcpragcrawl4ai container:

```python
from neo4j import GraphDatabase

# Use container name, not localhost
driver = GraphDatabase.driver(
    "bolt://neo4j-kg:7687",
    auth=("neo4j", "knowledge_graph_2024")
)

with driver.session() as session:
    result = session.run("RETURN 'Connected!' as message")
    print(result.single()["message"])

driver.close()
```

## Integration Example

### Add Neo4j to mcpragcrawl4ai

Update your mcpragcrawl4ai service to connect to Neo4j:

#### 1. Update mcpragcrawl4ai .env file

```bash
# Add to /home/robiloo/Documents/raid/mcpragcrawl4ai/deployments/server/.env
NEO4J_URI=bolt://neo4j-kg:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=knowledge_graph_2024
```

#### 2. Install neo4j driver in mcpragcrawl4ai

Update `requirements.txt`:
```
neo4j>=5.14.0
```

Rebuild the container:
```bash
cd /home/robiloo/Documents/raid/mcpragcrawl4ai/deployments/server
docker compose down
docker compose build --no-cache
docker compose up -d
```

#### 3. Add Knowledge Graph Extractor

Create `core/data/kg_extractor.py`:

```python
import os
from neo4j import GraphDatabase
from typing import List, Dict, Any

class KnowledgeGraphExtractor:
    """Extract entities and relationships to Neo4j"""

    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://neo4j-kg:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "knowledge_graph_2024")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def store_entities(self, url: str, entities: List[Dict[str, Any]]):
        """Store extracted entities in Neo4j"""
        with self.driver.session() as session:
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {text: $text, type: $type})
                    ON CREATE SET e.first_seen = timestamp()
                    SET e.source_urls = CASE
                        WHEN e.source_urls IS NULL THEN [$url]
                        WHEN NOT $url IN e.source_urls THEN e.source_urls + $url
                        ELSE e.source_urls
                    END
                """, text=entity['text'], type=entity['type'], url=url)

    def store_relationship(self, subject: str, predicate: str,
                          object: str, url: str, confidence: float = 1.0):
        """Store relationship between entities"""
        with self.driver.session() as session:
            session.run("""
                MATCH (s:Entity {text: $subject})
                MATCH (o:Entity {text: $object})
                MERGE (s)-[r:RELATES {type: $predicate}]->(o)
                ON CREATE SET r.confidence = $confidence
                SET r.source_url = $url,
                    r.last_updated = timestamp()
            """, subject=subject, object=object, predicate=predicate,
                url=url, confidence=confidence)

    def query_entity_context(self, entity_text: str, depth: int = 2) -> List[Dict]:
        """Query related entities for RAG context enrichment"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (e:Entity {text: $entity})-[*1..$depth]-(related)
                RETURN DISTINCT related.text as entity,
                       related.type as type,
                       [rel in relationships(path) | type(rel)] as relationships
                LIMIT 20
            """, entity=entity_text, depth=depth)
            return [dict(record) for record in result]
```

#### 4. Integrate with RAG Pipeline

Update `core/operations/crawler.py` to call KG extraction:

```python
from core.data.kg_extractor import KnowledgeGraphExtractor

# In your crawl function:
kg = KnowledgeGraphExtractor()
try:
    # After storing in SQLite:
    entities = extract_entities(content)  # Your NER logic
    kg.store_entities(url, entities)

    relationships = extract_relationships(content)  # Your RE logic
    for rel in relationships:
        kg.store_relationship(
            rel['subject'],
            rel['predicate'],
            rel['object'],
            url,
            rel.get('confidence', 1.0)
        )
finally:
    kg.close()
```

## Troubleshooting

### Issue: "Could not connect to neo4j-kg:7687"

**Solution:**
1. Verify both containers are running:
   ```bash
   docker ps | grep -E 'neo4j-kg|crawl4ai-rag-server'
   ```

2. Check both are on crawler_default network:
   ```bash
   docker network inspect crawler_default
   ```

3. Restart services in order:
   ```bash
   cd /home/robiloo/Documents/KG-project
   docker compose restart neo4j

   cd /home/robiloo/Documents/raid/mcpragcrawl4ai/deployments/server
   docker compose restart
   ```

### Issue: "Network crawler_default not found"

**Solution:**
```bash
# Create the network
docker network create crawler_default

# Restart both services
```

### Issue: "Connection refused on port 7687"

**Solution:**
1. Wait for Neo4j to fully start (can take 30-60 seconds)
2. Check Neo4j logs:
   ```bash
   docker logs neo4j-kg
   ```
3. Look for "Started." message

### Issue: DNS resolution fails

**Solution:**
- Use container names exactly as defined in docker-compose.yml
- Neo4j container: `neo4j-kg`
- mcpragcrawl4ai container: `crawl4ai-rag-server`

## Port Reference

| Service | Container Name | Internal Port | Host Port | Purpose |
|---------|---------------|---------------|-----------|---------|
| Neo4j HTTP | neo4j-kg | 7474 | 7474 | Browser UI |
| Neo4j Bolt | neo4j-kg | 7687 | 7687 | Driver connections |
| mcpragcrawl4ai API | crawl4ai-rag-server | 8080 | 8080 | REST API |
| mcpragcrawl4ai MCP | crawl4ai-rag-server | 3000 | 3000 | MCP Server |

## Environment Variables Summary

### For Local Development (Host Machine)
```bash
NEO4J_URI=bolt://localhost:7687
```

### For Container Communication
```bash
NEO4J_URI=bolt://neo4j-kg:7687
MCPRAG_API_URL=http://crawl4ai-rag-server:8080
```

## Best Practices

1. **Always use container names** for inter-container communication
2. **Use localhost** only when connecting from host machine
3. **Check logs** if connection fails:
   ```bash
   docker logs neo4j-kg
   docker logs crawl4ai-rag-server
   ```
4. **Verify network** membership before debugging connectivity
5. **Wait for health checks** to pass before connecting

## Next Steps

1. Implement entity extraction in mcpragcrawl4ai
2. Add relationship extraction logic
3. Create enriched search that combines:
   - Vector similarity (SQLite + sqlite-vec)
   - Entity relationships (Neo4j)
4. Build visualization dashboard using Neo4j Browser
