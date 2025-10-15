# Knowledge Graph System with Neo4j

A Neo4j-based knowledge graph system running in Docker, designed for entity extraction, relationship mapping, and graph analytics.

**Important:** For Docker container-to-container communication with mcpragcrawl4ai, see [DOCKER_NETWORKING.md](DOCKER_NETWORKING.md)

## Quick Start

### 1. Start Neo4j

```bash
cd /home/robiloo/Documents/KG-project
docker compose up -d
```

Wait ~30 seconds for Neo4j to initialize.

### 2. Access Neo4j Browser

Open your web browser and navigate to:
```
http://localhost:7474
```

**Login credentials:**
- Username: `neo4j`
- Password: `knowledge_graph_2024`

### 3. Test Connection

```bash
# Install Python dependencies first
pip install neo4j python-dotenv

# Run test script
python3 scripts/test_connection.py
```

## Directory Structure

```
KG-project/
├── docker-compose.yml       # Neo4j container configuration
├── .env                     # Environment variables (DO NOT COMMIT)
├── neo4j/
│   ├── data/               # Graph database files (persistent)
│   ├── logs/               # Neo4j logs
│   ├── import/             # Drop CSV/JSON files here for bulk import
│   └── plugins/            # Neo4j plugins (APOC auto-installed)
├── scripts/
│   └── test_connection.py  # Connection test script
└── README.md               # This file
```

## Common Commands

### Docker Management

```bash
# Start Neo4j
docker compose up -d

# Stop Neo4j
docker compose down

# View logs
docker compose logs -f neo4j

# Restart Neo4j
docker compose restart neo4j

# Check status
docker compose ps
```

### Neo4j Management

```bash
# Access Neo4j container shell
docker exec -it neo4j-kg bash

# Run cypher-shell (inside container)
docker exec -it neo4j-kg cypher-shell -u neo4j -p knowledge_graph_2024
```

## Python Integration

### Basic Connection

```python
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jConnection:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "knowledge_graph_2024")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, cypher, parameters=None):
        with self.driver.session() as session:
            result = session.run(cypher, parameters)
            return [record for record in result]

# Usage
db = Neo4jConnection()
results = db.query("MATCH (n) RETURN count(n) as count")
print(f"Total nodes: {results[0]['count']}")
db.close()
```

### Create Entities and Relationships

```python
def create_knowledge_graph(session):
    # Create entities
    session.run("""
        CREATE (e1:Entity {name: 'Python', type: 'Technology'})
        CREATE (e2:Entity {name: 'Machine Learning', type: 'Field'})
        CREATE (e3:Entity {name: 'Neo4j', type: 'Database'})

        # Create relationships
        CREATE (e1)-[:USED_IN]->(e2)
        CREATE (e3)-[:STORES]->(e1)
    """)
```

### Query Relationships

```python
def find_connections(session, entity_name):
    result = session.run("""
        MATCH (e:Entity {name: $name})-[r]->(target)
        RETURN e.name as source, type(r) as relationship, target.name as target
    """, name=entity_name)

    for record in result:
        print(f"{record['source']} -{record['relationship']}-> {record['target']}")
```

## Cypher Query Examples

### Create Nodes

```cypher
// Create a single entity
CREATE (p:Person {name: 'Alice', role: 'Engineer'})

// Create multiple related entities
CREATE (p:Person {name: 'Bob'})
CREATE (t:Technology {name: 'Python'})
CREATE (p)-[:KNOWS]->(t)
```

### Query Patterns

```cypher
// Find all persons
MATCH (p:Person)
RETURN p.name, p.role

// Find relationships
MATCH (p:Person)-[r]->(t:Technology)
RETURN p.name, type(r), t.name

// Find paths
MATCH path = (start)-[*1..3]->(end)
WHERE start.name = 'Alice'
RETURN path

// Count nodes by label
MATCH (n)
RETURN labels(n)[0] as label, count(n) as count
```

### Update and Delete

```cypher
// Update properties
MATCH (p:Person {name: 'Alice'})
SET p.email = 'alice@example.com'

// Delete specific nodes and their relationships
MATCH (p:Person {name: 'Bob'})
DETACH DELETE p

// Delete everything (careful!)
MATCH (n)
DETACH DELETE n
```

### Advanced Queries

```cypher
// Find shortest path
MATCH path = shortestPath(
  (start:Person {name: 'Alice'})-[*]-(end:Person {name: 'Bob'})
)
RETURN path

// Find highly connected nodes
MATCH (n)-[r]->()
RETURN n.name, count(r) as connections
ORDER BY connections DESC
LIMIT 10

// Pattern matching with WHERE clause
MATCH (p:Person)-[:KNOWS]->(t:Technology)
WHERE t.name CONTAINS 'Python'
RETURN p.name, t.name
```

## APOC Procedures

APOC (Awesome Procedures on Cypher) is pre-installed and provides extended functionality:

```cypher
// Generate UUID
RETURN apoc.create.uuid() as id

// Load JSON data
CALL apoc.load.json('file:///var/lib/neo4j/import/data.json')
YIELD value
CREATE (n:Entity {name: value.name})

// Export to JSON
CALL apoc.export.json.all('export.json', {})

// Run periodic commit (for large imports)
CALL apoc.periodic.iterate(
  "LOAD CSV FROM 'file:///import/data.csv' AS row RETURN row",
  "CREATE (n:Node {id: row[0], name: row[1]})",
  {batchSize: 1000, parallel: true}
)
```

## Integration with mcpragcrawl4ai

To integrate this knowledge graph with your RAG system:

### 1. Install Neo4j Python Driver

```bash
cd /home/robiloo/Documents/raid/mcpragcrawl4ai
source .venv/bin/activate
pip install neo4j
```

### 2. Extract Entities During Crawl

```python
# Add to your RAG pipeline
from neo4j import GraphDatabase

class KnowledgeGraphExtractor:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    def extract_and_store(self, content, url):
        """Extract entities from content and store in Neo4j"""
        entities = self.extract_entities(content)  # Your NER logic
        relationships = self.extract_relationships(content)  # Your RE logic

        with self.driver.session() as session:
            # Store entities
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {text: $text, type: $type})
                    SET e.source_urls = CASE
                        WHEN e.source_urls IS NULL THEN [$url]
                        ELSE e.source_urls + $url
                    END
                """, text=entity['text'], type=entity['type'], url=url)

            # Store relationships
            for rel in relationships:
                session.run("""
                    MATCH (s:Entity {text: $subject})
                    MATCH (o:Entity {text: $object})
                    MERGE (s)-[r:RELATES {type: $predicate}]->(o)
                    SET r.source_url = $url
                """, subject=rel['subject'], object=rel['object'],
                    predicate=rel['predicate'], url=url)
```

### 3. Query Graph During Search

```python
def enhanced_search(query, rag_db, neo4j_session):
    # Get RAG results
    rag_results = rag_db.search(query)

    # Get related entities from knowledge graph
    graph_results = neo4j_session.run("""
        MATCH (e:Entity)-[r*1..2]-(related)
        WHERE e.text CONTAINS $query
        RETURN e.text, type(r[0]) as relationship, related.text
        LIMIT 10
    """, query=query)

    # Combine results for enriched context
    return {
        'rag_content': rag_results,
        'graph_connections': [dict(record) for record in graph_results]
    }
```

## Configuration

### Memory Settings

Edit [.env](.env) to adjust memory allocation:

```bash
# For systems with 16GB+ RAM
NEO4J_HEAP_INITIAL=1G
NEO4J_HEAP_MAX=4G
NEO4J_PAGECACHE=2G

# For systems with 8GB RAM
NEO4J_HEAP_INITIAL=512m
NEO4J_HEAP_MAX=2G
NEO4J_PAGECACHE=1G
```

### Port Configuration

If ports 7474 or 7687 conflict with other services, change them in [.env](.env):

```bash
NEO4J_HTTP_PORT=7475
NEO4J_BOLT_PORT=7688
```

Then update [docker-compose.yml](docker-compose.yml) ports section.

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs neo4j

# Common issue: port already in use
sudo lsof -i :7474
sudo lsof -i :7687

# Kill conflicting process or change ports in .env
```

### Connection Refused

```bash
# Wait for Neo4j to fully start (can take 30-60 seconds)
docker compose logs -f neo4j

# Look for: "Started."
```

### Permission Errors

```bash
# Fix directory permissions
sudo chown -R $(whoami):$(whoami) neo4j/
```

### Out of Memory

```bash
# Reduce memory settings in .env
NEO4J_HEAP_MAX=1G
NEO4J_PAGECACHE=512m

# Restart
docker compose restart neo4j
```

## Backup and Restore

### Backup

```bash
# Stop Neo4j
docker compose down

# Backup data directory
tar -czf neo4j-backup-$(date +%Y%m%d).tar.gz neo4j/data/

# Restart
docker compose up -d
```

### Restore

```bash
# Stop Neo4j
docker compose down

# Restore data directory
rm -rf neo4j/data/
tar -xzf neo4j-backup-YYYYMMDD.tar.gz

# Restart
docker compose up -d
```

## Security Notes

1. **Change default password** - Edit `.env` and update `NEO4J_PASSWORD`
2. **Add .env to .gitignore** - Never commit credentials
3. **Use strong passwords** in production (20+ characters)
4. **Restrict network access** - In production, don't expose ports publicly
5. **Enable SSL/TLS** - For production deployments over networks

## Resources

- **Neo4j Documentation**: https://neo4j.com/docs/
- **Cypher Query Language**: https://neo4j.com/docs/cypher-manual/
- **Python Driver**: https://neo4j.com/docs/python-manual/
- **APOC Documentation**: https://neo4j.com/docs/apoc/
- **Graph Data Science**: https://neo4j.com/docs/graph-data-science/

## License

This setup is provided as-is for the KG-project. Neo4j Community Edition is licensed under GPLv3.
