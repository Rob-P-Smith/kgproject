"""
Neo4j Graph Schema Definition and Initialization

Defines the schema for the knowledge graph including:
- Node types (Document, Chunk, Entity)
- Relationship types (HAS_CHUNK, MENTIONED_IN, various semantic relationships)
- Indexes and constraints
- Schema initialization
"""

import logging
from typing import List, Dict, Any

from storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphSchema:
    """Knowledge graph schema manager"""

    # Node labels
    NODE_DOCUMENT = "Document"
    NODE_CHUNK = "Chunk"
    NODE_ENTITY = "Entity"

    # Relationship types (structural)
    REL_HAS_CHUNK = "HAS_CHUNK"
    REL_MENTIONED_IN = "MENTIONED_IN"
    REL_CO_OCCURS = "CO_OCCURS_WITH"

    # Common semantic relationship types
    # (Note: actual relationships are dynamic based on extraction)
    SEMANTIC_RELATIONSHIPS = [
        "USES", "IMPLEMENTS", "EXTENDS", "DEPENDS_ON", "REQUIRES",
        "PROVIDES", "SUPPORTS", "INTEGRATES_WITH", "BASED_ON",
        "BUILT_WITH", "POWERED_BY", "RUNS_ON", "COMPATIBLE_WITH",
        "SIMILAR_TO", "ALTERNATIVE_TO", "COMPETES_WITH", "DIFFERS_FROM",
        "REPLACES", "SUPERSEDES", "EVOLVED_FROM",
        "PART_OF", "CONTAINS", "INCLUDES", "COMPOSED_OF",
        "CATEGORY_OF", "TYPE_OF", "INSTANCE_OF", "SUBCLASS_OF",
        "PROCESSES", "GENERATES", "TRANSFORMS", "ANALYZES",
        "VALIDATES", "HANDLES", "MANAGES", "CONTROLS",
        "DEVELOPED_BY", "MAINTAINED_BY", "CREATED_BY", "DESIGNED_BY",
        "DOCUMENTED_IN", "DESCRIBED_IN", "DEFINED_IN", "REFERENCED_IN",
        "CONFIGURED_WITH", "SETTINGS_FOR", "PARAMETER_OF", "OPTION_FOR",
        "OPTIMIZES", "IMPROVES", "ACCELERATES", "SCALES_WITH"
    ]

    def __init__(self, client: Neo4jClient):
        """
        Initialize schema manager

        Args:
            client: Connected Neo4j client
        """
        self.client = client

    async def initialize_schema(self) -> Dict[str, Any]:
        """
        Initialize complete graph schema

        Creates:
        - Constraints for unique identifiers
        - Indexes for performance
        - Any required initial nodes

        Returns:
            Dictionary with initialization results
        """
        logger.info("Initializing Neo4j graph schema...")

        results = {
            "constraints_created": 0,
            "indexes_created": 0,
            "errors": []
        }

        # Create constraints
        constraint_results = await self._create_constraints()
        results["constraints_created"] = constraint_results["created"]
        results["errors"].extend(constraint_results["errors"])

        # Create indexes
        index_results = await self._create_indexes()
        results["indexes_created"] = index_results["created"]
        results["errors"].extend(index_results["errors"])

        logger.info(
            f"Schema initialization complete: "
            f"{results['constraints_created']} constraints, "
            f"{results['indexes_created']} indexes"
        )

        if results["errors"]:
            logger.warning(f"Encountered {len(results['errors'])} errors during initialization")

        return results

    async def _create_constraints(self) -> Dict[str, Any]:
        """Create uniqueness constraints"""

        constraints = [
            # Document: unique content_id
            {
                "name": "unique_document_content_id",
                "query": """
                    CREATE CONSTRAINT unique_document_content_id IF NOT EXISTS
                    FOR (d:Document) REQUIRE d.content_id IS UNIQUE
                """
            },
            # Chunk: unique vector_rowid
            {
                "name": "unique_chunk_rowid",
                "query": """
                    CREATE CONSTRAINT unique_chunk_rowid IF NOT EXISTS
                    FOR (c:Chunk) REQUIRE c.vector_rowid IS UNIQUE
                """
            },
            # Entity: unique normalized name
            {
                "name": "unique_entity_normalized",
                "query": """
                    CREATE CONSTRAINT unique_entity_normalized IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.normalized IS UNIQUE
                """
            }
        ]

        results = {"created": 0, "errors": []}

        async with self.client.driver.session(database=self.client.database) as session:
            for constraint in constraints:
                try:
                    await session.run(constraint["query"])
                    results["created"] += 1
                    logger.info(f"✓ Created constraint: {constraint['name']}")
                except Exception as e:
                    error_msg = f"Failed to create constraint {constraint['name']}: {e}"
                    logger.warning(error_msg)
                    results["errors"].append(error_msg)

        return results

    async def _create_indexes(self) -> Dict[str, Any]:
        """Create performance indexes"""

        indexes = [
            # Document indexes
            {
                "name": "index_document_url",
                "query": """
                    CREATE INDEX index_document_url IF NOT EXISTS
                    FOR (d:Document) ON (d.url)
                """
            },
            # Entity indexes
            {
                "name": "index_entity_type_primary",
                "query": """
                    CREATE INDEX index_entity_type_primary IF NOT EXISTS
                    FOR (e:Entity) ON (e.type_primary)
                """
            },
            {
                "name": "index_entity_type_full",
                "query": """
                    CREATE INDEX index_entity_type_full IF NOT EXISTS
                    FOR (e:Entity) ON (e.type_full)
                """
            },
            {
                "name": "index_entity_text",
                "query": """
                    CREATE INDEX index_entity_text IF NOT EXISTS
                    FOR (e:Entity) ON (e.text)
                """
            },
            # Chunk indexes
            {
                "name": "index_chunk_index",
                "query": """
                    CREATE INDEX index_chunk_index IF NOT EXISTS
                    FOR (c:Chunk) ON (c.chunk_index)
                """
            }
        ]

        results = {"created": 0, "errors": []}

        async with self.client.driver.session(database=self.client.database) as session:
            for index in indexes:
                try:
                    await session.run(index["query"])
                    results["created"] += 1
                    logger.info(f"✓ Created index: {index['name']}")
                except Exception as e:
                    error_msg = f"Failed to create index {index['name']}: {e}"
                    logger.warning(error_msg)
                    results["errors"].append(error_msg)

        return results

    async def get_schema_info(self) -> Dict[str, Any]:
        """
        Get current schema information

        Returns:
            Dictionary with schema metadata
        """
        info = {
            "constraints": [],
            "indexes": [],
            "node_counts": {},
            "relationship_counts": {}
        }

        async with self.client.driver.session(database=self.client.database) as session:
            # Get constraints
            result = await session.run("SHOW CONSTRAINTS")
            async for record in result:
                info["constraints"].append({
                    "name": record.get("name"),
                    "type": record.get("type"),
                    "entityType": record.get("entityType")
                })

            # Get indexes
            result = await session.run("SHOW INDEXES")
            async for record in result:
                info["indexes"].append({
                    "name": record.get("name"),
                    "type": record.get("type"),
                    "entityType": record.get("entityType"),
                    "properties": record.get("properties")
                })

            # Get node counts
            for label in [self.NODE_DOCUMENT, self.NODE_CHUNK, self.NODE_ENTITY]:
                result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
                record = await result.single()
                info["node_counts"][label] = record["count"]

            # Get relationship counts
            result = await session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(r) AS count
                ORDER BY count DESC
            """)
            async for record in result:
                info["relationship_counts"][record["rel_type"]] = record["count"]

        return info

    async def validate_schema(self) -> Dict[str, Any]:
        """
        Validate schema integrity

        Returns:
            Dictionary with validation results
        """
        validation = {
            "valid": True,
            "issues": []
        }

        async with self.client.driver.session(database=self.client.database) as session:
            # Check for orphaned chunks (no parent document)
            result = await session.run("""
                MATCH (c:Chunk)
                WHERE NOT EXISTS((c)<-[:HAS_CHUNK]-(:Document))
                RETURN count(c) AS orphaned_chunks
            """)
            record = await result.single()
            orphaned_chunks = record["orphaned_chunks"]

            if orphaned_chunks > 0:
                validation["valid"] = False
                validation["issues"].append(
                    f"Found {orphaned_chunks} orphaned chunks (no parent document)"
                )

            # Check for entities with no mentions
            result = await session.run("""
                MATCH (e:Entity)
                WHERE NOT EXISTS((e)-[:MENTIONED_IN]->(:Chunk))
                RETURN count(e) AS entities_without_mentions
            """)
            record = await result.single()
            entities_no_mentions = record["entities_without_mentions"]

            if entities_no_mentions > 0:
                validation["issues"].append(
                    f"Found {entities_no_mentions} entities with no chunk mentions "
                    f"(may be expected for entity normalization)"
                )

        return validation

    async def clear_all_data(self) -> int:
        """
        Clear all data from graph (DANGEROUS!)

        Returns:
            Number of nodes deleted
        """
        logger.warning("Clearing all data from Neo4j graph...")

        async with self.client.driver.session(database=self.client.database) as session:
            result = await session.run("""
                MATCH (n)
                DETACH DELETE n
                RETURN count(n) AS deleted
            """)
            record = await result.single()
            deleted = record["deleted"]

            logger.info(f"Deleted {deleted} nodes and all relationships")
            return deleted


async def initialize_graph_schema(client: Neo4jClient) -> Dict[str, Any]:
    """
    Convenience function to initialize schema

    Args:
        client: Connected Neo4j client

    Returns:
        Initialization results
    """
    schema = GraphSchema(client)
    return await schema.initialize_schema()
