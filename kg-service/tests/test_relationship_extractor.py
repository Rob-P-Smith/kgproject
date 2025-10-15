"""
Test relationship extractor with sample data
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractors.entity_extractor import get_entity_extractor
from extractors.relation_extractor import get_relation_extractor
from pipeline.chunk_mapper import get_chunk_mapper


async def test_relationship_extraction():
    """Test the full entity + relationship extraction pipeline"""

    # Sample technical document
    sample_text = """
# FastAPI: Modern Web Framework

FastAPI is a modern, high-performance web framework for building APIs with Python 3.7+
based on standard Python type hints. It was created by Sebastián Ramírez and is built
on top of Starlette for the web parts and Pydantic for the data validation.

## Key Features

FastAPI uses Pydantic models for request and response validation. This integration
provides automatic data validation, serialization, and documentation. The framework
also leverages Starlette's async capabilities for high-performance request handling.

## Performance

FastAPI competes with Node.js and Go in terms of performance. It is one of the fastest
Python frameworks available, on par with frameworks like Starlette and Uvicorn.

## Ecosystem

The framework integrates with SQLAlchemy for database operations and supports
OpenAPI for automatic API documentation. FastAPI also works seamlessly with
popular Python testing frameworks like pytest.

Developers often compare FastAPI to Flask and Django, but FastAPI differs from
these frameworks in its approach to type hints and async support.
    """

    # Sample chunks (simulating chunking)
    chunks = [
        {
            "vector_rowid": 1001,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 400,
            "text": sample_text[0:400]
        },
        {
            "vector_rowid": 1002,
            "chunk_index": 1,
            "char_start": 350,
            "char_end": 750,
            "text": sample_text[350:750]
        },
        {
            "vector_rowid": 1003,
            "chunk_index": 2,
            "char_start": 700,
            "char_end": len(sample_text),
            "text": sample_text[700:]
        }
    ]

    print("=" * 80)
    print("TESTING RELATIONSHIP EXTRACTION PIPELINE")
    print("=" * 80)
    print()

    # Step 1: Extract entities
    print("Step 1: Extracting entities with GLiNER...")
    entity_extractor = get_entity_extractor()
    entities = entity_extractor.extract(sample_text)

    print(f"✓ Extracted {len(entities)} entities")
    print()
    print("Sample entities:")
    for entity in entities[:5]:
        print(f"  - {entity['text']} ({entity['type_full']}) [conf: {entity['confidence']:.2f}]")
    print()

    # Step 2: Extract relationships
    print("Step 2: Extracting relationships with vLLM...")
    relation_extractor = get_relation_extractor()

    try:
        relationships = await relation_extractor.extract_relationships(
            sample_text,
            entities
        )

        print(f"✓ Extracted {len(relationships)} relationships")
        print()

        if relationships:
            print("Sample relationships:")
            for rel in relationships[:10]:
                print(
                    f"  - {rel['subject_text']} --[{rel['predicate']}]--> "
                    f"{rel['object_text']} [conf: {rel['confidence']:.2f}]"
                )
                print(f"    Context: {rel['context'][:100]}...")
        else:
            print("⚠ No relationships extracted (vLLM may not be available)")
        print()

    except Exception as e:
        print(f"⚠ Relationship extraction failed: {e}")
        print("  (This is expected if vLLM is not running)")
        relationships = []
        print()

    # Step 3: Map to chunks
    print("Step 3: Mapping entities and relationships to chunks...")
    chunk_mapper = get_chunk_mapper()

    # Map entities
    mapped_entities = chunk_mapper.map_entities_to_chunks(entities, chunks)
    print(f"✓ Mapped entities to chunks")

    # Show sample mapping
    print()
    print("Sample entity chunk mappings:")
    for entity in mapped_entities[:3]:
        appearances = entity.get("chunk_appearances", [])
        print(f"  - {entity['text']}:")
        print(f"    Appears in {len(appearances)} chunk(s)")
        for app in appearances[:2]:
            print(f"      Chunk {app['chunk_index']} (rowid {app['vector_rowid']})")
    print()

    # Map relationships
    if relationships:
        mapped_relationships = chunk_mapper.map_relationships_to_chunks(
            relationships,
            mapped_entities,
            chunks
        )
        print(f"✓ Mapped relationships to chunks")

        print()
        print("Sample relationship chunk mappings:")
        for rel in mapped_relationships[:3]:
            print(
                f"  - {rel['subject_text']} --[{rel['predicate']}]--> "
                f"{rel['object_text']}"
            )
            print(f"    Chunks: {rel['chunk_rowids']}")
            print(f"    Spans chunks: {rel['spans_chunks']}")
        print()

    # Step 4: Generate summary
    print("Step 4: Generating mapping summary...")
    summary = chunk_mapper.generate_mapping_summary(
        mapped_entities,
        mapped_relationships if relationships else [],
        chunks
    )

    print()
    print("MAPPING SUMMARY:")
    print("-" * 80)
    print(f"Total chunks: {summary['total_chunks']}")
    print(f"Chunks with entities: {summary['chunks_with_entities']}")
    print(f"Unique entities: {summary['unique_entities']}")
    print(f"Total entity appearances: {summary['total_entity_appearances']}")
    print(f"Multi-chunk entities: {summary['multi_chunk_entities']}")
    print(f"Avg entities per chunk: {summary['avg_entities_per_chunk']}")
    print(f"Total relationships: {summary['total_relationships']}")
    print(f"Cross-chunk relationships: {summary['cross_chunk_relationships']}")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_relationship_extraction())
