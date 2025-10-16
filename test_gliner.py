#!/usr/bin/env python3
"""
Test GLiNER entity extraction without writing to database
"""

import sys
import json

# Sample test texts
TEST_TEXTS = {
    "python_framework": """
FastAPI is a modern Python web framework. It uses Pydantic for data validation
and automatic API documentation. FastAPI is built on top of Starlette for the
web parts and uses type hints extensively.
""",

    "ai_tech": """
OpenAI developed GPT-4, a large language model trained on massive datasets.
Google also released Gemini, competing in the AI space. Meta's LLaMA models
are open source alternatives that run locally.
""",

    "cloud_services": """
Amazon Web Services (AWS) provides cloud computing with services like EC2 for
virtual machines and S3 for object storage. Microsoft Azure and Google Cloud
Platform are major competitors in the cloud infrastructure market.
""",

    "databases": """
PostgreSQL is a powerful relational database that supports JSON and full-text
search. MongoDB is a NoSQL document database. Neo4j is a graph database ideal
for connected data and relationships.
"""
}


def test_entity_extraction():
    """Test entity extraction with sample texts"""

    # Import here after adding to path
    from extractors.entity_extractor import get_entity_extractor

    print("=" * 80)
    print("GLiNER Entity Extraction Test")
    print("=" * 80)
    print()

    # Get the entity extractor instance
    extractor = get_entity_extractor()

    print(f"✓ GLiNER model loaded: {extractor.model_name}")
    print(f"✓ Entity types available: {len(extractor.entity_types)}")
    print(f"✓ Confidence threshold: {extractor.threshold}")
    print()

    # Test each text sample
    for test_name, text in TEST_TEXTS.items():
        print("=" * 80)
        print(f"Test: {test_name}")
        print("=" * 80)
        print(f"Text: {text.strip()[:200]}...")
        print()

        try:
            # Extract entities
            entities = extractor.extract_entities(text)

            print(f"✓ Extracted {len(entities)} entities:")
            print()

            # Display entities
            for i, entity in enumerate(entities, 1):
                print(f"{i}. {entity['text']}")
                print(f"   Type: {entity['type_full']}")
                print(f"   Primary: {entity['type_primary']}")
                if entity.get('type_sub1'):
                    print(f"   Sub1: {entity['type_sub1']}")
                if entity.get('type_sub2'):
                    print(f"   Sub2: {entity['type_sub2']}")
                print(f"   Confidence: {entity['confidence']:.3f}")
                print(f"   Position: {entity['span'][0]}-{entity['span'][1]}")
                print()

            # Show some statistics
            types_found = {}
            for entity in entities:
                primary = entity['type_primary']
                types_found[primary] = types_found.get(primary, 0) + 1

            print(f"Entity types found: {dict(types_found)}")
            print()

        except Exception as e:
            print(f"✗ Error extracting entities: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_entity_extraction()
