#!/usr/bin/env python3
"""
Test GLiNER entity extraction + vLLM relationship extraction
Extracts entities, then uses LLM to find relationships between them
"""

import sys
import json
import asyncio
import aiohttp

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


async def get_vllm_model_name():
    """Get the model name from vLLM API"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8078/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    if models:
                        return models[0]["id"]
    except:
        pass
    return "Qwen3-30B"  # fallback


async def extract_relationships_with_llm(text, entities, model_name):
    """
    Use vLLM to extract relationships between entities
    Returns: (relationships, llm_output)
    """
    if len(entities) < 2:
        return [], ""

    # Build entity list for prompt
    entity_list = []
    for i, ent in enumerate(entities):
        entity_list.append(f"{i+1}. {ent['text']} ({ent['type_full']})")

    # Build prompt for LLM
    prompt = f"""Extract relationships between entities in the following text.

Text:
{text}

Entities:
{chr(10).join(entity_list)}

For each relationship, output in this JSON format:
{{"subject": "entity1", "predicate": "relationship_type", "object": "entity2", "confidence": 0.9}}

Common relationship types: uses, implements, extends, depends_on, provides, supports, integrates_with, part_of, similar_to, competes_with, processes, manages, configured_with.

Output only valid JSON array of relationships:"""

    # Call vLLM API directly
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8078/v1/completions",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "stop": ["\n\n\n"]
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    llm_output = result.get("choices", [{}])[0].get("text", "").strip()

                    # Store original output before processing
                    original_output = llm_output

                    # Remove markdown code fences if present
                    llm_output = llm_output.replace("```json", "").replace("```", "").strip()

                    # Try to parse JSON output
                    try:
                        # Find all JSON arrays in output (there might be an example first)
                        arrays = []
                        start = 0
                        while True:
                            start_pos = llm_output.find("[", start)
                            if start_pos < 0:
                                break
                            end_pos = llm_output.find("]", start_pos) + 1
                            if end_pos > start_pos:
                                try:
                                    json_str = llm_output[start_pos:end_pos]
                                    arr = json.loads(json_str)
                                    arrays.append(arr)
                                except:
                                    pass
                                start = end_pos
                            else:
                                break

                        # Return the last (most likely actual) array, or the first if only one
                        if arrays:
                            # Prefer the longest array (actual data vs example)
                            relationships = max(arrays, key=len)
                            return relationships, original_output
                        return [], original_output
                    except json.JSONDecodeError as e:
                        print(f"Could not parse LLM output as JSON: {e}")
                        print(llm_output[:300])
                        return [], original_output
                else:
                    print(f"vLLM API error: {response.status}")
                    return [], ""
    except Exception as e:
        print(f"Error calling vLLM: {e}")
        return [], ""


async def test_entity_and_relationship_extraction():
    """Test entity extraction with GLiNER and relationship extraction with vLLM"""

    # Import here after adding to path
    from extractors.entity_extractor import get_entity_extractor

    print("=" * 80)
    print("GLiNER + vLLM Relationship Extraction Test")
    print("=" * 80)
    print()

    # Get the entity extractor
    entity_extractor = get_entity_extractor()

    # Get vLLM model name dynamically
    vllm_model_name = await get_vllm_model_name()

    print(f"✓ GLiNER model loaded: {entity_extractor.model_name}")
    print(f"✓ Entity types available: {len(entity_extractor.entity_types)}")
    print(f"✓ Confidence threshold: {entity_extractor.threshold}")
    print(f"✓ vLLM endpoint: http://localhost:8078")
    print(f"✓ vLLM model: {vllm_model_name}")
    print()

    # Test each text sample
    for test_name, text in TEST_TEXTS.items():
        print("=" * 80)
        print(f"Test: {test_name}")
        print("=" * 80)
        print(f"Text: {text.strip()[:200]}...")
        print()

        try:
            # STEP 1: Extract entities with GLiNER
            print("STEP 1: Extracting entities with GLiNER...")
            entities = entity_extractor.extract(text, threshold=0.4)
            print(f"✓ Extracted {len(entities)} entities")
            print()

            # Display entities
            for i, entity in enumerate(entities, 1):
                print(f"{i}. {entity['text']}")
                print(f"   Type: {entity['type_full']}")
                print(f"   Confidence: {entity['confidence']:.3f}")
            print()

            if len(entities) < 2:
                print("⚠ Need at least 2 entities for relationship extraction")
                print()
                continue

            # STEP 2: Extract relationships with vLLM
            print("STEP 2: Extracting relationships with vLLM...")
            print(f"Analyzing relationships between {len(entities)} entities...")
            print()

            relationships, llm_response = await extract_relationships_with_llm(text, entities, vllm_model_name)

            if relationships:
                print(f"✓ Extracted {len(relationships)} relationships:")
                print()

                for i, rel in enumerate(relationships, 1):
                    subj = rel.get('subject', 'unknown')
                    pred = rel.get('predicate', 'unknown')
                    obj = rel.get('object', 'unknown')
                    conf = rel.get('confidence', 0.0)

                    print(f"{i}. {subj} --[{pred}]--> {obj}")
                    print(f"   Confidence: {conf:.3f}")
                    print()
            else:
                print("⚠ No relationships extracted")
                print()

            # STEP 3: Summary statistics
            print("-" * 80)
            print(f"Summary:")
            print(f"  Entities: {len(entities)}")
            print(f"  Relationships: {len(relationships) if relationships else 0}")

            # Count relationship types
            if relationships:
                rel_types = {}
                for rel in relationships:
                    pred = rel.get('predicate', 'unknown')
                    rel_types[pred] = rel_types.get(pred, 0) + 1
                print(f"  Relationship types: {dict(rel_types)}")

            # Show LLM response
            if llm_response:
                print()
                print("-" * 80)
                print("LLM Response:")
                print(llm_response)

            print()

        except Exception as e:
            print(f"✗ Error during extraction: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run async test
    asyncio.run(test_entity_and_relationship_extraction())
