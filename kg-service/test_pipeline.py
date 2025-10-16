#!/usr/bin/env python3
"""
Test the complete KG pipeline: Entity extraction + Relationship extraction + Neo4j storage
"""

import asyncio
from pipeline.processor import KGProcessor

async def test_pipeline():
    """Test full pipeline"""

    # Initialize processor
    print("Initializing KG processor...")
    processor = KGProcessor()

    if not await processor.initialize():
        print("Failed to initialize processor")
        return

    print("✓ Processor initialized")
    print()

    # Test document with proper chunks
    markdown_content = """# FastAPI Framework

FastAPI is a modern Python web framework. It uses Pydantic for data validation
and automatic API documentation. FastAPI is built on top of Starlette for the
web parts and uses type hints extensively.

## Cloud Services

Amazon Web Services (AWS) provides cloud computing with services like EC2 for
virtual machines and S3 for object storage. Microsoft Azure and Google Cloud
Platform are major competitors in the cloud infrastructure market.
"""

    test_doc = {
        "content_id": 999,
        "url": "https://example.com/test",
        "title": "FastAPI Test Document",
        "markdown": markdown_content,
        "chunks": [
            {
                "vector_rowid": 99991,
                "chunk_index": 0,
                "char_start": 0,
                "char_end": 200,
                "text": "FastAPI is a modern Python web framework. It uses Pydantic for data validation and automatic API documentation. FastAPI is built on top of Starlette for the web parts.",
                "heading": "FastAPI Framework",
                "token_count": 45
            },
            {
                "vector_rowid": 99992,
                "chunk_index": 1,
                "char_start": 200,
                "char_end": 400,
                "text": "Amazon Web Services (AWS) provides cloud computing with services like EC2 for virtual machines and S3 for object storage. Microsoft Azure and Google Cloud Platform are major competitors.",
                "heading": "Cloud Services",
                "token_count": 38
            }
        ]
    }

    print("=" * 80)
    print("Processing test document:")
    print(f"Title: {test_doc['title']}")
    print(f"URL: {test_doc['url']}")
    print("=" * 80)
    print()

    try:
        # Process the document
        result = await processor.process_document(
            content_id=test_doc['content_id'],
            url=test_doc['url'],
            title=test_doc['title'],
            markdown=test_doc['markdown'],
            chunks=test_doc['chunks']
        )

        print("✓ Document processed successfully!")
        print()
        print("-" * 80)
        print("Results:")
        print(f"  Entities extracted: {result.get('entities_extracted', 0)}")
        print(f"  Relationships extracted: {result.get('relationships_extracted', 0)}")
        print(f"  Processing time: {result.get('processing_time_seconds', 0):.2f}s")
        print("-" * 80)
        print()

        # Show some sample entities
        if result.get('sample_entities'):
            print("Sample Entities:")
            for i, ent in enumerate(result['sample_entities'][:5], 1):
                print(f"  {i}. {ent['text']} ({ent['type']})")
            print()

        # Show some sample relationships
        if result.get('sample_relationships'):
            print("Sample Relationships:")
            for i, rel in enumerate(result['sample_relationships'][:5], 1):
                print(f"  {i}. {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")
            print()

        print("=" * 80)
        print("Pipeline test complete!")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Error processing document: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        pass  # No cleanup method needed

if __name__ == "__main__":
    asyncio.run(test_pipeline())
