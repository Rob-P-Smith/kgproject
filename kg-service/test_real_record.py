#!/usr/bin/env python3
"""
Test KG pipeline with real record from RAG database
"""

import asyncio
from pipeline.processor import KGProcessor


async def main():
    # Read the real test record (from Docker volume mount)
    with open('/app/test_record.md', 'r') as f:
        markdown = f.read()

    # Create simulated chunks (similar to how the RAG system would chunk this)
    # For simplicity, we'll split by major sections
    lines = markdown.split('\n')

    # Create 3 chunks from the document
    chunk_size = len(markdown) // 3
    chunks = [
        {
            "vector_rowid": 10001,
            "chunk_index": 0,
            "char_start": 0,
            "char_end": chunk_size,
            "text": markdown[0:chunk_size],
            "heading": "GitHub Spark - Introduction",
            "token_count": len(markdown[0:chunk_size].split())
        },
        {
            "vector_rowid": 10002,
            "chunk_index": 1,
            "char_start": chunk_size,
            "char_end": chunk_size * 2,
            "text": markdown[chunk_size:chunk_size * 2],
            "heading": "GitHub Spark - Features",
            "token_count": len(markdown[chunk_size:chunk_size * 2].split())
        },
        {
            "vector_rowid": 10003,
            "chunk_index": 2,
            "char_start": chunk_size * 2,
            "char_end": len(markdown),
            "text": markdown[chunk_size * 2:],
            "heading": "GitHub Spark - FAQ & Pricing",
            "token_count": len(markdown[chunk_size * 2:].split())
        }
    ]

    # Test document metadata
    test_doc = {
        "content_id": 9678,
        "url": "http://github.com/features/spark",
        "title": "GitHub Spark - Dream it. See it. Ship it.",
        "markdown": markdown,
        "chunks": chunks
    }

    print("=" * 80)
    print("Initializing KG processor...")
    print("=" * 80)

    processor = KGProcessor()
    await processor.initialize()

    print("✓ Processor initialized")
    print()
    print("=" * 80)
    print("Processing real document:")
    print(f"Title: {test_doc['title']}")
    print(f"URL: {test_doc['url']}")
    print(f"Content ID: {test_doc['content_id']}")
    print(f"Markdown length: {len(markdown)} chars")
    print(f"Number of chunks: {len(chunks)}")
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

        print("=" * 80)
        print("Real record pipeline test complete!")
        print("=" * 80)

    except Exception as e:
        print(f"✗ Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
