#!/usr/bin/env python3
"""
Direct test of vLLM with guided JSON to see truncation issue
"""

import asyncio
import httpx
import json
from pydantic import BaseModel, Field
from typing import List


class RelationshipItem(BaseModel):
    """Single relationship"""
    subject: str
    predicate: str
    object: str
    confidence: float
    context: str


class RelationshipResponse(BaseModel):
    """Response format"""
    relationships: List[RelationshipItem]


async def send_single_request(client, prompt, max_tokens, request_num):
    """Send a single request and return results"""
    import time

    request_data = {
        "model": "Qwen3-30B",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }

    start_time = time.time()
    print(f"[Request {request_num}] Sending with max_tokens={max_tokens}...")

    try:
        response = await client.post(
            "http://localhost:8078/v1/completions",
            json=request_data
        )

        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            completion_text = data["choices"][0]["text"]

            # Remove markdown code fences
            cleaned = completion_text.replace("```json", "").replace("```", "").strip()

            # Try to extract JSON arrays using bracket matching
            arrays = []
            start = 0
            while True:
                start_pos = cleaned.find("[", start)
                if start_pos < 0:
                    break

                # Find matching closing bracket
                bracket_count = 0
                end_pos = start_pos
                for i in range(start_pos, len(cleaned)):
                    if cleaned[i] == '[':
                        bracket_count += 1
                    elif cleaned[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i + 1
                            break

                if end_pos > start_pos:
                    try:
                        json_str = cleaned[start_pos:end_pos]
                        arr = json.loads(json_str)
                        if isinstance(arr, list):
                            arrays.append(arr)
                    except json.JSONDecodeError:
                        pass
                    start = end_pos
                else:
                    break

            if arrays:
                rels = max(arrays, key=len)  # Pick longest array
                return {
                    "request_num": request_num,
                    "max_tokens": max_tokens,
                    "success": True,
                    "elapsed": elapsed,
                    "response_length": len(completion_text),
                    "relationships": rels
                }
            else:
                return {
                    "request_num": request_num,
                    "max_tokens": max_tokens,
                    "success": False,
                    "elapsed": elapsed,
                    "error": "No valid JSON arrays found"
                }
        else:
            return {
                "request_num": request_num,
                "max_tokens": max_tokens,
                "success": False,
                "elapsed": elapsed,
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "request_num": request_num,
            "max_tokens": max_tokens,
            "success": False,
            "elapsed": elapsed,
            "error": str(e)
        }


async def test_vllm():
    print("="*80)
    print("CONCURRENT vLLM TEST")
    print("="*80)

    # Create a simple test prompt
    prompt = """Extract relationships from this text:

GitHub Spark is built by GitHub. GitHub Spark uses TypeScript and React.
GitHub Spark provides AI-powered development. GitHub Spark integrates with GitHub Copilot.
Microsoft owns GitHub. Azure hosts GitHub Spark applications.

Return relationships as JSON."""

    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"\nSending 7 concurrent requests...")
    print()

    # Send all requests concurrently
    async with httpx.AsyncClient(timeout=1800.0) as client:
        tasks = [
            send_single_request(client, prompt, 4096, 1),
            send_single_request(client, prompt, 8192, 2),
            send_single_request(client, prompt, 16384, 3),
            send_single_request(client, prompt, 65536, 4),
            send_single_request(client, prompt, 4096, 5),
            send_single_request(client, prompt, 8192, 6),
            send_single_request(client, prompt, 16384, 7),
        ]

        results = await asyncio.gather(*tasks)

    # Print results
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)

    for result in sorted(results, key=lambda x: x["request_num"]):
        print(f"\n[Request {result['request_num']}] max_tokens={result['max_tokens']}")
        print(f"  Time: {result['elapsed']:.2f}s")

        if result["success"]:
            rels = result["relationships"]
            print(f"  ✓ Success! Found {len(rels)} relationships")
            print(f"  Response length: {result['response_length']} chars")
            for i, rel in enumerate(rels[:3]):
                print(f"    {i+1}. {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")
        else:
            print(f"  ✗ Failed: {result['error']}")

    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_vllm())
