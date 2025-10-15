"""
vLLM Client with automatic model discovery and retry logic

Features:
- Auto-discovery of model name from /v1/models endpoint
- Automatic retry with exponential backoff on failure
- Model name reset to None on connection failure
- 30-second retry interval when unavailable
"""

import asyncio
import time
import logging
from typing import Optional, List, Dict, Any
import httpx
import backoff
from config import settings

logger = logging.getLogger(__name__)


class ModelUnavailableError(Exception):
    """Raised when vLLM model is not available"""
    pass


class VLLMClient:
    """Client for vLLM inference server with auto-recovery"""

    def __init__(
        self,
        base_url: str = None,
        timeout: int = None,
        retry_interval: int = None
    ):
        self.base_url = base_url or settings.VLLM_BASE_URL
        self.timeout = timeout or settings.VLLM_TIMEOUT
        self.retry_interval = retry_interval or settings.VLLM_RETRY_INTERVAL

        # Model state
        self.model_name: Optional[str] = None
        self.last_check: Optional[float] = None
        self.is_available: bool = False

        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10)
        )

        logger.info(f"VLLMClient initialized with base_url: {self.base_url}")

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def get_model_name(self) -> Optional[str]:
        """
        Query /v1/models endpoint to get the active model name

        Returns:
            str: Model name if available, None otherwise
        """
        try:
            logger.debug(f"Querying models endpoint: {self.base_url}/v1/models")
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()

            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                model_name = data["data"][0]["id"]
                logger.info(f"Discovered model: {model_name}")
                return model_name
            else:
                logger.warning("No models found in /v1/models response")
                return None

        except httpx.HTTPError as e:
            logger.error(f"Failed to get model name: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting model name: {e}")
            return None

    async def ensure_model(self) -> bool:
        """
        Ensure we have a valid model name

        Returns:
            bool: True if model is available, False otherwise
        """
        current_time = time.time()

        # Check if we need to query for model
        should_check = (
            self.model_name is None or
            self.last_check is None or
            (current_time - self.last_check) > self.retry_interval
        )

        if should_check:
            logger.debug("Checking for available model...")
            self.model_name = await self.get_model_name()
            self.last_check = current_time

            if self.model_name:
                self.is_available = True
                logger.info(f"Model available: {self.model_name}")
                return True
            else:
                self.is_available = False
                logger.warning("Model not available, will retry")
                return False

        return self.is_available

    async def wait_for_model(self, max_wait_time: int = 300) -> bool:
        """
        Wait for model to become available with periodic retries

        Args:
            max_wait_time: Maximum time to wait in seconds

        Returns:
            bool: True if model became available, False if timeout
        """
        start_time = time.time()
        attempt = 1

        while (time.time() - start_time) < max_wait_time:
            logger.info(f"Attempt {attempt}: Waiting for vLLM model...")

            if await self.ensure_model():
                return True

            logger.info(f"Model not available, retrying in {self.retry_interval}s...")
            await asyncio.sleep(self.retry_interval)
            attempt += 1

        logger.error(f"Model did not become available after {max_wait_time}s")
        return False

    def reset_model_state(self):
        """Reset model state after connection failure"""
        logger.warning("Resetting model state due to failure")
        self.model_name = None
        self.last_check = None
        self.is_available = False

    async def complete(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate completion using vLLM

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            str: Generated text

        Raises:
            ModelUnavailableError: If model is not available
        """
        # Ensure model is available
        if not await self.ensure_model():
            raise ModelUnavailableError("vLLM model is not available")

        try:
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens or settings.VLLM_MAX_TOKENS,
                "temperature": temperature or settings.VLLM_TEMPERATURE,
                **kwargs
            }

            if stop:
                request_data["stop"] = stop

            logger.debug(f"Sending completion request to vLLM")
            response = await self.client.post(
                f"{self.base_url}/v1/completions",
                json=request_data
            )
            response.raise_for_status()

            data = response.json()
            completion_text = data["choices"][0]["text"]

            logger.debug(f"Received completion: {len(completion_text)} chars")
            return completion_text

        except httpx.HTTPError as e:
            logger.error(f"HTTP error during completion: {e}")
            self.reset_model_state()
            raise ModelUnavailableError(f"vLLM request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during completion: {e}")
            self.reset_model_state()
            raise

    async def extract_json(
        self,
        prompt: str,
        max_tokens: int = None,
        temperature: float = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON completion using vLLM

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            dict: Parsed JSON response

        Raises:
            ModelUnavailableError: If model is not available
            ValueError: If response is not valid JSON
        """
        try:
            # Some vLLM models support response_format
            response_text = await self.complete(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["```", "\n\n\n"]
            )

            # Try to extract JSON from response
            import json

            # Try direct parsing
            try:
                return json.loads(response_text.strip())
            except json.JSONDecodeError:
                # Try to find JSON in markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))

                # Try to find any JSON object
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))

                raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")

        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if vLLM server is healthy

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed model information

        Returns:
            dict: Model information or None if unavailable
        """
        if not await self.ensure_model():
            return None

        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            data = response.json()

            if data.get("data"):
                return data["data"][0]
            return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None


# Global client instance
_vllm_client: Optional[VLLMClient] = None


async def get_vllm_client() -> VLLMClient:
    """Get or create global vLLM client instance"""
    global _vllm_client

    if _vllm_client is None:
        _vllm_client = VLLMClient()
        logger.info("Created global VLLMClient instance")

    return _vllm_client


async def close_vllm_client():
    """Close global vLLM client"""
    global _vllm_client

    if _vllm_client:
        await _vllm_client.close()
        _vllm_client = None
        logger.info("Closed global VLLMClient instance")


if __name__ == "__main__":
    # Test the vLLM client
    import asyncio

    async def test_client():
        client = VLLMClient()

        print("Testing vLLM client...")
        print(f"Base URL: {client.base_url}")

        # Test model discovery
        print("\n1. Testing model discovery...")
        model_name = await client.get_model_name()
        print(f"   Model: {model_name}")

        if model_name:
            # Test completion
            print("\n2. Testing completion...")
            try:
                response = await client.complete(
                    prompt="The capital of France is",
                    max_tokens=10
                )
                print(f"   Response: {response}")
            except Exception as e:
                print(f"   Error: {e}")

            # Test JSON extraction
            print("\n3. Testing JSON extraction...")
            try:
                json_response = await client.extract_json(
                    prompt='Return a JSON object with a "name" field set to "test": ',
                    max_tokens=50
                )
                print(f"   JSON: {json_response}")
            except Exception as e:
                print(f"   Error: {e}")
        else:
            print("\nModel not available, skipping completion tests")

        await client.close()

    asyncio.run(test_client())
