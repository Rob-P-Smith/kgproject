"""
GLiNER-based Entity Extractor with Hierarchical Type Support

Features:
- 300+ entity types with 3-level hierarchical classification
- Batch processing for efficiency
- Confidence threshold filtering
- Entity deduplication
- Context extraction for mentions
"""

import os
import logging
from typing import List, Dict, Any, Set
import yaml
from gliner import GLiNER
from config import settings

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities using GLiNER with hierarchical type classification"""

    def __init__(
        self,
        model_name: str = None,
        taxonomy_path: str = None,
        threshold: float = None
    ):
        self.model_name = model_name or settings.GLINER_MODEL
        self.threshold = threshold or settings.GLINER_THRESHOLD
        self.taxonomy_path = taxonomy_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            settings.ENTITY_TAXONOMY_PATH
        )

        # Load model
        logger.info(f"Loading GLiNER model: {self.model_name}")
        try:
            self.model = GLiNER.from_pretrained(self.model_name)
            logger.info("GLiNER model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GLiNER model: {e}")
            raise

        # Load entity taxonomy
        self.entity_types = self._load_taxonomy()
        logger.info(f"Loaded {len(self.entity_types)} entity types from taxonomy")

        # Cache for type parsing
        self._type_cache: Dict[str, Dict[str, str]] = {}

    def _load_taxonomy(self) -> List[str]:
        """
        Load entity types from YAML taxonomy file

        Returns:
            List of entity type strings
        """
        try:
            with open(self.taxonomy_path, 'r') as f:
                taxonomy_data = yaml.safe_load(f)

            entity_types = []
            for category, types in taxonomy_data.get('entity_categories', {}).items():
                entity_types.extend(types)

            logger.debug(f"Loaded {len(entity_types)} entity types")
            return entity_types

        except Exception as e:
            logger.error(f"Failed to load taxonomy from {self.taxonomy_path}: {e}")
            raise

    def _parse_type_hierarchy(self, type_string: str) -> Dict[str, str]:
        """
        Parse hierarchical type string into components

        Args:
            type_string: e.g., "Framework::Backend::Python"

        Returns:
            dict with keys: type_primary, type_sub1, type_sub2, type_sub3
        """
        if type_string in self._type_cache:
            return self._type_cache[type_string]

        parts = type_string.split("::")
        parsed = {
            "type_primary": parts[0] if len(parts) > 0 else None,
            "type_sub1": parts[1] if len(parts) > 1 else None,
            "type_sub2": parts[2] if len(parts) > 2 else None,
            "type_sub3": parts[3] if len(parts) > 3 else None,
        }

        self._type_cache[type_string] = parsed
        return parsed

    def _get_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 50
    ) -> Dict[str, str]:
        """
        Extract context around an entity mention

        Args:
            text: Full text
            start: Entity start position
            end: Entity end position
            window: Characters to include before/after

        Returns:
            dict with context_before, context_after, sentence
        """
        # Get surrounding context
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        context_before = text[context_start:start].strip()
        context_after = text[end:context_end].strip()

        # Try to extract full sentence
        sentence_start = text.rfind('.', 0, start) + 1
        sentence_end = text.find('.', end)
        if sentence_end == -1:
            sentence_end = len(text)

        sentence = text[sentence_start:sentence_end].strip()

        return {
            "context_before": context_before,
            "context_after": context_after,
            "sentence": sentence
        }

    def extract(
        self,
        text: str,
        custom_types: List[str] = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text with hierarchical typing

        Args:
            text: Input text to process
            custom_types: Optional list of entity types (defaults to taxonomy)
            threshold: Confidence threshold (defaults to settings)

        Returns:
            List of entity dictionaries with hierarchical type info
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for extraction")
            return []

        entity_types = custom_types or self.entity_types
        threshold = threshold or self.threshold

        try:
            logger.debug(f"Extracting entities from text ({len(text)} chars)")

            # Run GLiNER prediction
            predictions = self.model.predict_entities(
                text,
                entity_types,
                threshold=threshold
            )

            logger.debug(f"GLiNER returned {len(predictions)} predictions")

            # Process predictions
            entities = []
            seen_mentions: Set[str] = set()

            for pred in predictions:
                # Parse hierarchical type
                type_hierarchy = self._parse_type_hierarchy(pred["label"])

                # Get context
                context = self._get_context(
                    text,
                    pred["start"],
                    pred["end"],
                    window=settings.RELATION_CONTEXT_WINDOW
                )

                # Create entity mention key for deduplication
                mention_key = f"{pred['text'].lower()}:{pred['start']}:{pred['end']}"

                # Skip if already seen (deduplication)
                if settings.ENTITY_DEDUPLICATION and mention_key in seen_mentions:
                    continue
                seen_mentions.add(mention_key)

                # Build entity dict
                entity = {
                    # Identity
                    "text": pred["text"],
                    "normalized": pred["text"].lower().strip(),
                    "start": pred["start"],
                    "end": pred["end"],

                    # Type hierarchy
                    "type_full": pred["label"],
                    **type_hierarchy,

                    # Confidence
                    "confidence": float(pred["score"]),

                    # Context
                    **context,

                    # Metadata
                    "extraction_method": "gliner"
                }

                entities.append(entity)

            logger.info(f"Extracted {len(entities)} entities")
            return entities

        except Exception as e:
            logger.error(f"Error during entity extraction: {e}")
            raise

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Extract entities from multiple texts in batches

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of entity lists (one per input text)
        """
        batch_size = batch_size or settings.GLINER_BATCH_SIZE
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}")

            batch_results = [self.extract(text) for text in batch]
            results.extend(batch_results)

        return results

    def get_entity_types(self, category: str = None) -> List[str]:
        """
        Get available entity types, optionally filtered by category

        Args:
            category: Optional category filter (e.g., "Framework")

        Returns:
            List of entity type strings
        """
        if category:
            return [
                t for t in self.entity_types
                if t.startswith(category + "::")
            ]
        return self.entity_types

    def get_type_hierarchy_tree(self) -> Dict[str, Any]:
        """
        Build hierarchical tree structure of entity types

        Returns:
            Nested dictionary representing type hierarchy
        """
        tree: Dict[str, Any] = {}

        for entity_type in self.entity_types:
            parts = entity_type.split("::")
            current = tree

            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {} if i < len(parts) - 1 else {"_full_type": entity_type}
                current = current[part]

        return tree


# Global extractor instance
_entity_extractor: EntityExtractor = None


def get_entity_extractor() -> EntityExtractor:
    """Get or create global entity extractor instance"""
    global _entity_extractor

    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
        logger.info("Created global EntityExtractor instance")

    return _entity_extractor


if __name__ == "__main__":
    # Test the entity extractor
    print("Testing Entity Extractor...")

    # Test text
    test_text = """
    FastAPI is a modern Python web framework that uses Pydantic for data validation.
    It's built on top of Starlette and Uvicorn, and is designed for building APIs quickly.
    Many developers prefer it over Flask and Django for microservices.
    The framework supports async/await and integrates well with Neo4j and PostgreSQL databases.
    OpenAI and Anthropic provide LLM APIs that can be integrated with FastAPI backends.
    """

    extractor = get_entity_extractor()

    print(f"\nLoaded {len(extractor.entity_types)} entity types")
    print(f"Threshold: {extractor.threshold}")

    print("\nExtracting entities...")
    entities = extractor.extract(test_text)

    print(f"\nFound {len(entities)} entities:\n")
    for entity in entities:
        print(f"  {entity['text']}")
        print(f"    Type: {entity['type_primary']} > {entity['type_sub1']} > {entity['type_sub2']}")
        print(f"    Confidence: {entity['confidence']:.2f}")
        print(f"    Context: ...{entity['context_before'][-30:]} [{entity['text']}] {entity['context_after'][:30]}...")
        print()
