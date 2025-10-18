"""
Relationship Extractor using vLLM for LLM-based extraction

This module extracts semantic relationships between entities identified by GLiNER.
Uses vLLM to understand context and identify meaningful relationships.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import re

from pydantic import BaseModel, Field
from clients.vllm_client import get_vllm_client
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Pydantic models for vLLM guided JSON schema
class RelationshipItem(BaseModel):
    """Single relationship for vLLM guided JSON output"""
    subject: str = Field(description="Entity name from the entity list (exact match)")
    predicate: str = Field(description="Relationship type in snake_case (e.g., uses, implements, part_of)")
    object: str = Field(description="Target entity name from the entity list (exact match)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    context: str = Field(description="Brief supporting text from the document")


class RelationshipResponse(BaseModel):
    """vLLM guided JSON response format for relationship extraction"""
    relationships: List[RelationshipItem] = Field(description="Array of extracted relationships")


@dataclass
class EntityMention:
    """Entity mention with position and type information"""
    text: str
    normalized: str
    type_full: str
    type_primary: str
    start_pos: int
    end_pos: int
    confidence: float
    sentence: str
    context_before: str
    context_after: str


@dataclass
class ExtractedRelationship:
    """Relationship between two entities"""
    subject_text: str
    subject_normalized: str
    subject_type: str
    predicate: str
    object_text: str
    object_normalized: str
    object_type: str
    confidence: float
    context: str
    sentence: str
    subject_start: int
    subject_end: int
    object_start: int
    object_end: int


class RelationshipExtractor:
    """Extract relationships between entities using vLLM"""

    # Relationship types organized by category
    RELATIONSHIP_TYPES = {
        "technical": [
            "uses", "implements", "extends", "depends_on", "requires",
            "provides", "supports", "integrates_with", "based_on",
            "built_with", "powered_by", "runs_on", "compatible_with"
        ],
        "comparison": [
            "similar_to", "alternative_to", "competes_with", "differs_from",
            "replaces", "supersedes", "evolved_from"
        ],
        "hierarchical": [
            "part_of", "contains", "includes", "composed_of",
            "category_of", "type_of", "instance_of", "subclass_of"
        ],
        "functional": [
            "processes", "generates", "transforms", "analyzes",
            "validates", "handles", "manages", "controls"
        ],
        "development": [
            "developed_by", "maintained_by", "created_by", "designed_by",
            "contributed_to", "sponsored_by"
        ],
        "documentation": [
            "documented_in", "described_in", "defined_in",
            "referenced_in", "mentioned_in"
        ],
        "configuration": [
            "configured_with", "settings_for", "parameter_of",
            "option_for", "enabled_by"
        ],
        "performance": [
            "optimizes", "improves", "accelerates", "scales_with",
            "benchmarked_against"
        ]
    }

    def __init__(self):
        """Initialize relationship extractor"""
        self.vllm_client = None  # Will be initialized in async methods
        self.max_entity_distance = settings.RELATION_MAX_DISTANCE  # sentences
        self.min_confidence = settings.RELATION_MIN_CONFIDENCE
        self.context_window = settings.RELATION_CONTEXT_WINDOW  # characters

        # Flatten relationship types for prompt
        self.all_relationship_types = []
        for category_types in self.RELATIONSHIP_TYPES.values():
            self.all_relationship_types.extend(category_types)

        logger.info(f"Relationship extractor initialized with {len(self.all_relationship_types)} relationship types")

    async def _ensure_vllm_client(self):
        """Ensure vLLM client is initialized (lazy async initialization)"""
        if self.vllm_client is None:
            self.vllm_client = await get_vllm_client()
            # Ensure model is discovered and available
            await self.vllm_client.ensure_model()
            logger.info(f"vLLM client ready with model: {self.vllm_client.model_name}")
        return self.vllm_client

    def _build_extraction_prompt(
        self,
        text: str,
        entities: List[EntityMention]
    ) -> str:
        """Build prompt for LLM relationship extraction"""

        # Format entities for prompt
        entity_list = []
        for i, ent in enumerate(entities, 1):
            entity_list.append(
                f"{i}. **{ent.text}** ({ent.type_primary})"
            )

        entities_formatted = "\n".join(entity_list)

        # Format relationship types by category (show all types)
        rel_types_formatted = []
        for category, types in self.RELATIONSHIP_TYPES.items():
            rel_types_formatted.append(f"- **{category.title()}**: {', '.join(types)}")
        rel_types_formatted = "\n".join(rel_types_formatted)

        prompt = f"""You are an expert at extracting semantic relationships between entities in technical documentation.

        
DO NOT RETURN A SUMMARY.
DO NOT EXPLAIN YOUR CHOICES.
DO NOT ADD ANYTHING OUTSIDE OF THE JSON 

**Text:**
{text}

**Entities:**
{entities_formatted}

**Task:**
Identify meaningful relationships between the entities above. Focus on explicit relationships mentioned in the text.

**Relationship Types (organized by category):**
{rel_types_formatted}

Use the most appropriate relationship type from the categories above, or create similar snake_case predicates if needed.

**Output Format:**
Return a JSON array of relationships. Each relationship should have:
- subject: The entity name (must match one from the list above)
- predicate: The relationship type (use snake_case, e.g., "uses", "implements")
- object: The target entity name (must match one from the list above)
- confidence: Float between 0 and 1
- context: Brief supporting text from the document

**Example:**
```json
[
  {{
    "subject": "FastAPI",
    "predicate": "uses",
    "object": "Pydantic",
    "confidence": 0.95,
    "context": "FastAPI uses Pydantic for data validation"
  }},
  {{
    "subject": "FastAPI",
    "predicate": "based_on",
    "object": "Starlette",
    "confidence": 0.88,
    "context": "FastAPI is based on Starlette for async support"
  }}
]
```

**Important Rules:**
1. Only extract relationships explicitly stated in the text
2. Subject and object MUST be entity names from the list above (exact match)
3. Use lowercase snake_case for predicates (do not deviate from this)
4. Confidence should reflect how clearly the relationship is stated (the lower the number the lower the relationshipo strength)
5. Context should be a brief quote or paraphrase from the text (no more than 100 words no less than 50 words)
6. Return empty array [] if no clear relationships exist
7. Focus on meaningful relationships, not trivial mentions, something that matches the relationships selections earlier 

Return ONLY the JSON array, no additional text.
DO NOT RETURN A SUMMARY.
DO NOT EXPLAIN YOUR CHOICES.
DO NOT ADD ANYTHING OUTSIDE OF THE JSON """

        return prompt

    def _parse_llm_response(
        self,
        response: str,
        entities: List[EntityMention]
    ) -> List[Dict[str, Any]]:
        """Parse LLM response and validate relationships"""

        # Remove markdown code fences and extra backticks
        response = response.replace("```json", "").replace("```", "").strip()

        relationships = []

        # Extract JSON arrays from response - handle multiple arrays (example + actual)
        try:
            arrays = []
            start = 0
            while True:
                start_pos = response.find("[", start)
                if start_pos < 0:
                    break

                # Find the matching closing bracket
                bracket_count = 0
                end_pos = start_pos
                for i in range(start_pos, len(response)):
                    if response[i] == '[':
                        bracket_count += 1
                    elif response[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_pos = i + 1
                            break

                if end_pos > start_pos:
                    try:
                        json_str = response[start_pos:end_pos]
                        arr = json.loads(json_str)
                        if isinstance(arr, list):
                            arrays.append(arr)
                            logger.debug(f"Found valid JSON array with {len(arr)} items")
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse JSON array at position {start_pos}: {e}")
                    start = end_pos
                else:
                    break

            # Prefer the longest array (actual data vs example)
            if arrays:
                relationships = max(arrays, key=len)
                logger.debug(f"Extracted {len(relationships)} relationships from array parsing")
            else:
                logger.warning("No JSON array found in LLM response")
                logger.debug(f"Response preview: {response[:500]}")
                return []

        except Exception as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response: {response[:500]}")
            return []

        if not isinstance(relationships, list):
            logger.warning("LLM response is not a list")
            return []

        # Build entity lookup
        entity_lookup = {}
        for ent in entities:
            entity_lookup[ent.text.lower()] = ent
            entity_lookup[ent.normalized.lower()] = ent

        # Validate and enrich relationships
        validated = []
        for rel in relationships:
            try:
                # Required fields
                if not all(k in rel for k in ["subject", "predicate", "object", "confidence"]):
                    continue

                subject_text = rel["subject"].strip()
                object_text = rel["object"].strip()

                # Find matching entities
                subject_entity = entity_lookup.get(subject_text.lower())
                object_entity = entity_lookup.get(object_text.lower())

                if not subject_entity or not object_entity:
                    logger.debug(f"Entity not found: {subject_text} -> {object_text}")
                    continue

                # Skip self-relationships
                if subject_entity.normalized == object_entity.normalized:
                    continue

                # Validate confidence
                confidence = float(rel["confidence"])
                if confidence < self.min_confidence:
                    continue

                # Build validated relationship
                validated.append({
                    "subject_text": subject_entity.text,
                    "subject_normalized": subject_entity.normalized,
                    "subject_type": subject_entity.type_full,
                    "predicate": rel["predicate"].lower().replace(" ", "_"),
                    "object_text": object_entity.text,
                    "object_normalized": object_entity.normalized,
                    "object_type": object_entity.type_full,
                    "confidence": confidence,
                    "context": rel.get("context", "")[:500],
                    "subject_start": subject_entity.start_pos,
                    "subject_end": subject_entity.end_pos,
                    "object_start": object_entity.start_pos,
                    "object_end": object_entity.end_pos
                })

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Invalid relationship format: {e}")
                continue

        return validated

    def _deduplicate_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate relationships, keeping highest confidence"""

        # Group by (subject, predicate, object)
        rel_map = {}
        for rel in relationships:
            key = (
                rel["subject_normalized"],
                rel["predicate"],
                rel["object_normalized"]
            )

            if key not in rel_map or rel["confidence"] > rel_map[key]["confidence"]:
                rel_map[key] = rel

        return list(rel_map.values())

    async def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using vLLM

        Args:
            text: Full document text
            entities: List of entity dictionaries from EntityExtractor

        Returns:
            List of relationship dictionaries
        """
        if not entities:
            logger.info("No entities provided, skipping relationship extraction")
            return []

        # Convert entities to EntityMention objects
        entity_mentions = []
        for ent in entities:
            # Find entity position in text (use first occurrence if not provided)
            start_pos = ent.get("start_pos", text.find(ent["text"]))
            end_pos = ent.get("end_pos", start_pos + len(ent["text"]))

            entity_mentions.append(EntityMention(
                text=ent["text"],
                normalized=ent["normalized"],
                type_full=ent["type_full"],
                type_primary=ent["type_primary"],
                start_pos=start_pos,
                end_pos=end_pos,
                confidence=ent["confidence"],
                sentence=ent.get("sentence", ""),
                context_before=ent.get("context_before", ""),
                context_after=ent.get("context_after", "")
            ))

        logger.info(f"Extracting relationships from {len(entity_mentions)} entities")

        # For large documents, process in chunks to avoid context limits
        max_text_length = 30000  # characters
        if len(text) > max_text_length:
            logger.info(f"Document too long ({len(text)} chars), processing in sections")
            # Process sections with entity overlap
            all_relationships = []

            # Split into sections while keeping entities together
            section_size = max_text_length
            for i in range(0, len(text), section_size):
                section_text = text[i:i + section_size + 3000]  # Overlap (10% of section size)

                # Filter entities in this section
                section_entities = [
                    e for e in entity_mentions
                    if e.start_pos >= i and e.start_pos < i + len(section_text)
                ]

                if len(section_entities) < 2:
                    continue

                # Extract relationships for this section
                section_rels = await self._extract_from_section(
                    section_text,
                    section_entities
                )
                all_relationships.extend(section_rels)

            # Deduplicate across sections
            return self._deduplicate_relationships(all_relationships)

        else:
            # Process entire document at once
            return await self._extract_from_section(text, entity_mentions)

    async def _extract_from_section(
        self,
        text: str,
        entities: List[EntityMention]
    ) -> List[Dict[str, Any]]:
        """Extract relationships from a text section"""

        if len(entities) < 2:
            return []

        # Ensure vLLM client is initialized
        await self._ensure_vllm_client()

        logger.info(f"vLLM client initialized, model_name: {self.vllm_client.model_name}")

        # Build prompt
        prompt = self._build_extraction_prompt(text, entities)

        logger.debug(f"Built prompt with {len(entities)} entities, prompt length: {len(prompt)}")

        # Call vLLM for relationship extraction
        try:
            logger.info("Calling vLLM for relationship extraction...")
            response = await self.vllm_client.complete(
                prompt=prompt,
                max_tokens=settings.VLLM_MAX_TOKENS,
                temperature=settings.VLLM_TEMPERATURE
            )
            logger.info(f"vLLM response received, length: {len(response)}")

            # Parse response
            relationships = self._parse_llm_response(response, entities)

            logger.info(f"Extracted {len(relationships)} relationships from section")
            return relationships

        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

    def _find_entity_sentence(
        self,
        text: str,
        entity_start: int,
        entity_end: int
    ) -> str:
        """Find the sentence containing an entity"""

        # Find sentence boundaries
        sentence_start = max(0, entity_start - 500)
        sentence_end = min(len(text), entity_end + 500)

        # Look for sentence boundaries
        for i in range(entity_start, sentence_start, -1):
            if i < len(text) and text[i] in '.!?\n':
                sentence_start = i + 1
                break

        for i in range(entity_end, sentence_end):
            if i < len(text) and text[i] in '.!?':
                sentence_end = i + 1
                break

        return text[sentence_start:sentence_end].strip()


# Global instance
_relation_extractor: Optional[RelationshipExtractor] = None


def get_relation_extractor() -> RelationshipExtractor:
    """Get global relationship extractor instance"""
    global _relation_extractor
    if _relation_extractor is None:
        _relation_extractor = RelationshipExtractor()
    return _relation_extractor
