"""Pydantic schemas for test data validation.

This module defines the data models for all evaluation tasks using Pydantic
for automatic validation and type safety.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TestMetadata(BaseModel):
    """Metadata for test suite.

    Attributes
    ----------
    version : str
        Test data version (e.g., "1.0")
    description : str
        Brief description of the test suite
    language : str, optional
        Primary language of test data (default: "tr")
    domain : str, optional
        Domain of test data (e.g., "finance", "general", "medical") - optional
    created_date : str, optional
        ISO format date when test data was created
    total_tasks : int, optional
        Total number of tasks in the suite (default: 4)
    """

    version: str
    description: str
    language: str = "tr"
    domain: Optional[str] = None
    created_date: Optional[str] = None
    total_tasks: int = 4


class IRConfig(BaseModel):
    """Configuration for Information Retrieval task.

    Attributes
    ----------
    ndcg_at_k : list of int
        K values for NDCG calculation (default: [1, 3, 5, 10])
    map_at_k : list of int
        K values for MAP calculation (default: [1, 3, 5, 10])
    mrr_at_k : list of int
        K values for MRR calculation (default: [1, 3, 5, 10])
    precision_at_k : list of int, optional
        K values for Precision calculation
    recall_at_k : list of int, optional
        K values for Recall calculation
    """

    ndcg_at_k: List[int] = Field(default=[1, 3, 5, 10])
    map_at_k: List[int] = Field(default=[1, 3, 5, 10])
    mrr_at_k: List[int] = Field(default=[1, 3, 5, 10])
    precision_at_k: Optional[List[int]] = None
    recall_at_k: Optional[List[int]] = None


class InformationRetrievalData(BaseModel):
    """Data schema for Information Retrieval task.

    Attributes
    ----------
    name : str
        Task name
    enabled : bool, optional
        Whether this task is enabled (default: True)
    corpus : dict of str to str
        Document corpus mapping doc_id to document text
    queries : dict of str to str
        Query set mapping query_id to query text
    relevant_docs : dict of str to dict
        Relevance judgments mapping query_id to {doc_id: score}
        Scores: 0=not relevant, 1=somewhat relevant, 2=highly relevant
    config : IRConfig, optional
        Task configuration

    Examples
    --------
    >>> data = InformationRetrievalData(
    ...     name="IR Task",
    ...     corpus={"c0": "Document text..."},
    ...     queries={"q0": "Query text..."},
    ...     relevant_docs={"q0": {"c0": 2}}
    ... )
    """

    name: str
    enabled: bool = True
    corpus: Dict[str, str] = Field(..., description="Document corpus")
    queries: Dict[str, str] = Field(..., description="Query set")
    relevant_docs: Dict[str, Dict[str, int]] = Field(
        ...,
        description="Relevance judgments (query_id -> {doc_id: relevance_score})"
    )
    config: IRConfig = Field(default_factory=IRConfig)

    @field_validator('relevant_docs')
    @classmethod
    def validate_relevance_scores(cls, v):
        """Validate that relevance scores are 0, 1, or 2.

        Parameters
        ----------
        v : dict
            Relevance judgments to validate

        Returns
        -------
        dict
            Validated relevance judgments

        Raises
        ------
        ValueError
            If any relevance score is not 0, 1, or 2
        """
        for query_id, docs in v.items():
            for doc_id, score in docs.items():
                if score not in [0, 1, 2]:
                    raise ValueError(
                        f"Relevance score must be 0, 1, or 2. "
                        f"Got {score} for query={query_id}, doc={doc_id}"
                    )
        return v

    @field_validator('corpus', 'queries')
    @classmethod
    def validate_not_empty(cls, v, info):
        """Validate that corpus and queries are not empty."""
        if not v:
            raise ValueError(f"{info.field_name} cannot be empty")
        return v


class TripletData(BaseModel):
    """Single triplet for semantic similarity evaluation.

    Attributes
    ----------
    id : str
        Unique identifier for this triplet
    anchor : str
        Anchor text
    positive : str
        Text semantically similar to anchor
    negative : str
        Text semantically dissimilar to anchor
    difficulty : {'kolay', 'orta', 'zor'}, optional
        Difficulty level of this triplet (optional metadata)
    category : str, optional
        Category/topic of this triplet (e.g., "enflasyon", "borsa") - optional metadata
    subcategory : str, optional
        High-level category (e.g., "finansal", "genel_turkce") - optional metadata

    Examples
    --------
    >>> # Minimal triplet (only required fields)
    >>> triplet = TripletData(
    ...     id="t0",
    ...     anchor="Enflasyon yükseliyor",
    ...     positive="Fiyatlar artıyor",
    ...     negative="Hisse senetleri düştü"
    ... )
    >>>
    >>> # Full triplet with metadata
    >>> triplet = TripletData(
    ...     id="t0",
    ...     anchor="Enflasyon yükseliyor",
    ...     positive="Fiyatlar artıyor",
    ...     negative="Hisse senetleri düştü",
    ...     difficulty="kolay",
    ...     category="enflasyon",
    ...     subcategory="finansal"
    ... )
    """

    id: str
    anchor: str
    positive: str
    negative: str
    difficulty: Optional[Literal["kolay", "orta", "zor"]] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None

    @field_validator('anchor', 'positive', 'negative')
    @classmethod
    def validate_text_not_empty(cls, v, info):
        """Validate that text fields are not empty."""
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v.strip()


class SimilarityConfig(BaseModel):
    """Configuration for Semantic Similarity task.

    Attributes
    ----------
    success_threshold : float
        Minimum triplet accuracy for success (default: 0.85)
    margin_threshold : float
        Minimum average margin between positive and negative scores (default: 0.20)
    """

    success_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    margin_threshold: float = Field(default=0.20, ge=0.0)


class SemanticSimilarityData(BaseModel):
    """Data schema for Semantic Similarity task.

    Attributes
    ----------
    name : str
        Task name
    enabled : bool, optional
        Whether this task is enabled (default: True)
    triplets : list of TripletData
        List of triplet test cases
    config : SimilarityConfig, optional
        Task configuration
    """

    name: str
    enabled: bool = True
    triplets: List[TripletData] = Field(..., min_items=1)
    config: SimilarityConfig = Field(default_factory=SimilarityConfig)


class MorphologyPair(BaseModel):
    """Morphology test pair.

    Attributes
    ----------
    id : str
        Unique identifier
    original : str
        Original word
    modified : str
        Word with morphological suffix
    category : str
        Type of morphology (e.g., "iyelik", "çoğul", "hal_eki")
    expected_similarity : {'high', 'low'}
        Expected similarity level (default: "high")
    """

    id: str
    original: str
    modified: str
    category: str
    expected_similarity: Literal["high", "low"] = "high"


class TypoPair(BaseModel):
    """Typo test pair.

    Attributes
    ----------
    id : str
        Unique identifier
    correct : str
        Correctly spelled word
    typo : str
        Word with typo
    type : str
        Type of typo (e.g., "türkçe_ses", "klavye", "eksik_harf")
    expected_similarity : {'high', 'low'}
        Expected similarity level (default: "high")
    """

    id: str
    correct: str
    typo: str
    type: str
    expected_similarity: Literal["high", "low"] = "high"


class NegationPair(BaseModel):
    """Negation test pair.

    Attributes
    ----------
    id : str
        Unique identifier
    sentence1 : str
        First sentence
    sentence2 : str
        Second sentence (negated or opposite meaning)
    type : str
        Type of negation (e.g., "olumsuzluk_eki", "zıt_kelime", "bağlaç")
    expected_similarity : {'high', 'low'}
        Expected similarity level (default: "low")
    """

    id: str
    sentence1: str
    sentence2: str
    type: str
    expected_similarity: Literal["high", "low"] = "low"


class RobustnessConfig(BaseModel):
    """Configuration for Linguistic Robustness task.

    Attributes
    ----------
    morphology_threshold : float
        Threshold for morphology pairs (default: 0.85)
    typo_threshold : float
        Threshold for typo pairs (default: 0.75)
    negation_threshold : float
        Threshold for negation pairs (default: 0.50)
    """

    morphology_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    typo_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    negation_threshold: float = Field(default=0.50, ge=0.0, le=1.0)


class LinguisticRobustnessData(BaseModel):
    """Data schema for Linguistic Robustness task.

    Attributes
    ----------
    name : str
        Task name
    enabled : bool, optional
        Whether this task is enabled (default: True)
    morphology : list of MorphologyPair
        Morphology test pairs
    typo : list of TypoPair
        Typo test pairs
    negation : list of NegationPair
        Negation test pairs
    config : RobustnessConfig, optional
        Task configuration
    """

    name: str
    enabled: bool = True
    morphology: List[MorphologyPair] = Field(default_factory=list)
    typo: List[TypoPair] = Field(default_factory=list)
    negation: List[NegationPair] = Field(default_factory=list)
    config: RobustnessConfig = Field(default_factory=RobustnessConfig)


class AnalogyData(BaseModel):
    """Single analogy for vector arithmetic evaluation.

    Format: a - b + c ≈ d

    Attributes
    ----------
    id : str
        Unique identifier
    a : str
        First term
    b : str
        Second term
    c : str
        Third term
    d : str
        Expected result (fourth term)
    category : str
        Category of analogy (e.g., "coğrafya", "cinsiyet", "para_ülke")
    subcategory : {'finansal', 'genel_turkce'}
        High-level category
    formula : str, optional
        Human-readable formula description (default: "a - b + c ≈ d")

    Examples
    --------
    >>> analogy = AnalogyData(
    ...     id="a0",
    ...     a="Ankara",
    ...     b="Türkiye",
    ...     c="Almanya",
    ...     d="Berlin",
    ...     category="coğrafya",
    ...     subcategory="genel_turkce"
    ... )
    """

    id: str
    a: str
    b: str
    c: str
    d: str
    category: str
    subcategory: Literal["finansal", "genel_turkce"]
    formula: str = "a - b + c ≈ d"


class ArithmeticConfig(BaseModel):
    """Configuration for Vector Arithmetic task.

    Attributes
    ----------
    top_k : list of int
        K values for top-k accuracy calculation (default: [1, 5, 10])
    vocabulary_expansion : bool
        Whether to expand vocabulary beyond analogy terms (default: True)
    vocabulary_sources : list of str
        Sources for vocabulary expansion (default: ["analogies", "corpus"])
    """

    top_k: List[int] = Field(default=[1, 5, 10])
    vocabulary_expansion: bool = True
    vocabulary_sources: List[str] = Field(default=["analogies", "corpus"])


class VectorArithmeticData(BaseModel):
    """Data schema for Vector Arithmetic task.

    Attributes
    ----------
    name : str
        Task name
    enabled : bool, optional
        Whether this task is enabled (default: True)
    analogies : list of AnalogyData
        List of analogy test cases
    config : ArithmeticConfig, optional
        Task configuration
    """

    name: str
    enabled: bool = True
    analogies: List[AnalogyData] = Field(..., min_items=1)
    config: ArithmeticConfig = Field(default_factory=ArithmeticConfig)


class TasksModel(BaseModel):
    """Container for all evaluation tasks.

    Attributes
    ----------
    information_retrieval : InformationRetrievalData, optional
        Information Retrieval task data
    semantic_similarity : SemanticSimilarityData, optional
        Semantic Similarity task data
    linguistic_robustness : LinguisticRobustnessData, optional
        Linguistic Robustness task data
    vector_arithmetic : VectorArithmeticData, optional
        Vector Arithmetic task data
    """

    information_retrieval: Optional[InformationRetrievalData] = None
    semantic_similarity: Optional[SemanticSimilarityData] = None
    linguistic_robustness: Optional[LinguisticRobustnessData] = None
    vector_arithmetic: Optional[VectorArithmeticData] = None

    @model_validator(mode='after')
    def validate_at_least_one_task(self):
        """Validate that at least one task is provided."""
        tasks = [
            self.information_retrieval,
            self.semantic_similarity,
            self.linguistic_robustness,
            self.vector_arithmetic
        ]

        if not any(tasks):
            raise ValueError("At least one task must be provided")

        return self


class TestDataModel(BaseModel):
    """Complete test data schema.

    This is the root model that encompasses all evaluation tasks.

    Attributes
    ----------
    metadata : TestMetadata
        Test suite metadata
    tasks : TasksModel
        Container for all evaluation tasks

    Examples
    --------
    >>> from pathlib import Path
    >>> import json
    >>>
    >>> # Load from JSON
    >>> data = json.loads(Path("test_data.json").read_text())
    >>> test_data = TestDataModel(**data)
    >>>
    >>> # Access specific task
    >>> ir_data = test_data.tasks.information_retrieval
    >>> print(f"Corpus size: {len(ir_data.corpus)}")
    """

    metadata: TestMetadata
    tasks: TasksModel

    def get_enabled_tasks(self) -> List[str]:
        """Get list of enabled task names.

        Returns
        -------
        list of str
            Names of enabled tasks

        Examples
        --------
        >>> test_data = TestDataModel(...)
        >>> test_data.get_enabled_tasks()
        ['information_retrieval', 'semantic_similarity']
        """
        enabled = []

        if self.tasks.information_retrieval and self.tasks.information_retrieval.enabled:
            enabled.append('information_retrieval')
        if self.tasks.semantic_similarity and self.tasks.semantic_similarity.enabled:
            enabled.append('semantic_similarity')
        if self.tasks.linguistic_robustness and self.tasks.linguistic_robustness.enabled:
            enabled.append('linguistic_robustness')
        if self.tasks.vector_arithmetic and self.tasks.vector_arithmetic.enabled:
            enabled.append('vector_arithmetic')

        return enabled
