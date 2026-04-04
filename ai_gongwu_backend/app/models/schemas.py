"""Pydantic data models for questions, media extraction, and API responses."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class QuestionDimension(BaseModel):
    """Single scoring dimension declared by the question bank."""

    name: str
    score: float = Field(..., ge=0)


class QuestionDefinition(BaseModel):
    """Structured question definition loaded from the question bank."""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: str = ""
    province: str = ""
    fullScore: float = Field(..., ge=0)
    question: str
    dimensions: List[QuestionDimension]
    coreKeywords: List[str] = Field(default_factory=list)
    strongKeywords: List[str] = Field(default_factory=list)
    weakKeywords: List[str] = Field(default_factory=list)
    bonusKeywords: List[str] = Field(default_factory=list)
    penaltyKeywords: List[str] = Field(default_factory=list)
    scoringCriteria: List[str] = Field(default_factory=list)
    deductionRules: List[str] = Field(default_factory=list)


class MediaExtractionResult(BaseModel):
    """Normalized output of text/audio/video extraction."""

    transcript: str
    source: Literal["text", "audio", "video"]
    visual_observation: Optional[str] = None


class LLMEvaluationPayload(BaseModel):
    """Raw structured payload expected from the LLM."""

    model_config = ConfigDict(extra="ignore")

    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    deduction_details: List[str] = Field(default_factory=list)
    bonus_details: List[str] = Field(default_factory=list)
    evidence_quotes: List[str] = Field(default_factory=list)
    rationale: str = ""
    total_score: float = 0.0


class EvaluationResult(LLMEvaluationPayload):
    """Final response after deterministic normalization and validation."""

    question_id: str
    question_type: str = ""
    transcript: str
    visual_observation: Optional[str] = None
    matched_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    validation_notes: List[str] = Field(default_factory=list)


class EvaluationAPIResponse(BaseModel):
    """Common API envelope for evaluation responses."""

    code: int = 200
    message: str = "success"
    data: EvaluationResult
