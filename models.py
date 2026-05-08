from pydantic import BaseModel, Field, field_validator
from typing import List, Any
import json

class SepsisEvidenceRow(BaseModel):
    study: str = Field(default="Not Reported", description="Lead Author and Year")
    population: str = Field(default="Not Reported", description="Country, setting, and clinical condition")
    sample_size: str = Field(default="Not Reported", description="Total N; Died N; Survived N")
    predictor: str = Field(default="Not Reported", description="The clinical variable studied")
    outcome: str = Field(default="Not Reported", description="Specific mortality definition")
    measurement_timing: str = Field(default="Not Reported", description="When the predictor was measured")
    method: str = Field(default="Not Reported", description="Statistical approach")
    effect_size: str = Field(default="Not Reported", description="OR, HR, or Cutoff value")
    performance: str = Field(default="Not Reported", description="AUC with 95% CI and p-value")
    notes: str = Field(default="Not Reported", description="Adjustment variables or unique findings")

    @field_validator('*', mode='before')
    @classmethod
    def force_string(cls, value: Any) -> str:
        if isinstance(value, dict) or isinstance(value, list):
            return json.dumps(value)
        if value is None:
            return "Not Reported"
        return str(value)
    
class SepsisAtlas(BaseModel):
    evidence_base: List[SepsisEvidenceRow]