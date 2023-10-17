from pydantic import BaseModel, Field
from enum import Enum


class SexEnum(str, Enum):
    """Enumeration for the value of the 'sex' input of the model."""

    male = "male"
    female = "female"


class RegionEnum(str, Enum):
    """Enumeration for the value of the 'region' input of the model."""

    southwest = "southwest"
    southeast = "southeast"
    northwest = "northwest"
    northeast = "northeast"


class InsuranceChargesModelInput(BaseModel):
    """Schema for input of the model's predict method."""

    age: int = Field(None, title="Age", ge=18, le=65, description="Age of primary beneficiary in years.")
    sex: SexEnum = Field(None, title="Sex", description="Gender of beneficiary.")
    bmi: float = Field(None, title="Body Mass Index", ge=15.0, le=50.0, description="Body mass index of beneficiary.")
    children: int = Field(None, title="Children", ge=0, le=5, description="Number of children covered by health "
                                                                          "insurance.")
    smoker: bool = Field(None, title="Smoker", description="Whether beneficiary is a smoker.")
    region: RegionEnum = Field(None, title="Region", description="Region where beneficiary lives.")


class InsuranceChargesModelOutput(BaseModel):
    """Schema for output of the model's predict method."""

    charges: float = Field(None, title="Charges", description="Individual medical costs billed by health insurance to "
                                                              "customer in US dollars.")
