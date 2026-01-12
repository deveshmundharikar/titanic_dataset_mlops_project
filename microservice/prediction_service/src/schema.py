

from typing import Literal
from pydantic import BaseModel, Field
class TitanicRequest(BaseModel):
    pclass: Literal[1, 2, 3]
    sex: Literal["male", "female"]
    age: float = Field(..., ge=0, le=120)
    sibsp: int = Field(..., ge=0)
    parch: int = Field(..., ge=0)
    fare: float = Field(..., ge=0)
    embarked: Literal["C", "Q", "S"]
    deck: str

