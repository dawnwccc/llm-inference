from typing import List
from serve.entity.protocol import CompletionChoiceResponse
from pydantic import BaseModel


class TempCompletionResponse(BaseModel):
    choices: List[CompletionChoiceResponse]
    interrupted: bool = False
