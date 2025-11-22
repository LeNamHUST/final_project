from pydantic import BaseModel

class RetinalDiseaseClassificationResponse(BaseModel):
    retinal_classification: list