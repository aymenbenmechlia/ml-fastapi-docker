from pydantic import BaseModel


class PredictionRequest(BaseModel):
    operational_unit_id: int
    item_id: int