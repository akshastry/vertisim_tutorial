from pydantic import BaseModel

class DynamicWindState(BaseModel):
    locationname: int
    windspeed: float
    winddir: float

class StaticWindState(BaseModel):
    windspeed: float
    winddir: float