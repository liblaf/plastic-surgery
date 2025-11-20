import datetime

import pydantic


class MetaAcquisition(pydantic.BaseModel):
    datetime: datetime.datetime


class MetaPatient(pydantic.BaseModel):
    id: str
    name: str
    acquisitions: list[MetaAcquisition] = pydantic.Field(default_factory=list)


class MetaDataset(pydantic.BaseModel):
    patients: dict[str, MetaPatient] = pydantic.Field(default_factory=dict)
