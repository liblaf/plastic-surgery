from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import pydicom
import pydicom.valuerep
import pyvista as pv

if TYPE_CHECKING:
    from _typeshed import StrPath


class DicomReader:
    folder: Path

    def __init__(self, path: StrPath) -> None:
        path = Path(path)
        if path.is_file():
            self.folder = path.parent
        else:
            self.folder = path

    @functools.cached_property
    def dirfile(self) -> pydicom.FileDataset:
        return pydicom.dcmread(self.folder / "DIRFILE")

    @functools.cached_property
    def first_record(self) -> pydicom.FileDataset:
        return pydicom.dcmread(
            self.folder
            / self.dirfile["DirectoryRecordSequence"][0]["ReferencedFileID"][-1]
        )

    @functools.cached_property
    def image_data(self) -> pv.ImageData:
        return pv.read(self.folder, force_ext=".dcm")  # pyright: ignore[reportReturnType]

    # ------------------------------- Metadata ------------------------------- #

    @functools.cached_property
    def acquisition_datetime(self) -> pydicom.valuerep.DT:
        return pydicom.valuerep.DT(self.first_record["AcquisitionDateTime"].value)  # pyright: ignore[reportReturnType]

    @functools.cached_property
    def patient_age(self) -> str:
        return self.first_record["PatientAge"].value

    @functools.cached_property
    def patient_birth_date(self) -> pydicom.valuerep.DA:
        return pydicom.valuerep.DA(self.first_record["PatientBirthDate"].value)  # pyright: ignore[reportReturnType]

    @functools.cached_property
    def patient_id(self) -> str:
        return self.first_record["PatientID"].value

    @functools.cached_property
    def patient_name(self) -> str:
        return str(self.first_record["PatientName"].value)

    @functools.cached_property
    def patient_sex(self) -> str:
        return self.first_record["PatientSex"].value
