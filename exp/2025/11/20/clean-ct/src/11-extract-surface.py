import datetime
import logging
from pathlib import Path

import pyvista as pv

from liblaf import cherries, grapes, melon
from liblaf.plastic_surgery import DicomReader, MetaDataset

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    data_dir: Path = Path("~/datasets/CT").expanduser()

    output_dir: Path = cherries.output("11-surface")


def process_acquisition(acquisition_dir: Path, output_dir: Path) -> None:
    reader = DicomReader(acquisition_dir)
    logger.info(
        "%s (%s): %s",
        reader.patient_id,
        reader.patient_name,
        reader.acquisition_datetime.isoformat(),
    )
    image_data: pv.ImageData = reader.image_data
    image_data = image_data.gaussian_smooth()  # pyright: ignore[reportAssignmentType]
    skull: pv.PolyData = image_data.contour([-200.0])  # pyright: ignore[reportAssignmentType]
    skull.extract_largest(inplace=True)
    melon.save(output_dir / "skull.ply", skull)
    skin: pv.PolyData = image_data.contour([200.0])  # pyright: ignore[reportAssignmentType]
    skin.extract_largest(inplace=True)
    melon.save(output_dir / "skin.ply", skin)


def main(cfg: Config) -> None:
    meta: MetaDataset = grapes.load(cfg.data_dir / "dataset.json", type=MetaDataset)
    task_inputs: list[tuple[Path, Path]] = []
    patient_ids_to_del: list[str] = []
    for patient_id, meta_patient in meta.patients.items():
        datetimes: list[datetime.datetime] = [
            acquisition.datetime for acquisition in meta_patient.acquisitions
        ]
        interval: datetime.timedelta = max(datetimes) - min(datetimes)
        if interval < datetime.timedelta(days=30):
            logger.warning(
                "%s (%s): skipping due to short interval: %s",
                patient_id,
                meta_patient.name,
                interval,
            )
            patient_ids_to_del.append(patient_id)
            continue
        task_inputs.extend(
            (
                cfg.data_dir / patient_id / acquisition_datetime.strftime("%Y-%m-%d"),
                cfg.output_dir / patient_id / acquisition_datetime.strftime("%Y-%m-%d"),
            )
            for acquisition_datetime in datetimes
        )
    for patient_id in patient_ids_to_del:
        del meta.patients[patient_id]
    grapes.save(cfg.output_dir / "dataset.json", meta)
    for inputs in grapes.track(task_inputs):
        process_acquisition(*inputs)


if __name__ == "__main__":
    cherries.main(main)
