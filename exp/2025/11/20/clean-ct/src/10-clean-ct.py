import logging
import shutil
import statistics
from collections import defaultdict
from collections.abc import Generator
from datetime import timedelta
from pathlib import Path

from liblaf import cherries, grapes
from liblaf.plastic_surgery import (
    DicomReader,
    MetaAcquisition,
    MetaDataset,
    MetaPatient,
)

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    data_dir: Path = Path("~/datasets/CT资料").expanduser()
    output_dir: Path = Path("~/datasets/CT").expanduser()


def filter_by_volume(readers: list[DicomReader]) -> Generator[DicomReader]:
    mean_volume: float = statistics.mean(reader.image_data.volume for reader in readers)
    for reader in readers:
        if 0.5 * mean_volume <= reader.image_data.volume:
            yield reader
        else:
            logger.warning(
                "%s (%s) %s: small volume: %g (%g of mean)",
                reader.patient_id,
                reader.patient_name,
                reader.acquisition_datetime.isoformat(),
                reader.image_data.volume,
                reader.image_data.volume / mean_volume,
            )


def main(cfg: Config) -> None:
    readers: list[DicomReader] = [
        DicomReader(dirfile) for dirfile in cfg.data_dir.rglob("DIRFILE")
    ]
    readers = list(filter_by_volume(readers))
    patients: defaultdict[str, list[DicomReader]] = defaultdict(list)
    for r in readers:
        # Patient ID may vary across acquisitions, so group by name
        patients[r.patient_name].append(r)
    meta = MetaDataset()
    for patient_name, readers in patients.items():
        readers = sorted(readers, key=lambda r: r.acquisition_datetime)  # noqa: PLW2901
        patient_id: str = readers[-1].patient_id
        interval: timedelta = (
            readers[-1].acquisition_datetime - readers[0].acquisition_datetime
        )
        if interval < timedelta(days=30):
            logger.warning(
                "%s (%s): skipping due to short interval: %s",
                patient_id,
                patient_name,
                interval,
            )
            continue
        meta_patient: MetaPatient = meta.patients.get(
            patient_id, MetaPatient(id=patient_id, name=patient_name)
        )
        meta_patient.acquisitions.extend(
            [MetaAcquisition(datetime=r.acquisition_datetime) for r in readers]
        )
        meta.patients[patient_id] = meta_patient
        for r in readers:
            logger.info(
                "%s (%s): %s",
                patient_id,
                patient_name,
                r.acquisition_datetime.isoformat(),
            )
            target_dir: Path = (
                cfg.output_dir
                / patient_id
                / r.acquisition_datetime.strftime("%Y-%m-%d")
            )
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(r.folder, target_dir, dirs_exist_ok=True)
    grapes.save(cfg.output_dir / "dataset.json", meta, order="sorted")


if __name__ == "__main__":
    cherries.main(main)
