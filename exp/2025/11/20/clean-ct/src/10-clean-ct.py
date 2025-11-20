import collections
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from liblaf import cherries, grapes
from liblaf.plastic_surgery import DicomReader

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    data_dir: Path = Path("~/datasets/CT资料").expanduser()
    output_dir: Path = Path("~/datasets/CT").expanduser()


def process_acquisition(reader: DicomReader, patient_id: str, output_dir: Path) -> None:
    logger.info(
        "%s (%s): %s",
        patient_id,
        reader.patient_name,
        reader.acquisition_datetime.isoformat(),
    )
    target_dir: Path = (
        output_dir / patient_id / reader.acquisition_datetime.strftime("%Y-%m-%d")
    )
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(reader.folder, target_dir, dirs_exist_ok=True)


def main(cfg: Config) -> None:
    datasets: collections.defaultdict[str, list[DicomReader]] = collections.defaultdict(
        list
    )
    for dirfile in cfg.data_dir.rglob("DIRFILE"):
        reader = DicomReader(dirfile)
        # Patient ID may change over time for the same patient, so we group by name.
        datasets[reader.patient_name].append(reader)

    task_inputs: list[tuple[DicomReader, str, Path]] = []
    for readers in datasets.values():
        readers.sort(key=lambda r: r.acquisition_datetime)
        patient_id: str = readers[-1].patient_id
        task_inputs.extend([(reader, patient_id, cfg.output_dir) for reader in readers])
    with ThreadPoolExecutor() as executor:
        for _ in grapes.track(
            executor.map(process_acquisition, *zip(*task_inputs, strict=True)),
            total=len(task_inputs),
        ):
            pass


if __name__ == "__main__":
    cherries.main(main)
