import concurrent
import concurrent.futures
import logging
from concurrent.futures import Future, ProcessPoolExecutor
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
    skin: pv.PolyData = image_data.contour([-200.0])  # pyright: ignore[reportAssignmentType]
    skin.extract_largest(inplace=True)
    melon.save(output_dir / "skin.ply", skin)
    skull: pv.PolyData = image_data.contour([200.0])  # pyright: ignore[reportAssignmentType]
    skull.extract_largest(inplace=True)
    melon.save(output_dir / "skull.ply", skull)


def main(cfg: Config) -> None:
    meta: MetaDataset = grapes.load(cfg.data_dir / "dataset.json", type=MetaDataset)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    grapes.save(cfg.output_dir / "dataset.json", meta, order="sorted")
    with ProcessPoolExecutor() as executor:
        futures: list[Future[None]] = []
        for patient_id, meta_patient in meta.patients.items():
            for meta_acq in meta_patient.acquisitions:
                input_dir: Path = (
                    cfg.data_dir / patient_id / meta_acq.datetime.strftime("%Y-%m-%d")
                )
                output_dir: Path = (
                    cfg.output_dir / patient_id / meta_acq.datetime.strftime("%Y-%m-%d")
                )
                futures.append(
                    executor.submit(process_acquisition, input_dir, output_dir)
                )
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    cherries.main(main)
