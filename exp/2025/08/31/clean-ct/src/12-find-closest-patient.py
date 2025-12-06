import logging
from pathlib import Path

import numpy as np
import trimesh as tm
from jaxtyping import Float

from liblaf import cherries, grapes, melon
from liblaf.plastic_surgery import MetaDataset

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    surface_dir: Path = cherries.input("11-surface")


def main(cfg: Config) -> None:
    meta: MetaDataset = grapes.load(cfg.surface_dir / "dataset.json", type=MetaDataset)
    distances: list[tuple[str, float]] = []
    for patient_id, meta_patient in meta.patients.items():
        patient_dir: Path = cfg.surface_dir / patient_id
        pre_skin: tm.Trimesh = melon.load_trimesh(
            patient_dir
            / meta_patient.acquisitions[0].datetime.strftime("%Y-%m-%d")
            / "skin.ply"
        )
        post_skin: tm.Trimesh = melon.load_trimesh(
            patient_dir
            / meta_patient.acquisitions[-1].datetime.strftime("%Y-%m-%d")
            / "skin.ply"
        )
        transformed: Float[np.ndarray, "n 3"]
        cost: float
        _, transformed, cost = tm.registration.icp(
            post_skin.sample(10000),
            pre_skin.sample(10000),
            max_iterations=100,
            reflection=False,
            translation=True,
            scale=False,
        )
        ic(patient_id, cost)
        distance: Float[np.ndarray, " n"]
        distance, _ = pre_skin.nearest.vertex(transformed)
        distance_max: float = float(np.max(distance))
        ic(patient_id, distance_max)
        distances.append((patient_id, distance_max))
    distances = sorted(distances, key=lambda x: x[1])
    for patient_id, distance_max in distances:
        logger.info("%s: max skin distance: %g mm", patient_id, distance_max)


if __name__ == "__main__":
    cherries.main(main)
