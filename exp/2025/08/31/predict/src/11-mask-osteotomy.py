import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float

from liblaf import cherries, melon

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    post_mandible: Path = cherries.output("00-post-mandible.vtp")
    pre_mandible: Path = cherries.input("00-pre-mandible.vtp")

    osteotomy_distance_threshold: float = 1.5  # millimeters
    output: Path = cherries.output("11-pre-mandible.vtp")


def main(cfg: Config) -> None:
    pre_mandible: pv.PolyData = melon.load_polydata(cfg.pre_mandible)
    post_mandible: pv.PolyData = melon.load_polydata(cfg.post_mandible)

    matrix: Float[np.ndarray, "4 4"]
    cost: float
    matrix, _, cost = tm.registration.icp(
        post_mandible.points,
        pre_mandible.points,
        reflection=False,
        translation=True,
        scale=False,
    )
    logger.info("ICP cost: %g", cost)
    post_mandible.transform(matrix, inplace=True)
    melon.save(cherries.temp("11-post-mandible-aligned.vtp"), post_mandible)

    nearest: melon.NearestPointOnSurfaceResult = melon.nearest_point_on_surface(
        post_mandible,
        pre_mandible.cell_centers(),
        distance_threshold=1.0,
        normal_threshold=None,
    )

    pre_mandible.cell_data["Osteotomy"] = (
        nearest.distance > cfg.osteotomy_distance_threshold
    ) & ~pre_mandible.cell_data["Floating"]
    pre_mandible.cell_data["Distance"] = nearest.distance
    melon.save(cfg.output, pre_mandible)


if __name__ == "__main__":
    cherries.main(main)
