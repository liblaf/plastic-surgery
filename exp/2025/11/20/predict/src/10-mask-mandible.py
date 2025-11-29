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
    output: Path = cherries.output("10-pre-mandible.vtp")


def main(cfg: Config) -> None:
    pre_mandible: pv.PolyData = melon.load_polydata(cfg.pre_mandible)
    post_mandible: pv.PolyData = melon.load_polydata(cfg.post_mandible)

    pre_mandible.point_data["_PointId"] = np.arange(pre_mandible.n_points)
    pre_mandible_floating: pv.PolyData = melon.tri.extract_cells(
        pre_mandible, np.flatnonzero(pre_mandible.cell_data["Floating"])
    )
    melon.save(cherries.temp("10-pre-mandible-floating.vtp"), pre_mandible_floating)
    pre_mandible.point_data["Floating"] = np.zeros((pre_mandible.n_points,), bool)
    pre_mandible.point_data["Floating"][
        pre_mandible_floating.point_data["_PointId"]
    ] = True

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
    melon.save(cherries.temp("10-post-mandible-aligned.vtp"), post_mandible)

    nearest: melon.NearestPointOnSurfaceResult = melon.nearest_point_on_surface(
        post_mandible, pre_mandible, distance_threshold=1.0, normal_threshold=None
    )

    pre_mandible.point_data["Osteotomy"] = (
        nearest.distance > cfg.osteotomy_distance_threshold
    ) & ~pre_mandible.point_data["Floating"]
    pre_mandible.point_data["Distance"] = nearest.distance
    melon.save(cfg.output, pre_mandible)


if __name__ == "__main__":
    cherries.main(main)
