from pathlib import Path

import numpy as np
import pyvista as pv
from liblaf.apple.constants import DIRICHLET_MASK, DIRICHLET_VALUE, PRESTRAIN

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    pre_mandible: Path = cherries.input("00-pre-mandible.vtp")
    post_mandible: Path = cherries.input("00-post-mandible.vtp")
    tetmesh: Path = cherries.input("12-tetmesh.vtu")

    output: Path = cherries.output("13-tetmesh.vtu")

    osteotomy_to_post_threshold: float = 20.0  # millimeters
    skin_to_osteotomy_threshold: float = 20.0  # millimeters

    a0: float = 1e2
    a1: float = 1e-1
    a2: float = 1e-3


def main(cfg: Config) -> None:
    pre_mandible: pv.PolyData = melon.load_polydata(cfg.pre_mandible)
    post_mandible: pv.PolyData = melon.load_polydata(cfg.post_mandible)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    tetmesh.point_data["_PointId"] = np.arange(tetmesh.n_points)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    skin: pv.PolyData = melon.tri.extract_points(surface, surface.point_data["IsSkin"])
    osteotomy: pv.PolyData = melon.tri.extract_points(
        surface, surface.point_data["Osteotomy"]
    )

    surface.point_data[DIRICHLET_MASK] = (
        surface.point_data["IsCranium"] | surface.point_data["IsMandible"]
    ) & ~surface.point_data["Osteotomy"]
    surface.point_data[DIRICHLET_VALUE] = np.zeros((surface.n_points, 3))

    tetmesh = melon.transfer_tri_point_to_tet(
        surface,
        tetmesh,
        data=[DIRICHLET_MASK, DIRICHLET_VALUE],
        fill={DIRICHLET_MASK: False, DIRICHLET_VALUE: 0.0},
        point_id="_PointId",
    )

    osteotomy_to_post: melon.NearestPointOnSurfaceResult = (
        melon.nearest_point_on_surface(
            post_mandible,
            osteotomy,
            distance_threshold=cfg.osteotomy_to_post_threshold / post_mandible.length,
            normal_threshold=None,
        )
    )
    osteotomy.point_data["PreToPostMandible"] = osteotomy_to_post.distance
    melon.save(cherries.temp("13-osteotomy.vtp"), osteotomy)
    melon.save(cherries.temp("13-pre-mandible.vtp"), pre_mandible)
    melon.save(cherries.temp("13-post-mandible.vtp"), post_mandible)
    osteotomy = osteotomy.point_data_to_cell_data(pass_point_data=True)  # pyright: ignore[reportAssignmentType]

    skin_to_osteotomy: melon.NearestPointOnSurfaceResult = (
        melon.nearest_point_on_surface(
            osteotomy,
            skin,
            distance_threshold=cfg.skin_to_osteotomy_threshold / osteotomy.length,
            normal_threshold=None,
        )
    )
    skin.point_data["SkinToOsteotomy"] = skin_to_osteotomy.distance
    skin.point_data["PreToPostMandible"] = np.where(
        skin_to_osteotomy.missing,
        0.0,
        osteotomy.cell_data["PreToPostMandible"][skin_to_osteotomy.triangle_id],
    )

    skin.point_data[PRESTRAIN] = (
        -cfg.a0
        * np.exp(-cfg.a1 * skin.point_data["SkinToOsteotomy"])
        * (1.0 - np.exp(-cfg.a2 * skin.point_data["PreToPostMandible"]))
    )
    melon.save(cherries.temp("13-skin.vtp"), skin)

    tetmesh = melon.transfer_tri_point_to_tet(
        skin,
        tetmesh,
        data=["PreToPostMandible", "SkinToOsteotomy", PRESTRAIN],
        fill={"PreToPostMandible": 0.0, "SkinToOsteotomy": np.inf, PRESTRAIN: 0.0},
        point_id="_PointId",
    )
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
