from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    cranium: Path = cherries.input("00-cranium.vtp")
    mandible: Path = cherries.input("00-mandible.vtp")
    tetmesh: Path = cherries.input("00-tetmesh.vtu")

    output: Path = cherries.output("10-tetmesh.vtu")


def main(cfg: Config) -> None:
    cranium: pv.PolyData = melon.load_polydata(cfg.cranium)
    mandible: pv.PolyData = melon.load_polydata(cfg.mandible)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    tetmesh.point_data["_PointId"] = np.arange(tetmesh.n_points)

    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    nearest: melon.NearestPointOnSurfaceResult = melon.nearest_point_on_surface(
        pv.merge([cranium, mandible]),
        surface,
        distance_threshold=1.0,
        normal_threshold=None,
    )
    surface.point_data["SkinToSkull"] = nearest.distance

    tetmesh = melon.transfer_tri_point_to_tet(
        surface, tetmesh, data="SkinToSkull", point_id="_PointId"
    )

    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
