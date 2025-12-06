from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    predict: Path = cherries.input("20-prediction.vtu")
    truth: Path = cherries.input("00-post-skin.vtp")

    output: Path = cherries.output("21-evaluation.vtp")


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.predict)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface.warp_by_vector("Displacement", inplace=True)
    truth: pv.PolyData = melon.load_polydata(cfg.truth)

    nearest: melon.NearestPointOnSurfaceResult = melon.nearest_point_on_surface(
        truth, surface, distance_threshold=1.0, normal_threshold=None
    )
    surface.point_data["Error"] = nearest.distance

    melon.save(cfg.output, surface)


if __name__ == "__main__":
    cherries.main(main)
