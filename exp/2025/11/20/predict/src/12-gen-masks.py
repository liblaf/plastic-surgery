from pathlib import Path

import numpy as np
import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    skin: Path = cherries.input("00-pre-skin.vtp")
    cranium: Path = cherries.input("00-pre-cranium.vtp")
    mandible: Path = cherries.input("11-pre-mandible.vtp")
    tetmesh: Path = cherries.input("10-tetmesh.vtu")

    output: Path = cherries.output("12-tetmesh.vtu")


def main(cfg: Config) -> None:
    skin: pv.PolyData = melon.load_polydata(cfg.skin)
    cranium: pv.PolyData = melon.load_polydata(cfg.cranium)
    mandible: pv.PolyData = melon.load_polydata(cfg.mandible)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    tetmesh.point_data["_PointId"] = np.arange(tetmesh.n_points)
    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]

    skin.cell_data["IsSkin"] = np.ones((skin.n_cells,), bool)
    cranium.cell_data["IsSkin"] = np.zeros((cranium.n_cells,), bool)
    mandible.cell_data["IsSkin"] = np.zeros((mandible.n_cells,), bool)

    skin.cell_data["IsCranium"] = np.zeros((skin.n_cells,), bool)
    cranium.cell_data["IsCranium"] = np.ones((cranium.n_cells,), bool)
    mandible.cell_data["IsCranium"] = np.zeros((mandible.n_cells,), bool)

    skin.cell_data["IsMandible"] = np.zeros((skin.n_cells,), bool)
    cranium.cell_data["IsMandible"] = np.zeros((cranium.n_cells,), bool)
    mandible.cell_data["IsMandible"] = np.ones((mandible.n_cells,), bool)

    skin.cell_data["Osteotomy"] = np.zeros((skin.n_cells,), bool)
    cranium.cell_data["Osteotomy"] = np.zeros((cranium.n_cells,), bool)

    data_names: list[str] = ["IsSkin", "IsCranium", "IsMandible", "Osteotomy"]

    surface = melon.transfer_tri_cell_to_point_category(
        pv.merge([skin, cranium, mandible]),
        surface,
        data=data_names,
        fill=False,
        nearest=melon.NearestPointOnSurface(
            distance_threshold=0.01, normal_threshold=None
        ),
    )
    tetmesh = melon.transfer_tri_point_to_tet(
        surface, tetmesh, data=data_names, fill=False, point_id="_PointId"
    )

    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
