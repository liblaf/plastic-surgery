from pathlib import Path

import numpy as np
import pyvista as pv
from liblaf.apple import ARAP, Forward, MassSpringPrestrain, Model, ModelBuilder
from liblaf.apple.constants import MU, POINT_ID, STIFFNESS
from liblaf.peach.optim import ScipyOptimizer

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    tetmesh: Path = cherries.input("13-tetmesh.vtu")

    output: Path = cherries.output("20-prediction.vtu")


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    tetmesh.cell_data[MU] = np.full((tetmesh.n_cells,), 1e0)

    builder = ModelBuilder()
    tetmesh = builder.assign_global_ids(tetmesh)
    builder.add_dirichlet(tetmesh)

    tetmesh_energy: ARAP = ARAP.from_pyvista(tetmesh)
    builder.add_energy(tetmesh_energy)

    surface: pv.PolyData = tetmesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    edges: pv.PolyData = surface.extract_all_edges()  # pyright: ignore[reportAssignmentType]
    edges = edges.point_data_to_cell_data(pass_point_data=True)  # pyright: ignore[reportAssignmentType]
    edges.cell_data[STIFFNESS] = np.full((edges.n_cells,), 2e1)
    melon.save(cherries.temp("20-surface-edges.vtp"), edges)

    surface_energy: MassSpringPrestrain = MassSpringPrestrain.from_pyvista(edges)
    ic(surface_energy, short_arrays=False)
    # ic(jnp.count_nonzero(surface_energy.prestrain))
    builder.add_energy(surface_energy)

    model: Model = builder.finalize()
    ic(model)
    ic(surface_energy.fun(model.u_full))
    forward = Forward(
        model, optimizer=ScipyOptimizer(method="trust-constr", options={"verbose": 3})
    )

    solution: ScipyOptimizer.Solution = forward.step()
    ic(solution)
    tetmesh.point_data["Displacement"] = model.u_full[tetmesh.point_data[POINT_ID]]  # pyright: ignore[reportArgumentType]
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
