from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from liblaf.apple import ARAP, Forward, Gravity, Model, ModelBuilder
from liblaf.apple.constants import DIRICHLET_MASK, DIRICHLET_VALUE, MASS, MU, POINT_ID
from liblaf.peach.optim import ScipyOptimizer

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    tetmesh: Path = cherries.input("00-tetmesh.vtu")

    output: Path = cherries.output("10-solution.vtu")


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)
    tetmesh = tetmesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    tetmesh.point_data[DIRICHLET_MASK] = (
        tetmesh.point_data["IsCranium"] | tetmesh.point_data["IsMandible"]
    )
    tetmesh.point_data["IsSurface"] = False
    surface_idx = tetmesh.surface_indices()
    tetmesh.point_data["IsSurface"][surface_idx] = True
    tetmesh.point_data[DIRICHLET_MASK] |= (
        tetmesh.point_data["IsSurface"] & ~tetmesh.point_data["IsSkin"]
    )
    tetmesh.point_data[DIRICHLET_VALUE] = np.zeros((tetmesh.n_points, 3))
    tetmesh.cell_data[MASS] = 1e-3 * tetmesh.cell_data["Volume"]
    tetmesh = tetmesh.cell_data_to_point_data()  # pyright: ignore[reportAssignmentType]
    tetmesh.cell_data[MU] = np.full((tetmesh.n_cells,), 1e2)

    builder = ModelBuilder()
    tetmesh = builder.assign_global_ids(tetmesh)
    builder.add_dirichlet(tetmesh)

    tetmesh_elastic: ARAP = ARAP.from_pyvista(tetmesh)
    builder.add_energy(tetmesh_elastic)

    gravity: Gravity = Gravity.from_pyvista(
        tetmesh, gravity=jnp.asarray([0.0, 0.0, 9.81e3])
    )
    builder.add_energy(gravity)

    model: Model = builder.finalize()
    forward: Forward = Forward(
        model, optimizer=ScipyOptimizer(method="trust-constr", options={"verbose": 3})
    )

    solution: ScipyOptimizer.Solution = forward.step()
    ic(solution)
    tetmesh.point_data["Displacement"] = model.u_full[tetmesh.point_data[POINT_ID]]  # pyright: ignore[reportArgumentType]
    melon.save(cfg.output, tetmesh)


if __name__ == "__main__":
    cherries.main(main)
