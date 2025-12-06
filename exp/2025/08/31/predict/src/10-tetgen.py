from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    skin: Path = cherries.input("00-pre-skin.vtp")
    cranium: Path = cherries.input("00-pre-cranium.vtp")
    mandible: Path = cherries.input("00-pre-mandible.vtp")

    output: Path = cherries.output("10-tetmesh.vtu")

    lr: float = 0.05 * 0.5
    epsr: float = 1e-3 * 0.5


def main(cfg: Config) -> None:
    skin: pv.PolyData = melon.load_polydata(cfg.skin)
    cranium: pv.PolyData = melon.load_polydata(cfg.cranium)
    mandible: pv.PolyData = melon.load_polydata(cfg.mandible)

    skull: pv.PolyData = pv.merge([cranium, mandible])
    skull.flip_faces(inplace=True)

    mesh: pv.UnstructuredGrid = melon.tetwild(
        pv.merge([skull, skin]), lr=cfg.lr, epsr=cfg.epsr
    )
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
