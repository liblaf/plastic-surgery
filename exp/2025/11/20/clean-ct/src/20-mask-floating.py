from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    cranium: Path = cherries.input("00-sculptor-cranium.ply")
    mandible: Path = cherries.input("00-sculptor-mandible.ply")

    output_cranium: Path = cherries.output("20-template-cranium.vtp")
    output_mandible: Path = cherries.output("20-template-mandible.vtp")


def main(cfg: Config) -> None:
    cranium: pv.PolyData = melon.load_polydata(cfg.cranium)
    mandible: pv.PolyData = melon.load_polydata(cfg.mandible)
    skull: pv.PolyData = pv.merge([cranium, mandible])

    distance_threshold: float = 0.02 * skull.length

    nearest = melon.nearest_point_on_surface(
        mandible,
        cranium.cell_centers(),
        distance_threshold=distance_threshold / mandible.length,
        normal_threshold=None,
    )
    cranium.cell_data["Distance"] = nearest.distance
    cranium.cell_data["Floating"] = ~nearest.missing
    melon.save(cfg.output_cranium, cranium)
    melon.save_landmarks(cfg.output_cranium, melon.load_landmarks(cfg.cranium))

    nearest: melon.NearestPointOnSurfaceResult = melon.nearest_point_on_surface(
        cranium,
        mandible.cell_centers(),
        distance_threshold=distance_threshold / cranium.length,
        normal_threshold=None,
    )
    mandible.cell_data["Distance"] = nearest.distance
    mandible.cell_data["Floating"] = ~nearest.missing
    melon.save(cfg.output_mandible, mandible)
    melon.save_landmarks(cfg.output_mandible, melon.load_landmarks(cfg.mandible))


if __name__ == "__main__":
    cherries.main(main)
