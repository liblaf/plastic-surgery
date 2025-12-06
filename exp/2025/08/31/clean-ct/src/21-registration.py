import functools
import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float

from liblaf import cherries, grapes, melon
from liblaf.plastic_surgery import MetaDataset

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    inputs_dir: Path = cherries.input("11-surface")

    template_cranium: Path = cherries.input("20-template-cranium.vtp")
    template_mandible: Path = cherries.input("20-template-mandible.vtp")
    template_skin: Path = cherries.input(
        "00-XYZ_ReadyToSculpt_eyesClosed_GEO_PolyGroups.obj"
    )

    outputs_dir: Path = cherries.output("21-registration")


def icp(
    source: pv.PolyData, target: pv.PolyData
) -> tuple[Float[np.ndarray, "4 4"], float]:
    source_tm: tm.Trimesh = melon.as_trimesh(source)
    target_tm: tm.Trimesh = melon.as_trimesh(target)
    matrix: Float[np.ndarray, "4 4"]
    cost: float
    matrix, _, cost = tm.registration.icp(
        source_tm.sample(10000),
        target_tm.sample(10000),
        max_iterations=100,
        reflection=False,
        translation=True,
        scale=False,
    )
    return matrix, cost


def register_acquisition(
    folder: Path,
    *,
    template_skin: pv.PolyData,
    template_skin_landmarks: Float[np.ndarray, "l 3"],
    template_cranium: pv.PolyData,
    template_cranium_landmarks: Float[np.ndarray, "l 3"],
    template_mandible: pv.PolyData,
    template_mandible_landmarks: Float[np.ndarray, "l 3"],
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData] | None:
    skin_file: Path = folder / "skin.ply"
    skin_landmarks: Float[np.ndarray, "l 3"] = melon.load_landmarks(skin_file)
    if skin_landmarks.size == 0:
        return None
    cranium_landmarks: Float[np.ndarray, "l 3"] = melon.load_landmarks(
        folder / "cranium.landmarks.json"
    )
    if cranium_landmarks.size == 0:
        return None
    mandible_landmarks: Float[np.ndarray, "l 3"] = melon.load_landmarks(
        folder / "mandible.landmarks.json"
    )
    if mandible_landmarks.size == 0:
        return None
    skin: pv.PolyData = melon.load_polydata(skin_file)
    skull: pv.PolyData = melon.load_polydata(folder / "skull.ply")
    skin = melon.tri.fast_wrapping(
        template_skin,
        skin,
        source_landmarks=template_skin_landmarks,
        target_landmarks=skin_landmarks,
        free_polygons_floating=melon.tri.select_groups(
            template_skin,
            [
                "Caruncle",
                "EarSocket",
                "EyeSocketBottom",
                "EyeSocketTop",
                "LipInnerBottom",
                "LipInnerTop",
                "MouthSocketBottom",
                "MouthSocketTop",
                "NeckBack",
                "NeckFront",
                "Nostril",
            ],
        ),
    )
    cranium: pv.PolyData = melon.tri.fast_wrapping(
        template_cranium,
        skull,
        source_landmarks=template_cranium_landmarks,
        target_landmarks=cranium_landmarks,
        free_polygons_floating=template_cranium.cell_data["Floating"],
    )
    mandible: pv.PolyData = melon.tri.fast_wrapping(
        template_mandible,
        skull,
        source_landmarks=template_mandible_landmarks,
        target_landmarks=mandible_landmarks,
        free_polygons_floating=template_mandible.cell_data["Floating"],
    )
    return skin, cranium, mandible


def main(cfg: Config) -> None:
    meta: MetaDataset = grapes.load(cfg.inputs_dir / "dataset.json", type=MetaDataset)
    template_skin: pv.PolyData = melon.load_polydata(cfg.template_skin)
    template_skin.clean(inplace=True)
    template_skin_landmarks: Float[np.ndarray, "l 3"] = melon.load_landmarks(
        cfg.template_skin
    )
    template_cranium: pv.PolyData = melon.load_polydata(cfg.template_cranium)
    template_cranium_landmarks: Float[np.ndarray, "l 3"] = melon.load_landmarks(
        cfg.template_cranium
    )
    template_mandible: pv.PolyData = melon.load_polydata(cfg.template_mandible)
    template_mandible_landmarks: Float[np.ndarray, "l 3"] = melon.load_landmarks(
        cfg.template_mandible
    )
    register_acquisition_partial = functools.partial(
        register_acquisition,
        template_skin=template_skin,
        template_skin_landmarks=template_skin_landmarks,
        template_cranium=template_cranium,
        template_cranium_landmarks=template_cranium_landmarks,
        template_mandible=template_mandible,
        template_mandible_landmarks=template_mandible_landmarks,
    )
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    grapes.save(cfg.outputs_dir / "dataset.json", meta, order="sorted")
    for patient_id, meta_patient in meta.patients.items():
        patient_dir: Path = cfg.inputs_dir / patient_id
        pre_acq_dir: Path = patient_dir / meta_patient.acquisitions[
            0
        ].datetime.strftime("%Y-%m-%d")
        pre_skin: pv.PolyData
        pre_cranium: pv.PolyData
        pre_mandible: pv.PolyData
        result: tuple[pv.PolyData, pv.PolyData, pv.PolyData] | None = (
            register_acquisition_partial(pre_acq_dir)
        )
        if result is None:
            logger.warning("%s (%s): missing landmarks", patient_id, meta_patient.name)
            continue
        pre_skin, pre_cranium, pre_mandible = result

        post_acq_dir: Path = patient_dir / meta_patient.acquisitions[
            -1
        ].datetime.strftime("%Y-%m-%d")
        post_skin: pv.PolyData
        post_cranium: pv.PolyData
        post_mandible: pv.PolyData
        result = register_acquisition_partial(post_acq_dir)
        if result is None:
            logger.warning("%s: missing landmarks", patient_id)
            continue
        post_skin, post_cranium, post_mandible = result

        post_to_pre: Float[np.ndarray, "4 4"]
        cost: float
        post_to_pre, cost = icp(post_cranium, pre_cranium)
        logger.info("%s: (Post -> Pre) ICP cost: %g", patient_id, cost)
        post_skin.transform(post_to_pre, inplace=True)
        post_cranium.transform(post_to_pre, inplace=True)
        post_mandible.transform(post_to_pre, inplace=True)

        output_patient_dir: Path = cfg.outputs_dir / patient_id
        melon.save(output_patient_dir / "pre-skin.vtp", pre_skin)
        melon.save(output_patient_dir / "pre-cranium.vtp", pre_cranium)
        melon.save(output_patient_dir / "pre-mandible.vtp", pre_mandible)
        melon.save(output_patient_dir / "post-skin.vtp", post_skin)
        melon.save(output_patient_dir / "post-cranium.vtp", post_cranium)
        melon.save(output_patient_dir / "post-mandible.vtp", post_mandible)


if __name__ == "__main__":
    cherries.main(main)
