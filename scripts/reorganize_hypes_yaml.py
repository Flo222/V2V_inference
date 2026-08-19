#!/usr/bin/env python3
"""One-shot, lossless reorganization of OpenCOOD experiment YAML files."""
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path("opencood/hypes_yaml")


def move(source: str, target: str) -> None:
    src = ROOT / source
    dst = ROOT / target
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print("{} -> {}".format(source, target))


def move_glob(source_dir: str, target_dir: str) -> None:
    for src in sorted((ROOT / source_dir).glob("*.yaml")):
        move(str(src.relative_to(ROOT)), str(Path(target_dir) / src.name))


def main() -> None:
    # OPV2V baselines.
    move("point_pillar_where2comm.yaml",
         "opv2v/baselines/where2comm/point_pillar_where2comm.yaml")
    move("point_pillar_v2xvit_opv2v.yaml",
         "opv2v/baselines/v2xvit/point_pillar_v2xvit_opv2v.yaml")
    move("opv2v/lidar_only/pointpillar_cosdh_baseline.yaml",
         "opv2v/baselines/cosdh/pointpillar_cosdh_baseline.yaml")
    move("opv2v/lidar_only/pointpillar_cosdh_markov.yaml",
         "opv2v/baselines/cosdh/pointpillar_cosdh_markov.yaml")
    move("point_pillar_rocooper_opv2v.yaml",
         "opv2v/baselines/rocooper/point_pillar_rocooper_opv2v.yaml")
    move("point_pillar_rocooper_opv2v_markov_sync_add_coopdiff.yaml",
         "opv2v/baselines/rocooper/point_pillar_rocooper_opv2v_markov.yaml")
    for name in ("point_pillar_diff_teacher_opv2v_e20.yaml",
                 "point_pillar_nofusion_opv2v_e20.yaml",
                 "point_pillar_nofusion_opv2v_range140.yaml"):
        move(name, "opv2v/baselines/coopdiff/" + name)
    move("point_pillar_diff_teacher_opv2v_smoke.yaml",
         "opv2v/baselines/coopdiff/legacy/point_pillar_diff_teacher_opv2v_smoke.yaml")

    # OPV2V ARCE experiments.
    for name in ("point_pillar_where2comm_arce_c2mab.yaml",
                 "point_pillar_where2comm_arce_c2mab_comp.yaml",
                 "point_pillar_where2comm_arce_c2mab_comp_div.yaml",
                 "point_pillar_where2comm_arce_fixed.yaml",
                 "point_pillar_where2comm_arce_random.yaml",
                 "point_pillar_where2comm_arce_w2c_channel.yaml"):
        move("where2comm_arce_final/" + name,
             "opv2v/arce/where2comm/final/" + name)
    # The standalone ablation directory is canonical.  The differing copies
    # found inside the former final directory are retained as historical data.
    move_glob("where2comm_arce_ablation", "opv2v/arce/where2comm/ablation")
    move_glob("where2comm_arce_final",
              "opv2v/arce/where2comm/final/legacy_ablation")
    for name in ("point_pillar_v2xvit_opv2v_arce.yaml",
                 "point_pillar_v2xvit_opv2v_arce_dc2mab_full.yaml",
                 "point_pillar_v2xvit_opv2v_arce_markov.yaml",
                 "point_pillar_v2xvit_opv2v_arce_on.yaml"):
        move(name, "opv2v/arce/v2xvit/" + name)
    # Historical sweep sets.
    move_glob("arce_baselines/fixed_sweep", "opv2v/sweeps/v2xvit/ideal/fixed")
    move_glob("arce_baselines/random", "opv2v/sweeps/v2xvit/ideal/random")
    move_glob("arce_baselines_markov/fixed_sweep", "opv2v/sweeps/v2xvit/markov/fixed")
    move_glob("arce_baselines_markov/random", "opv2v/sweeps/v2xvit/markov/random")
    move_glob("arce_baselines_bad/fixed_sweep",
              "opv2v/sweeps/v2xvit/legacy/bad_channel")
    move_glob("arce_baselines_bad_no_completion/fixed_sweep",
              "opv2v/sweeps/v2xvit/legacy/bad_no_completion")

    # V2X-Real baselines and ARCE configurations.
    move("v2xreal/point_pillar_where2comm_v2xreal_vc.yaml",
         "v2xreal/baselines/where2comm/point_pillar_where2comm_v2xreal_vc.yaml")
    for name in ("point_pillar_v2xvit_v2xreal_vc.yaml",
                 "point_pillar_v2xvit_markov_v2xreal_vc.yaml"):
        move("v2xreal/" + name, "v2xreal/baselines/v2xvit/" + name)
    for name in ("point_pillar_cosdh_v2xreal_vc.yaml",
                 "point_pillar_cosdh_markov_v2xreal_vc.yaml"):
        move("v2xreal/" + name, "v2xreal/baselines/cosdh/" + name)
    for name in ("point_pillar_rocooper_v2xreal_vc.yaml",
                 "point_pillar_rocooper_markov_v2xreal_vc.yaml"):
        move("v2xreal/" + name, "v2xreal/baselines/rocooper/" + name)
    for name in ("point_pillar_diff_student_markov_v2xreal_vc.yaml",
                 "point_pillar_diff_student_v2xreal_vc.yaml",
                 "point_pillar_diff_teacher_v2xreal_vc.yaml",
                 "point_pillar_nofusion_v2xreal_vc.yaml"):
        move("v2xreal/" + name, "v2xreal/baselines/coopdiff/" + name)
    move("v2xreal/point_pillar_where2comm_arce_c2mab_v2xreal_vc.yaml",
         "v2xreal/arce/where2comm/point_pillar_where2comm_arce_c2mab_v2xreal_vc.yaml")
    move("v2xreal/point_pillar_v2xvit_native_payload_arce_markov_v2xreal_vc.yaml",
         "v2xreal/arce/v2xvit/point_pillar_v2xvit_native_payload_arce_markov_v2xreal_vc.yaml")

    # OpenCOOD upstream configurations unrelated to the active five baselines.
    for name in ("ciassd_early_fusion.yaml", "ciassd_intermediate_fusion.yaml"):
        move(name, "upstream/ciassd/" + name)
    move("fpvrcnn_intermediate_fusion.yaml", "upstream/fpvrcnn/fpvrcnn_intermediate_fusion.yaml")
    for name in ("pixor_early_fusion.yaml", "pixor_intermediate_fusion.yaml", "pixor_late_fusion.yaml"):
        move(name, "upstream/pixor/" + name)
    for name in ("point_pillar_coalign.yaml", "point_pillar_cobevt.yaml",
                 "point_pillar_early_fusion.yaml", "point_pillar_fcooper.yaml",
                 "point_pillar_intermediate_fusion.yaml", "point_pillar_intermediate_V2VAM.yaml",
                 "point_pillar_late_fusion.yaml", "point_pillar_v2vnet.yaml",
                 "point_pillar_v2xvit.yaml"):
        move(name, "upstream/point_pillar/" + name)
    for name in ("second_early_fusion.yaml", "second_intermediate_fusion.yaml", "second_late_fusion.yaml"):
        move(name, "upstream/second/" + name)
    move("visualization.yaml", "visualization/visualization.yaml")

    remaining = list(ROOT.glob("*.yaml"))
    if remaining:
        raise RuntimeError("Unexpected root YAML files: {}".format(remaining))
    print("HYPES_YAML_REORGANIZED")


if __name__ == "__main__":
    main()
