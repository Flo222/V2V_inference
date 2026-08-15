"""Model-module registry for the structured V2V_inference layout.

Existing YAML ``model.core_method`` values remain stable.  The registry
redirects the five active collaborative-perception baselines to their
canonical package locations, so no flat compatibility model files are
required under :mod:`opencood.models`.
"""
from importlib import import_module

MODEL_MODULES = {
    'point_pillar_base_multi_scale_teacher': 'opencood.models.baselines.coopdiff.models.point_pillar_base_multi_scale_teacher',
    'point_pillar_base_multi_scale_teacher_diff': 'opencood.models.baselines.coopdiff.models.point_pillar_base_multi_scale_teacher_diff',
    'point_pillar_base_multi_scale_teacher_diff_v2xreal': 'opencood.models.baselines.coopdiff.models.point_pillar_base_multi_scale_teacher_diff_v2xreal',
    'point_pillar_cosdh': 'opencood.models.baselines.cosdh.models.point_pillar_cosdh',
    'point_pillar_cosdh_markov': 'opencood.models.baselines.cosdh.models.point_pillar_cosdh_markov',
    'point_pillar_cosdh_markov_v2xreal': 'opencood.models.baselines.cosdh.models.point_pillar_cosdh_markov_v2xreal',
    'point_pillar_cosdh_v2xreal': 'opencood.models.baselines.cosdh.models.point_pillar_cosdh_v2xreal',
    'point_pillar_diff_stu': 'opencood.models.baselines.coopdiff.models.point_pillar_diff_stu',
    'point_pillar_diff_stu_markov': 'opencood.models.baselines.coopdiff.models.point_pillar_diff_stu_markov',
    'point_pillar_diff_stu_markov_v2xreal': 'opencood.models.baselines.coopdiff.models.point_pillar_diff_stu_markov_v2xreal',
    'point_pillar_diff_stu_v2xreal': 'opencood.models.baselines.coopdiff.models.point_pillar_diff_stu_v2xreal',
    'point_pillar_rocooper': 'opencood.models.baselines.rocooper.models.point_pillar_rocooper',
    'point_pillar_rocooper_v2xreal': 'opencood.models.baselines.rocooper.models.point_pillar_rocooper_v2xreal',
    'point_pillar_transformer': 'opencood.models.baselines.v2xvit.point_pillar_transformer',
    'point_pillar_transformer_opv2v': 'opencood.models.baselines.v2xvit.point_pillar_transformer_opv2v',
    'point_pillar_transformer_opv2v_arce': 'opencood.models.baselines.v2xvit.point_pillar_transformer_opv2v_arce',
    'point_pillar_transformer_v2xreal': 'opencood.models.baselines.v2xvit.point_pillar_transformer_v2xreal',
    'point_pillar_transformer_v2xreal_arce': 'opencood.models.baselines.v2xvit.point_pillar_transformer_v2xreal_arce',
    'point_pillar_where2comm': 'opencood.models.baselines.where2comm.point_pillar_where2comm',
    'point_pillar_where2comm_arce': 'opencood.models.baselines.where2comm.point_pillar_where2comm_arce',
    'point_pillar_where2comm_arce_v2xreal': 'opencood.models.baselines.where2comm.point_pillar_where2comm_arce_v2xreal',
    'point_pillar_where2comm_v2xreal': 'opencood.models.baselines.where2comm.point_pillar_where2comm_v2xreal',
}

def resolve_model_module(core_method: str) -> str:
    """Return the import path for a YAML ``core_method`` value."""
    return MODEL_MODULES.get(core_method, f"opencood.models.upstream.{core_method}")

def import_model_module(core_method: str):
    """Import the model module registered for ``core_method``."""
    return import_module(resolve_model_module(core_method))
