# -*- coding: utf-8 -*-
"""PointPillar + RoCooper adapter for V2X-Real.

This file reuses the existing OPV2V RoCooper implementation and only
changes the detection heads to match V2X-Real's 3-class target format.

V2X-Real convention used by VoxelPostprocessorV2XReal / point_pillar_loss_v2xreal:
    psm channels = anchor_number * num_class * num_class
    rm  channels = 7 * anchor_number * num_class
For the current V2X-Real setup: anchor_number=2, num_class=3 -> psm=18, rm=42.
"""

from typing import Any, Dict

import torch.nn as nn

from opencood.models.baselines.rocooper.models.point_pillar_rocooper import PointPillarRocooper


class PointPillarRocooperV2xreal(PointPillarRocooper):
    """V2X-Real wrapper for the existing RoCooper PointPillar model."""

    def __init__(self, args: Dict[str, Any]):
        # Build all RoCooper components from the existing implementation.
        super(PointPillarRocooperV2xreal, self).__init__(args)

        self.num_class = int(args.get("num_class", 3))
        self.anchor_number = self._get_anchor_number(args)

        # Replace OPV2V single-class heads with V2X-Real 3-class heads.
        self.cls_head = nn.Conv2d(
            self.feature_dim,
            self.anchor_number * self.num_class * self.num_class,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            self.feature_dim,
            7 * self.anchor_number * self.num_class,
            kernel_size=1,
        )

        # If config requests frozen backbone/head, freeze the newly-created heads too.
        if self.backbone_fix_flag:
            self.backbone_fix()


# Extra aliases for OpenCOOD's dynamic importer tolerance.
PointPillarRoCooperV2xreal = PointPillarRocooperV2xreal
PointPillarROCOOPERV2XREAL = PointPillarRocooperV2xreal
