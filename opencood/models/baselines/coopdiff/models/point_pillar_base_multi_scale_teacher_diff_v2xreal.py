# -*- coding: utf-8 -*-
"""V2X-Real adaptation of the CoopDiff teacher model.

The OPV2V CoopDiff teacher is single-class and therefore uses 2 cls channels
and 14 reg channels.  V2X-Real is a three-class benchmark, so the detection
heads must be changed to:
    psm = anchor_number * num_class * num_class = 18
    rm  = 7 * anchor_number * num_class       = 42

All feature extraction, multi-scale fusion and diffuser module names are kept
identical to the OPV2V teacher so teacher->student checkpoint mapping remains
simple.
"""

import torch.nn as nn

from opencood.models.baselines.coopdiff.models.point_pillar_base_multi_scale_teacher_diff import \
    PointPillarBaseMultiScaleTeacherdiff


class PointPillarBaseMultiScaleTeacherdiffV2xreal(PointPillarBaseMultiScaleTeacherdiff):
    def __init__(self, args):
        super(PointPillarBaseMultiScaleTeacherdiffV2xreal, self).__init__(args)
        self.num_class = int(args.get('num_class', 3))
        self.anchor_number = int(args.get('anchor_number', args.get('anchor_num', 2)))
        head_dim = int(args.get('head_dim', 256))

        self.cls_head_teacher = nn.Conv2d(
            head_dim,
            self.anchor_number * self.num_class * self.num_class,
            kernel_size=1,
        )
        self.reg_head_teacher = nn.Conv2d(
            head_dim,
            7 * self.anchor_number * self.num_class,
            kernel_size=1,
        )
