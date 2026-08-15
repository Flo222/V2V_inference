# -*- coding: utf-8 -*-
"""V2X-Real adaptation of CoopDiff student.

This wrapper keeps the OPV2V CoopDiff implementation intact and only replaces
student/teacher detection heads with V2X-Real three-class heads.  The inherited
forward still returns ``psm`` and ``rm``, so it is compatible with
point_pillar_loss_v2xreal and VoxelPostprocessorV2XReal.
"""

import torch.nn as nn

from opencood.models.baselines.coopdiff.models.point_pillar_diff_stu import PointPillarDiffStu


class PointPillarDiffStuV2xreal(PointPillarDiffStu):
    def __init__(self, args):
        super(PointPillarDiffStuV2xreal, self).__init__(args)
        self.num_class = int(args.get('num_class', 3))
        self.anchor_number = int(args.get('anchor_number', args.get('anchor_num', 2)))
        head_dim = int(args.get('head_dim', 256))

        self.cls_head = nn.Conv2d(
            head_dim,
            self.anchor_number * self.num_class * self.num_class,
            kernel_size=1,
        )
        self.reg_head = nn.Conv2d(
            head_dim,
            7 * self.anchor_number * self.num_class,
            kernel_size=1,
        )
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
