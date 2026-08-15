# -*- coding: utf-8 -*-
"""V2X-Real adaptation of CoopDiff student with Markov feature channel.

Use this only for evaluation under channel impairment.  The learnable module
names match point_pillar_diff_stu_v2xreal, and the extra Markov channel is
parameter-free, so vanilla CoopDiff-V2XReal checkpoints can be reused.
"""

import torch.nn as nn

from opencood.models.baselines.coopdiff.models.point_pillar_diff_stu_markov import PointPillarDiffStuMarkov


class PointPillarDiffStuMarkovV2xreal(PointPillarDiffStuMarkov):
    def __init__(self, args):
        super(PointPillarDiffStuMarkovV2xreal, self).__init__(args)
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
