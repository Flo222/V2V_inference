import torch.nn as nn

from opencood.models.upstream.point_pillar import PointPillar


class PointPillarV2xreal(PointPillar):
    """
    V2X-Real 3-class PointPillar wrapper.

    Output:
      psm: anchor_number * num_class * num_class = 2 * 3 * 3 = 18
      rm : 7 * anchor_number * num_class = 7 * 2 * 3 = 42
    """
    def __init__(self, args):
        super(PointPillarV2xreal, self).__init__(args)

        num_class = int(args.get("num_class", 3))
        anchor_number = int(args.get("anchor_number", args.get("anchor_num", 2)))

        cls_out = anchor_number * num_class * num_class
        reg_out = 7 * anchor_number * num_class

        self.cls_head = nn.Conv2d(128 * 3, cls_out, kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, reg_out, kernel_size=1)
