#!/usr/bin/env python3
from __future__ import annotations
import torch
from opencood.methods.arce.policies.complementarity import ego_complementarity, mask_iou, overlap_with_selected, union_masks


def main():
    ego = torch.zeros(1, 8, 8); ego[:, :4, :4] = 1
    cav1 = torch.zeros(1, 8, 8); cav1[:, 2:6, 2:6] = 1
    cav2 = torch.zeros(1, 8, 8); cav2[:, 4:, 4:] = 1
    print('comp cav1->ego:', ego_complementarity(cav1, ego))
    print('iou cav1 ego:', mask_iou(cav1, ego))
    u = union_masks([cav1])
    print('overlap cav2 with cav1:', overlap_with_selected(cav2, u))

if __name__ == '__main__':
    main()
