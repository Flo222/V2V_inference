import argparse
import torch

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.communication.transport.packetization.packetizer import FeaturePacketizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--c", type=int, default=256)
    parser.add_argument("--h", type=int, default=48)
    parser.add_argument("--w", type=int, default=176)
    args = parser.parse_args()

    hypes = load_yaml(args.hypes_yaml)
    arce_cfg = hypes["model"]["args"]["arce"]

    packetizer = FeaturePacketizer(arce_cfg)
    feature = torch.randn(args.c, args.h, args.w)

    result = packetizer.packetize(feature)

    areas = [m.spatial_area for m in result.metas]
    max_area = max(areas)
    min_area = min(areas)

    print("packetizer:", packetizer)
    print("feature shape:", tuple(feature.shape))
    print("mode:", result.mode)
    print("num_packets:", result.num_packets)
    print("packet tensor shape:", tuple(result.packets.shape))
    print("packet_grid_shape:", result.packet_grid_shape)
    print("min spatial_area:", min_area)
    print("max spatial_area:", max_area)

    patch_max_size = arce_cfg["packetizer"].get("patch_max_size", None)
    print("configured patch_max_size:", patch_max_size)

    if patch_max_size is not None:
        assert max_area <= int(patch_max_size), (
            f"max patch area {max_area} > patch_max_size {patch_max_size}"
        )

    recon = packetizer.unpacketize(
        packets=result.packets,
        metas=result.metas,
        original_shape=result.original_shape,
    )

    diff = (recon - feature).abs().max().item()
    print("unpacketize max abs diff:", diff)
    assert diff == 0.0, "packetize/unpacketize is not lossless before channel loss"


if __name__ == "__main__":
    main()