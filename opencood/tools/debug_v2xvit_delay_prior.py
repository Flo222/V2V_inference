import argparse
import torch
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    hypes = load_yaml(args.hypes_yaml)
    dataset = build_dataset(hypes, visualize=False, train=False)

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False
    )

    for i, batch_data in enumerate(loader):
        if i >= args.num_samples:
            break

        ego = batch_data["ego"]
        record_len = ego["record_len"]
        prior = ego["prior_encoding"]          # [B, max_cav, 3]
        delay = prior[0, :, 1]                 # channel 1 = time_delay / delay_slots

        print("=" * 80)
        print(f"sample index: {i}")
        print("record_len:", record_len.tolist())
        print("prior_encoding shape:", tuple(prior.shape))
        print("delay_slots prior_encoding[0, :, 1]:", delay.tolist())

        if "spatial_correction_matrix" in ego:
            scm = ego["spatial_correction_matrix"]  # [B, max_cav, 4, 4]
            eye = torch.eye(4).view(1, 1, 4, 4)
            max_dev = (scm - eye).abs().max().item()
            print("spatial_correction_matrix max |M-I|:", max_dev)

if __name__ == "__main__":
    main()
