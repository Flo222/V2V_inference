import argparse
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset


STATE_ID_TO_NAME = {
    -1: "ego_or_pad",
    0: "good",
    1: "medium",
    2: "bad",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--num_samples", type=int, default=30)
    args = parser.parse_args()

    hypes = load_yaml(args.hypes_yaml)
    dataset = build_dataset(hypes, visualize=False, train=False)

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
    )

    for i, batch_data in enumerate(loader):
        if i >= args.num_samples:
            break

        ego = batch_data["ego"]

        record_len = int(ego["record_len"][0])
        prior = ego["prior_encoding"][0, :record_len, :]
        delay_slots_from_prior = prior[:, 1].tolist()

        print("=" * 80)
        print("sample:", i)
        print("record_len:", record_len)
        print("prior delay_slots:", delay_slots_from_prior)

        if "channel_state_ids" in ego:
            ids = ego["channel_state_ids"][0, :record_len].tolist()
            names = [STATE_ID_TO_NAME.get(int(x), "unknown") for x in ids]
            print("channel_state_ids:", ids)
            print("channel_states:", names)

        if "channel_delay_ms" in ego:
            delay_ms = ego["channel_delay_ms"][0, :record_len].tolist()
            print("channel_delay_ms:", delay_ms)

        if "channel_delay_slots" in ego:
            delay_slots = ego["channel_delay_slots"][0, :record_len].tolist()
            print("channel_delay_slots:", delay_slots)


if __name__ == "__main__":
    main()