import argparse
import yaml

STATE_DELAY_MS = {
    "good": 100,
    "medium": 200,
    "bad": 400,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state", required=True, choices=["good", "medium", "bad"])
    args = parser.parse_args()

    delay_ms = STATE_DELAY_MS[args.state]

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("wild_setting", {})
    cfg["wild_setting"]["async"] = True
    cfg["wild_setting"]["async_mode"] = "sim"
    cfg["wild_setting"]["async_overhead"] = delay_ms
    cfg["wild_setting"]["frame_interval_ms"] = 100

    arce = cfg["model"]["args"]["arce"]
    arce["late_policy"] = "allow"
    arce["enable_deadline_drop"] = False

    arce.setdefault("latency", {})
    arce["latency"]["enabled"] = False
    arce["latency"]["late_policy"] = "allow"

    if "channel" in arce:
        arce["channel"]["mode"] = "fixed"
        arce["channel"]["fixed_state"] = args.state

    with open(args.config, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Updated {args.config}")
    print(f"channel_state = {args.state}")
    print(f"V2X-ViT async_overhead = {delay_ms} ms")
    print(f"delay_slots = {delay_ms // 100}")

if __name__ == "__main__":
    main()
