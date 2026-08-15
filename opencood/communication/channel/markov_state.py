import hashlib
import numpy as np


STATE_TO_ID = {
    "good": 0,
    "medium": 1,
    "bad": 2,
}

ID_TO_STATE = {
    0: "good",
    1: "medium",
    2: "bad",
}


class LinkLevelMarkovChannel(object):
    """
    Link-level three-state Markov channel.

    Each link has its own Markov chain:
        link_key = scenario_id + ego_id + cav_id

    Example:
        ego -> cav1: good -> good -> medium -> ...
        ego -> cav2: bad  -> bad  -> medium -> ...
        ego -> cav3: good -> medium -> bad -> ...

    This scheduler is deterministic for each link_key and frame_idx.
    """

    def __init__(self, cfg, frame_interval_ms=100):
        self.cfg = cfg or {}
        self.states = list(self.cfg.get("states", ["good", "medium", "bad"]))
        self.seed = int(self.cfg.get("seed", 2026))
        self.initial_state = str(self.cfg.get("initial_state", "good"))
        self.frame_interval_ms = float(frame_interval_ms)

        self.delay_ms = self.cfg.get(
            "delay_ms",
            {
                "good": 100,
                "medium": 200,
                "bad": 400,
            },
        )

        transition_matrix = self.cfg.get("transition_matrix", None)
        if transition_matrix is None:
            transition_matrix = {
                "good": {"good": 0.85, "medium": 0.12, "bad": 0.03},
                "medium": {"good": 0.15, "medium": 0.70, "bad": 0.15},
                "bad": {"good": 0.05, "medium": 0.25, "bad": 0.70},
            }

        self.P = self._build_matrix(transition_matrix)

        # cache: link_key -> [state_0, state_1, ..., state_t]
        self.cache = {}

    def _build_matrix(self, transition_matrix):
        P = np.zeros((len(self.states), len(self.states)), dtype=np.float64)

        for i, src in enumerate(self.states):
            row = transition_matrix[src]
            for j, dst in enumerate(self.states):
                P[i, j] = float(row[dst])

            row_sum = P[i].sum()
            if abs(row_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"Transition row for {src} must sum to 1, got {row_sum}"
                )

        return P

    def _stable_seed(self, link_key):
        """
        Python built-in hash() is randomized between processes,
        so use md5 to make the per-link seed stable.
        """
        s = f"{self.seed}_{link_key}"
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def _get_rng(self, link_key):
        return np.random.RandomState(self._stable_seed(link_key))

    def _generate_chain_until(self, link_key, frame_idx):
        frame_idx = int(frame_idx)

        if link_key in self.cache and len(self.cache[link_key]) > frame_idx:
            return

        rng = self._get_rng(link_key)

        if link_key not in self.cache:
            chain = [self.initial_state]
        else:
            chain = list(self.cache[link_key])

        # Important:
        # To make the result deterministic even if frame_idx is queried out of order,
        # regenerate from the beginning with the link-specific RNG.
        chain = [self.initial_state]

        while len(chain) <= frame_idx:
            prev_state = chain[-1]
            prev_idx = self.states.index(prev_state)
            next_state = rng.choice(self.states, p=self.P[prev_idx])
            chain.append(str(next_state))

        self.cache[link_key] = chain

    def get_state(self, link_key, frame_idx):
        link_key = str(link_key)
        frame_idx = int(frame_idx)

        self._generate_chain_until(link_key, frame_idx)
        return self.cache[link_key][frame_idx]

    def get_state_id(self, link_key, frame_idx):
        state = self.get_state(link_key, frame_idx)
        return STATE_TO_ID[state]

    def get_delay_ms(self, link_key, frame_idx):
        state = self.get_state(link_key, frame_idx)
        return float(self.delay_ms[state])

    def get_delay_slots(self, link_key, frame_idx):
        delay_ms = self.get_delay_ms(link_key, frame_idx)
        return int(delay_ms // self.frame_interval_ms)

    def get_info(self, link_key, frame_idx):
        state = self.get_state(link_key, frame_idx)
        delay_ms = float(self.delay_ms[state])
        delay_slots = int(delay_ms // self.frame_interval_ms)

        return {
            "channel_state": state,
            "channel_state_id": STATE_TO_ID[state],
            "delay_ms": delay_ms,
            "delay_slots": delay_slots,
        }