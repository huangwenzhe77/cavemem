"""Read the single-arm inputs for card 10; one row per actual randomized decision."""

import math
from pathlib import Path

from .formal import write_rows
from .gate import lines, signed
from .records import digest, save
from .replay import require


SCHEMA = "cave-online-logged-policy/v1"


def validate_rows(logged, episodes):
    by_id = {e["episode_id"]: e for e in episodes}
    require(len(episodes) == len(by_id), "duplicate episode")
    decisions = {}
    for row in logged:
        eid = row["episode_id"]
        require(eid in by_id and eid not in decisions, "unknown or duplicate randomized episode")
        prob, action = row["b"], row["z"]
        require(prob == {"silent": .5, "remind": .5} and type(action) is int and action in (0, 1)
                and row["p_logged"] == prob[("silent", "remind")[action]]
                and row["q_position"] == .25, "invalid assignment or schedule probability")
        require(row["cluster_id"] == by_id[eid]["cluster_id"]
                and row["decision_id"] == by_id[eid]["decision_id"]
                and row["arm"] == by_id[eid]["assigned_arm"] == ("silent", "remind")[action]
                and digest(row["x"]) == row["features_digest"], "broken decision lineage")
        require("D" not in row and "H" not in row and row["y"] in (None, 0, 1),
                "single-arm outcomes must be observed")
        require(row["r"] is None or (row["y"] is not None and math.isfinite(row["r"])),
                "missing observed reward components")
        decisions[eid] = row["decision_id"]
    require(len(set(decisions.values())) == len(decisions) and
            set(decisions) == {e["episode_id"] for e in episodes if e["assigned_arm"] is not None},
            "incomplete randomized denominator")


def load_logged_dataset(directory):
    """Return (assigned rows, full episode denominator, metadata), preserving unscored assignments."""
    directory = Path(directory)
    release = signed(directory / "dataset.json", "dataset_id")
    require(release["schema"] == SCHEMA, "expected an online randomized release")
    logged = lines(directory / "logged-policy.jsonl")
    episodes = lines(directory / "episodes.jsonl")
    require((len(logged), len(episodes)) == (release["assigned"], release["episodes"])
            and digest(logged) == release["logged_digest"] and digest(episodes) == release["episodes_digest"],
            "release content mismatch")
    validate_rows(logged, episodes)
    return logged, episodes, release


def write_dataset(directory, logged, episodes, meta):
    validate_rows(logged, episodes)
    write_rows(directory / "logged-policy.jsonl", logged)
    write_rows(directory / "episodes.jsonl", episodes)
    release = {"schema": SCHEMA, **meta, "episodes": len(episodes), "assigned": len(logged),
               "logged_digest": digest(logged), "episodes_digest": digest(episodes)}
    save(directory / "dataset.json", {**release, "dataset_id": digest(release)})
    load_logged_dataset(directory)
