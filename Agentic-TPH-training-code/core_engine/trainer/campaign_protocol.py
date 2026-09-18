"""Identify the split rules used to prepare a training workspace."""

SPLIT_PROTOCOL = "potential-endpoint-pair-v1"


def require_current_split(metadata):
    if metadata.get("split_protocol") != SPLIT_PROTOCOL:
        raise ValueError(
            "Legacy or missing split protocol: corrected workspaces must be "
            "prepared explicitly before running; do not relabel old inputs."
        )
