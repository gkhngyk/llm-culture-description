"""Prompt variant 1 — formal."""


def render(role: str) -> str:
    if role == "proposer":
        return (
            "State the offer you are delivering (the integer given above), "
            "predict whether the other participant will accept or decline, "
            "and provide two to three sentences of reasoning."
        )
    return (
        "Decide whether to accept or decline the offer. "
        "Provide two to three sentences of reasoning for your decision."
    )
