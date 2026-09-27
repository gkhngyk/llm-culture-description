"""Prompt variant 3 — scenario-as-vignette."""


def render(role: str) -> str:
    if role == "proposer":
        return (
            "Picture the moment: the offer has just been written down, exactly as "
            "stated above. You imagine the other person reading it. "
            "Walk me through what you think happens next — accept or decline — "
            "and say in two or three sentences why it plays out that way."
        )
    return (
        "Picture the moment: you are looking at the offer, exactly as stated "
        "above. You have to choose. Walk me through what happens next — "
        "accept or decline — in two or three sentences."
    )
