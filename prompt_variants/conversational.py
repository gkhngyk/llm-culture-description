"""Prompt variant 2 — conversational."""


def render(role: str) -> str:
    if role == "proposer":
        return (
            "So — the offer you're handing over is the number above. "
            "What's your read — will the other person go for it or walk away? "
            "Give me two or three sentences on why."
        )
    return (
        "So — what's your call? Take it or leave it? "
        "Tell me in two or three sentences why."
    )
