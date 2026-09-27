"""Profile variant 3 — first-person self-description."""


def render(profile) -> str:
    return (
        f"I am {profile.age}. My days revolve around {profile.occupation_or_livelihood}. "
        f"When I think about where I stand financially, the honest answer is: "
        f"{profile.economic_security}. The people around me — that is "
        f"{profile.social_ties_density}. When I think about how things get "
        f"exchanged and provided in my world, what I know best is "
        f"{profile.primary_economy_experience}. My trust in formal institutions "
        f"is {profile.institutional_trust}."
    )
