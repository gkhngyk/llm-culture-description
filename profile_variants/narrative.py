"""Profile variant 2 — narrative paragraph."""


def render(profile) -> str:
    return (
        f"A {profile.age}-year-old whose daily life is organized around "
        f"{profile.occupation_or_livelihood}. Their economic situation is best "
        f"described as: {profile.economic_security}. Socially, they live with "
        f"{profile.social_ties_density}. When it comes to exchange and provisioning, "
        f"their experience is rooted in {profile.primary_economy_experience}. "
        f"Their trust in formal institutions is {profile.institutional_trust}."
    )
