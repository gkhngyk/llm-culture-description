"""Profile variant 1 — structured metric list."""


def render(profile) -> str:
    return (
        f"age: {profile.age}\n"
        f"role in life: {profile.occupation_or_livelihood}\n"
        f"economic security: {profile.economic_security}\n"
        f"social ties density: {profile.social_ties_density}\n"
        f"primary economy experience: {profile.primary_economy_experience}\n"
        f"institutional trust: {profile.institutional_trust}\n"
    )
