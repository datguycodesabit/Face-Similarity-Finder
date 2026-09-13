from __future__ import annotations

from dataclasses import dataclass

CATALOG_VERSION = "haircut-catalog-v1"
VALID_PREFERENCES = {
    "length": {"any", "short", "medium", "long"},
    "texture": {"unsure", "straight", "wavy", "curly", "coily"},
    "effort": {"low", "moderate", "high"},
    "goal": {"none", "add-height", "add-width", "soften-angles", "emphasize-angles"},
}


@dataclass(frozen=True)
class Haircut:
    name: str
    lengths: frozenset[str]
    textures: frozenset[str]
    effort: str
    goals: frozenset[str]
    shapes: dict[str, float]
    principle: str
    caution: str


CATALOG = (
    Haircut(
        "Textured crop",
        frozenset({"short"}),
        frozenset({"straight", "wavy", "curly", "coily"}),
        "low",
        frozenset({"add-height", "soften-angles"}),
        {"round": 1.0, "oval": 0.9, "square": 0.8},
        "Texture and a little height keep the outline visually balanced.",
        "Avoid a heavy horizontal fringe if you do not want the face to appear wider.",
    ),
    Haircut(
        "Tapered volume",
        frozenset({"short", "medium"}),
        frozenset({"straight", "wavy", "curly", "coily"}),
        "moderate",
        frozenset({"add-height", "emphasize-angles"}),
        {"round": 1.0, "triangle": 0.9, "oval": 0.8},
        "Tighter sides with controlled height lengthen the visible silhouette.",
        "Keep the taper gradual when the temples are already narrow.",
    ),
    Haircut(
        "Side-parted layers",
        frozenset({"medium", "long"}),
        frozenset({"straight", "wavy", "curly"}),
        "moderate",
        frozenset({"add-width", "soften-angles"}),
        {"oblong": 1.0, "square": 0.9, "heart": 0.8},
        "An offset part and lateral movement interrupt strong vertical or angular lines.",
        "Avoid excessive crown height on an already long face.",
    ),
    Haircut(
        "Layered bob or lob",
        frozenset({"medium"}),
        frozenset({"straight", "wavy", "curly"}),
        "moderate",
        frozenset({"add-width", "soften-angles"}),
        {"oblong": 1.0, "heart": 0.9, "diamond": 0.9, "oval": 0.8},
        "Layers can place fullness near the jaw or cheek area where it is most useful.",
        "Choose the hemline deliberately; a blunt line can emphasize jaw width.",
    ),
    Haircut(
        "Curtain fringe with layers",
        frozenset({"medium", "long"}),
        frozenset({"straight", "wavy", "curly"}),
        "high",
        frozenset({"add-width", "soften-angles"}),
        {"oblong": 1.0, "heart": 0.9, "diamond": 0.8},
        "A split fringe frames the forehead without hiding the whole face.",
        "Fringe behavior depends strongly on growth pattern and daily styling.",
    ),
    Haircut(
        "Soft shag",
        frozenset({"medium", "long"}),
        frozenset({"straight", "wavy", "curly", "coily"}),
        "moderate",
        frozenset({"add-width", "soften-angles"}),
        {"square": 1.0, "oblong": 0.9, "oval": 0.8},
        "Distributed texture softens a strong jaw and adds movement around the sides.",
        "Too much short volume at the crown can exaggerate face length.",
    ),
    Haircut(
        "Long face-framing layers",
        frozenset({"long"}),
        frozenset({"straight", "wavy", "curly", "coily"}),
        "low",
        frozenset({"soften-angles", "add-width"}),
        {"square": 1.0, "triangle": 0.9, "diamond": 0.9, "oval": 0.8},
        "The first layer can be placed to balance the cheekbone or jaw line.",
        "Very flat, unbroken length can pull the eye downward on an oblong face.",
    ),
    Haircut(
        "Rounded layered shape",
        frozenset({"short", "medium", "long"}),
        frozenset({"curly", "coily", "wavy"}),
        "low",
        frozenset({"add-width", "soften-angles"}),
        {"oblong": 1.0, "heart": 0.9, "square": 0.8},
        "Natural texture can build controlled width and soften the outer contour.",
        "Ask for shrinkage-aware length and avoid removing all weight at the sides.",
    ),
)


def validate_preferences(preferences: dict[str, str]) -> dict[str, str]:
    normalized = {
        "length": preferences.get("length", "any"),
        "texture": preferences.get("texture", "unsure"),
        "effort": preferences.get("effort", "moderate"),
        "goal": preferences.get("goal", "none"),
    }
    for key, value in normalized.items():
        if value not in VALID_PREFERENCES[key]:
            raise ValueError(f"Unsupported {key} preference: {value}")
    return normalized


def recommend_haircuts(
    memberships: list[tuple[str, float]], preferences: dict[str, str]
) -> list[dict[str, str | float]]:
    selected = validate_preferences(preferences)
    shape_scores = dict(memberships[:2])
    ranked: list[tuple[float, Haircut]] = []
    for haircut in CATALOG:
        score = sum(
            weight * haircut.shapes.get(shape, 0.3) for shape, weight in shape_scores.items()
        )
        if selected["length"] != "any":
            score += 0.8 if selected["length"] in haircut.lengths else -1.0
        if selected["texture"] != "unsure":
            score += 0.5 if selected["texture"] in haircut.textures else -1.0
        score += 0.3 if selected["effort"] == haircut.effort else 0.0
        if selected["goal"] != "none":
            score += 0.5 if selected["goal"] in haircut.goals else 0.0
        ranked.append((score, haircut))
    return [
        {
            "name": haircut.name,
            "fit_score": round(score, 3),
            "rationale": haircut.principle,
            "styling_principle": haircut.principle,
            "caution": haircut.caution,
        }
        for score, haircut in sorted(ranked, key=lambda item: (-item[0], item[1].name))[:3]
    ]
