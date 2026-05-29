import json
import re
from pathlib import Path

GRAPH_PATH = "trajectory_graph_compacted.json"
OUT_PATH = "trajectory_graph_with_predicates.json"


PREDICATE_RULES = {
    "tool_near_pile": [
        "towards the material pile",
        "toward the material pile",
        "near the material pile",
        "approaching",
    ],
    "tool_in_pile": [
        "enters the material pile",
        "entering the material pile",
        "inside the pile",
        "through the material",
    ],
    "tool_contains_material": [
        "collects material",
        "collected material",
        "scooped some material",
        "contains material",
    ],
    "tool_lifted": [
        "moving upward",
        "moves upward",
        "lift",
        "away from the material pile",
    ],
    "tool_at_dump": [
        "dump zone",
        "transport",
        "toward another location",
        "horizontally",
    ],
    "material_deposited": [
        "dump",
        "releasing",
        "deposited",
        "falls",
        "falling",
        "material leaves",
    ],
    "no_skill": [
        "stationary",
        "no dominant skill",
        "none",
    ],
}


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def infer_predicates_from_reason(reason):
    reason = normalize(reason)
    preds = []

    for pred, phrases in PREDICATE_RULES.items():
        for phrase in phrases:
            if phrase in reason:
                preds.append(pred)
                break

    return sorted(set(preds))


def main():
    with open(GRAPH_PATH) as f:
        graph = json.load(f)

    state_predicates = {
        node["abstract_node_id"]: set()
        for node in graph["abstract_nodes"]
    }

    for e in graph["edges"]:
        source = e["source"]
        target = e["target"]

        skill = e.get("skill_label", "none")
        expected = e.get("expected_skill_label")

        parsed = e.get("parsed_vlm_output") or {}
        reason = parsed.get("reason", "")

        inferred = infer_predicates_from_reason(reason)

        # Use skill structure as backup when VLM reason is weak
        label = expected or skill

        if label == "approach":
            state_predicates[source].add("tool_away")
            state_predicates[target].add("tool_near_pile")

        elif label == "scoop":
            state_predicates[source].add("tool_near_pile")
            state_predicates[target].add("tool_contains_material")

        elif label == "lift":
            state_predicates[source].add("tool_contains_material")
            state_predicates[target].add("tool_lifted")

        elif label == "transport":
            state_predicates[source].add("tool_lifted")
            state_predicates[target].add("tool_at_dump")

        elif label == "dump":
            state_predicates[source].add("tool_at_dump")
            state_predicates[target].add("material_deposited")

        # Also attach predicates inferred from VLM reason
        for p in inferred:
            state_predicates[target].add(p)

        e["reason_inferred_predicates"] = inferred

    graph["state_predicates"] = {
        s: sorted(list(preds))
        for s, preds in state_predicates.items()
    }

    Path(OUT_PATH).write_text(json.dumps(graph, indent=2))
    print(f"Saved {OUT_PATH}")

    print("\n=== STATE PREDICATES ===")
    for s, preds in graph["state_predicates"].items():
        if preds:
            print(s, ":", preds)


if __name__ == "__main__":
    main()