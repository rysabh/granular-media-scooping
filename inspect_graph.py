# import json

# GRAPH_PATH = "trajectory_graph_compacted.json"

# def main():
#     with open(GRAPH_PATH) as f:
#         graph = json.load(f)

#     print("\n=== EDGE SUMMARY ===\n")

#     for e in graph["edges"]:
#         print(
#             e["trajectory_id"],
#             e["source"],
#             "--",
#             e["skill_label"],
#             "(expected:", e.get("expected_skill_label"), ")",
#             "-->",
#             e["target"],
#             "conf:",
#             round(e["confidence"], 3),
#         )

#     print("\n=== STATS ===\n")
#     print("num raw nodes:", len(graph["raw_nodes"]))
#     print("num abstract states:", len(graph["abstract_nodes"]))
#     print("num edges:", len(graph["edges"]))
import json
from graphviz import Digraph

from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
GRAPH_PATH = BASE_DIR / "trajectory_graph_compacted.json"

with open(GRAPH_PATH) as f:
    data = json.load(f)
OUT_PATH = "trajectory_chain"


def main():
    with open(GRAPH_PATH) as f:
        graph = json.load(f)

    dot = Digraph(format="png")

    # Force strict horizontal layout
    dot.attr(rankdir="LR")
    dot.attr(nodesep="1.2")
    dot.attr(ranksep="1.5")

    dot.attr("node",
             shape="circle",
             style="filled",
             fillcolor="lightblue")

    edges = graph["transitions"]

    # IMPORTANT: sort edges by trajectory order
    edges = sorted(edges, key=lambda x: x["trajectory_id"])

    # Build chain
    for i, e in enumerate(edges):
        s = e["source_state"]
        t = e["target_state"]
        label = e["skill_label"]
        conf = round(e["confidence"], 2)
        expected = e.get("expected_skill_label")

        label = f"{label} ({conf})"

        if expected and expected != label:
            label += f"\nexp:{expected}"
            color = "red"
        else:
            color = "black"

        dot.edge(s, t, label=label, color=color)

    dot.render(OUT_PATH, view=True)
    print("Saved horizontal chain graph")


if __name__ == "__main__":
    main()

