import argparse
import json
from pathlib import Path
from graphviz import Digraph


DEFAULT_GRAPH_PATH = "trajectory_graph_compacted.json"
DEFAULT_OUT_PATH = "trajectory_chain"


def trajectory_sort_key(edge):
    """Sort tau_0001-style ids numerically when possible."""
    tid = str(edge.get("trajectory_id", ""))
    digits = "".join(ch for ch in tid if ch.isdigit())
    if digits:
        return int(digits)
    return tid


def node_label(state_id, abstract_nodes_by_id):
    """Create a readable node label using abstract-state metadata if available."""
    node = abstract_nodes_by_id.get(state_id, {})
    frame_id = node.get("representative_frame_id")
    role = node.get("representative_role")

    label_parts = [state_id]
    if frame_id is not None:
        label_parts.append(f"frame {frame_id}")
    if role:
        label_parts.append(role)
    return "\n".join(label_parts)


def build_graphviz(graph, show_raw_ids=False, show_orphans=False):
    """
    Build a clean left-to-right trajectory-chain graph.

    Important: the JSON may contain debug abstract nodes for every raw pre/post
    snapshot. Some pre-state nodes become unused after temporal chaining because
    the generator intentionally rewires each segment's pre-state to the previous
    segment's post-state. Therefore, by default we draw ONLY states that are
    actually referenced by edges. This removes the vertical floating orphan nodes.
    """
    dot = Digraph(format="png")

    dot.attr(rankdir="LR")
    dot.attr(nodesep="0.9")
    dot.attr(ranksep="1.2")
    dot.attr(splines="true")

    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="lightblue",
        fontsize="10",
    )
    dot.attr("edge", fontsize="10")

    abstract_nodes = graph.get("abstract_nodes", [])
    abstract_nodes_by_id = {
        node.get("abstract_node_id"): node
        for node in abstract_nodes
        if node.get("abstract_node_id") is not None
    }

    edges = sorted(graph.get("edges", []), key=trajectory_sort_key)

    # Only draw states that participate in the transition graph.
    used_state_ids = set()
    for e in edges:
        if e.get("source") is not None:
            used_state_ids.add(e["source"])
        if e.get("target") is not None:
            used_state_ids.add(e["target"])

    # Optional debugging mode: draw all abstract nodes, including disconnected ones.
    state_ids_to_draw = set(abstract_nodes_by_id.keys()) if show_orphans else used_state_ids

    for state_id in sorted(state_ids_to_draw):
        label = node_label(state_id, abstract_nodes_by_id)
        node = abstract_nodes_by_id.get(state_id, {})
        image_path = node.get("representative_image_path")
        if image_path:
            label += f"\n{Path(image_path).name}"
        dot.node(state_id, label=label)

    # Add edges with skill labels.
    for e in edges:
        source = e.get("source")
        target = e.get("target")
        if source is None or target is None:
            continue

        # In case an edge references a state not listed in abstract_nodes.
        if source not in state_ids_to_draw:
            dot.node(source, label=source)
        if target not in state_ids_to_draw:
            dot.node(target, label=target)

        skill = e.get("skill_label", "unknown")
        confidence = e.get("confidence", None)
        trajectory_id = e.get("trajectory_id", "")
        expected = e.get("expected_skill_label", None)

        if confidence is None:
            edge_label = f"{trajectory_id}\n{skill}"
        else:
            edge_label = f"{trajectory_id}\n{skill} ({confidence:.2f})"

        if expected is not None and expected != skill:
            edge_label += f"\nexpected: {expected}"
            color = "red"
        else:
            color = "black"

        if show_raw_ids:
            raw_source = e.get("raw_source")
            raw_target = e.get("raw_target")
            if raw_source or raw_target:
                edge_label += f"\n{raw_source} → {raw_target}"

        dot.edge(source, target, label=edge_label, color=color)

    return dot

def print_summary(graph):
    print("\n=== EDGE SUMMARY ===\n")
    for e in sorted(graph.get("edges", []), key=trajectory_sort_key):
        conf = e.get("confidence", 0.0)
        print(
            e.get("trajectory_id"),
            e.get("source"),
            "--",
            e.get("skill_label"),
            "-->",
            e.get("target"),
            "conf:",
            round(float(conf), 3),
        )

    print("\n=== STATS ===\n")
    print("num raw nodes:", len(graph.get("raw_nodes", [])))
    print("num abstract states:", len(graph.get("abstract_nodes", [])))
    print("num edges:", len(graph.get("edges", [])))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a horizontal state-transition graph from trajectory_graph_compacted.json."
    )
    parser.add_argument(
        "--graph-path",
        default=None,
        help="Path to trajectory_graph_compacted.json. Defaults to script folder / trajectory_graph_compacted.json.",
    )
    parser.add_argument(
        "--out-path",
        default=DEFAULT_OUT_PATH,
        help="Output path without extension. Default: trajectory_chain",
    )
    parser.add_argument(
        "--show-raw-ids",
        action="store_true",
        help="Show raw pre/post node ids on each edge.",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Save PNG without opening the viewer.",
    )
    parser.add_argument(
        "--show-orphans",
        action="store_true",
        help="Also draw disconnected debug nodes. By default, only edge-referenced states are shown.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    graph_path = Path(args.graph_path) if args.graph_path else base_dir / DEFAULT_GRAPH_PATH

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Could not find JSON graph file: {graph_path}\n"
            f"Pass the file explicitly, for example:\n"
            f"python inspect_graph_revised.py --graph-path /path/to/trajectory_graph_compacted.json"
        )

    with open(graph_path, "r") as f:
        graph = json.load(f)

    if "edges" not in graph:
        raise KeyError(
            "This JSON does not contain graph['edges']. "
            "Check whether your generator saved a different schema."
        )

    print_summary(graph)

    dot = build_graphviz(graph, show_raw_ids=args.show_raw_ids, show_orphans=args.show_orphans)
    output_path = dot.render(args.out_path, view=not args.no_view, cleanup=True)
    print(f"\nSaved graph to: {output_path}")


if __name__ == "__main__":
    main()
