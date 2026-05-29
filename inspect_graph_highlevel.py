"""Visualize the high-level OWLv2 -> DINOv2 -> PELT(RBF) -> VLM state/skill graph.

Updated for the current design:

1. PELT/RBF detects rupture boundaries from object-crop + global-frame embeddings + relative spoon-location deltas.
2. State nodes are rupture-defined frame segments between boundaries.
3. Transition windows are compact boundary-centered evidence windows.
4. State nodes display VLM scene descriptions and visible conditions, not predicates.
5. Edges display closed-set skill labels inferred from PRE-to-POST semantic state change, with boundary evidence as verification.
"""

import argparse
import json
from pathlib import Path
from graphviz import Digraph

DEFAULT_GRAPH_PATH = "trajectory_graph_highlevel_boundary.json"
DEFAULT_OUT_PATH = "trajectory_highlevel_state_change_skill_chain"

SKILL_COLORS = {
    "approach": "gray",
    "scoop": "orange",
    "lift": "purple",
    "transport": "blue",
    "dump": "green",
    "return": "brown",
    "none": "black",
}


def trajectory_sort_key(edge):
    tid = str(edge.get("trajectory_id", ""))
    digits = "".join(ch for ch in tid if ch.isdigit())
    return int(digits) if digits else tid


def compact_text(text, max_len=90):
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def format_window(win):
    if not win or len(win) < 2:
        return None
    return f"{win[0]}-{win[1]}"


def condition_text(node, max_len=90):
    conditions = node.get("visible_conditions") or []
    if conditions:
        if isinstance(conditions, list):
            return compact_text(", ".join(str(c) for c in conditions), max_len)
        return compact_text(str(conditions), max_len)
    summary = node.get("state_summary")
    return compact_text(summary, max_len) if summary else ""


def node_label(state_id, abstract_nodes_by_id, show_conditions=True):
    node = abstract_nodes_by_id.get(state_id, {})
    frame_id = node.get("representative_frame_id")
    role = node.get("representative_role")
    state_name = node.get("state_name")
    summary = node.get("state_summary")
    state_window = node.get("state_window")
    image_path = node.get("representative_image_path")

    parts = [state_id]

    if state_name and state_name != state_id:
        parts.append(str(state_name))

    if frame_id is not None:
        parts.append(f"frame {frame_id}")

    if state_window:
        w = format_window(state_window)
        if w:
            parts.append(f"state={w}")

    if role:
        parts.append(str(role))

    if summary:
        parts.append(compact_text(summary, 90))

    if show_conditions:
        cond = condition_text(node, max_len=100)
        if cond:
            parts.append("visible: " + cond)

    if image_path:
        parts.append(Path(image_path).name)

    return "\n".join(parts)


def pipeline_label(graph):
    pipeline = graph.get("pipeline")
    if isinstance(pipeline, list):
        return "Pipeline: " + " → ".join(pipeline)
    detector = graph.get("object_detector", {}).get("model_id", "OWLv2")
    embedding = graph.get("embedding_model", {}).get("model_id", "DINOv2")
    cpd = graph.get("change_point_detector", {}).get("library", "ruptures")
    return f"Pipeline: {detector} → object crops + global frame + relative spoon deltas → {embedding} embeddings/features → aggregate → {cpd} → rupture-defined state segments → VLM state descriptions → PRE-to-POST state change + boundary evidence → VLM skill inference → graph"


def state_text(state_id, abstract_nodes_by_id, max_len=48):
    node = abstract_nodes_by_id.get(state_id, {})
    summary = node.get("state_summary")
    if summary:
        return compact_text(summary, max_len)
    cond = condition_text(node, max_len=max_len)
    return cond if cond else state_id


def build_graphviz(graph, show_raw_ids=False, show_orphans=False, show_conditions=True, show_reasons=False):
    dot = Digraph(format="png")
    dot.attr(rankdir="LR", nodesep="0.9", ranksep="1.2", splines="true")
    dot.attr(label=compact_text(pipeline_label(graph), 260), labelloc="t", fontsize="14")
    dot.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue", fontsize="9")
    dot.attr("edge", fontsize="9", penwidth="2")

    abstract_nodes = graph.get("abstract_nodes", [])
    abstract_nodes_by_id = {
        node.get("abstract_node_id"): node
        for node in abstract_nodes
        if node.get("abstract_node_id") is not None
    }

    edges = sorted(graph.get("edges", []), key=trajectory_sort_key)

    used_state_ids = set()
    for e in edges:
        if e.get("source") is not None:
            used_state_ids.add(e["source"])
        if e.get("target") is not None:
            used_state_ids.add(e["target"])

    state_ids_to_draw = set(abstract_nodes_by_id.keys()) if show_orphans else used_state_ids

    for state_id in sorted(state_ids_to_draw):
        dot.node(
            state_id,
            label=node_label(state_id, abstract_nodes_by_id, show_conditions=show_conditions),
        )

    for e in edges:
        source = e.get("source")
        target = e.get("target")
        if source is None or target is None:
            continue

        if source not in state_ids_to_draw:
            dot.node(source, label=source)
        if target not in state_ids_to_draw:
            dot.node(target, label=target)

        skill = e.get("skill_label", "unknown")
        conf = e.get("confidence")
        tid = e.get("trajectory_id", "")
        transition_window = e.get("transition_window")
        reason = e.get("reason")

        src_text = state_text(source, abstract_nodes_by_id, max_len=60)
        dst_text = state_text(target, abstract_nodes_by_id, max_len=60)

        pre_summary = e.get("source_state_summary")
        post_summary = e.get("target_state_summary")

        boundary = e.get("boundary_frame")
        edge_label = f"{tid}"

        if boundary is not None:
            edge_label += f"\nboundary={boundary}"

        if transition_window:
            w = format_window(transition_window)
            if w:
                edge_label += f"\nevidence={w}"

        if pre_summary:
            edge_label += "\nPRE:"
            edge_label += "\n" + compact_text(pre_summary, 90)
        else:
            edge_label += "\nPRE:"
            edge_label += "\n" + src_text

        edge_label += f"\n→ {skill} →"

        if post_summary:
            edge_label += "\nPOST:"
            edge_label += "\n" + compact_text(post_summary, 90)
        else:
            edge_label += "\nPOST:"
            edge_label += "\n" + dst_text

        if conf is not None:
            try:
                edge_label += f"\nconf={float(conf):.2f}"
            except Exception:
                pass

        if show_reasons and reason:
            edge_label += "\n" + compact_text(reason, 120)

        expected = e.get("expected_skill_label")
        color = SKILL_COLORS.get(skill, "black")
        if expected is not None and expected != skill:
            edge_label += f"\nexpected: {expected}"
            color = "red"

        if show_raw_ids:
            raw_source = e.get("raw_source")
            raw_target = e.get("raw_target")
            if raw_source or raw_target:
                edge_label += f"\n{raw_source} → {raw_target}"

        dot.edge(source, target, label=edge_label, color=color, fontcolor=color)

    with dot.subgraph(name="cluster_legend") as legend:
        legend.attr(label="Skill colors", fontsize="10", color="lightgray")
        for skill, color in SKILL_COLORS.items():
            legend.node(f"legend_{skill}", label=skill, shape="plaintext", fontcolor=color)

    return dot


def print_summary(graph):
    print("\n=== HIGH-LEVEL PIPELINE ===\n")
    print(pipeline_label(graph))
    print("\nsegmentation_method:", graph.get("segmentation_method"))
    print("embedding_model:", graph.get("embedding_model", {}).get("model_id"))
    print("change_point_detector:", graph.get("change_point_detector", {}))
    print("boundaries:", graph.get("boundaries"))

    embedding = graph.get("embedding_model", {})
    if embedding:
        print("object_streams:", embedding.get("object_streams"))
        print("uses_global_frame_embedding:", embedding.get("uses_global_frame_embedding"))
        print("uses_relative_spatial_deltas:", embedding.get("uses_relative_spatial_deltas"))
        print("spatial_delta_features:", embedding.get("spatial_delta_features"))
        print("aggregation:", embedding.get("aggregation"))
        bd = graph.get("boundary_debug", {})
        if bd:
            print("debug_embedding_streams:", bd.get("embedding_streams") or bd.get("object_streams"))
            print("debug_uses_global_frame_embedding:", bd.get("uses_global_frame_embedding"))
            print("debug_uses_relative_spatial_deltas:", bd.get("uses_relative_spatial_deltas"))
            print("debug_spatial_delta_features:", bd.get("spatial_delta_features"))

    print("\n=== STATE SUMMARY ===\n")
    for node in graph.get("abstract_nodes", []):
        sid = node.get("abstract_node_id")
        win = node.get("state_window")
        frame = node.get("representative_frame_id")
        print(f"{sid}: window={win}, frame={frame}")
        if node.get("state_summary"):
            print("  summary:", compact_text(node.get("state_summary"), 160))
        if node.get("visible_conditions"):
            print("  visible:", compact_text(", ".join(map(str, node.get("visible_conditions"))), 180))

    print("\n=== EDGE SUMMARY ===\n")
    abstract_nodes_by_id = {
        node.get("abstract_node_id"): node
        for node in graph.get("abstract_nodes", [])
        if node.get("abstract_node_id") is not None
    }

    for e in sorted(graph.get("edges", []), key=trajectory_sort_key):
        conf = e.get("confidence", 0.0)
        try:
            conf_print = round(float(conf), 3) if conf is not None else None
        except Exception:
            conf_print = conf

        src = e.get("source")
        dst = e.get("target")
        src_state = state_text(src, abstract_nodes_by_id, max_len=90)
        dst_state = state_text(dst, abstract_nodes_by_id, max_len=90)

        print(
            e.get("trajectory_id"),
            f"boundary={e.get('boundary_frame')}, evidence={e.get('transition_window')}",
            src,
            "--",
            e.get("skill_label"),
            "-->",
            dst,
            "conf:",
            conf_print,
        )
        print("  pre:", src_state)
        print("  skill:", e.get("skill_label"))
        print("  post:", dst_state)
        if e.get("reason"):
            print("  reason:", compact_text(e.get("reason"), 160))

    print("\n=== STATS ===\n")
    print("num raw nodes:", len(graph.get("raw_nodes", [])))
    print("num abstract states:", len(graph.get("abstract_nodes", [])))
    print("num edges:", len(graph.get("edges", [])))


def main():
    parser = argparse.ArgumentParser(description="Generate a horizontal graph from rupture-segment state-change skill JSON with object/global/spatial-delta embedding metadata.")
    parser.add_argument("--graph-path", default=None, help="Path to graph JSON. Defaults to script folder / trajectory_graph_highlevel_boundary.json.")
    parser.add_argument("--out-path", default=DEFAULT_OUT_PATH, help="Output path without extension.")
    parser.add_argument("--show-raw-ids", action="store_true", help="Show raw node ids on each edge when available.")
    parser.add_argument("--show-orphans", action="store_true", help="Draw disconnected debug nodes too.")
    parser.add_argument("--hide-conditions", action="store_true", help="Hide visible_conditions in node labels.")
    parser.add_argument("--show-reasons", action="store_true", help="Show VLM skill reasons on edges.")
    parser.add_argument("--no-view", action="store_true", help="Save PNG without opening the viewer.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    graph_path = Path(args.graph_path) if args.graph_path else base_dir / DEFAULT_GRAPH_PATH
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Could not find JSON graph file: {graph_path}\n"
            f"Pass it explicitly, e.g. python inspect_graph_highlevel_state_change_skill.py --graph-path /path/to/trajectory_graph_highlevel_boundary.json"
        )

    with open(graph_path, "r") as f:
        graph = json.load(f)

    if "edges" not in graph:
        raise KeyError("This JSON does not contain graph['edges'].")

    print_summary(graph)
    dot = build_graphviz(
        graph,
        show_raw_ids=args.show_raw_ids,
        show_orphans=args.show_orphans,
        show_conditions=not args.hide_conditions,
        show_reasons=args.show_reasons,
    )
    output_path = dot.render(args.out_path, view=not args.no_view, cleanup=True)
    print(f"\nSaved graph to: {output_path}")


if __name__ == "__main__":
    main()
