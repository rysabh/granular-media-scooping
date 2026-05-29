import json
import subprocess
from pathlib import Path
from clingo import Control

GRAPH_PATH = "trajectory_graph_with_predicates.json"
ASP_FILE = "domain_inference.lp"
MODEL_FILE = "asp_model.txt"

DOMAIN_FILE = "domain.pddl"
PROBLEM_FILE = "problem.pddl"


def sanitize(x):
    return x.replace("-", "_")


def write_asp_facts(graph):
    lines = []

    # states
    for node in graph["abstract_nodes"]:
        s = sanitize(node["abstract_node_id"])
        lines.append(f"state({s}).")

    # predicates from reason_to_predicates.py
    state_predicates = graph.get("state_predicates", {})

    all_preds = set()
    for state_id, preds in state_predicates.items():
        s = sanitize(state_id)
        for p in preds:
            p = sanitize(p)
            all_preds.add(p)
            lines.append(f"pred({p}).")
            lines.append(f"holds({s},{p}).")

    # transitions
    for e in graph["edges"]:
        src = sanitize(e["source"])
        tgt = sanitize(e["target"])

        # use expected label first for now, because VLM labels may still be noisy
        skill = e.get("expected_skill_label") or e["skill_label"]
        skill = sanitize(skill)

        lines.append(f"action({skill}).")
        lines.append(f"transition({src},{skill},{tgt}).")

    return "\n".join(lines)

ASP_RULES = r"""
% Preconditions: predicates true in source state
pre(A, P) :- transition(S, A, _), holds(S, P).

% Add effects: predicates true in target state but not source state
add(A, P) :- transition(S, A, T), holds(T, P), not holds(S, P).

% Delete effects: predicates true in source state but not target state
del(A, P) :- transition(S, A, T), holds(S, P), not holds(T, P).

#show action/1.
#show pred/1.
#show pre/2.
#show add/2.
#show del/2.
"""


def run_clingo():
    ctl = Control(["1"])
    ctl.load(ASP_FILE)
    ctl.ground([("base", [])])

    models = []

    def on_model(model):
        models.append(model.symbols(shown=True))

    result = ctl.solve(on_model=on_model)

    if not result.satisfiable:
        raise RuntimeError("ASP problem is unsatisfiable")

    if not models:
        raise RuntimeError("No ASP model found")

    atoms = [str(atom) for atom in models[0]]
    output = " ".join(atoms)

    Path(MODEL_FILE).write_text(output)
    return output

def parse_atoms(output):
    atoms = []

    for line in output.splitlines():
        if line.startswith("Answer:"):
            continue
        if line.startswith("SATISFIABLE"):
            continue
        if "(" in line and ")" in line:
            atoms.extend(line.strip().split())

    return atoms


def atom_args(atom):
    name = atom[:atom.index("(")]
    inside = atom[atom.index("(") + 1: atom.rindex(")")]
    args = inside.split(",")
    return name, args


def write_pddl(atoms, graph):
    actions = set()
    preds = set()
    pres = {}
    adds = {}
    dels = {}

    for atom in atoms:
        name, args = atom_args(atom)

        if name == "action":
            actions.add(args[0])
        elif name == "pred":
            preds.add(args[0])
        elif name == "pre":
            a, p = args
            pres.setdefault(a, set()).add(p)
        elif name == "add":
            a, p = args
            adds.setdefault(a, set()).add(p)
        elif name == "del":
            a, p = args
            dels.setdefault(a, set()).add(p)

    with open(DOMAIN_FILE, "w") as f:
        f.write("(define (domain scooping)\n")
        f.write("  (:predicates\n")
        for p in sorted(preds):
            f.write(f"    ({p})\n")
        f.write("  )\n\n")

        for a in sorted(actions):
            f.write(f"  (:action {a}\n")

            pre_list = sorted(pres.get(a, []))
            if pre_list:
                f.write("    :precondition (and\n")
                for p in pre_list:
                    f.write(f"      ({p})\n")
                f.write("    )\n")
            else:
                f.write("    :precondition ()\n")

            f.write("    :effect (and\n")
            for p in sorted(dels.get(a, [])):
                f.write(f"      (not ({p}))\n")
            for p in sorted(adds.get(a, [])):
                f.write(f"      ({p})\n")
            f.write("    )\n")
            f.write("  )\n\n")

        f.write(")\n")

    edges = graph["edges"]
    init_state = sanitize(edges[0]["source"])
    goal_state = sanitize(edges[-1]["target"])

    with open(PROBLEM_FILE, "w") as f:
        f.write("(define (problem scooping-task)\n")
        f.write("  (:domain scooping)\n")
        f.write(f"  (:init ({init_state}))\n")
        f.write(f"  (:goal ({goal_state}))\n")
        f.write(")\n")


def main():
    with open(GRAPH_PATH) as f:
        graph = json.load(f)

    facts = write_asp_facts(graph)
    Path(ASP_FILE).write_text(facts + "\n\n" + ASP_RULES)

    print(f"Wrote ASP program to {ASP_FILE}")

    output = run_clingo()
    atoms = parse_atoms(output)

    write_pddl(atoms, graph)

    print(f"Generated {DOMAIN_FILE}")
    print(f"Generated {PROBLEM_FILE}")
    print(f"Saved ASP output to {MODEL_FILE}")


if __name__ == "__main__":
    main()