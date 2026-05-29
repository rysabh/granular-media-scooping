from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, FrozenSet, Optional
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import networkx as nx


# ============================================================
# 1. DATA STRUCTURES
# ============================================================

Predicate = Tuple[str, ...]


@dataclass
class DemoSample:
    """
    One synchronized multimodal observation at time t.

    In a real system:
    - video_clip: short clip or frame tensor
    - audio_feat: mel spectrogram or waveform features
    - labels: optional supervision for training
    """
    demo_id: str
    t: int
    video_clip: torch.Tensor      # shape: [C, T, H, W] or [C, H, W]
    audio_feat: torch.Tensor      # shape: [1, F, A] e.g. mel spectrogram
    labels: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class SymbolicState:
    predicates: FrozenSet[Predicate]

    def pretty(self) -> List[str]:
        return sorted(["(" + ", ".join(p) + ")" for p in self.predicates])


@dataclass
class Transition:
    demo_id: str
    t: int
    state_before: SymbolicState
    action: Tuple[str, ...]
    state_after: SymbolicState


# ============================================================
# 2. MOCK DATASET
# ============================================================

class MockBrownSugarDataset(Dataset):
    """
    Toy dataset for demonstration.
    Replace with real video/audio loading later.
    """
    def __init__(self, num_demos: int = 5, timesteps_per_demo: int = 12):
        self.samples: List[DemoSample] = []

        for d in range(num_demos):
            demo_id = f"demo_{d:03d}"
            for t in range(timesteps_per_demo):
                # Mock video: [C, T, H, W]
                video = torch.randn(3, 4, 64, 64)

                # Mock audio spectrogram: [1, F, A]
                audio = torch.randn(1, 64, 64)

                # Simple fake phase progression
                if t < 2:
                    phase = 0   # approach
                    contact = 0
                    fill = 0    # empty
                    spill = 0
                elif t < 5:
                    phase = 1   # penetrate_drag
                    contact = 1
                    fill = 1    # low
                    spill = 0
                elif t < 8:
                    phase = 2   # lift_carry
                    contact = 0
                    fill = 2    # medium
                    spill = 0
                else:
                    phase = 3   # pour
                    contact = 0
                    fill = 1 if t < 10 else 0
                    spill = 1 if t == 9 else 0

                labels = {
                    "contact": contact,        # 0/1
                    "phase": phase,            # 0..3
                    "fill_level": fill,        # 0 empty, 1 low, 2 medium, 3 high
                    "spill": spill,            # 0/1
                    "near_pile": 1 if t < 5 else 0,
                    "near_bowl": 1 if t >= 8 else 0,
                }

                self.samples.append(
                    DemoSample(
                        demo_id=demo_id,
                        t=t,
                        video_clip=video,
                        audio_feat=audio,
                        labels=labels
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> DemoSample:
        return self.samples[idx]


def collate_demo_samples(batch: List[DemoSample]):
    video = torch.stack([b.video_clip for b in batch], dim=0)   # [B, C, T, H, W]
    audio = torch.stack([b.audio_feat for b in batch], dim=0)   # [B, 1, F, A]

    demo_ids = [b.demo_id for b in batch]
    times = torch.tensor([b.t for b in batch], dtype=torch.long)

    labels = {}
    if batch[0].labels is not None:
        for key in batch[0].labels.keys():
            labels[key] = torch.tensor([b.labels[key] for b in batch], dtype=torch.long)

    return {
        "demo_ids": demo_ids,
        "times": times,
        "video": video,
        "audio": audio,
        "labels": labels
    }


# ============================================================
# 3. VISION ENCODER
# ============================================================

class VisionEncoder(nn.Module):
    """
    Very small 3D-CNN-style encoder for short clips.
    Input: [B, C, T, H, W]
    Output: [B, D]
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)               # [B, 64, 1, 1, 1]
        h = h.flatten(1)              # [B, 64]
        return self.proj(h)           # [B, out_dim]


# ============================================================
# 4. AUDIO ENCODER
# ============================================================

class AudioEncoder(nn.Module):
    """
    CNN encoder for spectrogram-like audio features.
    Input: [B, 1, F, A]
    Output: [B, D]
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)      # [B, 64, 1, 1]
        h = h.flatten(1)     # [B, 64]
        return self.proj(h)  # [B, out_dim]


# ============================================================
# 5. MULTIMODAL PERCEPTION MODEL
# ============================================================

class MultimodalPerceptionModel(nn.Module):
    """
    Predicts symbolic concepts from video + audio.
    """
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.vision_encoder = VisionEncoder(out_dim=embed_dim)
        self.audio_encoder = AudioEncoder(out_dim=embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Heads for symbolic concept prediction
        self.contact_head = nn.Linear(hidden_dim, 2)      # contact / no-contact
        self.phase_head = nn.Linear(hidden_dim, 4)        # approach, penetrate_drag, lift_carry, pour
        self.fill_head = nn.Linear(hidden_dim, 4)         # empty, low, medium, high
        self.spill_head = nn.Linear(hidden_dim, 2)        # no spill / spill
        self.near_pile_head = nn.Linear(hidden_dim, 2)
        self.near_bowl_head = nn.Linear(hidden_dim, 2)

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        zv = self.vision_encoder(video)
        za = self.audio_encoder(audio)
        z = torch.cat([zv, za], dim=-1)
        h = self.fusion(z)

        return {
            "contact_logits": self.contact_head(h),
            "phase_logits": self.phase_head(h),
            "fill_logits": self.fill_head(h),
            "spill_logits": self.spill_head(h),
            "near_pile_logits": self.near_pile_head(h),
            "near_bowl_logits": self.near_bowl_head(h),
            "embedding": h,
        }


# ============================================================
# 6. LOSSES
# ============================================================

def compute_losses(outputs: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    losses = {}

    losses["contact"] = F.cross_entropy(outputs["contact_logits"], labels["contact"])
    losses["phase"] = F.cross_entropy(outputs["phase_logits"], labels["phase"])
    losses["fill"] = F.cross_entropy(outputs["fill_logits"], labels["fill_level"])
    losses["spill"] = F.cross_entropy(outputs["spill_logits"], labels["spill"])
    losses["near_pile"] = F.cross_entropy(outputs["near_pile_logits"], labels["near_pile"])
    losses["near_bowl"] = F.cross_entropy(outputs["near_bowl_logits"], labels["near_bowl"])

    losses["total"] = sum(losses.values())
    return losses


# ============================================================
# 7. SYMBOLIZATION
# ============================================================

class PredicateMapper:
    """
    Converts neural predictions into symbolic predicates.
    """

    PHASE_MAP = {
        0: "approach",
        1: "penetrate_drag",
        2: "lift_carry",
        3: "pour",
    }

    FILL_MAP = {
        0: "empty",
        1: "low",
        2: "medium",
        3: "high",
    }

    def __init__(
        self,
        contact_thresh: float = 0.5,
        spill_thresh: float = 0.5,
        near_thresh: float = 0.5,
    ):
        self.contact_thresh = contact_thresh
        self.spill_thresh = spill_thresh
        self.near_thresh = near_thresh

    def logits_to_state(self, outputs_for_one: Dict[str, torch.Tensor]) -> SymbolicState:
        preds: List[Predicate] = []

        preds.append(("entity", "spoon", "tool"))
        preds.append(("entity", "brown_sugar", "material"))
        preds.append(("entity", "target_bowl", "container"))
        preds.append(("entity", "sugar_pile", "pile"))

        # Convert logits to probabilities
        contact_prob = F.softmax(outputs_for_one["contact_logits"], dim=-1)[1].item()
        spill_prob = F.softmax(outputs_for_one["spill_logits"], dim=-1)[1].item()
        near_pile_prob = F.softmax(outputs_for_one["near_pile_logits"], dim=-1)[1].item()
        near_bowl_prob = F.softmax(outputs_for_one["near_bowl_logits"], dim=-1)[1].item()

        phase_idx = torch.argmax(outputs_for_one["phase_logits"]).item()
        fill_idx = torch.argmax(outputs_for_one["fill_logits"]).item()

        phase_name = self.PHASE_MAP[phase_idx]
        fill_name = self.FILL_MAP[fill_idx]

        preds.append(("phase", phase_name))
        preds.append(("tool_fill", "spoon", fill_name))

        if contact_prob >= self.contact_thresh:
            preds.append(("contacting", "spoon", "brown_sugar"))
        else:
            preds.append(("contact_free", "spoon", "brown_sugar"))

        if spill_prob >= self.spill_thresh:
            preds.append(("spill", "yes"))
        else:
            preds.append(("spill", "no"))

        if near_pile_prob >= self.near_thresh:
            preds.append(("near", "spoon", "sugar_pile"))

        if near_bowl_prob >= self.near_thresh:
            preds.append(("near", "spoon", "target_bowl"))

        # Extra symbolic rules
        if phase_name == "penetrate_drag" and contact_prob >= self.contact_thresh:
            preds.append(("embedded", "spoon", "brown_sugar"))

        if fill_name != "empty":
            preds.append(("holding_material", "spoon", "brown_sugar"))

        if phase_name == "pour" and fill_name != "empty":
            preds.append(("transfer_attempt", "spoon", "target_bowl"))

        if phase_name == "pour" and fill_name == "empty":
            preds.append(("tool_empty_during_pour",))

        return SymbolicState(predicates=frozenset(preds))


# ============================================================
# 8. ACTION EXTRACTION
# ============================================================

class ActionExtractor:
    """
    Defines operation u_t from the symbolic state's phase.
    """
    def state_to_action(self, state: SymbolicState) -> Tuple[str, ...]:
        for p in state.predicates:
            if p[0] == "phase":
                return ("action", p[1])
        return ("action", "unknown")


# ============================================================
# 9. TRANSITION BUILDER
# ============================================================

def build_transitions_for_demo(
    demo_id: str,
    time_to_state: Dict[int, SymbolicState],
    action_extractor: ActionExtractor,
) -> List[Transition]:
    times = sorted(time_to_state.keys())
    transitions: List[Transition] = []

    for i in range(len(times) - 1):
        t = times[i]
        t_next = times[i + 1]

        s_t = time_to_state[t]
        s_t1 = time_to_state[t_next]
        action = action_extractor.state_to_action(s_t)

        transitions.append(
            Transition(
                demo_id=demo_id,
                t=t,
                state_before=s_t,
                action=action,
                state_after=s_t1
            )
        )

    return transitions


# ============================================================
# 10. KNOWLEDGE GRAPH
# ============================================================

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_transitions(self, transitions: List[Transition]) -> None:
        if not transitions:
            return

        demo_id = transitions[0].demo_id
        demo_node = f"demo:{demo_id}"
        self.graph.add_node(demo_node, node_type="Demonstration", demo_id=demo_id)

        for tr in transitions:
            sb = f"state:{tr.demo_id}:{tr.t}:before"
            sa = f"state:{tr.demo_id}:{tr.t+1}:after"
            act = f"action:{tr.demo_id}:{tr.t}:{tr.action[1]}"

            self.graph.add_node(sb, node_type="State", t=tr.t)
            self.graph.add_node(sa, node_type="State", t=tr.t + 1)
            self.graph.add_node(act, node_type="Action", action=tr.action[1], t=tr.t)

            self.graph.add_edge(demo_node, sb, relation="contains_state")
            self.graph.add_edge(demo_node, act, relation="contains_action")
            self.graph.add_edge(demo_node, sa, relation="contains_state")

            self.graph.add_edge(sb, act, relation="before")
            self.graph.add_edge(act, sa, relation="results_in")

            self._attach_state_predicates(sb, tr.state_before)
            self._attach_state_predicates(sa, tr.state_after)

    def _attach_state_predicates(self, state_node: str, state: SymbolicState) -> None:
        for pred in state.predicates:
            pred_node = "pred:" + "|".join(pred)
            self.graph.add_node(pred_node, node_type="Predicate", predicate=pred)
            self.graph.add_edge(state_node, pred_node, relation="has_predicate")


# ============================================================
# 11. SIMPLE OPERATOR INDUCTION
# ============================================================

def induce_action_models(transitions: List[Transition]) -> Dict[str, Dict[str, List[Predicate]]]:
    grouped: Dict[str, List[Transition]] = {}
    for tr in transitions:
        a = tr.action[1]
        grouped.setdefault(a, []).append(tr)

    results = {}
    for action_name, trs in grouped.items():
        pre = set(trs[0].state_before.predicates)
        add = set(trs[0].state_after.predicates - trs[0].state_before.predicates)
        delete = set(trs[0].state_before.predicates - trs[0].state_after.predicates)

        for tr in trs[1:]:
            pre &= set(tr.state_before.predicates)
            add &= set(tr.state_after.predicates - tr.state_before.predicates)
            delete &= set(tr.state_before.predicates - tr.state_after.predicates)

        results[action_name] = {
            "preconditions": sorted(pre),
            "add_effects": sorted(add),
            "delete_effects": sorted(delete),
        }

    return results


# ============================================================
# 12. TRAINING LOOP
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        outputs = model(video, audio)
        losses = compute_losses(outputs, labels)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def infer_states(model, loader, mapper: PredicateMapper, device):
    model.eval()

    all_states_by_demo: Dict[str, Dict[int, SymbolicState]] = {}

    for batch in loader:
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)

        outputs = model(video, audio)

        B = video.shape[0]
        for i in range(B):
            sample_outputs = {
                "contact_logits": outputs["contact_logits"][i:i+1].squeeze(0),
                "phase_logits": outputs["phase_logits"][i:i+1].squeeze(0),
                "fill_logits": outputs["fill_logits"][i:i+1].squeeze(0),
                "spill_logits": outputs["spill_logits"][i:i+1].squeeze(0),
                "near_pile_logits": outputs["near_pile_logits"][i:i+1].squeeze(0),
                "near_bowl_logits": outputs["near_bowl_logits"][i:i+1].squeeze(0),
            }

            state = mapper.logits_to_state(sample_outputs)

            demo_id = batch["demo_ids"][i]
            t = int(batch["times"][i].item())
            all_states_by_demo.setdefault(demo_id, {})[t] = state

    return all_states_by_demo


# ============================================================
# 13. MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MockBrownSugarDataset(num_demos=3, timesteps_per_demo=12)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_demo_samples)

    model = MultimodalPerceptionModel(embed_dim=128, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train briefly on mock data
    for epoch in range(3):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}: loss = {loss:.4f}")

    # Inference
    eval_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_demo_samples)
    mapper = PredicateMapper()
    action_extractor = ActionExtractor()

    states_by_demo = infer_states(model, eval_loader, mapper, device)

    all_transitions: List[Transition] = []
    for demo_id, time_to_state in states_by_demo.items():
        transitions = build_transitions_for_demo(demo_id, time_to_state, action_extractor)
        all_transitions.extend(transitions)

    # Print example states
    for demo_id, time_to_state in states_by_demo.items():
        print("\n" + "=" * 70)
        print(f"Demo: {demo_id}")
        print("=" * 70)
        for t in sorted(time_to_state.keys())[:5]:
            print(f"\nTime {t}")
            for p in time_to_state[t].pretty():
                print(" ", p)

    # Build KG
    kg_builder = KnowledgeGraphBuilder()
    # add transitions per demo
    demos_grouped: Dict[str, List[Transition]] = {}
    for tr in all_transitions:
        demos_grouped.setdefault(tr.demo_id, []).append(tr)

    for demo_id, trs in demos_grouped.items():
        kg_builder.add_transitions(trs)

    G = kg_builder.graph
    print("\n" + "=" * 70)
    print("KG SAMPLE NODES")
    print("=" * 70)
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    for i, (node, data) in enumerate(G.nodes(data=True)):
        print(node, data)
        if i > 10:
           break
    
    print("\n" + "=" * 70)
    print("KG SAMPLE EDGES")
    print("=" * 70)

    for i, (u, v, data) in enumerate(G.edges(data=True)):
        print(f"{u} --[{data['relation']}]--> {v}")
        if i > 15:
            break


    # Induce action models
    action_models = induce_action_models(all_transitions)

    print("\n" + "=" * 70)
    print("INDUCED ACTION MODELS")
    print("=" * 70)
    for action_name, model_info in action_models.items():
        print(f"\nAction: {action_name}")
        print("  Preconditions:")
        for p in model_info["preconditions"]:
            print("   ", p)
        print("  Add effects:")
        for p in model_info["add_effects"]:
            print("   ", p)
        print("  Delete effects:")
        for p in model_info["delete_effects"]:
            print("   ", p)

    nx.write_gml(G, "multimodal_brown_sugar_kg.gml")
    print("\nSaved KG to multimodal_brown_sugar_kg.gml")


if __name__ == "__main__":
    main()