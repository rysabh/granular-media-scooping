import json
import re
import cv2
import torch
import gc
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)
from qwen_vl_utils import process_vision_info

try:
    import ruptures as rpt
except ImportError:
    rpt = None


# -------------------------
# Configuration
# -------------------------
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"

VIDEO_PATH = "training_data/demo.mp4"
OUT_PATH = "trajectory_graph_compacted.json"
NODE_IMAGE_DIR = "node_images"
DETECTION_DEBUG_PATH = "owlv2_detections.json"
SEGMENT_DEBUG_PATH = "change_point_segments.json"

# If True, use the manually specified segments below instead of OWLv2 + change point detection.
USE_MANUAL_SEGMENTS = False
MANUAL_SKILL_SEGMENTS = [
    (0, 60, "approach"),
    (60, 120, "scoop"),
    (120, 180, "lift"),
    (180, 260, "transport"),
    (260, 340, "dump"),
]

# Fallback only, used when OWLv2/change-point detection fails.
SEGMENT_LENGTH_FRAMES = 40

NUM_FRAMES_PER_SEGMENT = 6
IMAGE_SIZE = (224, 224)

# OWLv2 detection settings.
# Running OWLv2 on every frame can be slow, so we sample every N frames.
DETECTION_STRIDE = 3
OWL_SCORE_THRESHOLD = 0.12
MIN_SEGMENT_FRAMES = 8
TARGET_SEGMENT_FRAMES = 18
MAX_SEGMENT_FRAMES = 28

# Prompts for open-vocabulary object detection.
# The dictionary key is the canonical object name used in the feature vector.
OWL_OBJECT_PROMPTS = {
    "spoon": ["spoon", "metal spoon", "scoop spoon", "tool spoon"],
    "brown_sugar": ["brown sugar", "sugar pile", "brown sugar pile", "granular material"],
    "sugar_container": ["container with brown sugar", "container of brown sugar", "sugar container", "bowl with brown sugar"],
    "bucket": ["bucket", "black bucket", "container bucket", "dump bucket"],
}

# Human-provided skill vocabulary Lambda.
SKILL_LABELS = [
    "approach",
    "scoop",
    "lift",
    "transport",
    "dump",
    "return",
    "none",
]

# Domain-specific state definition/hints for scooping.
STATE_HINTS = """
A high-level state definition is defined by:
- the spoon position relative to the brown sugar
- the spoon position relative to the container with brown sugar
- the spoon position relative to the bucket
- whether the spoon contains brown sugar
- the shape, height, and disturbance of the sugar pile
- whether sugar has been deposited into the bucket

Use abstract symbolic predicates such as:
- tool_near_source, tool_touching_source, tool_loaded, tool_near_target, material_in_spoon, sugar_pile_disturbed

Important predicate rule:
- Do NOT output material_deposited directly from the VLM state prompt.
- material_deposited will be added later by the program only when bucket-contact / bucket-deposit evidence is strong.
- If deposition is uncertain, describe the tool/bucket relation but omit material_deposited.

Ignore:
- lighting changes
- small particle-level differences
- camera noise
- exact spoon coordinates
- tiny pose differences that do not change the task-level state
"""

STATE_EQUIVALENCE_CONFIDENCE_THRESHOLD = 0.98
DISABLE_STATE_MERGING_FOR_DEBUG = True

# Global cache used to pass OWLv2 motion evidence into the VLM skill prompt.
SEGMENT_MOTION_SUMMARIES = {}


@dataclass
class Detection:
    box: Optional[List[float]]
    score: float
    prompt: Optional[str]


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_json(text):
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# -------------------------
# Geometry / feature helpers
# -------------------------
def bbox_center(box: Optional[List[float]]) -> Optional[np.ndarray]:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def bbox_area(box: Optional[List[float]]) -> float:
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def iou(box_a: Optional[List[float]], box_b: Optional[List[float]]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    return float(inter / union) if union > 0 else 0.0


def safe_center(det: Dict[str, Detection], name: str, default: np.ndarray) -> np.ndarray:
    center = bbox_center(det[name].box)
    return center if center is not None else default


def normalize_features(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std


def smooth_features(X: np.ndarray, window: int = 5) -> np.ndarray:
    if len(X) < window:
        return X
    kernel = np.ones(window, dtype=np.float32) / window
    return np.stack([np.convolve(X[:, j], kernel, mode="same") for j in range(X.shape[1])], axis=1)


# -------------------------
# OWLv2 object detection
# -------------------------
def load_owlv2():
    print("Loading OWLv2 detector...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID).to(device)
    model.eval()
    return model, processor, device


def run_owlv2_on_image(
    model,
    processor,
    device: str,
    image: Image.Image,
) -> Dict[str, Detection]:
    """Run OWLv2 and return the best detection for each canonical object."""
    canonical_names = list(OWL_OBJECT_PROMPTS.keys())
    prompt_groups = [OWL_OBJECT_PROMPTS[name] for name in canonical_names]
    flat_prompts = [p for group in prompt_groups for p in group]

    inputs = processor(text=[flat_prompts], images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)  # (height, width)
    if hasattr(processor, "post_process_object_detection"):
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=OWL_SCORE_THRESHOLD,
        )[0]
    else:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=OWL_SCORE_THRESHOLD,
        )[0]

    # Map flat prompt index back to canonical object.
    prompt_to_object = {}
    idx = 0
    for name, prompts in OWL_OBJECT_PROMPTS.items():
        for prompt in prompts:
            prompt_to_object[idx] = (name, prompt)
            idx += 1

    best = {
        name: Detection(box=None, score=0.0, prompt=None)
        for name in canonical_names
    }

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_idx = int(label.item())
        obj_name, prompt = prompt_to_object[label_idx]
        score_val = float(score.item())
        if score_val > best[obj_name].score:
            best[obj_name] = Detection(
                box=[float(v) for v in box.detach().cpu().tolist()],
                score=score_val,
                prompt=prompt,
            )

    del inputs, outputs, results
    cleanup_cuda()
    return best


def read_frame_as_pil(cap: cv2.VideoCapture, frame_idx: int) -> Optional[Image.Image]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def detect_objects_over_video(video_path: str) -> Tuple[List[int], List[Dict[str, Detection]]]:
    owl_model, owl_processor, owl_device = load_owlv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = list(range(0, total_frames, DETECTION_STRIDE))
    detections = []

    for k, frame_idx in enumerate(frame_ids):
        if k % 20 == 0:
            print(f"OWLv2 detecting frame {frame_idx}/{total_frames}")
        image = read_frame_as_pil(cap, frame_idx)
        if image is None:
            continue
        detections.append(run_owlv2_on_image(owl_model, owl_processor, owl_device, image))

    cap.release()
    del owl_model, owl_processor
    cleanup_cuda()

    return frame_ids[: len(detections)], detections


def detections_to_jsonable(frame_ids, detections):
    rows = []
    for frame_id, det in zip(frame_ids, detections):
        row = {"frame_id": frame_id}
        for name, d in det.items():
            row[name] = {"box": d.box, "score": d.score, "prompt": d.prompt}
        rows.append(row)
    return rows


# -------------------------
# Time-series feature extraction and change point detection
# -------------------------
def build_time_series_features(
    frame_ids: List[int],
    detections: List[Dict[str, Detection]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert OWLv2 per-frame boxes into a time-series feature matrix.

    The important point is that we do not only use absolute box positions. We also
    include relative position, distance changes, IoU/contact proxies, and changes in
    those proxies. These features are much better for separating approach/scoop/lift/
    transport/dump than raw frames or fixed 120-frame chunks.
    """
    base_rows = []
    prev_centers = {
        "spoon": np.array([0.0, 0.0], dtype=np.float32),
        "brown_sugar": np.array([0.0, 0.0], dtype=np.float32),
        "sugar_container": np.array([0.0, 0.0], dtype=np.float32),
        "bucket": np.array([0.0, 0.0], dtype=np.float32),
    }
    prev_spoon = None

    for det in detections:
        centers = {}
        for name in prev_centers:
            c = bbox_center(det[name].box)
            if c is None:
                c = prev_centers[name]
            else:
                prev_centers[name] = c
            centers[name] = c

        spoon_c = centers["spoon"]
        sugar_c = centers["brown_sugar"]
        container_c = centers["sugar_container"]
        bucket_c = centers["bucket"]

        if prev_spoon is None:
            spoon_v = np.array([0.0, 0.0], dtype=np.float32)
        else:
            spoon_v = spoon_c - prev_spoon
        prev_spoon = spoon_c.copy()

        rel_sugar = spoon_c - sugar_c
        rel_container = spoon_c - container_c
        rel_bucket = spoon_c - bucket_c
        dist_sugar = float(np.linalg.norm(rel_sugar))
        dist_container = float(np.linalg.norm(rel_container))
        dist_bucket = float(np.linalg.norm(rel_bucket))

        base_rows.append([
            spoon_c[0], spoon_c[1],
            spoon_v[0], spoon_v[1], float(np.linalg.norm(spoon_v)),
            rel_sugar[0], rel_sugar[1], dist_sugar,
            rel_container[0], rel_container[1], dist_container,
            rel_bucket[0], rel_bucket[1], dist_bucket,
            iou(det["spoon"].box, det["brown_sugar"].box),
            iou(det["spoon"].box, det["sugar_container"].box),
            iou(det["spoon"].box, det["bucket"].box),
            bbox_area(det["spoon"].box),
            bbox_area(det["brown_sugar"].box),
            bbox_area(det["sugar_container"].box),
            bbox_area(det["bucket"].box),
            det["spoon"].score,
            det["brown_sugar"].score,
            det["sugar_container"].score,
            det["bucket"].score,
        ])

    base = np.array(base_rows, dtype=np.float32)
    if len(base) == 0:
        return base, []

    # First-order temporal changes. These are the most useful features for skill
    # boundaries: moving toward source, leaving source, moving toward bucket, contact
    # onset/offset, etc.
    delta = np.vstack([np.zeros((1, base.shape[1]), dtype=np.float32), np.diff(base, axis=0)])

    base_names = [
        "spoon_x", "spoon_y", "spoon_vx", "spoon_vy", "spoon_speed",
        "spoon_to_sugar_dx", "spoon_to_sugar_dy", "spoon_to_sugar_dist",
        "spoon_to_container_dx", "spoon_to_container_dy", "spoon_to_container_dist",
        "spoon_to_bucket_dx", "spoon_to_bucket_dy", "spoon_to_bucket_dist",
        "iou_spoon_sugar", "iou_spoon_container", "iou_spoon_bucket",
        "spoon_area", "brown_sugar_area", "sugar_container_area", "bucket_area",
        "spoon_score", "brown_sugar_score", "sugar_container_score", "bucket_score",
    ]
    delta_names = ["delta_" + n for n in base_names]

    # Keep all base features plus deltas, but repeat/highlight the most discriminative
    # derivatives so the change-point detector is sensitive to phase transitions.
    selected_delta_cols = [
        base_names.index("spoon_to_sugar_dist"),
        base_names.index("spoon_to_bucket_dist"),
        base_names.index("iou_spoon_sugar"),
        base_names.index("iou_spoon_bucket"),
        base_names.index("spoon_y"),
        base_names.index("spoon_speed"),
    ]
    X = np.concatenate([base, delta, 2.0 * delta[:, selected_delta_cols]], axis=1)
    feature_names = base_names + delta_names + ["weighted_" + delta_names[i] for i in selected_delta_cols]
    return X, feature_names


def summarize_motion_for_segment(
    segment_start: int,
    segment_end: int,
    frame_ids: List[int],
    X_raw: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Create a compact numeric summary to guide the VLM skill label."""
    idxs = [i for i, f in enumerate(frame_ids) if segment_start <= f < segment_end]
    if not idxs:
        return {}
    seg = X_raw[idxs]
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    def change(name: str) -> float:
        if name not in name_to_idx or len(seg) < 2:
            return 0.0
        col = seg[:, name_to_idx[name]]
        return float(col[-1] - col[0])

    def mean(name: str) -> float:
        if name not in name_to_idx:
            return 0.0
        return float(np.mean(seg[:, name_to_idx[name]]))

    return {
        "delta_distance_to_sugar": change("spoon_to_sugar_dist"),
        "delta_distance_to_bucket": change("spoon_to_bucket_dist"),
        "delta_spoon_y": change("spoon_y"),
        "delta_iou_spoon_sugar": change("iou_spoon_sugar"),
        "delta_iou_spoon_bucket": change("iou_spoon_bucket"),
        "mean_spoon_speed": mean("spoon_speed"),
        "mean_iou_spoon_sugar": mean("iou_spoon_sugar"),
        "mean_iou_spoon_bucket": mean("iou_spoon_bucket"),
    }


def heuristic_skill_prior(summary: Dict[str, float]) -> str:
    """
    Automatic motion-derived skill prior from OWLv2 features.

    This is NOT used as the final label. It is stored for debugging and passed to
    the VLM as structured evidence, so the final skill label remains VLM-generated.
    """
    if not summary:
        return "none"

    ds = summary.get("delta_distance_to_sugar", 0.0)
    db = summary.get("delta_distance_to_bucket", 0.0)
    dy = summary.get("delta_spoon_y", 0.0)
    diou_s = summary.get("delta_iou_spoon_sugar", 0.0)
    diou_b = summary.get("delta_iou_spoon_bucket", 0.0)
    miou_s = summary.get("mean_iou_spoon_sugar", 0.0)
    miou_b = summary.get("mean_iou_spoon_bucket", 0.0)
    speed = summary.get("mean_spoon_speed", 0.0)

    # Image coordinates usually have y increasing downward; negative dy means upward motion.
    moving_toward_sugar = ds < -5.0
    moving_away_from_sugar = ds > 5.0
    moving_toward_bucket = db < -5.0
    moving_away_from_bucket = db > 5.0
    lifting = dy < -6.0
    sugar_contact = miou_s > 0.025 or diou_s > 0.015
    bucket_contact = miou_b > 0.025 or diou_b > 0.015

    # Return phase: after dump, tool moves away from bucket and back toward source.
    if moving_away_from_bucket and moving_toward_sugar and speed > 1.0:
        return "return"

    # Transport/dump separation:
    # - transport = leaving source and moving toward bucket
    # - dump = already at/over bucket with clear bucket contact/overlap
    # Do not classify the whole source-to-bucket motion as dump too early.
    if moving_away_from_sugar and moving_toward_bucket:
        if bucket_contact and (miou_b > 0.05 or diou_b > 0.03):
            return "dump"
        return "transport"

    # Lift: mostly upward motion after/near sugar contact, but not yet clearly
    # moving toward the bucket.
    if lifting and (sugar_contact or moving_away_from_sugar):
        if abs(db) < 30.0:
            return "lift"
        if moving_toward_bucket:
            return "transport"
        return "lift"

    # Scoop: contact with sugar or increasing overlap with sugar.
    if sugar_contact:
        return "scoop"

    # Approach: moving toward sugar before contact.
    if moving_toward_sugar:
        return "approach"

    return "none"



def get_motion_evidence_for_trajectory(trajectory_id: str) -> Dict:
    """Return OWLv2 motion evidence with a stable schema for debugging."""
    evidence = SEGMENT_MOTION_SUMMARIES.get(trajectory_id)
    if evidence is None:
        return {
            "summary": {},
            "heuristic_prior": "none",
            "available": False,
            "debug_note": "No OWLv2 motion summary found for this trajectory_id.",
        }
    if "summary" not in evidence:
        evidence = {"summary": evidence, "heuristic_prior": "none"}
    evidence.setdefault("available", bool(evidence.get("summary")))
    evidence.setdefault(
        "debug_note",
        "OWLv2 motion summary attached." if evidence.get("summary") else "OWLv2 motion summary is empty for this segment.",
    )
    return evidence


def constrained_skill_candidates(motion_evidence: Dict) -> List[str]:
    """Choose a stricter automatic candidate set from OWLv2 motion evidence.

    The goal is to keep labeling automated while preventing Qwen from calling every
    source-visible segment ``scoop``. We only allow ``scoop`` when the motion actually
    supports source contact/collection. Otherwise, we remove it from the candidate set.
    """
    prior = motion_evidence.get("heuristic_prior", "none")
    summary = motion_evidence.get("summary", {}) or {}

    ds = summary.get("delta_distance_to_sugar", 0.0)
    db = summary.get("delta_distance_to_bucket", 0.0)
    dy = summary.get("delta_spoon_y", 0.0)
    diou_s = summary.get("delta_iou_spoon_sugar", 0.0)
    diou_b = summary.get("delta_iou_spoon_bucket", 0.0)
    miou_s = summary.get("mean_iou_spoon_sugar", 0.0)
    miou_b = summary.get("mean_iou_spoon_bucket", 0.0)

    moving_toward_sugar = ds < -5.0
    moving_away_from_sugar = ds > 5.0
    moving_toward_bucket = db < -5.0
    moving_away_from_bucket = db > 5.0
    lifting = dy < -6.0
    source_contact_increasing = diou_s > 0.04
    source_contact_decreasing = diou_s < -0.04
    source_contact_high = miou_s > 0.05
    bucket_contact_increasing = diou_b > 0.01
    bucket_contact_high = miou_b > 0.03

    # Pure/early approach: getting closer to sugar, before meaningful contact.
    # Keep scoop as a secondary option only if contact is starting to appear.
    if moving_toward_sugar and not source_contact_high and not source_contact_increasing:
        return ["approach", "none"]
    if moving_toward_sugar and source_contact_increasing:
        return ["scoop", "approach"]

    # Strong transport: leaving source and going toward bucket.
    # This should be transport unless bucket contact/overlap is strong enough
    # to indicate actual dumping. Put transport first so it does not disappear
    # into lift or dump.
    if moving_away_from_sugar and moving_toward_bucket:
        if bucket_contact_high or diou_b > 0.03:
            return ["dump", "transport"]
        return ["transport", "lift"]

    # Post-scoop lift: upward motion, leaving sugar, but bucket distance is not
    # changing much yet. Do not include scoop here; otherwise Qwen tends to choose
    # scoop again.
    if lifting and (moving_away_from_sugar or source_contact_decreasing):
        if abs(db) < 30.0:
            return ["lift", "transport"]
        if moving_toward_bucket:
            return ["transport", "lift"]
        return ["lift", "transport"]

    # Return: moving away from bucket and back to sugar/source. Do not allow scoop
    # unless source contact is increasing again.
    if moving_away_from_bucket and moving_toward_sugar:
        if source_contact_increasing:
            return ["approach", "scoop"]
        return ["return", "approach", "none"]

    # Fall back to the prior, but keep candidates strict.
    if prior == "scoop":
        return ["scoop", "approach"]
    if prior == "lift":
        return ["lift", "transport"]
    if prior == "transport":
        return ["transport", "dump", "lift"]
    if prior == "dump":
        return ["dump", "transport"]
    if prior == "return":
        return ["return", "approach", "none"]
    if prior == "approach":
        return ["approach", "scoop"]

    return SKILL_LABELS

def sanitize_state_inference(state_inference: Dict) -> Dict:
    """Remove unstable VLM predicates and duplicate predicates.

    In this task, Qwen often hallucinates material_deposited whenever the bucket is visible.
    We remove it from the free VLM state output; a conservative programmatic predicate
    can be added later if the motion/visual evidence supports deposit.
    """
    preds = state_inference.get("predicates", []) or []
    cleaned = []
    for pred in preds:
        if pred == "material_deposited":
            continue
        if pred not in cleaned:
            cleaned.append(pred)

    state_inference = dict(state_inference)
    state_inference["predicates"] = cleaned

    parsed = state_inference.get("parsed_vlm_output")
    if isinstance(parsed, dict):
        parsed = dict(parsed)
        parsed["predicates"] = cleaned
        state_inference["parsed_vlm_output"] = parsed
    return state_inference


def maybe_add_programmatic_deposit_predicate(node: Dict, trajectory_id: str, role: str) -> Dict:
    """Conservatively add material_deposited using OWLv2 bucket evidence.

    We only add it for post-states when the segment motion indicates bucket contact or
    bucket overlap has increased. This is automatic, not hand labeling.
    """
    evidence = get_motion_evidence_for_trajectory(trajectory_id)
    summary = evidence.get("summary", {}) or {}
    prior = evidence.get("heuristic_prior", "none")
    bucket_iou = summary.get("mean_iou_spoon_bucket", 0.0)
    delta_bucket_iou = summary.get("delta_iou_spoon_bucket", 0.0)

    strong_deposit = role == "post_state" and (
        prior == "dump" or bucket_iou > 0.05 or delta_bucket_iou > 0.04
    )
    if strong_deposit:
        preds = list(node.get("predicates", []) or [])
        if "material_deposited" not in preds:
            preds.append("material_deposited")
        node["predicates"] = preds
        inf = node.get("state_inference")
        if isinstance(inf, dict):
            inf = dict(inf)
            inf["predicates"] = preds
            parsed = inf.get("parsed_vlm_output")
            if isinstance(parsed, dict):
                parsed = dict(parsed)
                parsed["predicates"] = preds
                inf["parsed_vlm_output"] = parsed
            node["state_inference"] = inf
    return node

def attach_motion_evidence_to_vlm_label(vlm_result: Dict, trajectory_id: str) -> Dict:
    """
    Attach OWLv2 motion evidence to the VLM label without overriding the label.

    This keeps the pipeline automatic but not rule-based/manual: OWLv2 provides
    structured object-motion evidence; Qwen-VL remains responsible for the final
    semantic skill label.
    """
    evidence = get_motion_evidence_for_trajectory(trajectory_id)
    prior = evidence.get("heuristic_prior", "none")

    enriched = dict(vlm_result)
    enriched["vlm_skill_label"] = vlm_result.get("skill_label", "none")
    enriched["motion_heuristic_label"] = prior
    enriched["motion_evidence"] = evidence
    enriched["label_source"] = "vlm_with_owlv2_motion_context"
    return enriched

def postprocess_segments(
    candidate_segments: List[Tuple[int, int, Optional[str]]],
    total_frames: int,
) -> List[Tuple[int, int, Optional[str]]]:
    """Remove tiny segments, split overly long segments, and preserve temporal order."""
    clean = []
    for start, end, label in candidate_segments:
        start = max(0, int(start))
        end = min(total_frames, int(end))
        if end <= start:
            continue
        if end - start < MIN_SEGMENT_FRAMES:
            if clean:
                prev_start, _, prev_label = clean[-1]
                clean[-1] = (prev_start, end, prev_label)
            continue
        while end - start > MAX_SEGMENT_FRAMES:
            split_end = start + TARGET_SEGMENT_FRAMES
            clean.append((start, split_end, label))
            start = split_end
        if end - start >= MIN_SEGMENT_FRAMES:
            clean.append((start, end, label))

    if not clean:
        clean = [(0, total_frames, None)]
    if clean[0][0] > 0:
        clean[0] = (0, clean[0][1], clean[0][2])
    if clean[-1][1] < total_frames:
        s, _, lab = clean[-1]
        clean[-1] = (s, total_frames, lab)
    return clean


def fixed_length_fallback(total_frames: int) -> List[Tuple[int, int, Optional[str]]]:
    return [
        (start, min(start + TARGET_SEGMENT_FRAMES, total_frames), None)
        for start in range(0, total_frames, TARGET_SEGMENT_FRAMES)
    ]


def detect_change_point_segments(video_path: str, total_frames: int) -> List[Tuple[int, int, Optional[str]]]:
    """
    Use OWLv2 detections -> relative motion/contact features -> windowed change-point detection.

    For this scooping video, window-based detection is more suitable than a very
    conservative global PELT penalty because the repeated skills are short and local.
    We still post-process to avoid tiny/noisy segments.
    """
    global SEGMENT_MOTION_SUMMARIES

    if rpt is None:
        print("ruptures is not installed. Falling back to short fixed-length segments.")
        segments = fixed_length_fallback(total_frames)
        SEGMENT_MOTION_SUMMARIES = {
            f"tau_{i:04d}": {
                "summary": {},
                "heuristic_prior": "none",
                "available": False,
                "debug_note": "ruptures not installed; no OWLv2 motion summary generated.",
            }
            for i, _ in enumerate(segments)
        }
        return segments

    frame_ids, detections = detect_objects_over_video(video_path)
    with open(DETECTION_DEBUG_PATH, "w") as f:
        json.dump(detections_to_jsonable(frame_ids, detections), f, indent=2)

    X_raw, feature_names = build_time_series_features(frame_ids, detections)
    if len(X_raw) < 8:
        print("Too few OWLv2 feature points. Falling back to short fixed-length segments.")
        segments = fixed_length_fallback(total_frames)
        SEGMENT_MOTION_SUMMARIES = {
            f"tau_{i:04d}": {
                "summary": {},
                "heuristic_prior": "none",
                "available": False,
                "debug_note": "Too few OWLv2 feature points; fallback segmentation used.",
            }
            for i, _ in enumerate(segments)
        }
        return segments

    X_norm = smooth_features(normalize_features(X_raw), window=3)
    min_size = max(2, MIN_SEGMENT_FRAMES // DETECTION_STRIDE)
    width = max(2 * min_size + 1, TARGET_SEGMENT_FRAMES // DETECTION_STRIDE)
    n_bkps = max(1, min(len(X_norm) // min_size - 1, total_frames // TARGET_SEGMENT_FRAMES - 1))

    # Window is intentionally used here: it detects local distribution changes in
    # object-relative motion, which is what separates approach/scoop/lift/transport/dump.
    algo = rpt.Window(width=width, model="rbf", min_size=min_size).fit(X_norm)
    cp_indices = algo.predict(n_bkps=n_bkps)

    boundaries = [0]
    for cp in cp_indices:
        if cp < len(frame_ids):
            boundaries.append(frame_ids[cp])
        else:
            boundaries.append(total_frames)
    boundaries = sorted(set([b for b in boundaries if 0 <= b <= total_frames]))
    if boundaries[-1] != total_frames:
        boundaries.append(total_frames)

    raw_segments = [(boundaries[i], boundaries[i + 1], None) for i in range(len(boundaries) - 1)]
    segments = postprocess_segments(raw_segments, total_frames)

    SEGMENT_MOTION_SUMMARIES = {}
    debug_segments = []
    for i, (s, e, lab) in enumerate(segments):
        summary = summarize_motion_for_segment(s, e, frame_ids, X_raw, feature_names)
        prior = heuristic_skill_prior(summary)
        SEGMENT_MOTION_SUMMARIES[f"tau_{i:04d}"] = {
            "summary": summary,
            "heuristic_prior": prior,
            "available": bool(summary),
            "debug_note": "OWLv2 motion summary attached." if summary else "OWLv2 summary empty for this segment.",
        }
        debug_segments.append({
            "trajectory_id": f"tau_{i:04d}",
            "start_frame": s,
            "end_frame": e,
            "expected_skill_label": lab,
            "motion_summary": summary,
            "heuristic_skill_prior": prior,
        })

    debug = {
        "algorithm": "OWLv2 features + ruptures.Window(model='rbf')",
        "reason": "Windowed change-point detection is used because scooping primitives are short local motion phases.",
        "detection_stride": DETECTION_STRIDE,
        "owl_score_threshold": OWL_SCORE_THRESHOLD,
        "feature_names": feature_names,
        "min_segment_frames": MIN_SEGMENT_FRAMES,
        "target_segment_frames": TARGET_SEGMENT_FRAMES,
        "max_segment_frames": MAX_SEGMENT_FRAMES,
        "width_in_detection_steps": int(width),
        "n_bkps": int(n_bkps),
        "segments": debug_segments,
    }
    with open(SEGMENT_DEBUG_PATH, "w") as f:
        json.dump(debug, f, indent=2)

    print(f"Detected {len(segments)} segments using OWLv2 + window change-point detection.")
    return segments


# -------------------------
# VLM prompts
# -------------------------
def build_skill_prompt(skill_labels, trajectory_id, motion_evidence=None, candidate_labels=None):
    labels = ", ".join(candidate_labels or skill_labels)
    motion_text = "No OWLv2 motion summary is available. Use visual temporal evidence only."
    if motion_evidence:
        motion_text = json.dumps(motion_evidence, indent=2)
    return f"""
You are an expert robot operator analyzing ONE SHORT phase of a scooping demonstration.

This segment was produced by OWLv2 object detection followed by change-point detection.
It should contain one dominant primitive skill.

OWLv2 MOTION EVIDENCE FOR THIS SEGMENT:
{motion_text}

STATE DEFINITION:
{STATE_HINTS}

TASK:
Choose exactly one skill label from the CANDIDATE LABELS below.
The candidate labels were automatically selected from OWLv2 object-motion trajectories.
This is not manual labeling; it is an automatic constraint to avoid confusing all tool-source scenes as scoop.

Candidate labels:
{labels}

Strict definitions:
- approach: spoon moves toward the brown sugar/container before meaningful contact or collection
- scoop: spoon enters/pushes through brown sugar or increases contact with sugar to collect material
- lift: spoon moves upward away from the sugar after contact/collection
- transport: spoon moves away from the sugar and toward the bucket while carrying/holding material
- dump: spoon is at/over the bucket and material is released/deposited into the bucket
- return: spoon moves back from the bucket/target area toward the brown sugar container for the next scoop
- none: no clear primitive skill occurs

Critical rules:
- Do NOT label a segment as scoop merely because sugar or the source container is visible.
- Only choose scoop when source contact is increasing or the spoon is actively entering/pushing through sugar.
- If OWLv2 says distance_to_sugar increases and sugar overlap decreases, this is lift/transport, not scoop.
- If OWLv2 says spoon y decreases strongly after source contact AND bucket distance is not changing much, prefer lift.
- If OWLv2 says distance_to_sugar increases and distance_to_bucket decreases, prefer transport.
- Only prefer dump when the spoon is at/over the bucket and bucket overlap/contact is strong or increasing.
- If OWLv2 says distance_to_bucket increases and distance_to_sugar decreases, prefer return/approach over scoop.
- If the correct label is not in the candidate set, choose the closest candidate rather than inventing a different label.

Return ONLY valid JSON with all keys present:
{{
  "trajectory_id": "{trajectory_id}",
  "skill_label": "one_label_from_candidate_labels",
  "confidence": 0.0,
  "reason": "brief visual and motion-based reason"
}}
"""


def build_state_inference_prompt(node_id):
    return f"""
You are an expert robot operator describing a high-level symbolic state in a scooping task.

You are given one image snapshot for node {node_id}.

STATE DEFINITION:
{STATE_HINTS}

Infer the high-level state using task-level predicates. Focus on:
- spoon relation to brown sugar
- spoon relation to sugar container
- spoon relation to bucket
- whether spoon appears to contain material
- whether the sugar pile is disturbed
- whether sugar appears deposited in the bucket

Return ONLY valid JSON with all keys present:
{{
  "node_id": "{node_id}",
  "state_summary": "brief symbolic state description",
  "predicates": ["short_predicate_1", "short_predicate_2"],
  "confidence": 0.0
}}
"""


def build_state_equivalence_prompt(node_a_id, node_b_id):
    return f"""
You are an expert robot operator assessing whether two snapshots represent the SAME high-level world state.

You are given two images:
- Image A: snapshot for node A ({node_a_id})
- Image B: snapshot for node B ({node_b_id})

STATE DEFINITION:
{STATE_HINTS}

Question:
Do Image A and Image B depict the SAME high-level task state?

Two images are the SAME high-level state if the task phase is the same, even if the spoon has moved slightly.
Ignore exact spoon coordinates. Do not mark states different only because the spoon is in a slightly different pose.

Examples of SAME states:
- spoon slightly shifted while still approaching the sugar
- spoon slightly shifted while still scooping
- spoon slightly shifted while still transporting
- minor sugar pile deformation without a task-level change

Examples of DIFFERENT states:
- before contact with sugar vs actively scooping sugar
- empty spoon vs spoon loaded with material
- loaded spoon near source vs loaded spoon near bucket
- before dumping vs after material is deposited into bucket

Return ONLY valid JSON with all keys present:
{{
  "same_state": true,
  "confidence": 0.0,
  "reason": "brief reason"
}}
"""


# -------------------------
# VLM calls
# -------------------------
def run_vlm_on_images(model, processor, images, prompt, max_new_tokens=160):
    content = []
    for image in images:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        content.append({"type": "image", "image": image})

    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated_ids, generated_ids_trimmed
    cleanup_cuda()
    return output_text


def label_skill(model, processor, images, trajectory_id):
    motion_evidence = get_motion_evidence_for_trajectory(trajectory_id)
    candidate_labels = constrained_skill_candidates(motion_evidence)
    prompt = build_skill_prompt(SKILL_LABELS, trajectory_id, motion_evidence, candidate_labels)
    raw_output = run_vlm_on_images(
        model,
        processor,
        images,
        prompt,
        max_new_tokens=160,
    )
    parsed = extract_json(raw_output)

    if parsed is None:
        return {
            "skill_label": "none",
            "confidence": 0.0,
            "raw_vlm_output": raw_output,
            "parsed_vlm_output": None,
        }

    skill_label = parsed.get("skill_label", "none")
    if skill_label not in candidate_labels:
        # Keep the final label VLM-generated but force it to respect the automatic candidate set.
        # If the VLM outputs an invalid label, fall back to the automatic motion prior.
        skill_label = motion_evidence.get("heuristic_prior", "none")
        if skill_label not in candidate_labels:
            skill_label = candidate_labels[0] if candidate_labels else "none"

    result = {
        "skill_label": skill_label,
        "confidence": float(parsed.get("confidence", 0.0)),
        "raw_vlm_output": raw_output,
        "parsed_vlm_output": parsed,
        "candidate_skill_labels": candidate_labels,
    }
    return attach_motion_evidence_to_vlm_label(result, trajectory_id)


def infer_state(model, processor, image, node_id):
    prompt = build_state_inference_prompt(node_id)
    raw_output = run_vlm_on_images(
        model,
        processor,
        [image],
        prompt,
        max_new_tokens=160,
    )
    parsed = extract_json(raw_output)
    result = {
        "raw_vlm_output": raw_output,
        "parsed_vlm_output": parsed,
        "state_summary": None if parsed is None else parsed.get("state_summary"),
        "predicates": [] if parsed is None else parsed.get("predicates", []),
        "confidence": 0.0 if parsed is None else float(parsed.get("confidence", 0.0)),
    }
    return sanitize_state_inference(result)


def judge_same_state(model, processor, image_a, image_b, node_a_id, node_b_id):
    prompt = build_state_equivalence_prompt(node_a_id, node_b_id)
    raw_output = run_vlm_on_images(
        model,
        processor,
        [image_a, image_b],
        prompt,
        max_new_tokens=120,
    )
    parsed = extract_json(raw_output)

    if parsed is None:
        return {
            "same_state": False,
            "confidence": 0.0,
            "raw_vlm_output": raw_output,
            "parsed_vlm_output": None,
        }

    same_state = parsed.get("same_state", False)
    if isinstance(same_state, str):
        same_state = same_state.strip().lower() == "true"

    return {
        "same_state": bool(same_state),
        "confidence": float(parsed.get("confidence", 0.0)),
        "raw_vlm_output": raw_output,
        "parsed_vlm_output": parsed,
    }


# -------------------------
# Video sampling
# -------------------------
def sample_segment_frames(video_path, start_frame, end_frame, num_frames):
    cap = cv2.VideoCapture(video_path)
    frame_indices = [
        int(start_frame + i * (end_frame - start_frame - 1) / max(num_frames - 1, 1))
        for i in range(num_frames)
    ]

    images = []
    valid_frame_indices = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(frame_rgb))
        valid_frame_indices.append(idx)

    cap.release()
    return images, valid_frame_indices


def save_node_image(image, path):
    image.convert("RGB").save(path)


# -------------------------
# Graph construction
# -------------------------
def find_or_create_abstract_node(model, processor, abstract_nodes, candidate_node):
    """Merge visual states using VLM-based state equivalence.

    For debugging segmentation, DISABLE_STATE_MERGING_FOR_DEBUG=True creates one
    abstract node per raw state so we can see whether the states really change.
    Turn it off later when labels/segments are reliable.
    """
    if DISABLE_STATE_MERGING_FOR_DEBUG:
        abstract_id = f"state_{len(abstract_nodes):04d}"
        abstract_nodes.append(
            {
                "abstract_node_id": abstract_id,
                "representative_raw_node_id": candidate_node["raw_node_id"],
                "representative_image_path": candidate_node["image_path"],
                "representative_frame_id": candidate_node["frame_id"],
                "representative_role": candidate_node["role"],
                "representative_image": candidate_node["image"],
                "state_summary": candidate_node.get("state_summary"),
                "predicates": candidate_node.get("predicates", []),
                "state_inference": candidate_node.get("state_inference"),
                "members": [candidate_node["raw_node_id"]],
                "equivalence_checks": [],
            }
        )
        return abstract_id, False

    for abstract_node in abstract_nodes:
        result = judge_same_state(
            model,
            processor,
            candidate_node["image"],
            abstract_node["representative_image"],
            candidate_node["raw_node_id"],
            abstract_node["abstract_node_id"],
        )

        abstract_node.setdefault("equivalence_checks", []).append(
            {
                "candidate_raw_node_id": candidate_node["raw_node_id"],
                "same_state": result["same_state"],
                "confidence": result["confidence"],
                "parsed_vlm_output": result["parsed_vlm_output"],
                "raw_vlm_output": result["raw_vlm_output"],
            }
        )

        if result["same_state"] and result["confidence"] >= STATE_EQUIVALENCE_CONFIDENCE_THRESHOLD:
            abstract_node["members"].append(candidate_node["raw_node_id"])
            return abstract_node["abstract_node_id"], True

    abstract_id = f"state_{len(abstract_nodes):04d}"
    abstract_nodes.append(
        {
            "abstract_node_id": abstract_id,
            "representative_raw_node_id": candidate_node["raw_node_id"],
            "representative_image_path": candidate_node["image_path"],
            "representative_frame_id": candidate_node["frame_id"],
            "representative_role": candidate_node["role"],
            "representative_image": candidate_node["image"],
            "state_summary": candidate_node.get("state_summary"),
            "predicates": candidate_node.get("predicates", []),
            "state_inference": candidate_node.get("state_inference"),
            "members": [candidate_node["raw_node_id"]],
            "equivalence_checks": [],
        }
    )
    return abstract_id, False


def strip_runtime_images(graph):
    """Remove PIL images before JSON serialization."""
    for node in graph["abstract_nodes"]:
        node.pop("representative_image", None)
    for node in graph["raw_nodes"]:
        node.pop("image", None)
    return graph


def load_vlm():
    print("Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    return model, processor


def resolve_video_path(path_str: str) -> str:
    """Resolve relative paths robustly from either current working directory or script directory."""
    p = Path(path_str).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        candidates.append(Path(__file__).resolve().parent / p)
    for c in candidates:
        if c.exists():
            return str(c)
    tried = "\n".join(str(c) for c in candidates)
    raise RuntimeError(f"Could not find video: {path_str}\nTried:\n{tried}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build VLM trajectory graph with OWLv2 + change-point segmentation.")
    parser.add_argument("--video-path", default=VIDEO_PATH, help="Path to input demonstration video.")
    parser.add_argument("--out-path", default=OUT_PATH, help="Path to output JSON graph.")
    return parser.parse_args()


def main():
    cleanup_cuda()
    args = parse_args()
    video_path = resolve_video_path(args.video_path)
    out_path = args.out_path

    print(f"Using video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if USE_MANUAL_SEGMENTS:
        segment_iter = MANUAL_SKILL_SEGMENTS
        segmentation_method = "manual"
    else:
        segment_iter = detect_change_point_segments(video_path, total_frames)
        segmentation_method = "owlv2_features_window_rbf"

    model, processor = load_vlm()

    graph = {
        "raw_nodes": [],
        "abstract_nodes": [],
        "edges": [],
        "skill_vocabulary": SKILL_LABELS,
        "state_hints": STATE_HINTS,
        "segmentation_method": segmentation_method,
        "skill_labeling_method": "Qwen2.5-VL final label constrained by automatic OWLv2 motion candidate labels; transport is prioritized when motion moves away from sugar toward bucket; dump requires strong bucket contact; heuristic prior is used only when VLM outputs an invalid candidate",
        "object_detector": {
            "model_id": OWL_MODEL_ID,
            "object_prompts": OWL_OBJECT_PROMPTS,
            "score_threshold": OWL_SCORE_THRESHOLD,
            "detection_stride": DETECTION_STRIDE,
        },
        "change_point_detector": {
            "algorithm": "ruptures.Window(model='rbf')",
            "input_features": "relative object positions, velocities, distance changes, IoU/contact changes, detection scores",
            "min_segment_frames": MIN_SEGMENT_FRAMES,
            "max_segment_frames": MAX_SEGMENT_FRAMES,
        },
        "state_equivalence_confidence_threshold": STATE_EQUIVALENCE_CONFIDENCE_THRESHOLD,
        "disable_state_merging_for_debug": DISABLE_STATE_MERGING_FOR_DEBUG,
    }

    os.makedirs(NODE_IMAGE_DIR, exist_ok=True)
    trajectory_id = 0
    prev_post_state_id = None

    for start, end, expected_skill in segment_iter:
        start = max(0, int(start))
        end = min(int(end), total_frames)
        if end - start < 5:
            continue

        traj_name = f"tau_{trajectory_id:04d}"
        print(f"Processing trajectory {traj_name}: frames {start}-{end}")

        images, frame_indices = sample_segment_frames(
            video_path,
            start,
            end,
            NUM_FRAMES_PER_SEGMENT,
        )
        if len(images) < 2:
            continue

        pre_raw_id = f"raw_node_{trajectory_id:04d}_pre"
        post_raw_id = f"raw_node_{trajectory_id:04d}_post"

        pre_image_path = f"{NODE_IMAGE_DIR}/{pre_raw_id}.jpg"
        post_image_path = f"{NODE_IMAGE_DIR}/{post_raw_id}.jpg"
        save_node_image(images[0], pre_image_path)
        save_node_image(images[-1], post_image_path)

        # Step 3: after segmentation, infer symbolic states with VLM.
        pre_state_inference = infer_state(model, processor, images[0], pre_raw_id)
        post_state_inference = infer_state(model, processor, images[-1], post_raw_id)

        pre_candidate = {
            "raw_node_id": pre_raw_id,
            "trajectory_id": traj_name,
            "role": "pre_state",
            "frame_id": frame_indices[0],
            "image_path": pre_image_path,
            "image": images[0],
            "state_summary": pre_state_inference["state_summary"],
            "predicates": pre_state_inference["predicates"],
            "state_inference": pre_state_inference,
        }

        post_candidate = {
            "raw_node_id": post_raw_id,
            "trajectory_id": traj_name,
            "role": "post_state",
            "frame_id": frame_indices[-1],
            "image_path": post_image_path,
            "image": images[-1],
            "state_summary": post_state_inference["state_summary"],
            "predicates": post_state_inference["predicates"],
            "state_inference": post_state_inference,
        }

        pre_candidate = maybe_add_programmatic_deposit_predicate(pre_candidate, traj_name, "pre_state")
        post_candidate = maybe_add_programmatic_deposit_predicate(post_candidate, traj_name, "post_state")

        graph["raw_nodes"].append(pre_candidate)
        graph["raw_nodes"].append(post_candidate)

        pre_state_id, pre_was_merged = find_or_create_abstract_node(
            model,
            processor,
            graph["abstract_nodes"],
            pre_candidate,
        )
        post_state_id, post_was_merged = find_or_create_abstract_node(
            model,
            processor,
            graph["abstract_nodes"],
            post_candidate,
        )

        # Keep temporal continuity for one video trajectory.
        if prev_post_state_id is not None:
            pre_state_id = prev_post_state_id
        prev_post_state_id = post_state_id

        # Step 3: label the segmented skill with VLM.
        skill_result = label_skill(model, processor, images, traj_name)

        graph["edges"].append(
            {
                "trajectory_id": traj_name,
                "segment_start_frame": start,
                "segment_end_frame": end,
                "source": pre_state_id,
                "target": post_state_id,
                "raw_source": pre_raw_id,
                "raw_target": post_raw_id,
                "source_was_merged": pre_was_merged,
                "target_was_merged": post_was_merged,
                "expected_skill_label": expected_skill,
                "skill_label": skill_result["skill_label"],
                "vlm_skill_label": skill_result.get("vlm_skill_label", skill_result["skill_label"]),
                "motion_heuristic_label": skill_result.get("motion_heuristic_label"),
                "label_source": skill_result.get("label_source", "vlm"),
                "motion_evidence": skill_result.get("motion_evidence", {}).get("summary", {}),
                "motion_evidence_available": skill_result.get("motion_evidence", {}).get("available", False),
                "motion_evidence_debug_note": skill_result.get("motion_evidence", {}).get("debug_note"),
                "full_motion_context": skill_result.get("motion_evidence"),
                "candidate_skill_labels": skill_result.get("candidate_skill_labels"),
                "confidence": skill_result["confidence"],
                "raw_vlm_output": skill_result["raw_vlm_output"],
                "parsed_vlm_output": skill_result["parsed_vlm_output"],
            }
        )

        del images
        cleanup_cuda()
        trajectory_id += 1

    graph["segment_motion_summaries"] = SEGMENT_MOTION_SUMMARIES

    graph = strip_runtime_images(graph)
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Saved compacted trajectory graph to {out_path}")


if __name__ == "__main__":
    main()
