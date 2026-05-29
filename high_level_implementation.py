"""
High-level scooping pipeline (boundary-centered, learning-based version):

Video
  -> OWLv2 open-vocabulary detection of task entities
  -> object-centric crops: spoon, source sugar pile
  -> interaction-centric crops: spoon-source pile, spoon-target container
  -> global-frame crop
  -> DINOv2 embeddings for object, interaction, and scene crops
  -> aggregate learned visual signal Z(t)
  -> optional temporal embedding deltas Delta Z(t)
  -> ruptures PELT(model="rbf") change-point detection
  -> rupture-defined state segments between boundaries
  -> VLM scene description for each rupture segment
  -> compact boundary-centered transition evidence
  -> VLM closed-set skill inference from semantic state change
  -> symbolic state / skill graph.


"""

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    Owlv2ForObjectDetection,
    Owlv2Processor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

try:
    import ruptures as rpt
except ImportError:
    rpt = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None


# -------------------------
# Model / path configuration
# -------------------------
VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
DINO_MODEL_ID = "facebook/dinov2-base"

VIDEO_PATH = "training_data/demo.mp4"
OUT_PATH = "trajectory_graph_highlevel_boundary.json"
NODE_IMAGE_DIR = "node_images"
CROP_DIR = "object_crops"
DETECTION_DEBUG_PATH = "owlv2_detections.json"
EMBEDDING_DEBUG_PATH = "dino_object_interaction_global_embeddings.npz"
BOUNDARY_DEBUG_PATH = "change_point_boundaries.json"

# Detection / embedding cadence.
DETECTION_STRIDE = 1
OWL_SCORE_THRESHOLD = 0.12
CROP_PADDING = 0.12

# Include full-frame/global DINO embedding so rupture can capture scene layout
# and object locations that are partly lost by object crops.
USE_GLOBAL_FRAME_EMBEDDING = True

# Use learned object-centric, interaction-centric, and scene-centric DINO embeddings only.
# No explicit distance, centroid, optical-flow, or handcrafted motion features are used.
USE_RELATIVE_SPATIAL_DELTAS = False

# Object/interaction embedding streams used for rupture segmentation.
SEGMENTATION_OBJECT_STREAMS = ["spoon", "sugar_pile"]
SEGMENTATION_INTERACTION_STREAMS = ["spoon_source_interaction", "spoon_target_interaction"]
INTERACTION_CROP_PADDING = 0.20

# Embedding signal options. PCA is disabled by default; enable only if runtime/noise requires it.
PCA_DIMS = None
USE_DELTA_EMBEDDINGS = True
SMOOTHING_WINDOW = 3

# Change point detection. PELT is the search algorithm; RBF is the cost model.
# RBF measures nonlinear similarity changes in the joint DINO embedding signal.
# No manual segment length is imposed: PELT decides boundaries from the embedding signal.
RUPTURES_MODEL = "rbf"
RUPTURES_PENALTY = 4

# Boundary-centered VLM evidence budget.
# These caps are NOT segmentation parameters. They only limit how many images are sent
# to the VLM from adaptive windows derived from neighboring rupture boundaries.
MAX_IMAGES_PER_STATE_WINDOW = 8
MAX_IMAGES_PER_TRANSITION_WINDOW = 8
IMAGE_SIZE = (224, 224)

# Open-vocabulary detection prompts.
OWL_OBJECT_PROMPTS = {
    "spoon": [
        "spoon",
        "metal spoon",
        "scoop spoon",
        "spoon containing brown sugar",
        "empty spoon",
    ],
    "sugar_pile": [
        "brown sugar pile",
        "sugar pile",
        "brown sugar",
        "granular material",
        "disturbed sugar pile",
    ],
    "source_container": [
        "container with brown sugar",
        "bowl with brown sugar",
        "sugar container",
        "source container",
    ],
    "target_container": [
        "bucket",
        "dump bucket",
        "target bucket",
        "container bucket",
    ],
}

SKILL_LABELS = [
    "approach",
    "scoop",
    "lift",
    "transport",
    "dump",
    "return",
    "none",
]

STATE_HINTS = """
State clips are rupture-defined frame segments between embedding change boundaries.

Describe ONLY visible scene conditions.
A state is not an action.

Output ONLY conditions observable in the current segment.

Required fields:
- spoon_occupancy:
    empty / contains_material

- spoon_to_source:
    inside / near / far

- spoon_to_target:
    near / far

- sugar_pile:
    intact / disturbed

- target_container:
    empty / contains_material

Rules:
- Do NOT infer action.
- Do NOT infer skill.
- Do NOT infer what happened before.
- Do NOT infer what happens next.
- Ignore human intent.
- Use ONLY the allowed values above.
- Do not output unclear. Choose the closest visible state from the allowed vocabulary.

State summary should be a compact condition list.

Good:
"spoon_occupancy=contains_material; spoon_to_source=inside; spoon_to_target=far; sugar_pile=disturbed; target_container=empty"

Bad:
"spoon scooping sugar"
"spoon moving toward bucket"
"spoon_occupancy=value"
"spoon_occupancy=unclear"
"""

STATE_EQUIVALENCE_CONFIDENCE_THRESHOLD = 0.95
DISABLE_STATE_MERGING_FOR_DEBUG = True


@dataclass
class Detection:
    box: Optional[List[float]]
    score: float
    prompt: Optional[str]


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_json(text: str) -> Optional[Dict]:
    """Extract the first valid JSON object from VLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    # Try direct parse first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: use the outermost JSON-looking object.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def resolve_video_path(path_str: str) -> str:
    p = Path(path_str).expanduser()
    candidates = [p] if p.is_absolute() else [Path.cwd() / p, Path(__file__).resolve().parent / p]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError("Could not find video. Tried:\n" + "\n".join(str(c) for c in candidates))


def read_frame_as_pil(cap: cv2.VideoCapture, frame_idx: int) -> Optional[Image.Image]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# -------------------------
# OWLv2 detection and crops
# -------------------------
def load_owlv2():
    print("Loading OWLv2 detector...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID).to(device)
    model.eval()
    return model, processor, device


def run_owlv2_on_image(model, processor, device: str, image: Image.Image) -> Dict[str, Detection]:
    canonical_names = list(OWL_OBJECT_PROMPTS.keys())
    flat_prompts = [p for name in canonical_names for p in OWL_OBJECT_PROMPTS[name]]

    inputs = processor(text=[flat_prompts], images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    if hasattr(processor, "post_process_object_detection"):
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=OWL_SCORE_THRESHOLD
        )[0]
    else:
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=OWL_SCORE_THRESHOLD
        )[0]

    prompt_to_object = {}
    idx = 0
    for name in canonical_names:
        for prompt in OWL_OBJECT_PROMPTS[name]:
            prompt_to_object[idx] = (name, prompt)
            idx += 1

    best = {name: Detection(None, 0.0, None) for name in canonical_names}
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


def detect_objects_over_video(video_path: str) -> Tuple[List[int], List[Dict[str, Detection]]]:
    model, processor, device = load_owlv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ids = list(range(0, total_frames, DETECTION_STRIDE))
    detections: List[Dict[str, Detection]] = []

    for k, frame_idx in enumerate(frame_ids):
        if k % 5 == 0:
            print(f"OWLv2 detecting frame {frame_idx}/{total_frames}")
        image = read_frame_as_pil(cap, frame_idx)
        if image is None:
            continue
        det = run_owlv2_on_image(model, processor, device, image)
        detections.append(det)
        if k % 5 == 0:
            print_detection_row(frame_idx, det)

    print_detection_summary(detections)
    cap.release()
    del model, processor
    cleanup_cuda()
    return frame_ids[: len(detections)], detections


def detections_to_jsonable(frame_ids: List[int], detections: List[Dict[str, Detection]]) -> List[Dict]:
    rows = []
    for frame_id, det in zip(frame_ids, detections):
        row = {"frame_id": frame_id}
        for name, d in det.items():
            row[name] = {"box": d.box, "score": d.score, "prompt": d.prompt}
        rows.append(row)
    return rows


def print_detection_row(frame_idx: int, det: Dict[str, Detection]) -> None:
    """Print OWLv2 detections for one sampled frame."""
    print(f"\nDetected OWLv2 objects at frame {frame_idx}:")
    print("-" * 60)
    for name in OWL_OBJECT_PROMPTS.keys():
        d = det.get(name)
        if d is None or d.box is None:
            print(f"{name}: NOT DETECTED")
        else:
            box = [round(float(v), 2) for v in d.box]
            print(f"{name}: score={d.score:.3f}, prompt={d.prompt}, box={box}")
    print("-" * 60)


def print_detection_summary(detections: List[Dict[str, Detection]]) -> None:
    """Print aggregate detection coverage for the four OWLv2 object streams."""
    if not detections:
        return
    print("\n========= OWLv2 Detection Summary =========")
    for name in OWL_OBJECT_PROMPTS.keys():
        found = sum(1 for det in detections if det.get(name) is not None and det[name].box is not None)
        coverage = 100.0 * found / len(detections)
        avg_score = np.mean([det[name].score for det in detections if det.get(name) is not None and det[name].box is not None]) if found else 0.0
        print(f"{name}: {found}/{len(detections)} frames ({coverage:.1f}%), avg_score={avg_score:.3f}")
    print("==========================================\n")


def padded_crop(image: Image.Image, box: Optional[List[float]], padding: float = CROP_PADDING) -> Image.Image:
    if box is None:
        return image.copy()
    w, h = image.size
    x1, y1, x2, y2 = box
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0, int(x1 - padding * bw))
    y1 = max(0, int(y1 - padding * bh))
    x2 = min(w, int(x2 + padding * bw))
    y2 = min(h, int(y2 + padding * bh))
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image.crop((x1, y1, x2, y2))





def crop_two_object_region(
    image: Image.Image,
    box_a: Optional[List[float]],
    box_b: Optional[List[float]],
    padding: float = INTERACTION_CROP_PADDING,
) -> Image.Image:
    """Crop the union of two OWLv2 boxes with padding.

    This produces interaction-centric crops, e.g. spoon-source-pile and
    spoon-target-container regions. The crop is automatically determined from
    detections; no explicit distance/contact/flow feature is computed.
    """
    if box_a is None or box_b is None:
        return image.copy()
    w, h = image.size
    x1 = min(float(box_a[0]), float(box_b[0]))
    y1 = min(float(box_a[1]), float(box_b[1]))
    x2 = max(float(box_a[2]), float(box_b[2]))
    y2 = max(float(box_a[3]), float(box_b[3]))
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0, int(x1 - padding * bw))
    y1 = max(0, int(y1 - padding * bh))
    x2 = min(w, int(x2 + padding * bw))
    y2 = min(h, int(y2 + padding * bh))
    if x2 <= x1 or y2 <= y1:
        return image.copy()
    return image.crop((x1, y1, x2, y2))


def normalized_box_center(box: Optional[List[float]], image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Return normalized box center [cx, cy], or None for missing boxes."""
    if box is None:
        return None
    image_w, image_h = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2) / max(float(image_w), 1.0)
    cy = 0.5 * (y1 + y2) / max(float(image_h), 1.0)
    return np.array([cx, cy], dtype=np.float32)


def relative_center_delta(
    moving_box: Optional[List[float]],
    reference_box: Optional[List[float]],
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Return normalized [dx, dy] = center(moving_box) - center(reference_box).

    Missing boxes are encoded as [0, 0] so the time series remains dense.
    """
    moving_center = normalized_box_center(moving_box, image_size)
    reference_center = normalized_box_center(reference_box, image_size)
    if moving_center is None or reference_center is None:
        return np.zeros(2, dtype=np.float32)
    return (moving_center - reference_center).astype(np.float32)



# -------------------------
# DINOv2 object/crop embeddings
# -------------------------
def load_dino():
    print("Loading DINOv2 embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModel.from_pretrained(DINO_MODEL_ID).to(device)
    model.eval()
    return model, processor, device


def embed_images_dino(model, processor, device: str, images: List[Image.Image]) -> np.ndarray:
    if not images:
        return np.zeros((0, 1), dtype=np.float32)
    rgb_images = [img.convert("RGB") for img in images]
    inputs = processor(images=rgb_images, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    arr = emb.detach().cpu().numpy().astype(np.float32)
    del inputs, outputs, emb
    cleanup_cuda()
    return arr


def extract_object_crop_embeddings(
    video_path: str,
    frame_ids: List[int],
    detections: List[Dict[str, Detection]],
    save_crops: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Return aggregate learned visual sequence Z[t] and per-stream embeddings.

    The rupture signal uses only learned DINOv2 visual representations from:
      1. object-centric crops:
            z_spoon = DINO(crop(spoon))
            z_source_pile = DINO(crop(source sugar pile))
      2. interaction-centric crops:
            z_spoon_source = DINO(union(spoon, source sugar pile))
            z_spoon_target = DINO(union(spoon, target container))
      3. optional global-frame embedding:
            z_global = DINO(full frame)

    No explicit centroid distance, velocity, contact, optical-flow, or spatial
    handcrafted feature is added. Relative layout and material transfer are
    captured implicitly by the interaction-region embeddings.
    """
    dino_model, dino_processor, dino_device = load_dino()
    cap = cv2.VideoCapture(video_path)

    stream_names_config = list(SEGMENTATION_OBJECT_STREAMS) + list(SEGMENTATION_INTERACTION_STREAMS)
    if USE_GLOBAL_FRAME_EMBEDDING:
        stream_names_config.append("global_frame")

    streams: Dict[str, List[np.ndarray]] = {name: [] for name in stream_names_config}

    crop_paths_debug: List[Dict] = []
    if save_crops:
        Path(CROP_DIR).mkdir(parents=True, exist_ok=True)

    for idx, (frame_id, det) in enumerate(zip(frame_ids, detections)):
        image = read_frame_as_pil(cap, frame_id)
        if image is None:
            continue

        crops: List[Image.Image] = []
        names_for_frame: List[str] = []
        debug_row = {"frame_id": frame_id, "crops": {}}

        # Object-centric crops.
        object_crop_specs = {
            "spoon": det["spoon"].box,
            "sugar_pile": det["sugar_pile"].box,
        }
        for name in SEGMENTATION_OBJECT_STREAMS:
            crop = padded_crop(image, object_crop_specs[name])
            crops.append(crop)
            names_for_frame.append(name)
            if save_crops and idx % 10 == 0:
                path = Path(CROP_DIR) / f"frame_{frame_id:06d}_{name}.jpg"
                crop.convert("RGB").save(path)
                debug_row["crops"][name] = str(path)

        # Interaction-centric crops from detected object pairs.
        interaction_crop_specs = {
            "spoon_source_interaction": (det["spoon"].box, det["sugar_pile"].box),
            "spoon_target_interaction": (det["spoon"].box, det["target_container"].box),
        }
        for name in SEGMENTATION_INTERACTION_STREAMS:
            box_a, box_b = interaction_crop_specs[name]
            crop = crop_two_object_region(image, box_a, box_b)
            crops.append(crop)
            names_for_frame.append(name)
            if save_crops and idx % 10 == 0:
                path = Path(CROP_DIR) / f"frame_{frame_id:06d}_{name}.jpg"
                crop.convert("RGB").save(path)
                debug_row["crops"][name] = str(path)

        # Full-scene/global embedding.
        if USE_GLOBAL_FRAME_EMBEDDING:
            crops.append(image)
            names_for_frame.append("global_frame")
            if save_crops and idx % 10 == 0:
                path = Path(CROP_DIR) / f"frame_{frame_id:06d}_global_frame.jpg"
                image.convert("RGB").save(path)
                debug_row["crops"]["global_frame"] = str(path)

        embs = embed_images_dino(dino_model, dino_processor, dino_device, crops)
        for name, emb in zip(names_for_frame, embs):
            streams[name].append(emb)

        if debug_row["crops"]:
            crop_paths_debug.append(debug_row)

    cap.release()
    del dino_model, dino_processor
    cleanup_cuda()

    stream_arrays = {name: np.stack(vals, axis=0) for name, vals in streams.items() if vals}
    if not stream_arrays:
        raise RuntimeError("No DINO embeddings were extracted.")

    T = min(arr.shape[0] for arr in stream_arrays.values())
    stream_arrays = {k: v[:T] for k, v in stream_arrays.items()}

    stream_names = [name for name in stream_names_config if name in stream_arrays]
    Z = np.concatenate([stream_arrays[name] for name in stream_names], axis=1).astype(np.float32)

    with open(Path(EMBEDDING_DEBUG_PATH).with_suffix(".crops.json"), "w") as f:
        json.dump(crop_paths_debug, f, indent=2)

    return Z, stream_arrays, stream_names

def normalize_features(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mean) / std


def smooth_features(X: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    if window <= 1 or len(X) < window:
        return X
    kernel = np.ones(window, dtype=np.float32) / window
    return np.stack([np.convolve(X[:, j], kernel, mode="same") for j in range(X.shape[1])], axis=1)


def prepare_signal_for_ruptures(Z: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Prepare joint embedding signal for PELT+RBF.

    Z is the concatenated learned visual embedding vector:
        Z(t) = [z_spoon, z_source_pile, z_spoon_source_interaction,
                z_spoon_target_interaction, z_global]

    RBF inside PELT detects nonlinear similarity/distribution changes in this
    object-centric, interaction-centric, and scene-centric representation.
    """
    Z_norm = normalize_features(Z)

    if PCA is not None and PCA_DIMS and Z_norm.shape[0] > 3 and Z_norm.shape[1] > PCA_DIMS:
        n_components = min(PCA_DIMS, Z_norm.shape[0] - 1, Z_norm.shape[1])
        pca = PCA(n_components=n_components, random_state=0)
        Z_low = pca.fit_transform(Z_norm).astype(np.float32)
        pca_info = {
            "used": True,
            "n_components": int(n_components),
            "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        }
    else:
        Z_low = Z_norm
        pca_info = {"used": False, "reason": "sklearn unavailable, PCA disabled, or dimensions already small"}

    if USE_DELTA_EMBEDDINGS:
        dZ = np.vstack([np.zeros((1, Z_low.shape[1]), dtype=np.float32), np.diff(Z_low, axis=0)])
        signal = np.concatenate([Z_low, dZ], axis=1)
    else:
        signal = Z_low

    signal = smooth_features(signal)
    info = {
        "raw_embedding_dim": int(Z.shape[1]),
        "signal_dim": int(signal.shape[1]),
        "pca": pca_info,
        "uses_delta_embeddings": USE_DELTA_EMBEDDINGS,
        "smoothing_window": SMOOTHING_WINDOW,
        "interpretation": "learned object-centric + interaction-centric + global-scene DINO embedding signal; PELT+RBF detects distribution/similarity shifts",
    }
    return signal.astype(np.float32), info


# -------------------------
# Change-point detection
# -------------------------
def detect_change_points(video_path: str, total_frames: int, save_crops: bool = False) -> Tuple[List[int], Dict]:
    """Detect rupture boundaries using PELT(model='rbf') on joint DINO embeddings.

    Returns boundary frame indices. Boundaries are internal change points only; 0 and
    total_frames are kept separately in the debug metadata.
    """
    if rpt is None:
        raise RuntimeError(
            "ruptures is required for this version. Install it with `pip install ruptures`. "
            "No fixed-length fallback is used because segmentation must come from change-point detection."
        )

    frame_ids, detections = detect_objects_over_video(video_path)
    with open(DETECTION_DEBUG_PATH, "w") as f:
        json.dump(detections_to_jsonable(frame_ids, detections), f, indent=2)

    Z, stream_arrays, object_names = extract_object_crop_embeddings(video_path, frame_ids, detections, save_crops=save_crops)
    signal, signal_info = prepare_signal_for_ruptures(Z)

    if len(signal) < 6:
        raise RuntimeError(
            f"Too few embedding points for rupture segmentation: {len(signal)}. "
            "Increase video length or reduce DETECTION_STRIDE."
        )

    np.savez_compressed(
        EMBEDDING_DEBUG_PATH,
        frame_ids=np.array(frame_ids[: len(signal)], dtype=np.int32),
        aggregate_embeddings=Z[: len(signal)],
        ruptures_signal=signal,
        object_names=np.array(object_names),
        **{f"stream_{name}": arr[: len(signal)] for name, arr in stream_arrays.items()},
    )

    min_size = 1
    algo = rpt.Pelt(model=RUPTURES_MODEL, min_size=min_size, jump=1).fit(signal)
    cp_indices = algo.predict(pen=RUPTURES_PENALTY)

    # ruptures includes the final endpoint len(signal). Keep only internal changes.
    internal_boundaries = []
    for cp in cp_indices:
        if 0 < cp < len(frame_ids):
            internal_boundaries.append(int(frame_ids[cp]))

    internal_boundaries = sorted(set(b for b in internal_boundaries if 0 < b < total_frames))

    debug = {
        "pipeline": "OWLv2 detection -> object-centric crops + interaction-centric pair crops + global frame -> DINOv2 embeddings -> aggregate -> PELT(model='rbf')",
        "algorithm": f"ruptures.Pelt(model='{RUPTURES_MODEL}', pen={RUPTURES_PENALTY})",
        "rbf_interpretation": "RBF measures nonlinear similarity changes in the joint DINO embedding signal.",
        "embedding_streams": object_names,
        "detection_stride": DETECTION_STRIDE,
        "owl_score_threshold": OWL_SCORE_THRESHOLD,
        "dino_model_id": DINO_MODEL_ID,
        "signal_info": signal_info,
        "min_size_detection_steps": int(min_size),
        "min_size_interpretation": "minimal numerical constraint only; no manual segment duration is imposed",
        "boundary_frames": internal_boundaries,
        "full_timeline_boundaries": [0] + internal_boundaries + [total_frames],
        "uses_global_frame_embedding": USE_GLOBAL_FRAME_EMBEDDING,
        "uses_relative_spatial_deltas": False,
        "spatial_delta_features": [],
        "interaction_embedding_streams": SEGMENTATION_INTERACTION_STREAMS,
    }
    with open(BOUNDARY_DEBUG_PATH, "w") as f:
        json.dump(debug, f, indent=2)

    print(f"Detected {len(internal_boundaries)} internal boundaries using OWLv2 + DINOv2 object/interaction/global embeddings + PELT(RBF).")
    return internal_boundaries, debug


# -------------------------
# VLM prompts / calls
# -------------------------
def build_state_inference_prompt(node_id: str, role: str) -> str:
    return f"""
You are given ONE rupture-defined STATE segment from a scooping demonstration.

This segment represents a visual state region between embedding change boundaries.
Your job is to describe ONLY what is visibly true in this segment.

A state is NOT an action.
Do NOT infer the skill.
Do NOT infer motion direction.
Do NOT infer what happened before this segment.
Do NOT infer what happens after this segment.
Do NOT describe the human action or intent.

Use these controlled state variables and ONLY these allowed values:

1. spoon_occupancy:
   empty / contains_material

2. spoon_to_source:
   inside / near / far

3. spoon_to_target:
   near / far

4. sugar_pile:
   intact / disturbed

5. target_container:
   empty / contains_material

Important:
- Do NOT output unclear.
- Do NOT use values outside the allowed vocabulary.
- Choose the closest visible state from the allowed values.
- Do not use action words such as approach, scoop, lift, transport, dump, return, move, collect, transfer, pour, use, using.
- Do not copy the placeholder string "value".
- visible_conditions must contain the actual selected values, for example "spoon_occupancy=contains_material".
- state_summary must be a compact condition list using the selected values.

Good state_summary:
"spoon_occupancy=contains_material; spoon_to_source=inside; spoon_to_target=far; sugar_pile=disturbed; target_container=empty"

Bad state_summary:
"a person is using a spoon to scoop sugar"
"the spoon is moving toward the bucket"
"spoon_occupancy=value"
"spoon_occupancy=unclear"

Return ONLY valid JSON. Do not use markdown. No text outside JSON.

{{
  "node_id": "{node_id}",
  "state_name": "{node_id}",
  "state_summary": "compact condition list without action verbs",
  "state_variables": {{
    "spoon_occupancy": "empty/contains_material",
    "spoon_to_source": "inside/near/far",
    "spoon_to_target": "near/far",
    "sugar_pile": "intact/disturbed",
    "target_container": "empty/contains_material"
  }},
  "visible_conditions": [
    "spoon_occupancy=actual_value",
    "spoon_to_source=actual_value",
    "spoon_to_target=actual_value",
    "sugar_pile=actual_value",
    "target_container=actual_value"
  ],
  "confidence": 0.0,
  "reason": "brief visual evidence only"
}}
"""


def build_transition_skill_prompt(
    skill_labels: List[str],
    trajectory_id: str,
    pre_state: Optional[Dict] = None,
    post_state: Optional[Dict] = None,
) -> str:
    labels = ", ".join(skill_labels)

    def compact_state_context(state: Optional[Dict]) -> Dict:
        if state is None:
            return {}
        return {
            "state_id": state.get("abstract_node_id") or state.get("state_name"),
            "state_variables": state.get("state_variables", {}),
            "state_summary": state.get("state_summary"),
        }

    pre_context = json.dumps(compact_state_context(pre_state), indent=2)
    post_context = json.dumps(compact_state_context(post_state), indent=2)

    return f"""
You are given PRE and POST structured state variables from a scooping demonstration.

Your job:
Infer exactly ONE primitive skill from the explicit difference between PRE.state_variables and POST.state_variables.

Closed skill vocabulary:
{labels}

PRE:
{pre_context}

POST:
{post_context}

CRITICAL GROUNDING RULES:
- Use ONLY PRE.state_variables and POST.state_variables as primary evidence.
- Boundary images are secondary verification only.
- Boundary images MUST NOT override state_variables.
- Do NOT infer hidden intermediate actions.
- Do NOT infer what happened before PRE.
- Do NOT infer what happens after POST.
- Do NOT invent state changes absent from PRE and POST.
- Do NOT rewrite, reinterpret, or relabel PRE or POST.
- If your reason mentions a PRE or POST value, that value MUST exactly match the provided state_variables.
- If PRE and POST task-relevant variables are the same, output skill_label="none".

Task-relevant variables:
- spoon_occupancy
- spoon_to_source
- spoon_to_target
- sugar_pile
- target_container

Ignore any variable outside the task-relevant variables when choosing the skill.

Compute the semantic delta first:
delta = all task-relevant variables whose PRE value differs from POST value.

If delta is empty:
skill_label = "none"
reason = "No task-relevant PRE→POST state-variable change."

Skill definitions:

approach:
PRE spoon_to_source is far
POST spoon_to_source is near or inside
spoon_occupancy remains empty

scoop:
PRE spoon_occupancy is empty
POST spoon_occupancy is contains_material
Optional support: sugar_pile changes from intact to disturbed
FORBIDDEN if PRE spoon_occupancy is contains_material

lift:
PRE spoon_occupancy is contains_material AND PRE spoon_to_source is inside
POST spoon_occupancy is contains_material AND POST spoon_to_source is near or far
FORBIDDEN if PRE spoon_to_source and POST spoon_to_source are the same

transport:
PRE spoon_occupancy is contains_material
POST spoon_occupancy is contains_material
AND spoon_to_target changes from far to near

dump:
PRE spoon_occupancy is contains_material
AND (
  POST spoon_occupancy is empty
  OR POST target_container is contains_material
)
FORBIDDEN if spoon_occupancy remains contains_material AND target_container does not change to contains_material

return:
PRE spoon_occupancy is empty AND PRE spoon_to_target is near
POST spoon_occupancy is empty AND POST spoon_to_source is near or inside

none:
Choose none if:
- delta is empty
- only non-task-relevant visibility changed
- the change is ambiguous
- the proposed skill would require assuming hidden history

Hard validation before final answer:
1. If PRE spoon_occupancy = contains_material and POST spoon_occupancy = contains_material, skill_label cannot be scoop.
2. If PRE spoon_to_source = POST spoon_to_source, skill_label cannot be lift.
3. If PRE == POST for task-relevant variables, skill_label must be none.
4. Reason must cite actual PRE and POST variable values exactly as provided.
5. Do not say "PRE spoon_occupancy=empty" unless PRE actually has spoon_occupancy="empty".

Return ONLY valid JSON. Do not use markdown. No text outside JSON.

{{
  "trajectory_id": "{trajectory_id}",
  "skill_label": "one_label_from_vocabulary",
  "confidence": 0.0,
  "reason": "explicit PRE→POST state-variable comparison only"
}}
"""


def build_state_equivalence_prompt(node_a_id: str, node_b_id: str) -> str:
    return f"""
You are an expert robot operator assessing whether two visual windows represent
the SAME high-level world state.

Images:
- A: {node_a_id}
- B: {node_b_id}

STATE DEFINITION:
{STATE_HINTS}

Same high-level state means task-relevant conditions are the same, even if pose or lighting differs.
Different high-level state means spoon fill status, source pile condition, target bucket content condition, or task-relevant tool relation changed.

Return ONLY valid JSON:
{{
  "same_state": true,
  "confidence": 0.0,
  "reason": "brief reason"
}}
"""


def load_vlm():
    print("Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(VLM_MODEL_ID)
    return model, processor


def run_vlm_on_images(model, processor, images: List[Image.Image], prompt: str, max_new_tokens: int = 220) -> str:
    content = []
    for image in images:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    del inputs, generated_ids, generated_ids_trimmed
    cleanup_cuda()
    return output_text



def infer_state_window(model, processor, images: List[Image.Image], node_id: str, role: str) -> Dict:
    raw_output = run_vlm_on_images(
        model,
        processor,
        images,
        build_state_inference_prompt(node_id, role),
        max_new_tokens=260,
    )
    parsed = extract_json(raw_output)
    return {
        "raw_vlm_output": raw_output,
        "parsed_vlm_output": parsed,
        "state_name": node_id if parsed is None else parsed.get("state_name", node_id),
        "state_summary": None if parsed is None else parsed.get("state_summary"),
        "state_variables": {} if parsed is None else parsed.get("state_variables", {}),
        "visible_conditions": [] if parsed is None else parsed.get("visible_conditions", []),
        "confidence": 0.0 if parsed is None else float(parsed.get("confidence", 0.0)),
        "reason": None if parsed is None else parsed.get("reason"),
    }



def validate_skill_against_state_variables(
    skill: str,
    pre_state: Optional[Dict],
    post_state: Optional[Dict],
) -> Tuple[str, Optional[str]]:
    """Deterministically correct impossible VLM skill labels using state_variables."""
    pre_vars = (pre_state or {}).get("state_variables", {}) or {}
    post_vars = (post_state or {}).get("state_variables", {}) or {}

    task_keys = [
        "spoon_occupancy",
        "spoon_to_source",
        "spoon_to_target",
        "sugar_pile",
        "target_container",
    ]
    delta = {
        k: (pre_vars.get(k), post_vars.get(k))
        for k in task_keys
        if pre_vars.get(k) != post_vars.get(k)
    }

    if not delta:
        return "none", "corrected_to_none: no task-relevant PRE→POST state-variable change"

    pre_occ = pre_vars.get("spoon_occupancy")
    post_occ = post_vars.get("spoon_occupancy")
    pre_src = pre_vars.get("spoon_to_source")
    post_src = post_vars.get("spoon_to_source")
    pre_tgt = pre_vars.get("spoon_to_target")
    post_tgt = post_vars.get("spoon_to_target")
    pre_target = pre_vars.get("target_container")
    post_target = post_vars.get("target_container")

    if skill == "scoop" and pre_occ == "contains_material" and post_occ == "contains_material":
        return "none", "corrected_to_none: scoop forbidden because PRE and POST both contain material"

    if skill == "lift" and pre_src == post_src:
        return "none", "corrected_to_none: lift forbidden because spoon_to_source did not change"

    if skill == "dump" and post_occ == "contains_material" and post_target != "contains_material":
        return "none", "corrected_to_none: dump forbidden because spoon still contains material and target did not contain material"

    # If the VLM says none but the change has a clear deterministic match, keep none for now.
    # We only correct impossible labels, not force a label.
    return skill, None



def label_transition_skill(
    model,
    processor,
    images: List[Image.Image],
    trajectory_id: str,
    pre_state: Optional[Dict] = None,
    post_state: Optional[Dict] = None,
) -> Dict:
    raw_output = run_vlm_on_images(
        model,
        processor,
        images,
        build_transition_skill_prompt(SKILL_LABELS, trajectory_id, pre_state=pre_state, post_state=post_state),
        max_new_tokens=180,
    )
    parsed = extract_json(raw_output)
    if parsed is None:
        return {"skill_label": "none", "confidence": 0.0, "raw_vlm_output": raw_output, "parsed_vlm_output": None}
    skill = parsed.get("skill_label", "none")
    if skill not in SKILL_LABELS:
        skill = "none"
    corrected_skill, correction_reason = validate_skill_against_state_variables(skill, pre_state, post_state)
    reason = parsed.get("reason")
    if correction_reason is not None:
        reason = f"{correction_reason}; original_vlm_label={skill}; original_reason={reason}"
    return {
        "skill_label": corrected_skill,
        "confidence": float(parsed.get("confidence", 0.0)),
        "reason": reason,
        "raw_vlm_output": raw_output,
        "parsed_vlm_output": parsed,
        "label_source": "vlm_pre_post_state_change_with_boundary_evidence_closed_set_skill_validated",
    }


# -------------------------
# Rupture-segment frame sampling and graph construction
# -------------------------
def sample_frame_indices(start_frame: int, end_frame: int, max_images: int) -> List[int]:
    """Sample up to max_images frames from an inclusive frame interval."""
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame < start_frame:
        return []
    if max_images <= 0:
        return []
    if end_frame == start_frame:
        return [start_frame]
    n = min(max_images, end_frame - start_frame + 1)
    return sorted(set(int(round(x)) for x in np.linspace(start_frame, end_frame, n)))


def load_frames(video_path: str, frame_indices: List[int]) -> Tuple[List[Image.Image], List[int]]:
    cap = cv2.VideoCapture(video_path)
    images, valid = [], []
    for idx in frame_indices:
        image = read_frame_as_pil(cap, idx)
        if image is not None:
            images.append(image)
            valid.append(int(idx))
    cap.release()
    return images, valid


def boundary_windows(
    boundary: int,
    prev_boundary: int,
    next_boundary: int,
    total_frames: int,
) -> Dict[str, Tuple[int, int]]:
    """Return a compact boundary-centered transition evidence window.

    Rupture boundaries define state changes. The transition window is not a
    long skill segment; it is only a small visual neighborhood around the
    boundary used to verify the semantic state change.
    """
    b = int(boundary)
    prev_b = max(0, int(prev_boundary))
    next_b = min(total_frames, int(next_boundary))

    left_len = max(1, b - prev_b)
    right_len = max(1, next_b - b)

    # Compact boundary evidence. This is intentionally smaller than the old
    # 20% window because skills are now inferred primarily from PRE→POST state
    # change, not from a long action clip.
    transition_radius = min(3, max(1, int(round(0.10 * min(left_len, right_len)))))

    trans_start = max(prev_b, b - transition_radius)
    trans_end = min(next_b - 1, b + transition_radius)

    return {
        "transition": (trans_start, trans_end),
        "neighbor_boundaries": (prev_b, b, next_b),
        "adaptive_transition_radius": transition_radius,
    }


def build_windows_from_boundaries(boundaries: List[int], total_frames: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Create rupture-defined state windows and compact transition evidence.

    State windows are the rupture segments themselves:
        S0 = [0, b0-1]
        S1 = [b0, b1-1]
        ...
        Sn = [b_last, total_frames-1]

    Each edge i connects:
        state_i -- skill(boundary_i) --> state_{i+1}

    The skill is inferred from the semantic change between adjacent states.
    Boundary-centered transition images are only verification evidence.
    """
    boundaries = sorted(set(int(b) for b in boundaries if 0 < int(b) < total_frames))
    timeline = [0] + boundaries + [total_frames]

    state_windows: List[Tuple[int, int]] = []
    for i in range(len(timeline) - 1):
        start = timeline[i]
        end = timeline[i + 1] - 1
        if end < start:
            end = start
        state_windows.append((start, min(end, total_frames - 1)))

    transition_windows: List[Tuple[int, int]] = []
    for i, boundary in enumerate(boundaries):
        prev_boundary = timeline[i]
        next_boundary = timeline[i + 2]
        win = boundary_windows(boundary, prev_boundary, next_boundary, total_frames)
        transition_windows.append(win["transition"])

    return transition_windows, state_windows


def save_node_image(image: Image.Image, path: str) -> None:
    image.convert("RGB").save(path)



def strip_runtime_images(graph: Dict) -> Dict:
    for node in graph["abstract_nodes"]:
        node.pop("representative_image", None)
    for node in graph["raw_nodes"]:
        node.pop("image", None)
    return graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build rupture-segment high-level graph using OWLv2 object/interaction crops + DINOv2 + PELT(RBF) + VLM."
    )
    parser.add_argument("--video-path", default=VIDEO_PATH, help="Path to input demonstration video.")
    parser.add_argument("--out-path", default=OUT_PATH, help="Path to output JSON graph.")
    parser.add_argument("--save-crops", action="store_true", help="Save sampled object crops for debugging.")
    parser.add_argument("--penalty", type=float, default=RUPTURES_PENALTY, help="PELT penalty. Higher means fewer boundaries.")
    return parser.parse_args()


def main() -> None:
    cleanup_cuda()
    args = parse_args()

    global RUPTURES_PENALTY
    RUPTURES_PENALTY = args.penalty

    video_path = resolve_video_path(args.video_path)
    out_path = args.out_path

    print(f"Using video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    boundaries, boundary_debug = detect_change_points(video_path, total_frames, save_crops=args.save_crops)
    if not boundaries:
        raise RuntimeError(
            "PELT(RBF) found no internal boundaries. Try lowering --penalty or reducing DETECTION_STRIDE."
        )

    model, processor = load_vlm()

    graph = {
        "raw_nodes": [],
        "abstract_nodes": [],
        "edges": [],
        "pipeline": [
            "OWLv2 open-vocabulary object localization",
            "object/task-centric crop extraction",
            "global-frame extraction for scene layout",
            "object-centric crop extraction for spoon and source sugar pile",
            "interaction-centric pair crop extraction for spoon-source and spoon-target regions",
            "global-frame extraction for scene layout",
            "DINOv2 object, interaction, and global-frame embedding extraction",
            "object-centric + interaction-centric + scene-level feature aggregation",
            "temporal embedding delta construction; optional PCA disabled by default",
            "ruptures PELT with RBF cost on joint embedding signal",
            "rupture-defined state segments and compact boundary evidence windows",
            "VLM state semantics and PRE-to-POST skill inference",
            "symbolic state-skill graph construction",
        ],
        "skill_vocabulary": SKILL_LABELS,
        "state_hints": STATE_HINTS,
        "segmentation_method": "PELT_RBF_on_joint_DINOv2_object_interaction_global_embeddings",
        "boundaries": boundaries,
        "object_detector": {
            "model_id": OWL_MODEL_ID,
            "object_prompts": OWL_OBJECT_PROMPTS,
            "score_threshold": OWL_SCORE_THRESHOLD,
            "detection_stride": DETECTION_STRIDE,
        },
        "embedding_model": {
            "model_id": DINO_MODEL_ID,
            "object_streams": SEGMENTATION_OBJECT_STREAMS,
            "interaction_streams": SEGMENTATION_INTERACTION_STREAMS,
            "uses_global_frame_embedding": USE_GLOBAL_FRAME_EMBEDDING,
            "uses_relative_spatial_deltas": False,
            "spatial_delta_features": [],
            "aggregation": "concatenate DINO embeddings from object-centric crops, interaction-centric pair crops, and optional global frame per sampled frame",
            "uses_delta_embeddings_for_ruptures": USE_DELTA_EMBEDDINGS,
            "pca_dims": PCA_DIMS,
        },
        "change_point_detector": {
            "library": "ruptures",
            "algorithm": "PELT",
            "model": RUPTURES_MODEL,
            "penalty": RUPTURES_PENALTY,
            "input_features": "joint DINOv2 embeddings from OWLv2 object-centric crops, interaction-centric pair crops, and optional global-frame embedding; optional temporal embedding deltas; no explicit geometry features",
            "rbf_interpretation": "RBF measures nonlinear similarity shifts in embedding space; PELT searches for optimal change-point boundaries.",
            "min_size_detection_steps": 1,
            "min_size_interpretation": "minimal numerical constraint only; no manual segment duration is imposed",
        },
        "vlm_reasoning": {
            "model_id": VLM_MODEL_ID,
            "role": "infer structured state variables from rupture segments, then infer and validate skills only from explicit PRE-to-POST state-variable changes",
            "state_windows": "rupture-defined frame segments; each state segment is inferred once",
            "transition_windows": "compact neighborhoods around rupture boundaries; used only to verify PRE-to-POST state-change skill inference",
            "no_handcrafted_state_rules": True,
            "segmentation_uses_explicit_geometry": False,
        },
        "boundary_debug": boundary_debug,
        "state_equivalence_confidence_threshold": STATE_EQUIVALENCE_CONFIDENCE_THRESHOLD,
        "disable_state_merging_for_debug": DISABLE_STATE_MERGING_FOR_DEBUG,
    }

    os.makedirs(NODE_IMAGE_DIR, exist_ok=True)

    transition_windows, state_windows = build_windows_from_boundaries(boundaries, total_frames)

    # Infer each rupture-defined state segment once.
    state_node_ids = []
    for state_idx, state_win in enumerate(state_windows):
        state_id = f"state_{state_idx:04d}"
        raw_id = f"raw_{state_id}"
        state_image_path = f"{NODE_IMAGE_DIR}/{raw_id}.jpg"

        frame_idx = sample_frame_indices(*state_win, max_images=MAX_IMAGES_PER_STATE_WINDOW)
        state_images, state_valid = load_frames(video_path, frame_idx)
        if not state_images:
            print(f"Skipping {state_id}: no valid frames in state window {state_win}.")
            continue

        # Save a representative snapshot from the middle of the state window.
        rep_image = state_images[len(state_images) // 2]
        rep_frame = state_valid[len(state_valid) // 2]
        save_node_image(rep_image, state_image_path)

        state_inference = infer_state_window(
            model,
            processor,
            state_images,
            state_id,
            role="state",
        )

        raw_node = {
            "raw_node_id": raw_id,
            "abstract_node_id": state_id,
            "role": "state",
            "frame_id": rep_frame,
            "window_frames": state_valid,
            "state_window": list(state_win),
            "image_path": state_image_path,
            "state_name": state_inference.get("state_name", state_id),
            "state_summary": state_inference.get("state_summary"),
            "state_variables": state_inference.get("state_variables", {}),
            "visible_conditions": state_inference.get("visible_conditions", []),
            "state_inference": state_inference,
        }
        graph["raw_nodes"].append(raw_node)

        abstract_node = {
            "abstract_node_id": state_id,
            "representative_raw_node_id": raw_id,
            "representative_image_path": state_image_path,
            "representative_frame_id": rep_frame,
            "representative_role": "state",
            "state_window": list(state_win),
            "state_name": state_inference.get("state_name", state_id),
            "state_summary": state_inference.get("state_summary"),
            "state_variables": state_inference.get("state_variables", {}),
            "visible_conditions": state_inference.get("visible_conditions", []),
            "state_inference": state_inference,
            "members": [raw_id],
            "equivalence_checks": [],
        }
        graph["abstract_nodes"].append(abstract_node)
        state_node_ids.append(state_id)

        del state_images
        cleanup_cuda()

    # Infer each skill from PRE->POST state change, using compact transition images as verification.
    for transition_idx, trans_win in enumerate(transition_windows):
        if transition_idx + 1 >= len(state_node_ids):
            print(f"Skipping transition {transition_idx}: missing source/target state.")
            continue

        traj_name = f"tau_{transition_idx:04d}"
        trans_idx = sample_frame_indices(*trans_win, max_images=MAX_IMAGES_PER_TRANSITION_WINDOW)
        trans_images, trans_valid = load_frames(video_path, trans_idx)
        if not trans_images:
            print(f"Skipping {traj_name}: no valid frames in transition window {trans_win}.")
            continue

        pre_state_context = graph["abstract_nodes"][transition_idx]
        post_state_context = graph["abstract_nodes"][transition_idx + 1]

        skill_result = label_transition_skill(
            model,
            processor,
            trans_images,
            traj_name,
            pre_state=pre_state_context,
            post_state=post_state_context,
        )

        graph["edges"].append(
            {
                "trajectory_id": traj_name,
                "boundary_frame": int(boundaries[transition_idx]) if transition_idx < len(boundaries) else None,
                "transition_window": list(trans_win),
                "transition_frame_indices": trans_valid,
                "source": state_node_ids[transition_idx],
                "target": state_node_ids[transition_idx + 1],
                "source_state_summary": pre_state_context.get("state_summary"),
                "target_state_summary": post_state_context.get("state_summary"),
                "source_state_variables": pre_state_context.get("state_variables", {}),
                "target_state_variables": post_state_context.get("state_variables", {}),
                "source_visible_conditions": pre_state_context.get("visible_conditions", []),
                "target_visible_conditions": post_state_context.get("visible_conditions", []),
                "skill_label": skill_result["skill_label"],
                "label_source": skill_result.get("label_source"),
                "confidence": skill_result["confidence"],
                "reason": skill_result.get("reason"),
                "raw_vlm_output": skill_result["raw_vlm_output"],
                "parsed_vlm_output": skill_result["parsed_vlm_output"],
            }
        )

        del trans_images
        cleanup_cuda()

    graph = strip_runtime_images(graph)
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"Saved rupture-segment high-level trajectory graph to {out_path}")


if __name__ == "__main__":
    main()
