import json
import re
import cv2
import torch
import gc
from PIL import Image
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
VIDEO_PATH = "training_data/demo1.mp4"
OUT_PATH = "trajectory_graph_compacted.json"


# Manual skill segments: (start_frame, end_frame, expected_skill_name)
# Adjust these after watching demo.mp4
MANUAL_SKILL_SEGMENTS = [
    (0, 60, "approach"),
    (60, 120, "scoop"),
    (120, 180, "lift"),
    (180, 260, "transport"),
    (260, 340, "dump"),
]

USE_MANUAL_SEGMENTS = False

# Each trajectory segment tau is assumed to contain one dominant skill.
SEGMENT_LENGTH_FRAMES = 120
NUM_FRAMES_PER_SEGMENT = 8
IMAGE_SIZE = (224, 224)

# Human-provided skill vocabulary Lambda.
SKILL_LABELS = [
    "approach",
    "scoop",
    "lift",
    "transport",
    "dump",
    "none",
]

# Domain-specific state definition/hints for scooping.
STATE_HINTS = """
A high-level state definition is defined by:
- the spoon position relative to the brown sugar
- the spoon position relatvie to the container with brown sugar
- whether the spoon contains brown sugar
- the shape, height, and disturbance of the sugar pile
- whether sugar has been deposited into the black bucket

Ignore:
- lighting changes
- small particle-level differences
- camera noise
- tiny pose differences that do not change the task-level state
"""

# Bisimulation/state-merging threshold.
# A candidate node is merged with an existing abstract node if the VLM says same_state=true
# and confidence >= this value.
STATE_EQUIVALENCE_CONFIDENCE_THRESHOLD = 0.70


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
# VLM prompts
# -------------------------
def build_skill_prompt(skill_labels, trajectory_id):
    labels = ", ".join(skill_labels)

    return f"""
You are an expert robot operator analyzing one phase of a tool-used demonstration

This trajectory segment contains exactly one dominant skill.

STATE DEFINITION:
{STATE_HINTS}

TASK:
1. Inspect the whole sequence of images.
2. Identify the single dominant skill performed across the sequence.
3. Choose exactly one label from the allowed skill vocabulary.

IMPORTANT TEMPORAL REASONING:
You are analyzing images in chronological order from the beginning to the end of the segment.
Focus on how the tool moves and how the material changes over time.

Use these definitions:
- approach: tool moves toward the material pile but does not scoop yet
- scoop: tool enters or pushes through the material pile to collect material
- lift: tool containing or contacting material moves upward away from the pile
- transport: tool moves horizontally/sideways away from the pile while carrying material
- dump: material falls into a location away from the pile
- none: no clear meaningful skill occurs

Do NOT label every segment as scoop just because the tool is near the pile.
Choose the label based on the dominant motion across the whole sequence.


Allowed skill labels:
{labels}

Return ONLY valid JSON with all keys present:
{{
  "trajectory_id": "{trajectory_id}",
  "skill_label": "one_label_from_allowed_list",
  "confidence": 0.0,
  "reason": "brief visual reason"
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


Two images are DIFFERENT states ONLY if they correspond to different stages of the task (e.g., before scooping vs after scooping).

Small differences in tool position, partial motion, or pile shape should still be considered the SAME state.

Return ONLY valid JSON with all keys present:
{{
  "same_state": true,
  "confidence": 0.0,
  "reason": "brief reason"
}}
"""


# -------------------------
# calling the function
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
    prompt = build_skill_prompt(SKILL_LABELS, trajectory_id)
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
    if skill_label not in SKILL_LABELS:
        skill_label = "none"

    return {
        "skill_label": skill_label,
        "confidence": float(parsed.get("confidence", 0.0)),
        "raw_vlm_output": raw_output,
        "parsed_vlm_output": parsed,
    }


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
# Video Segmentation
# -------------------------
def sample_segment_frames(video_path, start_frame, end_frame, num_frames):
    cap = cv2.VideoCapture(video_path)

    frame_indices = [
        int(start_frame + i * (end_frame - start_frame - 1) / max(num_frames - 1, 1))
        for i in range(num_frames)
    ]

    images = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(frame_rgb))

    cap.release()
    return images, frame_indices


def save_node_image(image, path):
    image.convert("RGB").save(path)


# -------------------------
# graph construction
# -------------------------
def find_or_create_abstract_node(model, processor, abstract_nodes, candidate_node):
    """
    Implements VLM-based state merging.

    candidate_node is merged into an existing abstract node if the VLM judges the two
    visual snapshots to be the same high-level state.
    """
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

        if (
            result["same_state"]
            and result["confidence"] >= STATE_EQUIVALENCE_CONFIDENCE_THRESHOLD
        ):
            abstract_node["members"].append(candidate_node["raw_node_id"])
            return abstract_node["abstract_node_id"], True

    # No equivalent abstract node found, so create a new abstract state.
    abstract_id = f"state_{len(abstract_nodes):04d}"
    abstract_nodes.append(
        {
            "abstract_node_id": abstract_id,
            "representative_raw_node_id": candidate_node["raw_node_id"],
            "representative_image_path": candidate_node["image_path"],
            "representative_frame_id": candidate_node["frame_id"],
            "representative_role": candidate_node["role"],
            "representative_image": candidate_node["image"],
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


def main():
    cleanup_cuda()
    print("Loading model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    graph = {
        "raw_nodes": [],
        "abstract_nodes": [],
        "edges": [],
        "skill_vocabulary": SKILL_LABELS,
        "state_hints": STATE_HINTS,
        "state_equivalence_confidence_threshold": STATE_EQUIVALENCE_CONFIDENCE_THRESHOLD,
    }

    trajectory_id = 0

    if USE_MANUAL_SEGMENTS:
        segment_iter = MANUAL_SKILL_SEGMENTS
    else:
        segment_iter = [
           (start, min(start + SEGMENT_LENGTH_FRAMES, total_frames), None)
           for start in range(0, total_frames, SEGMENT_LENGTH_FRAMES)
    ]
    prev_post_state_id = None

    for start, end, expected_skill in segment_iter:
        start = max(0, start)
        end = min(end, total_frames)

        if end - start < 5:
            continue

        traj_name = f"tau_{trajectory_id:04d}"
        print(f"Processing trajectory {traj_name}: frames {start}-{end}")

        images, frame_indices = sample_segment_frames(
            VIDEO_PATH,
            start,
            end,
            NUM_FRAMES_PER_SEGMENT,
        )

        if len(images) < 2:
            continue

        # Raw visual snapshots before and after the skill execution.
        pre_raw_id = f"raw_node_{trajectory_id:04d}_pre"
        post_raw_id = f"raw_node_{trajectory_id:04d}_post"

        os.makedirs("node_images", exist_ok=True)

        pre_image_path = f"node_images/{pre_raw_id}.jpg"
        post_image_path = f"node_images/{post_raw_id}.jpg"
        save_node_image(images[0], pre_image_path)
        save_node_image(images[-1], post_image_path)

        pre_candidate = {
            "raw_node_id": pre_raw_id,
            "trajectory_id": traj_name,
            "role": "pre_state",
            "frame_id": frame_indices[0],
            "image_path": pre_image_path,
            "image": images[0],
        }

        post_candidate = {
            "raw_node_id": post_raw_id,
            "trajectory_id": traj_name,
            "role": "post_state",
            "frame_id": frame_indices[-1],
            "image_path": post_image_path,
            "image": images[-1],
        }

        graph["raw_nodes"].append(pre_candidate)
        graph["raw_nodes"].append(post_candidate)

        # Paper-style step (i): identify/merge high-level states before and after skill.
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

        # Force sequential connection for manually segmented single video
        if USE_MANUAL_SEGMENTS and prev_post_state_id is not None:
            pre_state_id = prev_post_state_id

        prev_post_state_id = post_state_id

        # assign a skill label to the directed transition edge.
        skill_result = label_skill(model, processor, images, traj_name)

        graph["edges"].append(
            {
                "trajectory_id": traj_name,
                "source": pre_state_id,
                "target": post_state_id,
                "raw_source": pre_raw_id,
                "raw_target": post_raw_id,
                "source_was_merged": pre_was_merged,
                "target_was_merged": post_was_merged,
                "expected_skill_label": expected_skill,
                "skill_label": skill_result["skill_label"],
                "confidence": skill_result["confidence"],
                "raw_vlm_output": skill_result["raw_vlm_output"],
                "parsed_vlm_output": skill_result["parsed_vlm_output"],
            }
        )

        del images
        cleanup_cuda()
        trajectory_id += 1

    graph = strip_runtime_images(graph)

    with open(OUT_PATH, "w") as f:
        json.dump(graph, f, indent=2)

    print(f"Saved compacted trajectory graph to {OUT_PATH}")


if __name__ == "__main__":
    main()
