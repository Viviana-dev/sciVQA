import ast
import random
import re
import sys
from os import path
from pathlib import Path

from PIL import Image

from preprocessing.ocr_utils import visualize_ocr_boxes

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from helpers.data_load import load_real_image_path


def parse_qa_types(qa_type_raw: str) -> set[str]:
    qa_str = str(qa_type_raw).lower()

    ordered_tokens = [
        "closed-ended",
        "unanswerable",
        "infinite answer set",
        "finite answer set",
        "non-binary",
        "binary",
        "non-visual",
        "visual",
    ]

    found = set()
    for token in ordered_tokens:
        # match token as a whole word; allow spaces or semicolons as separators
        pattern = r"(?:^|[\s;])" + re.escape(token) + r"(?:[\s;]|$)"
        if re.search(pattern, qa_str):
            found.add(token)
            # strip out the matched portion to prevent nested matches
            qa_str = re.sub(pattern, " ", qa_str, count=1)

    return found


def build_dynamic_prompt(entry, split="validation", save_sample_path: Path = None):
    instance_id = entry["instance_id"]
    image_path = entry["image_file"]
    question = entry["question"]
    qa_type_raw = entry["qa_pair_type"]
    caption = entry.get("caption", "")
    figure_type = entry.get("figure_type", "figure")
    compound = entry.get("compound", False)
    figs_numb = entry.get("figs_numb", 0)
    answer_options = entry.get("answer_options", "")
    region_block = entry.get("region_block", None)
    random_state: bool = random.choice([True, False])

    qa_types = parse_qa_types(qa_type_raw)

    prompt = f"You are looking at a {figure_type}"
    if compound:
        prompt += f" with {figs_numb} subfigures"
    prompt += "."

    if region_block:
        prompt += f"\nThe following chart regions were detected with OCR (they may contain" " errors): \n{region_block}"

    if caption:
        prompt += f"\nThe caption is: '{caption}'."

    prompt += f"\nQuestion: {question}"

    if "visual" in qa_types:
        prompt += "\n[Visual cue] Pay attention to color, position, shape, size, height, or direction."
    elif "non-visual" in qa_types:
        prompt += "\n[Data-only cue] Focus your response more on numeric or textual values."
    prompt += "\nPlease also consider the caption of the figure to respond to the question."

    if "infinite answer set" in qa_types:
        prompt += (
            "\nRespond with a concise, one-word or very short phrase. No full sentences, no explanations."
            "\nIf the response is numeric, use digits only and include any units or suffixes (e.g., %, kg, $)."
        )
    elif "finite answer set" in qa_types:
        if "binary" in qa_types:
            prompt += "\nPlease answer with 'Yes' or 'No' only."
        else:
            parsed_options = ast.literal_eval(answer_options)
            options = {k: v for d in parsed_options for k, v in d.items()}
            prompt += f"\nAvailable options: {options}."
            prompt += "\nRespond only with the corresponding option keyword(s) (e.g., 'A' or 'A,B' if multiple apply)."
            prompt += "\nDo not include explanations, full sentences, or option text."

    prompt += (
        "\nIf the answer cannot be inferred from the figure and caption, please reply with the sentence: 'It is not possible to answer this question based only on the provided data.'"
        "\n\nResponse:"
    )

    if random_state and save_sample_path:
        save_path = Path(path.join(save_sample_path, instance_id))
        save_path.mkdir(parents=True, exist_ok=True)
        prompt_file_path = path.join(save_path, "prompt.txt")
        image_file_path = path.join(save_path, "image.png")
        root_image_path = load_real_image_path(image_path, **{split: True})
        image = Image.open(root_image_path).convert("RGB")
        image.save(image_file_path)
        with open(prompt_file_path, "w", encoding="utf-8") as f:
            f.write(f"QA-Type: {qa_type_raw}")
            f.write(f"Figure type: {figure_type}")
            f.write(f"\n\n{prompt}")

    return prompt.strip(), random_state
