import ast
import re


def parse_qa_types(qa_type_raw: str) -> set[str]:
    """
    Parse the QA type string to identify the types of questions.
    The function looks for specific tokens in the input string and returns a set of identified types.

    Parameters
    ----------
    qa_type_raw : str
        The raw QA type string to be parsed.

    Returns
    -------
    set[str]
        A set of identified QA types based on the input string.
    """
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


def build_dynamic_prompt(entry: dict) -> str:
    """
    Build a dynamic prompt for the model based on the provided entry.
    The prompt includes information about the figure type, caption, question,
    and specific instructions based on the QA type.
    The function also includes reasoning steps for the model to follow.

    Parameters
    ----------
    entry : dict
        A dictionary containing the information needed to build the prompt.
        Based on the Dataset format from SciVQA

    Returns
    -------
    str
        The constructed prompt string.
    """
    question = entry["question"]
    qa_type_raw = entry["qa_pair_type"]
    caption = entry.get("caption", "")
    figure_type = entry.get("figure_type", "figure")
    compound = entry.get("compound", False)
    figs_numb = entry.get("figs_numb", 0)
    answer_options = entry.get("answer_options", "")

    qa_types = parse_qa_types(qa_type_raw)

    prompt = f"You are looking at a {figure_type}"
    if compound:
        prompt += f" with {figs_numb} subfigures"
    prompt += "."

    if caption:
        prompt += f"\nThe caption is: '{caption}'."

    prompt += f"\nQuestion: {question}"

    if "visual" in qa_types:
        prompt += "\n[Visual cue] Pay attention to color, position, shape, size, height, or direction."
    elif "non-visual" in qa_types:
        prompt += "\n[Data-only cue] Focus your response more on numeric or textual values."
    prompt += "\nUse information from the caption when it directly supports your answer; otherwise, focus on data present in the visual itself."
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
            prompt += "\nRespond only with the corresponding option keyword(s) (e.g., 'A' or 'A,B' if multiple apply, without space between)."
            prompt += "\nDo not include explanations, full sentences, or option text."

    prompt += "\nIf the answer cannot be inferred from the figure and caption, please reply with the sentence: 'It is not possible to answer this question based only on the provided data.'"

    prompt += (
        "\n---\n"
        "<thinking> Reasoning (do NOT respond yet)\n"
        "Step 1 Identify the figure type and its axes/legend.\n"
        "Step 2 Locate the graphical elements relevant to the question.\n"
        "Step 3 Extract the key-value information.\n"
        "Step 4 Determine the required values or qualitative trends.\n"
        "Step 5 Integrate insights from the caption when necessary.\n"
        f"Step 6 {'Evaluate the provided answer choices and select the best one' if 'finite answer set' in qa_types and 'binary' not in qa_types else 'Ensure the answer is either Yes or No as required'}\n"
        "Step 7 Produce the concise answer following the formatting rules above.\n"
        "---\n"
        "Final respond:\n"
        "<answer>\n"
    )

    return prompt.strip()


def build_general_prompt(entry: dict) -> str:
    """
    Build a more general prompt for the model based on the provided entry.
    The function also includes reasoning steps for the model to follow.

    Parameters
    ----------
    entry : dict
        A dictionary containing the information needed to build the prompt.
        Based on the Dataset format from SciVQA
        - "question" (str): The question to be answered.
        - "answer_options" (list[dict]): The available answer options (if any).
            -> e.g [{"A": "The blue line","B": null},{"A": null,"B": "The red line"}]

    Returns
    -------
    str
        The constructed prompt string.
    """
    question = entry["question"]
    answer_options = entry.get("answer_options", "")

    prompt = (
        "You are looking at one or more charts or graphs.\n"
        "While inspecting the visual, pay attention to: color, position, shape, size, height, direction, and any numeric values on axes, legends, or labels.\n"
        "Use the caption only if it clarifies the figure; otherwise rely on the visual itself.\n"
        "\n"
        "Answer format:\n"
        "- Yes/No question -> reply 'Yes' or 'No' only.\n"
        "- Multiple-choice question -> reply with the capital letter(s) of the chosen option(s) (e.g. `A` or `A,B`, no spaces).\n"
        "- Numeric answer -> digits only, include any units or symbols (e.g., %, kg, $).\n"
        "- If the answer cannot be inferred -> reply exactly: 'It is not possible to answer this question based only on the provided data.'\n"
        "- Please be concise and avoide explenations or reasoning in your final answer.\n"
    )

    if answer_options and answer_options != "[]":
        parsed_options = ast.literal_eval(answer_options)
        options = {k: v for d in parsed_options for k, v in d.items()}
        prompt += f"\nAvailable options: {options}."

    prompt += (
        f"\n\nQuestion: {question}\n"
        "\n---\n"
        "<thinking> Reasoning (do NOT respond yet)\n"
        "1. Identify the chart type, axes, and legend.\n"
        "2. Locate the graphical elements relevant to the question.\n"
        "3. Extract the key values or qualitative trends.\n"
        "4. Integrate helpful details from the caption (if any).\n"
        "5. If multiple choice, match your finding to the option(s); if yes/no, decide 'Yes' or 'No'.\n"
        "6. Produce the concise answer following the formatting rules above.\n"
        "---\n"
        "Final respond:\n"
        "<answer>\n"
    )

    return prompt.strip()
