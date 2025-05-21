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
