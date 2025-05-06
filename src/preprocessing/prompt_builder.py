def build_prompt(
    question: str,
    qa_pair_type: str,
    ocr_boxes_norm: list[tuple[float, float, float, float, str]],
    table_str: str | None = None,
) -> str:
    """Return a complete user prompt with region tags and dynamic instructions."""

    # --- Region tags ---------------------------------------------------
    region_lines = []
    for x1, y1, x2, y2, text in ocr_boxes_norm:
        region_lines.append(f'<box>({x1},{y1},{x2},{y2}): "{text}"</box>')
    region_block = "\n".join(region_lines)

    # --- Table block ----------------------------------------------------
    table_block = f"\nDetected table:\n{table_str}" if table_str else ""

    # --- Dynamic instructions -----------------------------------------
    prompt = (
        f"You are looking at a {qa_pair_type} type chart.\n"
        f"The following chart regions were detected:\n{region_block}{table_block}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    return prompt.strip()
