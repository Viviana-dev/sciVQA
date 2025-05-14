from evaluate import load
from rouge_score import rouge_scorer


def rouge(predictions: list[str], references: list[str], r_type: str = "", merged=None):
    precision = []
    recall = []
    f1 = []

    scorer = rouge_scorer.RougeScorer([r_type], use_stemmer=True)
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        precision.append(score[r_type].precision)
        recall.append(score[r_type].recall)
        f1.append(score[r_type].fmeasure)

    if merged is not None:
        merged[f"{r_type}_precision"] = precision
        merged[f"{r_type}_recall"] = recall
        merged[f"{r_type}_fmeasure"] = f1

    f1 = sum(f1) / len(f1)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    return f1, precision, recall, merged


def bertS(predictions: list[str], references: list[str], merged=None):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    precision = results["precision"]
    recall = results["recall"]
    f1 = results["f1"]
    if merged is not None:
        merged["bertscore_precision"] = precision
        merged["bertscore_recall"] = recall
        merged["bertscore_f1"] = f1

    f1 = sum(results["f1"]) / len(results["f1"])
    precision = sum(results["precision"]) / len(results["precision"])
    recall = sum(results["recall"]) / len(results["recall"])
    return f1, precision, recall, merged
