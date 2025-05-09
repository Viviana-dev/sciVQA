import csv
import sys
from os import makedirs, path

import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import METRIC_PATH, PREDICITION_PATH, SCORES_PATH
from helpers.data_load import load_datasets


def generate_golden_file():
    validation_ds = load_datasets(validation=True, train=False, test=False)
    golden_file_path = path.join(PREDICITION_PATH, "golden", "golden.json")
    if path.exists(golden_file_path):
        print(f"Golden file already exists at {golden_file_path}")
        return
    else:
        print(f"Creating golden file at {golden_file_path}")
        if not path.exists(path.dirname(golden_file_path)):
            print(f"Creating directory: {path.dirname(golden_file_path)}")
            makedirs(path.dirname(golden_file_path))
        golden_df = pd.DataFrame(columns=["instance_id", "answer"])
        for i, row in validation_ds.iterrows():
            instance_id = row["instance_id"]
            answer = row["answer"]
            qa_type = row["qa_pair_type"]
            figure_type = row["figure_type"]
            golden_df = pd.concat(
                [
                    golden_df,
                    pd.DataFrame(
                        {
                            "instance_id": [instance_id],
                            "answer": [answer],
                            "qa_type": [qa_type],
                            "figure_type": [figure_type],
                        }
                    ),
                ],
                ignore_index=True,
            )
        golden_df.to_json(golden_file_path, orient="records")
        print(f"Golden file created at {golden_file_path}")


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


def compute_evaluation_scores(version: str):
    scores_path = path.join(SCORES_PATH, version)
    if not path.exists(scores_path):
        makedirs(scores_path)

    golden_file_path = path.join(PREDICITION_PATH, "golden", "golden.json")
    if not path.exists(golden_file_path):
        generate_golden_file()

    prediction_foler_path = path.join(PREDICITION_PATH, "predictions")
    prediction_file_path = path.join(prediction_foler_path, "predictions.csv")
    if not path.exists(prediction_file_path):
        raise ValueError(
            f"Prediction file not found at {prediction_file_path}. Please run the prediction script first."
        )

    output_filename = path.join(scores_path, "scores.txt")
    output_file = open(output_filename, "w")

    gold_df = pd.read_json(golden_file_path)
    pred_df = pd.read_csv(prediction_file_path, index_col=0)

    if len(gold_df) != len(pred_df):
        raise ValueError("The lengths of references and predictions do not match.")

    merged: pd.DataFrame = gold_df.merge(pred_df, on="instance_id", how="left")
    references = merged["answer"].tolist()
    predictions = merged["answer_pred"].tolist()

    rouge1_score_f1, rouge1_score_precision, rouge1_score_recall, merged = rouge(
        predictions, references, "rouge1", merged
    )
    rougeL_score_f1, rougeL_score_precision, rougeL_score_recall, merged = rouge(
        predictions, references, "rougeL", merged
    )
    bert_score_f1, bert_score_precision, bert_score_recall, merged = bertS(predictions, references, merged)

    output_file.write("rouge1.f1: " + str(rouge1_score_f1) + "\n")
    print(f"rouge1.f1: {rouge1_score_f1}")
    output_file.write("rouge1.precision: " + str(rouge1_score_precision) + "\n")
    print(f"rouge1.precision: {rouge1_score_precision}")
    output_file.write("rouge1.recall: " + str(rouge1_score_recall) + "\n")
    print(f"rouge1.recall: {rouge1_score_recall}")

    output_file.write("rougeL.f1: " + str(rougeL_score_f1) + "\n")
    print(f"rougeL.f1: {rougeL_score_f1}")
    output_file.write("rougeL.precision: " + str(rougeL_score_precision) + "\n")
    print(f"rougeL.precision: {rougeL_score_precision}")
    output_file.write("rougeL.recall: " + str(rougeL_score_recall) + "\n")
    print(f"rougeL.recall: {rougeL_score_recall}")

    output_file.write("bertS.f1: " + str(bert_score_f1) + "\n")
    print(f"bertS.f1: {bert_score_f1}")
    output_file.write("bertS.precision: " + str(bert_score_precision) + "\n")
    print(f"bertS.precision: {bert_score_precision}")
    output_file.write("bertS.recall: " + str(bert_score_recall) + "\n")
    print(f"bertS.recall: {bert_score_recall}")

    output_file.close()
    list_of_metric_dfs = []

    for figure_type in merged["figure_type"].unique():
        figure_df = merged[merged["figure_type"] == figure_type]
        metric_figure = []
        for qa_type in figure_df["qa_type"].unique():
            qa_df = figure_df[figure_df["qa_type"] == qa_type]
            metric_figure.append(
                {
                    "figure_type": figure_type,
                    "qa_type": qa_type,
                    "rouge1_fmeasure": round(qa_df["rouge1_fmeasure"].mean(), 2),
                    "rouge1_precision": round(qa_df["rouge1_precision"].mean(), 2),
                    "rouge1_recall": round(qa_df["rouge1_recall"].mean(), 2),
                    "rougeL_fmeasure": round(qa_df["rougeL_fmeasure"].mean(), 2),
                    "rougeL_precision": round(qa_df["rougeL_precision"].mean(), 2),
                    "rougeL_recall": round(qa_df["rougeL_recall"].mean(), 2),
                    "bertscore_f1": round(qa_df["bertscore_f1"].mean(), 2),
                    "bertscore_precision": round(qa_df["bertscore_precision"].mean(), 2),
                    "bertscore_recall": round(qa_df["bertscore_recall"].mean(), 2),
                }
            )
        metric_df = pd.DataFrame(metric_figure)
        list_of_metric_dfs.append(metric_df)

    # join the dataframe on one csv and add a headline to every csv table
    with open(path.join(METRIC_PATH, version, "metrics.csv"), "w") as f:
        for i, metric_df in enumerate(list_of_metric_dfs):
            if i != 0:
                f.write("\n")
            f.write(f"Figure Type: {metric_df['figure_type'].iloc[0]}\n")
            f.write(f"QA Type: {metric_df['qa_type'].iloc[0]}\n")
            metric_df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
            f.write("\n")


if __name__ == "__main__":
    compute_evaluation_scores()
