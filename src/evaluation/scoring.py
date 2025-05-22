import csv
import sys
from os import makedirs, path
from pathlib import Path
from typing import Literal

import pandas as pd

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from evaluation.metrics import bertS, rouge
from helpers.constants import METRIC_PATH, PREDICITION_PATH, SCORES_PATH
from helpers.dataset_utils import ChartQAEvaluationDataset, load_datasets
from helpers.logging_utils import setup_logger

logger = setup_logger(__name__)


def generate_golden_file(golden_file_path: str, dataset_name: Literal["scivqa", "chartqa"] = "scivqa"):
    """Generate a golden file for evaluation.
    The golden file contains the expected answers for the validation dataset.
    It is used to compute the evaluation scores.

    Parameters
    ----------
    dataset_name : str default "scivqa"
        The name of the dataset to generate the golden file for.
        Can be either "scivqa" or "chartqa".
    """
    if dataset_name == "scivqa":
        validation_ds: pd.DataFrame = load_datasets(validation=True, train=False, test=False)
    elif dataset_name == "chartqa":
        validation_ds: pd.DataFrame = ChartQAEvaluationDataset(split="validation").table

    if path.exists(golden_file_path):
        logger.info(f"Golden file already exists at {golden_file_path}")
        return
    else:
        logger.info(f"Creating golden file at {golden_file_path}")
        if not path.exists(path.dirname(golden_file_path)):
            logger.info(f"Creating directory: {path.dirname(golden_file_path)}")
            makedirs(path.dirname(golden_file_path))
        golden_df = pd.DataFrame(columns=["instance_id", "answer"])
        if dataset_name == "scivqa":
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
        elif dataset_name == "chartqa":
            for i, row in validation_ds.iterrows():
                instance_id = row["instance_id"]
                answer = row["answer"]
                golden_df = pd.concat(
                    [
                        golden_df,
                        pd.DataFrame(
                            {
                                "instance_id": [instance_id],
                                "answer": [answer],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
        golden_df.to_json(golden_file_path, orient="records")
        logger.info(f"Golden file created at {golden_file_path}")


def compute_evaluation_scores(version: str, dataset_name: Literal["scivqa", "chartqa"] = "scivqa"):
    """Compute evaluation scores for the given version.
    The scores are computed using the ROUGE and BERTScore metrics.
    The scores are saved to a file in the Scores_versions directory.

    Parameters
    ----------
    version : str
        The version of the model to evaluate.
    dataset_name : str default "scivqa"
        The name of the dataset to evaluate. Can be either "scivqa" or "chartqa".
    """

    scores_path = path.join(SCORES_PATH, version)
    if not path.exists(scores_path):
        makedirs(scores_path)

    golden_file_path = path.join(PREDICITION_PATH, "golden", dataset_name, "golden.json")
    if not path.exists(golden_file_path):
        logger.info(f"Golden file not found at {golden_file_path}. Generating a new one.")
        generate_golden_file(golden_file_path=golden_file_path, dataset_name=dataset_name)

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
    output_file.write("rouge1.precision: " + str(rouge1_score_precision) + "\n")
    output_file.write("rouge1.recall: " + str(rouge1_score_recall) + "\n")

    output_file.write("rougeL.f1: " + str(rougeL_score_f1) + "\n")
    output_file.write("rougeL.precision: " + str(rougeL_score_precision) + "\n")
    output_file.write("rougeL.recall: " + str(rougeL_score_recall) + "\n")

    output_file.write("bertS.f1: " + str(bert_score_f1) + "\n")
    output_file.write("bertS.precision: " + str(bert_score_precision) + "\n")
    output_file.write("bertS.recall: " + str(bert_score_recall) + "\n")

    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "ROUGE-1",
                "F1 (%)": round(rouge1_score_f1 * 100, 3),
                "Precision (%)": round(rouge1_score_precision * 100, 3),
                "Recall (%)": round(rouge1_score_recall * 100, 3),
            },
            {
                "Metric": "ROUGE-L",
                "F1 (%)": round(rougeL_score_f1 * 100, 3),
                "Precision (%)": round(rougeL_score_precision * 100, 3),
                "Recall (%)": round(rougeL_score_recall * 100, 3),
            },
            {
                "Metric": "BERTScore",
                "F1 (%)": round(bert_score_f1 * 100, 3),
                "Precision (%)": round(bert_score_precision * 100, 3),
                "Recall (%)": round(bert_score_recall * 100, 3),
            },
        ]
    )
    logger.info("\n%s", metrics_df.to_string(index=False))

    output_file.close()

    if dataset_name == "scivqa":
        # If the Dataset is scivqa we can print metrics based on Graph Type and QA Type
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
        metrics_path = Path(path.join(METRIC_PATH, version, "metrics.csv"))
        if not path.exists(path.dirname(metrics_path)):
            logger.info(f"Creating directory: {path.dirname(metrics_path)}")
            makedirs(path.dirname(metrics_path))
        with open(metrics_path, "w") as f:
            for i, metric_df in enumerate(list_of_metric_dfs):
                if i != 0:
                    f.write("\n")
                f.write(f"Figure Type: {metric_df['figure_type'].iloc[0]}\n")
                f.write(f"QA Type: {metric_df['qa_type'].iloc[0]}\n")
                metric_df.to_csv(f, index=False, quoting=csv.QUOTE_ALL)
                f.write("\n")
