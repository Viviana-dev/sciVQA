import sys
from os import makedirs, path

import pandas as pd
from evaluate import load
from rouge_score import rouge_scorer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from helpers.constants import PREDICITION_PATH
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
            golden_df = pd.concat(
                [golden_df, pd.DataFrame({"instance_id": [instance_id], "answer": [answer]})], ignore_index=True
            )
        golden_df.to_json(golden_file_path, orient="records")
        print(f"Golden file created at {golden_file_path}")


def rouge(predictions: list[str], references: list[str], r_type: str = ""):
    precision = []
    recall = []
    f1 = []
    scorer = rouge_scorer.RougeScorer([r_type], use_stemmer=True)
    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)
        precision.append(score[r_type].precision)
        recall.append(score[r_type].recall)
        f1.append(score[r_type].fmeasure)

    f1 = sum(f1) / len(f1)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    return f1, precision, recall


def bertS(predictions: list[str], references: list[str]):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")

    f1 = sum(results["f1"]) / len(results["f1"])
    precision = sum(results["precision"]) / len(results["precision"])
    recall = sum(results["recall"]) / len(results["recall"])
    return f1, precision, recall


def main():
    output_file_path = path.join(PREDICITION_PATH, "output")
    if not path.exists(output_file_path):
        print(f"Creating directory: {output_file_path}")
        makedirs(output_file_path)
    golden_file_path = path.join(PREDICITION_PATH, "golden", "golden.json")
    if not path.exists(golden_file_path):
        generate_golden_file()

    prediction_file_path = path.join(PREDICITION_PATH, "predictions", "predictions.csv")
    if not path.exists(prediction_file_path):
        raise ValueError(
            f"Prediction file not found at {prediction_file_path}. Please run the prediction script first."
        )

    output_filename = path.join(output_file_path, "scores.txt")
    output_file = open(output_filename, "w")

    gold_df = pd.read_json(golden_file_path)
    pred_df = pd.read_csv(prediction_file_path, index_col=0)

    if len(gold_df) != len(pred_df):
        raise ValueError("The lengths of references and predictions do not match.")

    merged = gold_df.merge(pred_df, on="instance_id", how="left")
    references = merged["answer"].tolist()
    predictions = merged["answer_pred"].tolist()

    rouge1_score_f1, rouge1_score_precision, rouge1_score_recall = rouge(predictions, references, "rouge1")
    rougeL_score_f1, rougeL_score_precision, rougeL_score_recall = rouge(predictions, references, "rougeL")
    bert_score_f1, bert_score_precision, bert_score_recall = bertS(predictions, references)

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


if __name__ == "__main__":
    main()
