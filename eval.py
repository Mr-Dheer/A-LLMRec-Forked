import argparse
import numpy as np


# ---------------------------------------------------------------------------
# ID-prediction evaluation
# ---------------------------------------------------------------------------

def evaluate_id(results_file):
    """
    Compute Hit@1 from the ID-prediction output file.

    The file written by generate_id() has lines of the form:
        Target: <int> | Predicted: <int> | Hit: <0 or 1>

    Returns (hit_at_1, total_count).
    """
    total = 0
    hits = 0
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Target:'):
                parts = {kv.split(':')[0].strip(): kv.split(':')[1].strip()
                         for kv in line.split('|')}
                hits += int(parts.get('Hit', 0))
                total += 1
    if total == 0:
        return 0.0, 0
    return hits / total, total


# ---------------------------------------------------------------------------
# Text-generation evaluation
# ---------------------------------------------------------------------------

def get_answers_predictions(file_path):
    answers = []
    llm_predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'Answer:' == line[:len('Answer:')]:
                answer = line.replace('Answer:', '').strip()[1:-1].lower()
                answers.append(answer)
            if 'LLM:' == line[:len('LLM:')]:
                # Use replace('LLM:', '', 1) to avoid clobbering 'LLM' in titles.
                llm_prediction = line.replace('LLM:', '', 1).strip().lower()
                # Format A (SmolVLM-2B with chat template): LLM: "title"
                # Format B (SmolVLM-500M without template): LLM: title"
                # For format A, first and last " differ → extract between them.
                # For format B, only a trailing " → strip it directly.
                start = llm_prediction.find('"')
                end = llm_prediction.rfind('"')
                if start != -1 and end != -1 and start != end:
                    # Format A: text between first and last quote.
                    llm_prediction = llm_prediction[start + 1:end]
                elif llm_prediction.endswith('"'):
                    # Format B: just a trailing quote, strip it.
                    llm_prediction = llm_prediction[:-1]
                llm_predictions.append(llm_prediction.strip())

    return answers, llm_predictions


def evaluate(answers, llm_predictions, k=1):
    NDCG = 0.0
    HT = 0.0
    predict_num = len(answers)
    print(predict_num)
    for answer, prediction in zip(answers, llm_predictions):
        if k > 1:
            rank = prediction.index(answer)
            if rank < k:
                NDCG += 1 / np.log2(rank + 1)
                HT += 1
        elif k == 1:
            if answer in prediction:
                NDCG += 1
                HT += 1

    return NDCG / predict_num, HT / predict_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file',
        type=str,
        default='./recommendation_output.txt',
        help='Path to the text-generation output file',
    )
    parser.add_argument(
        '--id_file',
        type=str,
        default='./recommendation_output_id.txt',
        help='Path to the ID-prediction output file (used with --id)',
    )
    parser.add_argument(
        '--id',
        action='store_true',
        help='Evaluate ID-prediction results instead of text-generation results',
    )
    args = parser.parse_args()

    if args.id:
        hit, count = evaluate_id(args.id_file)
        print(f"ID-prediction Hit@1: {hit:.4f}  ({int(hit * count)}/{count})")
    else:
        answers, llm_predictions = get_answers_predictions(args.file)
        print(len(answers), len(llm_predictions))
        assert len(answers) == len(llm_predictions)

        ndcg, ht = evaluate(answers, llm_predictions, k=1)
        print(f"ndcg at 1: {ndcg}")
        print(f"hit at 1: {ht}")