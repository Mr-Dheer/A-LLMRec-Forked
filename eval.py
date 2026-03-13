import numpy as np


# ---------------------------------------------------------------------------
# ID-prediction evaluation  (new)
# ---------------------------------------------------------------------------

def evaluate_id(results_file='./recommendation_output_id.txt'):
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
# Original text-generation evaluation (unchanged)
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
                llm_prediction = line.replace('LLM', '').strip().lower()
                try:
                    llm_prediction = llm_prediction.replace("\"item title\" : ", '')
                    start = llm_prediction.find('"')
                    end = llm_prediction.rfind('"')

                    if (start + end < start) or (start + end < end):
                        print(1/0)
                        
                    llm_prediction = llm_prediction[start+1:end]
                except Exception as e:
                    print()
                    
                llm_predictions.append(llm_prediction)
                
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
    import sys

    if '--id' in sys.argv:
        # Evaluate ID-prediction results.
        hit, count = evaluate_id('./recommendation_output_id.txt')
        print(f"ID-prediction Hit@1: {hit:.4f}  ({int(hit*count)}/{count})")
    else:
        # Evaluate text-generation results (original behaviour).
        inferenced_file_path = './recommendation_output.txt'
        answers, llm_predictions = get_answers_predictions(inferenced_file_path)
        print(len(answers), len(llm_predictions))
        assert len(answers) == len(llm_predictions)

        ndcg, ht = evaluate(answers, llm_predictions, k=1)
        print(f"ndcg at 1: {ndcg}")
        print(f"hit at 1: {ht}")