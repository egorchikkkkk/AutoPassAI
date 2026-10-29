def edit_distance(reference, prediction):
    # Levenshtein distance:
    # minimal count of insertions, deletes, substitutions
    previous = list(range(len(prediction) + 1))

    for i, ref_char in enumerate(reference, start=1):
        current = [i]

        for j, pred_char in enumerate(prediction, start=1):
            current.append(
                min(
                    previous[j] + 1,  # delete
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_char != pred_char),  # substitution
                )
            )

        previous = current

    return previous[-1]


def evaluate_metrics(references, predictions):
    if len(references) != len(predictions):
        raise ValueError(
            f"References size ({len(references)}) should be equal to Predictions size ({len(predictions)})."
        )

    if len(references) == 0:
        raise ValueError("Size of array should be more than 0.")

    fully_correct_cnt = 0
    total_errors = 0
    lengthes_sum = 0

    for ref, pred in zip(references, predictions, strict=True):
        fully_correct_cnt += int(ref == pred)
        total_errors += edit_distance(ref, pred)
        lengthes_sum += len(ref)

    accuracy = fully_correct_cnt / len(references)
    if lengthes_sum == 0:
        raise ValueError("All of Refrences examples are empty.")
    cer = total_errors / lengthes_sum

    return accuracy, cer
