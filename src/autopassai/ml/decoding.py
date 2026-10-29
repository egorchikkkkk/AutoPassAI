from autopassai.ml.charset import IDX_TO_CHAR


def decode_predictions(logits):
    # [T, B, C] → [B, T]
    predictions = logits.argmax(dim=2).transpose(0, 1).cpu().tolist()

    texts = []

    for sequence in predictions:
        characters = []
        previous = None

        for index in sequence:
            # Delete neighbour repeats and skip 'blank'
            if index != previous and index != 0:
                characters.append(IDX_TO_CHAR[index])

            previous = index

        texts.append("".join(characters))

    return texts
