ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"

CHAR_TO_IDX = {char: index for index, char in enumerate(ALPHABET, start=1)}

IDX_TO_CHAR = {index: char for char, index in CHAR_TO_IDX.items()}

NUM_CLASSES = len(ALPHABET) + 1
