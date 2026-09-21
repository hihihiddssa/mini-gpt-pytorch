class CharTokenizer:
    """One token per Unicode character. Vocabulary comes from the training text."""

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {char: i for i, char in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"Characters absent from training vocabulary: {''.join(sorted(unknown))!r}")
        return [self.stoi[char] for char in text]

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in ids)
