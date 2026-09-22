"""最简单的字符级 tokenizer：一个 Unicode 字符对应一个 token ID。"""


class CharTokenizer:
    """保存字符表，并负责「字符 ↔ 整数 ID」的双向转换。"""

    def __init__(self, chars):
        # set(chars) 找出文本中出现过的不同字符，自动去重。
        # 例如 chars="banana"，最终只剩 b、a、n 各一份。
        # sorted(...) 再按 Unicode 顺序排序，保证同一份语料得到同一套 ID。
        # self.chars 是「ID → 字符」的表：第 i 个元素就是 ID=i 的字符。
        # 空格、标点和换行也都是字符，也会各自占一个 ID。
        self.chars = sorted(set(chars))

        # enumerate 为每个字符提供从 0 开始的下标 i。
        # 字典推导式建立反向查询表「字符 → ID」，如 {'a': 0, 'b': 1}。
        self.stoi = {char: i for i, char in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        """根据训练文本建立 tokenizer；这里返回的是对象，还不是 ID 列表。"""
        # cls(text) 会调用上面的 __init__(text)，生成字符表。
        return cls(text)

    @property
    def vocab_size(self):
        """词表大小 V，即模型每个位置要给多少个字符分别打分。"""
        # @property 允许写 tokenizer.vocab_size，不必写 vocab_size()。
        return len(self.chars)

    def encode(self, text):
        """把字符串转换成整数列表，例如「ab」→ [0, 1]。"""
        # set(text) 是输入出现过的字符；set(self.stoi) 是训练词表中的字符。
        # 两个集合相减后，unknown 就是模型从未见过、无法编码的字符。
        unknown = set(text) - set(self.stoi)
        if unknown:
            # !r 让错误信息显示特殊字符的形式，例如换行显示为 '\n'。
            raise ValueError(
                f"Characters absent from training vocabulary: {''.join(sorted(unknown))!r}"
            )
        # 逐字符查字典。这里返回普通 Python 列表；train.py 随后把它变成张量。
        return [self.stoi[char] for char in text]

    def decode(self, ids):
        """把 ID 序列转换回文字，供我们阅读生成结果。"""
        # int(i) 也兼容 PyTorch 的标量 tensor，例如 tensor(3)。
        return "".join(self.chars[int(i)] for i in ids)
