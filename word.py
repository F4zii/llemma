class Word(tuple):
    def __new__(cls, subtokens: tuple[bytes, ...], is_special: bool = False):
        # Pass subtokens directly to tuple.__new__
        instance = super().__new__(cls, subtokens)
        instance.is_special = is_special
        return instance

    def __repr__(self):
        prefix = "Special " if self.is_special else ""
        return f"{prefix}Word({b', '.join(self)})"

    def __new__(cls, subtokens: tuple[bytes, ...], is_special: bool = False):
        instance = super().__new__(cls, subtokens)
        instance.is_special = is_special
        return instance

    def merge(self, pair: tuple[bytes, bytes]) -> "Word":
        if self.is_special or len(self) < 2:
            return self

        first, second = pair
        if first not in self:
            return self

        new_subtokens = []
        i = 0
        n = len(self)

        while i < n:
            if i < n - 1 and self[i] == first and self[i + 1] == second:
                new_subtokens.append(first + second)
                i += 2  # Skip BOTH merged elements (non-overlapping)
            else:
                new_subtokens.append(self[i])
                i += 1

        if len(new_subtokens) == n:
            return self

        return Word(tuple(new_subtokens), is_special=self.is_special)

    def __repr__(self):
        prefix = "Special " if self.is_special else ""
        return f"{prefix}Word({b', '.join(self)})"
