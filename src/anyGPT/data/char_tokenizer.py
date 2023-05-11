def encode(string, mapping):
    return [mapping[char] for char in string]


def decode(ints, mapping):
    return "".join([mapping[i] for i in ints])
