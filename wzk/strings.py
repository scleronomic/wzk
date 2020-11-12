import re
import uuid

abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
       'h', 'o', 'j', 'k', 'l', 'matrix', 'n',
       'c', 'p', 'q', 'r', 's', 't', 'u',
       'v', 'w', 'x', 'y', 'z']
ABC = [letter.upper() for letter in abc]


def remove_non_numeric(*s, squeeze=True):
    res = [re.sub(r"\D", "", s_i) for s_i in s]

    if squeeze and len(res) == 1:
        return res[0]
    else:
        return res


def string_0_to_n(s, n):
    if s is None:
        s = ['test']
    elif isinstance(s, str):
        s = [s]

    if len(s) == n:
        return s
    else:
        return s[:-1] + [s[-1] + '_' + str(i) for i in range(n-len(s)+1)]


def str2number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def uuid4():
    """
    Universally Unique Identifier (UUID)
    128-Bit x 4-Bit alpha-numeric representation
     -> len() = 32
     """
    return uuid.uuid4().hex

