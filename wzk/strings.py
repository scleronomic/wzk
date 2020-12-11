import os
import re
import uuid

abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
       'h', 'o', 'j', 'k', 'l', 'm', 'n',
       'c', 'p', 'q', 'r', 's', 't', 'u',
       'v', 'w', 'x', 'y', 'z']
ABC = [letter.upper() for letter in abc]


# https://en.wikipedia.org/wiki/Bracket
__brackets_round = '(', ')'
__brackets_square = '[', ']'
__brackets_curly = '{', '}'
__brackets_angle = '<', '>'
brackets_dict = dict(round=__brackets_round, r=__brackets_round,
                     square=__brackets_square, s=__brackets_square,
                     curly=__brackets_curly, c=__brackets_curly,
                     angles=__brackets_angle, a=__brackets_angle)

brackets_rir = '({i})'
brackets_sis = '[{i}]'
brackets_rijr = '({i}, {j})'
brackets_sijs = '[{i}, {j}]'
brackets_sissjs = '[{i}][{j}]'
brackets_rirrjr = '({i})({j})'


def brackets_wrapper(bracket, idx, multi=True):
    n = len(idx)
    b = brackets_dict[bracket]

    if multi:
        c = ', '
        s = b[0] + ''.join(idx_i + c for idx_i in idx)[:-len(c)] + b[1]
    else:
        s = ''.join(b[0] + idx_i + b[1] for idx_i in idx)


def remove_non_numeric(*s, squeeze=True):
    res = [re.sub(r"\D", "", s_i) for s_i in s]

    if squeeze and len(res) == 1:
        return res[0]
    else:
        return res


def str0_to_n(s, n):
    if s is None:
        s = ['test']
    elif isinstance(s, str):
        s = [s]

    if len(s) == n:
        return s
    else:
        return s[:-1] + [s[-1] + '_' + str(i) for i in range(n-len(s)+1)]


def split_insert_join(s, split, insert_pre='', insert_after=''):
    return split.join([insert_pre + ss + insert_after for ss in s.split(split)])


def test_split_insert_joint():
    s0 = 'abcdx1234xikjx789xxtest'
    s1 ='YabcdxY1234xYikjxY789xYxYtest'
    s1 = split_insert_join(s=s, split='x', insert_pre='Y')

def tab_str(*args, tab=4, squeeze=True):
    res = tuple(split_insert_join(s=s, split='\n', insert_pre=' ' * tab) for s in args)
    if len(res) == 1:
        return res[0]
    else:
        return res


def str2number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def str2file(directory, **kwargs):

    for key in kwargs:
        file, ext = os.path.splitext(key)

        with open(f"{directory}/{file}{ext}", 'w') as f:
            f.write(kwargs[key])


def uuid4():
    """
    Universally Unique Identifier (UUID)
    128-Bit x 4-Bit alpha-numeric representation
     -> len() = 32
     """
    return uuid.uuid4().hex

