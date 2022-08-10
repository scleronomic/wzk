import os
import sys
import numpy as np

from wzk.numpy2 import get_stats


def quiet_mode_on():
    # can not be changed back
    # copy.copy does not work
    # stdout_copy = sys.stdout
    sys.stdout = open(os.devnull, "w")
    #

# def quiet_mode_off():
#     sys.stdout = stdout_copy
#
#
# def toggle_quiet_mode():
#     pass


def pre_string_suf(s: str, prefix: str = '', suffix: str = '', delimiter: str = ' | ') -> str:
    if s == '':
        s = prefix
    elif prefix == '':
        s = s
    else:
        s = f"{prefix}{delimiter}{s}"

    if s == '':
        s = suffix
    elif suffix == '':
        s = s
    else:
        s = f"{s}{delimiter}{suffix}"

    return s


def get_progress_bar(i, n, prefix='', suffix='', bar='█'):
    bar = bar * i + '-' * (n - i)
    return f"\r{prefix} |{bar}| {suffix}"


def print_progress_bar(i, n, prefix='', suffix='', bar_length=None):
    bar_length_max = 100
    if bar_length is None:
        bar_length = n

    bar_length = min(bar_length, bar_length_max)

    if n == 0:
        i, n = 0, 1

    i += 1
    filled_length = int(round(bar_length * i / float(n)))
    s = get_progress_bar(i=filled_length, n=bar_length, prefix=prefix, suffix=suffix)

    sys.stdout.write(s)

    if i == n:
        sys.stdout.write('\n')
    sys.stdout.flush()


def print_table(rows, columns, data, min_cell_size=10, cell_format='.5f'):
    max_cell_size_c = max([len(c) for c in columns] + [min_cell_size])
    max_cell_size_r = max([len(r) for r in rows] + [min_cell_size])

    row_format = '{:>' + str(max_cell_size_r) + '}'
    header_format = row_format + ('{:>' + str(max_cell_size_c) + '}') * len(columns)
    data_format = row_format + ('{:>' + str(max_cell_size_c) + cell_format + '}') * len(columns)

    print(header_format.format('', *columns))
    for row_name, row_data in zip(rows, data):
        print(data_format.format(row_name, *row_data))


def print_dict(d, newline=True, message=None):
    if message is not None:
        print(message)

    for key in d:
        print(key, ':')
        print(repr(d[key]))

        if newline:
            print()


def print_stats(*args, names=None, dec=4):

    if names is None:
        names = [str(i) for i in range(len(args))]
    if isinstance(names, str):
        names = [names]
    assert len(names) == len(args)

    stats = []
    s = None
    for a in args:
        s = get_stats(a)
        stats.append([s[key] for key in s])

    cols = [key for key in s]

    print_table(rows=names, columns=cols, data=stats, cell_format=f'.{dec}f')
    return np.array(stats)


def print_stats_bool(b, name='', dec=4):
    print(f"{name}: {b.sum()}/{len(b)} = {b.mean():.{dec}f}")


def print_correlation(bool_lists, names, dec=4):
    arr = np.zeros((len(bool_lists), len(bool_lists)))
    total = np.ones_like(bool_lists[0], dtype=bool)
    for i, b1 in enumerate(bool_lists):
        total = np.logical_and(total, b1)
        for j, b2 in enumerate(bool_lists):
            arr[i, j] = np.logical_and(b1, b2).mean()

    if dec:
        arr = np.round(arr, decimals=dec)

    print_table(rows=names, columns=names, data=arr)
    print(f"Total: {total.sum()}/{len(total)} = {total.mean()}")

    return total


def verbose_level_wrapper(verbose=None, level=0):  # TODO make more convenient and use it consistently
    if isinstance(verbose, tuple):
        verbose, level = verbose
    elif verbose is None:
        verbose = 1
    return verbose, level


def check_verbosity(verbose, threshold=0):
    if isinstance(verbose, int):
        return verbose > threshold
    elif isinstance(verbose, tuple):
        return verbose[0] > threshold


def print2(*args, verbose=None, level=0,
           sep=' ', end='\n', file=None, flush=False):
    verbose, level = verbose_level_wrapper(verbose=verbose, level=level)
    level = max(0, level)

    if verbose > 0:
        args = [str(a) for a in args]
        t = '\t'*level
        print(f"{t}{sep.join(args)}", sep=sep, end=end, file=file, flush=flush)


def print_array_3d(array_3d,
                   verbose=None, level=0):
    l, k, n = array_3d.shape
    s = [''] * k
    for ll in range(l):
        ss = repr(array_3d[ll])
        ss = ss.replace('array', f'   {ll}:')
        ss = ss.replace('(', ' ')
        ss = ss.replace(')', '')
        ss = ss.split('\n')
        for kk in range(k):
            s[kk] += ss[kk]

    print2('\n'.join(s), verbose=verbose, level=level)


def clear_previous_line():
    sys.stdout.write("\033[F")  # back to previous line
    sys.stdout.write("\033[K")  # clear line


def color_text(s, color, background='w', weight=0):
    """you have to print normally (black / white) once to don't have any sight effects """
    color_dict = dict(black=0, k=0,
                      red=1, r=1,
                      green=2, g=2,
                      yellow=3, y=3,
                      blue=4, b=4,
                      magenta=5, m=5,
                      cyan=6, c=6,
                      gray=7, l=7,
                      white=8, w=8)
    tc = color_dict[color.lower()]
    bc = color_dict[background.lower()]
    return f"\033[{weight};3{tc};4{bc}m{s}\033"
