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


def print_progress(i, n, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        i           - Required  : current iteration (Int)
        n           - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """

    if n == 0:
        n = 1
        i = 0

    i += 1
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (i / float(n)))
    filled_length = int(round(bar_length * i / float(n)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

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
        print(d[key])

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


def verbose_level_wrapper(verbose=None, level=0):
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
           sep=' ', end='\n', file=None):
    verbose, level = verbose_level_wrapper(verbose=verbose, level=level)
    level = max(0, level)

    if verbose > 0:
        args = [str(a) for a in args]
        t = '\t'*level
        print(f"{t}{sep.join(args)}", sep=sep, end=end, file=file)


def test_print2():
    print2("aaa", 1, 2, verbose=(1, 0))
    print2(dict(b=1, bb=2), 11, 22, verbose=(1, 1))
    print2("ccc", [3, "cc", 333], 33, verbose=(1, 2), sep='---')
    print2("nice", "a", "staircase", verbose=(1, 1), sep='    ')
    print2("back", "to", "level", "zero", verbose=(1, 0), sep='::')


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


def test_print_array_3d():
    array_3d = np.arange(4*5*6).reshape((4, 5, 6))
    print_array_3d(array_3d)
