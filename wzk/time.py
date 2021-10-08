from time import time, sleep  # noqa: F401 unused import
from datetime import datetime


def get_timestamp(year=True, month=True, day=True, hour=True, minute=True, second=True, millisecond=False,
                  date_separator='-', date_time_separator='_', time_separator=':'):
    """
    Crete a datetime string including year, month, day; hour, minute, second, millisecond.
    With options for each of the elements if it should be included and how the symbol separating
    two elements should look like.
    """

    # Create a boolean list indicating which elements to keep
    bool_list = [bool(b) for b in [year, month, day, hour, minute, second, millisecond]]

    # Create a list of symbols which separate the elements of teh datetime string
    separator_list = [date_separator, date_separator,
                      date_time_separator,
                      time_separator, time_separator, time_separator]

    # Clean the datetime string and split it into separate elements
    stamp_list = str(datetime.now())
    stamp_list = stamp_list.replace('-', 'x')
    stamp_list = stamp_list.replace(':', 'x')
    stamp_list = stamp_list.replace(' ', 'x')
    stamp_list = stamp_list.replace('.', 'x')
    stamp_list = stamp_list.split('x')

    # Loop over the elements of the datetime string and concatenate them with the corresponding symbols
    stamp_str = ''
    for i in range(len(stamp_list)):
        if bool_list[i]:
            stamp_str += stamp_list[i]
            if i < len(stamp_list) and sum(bool_list[i+1:]) > 0:
                stamp_str += separator_list[i]

    return stamp_str


__start_stack = []
__start_named = {}


def tic(name: str = None):
    if name is None:
        __start_stack.append(time())
    else:
        __start_named[name] = time()


def toc(name: str = None, decimals: int = 6) -> float:
    if name is None:
        start = __start_stack.pop()
    else:
        try:
            start = __start_named.pop(name)
        except KeyError:
            start = __start_stack.pop()

    elapsed = time() - start

    if name is None:
        name = 'Elapsed time'
    elif name == '':
        return elapsed

    name += ': '
    print(name + f"{elapsed:.{decimals+1}}")

    return elapsed
