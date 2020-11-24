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


# Tic - Toc (like in MATLAB)
def tic_toc_generator():
    """
    Generator that returns time differences
    https://stackoverflow.com/a/26695514/7570817
    """
    tf = time()
    while True:
        ti = tf
        tf = time()
        yield tf - ti


TicToc = tic_toc_generator()


def toc(message=None, temp_bool=True, decimals=6):

    # Prints the time difference yielded by generator instance TicToc
    temp_time_interval = next(TicToc)
    if temp_bool:
        if message is None:
            message = 'Elapsed time'
        elif message == '':
            return temp_time_interval

        message += ': '
        print(message + f"{temp_time_interval:.{decimals+1}}")
        return temp_time_interval


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(temp_bool=False)


