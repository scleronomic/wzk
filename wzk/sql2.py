from contextlib import contextmanager
# from threading import Lock  # lock = Lock()
import numpy as np
import pandas as pd
import sqlite3 as sql

from wzk.numpy2 import numeric2object_array
from wzk.dicts_lists_tuples import change_tuple_order
from wzk.dtypes import str2np

_CMP = '_cmp'


def idx2sql(rows: object, dtype: object = str, values=None) -> object:
    if isinstance(rows, (int, np.int16, np.int32, np.int64)):
        if rows == -1:
            if values is None:
                return -1
            else:
                rows = np.arange(len(values))
        else:
            rows = [int(rows)]
    elif isinstance(rows, np.ndarray) and rows.dtype == bool:
        rows = np.nonzero(rows)[0]

    rows = np.array(rows) + 1  # Attention! Unlike in Python, SQL indices start at 1

    if dtype == str:
        return ', '.join(map(str, rows))

    elif dtype == list:
        return rows.tolist()

    else:
        raise ValueError


@contextmanager
def open_db_connection(file, close=True,
                       lock=None, check_same_thread=False, isolation_level=None):
    """
    Safety wrapper for the database call.
    """

    if lock is not None:
        lock.acquire()

    con = sql.connect(database=file, check_same_thread=check_same_thread, isolation_level=isolation_level)

    try:
        yield con

    finally:
        if close:
            con.close()
        if lock is not None:
            lock.release()


def execute(file, query, lock=None):
    with open_db_connection(file=file, close=True, lock=lock) as con:
        con.execute(query)


def executemany(file, query, args, lock=None):
    with open_db_connection(file=file, close=True, lock=lock) as con:
        con.executemany(query, args)


def executescript(file, query, lock=None):
    with open_db_connection(file=file, close=True, lock=lock) as con:
        con.executescript(query)


def vacuum(file):
    execute(file=file, query='VACUUM')


def get_table_name(file: str) -> list:
    with open_db_connection(file=file, close=True) as con:
        res = pd.read_sql_query(sql="SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%'",
                                con=con)
        return res['name'].values


def rename_tables(file: str, tables: dict) -> None:
    old_names = get_table_name(file=file)

    with open_db_connection(file=file, close=True) as con:
        cur = con.cursor()
        for old in old_names:
            if old in tables:
                new = tables[old]
                cur.execute(f"ALTER TABLE `{old}` RENAME TO `{new}`")


def rename_columns(file: str, table: str, columns: dict) -> None:
    with open_db_connection(file=file, close=True) as con:
        cur = con.cursor()
        for old in columns:
            cur.execute(f"ALTER TABLE `{table}` RENAME COLUMN `{old}` TO `{columns[old]}`")


def get_n_rows(file, table):
    """
    Only works if the rowid's are [0, ....i_max]
    """
    with open_db_connection(file=file, close=True) as con:
        return pd.read_sql_query(con=con, sql=f"SELECT COALESCE(MAX(rowid), 0) FROM {table}").values[0, 0]


def get_n_samples(file, i_worlds=-1):
    i_worlds_all = get_values_sql(file=file, columns='i_world', values_only=True, table='paths')
    unique, counts = np.unique(i_worlds_all, return_counts=True)
    if i_worlds == -1:
        return counts
    else:
        return counts[i_worlds]


def get_columns(file, table):
    with open_db_connection(file=file, close=True, lock=None) as con:
        c = pd.read_sql_query(con=con, sql=f"pragma table_info({table})")
    c = [cc[1] for cc in c.values]
    return c


def concatenate_tables(file, table, table2, file2=None, lock=None):
    if file2 is None:
        execute(file=file, query=f"INSERT INTO {table} SELECT * FROM {table2}", lock=lock)
    else:

        query = f"ATTACH DATABASE '{file2}' AS filetwo; INSERT INTO {table} SELECT * FROM filetwo.{table2}"
        executescript(file=file, query=query, lock=None)


def __decompress_values(value, column: str):
    # SQL saves everything in binary form -> convert back to numeric, expect the columns which are marked as cmp

    if isinstance(value[0], bytes) and column[-4:] != _CMP:
        value = np.array([np.frombuffer(v, dtype=str2np(s=column)) for v in value])

    return value


def delete_rows(file: str, table: str, rows, lock=None):
    rows = idx2sql(rows, dtype=str)
    execute(file=file, lock=lock, query=f"DELETE FROM {table} WHERE ROWID in ({rows})")
    vacuum(file)


# Get and Set SQL values
def get_values_sql(file: str, table: str, columns=None, rows=-1,
                   values_only: bool = False, squeeze_col: bool = True, squeeze_row: bool = True):
    """
    'i_samples' == i_samples_global
    """

    lock = None  # Lock is not necessary fo reading
    if columns is None:
        columns = '*'
    if isinstance(columns, str):
        columns = [columns]
    columns_str = ', '.join(map(str, columns))

    rows = idx2sql(rows, dtype=str)

    if rows == -1:  # All samples
        with open_db_connection(file=file, close=True, lock=lock) as con:
            df = pd.read_sql_query(con=con, sql=f"SELECT {columns_str} FROM {table}")  # path_db

    else:
        with open_db_connection(file=file, close=True, lock=lock) as con:
            df = pd.read_sql_query(con=con, sql=f"SELECT {columns_str} FROM {table} WHERE ROWID in ({rows})",
                                   index_col=None)

    value_list = []
    if np.any(columns == ['*']):
        columns = df.columns.values

    if values_only:
        for col in columns:
            value = __decompress_values(value=df.loc[:, col].values, column=col)
            value_list.append(value)

        if len(df) == 1 and squeeze_row:
            for i in range(len(columns)):
                value_list[i] = value_list[i][0]

        if len(value_list) == 1 and squeeze_col:
            value_list = value_list[0]

        return value_list

    # Return pandas.DataFrame
    else:
        for col in columns:
            value = __decompress_values(value=df.loc[:, col].values, column=col)
            df.loc[:, col] = numeric2object_array(value)

        return df


def set_values_sql(file, table,
                   values, columns, rows=-1, lock=None):
    """
    Attention! multidimensional numpy arrays have to be saved as flat byte string

    values = ([...], [...], [...], ...)
    """

    # TODO handle arry inputs, automatically convert to correct datatype
    rows = idx2sql(rows, values=values[0], dtype=list)

    # Handle columns argument
    if isinstance(columns, str):
        columns = [columns]

    columns_str = '=?, '.join(map(str, columns))
    columns_str += '=?'

    values_rows_sql = change_tuple_order(values + (rows,))
    values_rows_sql = list(values_rows_sql)
    query = f"UPDATE {table} SET {columns_str} WHERE ROWID=?"

    executemany(file=file, query=query, args=values_rows_sql, lock=lock)


# TODO
def dict2cv(d):
    pass
    # c = d.keys()
    raise NotImplementedError


def df2sql(df, file, table, if_exists='fail', lock=None):
    """
    From DataFrame.to_sql():
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
                   - fail: If table exists, do nothing.
                   - replace: If table exists, drop it, recreate it, and insert Measurements.
                   - append: If table exists, insert Measurements. Create if does not exist.
    """
    if df is None:
        print('No DataFrame was provided...')
        return

    with open_db_connection(file=file, close=True, lock=lock) as con:
        df.to_sql(name=table, con=con, if_exists=if_exists, index=False)
