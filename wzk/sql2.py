import numpy as np
import pandas as pd

import sqlite3 as sql
from contextlib import contextmanager

from wzk.numpy2 import numeric2object_array
from wzk.dicts_lists_tuples import change_tuple_order
from wzk.dtypes import str2np

_CMP = '_cmp'


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


def execute(file, command):
    with open_db_connection(file=file, close=True) as con:
        con.execute(command)


def vacuum(file):
    execute(file=file, command='VACUUM')


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


def __decompress_values(value, column: str):
    # SQL saves everything in binary form -> convert back to numeric, expect the columns which are marked as cmp
    if isinstance(value[0], bytes) and column[-4:] != _CMP:
        value = np.array([np.frombuffer(v, dtype=str2np(s=column)) for v in value])
    return value


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

    if isinstance(rows, int):
        rows = [rows]
    rows = np.array(rows)

    if rows[0] == -1:  # All samples
        with open_db_connection(file=file, close=True, lock=lock) as con:
            df = pd.read_sql_query(con=con, sql=f"SELECT {columns_str} FROM {table}")  # path_db

    else:
        rows_str = rows + 1  # Attention! Unlike in Python, SQL indices start at 1
        rows_str = ', '.join(map(str, rows_str))
        with open_db_connection(file=file, close=True, lock=lock) as con:
            df = pd.read_sql_query(sql=f"SELECT {columns_str} FROM {table} WHERE ROWID in ({rows_str})",
                                   index_col=rows, con=con)

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
    Important multidimensional numpy arrays have to be saved as flat to SQL otherwise the order is messed up

    'i_samples' == i_samples_global
    values = ([...], [...], [...], ...)
    """

    # Handle rows argument
    if isinstance(rows, int):
        if rows == -1:
            rows = np.arange(len(values[0])).tolist()
        else:
            rows = [rows]

    rows_sql = (np.array(rows) + 1).tolist()  # Attention! Unlike in Python, SQL indices start at 1

    # Handle columns argument
    if isinstance(columns, str):
        columns = [columns]

    columns_str = '=?, '.join(map(str, columns))
    columns_str += '=?'

    values_rows_sql = change_tuple_order(values + (rows_sql,))
    values_rows_sql = list(values_rows_sql)
    query = f"UPDATE {table} SET {columns_str} WHERE ROWID=?"

    with open_db_connection(file=file, close=True, lock=lock) as con:
        cur = con.cursor()
        if len(values_rows_sql) == 1:
            cur.execute(query, values_rows_sql[0])
        else:
            cur.executemany(query, values_rows_sql)

        con.commit()


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


# def __test_speed_get_values():
#     from wzk import tic, toc
#     from definitions import START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP, PATH_Q
#     sib = np.random.choice(np.arange(5000 * 1000), 256, replace=False)
#     file = 'abc/path.db'
#
#     tic()
#     _ = get_values_sql(rows=sib, values_only=True, file=file,
#                        columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP, PATH_Q])
#     toc()
#
#     tic()
#     _ = get_values_sql(rows=sib, values_only=True, file=file,
#                        columns=[START_IMG_CMP, END_IMG_CMP, PATH_IMG_CMP])
#     _ = get_values_sql(rows=sib, values_only=True, file=file,
#                        columns=[PATH_Q])
#     toc()
#     # FINDING it is faster to load all wanted columns at once from the sql file and not one after the other
#
#
# if __name__ == '__main__':
#     __test_speed_get_values()
