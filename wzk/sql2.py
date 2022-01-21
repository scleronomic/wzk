from contextlib import contextmanager
# from threading import Lock  # lock = Lock()
import os
import numpy as np
import pandas as pd
import sqlite3 as sql

from wzk.numpy2 import numeric2object_array
from wzk.dicts_lists_tuples import change_tuple_order, atleast_list
from wzk.dtypes import str2np
from wzk.strings import uuid4
_CMP = '_cmp'


TYPE_TEXT = 'TEXT'
TYPE_NUMERIC = 'NUMERIC'
TYPE_INTEGER = 'INTEGER'
TYPE_REAL = 'REAL'
TYPE_BLOB = 'BLOB'


def __handle_file_extension(file):
    file, ext = os.path.splitext(file)
    ext = ext or '.db'
    file = f"{file}{ext}"
    return file


def rows2sql(rows: object, dtype: object = str, values=None) -> object:
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


def columns2sql(columns: object, dtype: object):
    if columns is None:
        return '*'
    if isinstance(columns, str):
        columns = [columns]

    if dtype == str:
        return ', '.join(map(str, columns))
    elif dtype == list:
        return columns
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

    file = __handle_file_extension(file)
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


def set_journal_mode_wal(file, lock=None):
    # https://www.sqlite.org/pragma.html#pragma_journal_mode
    # speed up through smarter journal mode https://sqlite.org/wal.html
    execute(file=file, query='PRAGMA journal_mode=WAL', lock=lock)


def executescript(file, query, lock=None):
    with open_db_connection(file=file, close=True, lock=lock) as con:
        con.executescript(query)


def vacuum(file):

    # https://stackoverflow.com/a/23251896/7570817
    # To allow the VACUUM command to run, change the directory for temporary files to one that has enough free space.
    # assumption, that this is the case for the directory where the file itself leads
    # temp_store_directory is deprecated, but hte alternatives did not work
    directory = os.path.split(file)[0]
    execute(file=file, query=f"PRAGMA temp_store_directory = '{directory}'")

    execute(file=file, query='VACUUM')


def get_tables(file: str) -> list:
    with open_db_connection(file=file, close=True) as con:
        t = pd.read_sql_query(sql="SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%'",
                              con=con)
    return t['name'].values


def get_columns(file, table, mode: object = None):
    with open_db_connection(file=file, close=True, lock=None) as con:
        c = pd.read_sql_query(con=con, sql=f"pragma table_info({table})")

    if mode is None:
        return c

    res = []
    if 'name' in mode:
        res.append(c.name.values)

    if 'type' in mode:
        res.append(c.type.values)

    if len(res) == 1:
        res = res[0]

    return res


def rename_tables(file: str, tables: dict) -> None:
    old_names = get_tables(file=file)

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
            new = columns[old]
            cur.execute(f"ALTER TABLE `{table}` RENAME COLUMN `{old}` TO `{new}`")


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


def integrity_check(file):
    with open_db_connection(file=file, close=True, lock=None) as con:
        c = pd.read_sql_query(con=con, sql=f"pragma integrity_check")
    print(c)
    return c.values[0][0]


def concatenate_tables(file, table, table2, file2=None, lock=None):

    if file2 is None:
        execute(file=file, query=f"INSERT INTO {table} SELECT * FROM {table2}", lock=lock)

    else:
        query = f"ATTACH DATABASE '{file2}' AS filetwo; INSERT INTO {table} SELECT * FROM filetwo.{table2}"
        executescript(file=file, query=query, lock=None)


def __decompress_values(value, column: str):
    # SQL saves everything in binary form -> convert back to numeric, expect the columns which are marked as cmp
    # value = atleast_list(value, convert=False)
    # print(value)
    # print(column)
    if isinstance(value[0], bytes) and column[-4:] != _CMP:
        dtype = str2np(s=column)
        value = np.array([np.frombuffer(v, dtype=dtype) for v in value])

    return value


def delete_tables(file, tables):
    tables = atleast_list(tables)
    for t in tables:
        execute(file=file, query=f"DROP TABLE {t}")
    vacuum(file=file)


def delete_rows(file: str, table: str, rows, lock=None):
    rows = rows2sql(rows, dtype=str)
    execute(file=file, lock=lock, query=f"DELETE FROM {table} WHERE ROWID in ({rows})")
    vacuum(file)


def delete_columns(file: str, table: str, columns, lock=None):
    columns = columns2sql(columns, dtype=list)
    for col in columns:
        execute(file=file, lock=lock, query=f"ALTER TABLE {table} DROP COLUMN {col}")
    vacuum(file)


def add_column(file, table, column, dtype, lock=None):
    execute(file=file, query=f"ALTER TABLE {table} ADD COLUMN {column} {dtype}", lock=lock)


def copy_column(file, table, column_src, column_dst, dtype, lock=None):
    column_list = get_columns(file, table, mode='name')
    assert column_src in column_list
    if column_dst not in column_list:
        add_column(file=file, table=table, column=column_dst, dtype=dtype, lock=lock)
    execute(file=file, query=f"UPDATE {table} SET {column_dst} = CAST({column_src} as {dtype})", lock=lock)


def copy_table(file, table_src, table_dst, columns=None, dtypes=None):
    columns_old = get_columns(file=file, table=table_src, mode=None)
    if columns is None:
        columns = columns_old.name.values
    if dtypes is None:
        dtypes = columns_old.type.values

    columns = columns2sql(columns, dtype=list)
    dtypes = columns2sql(dtypes, dtype=list)
    assert len(columns) == len(dtypes)

    columns_dtype_str = ', '.join([f"{c} {d}" for c, d in zip(columns, dtypes)])
    columns_cast_dtype_str = ', '.join([f"CAST({c} AS {d})" for c, d in zip(columns, dtypes)])

    execute(file=file, query=f"CREATE TABLE {table_dst}({columns_dtype_str})")
    execute(file=file, query=f"INSERT INTO {table_dst} SELECT {columns_cast_dtype_str} FROM {table_src}")


def alter_table(file, table, columns, dtypes):
    table_tmp = table + uuid4()
    copy_table(file=file, table_src=table, table_dst=table_tmp, columns=columns, dtypes=dtypes)
    delete_tables(file, tables=table)
    rename_tables(file, tables={table_tmp: table})


def squeeze_table(file, table, verbose=1):
    columns = get_columns(file=file, table=table, mode='name')

    for c in zip(columns):
        v0 = get_values_sql(file=file, table=table, columns=c, rows=0, values_only=True)
        if np.size(v0) == 1:
            if verbose > 0:
                print(c)
            v = get_values_sql(file=file, table=table, columns=c, values_only=True)
            v = np.squeeze(v)
            set_values_sql(file=file, table=table, values=(v.tolist(),), columns=c)


def change_column_dtype(file, table, column, dtype, lock=None):
    column_tmp = f"{column}{uuid4()}"
    copy_column(file=file, table=table, column_src=column, column_dst=column_tmp, dtype=dtype)
    delete_columns(file=file, table=table, columns=column, lock=lock)
    copy_column(file=file, table=table, column_src=column_tmp, column_dst=column, dtype=dtype)
    delete_columns(file=file, table=table, columns=column_tmp, lock=lock)


# Get and Set SQL values
def get_values_sql(file: str, table: str, columns=None, rows=-1,
                   values_only: bool = False, squeeze_col: bool = True, squeeze_row: bool = True):
    """
    'i_samples' == i_samples_global
    """

    lock = None  # Lock is not necessary fo reading

    columns = columns2sql(columns=columns, dtype=list)
    columns_str = columns2sql(columns=columns, dtype=str)

    rows = rows2sql(rows, dtype=str)

    if rows == -1:  # All samples
        with open_db_connection(file=file, close=True, lock=lock) as con:
            df = pd.read_sql_query(con=con, sql=f"SELECT {columns_str} FROM {table}")

    else:
        with open_db_connection(file=file, close=True, lock=lock) as con:
            df = pd.read_sql_query(con=con, sql=f"SELECT {columns_str} FROM {table} WHERE ROWID in ({rows})",
                                   index_col=None)

    value_list = []
    if np.any(columns == '*'):
        columns = df.columns.values

    # print('rows', rows)
    # print(df)
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

    set_journal_mode_wal(file=file, lock=lock)

    # TODO handle array inputs, automatically convert to correct datatype
    rows = rows2sql(rows, values=values[0], dtype=list)
    columns = columns2sql(columns, dtype=list)

    columns = '=?, '.join(map(str, columns))
    columns += '=?'

    values_rows_sql = change_tuple_order(values + (rows,))
    values_rows_sql = list(values_rows_sql)
    query = f"UPDATE {table} SET {columns} WHERE ROWID=?"

    executemany(file=file, query=query, args=values_rows_sql, lock=lock)


def dict2cv():
    pass
    # c = d.keys()
    raise NotImplementedError


def df2sql(df, file, table, if_exists='fail'):
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

    with open_db_connection(file=file, close=True, lock=None) as con:
        df.to_sql(name=table, con=con, if_exists=if_exists, index=False, chunksize=None)

    set_journal_mode_wal(file=file)
