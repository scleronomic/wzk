from unittest import TestCase

from wzk.sql2 import *
from wzk.files import safe_rmdir


class Test(TestCase):
    
    @staticmethod
    def __create_dummy_db(mode):
        if mode == 'A':
            columns = ['A', 'B', 'C', 'DD']
            data = [[1,   2,   3.3,  4],
                    [10,  20,  33.3,  40],
                    [100, 300, 333.3, 400]]
            df = pd.DataFrame(columns=columns, data=data, index=None)
            return df
        elif mode == 'B':

            columns = ['aa', 'bb', 'cc']
            n = 11
            data = np.random.random((n, 3)).tolist()
            df = pd.DataFrame(columns=columns, data=data, index=None)
            return df

        elif mode == 'C':
            columns = ['A', 'B', 'C']
            data = [[1, 1, 33.3],
                    [1, 2, 33.3],
                    [3, 3, 33.3],
                    [1, 4, 33.3],
                    [2, 5, 33.3],
                    [1, 6, 33.3],
                    [1, 7, 33.3],
                    [4, 8, 33.3]]
            df = pd.DataFrame(columns=columns, data=data, index=None)
            return df

        elif mode == 'D':
            columns = ['A_f32', 'B', 'C']
            data = [[np.random.random((20, 4)), 1, 33.3],
                    [np.random.random((20, 4)), 2, 33.3],
                    [np.random.random((20, 4)), 3, 33.3],
                    [np.random.random((20, 4)), 4, 33.3],
                    [np.random.random((20, 4)), 5, 33.3],
                    [np.random.random((20, 4)), 6, 33.3],
                    [np.random.random((20, 4)), 7, 33.3],
                    [np.random.random((20, 4)), 8, 33.3]]
            df = pd.DataFrame(columns=columns, data=data, index=None)
            return df

        else:
            raise ValueError

    @staticmethod
    def get_path():
        return os.path.split(__file__)[0]

    def test_concatenate_tables(self):

        df = self.__create_dummy_db('A')

        file1 = f"{self.get_path()}/dummy_test_concatenate_tables_1.db"
        file2 = f"{self.get_path()}/dummy_test_concatenate_tables_2.db"
        tableA = 'tableA'
        tableB = 'tableB'
        df2sql(df=df, file=file1, table=tableA, if_exists='replace')
        df2sql(df=df, file=file1, table=tableB, if_exists='replace')
        df2sql(df=df, file=file2, table=tableA, if_exists='replace')

        #
        concatenate_tables(file=file1, table=tableB, table2=tableA)
        df2 = get_values_sql(file=file1, table=tableB, rows=[3, 4, 5])
        self.assertTrue(np.all(df == df2))

        concatenate_tables(file=file2, table=tableA, file2=file1, table2=tableB)
        df2 = get_values_sql(file=file2, table=tableA, rows=[6, 7, 8])
        self.assertTrue(np.all(df == df2))
        #

        safe_rmdir(file1)
        safe_rmdir(file2)

    def test_set_values(self):
        file = f"{self.get_path()}/dummy_test_set_values.db"
        table = 'dummytable'
        df = self.__create_dummy_db('A')
        df2sql(df=df, file=file, table=table, if_exists='replace')

        #
        c = 'B'
        r = [0, 1]
        v = [-2, -20]
        set_values_sql(file=file, table=table, values=(v,), columns=c, rows=r)
        v1 = get_values_sql(file=file, table=table, rows=r, columns=c, values_only=True)
        self.assertTrue(np.array_equal(v, v1))
        #
        safe_rmdir(file)

    def test_set_values_2(self):
        file = f"{self.get_path()}/dummy_test_set_values_2.db"
        table = 'dummytable'
        df = self.__create_dummy_db('D')
        df2sql(df=df, file=file, table=table, if_exists='replace')

        #
        c = 'A_f32'
        r = [0, 1]
        v = np.arange(2*20*4).reshape((2, 20, 4))
        v0 = get_values_sql(file=file, table=table, rows=r, columns=c, values_only=True)
        set_values_sql(file=file, table=table, values=(v,), columns=c, rows=r)
        v1 = get_values_sql(file=file, table=table, rows=r, columns=c, values_only=True)
        v1 = v1.reshape((2, 20, 4))
        self.assertTrue(np.array_equal(v, v1))
        #

        safe_rmdir(file)

    def test_add_column(self):
        file = f"{self.get_path()}/dummy_test_set_values_3.db"
        table = 'dummytable'
        df = self.__create_dummy_db('B')
        df2sql(df=df, file=file, table=table, if_exists='replace')

        #
        c = 'dd'
        r = np.arange(11)
        v = np.random.random(11)
        add_column(file=file, table=table, column=c, dtype=TYPE_REAL)
        set_values_sql(file=file, table=table, values=(v,), columns=c, rows=r)
        v1 = get_values_sql(file=file, table=table, rows=r, columns=c, values_only=True)
        self.assertTrue(np.array_equal(v, v1))
        #

        safe_rmdir(file)

    def test_delete_rows(self):
        file = f"{self.get_path()}/dummy_test_delete_rows.db"
        table = 'dummytable'
        df = self.__create_dummy_db('A')
        df2sql(df=df, file=file, table=table, if_exists='replace')

        #
        n_rows = get_n_rows(file=file, table=table)
        self.assertTrue(n_rows == 3)

        delete_rows(file=file, table=table, rows=[0, 1])
        n_rows = get_n_rows(file=file, table=table)
        self.assertTrue(n_rows == 1)

        df2 = get_values_sql(file=file, table=table)
        self.assertTrue(np.array_equal(df.iloc[2:3, :].values, df2.values))
        #

        safe_rmdir(file)

    def test_get_columns(self):
        file = f"{self.get_path()}/dummy_test_get_columns.db"
        table = 'dummytable'
        df = self.__create_dummy_db('A')
        df2sql(df=df, file=file, table=table, if_exists='replace')

        #
        c = get_columns(file=file, table=table, mode='name')
        self.assertTrue(np.array_equal(df.columns.values, c))
        #

        safe_rmdir(file)

    def test_change_column_dtype(self):
        file = f"{self.get_path()}/dummy_test_change_column_dtype.db"
        table = 'dummytable'
        df = self.__create_dummy_db('B')
        df2sql(df=df, file=file, table=table, if_exists='replace')

        #
        change_column_dtype(file, table, column='aa', dtype=TYPE_INTEGER)
        delete_columns(file=file, table=table, columns=['bb', 'cc'])
        self.assertTrue(np.all(get_values_sql(file=file, table=table, values_only=True) == 0))
        #

        safe_rmdir(file)

    def test_alter_table(self):
        file = f"{self.get_path()}/dummy_test_alter_table.db"
        df = self.__create_dummy_db('B')
        table = 'dummytable'
        df2sql(df=df, file=file, table=table, if_exists='replace')

        #
        new_columns = ['bb', 'aa', 'cc']
        new_types = [TYPE_TEXT, TYPE_INTEGER, TYPE_REAL]
        alter_table(file=file, table=table, columns=new_columns, dtypes=new_types)

        data2 = get_values_sql(file=file, table=table, values_only=True)
        bb, aa, cc = data2

        c = get_columns(file=file, table=table, mode='name')
        self.assertTrue(np.all(c == new_columns))

        self.assertTrue(isinstance(bb[0], str))
        self.assertTrue(isinstance(aa[0], np.int64))
        self.assertTrue(isinstance(cc[0], float))
        #

        safe_rmdir(file)

    def test_sort_table(self):
        file = f"{self.get_path()}/dummy_test_sort_table.db"
        df = self.__create_dummy_db('C')
        table = 'dummytable'
        df2sql(df=df, file=file, table=table, if_exists='replace')

        #
        sort_table(file=file, table=table, order_by=['A', 'ROWID'])
        data2 = get_values_sql(file=file, table=table)
        print(data2)
        a2 = get_values_sql(file=file, table=table, columns='A', values_only=True)
        self.assertTrue(np.all(np.argsort(a2) == np.arange(len(a2))))
        #

        safe_rmdir(file)


