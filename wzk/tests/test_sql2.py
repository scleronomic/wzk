from unittest import TestCase

from wzk.sql2 import *


class Test(TestCase):

    def test_all(self):
        columns = ['A', 'B', 'C', 'DD']
        data = [[1,   2,   3.3,  'four'],
                [10,  20,  33.3,  'forty'],
                [100, 300, 333.3, 'four-hundred']]

        index = None
        df = pd.DataFrame(columns=columns, data=data, index=index)

        file1 = '/Users/jote/Documents/Code/Python/DLR/wzk/wzk/tests/dummy1.db'
        file2 = '/Users/jote/Documents/Code/Python/DLR/wzk/wzk/tests/dummy2.db'
        print(__file__)

        tableA = 'testA'
        tableB = 'testB'
        df2sql(df=df, file=file1, table=tableA, if_exists='replace')
        df2sql(df=df, file=file1, table=tableB, if_exists='replace')
        df2sql(df=df, file=file2, table=tableA, if_exists='replace')

        concatenate_tables(file=file1, table=tableB, table2=tableA)
        df2 = get_values_sql(file=file1, table=tableB, rows=[3, 4, 5])
        self.assertTrue(np.all(df == df2))

        c = 'B'
        r = [0, 1]
        v = [-2, -20]
        set_values_sql(file=file1, table=tableA, values=(v,), columns=c, rows=r)
        v1 = get_values_sql(file=file1, table=tableA, rows=r, columns=c, values_only=True)
        self.assertTrue(np.array_equal(v, v1))

        n_rows = get_n_rows(file=file1, table=tableA)
        self.assertTrue(n_rows == 3)

        delete_rows(file=file1, table=tableA, rows=[0, 1])
        n_rows = get_n_rows(file=file1, table=tableA)
        print(get_values_sql(file=file1, table=tableA))
        self.assertTrue(n_rows == 1)

        df2 = get_values_sql(file=file1, table=tableA)
        self.assertTrue(np.array_equal(df.iloc[2:3, :].values, df2.values))

        c = get_columns(file=file1, table=tableA)
        self.assertTrue(np.array_equal(columns, c))

        concatenate_tables(file=file2, table=tableA, file2=file1, table2=tableB)
        print(get_values_sql(file=file2, table=tableA))


