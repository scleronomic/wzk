import numpy as np
import tables as tb

from wzk import tictoc


def robot_paths_table(n_wp, n_dof):

    class RobotPaths(tb.IsDescription):
        i_world = tb.UInt32Col()
        i_sample = tb.UInt32Col()
        # q = tb.Float32Col(shape=(n_wp, n_dof))
        q = tb.BoolCol(shape=(n_wp, n_dof, 64))
        objective = tb.Float32Col()
        feasibility = tb.Float32Col()

    return RobotPaths


print()


""" open a file and create the table """
fileh = tb.open_file('RobotPaths.h5', mode='w')
filters = tb.Filters(complevel=9, complib='zlib')
table = fileh.create_table(where='/',
                           name='paths',
                           description=robot_paths_table(n_wp=64, n_dof=64),
                           title='RobotPaths',
                           filters=filters)

""" get the last row """
row = table.row

n = 10000
q_list = np.random.random((n, 64, 64, 64)).astype(np.float32) < 0.9


with tictoc('loop') as _:
    for i in range(n):
        row['q'] = q_list[i]
        row.append()


""" write to disk and close the file"""
with tictoc('write') as _:
    table.flush()
    fileh.close()


""" check it """
fileh = tb.open_file('RobotPaths.h5', mode='r')
print(len(fileh.root.paths))
print(fileh.root.paths[0][''])
fileh.close()
