import numpy as np
from wzk import spatial, pv2


def understand_rotvec():
    x = np.zeros((5, 3))
    x[:, 2] = np.arange(5)
    rv = np.zeros((5, 3))
    # rv[:, :] = np.array([[0.3, 0.2, 0.1]])  # its not the same as multiplying a z rotation matrix on that frame
    rv[1, 2] += np.pi/2
    rv[2, 2] += np.pi
    rv[3, 2] += 2*np.pi
    rv[4, 2] += -np.pi
    print(rv)
    f = spatial.trans_rotvec2frame(trans=x, rotvec=rv)

    pl = pv2.Plotter()
    pv2.plot_coordinate_frames(f=f, pl=pl)
    pl.add_axes_at_origin(labels_off=False)

    pl.show()
