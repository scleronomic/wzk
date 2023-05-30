import numpy as np
from wzk import pv2, perlin


img = np.array([perlin.perlin_noise_3d(shape=(64, 64, 64)) < 0.5 for _ in range(100)])

s = img.sum(axis=(-1, -2, -3))
idx_s_sorted = np.argsort(s)
limits = np.array([[-1., 1.],
                   [-1., 1.],
                   [-1., 1.]])

f0 = np.eye(4)
f1 = np.eye(4)
f1[:, -1] = 1

for ii, i in enumerate(idx_s_sorted[::20]):
    print(i)
    p = pv2.Plotter(off_screen=True)
    p.open_gif(f"camera_test_{ii}_{i}.gif")

    pv2.plot_coordinate_frames(p=p, f=f0, scale=0.3)
    pv2.plot_bimg(p=p, img=img[i], limits=limits, color="grey", mode="mesh")
    pv2.plot_cube(p=p, limits=limits, color="black")

    for j, cpp in enumerate(pv2.camera_motion(pos=np.array([0, 0, 4]),
                                              focal=np.array([0., 0., 0.]),
                                              viewup=np.array([0., 0., 1.]),
                                              mode="circle_xy", radius=4, n=100)):
        # print(j)
        p.camera_position = cpp
        p.render()
        p.write_frame()

    p.close()
