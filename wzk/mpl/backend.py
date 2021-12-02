import os
import platform
import matplotlib as mpl


headless = False
if platform.system() == 'Linux':
    try:
        display = os.environ['DISPLAY']
        mpl.use('TkAgg')

    except KeyError:
        print("Matplotlib - Backend: 'headless' mode detected -> use 'Agg'")
        headless = True
        mpl.use('Agg')

elif platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Alternative for Mac: 'Qt5Agg', interplay with PyVista often a bit tricky otherwise

import matplotlib.pyplot as plt  # noqa
