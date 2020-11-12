import os
import platform
import matplotlib as mpl


headless = False
# Must be before importing matplotlib.pyplot or pylab!
if platform.system() == 'Linux':
    try:
        display = os.environ['DISPLAY']
        # mpl.use('TkAgg')

    except KeyError:
        print("Matplotlib - Backend: 'headless' mode detected-> use 'Agg'")
        headless = True
        # mpl.use('Agg')

elif platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Alternative for Mac: 'Qt5Agg'

import matplotlib.pyplot as plt
