import matplotlib as mpl
from wzk.mpl.figure import ieee1c, ieee2c

# fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)[source]

def __no_borders(pad=0.0):
    return {'figure.subplot.left': 0.0 + pad,
            'figure.subplot.right': 1.0 - pad,
            'figure.subplot.bottom': 0.0 + pad,
            'figure.subplot.top': 1.0 - pad}

def set_borders(left=0.125, right=0.9, bottom=0.1, top=0.9):
    mpl.rcParams.update({'figure.subplot.left': left,
                         'figure.subplot.right': right,
                         'figure.subplot.bottom': bottom,
                         'figure.subplot.top': top})


def set_style(s=('ieee', )):
    params = {}
    if 'ieee' in s:
        params.update({
            # 'text.usetex': False,
            'font.family': 'serif',
            'font.serif':  'CMU Serif, Times New Roman',  # , ['Times', 'Times New Roman', 'CMU Serif']
            'font.size': 8,
            'axes.linewidth': 1,
            'axes.labelsize': 8,
            'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,

            # 'figure.figsize': ieee2c,
            # 'figure.dpi': 300,

            'savefig.dpi': 300,
            'savefig.bbox': 'standard',
            'savefig.pad_inches': 0.1,
            'savefig.transparent': True
        })

    if 'no_borders' in s:
        params.update(__no_borders(pad=0.005))

    mpl.rcParams.update(params)
