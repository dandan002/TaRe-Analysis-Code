# tare_analysis/plots/style.py
import matplotlib as mpl

OKABE_ITO = {
    'blue':       '#0072B2',
    'sky':        '#56B4E9',
    'green':      '#009E73',
    'yellow':     '#F0E442',
    'orange':     '#E69F00',
    'vermillion': '#D55E00',
    'pink':       '#CC79A7',
    'black':      '#000000',
}

SPECIES_COLOR = {
    'metal':     OKABE_ITO['blue'],
    'interface': OKABE_ITO['sky'],
    'alloy':     OKABE_ITO['green'],
    'Ta1':       OKABE_ITO['orange'],
    'Ta3':       OKABE_ITO['yellow'],
    'Ta5':       OKABE_ITO['vermillion'],
    'Re_metal':  OKABE_ITO['blue'],
    'ReO2':      OKABE_ITO['orange'],
    'ReO3':      OKABE_ITO['vermillion'],
    'Re2O7':     OKABE_ITO['pink'],
    # S7 two-component oxide
    'Ta5a':      OKABE_ITO['vermillion'],
    'Ta5b':      OKABE_ITO['pink'],
}

SPECIES_LABEL = {
    'metal':     'Ta metal',
    'interface': 'Ta interface',
    'alloy':     r'Ta–Re alloy',
    'Ta1':       r'Ta$^{1+}$',
    'Ta3':       r'Ta$^{3+}$',
    'Ta5':       r'Ta$_2$O$_5$',
    'Re_metal':  'Re metal',
    'ReO2':      r'ReO$_2$',
    'ReO3':      r'ReO$_3$',
    'Re2O7':     r'Re$_2$O$_7$',
    # S7 two-component oxide
    'Ta5a':      r'Ta$_2$O$_5$ (stoich.)',
    'Ta5b':      r'TaO$_x$ (sub-stoich.)',
}

GROUP_COLOR = {
    'BOE':     OKABE_ITO['blue'],
    'Control': OKABE_ITO['vermillion'],
}

OVERLAY_CYCLE = [v for k, v in OKABE_ITO.items() if k != 'yellow']

# APS figure sizes (inches)
FIG_SINGLE      = (3.375, 2.8)
FIG_SINGLE_T    = (3.375, 3.5)   # tall single-column (fit components)
FIG_SINGLE_TALL = (3.375, 5.0)   # extra-tall single-column (2-panel stacked)
FIG_DOUBLE      = (6.75,  3.0)
FIG_DOUBLE_T    = (6.75,  3.5)   # tall double-column

mpl.rcParams.update({
    "figure.dpi":        300,
    "figure.figsize":    FIG_SINGLE,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "lines.linewidth":   1.0,
    "legend.frameon":    False,
})
