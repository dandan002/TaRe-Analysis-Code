# source for all data copied here:
# Handbook of X-ray Photoelectron Spectroscopy, CD Wagner, WM Riggs, LE Davis, JF Moulder, and GE Mullenberg (1979)

# Dictionary of binding energy peaks for the addPeakLabel function. Format is:
# {<string label of the element>: {<string label of the peak>: <binding energy of peak position>, ...}
peakDict = {'Ta': {'4f': 19,  # doublet
                   '5p': 41,  # doublet
                   '5s': 71,
                   '4d': 235,  # doublet
                   '4p_1/2': 464,
                   '4p_3/2': 403,
                   '4s': 566},
            'O': {'1s': 531,
                  '2s': 23},
            'C': {'1s': 287},
            'F': {'1s': 686,
                  '2s': 30},
            'Na': {'1s': 1072,
                   '2s': 64,
                   '2p': 31},
            'Mg': {'1s': 1305,
                   '2s': 90,
                   '2p': 51},
            'Al': {'2s': 119,
                   '2p': 74},
            'Si': {'2s': 153,
                   '2p': 102.5},  # doublet
            'Cl': {'2s': 270,
                   '2p': 200,  # doublet
                   '3s': 17},
            'K': {'2s': 378,
                  '2p': 294,
                  '3s': 33,
                  '3p': 17},  # doublet
            'Nb': {'3s': 470,
                   '3p_1/2': 379,
                   '3p_3/2': 364,
                   '3d': 207,  # doublet
                   '4s': 59,
                   '4p': 35},
            'Mo': {'3s': 470,
                   '3p_1/2': 379,
                   '3p_3/2': 364,
                   '3d': 207,  # doublet
                   '4s': 59,
                   '4p': 35
                   },
            'Pd': {'3s': 673,
                   '3p_1/2': 561,
                   '3p_3/2': 534,
                   '3d': 340,  # doublet
                   '4s': 88,
                   '4p_3/2': 54
                   },
            'Re': {'4s': 628,
                   '4p_1/2': 521,
                   '4p_3/2': 449,
                   '4d_3/2': 227,
                   '4d_5/2': 263,
                   '4f': 44,  # doublet
                   '5s': 81,
                   '5p_1/2': 44,
                   '5p_3/2': 33
                    },
            'Pt': {'4s': 726,
                   '4p_1/2': 610,
                   '4p_3/2': 521,
                   '4d_3/2': 333,
                   '4d_5/2': 316,
                   '4f': 74,  # doublet
                   '5s': 105,
                   '5p_1/2': 69,
                   '5p_3/2': 53
                   },
            'Au': {'4s': 763,
                   '4p_1/2': 643,
                   '4p_3/2': 547,
                   '4d_3/2': 354,
                   '4d_1/2': 336,
                   '4f': 87,
                   '5s': 110,
                   '5p_1/2': 75,
                   '5p_3/2': 57}
            }

# Dictionary of auger energy peaks for the addPeakLabel function. Format is:
# {<string label of the element>: {<string label of the peak>, <kinetic energy of peak position>}
augerDict = {'O': {'KL23L23': 1487-976,
                  'KL1L23': 1487-997,
                  'KL1L1': 1487-1012},
            'C': {'KL23L23': 1487-1226},
            'F': {'KL1L1': 1487-878,
                  'KL1L23': 1487-859,
                  'KL23L23': 1487-832},
            'Na': {'KL1L1': 1487-565,
                   'KL1L23': 1487-536,
                   'KL23L23': 1478-497},
            'Mg': {'KL1L1': 1487-384,
                   'KL1L23': 1487-350,
                   'KL23L23': 1487-305},
            'Al': {'KL23L23': 1487-100},
            'Cl': {'LMM': 1487-1304},
            'K': {'LMM': 1487-1237},  # doublet
            'Nb': {'MNV': 1487-1321,
                   'MNN': 1487-1289},
            'Mo': {'MNV': 1487-1301,
                   'MNN': 1487-1266},
            'Pd': {'MNV': 1487-1212,
                   'MNN': 1487-1161},
            'Pt': {'NOO': 1487-1425},
            'Au': {'NOO': 1487-1417}
            }
