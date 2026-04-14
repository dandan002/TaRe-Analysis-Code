# tare_analysis/analysis/data_loader.py
"""Shared data-loading helper used by main.py variant scripts (run_s6.py, etc.)."""
from pathlib import Path

from config import DATA_DIR
from xps.import_ import import_ThermoAlpha
from analysis.fitting import correct_data


def load_tare_data(data_dir: Path = DATA_DIR) -> dict:
    """Import all ThermoAlpha scans and return Shirley-corrected CorrectedData lists.

    Returns a dict with keys:
        scans            – raw list[XPSMeas]
        samples          – sorted list of sample name strings
        elements         – list of element strings (excluding XPS survey)
        indDict          – {element: [scan indices]}
        survey_inds      – indices of survey (XPS) scans
        Ta4fCorrected    – list[CorrectedData]
        Re4fCorrected    – list[CorrectedData]
    """
    BOE_files = [
        data_dir / 'TaRe_1124' / 'BOE_1124.xlsx',
        data_dir / 'TaRe_1201' / 'BOE_1201.xlsx',
        data_dir / 'TaRe_1208' / 'BOE_Small_1208.xlsx',
        data_dir / 'TaRe_1215' / 'BOE_Small_1215.xlsx',
    ]
    Control_files = [
        data_dir / 'TaRe_1124' / 'Control_1124.xlsx',
        data_dir / 'TaRe_1201' / 'Control_1201.xlsx',
        data_dir / 'TaRe_1208' / 'Control_1208.xlsx',
        data_dir / 'TaRe_1215' / 'Control_1215.xlsx',
    ]
    filepaths = BOE_files + Control_files
    missing = [str(f) for f in filepaths if not f.exists()]
    assert not missing, f'Missing files: {missing}'

    scans = []
    for fp in filepaths:
        scans += import_ThermoAlpha(fp)
    print(f'Imported {len(scans)} scans.')

    samples = sorted({s.sample for s in scans})
    elements = []
    for s in scans:
        if s.element != "XPS" and s.element not in elements:
            elements.append(s.element)

    indDict = {el: [] for el in elements}
    for i, s in enumerate(scans):
        if s.element in indDict:
            indDict[s.element].append(i)

    survey_inds = [i for i, s in enumerate(scans) if s.element == "XPS"]
    print("Samples:", samples)
    print("Elements:", elements)

    ta4f_inds = indDict.get("Ta4f", [])
    re4f_inds = indDict.get("Re4f", [])
    Ta4fCorrected = correct_data(scans, ta4f_inds, be_max=30.5)
    Re4fCorrected = correct_data(scans, re4f_inds, be_max=70.0)

    return {
        'scans': scans,
        'samples': samples,
        'elements': elements,
        'indDict': indDict,
        'survey_inds': survey_inds,
        'Ta4fCorrected': Ta4fCorrected,
        'Re4fCorrected': Re4fCorrected,
    }
