import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from xps.peaks import peakDict, augerDict

# Class for the XPS data
class XPSMeas:
    def __init__(self, sample=None, xrayEnergy=None, element=None, comment=None, sweeps=None, BE=None, intensity=None,
                 intensityErr=None, background=None, datestamp=None):
        self.sample = sample
        self.xrayEnergy = xrayEnergy          # [eV] incident beam energy
        self.element = element                # element and orbital, str
        self.comment = comment                # measurement comment
        self.sweeps = sweeps                  # number of data sweeps taken for this run
        self.BE = BE                          # [eV] Binding of photoelectrons
        self.intensity = intensity            # [counts] photoelectron intensity
        self.intensityErr = intensityErr      # [counts] error in photoelectron intensity
        self.background = background          # [counts] calculated background. Includes all contributions
        self.datestamp = datestamp

class XPSMeas_ion_milled:
    def __init__(self, sample=None, xrayEnergy=None, element=None, comment=None, sweeps=None, BE=None, intensity=None,
                 intensityErr=None, background=None, datestamp=None, etchlevel=None, etchtime=None):
        self.sample = sample
        self.xrayEnergy = xrayEnergy        # [eV] incident beam energy
        self.element = element              # element and orbital, str
        self.comment = comment
        self.sweeps = sweeps                # number of data sweeps taken for this run
        self.BE = BE                        # [eV] Binding of photoelectrons
        self.intensity = intensity          # [counts] photoelectron intensity
        self.intensityErr = intensityErr    # [counts] error in photoelectron intensity
        self.background = background        # [counts] calculated background. Includes all contributions
        self.datestamp = datestamp
        self.etchtime = etchtime
        self.etchlevel = etchlevel


# Shifts all binding energies to match a given reference. Typically Au or Ag as of Feb 2022
def xShiftCorrection(uncorrectedData, sampleName, elementName, expectedBindingEnergy):
    # List to store (xray_energy, binding_energy_shift) tuples
    shiftXAxis = []

    # Loop through all data elements to find matching sample and element data
    for i in range(len(uncorrectedData)):
        if uncorrectedData[i].sample == sampleName and uncorrectedData[i].element == elementName:
            # Find the peak position by getting index of maximum intensity
            maxIndex = uncorrectedData[i].intensity.argmax()

            # Calculate shift needed and store if this xray energy hasn't been processed
            if uncorrectedData[i].xrayEnergy not in [x[0] for x in shiftXAxis]:
                shift = round(expectedBindingEnergy - uncorrectedData[i].BE[maxIndex], 2)
                shiftXAxis.append((uncorrectedData[i].xrayEnergy, shift))

    # Create deep copy of data to store corrected values
    correctedData = copy.deepcopy(uncorrectedData)

    # Apply the calculated binding energy shifts to all matching data
    for i in range(len(correctedData)):
        for j in range(len(shiftXAxis)):
            if correctedData[i].xrayEnergy == shiftXAxis[j][0]:
                correctedData[i].BE = correctedData[i].BE + shiftXAxis[j][1]

    return correctedData


# Iterates the Shirley background until the maximum pointwise change between
# successive background estimates falls below shirleyCutoff * step_height,
# or until max_iter iterations are reached.
def iteratedShirleyCorrect(Y, YErr, shirleyCutoff, max_iter=50):
    step_height = abs(Y[0] - Y[-1])
    background = np.zeros_like(Y)

    for iteration in range(max_iter):
        residual = Y - background
        At = np.sum(residual)
        if At <= 0:
            break
        new_background = np.zeros_like(Y)
        for i in range(len(Y)):
            g_i = np.sum(residual[i:(len(residual) - 1)]) / At
            new_background[i] = g_i * step_height

        change = np.max(np.abs(new_background - background))
        background = new_background
        if change < shirleyCutoff * max(step_height, 1e-12):
            break

    newY = np.maximum(Y - background, 0)

    # Propagate errors: background uncertainty approximated from endpoint errors.
    # Each b_i = g_i * (Y[0]-Y[-1]), so var(b_i) ≈ g_i^2 * (YErr[0]^2 + YErr[-1]^2)
    endpoint_var = YErr[0] ** 2 + YErr[-1] ** 2
    At_final = max(np.sum(Y - background), 1e-12)
    newYErr = np.zeros_like(YErr)
    for i in range(len(Y)):
        g_i = np.sum((Y - background)[i:(len(Y) - 1)]) / At_final
        b_err_sq = g_i ** 2 * endpoint_var
        newYErr[i] = np.sqrt(YErr[i] ** 2 + b_err_sq)

    return newY, newYErr, iteration + 1

# Background corrects the data using either a flat, linear, or shirley background.
# lowBETarget and highBETarget are the binding energies (in eV) where the
# background is estimated. BEWindow is the width (in eV) around these targets to average over.
# shirleyCutoff is the fractional change in area under the curve used to determine when to
# stop iterating the shirley background correction.
def backgroundCorrect(xShiftData, lowBETarget, highBETarget, BEWindow=0, func='shirley', shirleyCutoff=0.01):
    correctedData = copy.deepcopy(xShiftData)

    # locate end values
    lowBEinds = []
    highBEinds = []
    for ind, energyPoint in enumerate(xShiftData.BE):
        if energyPoint >= lowBETarget - BEWindow / 2 and energyPoint <= lowBETarget + BEWindow / 2:
            lowBEinds.append(ind)
        if energyPoint >= highBETarget - BEWindow / 2 and energyPoint <= highBETarget + BEWindow / 2:
            highBEinds.append(ind)

    # Handle empty indices lists - use nearest point if no points in window
    if len(highBEinds) == 0:
        highBEinds = [np.argmin(np.abs(xShiftData.BE - highBETarget))]
    if len(lowBEinds) == 0:
        lowBEinds = [np.argmin(np.abs(xShiftData.BE - lowBETarget))]

    highBEcenterInd = round(np.mean(highBEinds))
    lowBEcenterInd = round(np.mean(lowBEinds))

    # copy into new lists
    plotXBG = xShiftData.BE[highBEcenterInd:lowBEcenterInd]
    plotYBG = xShiftData.intensity[highBEcenterInd:lowBEcenterInd]
    plotYBGerr = xShiftData.intensityErr[highBEcenterInd:lowBEcenterInd]

    if func == 'flat':
        # flat background correct - handle empty indices safely
        lowBEval = np.nanmean([xShiftData.intensity[i] for i in lowBEinds]) if len(lowBEinds) > 0 else 0.0
        lowBEerr_list = [xShiftData.intensityErr[i] ** 2 for i in lowBEinds]
        lowBEerr = np.sqrt(np.nansum(lowBEerr_list) / len(lowBEinds)) if len(lowBEinds) > 0 else 1.0

        xShiftDataBGLen = len(plotXBG)

        B1 = np.full(xShiftDataBGLen, lowBEval)
        B1err = np.full(xShiftDataBGLen, lowBEerr)

        correctedData.intensity = np.array(plotYBG) - B1
        correctedData.intensityErr = np.sqrt(np.array(plotYBGerr) ** 2 + B1err ** 2)
        correctedData.background = B1

    elif func == 'linear':
        lowBEval = np.nanmean([xShiftData.intensity[i] for i in lowBEinds]) if len(lowBEinds) > 0 else 0.0
        lowBEerr_list = [xShiftData.intensityErr[i] ** 2 for i in lowBEinds]
        lowBEerr = np.sqrt(np.nansum(lowBEerr_list) / len(lowBEinds)) if len(lowBEinds) > 0 else 1.0

        highBEval = np.nanmean([xShiftData.intensity[i] for i in highBEinds]) if len(highBEinds) > 0 else 0.0
        highBEerr_list = [xShiftData.intensityErr[i] ** 2 for i in highBEinds]
        highBEerr = np.sqrt(np.nansum(highBEerr_list) / len(highBEinds)) if len(highBEinds) > 0 else 1.0

        lowBE_mean = np.nanmean([xShiftData.BE[i] for i in lowBEinds]) if len(lowBEinds) > 0 else lowBETarget
        highBE_mean = np.nanmean([xShiftData.BE[i] for i in highBEinds]) if len(highBEinds) > 0 else highBETarget

        deltaBE = highBE_mean - lowBE_mean
        if np.isclose(deltaBE, 0):
            deltaBE = 1.0  # avoid division by zero

        slope = (highBEval - lowBEval) / deltaBE

        B1 = (plotXBG - lowBE_mean) * slope + lowBEval
        B1err = np.sqrt((plotXBG / deltaBE * highBEerr) ** 2 + ((1 - plotXBG / deltaBE) * lowBEerr) ** 2)

        correctedData.intensity = np.array(plotYBG) - B1
        correctedData.intensityErr = np.sqrt(np.array(plotYBGerr) ** 2 + B1err ** 2)
        correctedData.background = B1

    elif func == 'shirley':
        lowBEval = np.nanmean([xShiftData.intensity[i] for i in lowBEinds]) if len(lowBEinds) > 0 else 0.0
        lowBEerr_list = [xShiftData.intensityErr[i] ** 2 for i in lowBEinds]
        lowBEerr = np.sqrt(np.nansum(lowBEerr_list) / len(lowBEinds)) if len(lowBEinds) > 0 else 1.0

        xShiftDataBGLen = len(plotXBG)

        B1 = np.full(xShiftDataBGLen, lowBEval)
        B1err = np.full(xShiftDataBGLen, lowBEerr)

        currY = np.array(plotYBG) - B1
        currYErr = np.sqrt(np.array(plotYBGerr) ** 2 + B1err ** 2)

        # shirley correct
        correctedData.intensity, correctedData.intensityErr, _ = iteratedShirleyCorrect(currY, currYErr,
                                                                                         shirleyCutoff)
        correctedData.background = B1 + currY - correctedData.intensity

    else:
        assert False, 'Unknown B1func'

    correctedData.BE = plotXBG

    return correctedData


def plotElement(element, dataInds, xShiftData):
    """
    Plot XPS data for a specific element across multiple measurements.

    Args:
        element (str): Element name to plot
        dataInds (list): List of indices into xShiftData to plot
        xShiftData (list): List of XPSMeas objects containing measurement data

    Returns:
        matplotlib.figure.Figure: The generated plot figure
    """
    # Set matplotlib style parameters
    font_names = [f.name for f in fm.fontManager.ttflist]
    mpl.rcParams['font.family'] = 'Arial Unicode MS'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    colorCode = ['b', 'g', 'r', 'grey', 'olive', 'cyan', 'orange', 'brown']

    # Create figure and axis
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    # Configure axis display
    ax.invert_xaxis()
    ax.set_title(element + ', ' + xShiftData[dataInds[0]].sample, fontsize=20, pad=10)
    ax.set_xlabel('Binding Energy (eV)', fontsize=18, labelpad=10)
    ax.set_ylabel('Counts', fontsize=18, labelpad=10)

    # Plot data for each measurement
    for pltInd in dataInds:
        ax.errorbar(xShiftData[pltInd].BE,
                   xShiftData[pltInd].intensity,
                   xShiftData[pltInd].intensityErr,
                   linewidth=2,
                   fmt='.:',
                   capsize=3,
                   label=str(xShiftData[pltInd].xrayEnergy) + ' eV')

    ax.legend(bbox_to_anchor=(1, 1), loc='best', fontsize=12)

    return fig


def calculateNoiseFloorRatio(dataInds, xShiftData, lowBE, highBE):
    """
    Calculate the ratio between measured noise floor and expected Poisson noise floor.

    Args:
        dataInds (list): List of indices into xShiftData to analyze
        xShiftData (list): List of XPSMeas objects containing measurement data
        lowBE (float): Lower binding energy boundary for noise analysis
        highBE (float): Higher binding energy boundary for noise analysis

    Returns:
        list: Ratio of measured noise floor to Poisson noise floor for each dataset
    """
    # Initialize empty lists to store results
    noiseFloor = [[] for i in enumerate(dataInds)]          # Measured noise floor
    poissonNoiseFloor = [[] for i in enumerate(dataInds)]   # Expected Poisson noise floor
    noiseFloorRatio = [[] for i in enumerate(dataInds)]     # Ratio between measured and expected

    for i, dataInd in enumerate(dataInds):
        # Find indices closest to specified binding energy boundaries
        lowBEInd = np.abs(np.array(xShiftData[dataInd].BE) - lowBE).argmin()
        highBEInd = np.abs(np.array(xShiftData[dataInd].BE) - highBE).argmin()

        # Calculate measured noise floor (standard deviation of intensity)
        noiseFloor[i] = np.std(xShiftData[dataInd].intensity[highBEInd:lowBEInd])

        # Calculate expected Poisson noise floor (sqrt of mean intensity)
        poissonNoiseFloor[i] = np.sqrt(np.mean(xShiftData[dataInd].intensity[highBEInd:lowBEInd]))

        # Calculate ratio between measured and expected noise floors
        noiseFloorRatio[i] = noiseFloor[i] / poissonNoiseFloor[i]

    return noiseFloorRatio


def addPeakLabel(hax, element, peaks=None, auger=False, xrayEn=0):
    """
    Add vertical lines and labels for XPS peaks and Auger peaks to a matplotlib plot.

    Args:
        hax: matplotlib axis handle to add labels to
        element (str): Chemical element symbol to label peaks for
        peaks (list, optional): Specific peaks to label. If None, all peaks are labeled
        auger (bool): Whether to include Auger peaks
        xrayEn (float): X-ray energy in eV, required for Auger peak positions

    Returns:
        matplotlib.axes: The axis handle with added peak labels
    """
    # Add photoelectron peak labels if element exists in peak dictionary
    if element in peakDict.keys():
        BEDict = peakDict[element]
        # Use all peaks if none specified
        if peaks is None:
            BEpeaks = BEDict.keys()
        else:
            BEpeaks = peaks
        # Add vertical line and rotated text label for each peak
        for peak in BEpeaks:
            if peak in BEDict.keys():
                xloc = BEDict[peak]
                hax.axvline(xloc, color='k', alpha=0.35)
                hax.annotate(' ' + element + peak, (xloc, hax.get_ylim()[1]),
                             rotation='vertical', verticalalignment='bottom', horizontalalignment='center')

    # Add Auger peak labels if requested and element exists in Auger dictionary
    if auger:
        if xrayEn == 0:
            assert False, 'set xrayEn to a finite value when plotting auger peaks'
        if element in augerDict.keys():
            KEDict = augerDict[element]
            # Use all peaks if none specified
            if peaks is None:
                KEpeaks = KEDict.keys()
            else:
                KEpeaks = peaks
            # Add dotted vertical line and rotated text label for each Auger peak
            for peak in KEpeaks:
                if peak in KEDict.keys():
                    xloc = xrayEn - KEDict[peak]  # Convert KE to BE
                    hax.axvline(xloc, linestyle=':', color='k', alpha=0.35)
                    hax.annotate(' ' + element + ' ' + peak, (xloc, hax.get_ylim()[1]),
                                 rotation='vertical', verticalalignment='bottom', horizontalalignment='center')

    return hax
