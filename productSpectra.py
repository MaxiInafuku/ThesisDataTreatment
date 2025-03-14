from constants import*
from functions import*
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

dataFolder = 'data/'
dataFile = 'TASavgMatrix.csv'
UVvisFile = 'absorbance30uJafter.csv'

#Set the delay at which you want to get the product spectrum.
testDelay = 1

#Remove the baseline from the UVvis static spectrum
baseline = 0.03

#Set gaussian parameters
sigma = 1               #nm
xAxisMax = 100 
A = 0.3                 #amplitude to generate product spectrum

#Scattering removal (see productSpectraTest)
lowCut = 375            #nm
highCut = 440           #nm

#Load and plot the DOD, invert DODwavelength and singleDelaySpectrum, because if they're not 
DODdf = pd.read_csv(dataFolder+dataFile, header = None, na_values=["Inf", "-Inf", "NaN"])
DODdf.replace([np.inf, -np.inf], np.nan, inplace=True)
DODwavelength = DODdf.iloc[1:,0].to_numpy(dtype=float)[::-1]
DODdelay = DODdf.iloc[0,1:].to_numpy(dtype=float)
DODspectrum = DODdf.iloc[1:,1:].to_numpy(dtype=float)

delayIndex = np.searchsorted(DODdelay, testDelay, side="left")
closestTime = DODdelay[delayIndex]
print('Closest time: {:.4g}ps' .format(closestTime))

singleDelaySpectrum = DODspectrum[:,delayIndex][::-1]


#Load the UVvis, subtract the baseline
UVvisData = np.loadtxt(dataFolder+UVvisFile, delimiter = ',')
UVvisDataWavelength = UVvisData[:,0][::-1]
UVvisDataSpectrum = (UVvisData[:,1]-baseline)[::-1]

Figure, ax=plt.subplots()
plt.title('DOD at {:.4g}ps' .format(closestTime))
ax.plot(DODwavelength, singleDelaySpectrum)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('DOD')
plt.show()

#Convolute
UVvisConvoluted, gaussianKernel = convolutionGaussian(UVvisDataSpectrum, sigma, xAxisMax)
maximums, _ = find_peaks(UVvisDataSpectrum/max(UVvisDataSpectrum), height=0.5 )

Figure, ax = plt.subplots()
ax.plot(range(len(gaussianKernel)), gaussianKernel/max(gaussianKernel), linestyle = '--',label = 'Gaussian')
ax.plot(range(len(UVvisDataSpectrum)), UVvisDataSpectrum/UVvisDataSpectrum[maximums[0]], linestyle = '--', label = 'Data')
ax.plot(range(len(UVvisConvoluted)), UVvisConvoluted/max(UVvisConvoluted), label = 'Convolution')
ax.set_xlabel('pixel')
ax.legend()
plt.show()

DODInterpolated = funcInterpolate(DODwavelength, singleDelaySpectrum, UVvisDataWavelength)

#trimm data
lowCutIndex = np.searchsorted(UVvisDataWavelength, lowCut, side="left")
highCutIndex = np.searchsorted(UVvisDataWavelength, highCut, side="right")
UVvisDataWavelength = np.delete(UVvisDataWavelength, np.arange(lowCutIndex,highCutIndex+1))
UVvisConvoluted = np.delete(UVvisConvoluted, np.arange(lowCutIndex,highCutIndex+1))
DODInterpolated = np.delete(DODInterpolated, np.arange(lowCutIndex,highCutIndex+1))

productSpetrum = DODInterpolated + A*UVvisConvoluted

Figure, ax = plt.subplots()
ax.scatter(UVvisDataWavelength, -UVvisConvoluted*A, linestyle = '--', label = 'UV vis', s=1)
ax.scatter(UVvisDataWavelength, DODInterpolated, linestyle = '--', label = 'DOD', s=1)
ax.scatter(UVvisDataWavelength, productSpetrum, label = 'prod spectrum', s=1)
ax.set_xlabel('Wavelength(nm)')
ax.set_ylabel('DOD')
ax.legend()
plt.show()