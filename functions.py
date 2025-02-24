import numpy as np
from scipy import integrate
from scipy import interpolate
from constants import * 
from scipy.optimize import least_squares
import inspect

#------------------------------------------#
#-----------------DRUDE--------------------#
#------------------------------------------#
def funcDrude(w, wp, epsilonInf,gamma):
    epsilonDreal = epsilonInf - wp**2/(w**2 + gamma**2)
    epsilonDimag = wp**2 *gamma / (w*(w**2+gamma**2))
    return epsilonDreal, epsilonDimag

def funcDrudeReal(w, wp, epsilonInf, gamma):
    return epsilonInf - wp**2/(w**2 + gamma**2)

def funcDrudeImag(w,wp,gamma):
    return wp**2 *gamma / (w*(w**2+gamma**2))

def funcDrudeImagBaseline(w,wp,gamma,y0):
    return wp**2 *gamma / (w*(w**2+gamma**2)) + y0

def funcDrudeImagRes(params, w, data):
    wp, gamma, y0 = params
    return funcDrudeImagBaseline(np.array(w), wp, gamma, y0) - np.array(data)

#r in nanometer. 
#Other parameters in Hz if freqBoolean = True
#in eV if freqBoolean = False
def funcDrudeSize(w, wp, gammaBulkDrude, r, freqBoolean):
    w = np.asarray(w, dtype=float)
    if freqBoolean == False:
        gamma = gammaBulkDrude + Cnuf/r
    elif freqBoolean == True:
        gamma = gammaBulkDrude + C*nuf/r
    else: 
        'FreqBoolean must be a Boolean'

    return 1 - wp**2 / (w**2 + gamma*w*1j)


#------------------------------------------#
#--------IMAGINARY INTERBAND---------------#
#------------------------------------------#
def funcFermi(E,Ef,T):
    return 1 / (1 + np.exp((E-Ef)/(k*T)))

def funcDielIBrealIntegrand(x, E, Eg, gammaIB, Ef, T):
    return (x**2-E**2+gammaIB**2)/((x**2-E**2+gammaIB**2)**2 + 4 * E**2 * gammaIB**2) * (1-funcFermi(x,Ef,T)) * np.sqrt(x-Eg)/x

#divide by sqrt hbar to integrate in the energy domain, instead of frequency
def funcDielIBreal(E, Eg, gammaIB, Kbulk, Ef, T):
    integral, err = integrate.quad(funcDielIBrealIntegrand, Eg, np.inf, args = (E, Eg, gammaIB, Ef, T))
    return Kbulk*integral*hbarE**(3/2)


def funcDielIBimagIntegrand(x, E, Eg, gammaIB, Ef, T):
    return 2*E*gammaIB/((x**2-E**2+gammaIB**2)**2 + 4 * E**2 * gammaIB**2) * (1-funcFermi(x,Ef,T)) * np.sqrt(x-Eg)/x

#divide by sqrt hbar to integrate in the energy domain, instead of frequency
def funcDielIBimag(E, Eg, gammaIB, Kbulk, Ef, T):
    integral, err = integrate.quad(funcDielIBimagIntegrand, Eg, np.inf, args = (E, Eg, gammaIB, Ef, T))
    return Kbulk*integral*hbarE**(3/2)

'''def funcDielIBrealSize(E, Eg, gammaIB, r, Kbulk, Ef, T):
    Ksize = Kbulk*(1-np.exp(-r/r0))
    integral, err = integrate.quad(funcDielIBrealIntegrand, Eg, np.inf, args = (E, Eg, gammaIB, Ef, T))
    return Ksize*integral/hbarE**(3/2)

def funcDielIBimagSize(E, Eg, gammaIB, r, Kbulk, Ef, T):
    Ksize = Kbulk*(1-np.exp(-r/r0))
    integral, err = integrate.quad(funcDielIBimagIntegrand, Eg, np.inf, args = (E, Eg, gammaIB, Ef, T))
    return Ksize*integral/hbarE**(3/2)'''

def funcDielIBsizeIntegrand(x,E,Eg,gammaIB,Ef,T):
    return np.sqrt(x-Eg)/x * (1-funcFermi(x,Ef,T)) * ((x**2 - E**2 + gammaIB**2 + E*gammaIB*2j)) / ((x**2 - E**2 + gammaIB**2)**2 + 4*E**2*gammaIB**2)

def realIntegrand(x,E,Eg,gammaIB,Ef,T):
    return funcDielIBsizeIntegrand(x,E,Eg,gammaIB,Ef,T).real

def imagIntegrand(x,E,Eg,gammaIB,Ef,T):
    return funcDielIBsizeIntegrand(x,E,Eg,gammaIB,Ef,T).imag

def funcDielIBsize(E,r,Eg,gammaIB,Ef,T):
    Ksize = Kbulk*(1-np.exp(-r/r0))
    realIntegral, realErr = integrate.quad(lambda x: funcDielIBsizeIntegrand(x,E,Eg, gammaIB, Ef,T).real, Eg, np.inf)
    imagIntegral, imagErr = integrate.quad(lambda x: funcDielIBsizeIntegrand(x,E,Eg, gammaIB, Ef,T).imag, Eg, np.inf)
    
    return Ksize*hbarE**(3/2)*(np.array(realIntegral) + np.array(imagIntegral)*1j)

#------------------------------------------#
#--------EXTINCTION COEFFICIENT------------#
#------------------------------------------#
#Absorbance
def funcAbsorbanceHodak(energies, diel, dielMedium):
    abs = dielMedium.real**3/2 * energies*diel.imag / ((diel.real + 2*dielMedium.real)**2 + diel.imag**2)
    return abs


#Wavelength of medium (nm), Energies (eV), radius (nm), medium complex refraction index, real part and imaginary part of the dielectric function. 
#We assume there's no shell, which is equivalent to say r=r'
def funcExtCoeff(r, wavelength, diel, nMedium):
    wavelength = np.asarray(wavelength)
    dielMedium = refractionIndexToDielectricFunc(nMedium)
    
    polarizability = 4*np.pi*r**3 * (diel-dielMedium)/(diel+2*dielMedium)

    k = 2*np.pi*nMedium.real/wavelength
    Cext = k* polarizability.imag
    Qext = Cext/(2*np.pi*r**2)

    return Qext

#p163 Absorption and scattering of light by small particles - Bohren 1998
def funcScaCoeffCoreShell(rCore, rTot, wavelength, dielCore, dielShell, nMedium):
    wavelength = np.asarray(wavelength)
    dielMedium = refractionIndexToDielectricFunc(nMedium)

    f = (rCore/rTot)**3
    polarizability = 4*np.pi*rTot**3 * ((dielShell-dielMedium)*(dielCore+2*dielShell)+f*(dielCore-dielShell)*(dielMedium+2*dielShell))/((dielShell+2*dielMedium)*(dielCore+2*dielShell)+f*(2*dielShell-2*dielMedium)*(dielCore-dielShell))

    k = 2*np.pi*nMedium.real/wavelength
    Csca = k**4 * np.abs(polarizability)**2 / (6*np.pi)
    Qsca = Csca/(2*np.pi*rTot**2)

    return Qsca


def funcAbsCoeffCoreShell(rCore, rTot, wavelength, dielCore, dielShell, nMedium):
    wavelength = np.asarray(wavelength)
    dielMedium = refractionIndexToDielectricFunc(nMedium)

    f = (rCore/rTot)**3
    polarizability = 4*np.pi*rTot**3 * ((dielShell-dielMedium)*(dielCore+2*dielShell)+f*(dielCore-dielShell)*(dielMedium+2*dielShell))/((dielShell+2*dielMedium)*(dielCore+2*dielShell)+f*(2*dielShell-2*dielMedium)*(dielCore-dielShell))

    k = 2*np.pi*nMedium.real/wavelength
    Cabs = k* polarizability.imag
    Qabs = Cabs/(2*np.pi*rTot**2)

    return Qabs

def funcExtCoeffCoreShell(rCore, rTot, wavelength, dielCore, dielShell, nMedium):
    return funcScaCoeffCoreShell(rCore, rTot, wavelength, dielCore, dielShell, nMedium) + funcAbsCoeffCoreShell(rCore, rTot, wavelength, dielCore, dielShell, nMedium)

def singleFullAbsCoeffCore(r, E, wp, gammaBulk, Eg, gammaIB, Ef, T, nMedium):
    diel = funcDrudeSize(E, wp, gammaBulk, r, False) + funcDielIBsize(E, r, Eg, gammaIB, Ef, T)
    wavelength = hE*c*1e-6/E
    dielMedium = refractionIndexToDielectricFunc(nMedium)

    polarizability = 4*np.pi*r**3 * (diel-dielMedium)/(diel+2*dielMedium)

    k = 2*np.pi*nMedium.real/wavelength
    Cabs = k* polarizability.imag
    Qabs = Cabs/(2*np.pi*r**2)

    return Qabs

def fullAbsCoeffCore(r, Es, Ep, GammaBulk, Eg, GammaIB, Ef, T, nMedium):
    if Es.size == nMedium.size:
        return np.array([singleFullAbsCoeffCore(r, Es[i], Ep, GammaBulk, Eg, GammaIB, Ef, T, nMedium[i]) for i in range(len(Es))])
    else:
        print('Number of elements in energies are not equal to the ones in nMedium. Check those arrays.')

def singleFullAbsCoeffCoreShell(rCore, rTot, E, wp, gammaBulk, Eg, gammaIB, Ef, T, dielShell, nMedium):
    dielCore = funcDrudeSize(E, wp, gammaBulk, rCore, False) + funcDielIBsize(E, rCore, Eg, gammaIB, Ef, T)
    wavelength = hE*c*1e-6/E
    dielMedium = refractionIndexToDielectricFunc(nMedium)
    
    f = (rCore/rTot)**3
    polarizability = 4*np.pi*rTot**3 * ((dielShell-dielMedium)*(dielCore+2*dielShell)+f*(dielCore-dielShell)*(dielMedium+2*dielShell))/((dielShell+2*dielMedium)*(dielCore+2*dielShell)+f*(2*dielShell-2*dielMedium)*(dielCore-dielShell))
    k = 2*np.pi*nMedium.real/wavelength
    Cabs = k* polarizability.imag
    Qabs = Cabs/(2*np.pi*rTot**2)

    return Qabs

#------------------------------------------#
#-------------ABS FITTING------------------#
#------------------------------------------#
def funcDielIBsizeFit(E,r,Eg,gammaIB,Ef,T):
    realIntegral, realErr = integrate.quad(lambda x: funcDielIBsizeIntegrand(x,E,Eg, gammaIB, Ef,T).real, Eg, np.inf)
    realIntegral = np.nan_to_num(realIntegral, nan=0, posinf=0, neginf=0)
    imagIntegral, imagErr = integrate.quad(lambda x: funcDielIBsizeIntegrand(x,E,Eg, gammaIB, Ef,T).imag, Eg, np.inf)
    imagIntegral = np.nan_to_num(imagIntegral, nan=0, posinf=0, neginf=0)
    Ksize = Kbulk*hbarE**(3/2)*(1-np.exp(-r/r0))
    return Ksize*(np.array(realIntegral) + np.array(imagIntegral)*1j)

def singleFullAbsCoeffCoreFit(E, A, r, Ep, gammaBulk, Eg, gammaIB, Ef, T, nMedium):
    diel = funcDrudeSize(E, Ep, gammaBulk, r, False) + funcDielIBsizeFit(E,r,Eg, gammaIB, Ef, T)
    wavelength = hE*c*1e-6/E
    dielMedium = refractionIndexToDielectricFunc(nMedium)

    polarizability = (diel-dielMedium)/(diel+2*dielMedium)

    k = nMedium.real/wavelength
    Cabs = k* polarizability.imag

    return A*Cabs

def fullAbsCoeffCoreFit(Es, A, Ep, gammaBulk, r, Eg, gammaIB, Ef, T, nMedium):
    if len(Es) == len(nMedium):
        return np.array([singleFullAbsCoeffCoreFit(Es[i], A, r, Ep, gammaBulk, Eg, gammaIB, Ef, T, nMedium[i]) for i in range(len(Es))])
    else: 
        print('Energies has length {:.4g} and nMedium has length {:.4g}' .format(len(Es), len(nMedium)))
    

#------------------------------------------#
#----------------UTILS---------------------#
#------------------------------------------#
#given two sets of data, each one with its own x axis and y axis, this function interpolates to a third axis and returns a numpy array of complex numbers
def convertToInterpolatedComplex(xReal,yReal, xImag,yImag, xInterpolated):
    interpolatedComplex = np.empty(xInterpolated.shape, dtype=complex)
    
    realInterpolator = interpolate.interp1d(xReal,yReal,kind='cubic', fill_value='extrapolate')
    interpolatedComplex.real = realInterpolator(xInterpolated)
    
    imagInterpolator = interpolate.interp1d(xImag,yImag,kind='cubic', fill_value='extrapolate')
    interpolatedComplex.imag = imagInterpolator(xInterpolated)

    return interpolatedComplex

#Input: complex np.array
#Ouput: complex np.array
def refractionIndexToDielectricFunc(n):
    #If n is a np array do nothing, otherwise make it a np.array so multiplication is element wise.
    n = np.asarray(n)
    diel = np.empty(n.real.shape, dtype=complex)
    diel.real = n.real**2 - n.imag**2
    diel.imag = 2*n.real*n.imag

    return diel

def logNormalDistribution(x,mean,std):
    mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
    sigma = np.sqrt(np.log(1 + (std**2/mean**2)))
    return 1/(x*sigma*np.sqrt(2*np.pi)) * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))

#arguments = all but the integration variable
def average(mean, std, f, *args):
    def integrand(x):
        return f(x,*args)*logNormalDistribution(x,mean, std)
    result, _ = integrate.quad(integrand, 0, np.inf)
    return result 

#args = all but radius and wavelengths
def averageVectorized(mean, std, yAxis, f, *args, numPoints = 1000):
    xMin = 1e-6
    xMax = mean + 7*std 
    x = np.linspace(xMin, xMax, numPoints)

    logNorm = logNormalDistribution(x, mean, std)
    #Normalize just in case. Use trapz because this is a numerical integration
    logNorm /= np.trapz(logNorm, x) 

    #Create grid
    yGrid = yAxis[:, None] #Reshape values as column vector
    fValues = f(x, yGrid, )

#Creates the generalized residuals to use in the least square fitting.
def generalizedResiduals(function, xName, xData, yData, paramsFit, paramsFixed):
# Combine parameters in a single dict
    allParams = {**paramsFit, **paramsFixed}

    # Get the parameter names from the function signature
    paramNames = list(inspect.signature(function).parameters.keys())

    # Identify the indices of the parameters to be fitted
    fitIndices = [paramNames.index(key) for key in paramsFit]

    def residualsFunc(fitValues):
        # Update the values of parameters to be fitted with the new guesses
        for i, idx in enumerate(fitIndices):
            allParams[paramNames[idx]] = fitValues[i]

        # Reorder the parameters to match the function's signature, inserting xData at the correct position
        residuals = []
        for xValue in xData:
            orderedParams = []    
            for key in paramNames:
                if key == xName:
                    orderedParams.append(xValue)  # Add xData (independent variable)
                else:
                    orderedParams.append(allParams[key])  # Add the fitted or fixed parameters
            yModel = function(*orderedParams)
            residuals.append(yModel - yData)

    return residualsFunc

