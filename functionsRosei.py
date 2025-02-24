from constantsRosei import *
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
#from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm

#------------------------------------------#
#-----------------DRUDE--------------------#
#------------------------------------------#
def drude(w, wp, epsilonInf,gamma):
    epsilonDreal = epsilonInf - wp**2/(w**2 + gamma**2)
    epsilonDimag = wp**2 *gamma / (w*(w**2+gamma**2))
    return epsilonDreal, epsilonDimag

def drudeReal(w, wp, epsilonInf, gamma):
    return epsilonInf - wp**2/(w**2 + gamma**2)

def drudeImag(w,wp,gamma):
    return wp**2 *gamma / (w*(w**2+gamma**2))

def drudeImagBaseline(w,wp,gamma,y0):
    return wp**2 *gamma / (w*(w**2+gamma**2)) + y0

def drudeImagRes(params, w, data):
    wp, gamma, y0 = params
    return drudeImagBaseline(np.array(w), wp, gamma, y0) - np.array(data)
   

#------------------------------------------#
#-------------INTERBAND--------------------#
#------------------------------------------#

#--------------#
#Imaginary part#
#--------------#
def fermi(E,Ef,T):
    return 1 / (1 + np.exp((E-Ef)/(k*T)))
    '''
    #This code doesn't overflow, but it's way slower than the normal one.
    # Calculate the exponent
    exponent = (E - Ef) / (k * T)
    
    # Apply conditional logic to avoid overflow
    with np.errstate(over='ignore'):  # Temporarily suppress overflow warnings
        result = np.where(exponent > 700, 0, 
                 np.where(exponent < -700, 1, 
                 1 / (1 + np.exp(exponent))))
    return result
'''
'''
def EDJDOS(E, w):
    factor = 1e-37 #all exponents are here. Check reduced constants where all the x10^{---} have been removed. 
    return factor*Fps/(16*np.pi**2 * hbarE**2 * np.sqrt(((E+w-wg)*hbarE/massPperp - (E+wf)*hbarE/massSperp)*1e-3)) #The 1e-3 is added because we removed all exponents, check reducedConstants
'''

def EDJDOSps(E,w):
    return np.where(
        (E+w-wg)/massPperp - (E+wf)/massSperp >0,
        1/np.sqrt((E+w-wg)/massPperp - (E+wf)/massSperp),
        0)

def funcEminPS(w):
    if w < wg + wf:
        return -wf + massSperp/(massSperp-massPperp) * (wg + wf - w)
    
    else: 
        return -wf - massSpar/(massPpar+massSpar) * (w - (wf + wg))

def funcEmaxPS(w):
    return -wf - C1 + massSperp * (C2 + (wg + wf - w)) / (massSperp - massPperp)


def integrandIBimagPS(E,w,Ef,T):
    return EDJDOSps(E,w) * fermi(E, Ef, T)

def JDOSPSsinglePoint(w,Ef,T):
    if w < rootPSanalytical:
        Emin = funcEminPS(w)
        Emax = funcEmaxPS(w)
        '''
        value = ' '
        if w >= Emax or w<= Emin:
            value = 'good'
        else:
            value = 'bad'
        print(value)
        print('w:{:.4g}, Emin:{:.4g}, Emax:{:.4g}' .format(w, Emin, Emax))
        '''
        integral, err = integrate.quad(integrandIBimagPS, Emin, Emax, args = (w,Ef,T))
        return integral 
    
    else: 
        return 0

def funcJDOSps(ws, Ef, T):
    JDOS = []
    for w in ws:
        jdos = JDOSPSsinglePoint(w,Ef,T)
        JDOS.append(jdos)
    return JDOS

def EDJDOSdp(E,w):
    return np.where( 
        (((w-wo-wf-E)/massPperp - (E+wf)/massDperp)>0), 
        1/np.sqrt((w-wo-wf-E)/massPperp - (E+wf)/massDperp), 
        0)

def funcEminDP(w):
    return -wf - C1 + massPperp/(massPperp + massDperp) * (C3 + w - wo)

def funcEmaxDP(w):
    if w < wo:
        return -wf + massDpar * (w-wo) / (massDpar - massPpar)
    else:
        return -wf + massDperp * (w-wo) / (massPperp + massDperp)

def integrandIBimagDP(E,w,Ef,T):
    return EDJDOSdp(E,w) * (1-fermi(E,Ef,T))

def JDOSDPsinglePoint(w,Ef,T):
    Emin = funcEminDP(w)
    Emax = funcEmaxDP(w)
    integral, err = integrate.quad(integrandIBimagDP, Emin, Emax, args = (w,Ef,T))
    return integral 

def funcJDOSdp(ws, Ef, T):
    JDOS = []
    for w in ws:
        jdos = JDOSDPsinglePoint(w,Ef,T)
        JDOS.append(jdos)
    return JDOS

#relative factor between imaginary parts#
#---------------------------------------#
def epsIBimagSinglePointPS(w, Ef, T):
    JDOSps = JDOSPSsinglePoint(w,Ef,T)
    epsIBimaginaryPS = JDOSps/w**2
    return epsIBimaginaryPS

def epsIBimagSinglePointDP(w, Ef, T):
    integralDP = JDOSDPsinglePoint(w,Ef,T)
    epsIBimaginaryDP = integralDP/w**2 #The 15.43 is because the relative factor between the JDOSps and dp is that number. 2.21 is the aditional relative factor from the paper.
    return epsIBimaginaryDP

#To fit the factor between both contributions, we used a spectrum at T=300K, so we fixed this T.
def epsIBimagFactorFit(ws,f1, f2, y0):
    epsIBimags = []
    for w in ws:
        epsIBimag = f1*epsIBimagSinglePointPS(w,0,300) + f2*epsIBimagSinglePointDP(w,0,300) + y0
        epsIBimags.append(epsIBimag)
    return epsIBimags

def epsIBimagFactorFit2(ws,f1, f2):
    epsIBimags = []
    for w in ws:
        epsIBimag = f1*epsIBimagSinglePointPS(w,0,300) + f2*epsIBimagSinglePointDP(w,0,300)
        epsIBimags.append(epsIBimag)
    return epsIBimags

#np.trapz doesn't work very well.
def epsIBimagSinglePoint(w, Ef, T, f1, f2):
    EminPS = funcEminPS(w)
    EmaxPS = funcEmaxPS(w)
    integralPS, err = integrate.quad(integrandIBimagPS, EminPS, EmaxPS, args=(w, Ef, T))
    
    EminDP = funcEminDP(w)
    EmaxDP = funcEmaxDP(w)
    integralDP, err2 = integrate.quad(integrandIBimagDP, EminDP, EmaxDP, args=(w,Ef,T))

    epsIBimaginary = f1*integralPS/w**2 + f2*integralDP/w**2 #The 15.43 is because the relative factor between the JDOSps and dp is that number. 2.21 is the aditional relative factor from the paper.
    return epsIBimaginary

def epsIBimag(ws,Ef,T,f1,f2):
    epsIBimags = []
    for w in ws:
        epsIM = epsIBimagSinglePoint(w, Ef, T,f1,f2)
        epsIBimags.append(epsIM)
    return np.array(epsIBimags)


#---------#
#Real part#
#---------#
def integrandIBreal(wprime, w,Ef,T, f1, f2):
    integrand = wprime*epsIBimagSinglePoint(wprime,Ef,T,f1,f2)/(wprime**2 - w**2)
    return integrand 

#This integrand is mutiplied by (wprime-w) because cauchy weighting, which is used for optimizing principal value calculations, implicitly multiplies by 1/(x-xsingularity)
def integrandIBrealCauchy(wprime, w, Ef, T, f1, f2):
    integrand = wprime*epsIBimagSinglePoint(wprime,Ef,T,f1,f2)/(wprime + w)
    return integrand 

#https://stackoverflow.com/questions/52693899/numerical-integration-with-singularities-in-python-principal-value
def epsIBrealSlow(ws, Ef, T, f1, f2):
    epsIBreals = []
    for w in ws:
        integralCauchy,err = integrate.quad(integrandIBrealCauchy, 1e-5, 80, args=(w, Ef, T, f1, f2), weight='cauchy', wvar=w)
        #integral1,err1 = integrate.quad(integrandIBreal, 0, w, args=(w, Ef, T))#, weight='cauchy', wvar=w)
        #integral2,err2 = integrate.quad(integrandIBreal, w, np.inf, args=(w, Ef, T))#, weight='cauchy', wvar=w)
        #epsIBreal = 2/np.pi*(integral1+integral2)
        epsIBreal = 2/np.pi*integralCauchy
        epsIBreals.append(epsIBreal)
    return np.array(epsIBreals)

def epsIB_single(w, Ef, T, f1, f2):
    integralCauchy, err = integrate.quad(
        integrandIBrealCauchy, 1e-5, 80, args=(w, Ef, T, f1, f2), weight='cauchy', wvar=w
    )
    return 2 / np.pi * integralCauchy

def epsIBreal(ws, Ef, T, f1, f2, n_jobs=-1):
    epsIBreals = Parallel(n_jobs=n_jobs)(
        delayed(epsIB_single)(w, Ef, T, f1, f2) for w in ws
    )
    return np.array(epsIBreals)

#Engendro Luis
# Precompute
def precompute_epsIBimag_grid(w_grid, Ef, T, f1, f2):
    precomputed = [epsIBimagSinglePoint(wprime, Ef, T, f1, f2) for wprime in w_grid]
    return interp1d(w_grid, precomputed, kind='cubic', fill_value="extrapolate")

# Real Par
def integrandIBrealCauchy2(wprime, w, epsIBimag_interp):
    return wprime * epsIBimag_interp(wprime) / (wprime + w)

# Parallelized 
def epsIB_single2(w, Ef, T, epsIBimag_interp):
    integralCauchy, err = integrate.quad(
        integrandIBrealCauchy2,
        1e-5,
        50,
        args=(w, epsIBimag_interp),
        weight='cauchy',
        wvar=w
    )
    return 2 / np.pi * integralCauchy

def epsIBreal2(ws, Ef, T, f1, f2, n_jobs=-1):
    # Precompute interpolated epsIBimag 
    w_grid = np.linspace(1e-5, 50, 1000)  # Adjust grid resolution
    epsIBimag_interp = precompute_epsIBimag_grid(w_grid, Ef, T, f1, f2)

    # Parallelized over ws
    epsIBreals = Parallel(n_jobs=n_jobs)(
        delayed(epsIB_single2)(w, Ef, T, epsIBimag_interp) for w in ws
    )
    return np.array(epsIBreals)
#------------------------------------------#
#-------------ABSORBANCE-------------------#
#------------------------------------------#
def epsilon(w,wp,epsInf,gamma,Ef,T,f):
    epsDreal, epsDimag = drude(w, wp, epsInf,gamma)
    e1 = epsDreal + f*epsIBreal(w,Ef,T)
    e2 = epsDimag + f*epsIBimag(w,Ef,T)
    return e1, e2 

def absorbance(w,e1,e2, eM):
    abs = eM**3/2 * w*e2 / ((e1 + 2*eM)**2 + e2**2)
    return abs

def absorbanceFit(w,wp,epsInf,gamma,eM,Ef,T,f):
    epsDreal, epsDimag = drude(w, wp, epsInf,gamma)
    e1 = epsDreal + f*epsIBreal(w,Ef,T)
    e2 = epsDimag + f*epsIBimag(w,Ef,T)
    abs = eM**3/2 * w*e2 / ((e1 + 2*eM)**2 + e2**2)
    return abs 

#------------------------------------------#
#-------------CURVE FITTING----------------#
#------------------------------------------#
def epsilonImaginary(w, wp, gamma, f):
    epsilonDimag = drudeImag(w,wp,gamma)
    epsilonIBimag = np.array(epsIBimag(w,0,298.15))
    return epsilonDimag + f*epsilonIBimag

#------------------------------------------#
#-----------------UTILS--------------------#
#------------------------------------------#
def getValuesBelow(threshold,array):
    index = np.searchsorted(array, threshold, side='left')
    return array[:index]

def getValues(lowerThreshold, upperThreshold, arrayX, arrayY):
    indexLow = np.searchsorted(arrayX, lowerThreshold, side = 'left')
    indexUp = np.searchsorted(arrayX, upperThreshold, side = 'right')
    return arrayX[indexLow:indexUp], arrayY[indexLow:indexUp]

#------------------------------------------#
#-----------TEST FUNCTIONS-----------------#
#------------------------------------------#

#This integrand is mutiplied by (wprime-w) because cauchy weighting, which is used for optimizing principal value calculations, implicitly multiplies by 1/(x-xsingularity)
def integrandIBrealCauchyPS(wprime, w, Ef, T):
    integrand = wprime*epsIBimagSinglePointPS(wprime,Ef,T)/(wprime + w)
    return integrand 

#https://stackoverflow.com/questions/52693899/numerical-integration-with-singularities-in-python-principal-value
def epsIBrealPS(ws, Ef, T):
    epsIBreals = []
    for w in ws:
        integralCauchy,err = integrate.quad(integrandIBrealCauchyPS, 1e-5, 50, args=(w, Ef, T), weight='cauchy', wvar=w)
        #integral1,err1 = integrate.quad(integrandIBreal, 0, w, args=(w, Ef, T))#, weight='cauchy', wvar=w)
        #integral2,err2 = integrate.quad(integrandIBreal, w, np.inf, args=(w, Ef, T))#, weight='cauchy', wvar=w)
        #epsIBreal = 2/np.pi*(integral1+integral2)
        print(integralCauchy)
        epsIBreal = 2/np.pi*integralCauchy
        epsIBreals.append(epsIBreal)
    return np.array(epsIBreals)

#This integrand is mutiplied by (wprime-w) because cauchy weighting, which is used for optimizing principal value calculations, implicitly multiplies by 1/(x-xsingularity)
def integrandIBrealCauchyDP(wprime, w, Ef, T):
    integrand = wprime*epsIBimagSinglePointDP(wprime,Ef,T)/(wprime + w)
    return integrand 

#https://stackoverflow.com/questions/52693899/numerical-integration-with-singularities-in-python-principal-value
def epsIBrealDP(ws, Ef, T):
    epsIBreals = []
    for w in ws:
        integralCauchy,err = integrate.quad(integrandIBrealCauchyDP, 1e-5, 50, args=(w, Ef, T), weight='cauchy', wvar=w)
        #integral1,err1 = integrate.quad(integrandIBreal, 0, w, args=(w, Ef, T))#, weight='cauchy', wvar=w)
        #integral2,err2 = integrate.quad(integrandIBreal, w, np.inf, args=(w, Ef, T))#, weight='cauchy', wvar=w)
        #epsIBreal = 2/np.pi*(integral1+integral2)
        print(integralCauchy)
        epsIBreal = 2/np.pi*integralCauchy
        epsIBreals.append(epsIBreal)
    return np.array(epsIBreals)

#------------------------------------------#
#-----------HODAK FUNCTIONS----------------#
#------------------------------------------#
def funcHodakJDOSintegrand(E,w,wIB,nud,T):
    return (1-fermi(E,0,T))/np.sqrt(nud*(w-wib)-E)

def funcHodakJDOS(ws, wIB, nud,T):
    JDOS = []
    for w in ws:
        Emax = nud*(w-wib)
        JDOSsinglePoint, err = integrate.quad(funcHodakJDOSintegrand,0,Emax,args=(w,wIB,nud,T))
        JDOS.append(JDOSsinglePoint)
    return JDOS


