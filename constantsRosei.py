import numpy as np

massElectron = 9.10938188           #e-31 kg
massSperp = 5.160*massElectron      #*me
massSpar = 0.128*massElectron       #*me
massPperp = 0.32*massElectron       #*me
massPpar = 0.172*massElectron       #*me
wg = 3.85                           #eV 3.85 is the true value
wf = 0.31                           #eV 0.31 is the true value
k = 8.617333262e-5                  #eV/K
a = 0.409                           #meters e-9
hbarJ = 1.054571817                 #J.s e-34
hbarE = 6.582119569                 #eV.s e-16
#wf + wg = 4.16eV
Fps = np.sqrt((massSpar*massPpar * massSperp*massPperp)/(massSpar*massPperp + massSperp*massPpar))*1e-31 #kg

#hbarJ**2 ~ e-68
#a**2 ~ e-18
#6.241509e18 from J to eV
C1 = hbarJ**2 * np.pi**2 / (32*massPpar*a**2) * 6.241509e-1                     #~ x10^(-68 + 31 +18 +18 = -1) eV
C2 = hbarJ**2 * np.pi**2 * (1/massSpar + 1/massPpar) / (32*a**2) * 6.241509e-1  #eV 

#root for Emax-Emin for p->s band
rootPSanalytical = wg + wf + hbarJ**2*np.pi**2*(massSpar+massPpar)*6.241509e-1/(32*a**2 *massSpar*massPpar) #~ x10^(-68+18 +18+31)

massDperp = 2.580*massElectron      #*me
massDpar = 2.075*massElectron       #*me
wo = 3.68                           #eV

Fdp = np.sqrt((massDpar*massPperp * massDperp*massPpar) / (massDpar*massPperp + massDperp*massPpar))*1e-31 #kg 
C3 = hbarJ**2 * np.pi**2 * (1/massPpar - 1/massDpar) / (32*a**2) * 6.241509e-1  #eV

epsWater = 1.77

#Hodak from Rosei
nud = massDperp/(massDperp+massPperp)
wib = wo + wf*(1+massPperp/massDperp)

test1 = 1-massPperp/massSperp
#print(test1)

#Hodak constants:
wIBHodak = 3.9                           #eV
nudHodak = 0.92


#------------------------------------------#
#-----------------SANTILLAN----------------#
#------------------------------------------#
wpSan = 13.8*hbarE*0.1                      #eV
gammaBulkSan = 2.7*hbarE*0.1                #eV
sizeConstant = 0.8*14.1*hbarE*0.1           #nm.eV
Kbulk = 2e24
r0 = 0.35                                   #nm