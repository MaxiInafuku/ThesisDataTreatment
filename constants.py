#Constants
hbarE = 6.582119569                 #eV.s e-16
hE = 4.1357                         #eV.s e-15
k = 8.617333262e-5                  #eV/K
c = 299792458                       #m/s
epsWater = 1.77

#Drude
gammaBulk = 2.7                     #Hz e13
wp = 13.8                           #Hz e15
nuf = 14.1                          #nm.s-1 e14
C = 0.8
r0 = 0.35                           #nm

#Interband
Eg = 1.91                           #eV
Ef = 4.12                           #eV
gammaIBfreq = 1.5                   #Hz e14
Kbulk = 2                           #e24

gammaBu = hbarE*gammaBulk*0.001     #eV
Eplasma = hbarE*wp*0.1              #eV
Cnuf = hbarE*C*nuf*0.01             #eV
gammaIB = hbarE*gammaIBfreq*0.01    #eV

wg = Eg/hbarE*1e16
wf = Ef/hbarE*1e16

print(gammaBu)



'''
#Constants
hbarE = 6.582119569e-16             #eV.s 
hE = 4.1357e-15                     #eV.s 
k = 8.617333262e-5                  #eV/K
c = 299792458                       #m/s

gammaBulk = 2.7e13                  #Hz
wp = 13.8e15                        #Hz
nuf = 14.1e14                       #nm.s-1 
C = 0.8
r0 = 0.35                           #nm

Eg = 1.91                           #eV
Ef = 4.12                           #eV
gammaIBfreq = 1.5e14                #Hz e14
Kbulk = 2e24                        #e24

wg = Eg/hbarE*1e16

gammaBu = hbarE*gammaBulk*1e-3      #eV
Eplasma = hbarE*wp*0.1              #eV
Cnuf = hbarE*C*nuf*0.01             #eV
gammaIB = hbarE*gammaIBfreq*0.01    #eV
'''