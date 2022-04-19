from numpy import exp, cos, linspace
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os, time, glob
import numpy as np
import pandas as pd
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams['font.family'] = 'Helvetica'

def odes(x, t, altitude, temperature, pressure):
    # convert to per cm^3 from m^3
    fac=1e6
    O3 = x[0]/fac
    O = x[1]/fac
    O1D = x[2]/fac
    NO = x[3]/fac
    NO2 = x[4]/fac
    Cl = x[5]/fac
    ClO = x[6]/fac
    OH = x[7]/fac
    HO2 = x[8]/fac
    H = x[9]/fac
    Br = x[10]/fac
    BrO = x[11]/fac
    CH4 = x[12]/fac
    H2O = x[13]/fac
    N2O = x[14]/fac
    HNO3 = x[15]/fac
    HCl =  x[16]/fac
    ClONO2 = x[17]/fac
    HOCl = x[18]/fac
    ClOOCl = x[19]/fac
    ClOO = x[20]/fac
    BrCl = x[21]/fac
    OClO = x[22]/fac
    ### function to solve system of coupled differential rate equations
    ### for explanations see e.g. also https://www.youtube.com/watch?v=MXUMJMrX2Gw
    ### approximate air density at altitude; https://en.wikipedia.org/wiki/Atmospheric_pressure and Lecture 1 slides
    nr_timesteps_per_day = int(24*60/30.0)
    nr_seconds_per_day = 24*60*60
    # pressure = 1e5*np.exp(-9.8*altitude*1000.0*0.02896968 / (288.16*8.31446))
    # air_density_molecules_per_cm3 = pressure*28.97/(8.31446*temperature)
    ### in molecules per m^3
    M = pressure*6.02214e23/(8.31446*temperature*fac)
    M_O2 = 0.21*M
    ### find/define photolysis rates j (lat, 0 lon, all altitudes) for all relevant reactions # TODO
    ### also find/define reaction rates k(p, T) on IUPAC/NASA
    # i.e. define numerical values for rate constants and photolysis rates, use: https://jpldataeval.jpl.nasa.gov/pdf/NASA-JPL%20Evaluation%2019-5.pdf and https://iupac-aeris.ipsl.fr/
    if np.sin((t/nr_seconds_per_day)*np.pi) > 0.0:
        j1 = 3e-12*np.exp(-(50.0-altitude)/50.0)*np.sin((t/nr_seconds_per_day)*np.pi)#O2+hv->2O; in s-1;
    else:
        j1 = 0.0
    # read from plot https://www.briangwilliams.us/atmospheric-chemistry/photolysis-rate-as-a-function-of-altitude.html - should later introduce dependence on season and latitude
    k2 = 6e-34*(temperature/300.0)**-2.6 # cm^3 molecule-1 s-1; IIUPAC; over 200-300K; O+O2+M
    if np.sin((t/nr_seconds_per_day)*np.pi) > 0.0:
        j3 = 1e-18*np.exp(-(50.0-altitude)/50.0)*np.sin((t/nr_seconds_per_day)*np.pi) # as for j1; needs further specification. This is only a first test; O3+hv
    else:
        j3=0.0
    k3 = 2.15e-11*np.exp(-110.0/temperature) # O1D+M -> O+M , NASA: p.21 for M=N2; branching ratio; note O2 has a different one
    # k4 = 8e-12*np.exp(2060.0/temperature) # O+O3->2O2; IUPAC; range 200-400K; cm3 molecule=1 s-1
    k4 = 1e-13*np.exp(2060.0/temperature) # O+O3->2O2; IUPAC; range 200-400K; cm3 molecule=1 s-1
    k5 = 2.07e-12*np.exp(-1400.0/temperature) # range 195-310K cm3 molecule-1 s-1; NO+O3->NO2+O2; IUPAC
    k6 = 5.1e-12*np.exp(198.0/temperature) # range 220-420K; cm3 molecule-1 s-1; NO2+O->NO+O2; IUPAC
    k7 = 2.8e-11*np.exp(-250.0/temperature) # range 180-300K; cm3 molecule-1 s-1; Cl+O3->ClO+O2; IUPAC
    k8 = 2.5e-11*np.exp(110.0/temperature) # cm3 molecule-1 s-1; range 220-390K; ClO+O->Cl+O2; IUPAC
    k9 = 1.7e-12*np.exp(-940.0/temperature) # cm3 molecule-1 s-1; range 220-450K; OH+O3->HO2+O2; IUPAC
    k10 = 2.7e-11*np.exp(224.0/temperature) # cm3 molecule-1 s-1; range 220-400K; HO2+O->OH+O2; IUPAC
    # k10 = 3e-11*np.exp(-200.0/temperature) # NASA; p.67; O+HO2->OH+O2; range 229-391K; cm3 molecule-1 s-1
    k11 = 1.4e-10*np.exp(470.0/temperature) # NASA p.67; H+O3->OH+O2; range 196-424K; cm3 molecule-1 s-1
    k12 = 1.8e-11*np.exp(-180.0/temperature) # NASA; p.67; O+OH->H+O2; range 136-515K; cm3 molecule-1 s-1
    k13 = 1.6e-11*np.exp(780.0/temperature) # NASA; p.325; Br+O3->BrO+O2; range 195-422K; cm3 molecule-1 s-1
    k14 = 1.9e-11*np.exp(-230.0/temperature) # NASA; p.325; BrO+O->Br+O2; range 231-328K; cm3 molecule-1 s-1
    k15 = 5.3e-32*(temperature/298.0)**-1.8 # NASA; p.434; H+O2+M->HO2+M; 
    k16 = 1e-14*np.exp(490.0/temperature) # NASA; p.67; HO2+O3->OH+2O2; range 197-413K; cm3 molecule-1 s-1
    k17 = 1.75e-10 # NASA; p.22; CH4+O1D->total products; note that the rate considered is for all possible outcomes not just the one provided in the documentation; cm3 molecule-1 s-1
    k18 = 2.45e-12*np.exp(1775.0/temperature) # NASA; p.106; CH4+OH->CH3+H2O; range 178-2025K; cm3 molecule-1 s-1
    k19 = 1.63e-10*np.exp(-60.0/temperature) # NASA; p.21; O1D+H2O->mostly 2OH; range 217-453K; cm3 molecule-1 s-1
    k20 = 1.19e-10*np.exp(-20.0/temperature)*0.61 # NASA; p.21; O1D+N2O->2NO; range 195-719K; cm3 molecule-1 s-1
    k21 = 1.8e-30*(temperature/298.0)**-3.0 # NASA; p.434; HNO3 formation; taking the low pressure limit for simplicity; cm6 molecule-2 s-1
    if np.sin((t/nr_seconds_per_day)*np.pi) > 0.0:
        j22 = 0.5e-5*np.sin((t/nr_seconds_per_day)*np.pi) # dummy variable; currently hardly HNO3 photolysed; have to find a way to treat this.
    else:
        j22 = 0.0
    k23 = 7.1e-12*np.exp(1270.0/temperature) # NASA; p.243; Cl+CH4->HCl+CH3; 181-1550K; cm3 molecule-1 s-1
    k24 = 1.8e-31*(temperature/298.0)**-3.4 # NASA; p.436; ClO+NO2+M->ClONO2+M; cm6 molecule-2 s-1
    k25 = 2.6e-12*np.exp(-290.0/temperature) # NASA; p.242; ClO+HO2->HOCl+O2; cm3 molecule-1 s-1
    if np.sin((t/nr_seconds_per_day)*np.pi) > 0.0:
        j26 = 0.1*np.sin((t/nr_seconds_per_day)*np.pi) # NO2+hv->NO+O; have to treat the photolysis of NO2 later
    else:
        j26 = 0.0
    k27 = 1.9e-32*(temperature/298.0)**-3.6 # NASA; ClO+ClO+M->ClOOCl+M
    if np.sin((t/nr_seconds_per_day)*np.pi) > 0.0:
        j28 = 0.3*k27*np.sin((t/nr_seconds_per_day)*np.pi) ### ClOOCl+hv->Cl+ClOO; assume part is photolysed again - to be specified later
    else:
        j28=0.0
    k29 = 2.8e-10*np.exp(-1820.0/temperature)*0.79 # ClOO+M->Cl+O2+M; IUPAC, temperature range 160-300K
    k30 = 2.3e-12*np.exp(-260.0/temperature) # NASA; BrO+ClO->Br+ClOO
    k31 = 9.5e-13*np.exp(-550.0/temperature) # NASA; BrO+ClO->Br+OClO; range 200-400K
    k32 = 2.6e-11*np.exp(1300.0/temperature) # NASA; p.325, range 267-423K; Br + OClO -> BrO+ClO
    k33 = 4.1e-13*np.exp(-290.0/temperature) # NASA, p.326, range 200-400K; BrO+ClO->BrCl+O2
    if np.sin((t/nr_seconds_per_day)*np.pi) > 0.0:
        j34 = 0.1e-3*np.sin((t/nr_seconds_per_day)*np.pi) # replace later
    else:
        j34 = 0.0
        ### then define individual rate equations for all species involved
    # d[O3]/dt = k2*[O][O2][M] - j3*[O3] - k4[O3][O] - k5[NO][O3] - k9[OH][O3] - k11[H][O3] - k13[Br][O3] - k16[HO2][O3]
    dO3dt = k2*O*M_O2*M - j3*O3 - k4*O3*O - k5*NO*O3 - k9*OH*O3 - k11*H*O3 - k13*Br*O3 - k16*HO2*O3
    ### prevent negative concentration values are possible; first enfore this as hard limit
    if O3+dO3dt < 0.0:
    #     # dO3dt = dO3dt - O3
        dO3dt = 0.0
    #     # d[O]/dt = 2*j1*[O2] - k2*[O][O2][M] - k4[O3][O] - k6[NO2][O] - k8[ClO][O] - k10[HO2][O] - k12[OH][O] - k14[BrO][O] + j26[NO2]
    dOdt = j1*M_O2 - k2*O*M_O2*M - k4*O3*O + k3*O1D*M
#    dOdt = 0#2*j1*M_O2 - k2*O*M_O2*M - k4*O3*O #- k6*NO2*O - k8*ClO*O - k10*HO2*O - k12*OH*O - k14*BrO*O + j26*NO2
    # if O+dOdt < 0.0:
    #     dOdt = 0.0
    # # d[O(1D)]/dt = j3[O3] - k3*[O(1D)][M] - k17[CH4][O1D] - k19[H2O][O1D] - k20[N2O][O1D]
    dO1Ddt = j3*O3 - k3*O1D*M*1e-9 #- k17*CH4*O1D - k19*H2O*O1D - k20*N2O*O1D
    if O1D+dO1Ddt < 0.0:
        dO1Ddt = 0.0
    # d[NO]/dt = -k5[NO][O3] + k6[NO2][O] + 2*k20[N2O][O1D] + j26[NO2]
    dNOdt = -k5*NO*O3 + k6*NO2*O #+ 2*k20*N2O*O1D + j26*NO2
    if NO+dNOdt < 0.0:
        dNOdt = 0.0
    # d[NO2]/dt = k5[NO][O3] - k6[NO2][O] - k21[OH][NO2][M] + j22[HNO3] - k24[ClO][NO2][M] - j26[NO2]
    dNO2dt = k5*NO*O3 - k6*NO2*O - k21*OH*NO2*M + j22*HNO3 - k24*ClO*NO2*M #- j26*NO2
    if NO2+dNO2dt < 0.0:
        dNO2dt = 0.0
    # d[Cl]/dt = -k7[Cl][O3] + k8[ClO][O] - k23[Cl][CH4] + j28[ClOOCl] + k29[ClOO][M] + j34[BrCl]
    dCldt = -k7*Cl*O3 + k8*ClO*O + j28*ClOOCl #+ k29*ClOO*M + j34*BrCl #- k23*Cl*CH4 
    if Cl+dCldt < 0.0:
        dCldt = 0.0
    # d[ClO]/dt = k7[Cl][O3] - k8[ClO][O] - k24[ClO][NO2][M] - k25[ClO][HO2] - k27[ClO][ClO][M] - k30[ClO][BrO] - k31[ClO][BrO] + k32[Br][OClO] - k33[BrO][ClO]
    dClOdt = k7*Cl*O3 - k8*ClO*O - k24*ClO*NO2*M - k25*ClO*HO2 - k27*ClO*ClO*M - k30*ClO*BrO - k31*ClO*BrO + k32*Br*OClO - k33*BrO*ClO
    if ClO+dClOdt < 0.0:
        dClOdt = 0.0
    # d[OH]/dt = -k9[OH][O3] + k10[HO2][O] + k11[H][O3] - k12[OH][O] + k16[HO2][O3] + k17[CH4][O1D] - k18[CH4][OH] + 2*k19[H2O][O1D] - k21[OH][NO2][M] + j22[HNO3]
    dOHdt = -k9*OH*O3 + k10*HO2*O + k11*H*O3 - k12*OH*O + k16*HO2*O3 + 2*k19*H2O*O1D - k21*OH*NO2*M + j22*HNO3#+ k17*CH4*O1D - k18*CH4*OH  + j22*HNO3
    if OH+dOHdt < 0.0:
        dOHdt = 0.0
    # d[HO2]/dt = k9[OH][O3] - k10[HO2][O] + k15[H][O2][M] - k16[HO2][O3] - k25[ClO][HO2]
    dHO2dt = k9*OH*O3 - k10*HO2*O + k15*H*M_O2*M - k16*HO2*O3 - k25*ClO*HO2
    if HO2+dHO2dt < 0.0:
        dHO2dt = 0.0
    # d[H]/dt = -k11[H][O3] + k12[OH][O] - k15[H][O2][M]
    dHdt = -k11*H*O3 + k12*OH*O - k15*H*M_O2*M
    if H+dHdt < 0.0:
        dHdt = 0.0
    # d[Br]/dt = -k13[Br][O3] + k14[BrO][O] + k30[BrO][ClO] + k31[ClO][BrO] - k32[Br][OClO] + j34[BrCl]
    dBrdt = -k13*Br*O3 + k14*BrO*O + k30*BrO*ClO + k31*ClO*BrO - k32*Br*OClO + j34*BrCl
    if Br+dBrdt < 0.0:
        dBrdt = 0.0
    # d[BrO]/dt = k13[Br][O3] - k14[BrO][O] - k30[BrO][ClO] - k31[ClO][BrO] + k32[OClO][Br] - k33[BrO][ClO]
    dBrOdt = k13*Br*O3 - k14*BrO*O - k30*BrO*ClO - k31*ClO*BrO + k32*OClO*Br - k33*BrO*ClO
    if BrO+dBrOdt < 0.0:
        dBrOdt = 0.0
    # d[CH4]/dt = -k17[CH4][O1D] - k18[CH4][OH] - k23[Cl][CH4]
    dCH4dt = 0#-k17*CH4*O1D - k18*CH4*OH - k23*Cl*CH4
    if CH4+dCH4dt < 0.0:
        dCH4dt = 0.0
    # d[H2O]/dt = k18[CH4][OH] - k19[H2O][O1D]
    dH2Odt = 0#k18*CH4*OH - k19*H2O*O1D
    if H2O+dH2Odt < 0.0:
        dH2Odt = 0.0
    # d[N2O]/dt = -k20[N2O][O1D]
    dN2Odt = 0#-k20*N2O*O1D
    if N2O+dN2Odt < 0.0:
        dN2Odt = 0.0
    # d[HNO3]/dt = k21[OH][NO2][M] - j22[HNO3]
    dHNO3dt = k21*OH*NO2*M - j22*HNO3
    if HNO3+dHNO3dt < 0.0:
        dHNO3dt = 0.0
    # d[HCl]/dt = k23[Cl][CH4]
    dHCldt = k23*Cl*CH4
    if HCl+dHCldt < 0.0:
        dHCldt = 0.0
    # d[ClONO2]/dt = k24[ClO][NO2][M]
    dClONO2dt = 0#k24*ClO*NO2*M
    if ClONO2+dClONO2dt < 0.0:
        dClONO2dt = 0.0
    # d[HOCl]/dt = k25[ClO][HO2]
    dHOCldt = 0#k25*ClO*HO2
    if HOCl+dHOCldt < 0.0:
        dHOCldt = 0.0
    # d[ClOOCl]/dt = k27[ClO][ClO][M] - j28[ClOOCl]
    dClOOCldt = 0#k27*ClO*ClO*M - j28*ClOOCl
    if ClOOCl+dClOOCldt < 0.0:
        dClOOCldt = 0.0
    # d[ClOO]/dt = j28[ClOOCl] - k29[ClOO][M] + k30[BrO][ClO]
    dClOOdt = j28*ClOOCl - k29*ClOO*M + k30*BrO*ClO
    if ClOO+dClOOdt < 0.0:
        dClOOdt = 0.0
    # d[BrCl]/dt = k33[BrO][ClO] - j34[BrCl]
    dBrCldt = 0#k33*BrO*ClO - j34*BrCl
    if BrCl+dBrCldt < 0.0:
        dBrCldt = 0.0
    # d[OClO]/dt = k31[ClO][BrO] - k32[OClO][Br]
    dOClOdt = k31*ClO*BrO - k32*OClO*Br
    if OClO+dOClOdt < 0.0:
        dOClOdt = 0.0
    ### then find numerical Python solver to solve the coupled equation system as a function of time and initial conditions
    gradients_in_cm3 = [dO3dt, dOdt, dO1Ddt, dNOdt, dNO2dt, dCldt, dClOdt, dOHdt, dHO2dt, dHdt, dBrdt, dBrOdt, dCH4dt, dH2Odt, dN2Odt, dHNO3dt, dHCldt, dClONO2dt, dHOCldt, dClOOCldt, dClOOdt, dBrCldt, dOClOdt]
    gradients_in_m3 = [element * fac for element in gradients_in_cm3] 
    return gradients_in_m3

def compute_chemistry(integrationtime, altitude, starttime, latitude, temperature, O3, O, O1D, NO, NO2, Cl, ClO, OH, HO2, H, Br,\
                      BrO, CH4, H2O, N2O, HNO3, HCl, ClONO2, HOCl, ClOOCl, ClOO, BrCl, OClO, resolution=500):
    list_of_species = ['H$_2$O (ppm)', 'CH$_4$ (ppb)', 'N$_2$O (ppb)', 'O$_3$ (ppm)', 'NO$_2$ (ppb)', 'NO (ppt)', 'HNO$_3$ (ppb)', 'ClO (ppb)',\
                       'BrO (ppt)', 'Br (ppq)', 'Cl (ppq)', 'HCl (ppb)', 'OH (ppt)', 'HO2 (ppt)', 'O($^3$P) (ppq)', 'O($^1$D) (ppq)'] 
    # first convert all species into number densities (molecules/m^3)
    # for this we need the air density, using the ideal gas law
    pressure = 1e5*np.exp(-9.81*altitude*1000.0*0.02896968 / (temperature*8.31446))
    ### air density in molecules per m^3
    M_air = pressure*6.02214e23/(8.31446*temperature)
    ### molar mass air
    MMR_air = 28.9644
    ### factors to convert later mass mixing ratios of each species into number densities in molecules/m^3
    fac_O3 = 1e-6*M_air*MMR_air/47.9982
    fac_O = 1e-15*M_air*MMR_air/15.999
    fac_O1D = 1e-15*M_air*MMR_air/15.999
    fac_NO = 1e-12*M_air*MMR_air/30.01
    fac_NO2 = 1e-9*M_air*MMR_air/46.0055
    fac_Cl = 1e-15*M_air*MMR_air/35.453
    fac_ClO = 1e-9*M_air*MMR_air/51.4521
    fac_OH = 1e-12*M_air*MMR_air/17.008
    fac_HO2 = 1e-12*M_air*MMR_air/18.01528
    fac_H = 1e-15*M_air*MMR_air/1.0078
    fac_Br = 1e-15*M_air*MMR_air/79.904
    fac_BrO = 1e-12*M_air*MMR_air/95.904
    fac_CH4 = 1e-9*M_air*MMR_air/16.04
    fac_H2O = 1e-6*M_air*MMR_air/18.01528
    fac_N2O = 1e-6*M_air*MMR_air/44.013
    fac_HNO3 = 1e-9*M_air*MMR_air/63.01
    fac_HCl = 1e-9*M_air*MMR_air/36.458
    fac_ClONO2 = 1e-9*M_air*MMR_air/97.46
    fac_HOCl = 1e-9*M_air*MMR_air/52.46
    fac_ClOOCl = 1e-9*M_air*MMR_air/102.9042
    fac_ClOO = 1e-12*M_air*MMR_air/67.45
    fac_BrCl = 1e-12*M_air*MMR_air/115.357
    fac_OClO = 1e-12*M_air*MMR_air/67.45
    ### convert mass mixing ratios into volume mixing ratios
    ### then multiply with M_air to obtain number densities with respect to each species, also consider input scaling
    O3 = O3*fac_O3
    O = O*fac_O
    O1D = O1D*fac_O1D
    NO = NO*fac_NO
    NO2 = NO2*fac_NO2
    Cl = Cl*fac_Cl
    ClO = ClO*fac_ClO
    OH = OH*fac_OH
    HO2 = HO2*fac_HO2
    H = H*fac_H
    Br = Br*fac_Br
    BrO = BrO*fac_BrO
    CH4 = CH4*fac_CH4
    H2O = H2O*fac_H2O
    N2O = N2O*fac_N2O
    HNO3 = HNO3*fac_HNO3
    HCl = HCl*fac_HCl
    ClONO2 = ClONO2*fac_ClONO2
    HOCl = HOCl*fac_HOCl
    ClOOCl = ClOOCl*fac_ClOOCl
    ClOO = ClOO*fac_ClOO
    BrCl = BrCl*fac_BrCl
    OClO = OClO*fac_OClO    
    ### define time vector and a few useful time quantities, all in seconds
    nr_timesteps_per_day = int(24*60/30.0)
    ### 30 min timestep in seconds
    dt = 30*60
    ### assume dt = 30 mins; input fractions of days
    nr_seconds_per_day = 24*60*60
    ### 30 min timestep in seconds
    dt = 30*60
    ### integration time in seconds
    dT = integrationtime*nr_seconds_per_day
    t = np.arange(0,dT+dt,dt)
    t_actual = (t+starttime*nr_seconds_per_day)/nr_seconds_per_day
    ### define initial state vector
    ### still need to convert inputs to number molecules per m^3 TODO!!!
    x0 = [O3, O, O1D, NO, NO2, Cl, ClO, OH, HO2, H, Br, BrO, CH4, H2O, N2O, HNO3, HCl, ClONO2, HOCl, ClOOCl, ClOO, BrCl, OClO]
    unit = '(molecules_m^-3)'
    names_no_unit = ['O3', 'O', 'O1D', 'NO', 'NO2', 'Cl', 'ClO', 'OH', 'HO2', 'H', 'Br', 'BrO', 'CH4', 'H2O', 'N2O', 'HNO3', 'HCl', 'ClONO2', 'HOCl', 'ClOOCl', 'ClOO', 'BrCl', 'OClO']
    names = [element + unit for element in names_no_unit]
    # sunlight =
    args = (altitude, temperature, pressure)
    x = odeint(odes, x0, t, args, printmessg=True)
    O3 = x[:,0]
    O = x[:,1]
    O1D = x[:,2]
    NO = x[:,3]
    NO2 = x[:,4]
    Cl = x[:,5]
    ClO = x[:,6]
    OH = x[:,7]
    HO2 = x[:,8]
    H = x[:,9]
    Br = x[:,10]
    BrO = x[:,11]
    CH4 = x[:,12]
    H2O = x[:,13]
    N2O = x[:,14]
    HNO3 = x[:,15]
    HCl =  x[:,16]
    ClONO2 = x[:,17]
    HOCl = x[:,18]
    ClOOCl = x[:,19]
    ClOO = x[:,20]
    BrCl = x[:,21]
    OClO = x[:,22]
    ### convert data to Dataframe for later save of csv file
    x = np.array(x)
    # print(x)
    df = pd.DataFrame(x,columns=names,index=t_actual)
    df.index.name='Time(days)'
    plt.figure()  # needed to avoid adding curves in plot
    fig, axs = plt.subplots(4,4,constrained_layout=True)
    # for i in range(0,3):
    #     for j in range(0,3):
    axs[0,0].plot(t_actual,H2O/fac_H2O,linewidth=2,color='red')
    axs[0,0].set_xlabel('Time (days)',size=6)
    axs[0,0].set_ylabel(list_of_species[0],size=6)
    axs[0,0].tick_params(axis='x', labelsize=4)
    axs[0,0].tick_params(axis='y', labelsize=4)
    
    axs[0,1].plot(t_actual,CH4/fac_CH4,linewidth=2,color='red')
    axs[0,1].set_xlabel('Time (days)',size=6)
    axs[0,1].set_ylabel(list_of_species[1],size=6)
    axs[0,1].tick_params(axis='x', labelsize=4)
    axs[0,1].tick_params(axis='y', labelsize=4)

    axs[0,2].plot(t_actual,N2O/fac_N2O,linewidth=2,color='red')
    axs[0,2].set_xlabel('Time (days)',size=6)
    axs[0,2].set_ylabel(list_of_species[2],size=6)
    axs[0,2].tick_params(axis='x', labelsize=4)
    axs[0,2].tick_params(axis='y', labelsize=4)

    axs[0,3].plot(t_actual,O3/fac_O3,linewidth=2,color='red')
    axs[0,3].set_xlabel('Time (days)',size=6)
    axs[0,3].set_ylabel(list_of_species[3],size=6)
    axs[0,3].tick_params(axis='x', labelsize=4)
    axs[0,3].tick_params(axis='y', labelsize=4)

    axs[1,0].plot(t_actual,NO2/fac_NO2,linewidth=2,color='red')
    axs[1,0].set_xlabel('Time (days)',size=6)
    axs[1,0].set_ylabel(list_of_species[4],size=6)
    axs[1,0].tick_params(axis='x', labelsize=4)
    axs[1,0].tick_params(axis='y', labelsize=4)

    axs[1,1].plot(t_actual,NO/fac_NO,linewidth=2,color='red')
    axs[1,1].set_xlabel('Time (days)',size=6)
    axs[1,1].set_ylabel(list_of_species[5],size=6)
    axs[1,1].tick_params(axis='x', labelsize=4)
    axs[1,1].tick_params(axis='y', labelsize=4)

    axs[1,2].plot(t_actual,HNO3/fac_HNO3,linewidth=2,color='red')
    axs[1,2].set_xlabel('Time (days)',size=6)
    axs[1,2].set_ylabel(list_of_species[6],size=6)
    axs[1,2].tick_params(axis='x', labelsize=4)
    axs[1,2].tick_params(axis='y', labelsize=4)

    axs[1,3].plot(t_actual,ClO/fac_ClO,linewidth=2,color='red')
    axs[1,3].set_xlabel('Time (days)',size=6)
    axs[1,3].set_ylabel(list_of_species[7],size=6)
    axs[1,3].tick_params(axis='x', labelsize=4)
    axs[1,3].tick_params(axis='y', labelsize=4)

    axs[2,0].plot(t_actual,BrO/fac_BrO,linewidth=2,color='red')
    axs[2,0].set_xlabel('Time (days)',size=6)
    axs[2,0].set_ylabel(list_of_species[8],size=6)
    axs[2,0].tick_params(axis='x', labelsize=4)
    axs[2,0].tick_params(axis='y', labelsize=4)

    axs[2,1].plot(t_actual,Br/fac_Br,linewidth=2,color='red')
    axs[2,1].set_xlabel('Time (days)',size=6)
    axs[2,1].set_ylabel(list_of_species[9],size=6)
    axs[2,1].tick_params(axis='x', labelsize=4)
    axs[2,1].tick_params(axis='y', labelsize=4)

    axs[2,2].plot(t_actual,Cl/fac_Cl,linewidth=2,color='red')
    axs[2,2].set_xlabel('Time (days)',size=6)
    axs[2,2].set_ylabel(list_of_species[10],size=6)
    axs[2,2].tick_params(axis='x', labelsize=4)
    axs[2,2].tick_params(axis='y', labelsize=4)

    axs[2,3].plot(t_actual,HCl/fac_HCl,linewidth=2,color='red')
    axs[2,3].set_xlabel('Time (days)',size=6)
    axs[2,3].set_ylabel(list_of_species[11],size=6)
    axs[2,3].tick_params(axis='x', labelsize=4)
    axs[2,3].tick_params(axis='y', labelsize=4)
    
    axs[3,0].plot(t_actual,OH/fac_OH,linewidth=2,color='red')
    axs[3,0].set_xlabel('Time (days)',size=6)
    axs[3,0].set_ylabel(list_of_species[12],size=6)
    axs[3,0].tick_params(axis='x', labelsize=4)
    axs[3,0].tick_params(axis='y', labelsize=4)

    axs[3,1].plot(t_actual,HO2/fac_HO2,linewidth=2,color='red')
    axs[3,1].set_xlabel('Time (days)',size=6)
    axs[3,1].set_ylabel(list_of_species[13],size=6)
    axs[3,1].tick_params(axis='x', labelsize=4)
    axs[3,1].tick_params(axis='y', labelsize=4)

    axs[3,2].plot(t_actual,O/fac_O,linewidth=2,color='red')
    axs[3,2].set_xlabel('Time (days)',size=6)
    axs[3,2].set_ylabel(list_of_species[14],size=6)
    axs[3,2].tick_params(axis='x', labelsize=4)
    axs[3,2].tick_params(axis='y', labelsize=4)

    axs[3,3].plot(t_actual,O1D/fac_O1D,linewidth=2,color='red')
    axs[3,3].set_xlabel('Time (days)',size=6)
    axs[3,3].set_ylabel(list_of_species[15],size=6)
    axs[3,3].tick_params(axis='x', labelsize=4)
    axs[3,3].tick_params(axis='y', labelsize=4)

    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
    # Use time since Jan 1, 1970 in filename in order make
    # a unique filename that the browser has not chached
    # plotfile = os.path.join('static/images', str(time.time()) + '.png')
    plotfile = os.path.join('static/images/', 'output.png')
    plt.savefig(plotfile,dpi=resolution)
    # dataout=np.random((10))
    ### save time series in csv file
    df.to_csv(os.path.join('static/images/', 'output.csv'))
    return plotfile


# if __name__ == '__main__':
#     print(compute(1, 0.1, 1, 20))
