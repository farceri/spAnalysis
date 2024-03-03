'''
Created by Francesco
14 July 2023
'''
#functions for clustering visualization
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import splev, splrep
import scipy.stats as st
from scipy.special import kn
import itertools
import sys
import os
import utils
import spCluster as cluster
import spInterface as interface

def lineFit(x, a, b):
    return a*x + b

def powerLaw(x, a, b):
    return a*x**b

def powerLawMass(x, a, b, c):
    return a*(x**b + c)

def hyperbolicTan(x, a, b, x0, w):
    return 0.5*(a+b) - 0.5*(a-b)*np.tanh(2*(x-x0)/w)

def hyperbolicSecSq(x, a, x0, w):
    return a/np.cosh(2*(x-x0)/w)*2/w

def curveCumSum(x, a, b, c):
    return 1 - c*np.exp(-(x*a)**b)

def besselFunc(x, a, b):
    return b*kn(0,(x/a))

def curveCvv(x, a, b):
    return a * np.exp(-b*x)

def curve4Poly(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def curve2Poly(x, a, b, c):
    return a*x**2 + b*x + c

#################################################################################
############################# Interface properties ##############################
#################################################################################
def plotSPClusterHeightFluctuations(dirName, figureName, which='active', qmax=1):
    sigma = utils.readFromDynParams(dirName, 'sigma')
    fig, ax = plt.subplots(2, 1, figsize=(7,7), dpi = 120)
    if not(os.path.exists(dirName + os.sep + "heightFluctuations.dat")):
        if(which=='lj'):
            cluster.averageClusterHeightFluctuations(dirName, 0.3)
        elif(which=='active'):
            cluster.averageClusterHeightFluctuations(dirName, 0.78)
    data = np.loadtxt(dirName + os.sep + "heightFluctuations.dat")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    ly = boxSize[1]*sigma
    data[:,4:] *= ly
    ax[0].errorbar(data[:,0]/sigma, data[:,1]/sigma**2, data[:,2]/sigma**2, lw=1, marker='o', markersize=6, color='k', fillstyle='none', capsize=3, elinewidth=0.5)
    ax[1].errorbar(data[1:,3], data[1:,4], data[1:,5], lw=1, marker='o', markersize=6, color='k', fillstyle='none', capsize=3, elinewidth=0.5)
    data = data[data[:,3]<qmax]
    x = data[1:,3]
    y = data[1:,4]
    failed = False
    try:
        popt, pcov = curve_fit(powerLawMass, x, y, bounds=([0, -2.01, -np.inf], [np.inf, -1.99, np.inf]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        ax[1].plot(x, powerLawMass(x, *popt), color='b', lw=1.2, ls='solid', label="$Fit,$ $a(q^b+c):$ $a=$" + str(np.around(popt[0],3)) + ", $b=$" + str(np.around(popt[1],3)) + ", $c=$" + str(np.around(popt[2],3)))
        print("fitting parameters - a:", popt[0], "b:", popt[1])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].legend(fontsize=12, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$q$", fontsize=16)
    ax[1].set_ylabel("$\\langle |\delta h(q)|^2 \\rangle L$", fontsize=16)
    ax[0].set_xlabel("$y$", fontsize=16)
    ax[0].set_ylabel("$\\langle |\delta h(y)|^2 \\rangle$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pHeightFlu-" + figureName + ".png"
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPClusterHeightVSTemp(dirName, figureName, which='active', qmax=0.4):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        dirList = np.array(['0.35', '0.36', '0.37', '0.38'])
        labelName = "$T=$"
    elif(which=='active'):
        dirList = np.array(['4e-04', '3e-04', '2e-04', '1.5e-04', '1.2e-04', '1e-04', '9e-05', '7e-05', '5e-05'])
        labelName = "$D_r=$"
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('plasma', dirList.shape[0]+2)
    taup = np.zeros(dirList.shape[0])
    temp = np.zeros((dirList.shape[0],2))
    slope = np.zeros(dirList.shape[0])
    power = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(which=='lj'):
            dirSample = dirName + "/box6-2/0.238/langevin-lj/T" + dirList[d] + "/dynamics/"
            labelName = '$T$ = '
        elif(which=='active'):
            dirSample = dirName + "/box6-2/0.30/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            labelName = '$D_r$ = '
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "heightFluctuations.dat")):
            if(which=='lj'):
                cluster.averageClusterHeightFluctuations(dirSample, 0.3)
            elif(which=='active'):
                cluster.averageClusterHeightFluctuations(dirSample, 0.78)
        boxSize = np.loadtxt(dirSample + os.sep + "boxSize.dat")
        ly = boxSize[1]*sigma
        data = np.loadtxt(dirSample + "heightFluctuations.dat")
        data[:,4:] *= ly
        ax.plot(data[1:,3], data[1:,4], lw=1, markersize=6, color=colorList(d/dirList.shape[0]), fillstyle='none', label=labelName + dirList[d])
        data = data[1:,:]
        data = data[data[:,3]<qmax]
        x = data[2:,3]
        y = data[2:,4]
        failed = False
        try:
            #popt, pcov = curve_fit(powerLaw, x, y)
            popt, pcov = curve_fit(powerLawMass, x, y, bounds=([0, -2.01, -np.inf], [np.inf, -1.99, np.inf]))
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            #ax.plot(x, powerLaw(x, *popt), color='k', lw=1.2, ls='dashed')#, label="$Fit,$ $aq^b:$ $a=$" + str(np.around(popt[0],3)) + ", $b=$" + str(np.around(popt[1],3)))
            ax.plot(x, powerLawMass(x, *popt), color='k', lw=1.2, ls='dashed')#, label="$Fit,$ $aq^b:$ $a=$" + str(np.around(popt[0],3)) + ", $b=$" + str(np.around(popt[1],3)))
            #ax.set_yscale('log')
            #ax.set_xscale('log')
            #plt.pause(0.5)
            print("Dr", dirList[d], "slope (stiffness):", popt[0], "power(2):", popt[1])
            slope[d] = popt[0]
            power[d] = popt[1]
        # compute length of interface
        if not(os.path.exists(dirSample + os.sep + "clusterTemperature.dat")):
            cluster.computeClusterTemperatureVSTime(dirSample)
        data = np.loadtxt(dirSample + os.sep + "clusterTemperature.dat")
        temp[d,0] = np.mean(data[:,0])
        temp[d,1] = np.std(data[:,0])
    ax.legend(fontsize=11, ncol=2, loc='best')
    ax.set_xlabel("$q$", fontsize=16)
    ax.set_ylabel("$\\langle |\delta h(q)|^2 \\rangle L$", fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/pHeightFluVSLength-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    # second plot
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        x = temp[:,0]
        xlabel = "$Temperature,$ $T$"
    elif(which=='active'):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Fitting$ $parameters$", fontsize=16)
    ax.plot(x, slope, lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', label="$Slope,$ $\\frac{U}{\\gamma}$")
    ax.plot(x, power, lw=1.2, marker='v', markersize=8, color='b', fillstyle='none', label="$Power$")
    ax.legend(fontsize=14, loc='best')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/stiffVSTemp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterHeightCorrelation(dirName, figureName, which='Dr'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='Dr'):
        dirList = np.array(['3e-04', '2e-04', '1.5e-04', '1.2e-04', '9e-05', '7e-05', '5e-05'])
    elif(which=='T'):
        dirList = np.array(['0.41', '0.35'])
    markerList = ['o', 'v', '^', 's', '*', 'd', 'D', '+', 'x']
    gamma = np.zeros(dirList.shape[0])
    temp = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(which=='Dr'):
            dirSample = dirName + os.sep + which + dirList[d] + "/dynamics/"
        elif(which=='T'):
            dirSample = dirName + os.sep + which + dirList[d] + "-removed/dynamics/"
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + os.sep + "clusterHeightCorr.dat")):
            cluster.computeClusterInterfaceCorrelation(dirSample)
        data = np.loadtxt(dirSample + os.sep + "clusterHeightCorr.dat")
        #ax.errorbar(data[:,0], data[:,1], data[:,2], lw=1, marker=markerList[d], markersize=6, color='k', fillstyle='none', capsize=3, elinewidth=0.5, label=which + '$=$' + dirList[d])
        ax.plot(data[1:,0], data[1:,1]/data[0,1], lw=1, marker=markerList[d], markersize=6, color='k', fillstyle='none', label=which + '$=$' + dirList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Distance,$ $\Delta x$", fontsize=16)
    ax.set_ylabel("$g_h(\Delta x)$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pHeightCorr-" + figureName + "-vs" + which + ".png"
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPDensityProfile(dirName, figureName):
    #print("taup:", 1/(utils.readFromDynParams(dirName, "Dr")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if not(os.path.exists(dirName + "densityProfile.dat")):
        cluster.averageLinearDensityProfile(dirName)
    data = np.loadtxt(dirName + "densityProfile.dat")
    center = np.mean(data[:,0])
    x = np.abs(data[:,0]-center)
    y = data[np.argsort(x),1]
    yerr = data[np.argsort(x),2]
    x = np.sort(x)
    ax.errorbar(x, y, yerr, lw=1, marker='o', markersize=6, color='k', capsize=3, fillstyle='none', label='$Profile$')
    failed = False
    try:
        popt, pcov = curve_fit(hyperbolicTan, x, y, bounds=([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        ax.plot(x, hyperbolicTan(x, *popt), color='g', lw=2, linestyle='dashed', label='$Fit$')
        print("center - x0:", popt[2], "width:", popt[3], 'phi-:', popt[0] - popt[1], 'phi+:', popt[0] + popt[1])
        width = popt[3]
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$\\varphi(x)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/phiProfile-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterWidth(dirName, figureName, which='active', param='1e-04'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    dirList = np.array(['0.6', '0.8', '1', '1.2', '1.4', '1.6', '1.8', '2', '2.2'])
    markerList = ['o', 's', 'd', 'v', '^', 'D', 'x', '*', 'h']
    colorList = cm.get_cmap('plasma', dirList.shape[0]+2)
    ly = dirList.astype(np.float64)
    length = np.zeros((dirList.shape[0],2))
    width1 = np.zeros((dirList.shape[0],2))
    width2 = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(which=='lj'):
            dirSample = dirName + "/box6-" + dirList[d] + "/0.238/langevin-lj/T" + param + "/dynamics/"
        elif(which=='active'):
            dirSample = dirName + "/box6-" + dirList[d] + "/0.30/active-langevin/Dr" + param + "/dynamics/"
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "energyLength.dat")):
            if(which=='lj'):
                cluster.averageClusterHeightFluctuations(dirSample, 0.3)
            elif(which=='active'):
                cluster.averageClusterHeightFluctuations(dirSample, 0.78)
        energyLength = np.loadtxt(dirSample + "energyLength.dat")
        energyLength[:,1] /= sigma
        length[d,0] = np.mean(energyLength[:,1])
        length[d,1] = np.std(energyLength[:,1])
        if not(os.path.exists(dirSample + "densityProfile.dat")):
            if(which=='lj'):
                cluster.averageLinearDensityProfile(dirSample, 0.3)
            elif(which=='active'):
                cluster.averageLinearDensityProfile(dirSample, 0.78)
        data = np.loadtxt(dirSample + "densityProfile.dat")
        center = np.mean(data[:,0])
        x = np.abs(data[:,0]-center)
        y = data[np.argsort(x),1]
        yerr = data[np.argsort(x),2]
        x = np.sort(x)
        ax.errorbar(x, y, yerr, lw=1, color=colorList(d/dirList.shape[0]), marker=markerList[d], markersize=6, capsize=3, fillstyle='none', label="$L_y=$" + dirList[d])
        #ax.plot(data[:,0], data[:,1], lw=1, marker=markerList[d], markersize=6, color='k', fillstyle='none', label=labelName + dirList[d])
        width2[d] = utils.computeInterfaceWidth(x, y)/sigma
        data = np.loadtxt(dirSample + "interfaceWidth.dat")/sigma
        width1[d,0] = np.mean(data[:,1]**2)
        width1[d,1] = np.std(data[:,1]**2)/np.sqrt(data.shape[0])
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$\\varphi(x)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/pProfileVSLength-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    # second plot
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax.set_ylabel("$Squared$ $width,$ $w^2$", fontsize=16)
    ax.errorbar(ly, width1[:,0], width1[:,1], lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', capsize=3, label='$Average$ $from$ $profiles$')
    ax.plot(ly, width2**2, lw=1.2, marker='v', markersize=8, color='b', fillstyle='none', label='$From$ $averaged$ $profile$')
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/widthVSLength-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterWidthVSTemp(dirName, figureName, which='passive'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='passive'):
        dirList = np.array(['0.30', '0.31', '0.32', '0.33', '0.34', '0.35', '0.36', '0.37', '0.38',
                            '0.39', '0.40', '0.41', '0.42', '0.43', '0.44', '0.45'])
    elif(which=='active'):
        dirList = np.array(['1', '2', '4', '8'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    temp = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    width = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirSample = dirName + "/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "/T0.30-tp1e-03-f0" + dirList[d] + "/dynamics/"
            labelName = "$f_0=$"
        if not(os.path.exists(dirSample + "densityProfile.dat")):
            interface.averageLinearDensityProfile(dirSample)
        data = np.loadtxt(dirSample + "densityProfile.dat")
        #ax.errorbar(data[:,0], data[:,1], data[:,2], lw=1, marker='o', markersize=6, color=colorList(d/dirList.shape[0]), capsize=3, fillstyle='none', label=labelName + dirList[d])
        center = np.mean(data[:,0])
        x = data[:,0]-center
        y = data[:,1]
        yerr = data[:,2]
        y = y[x>0]
        yerr = yerr[x>0]
        x = x[x>0]
        #x = (x[1:] + x[:-1])/2
        #y = (y[1:] - y[:-1])/2
        #yerr = (yerr[1:] + yerr[:-1])/2
        ax.errorbar(x, y, yerr, lw=1, marker='o', markersize=6, color=colorList(d/dirList.shape[0]), capsize=3, fillstyle='none', label=labelName + dirList[d])
        failed = False
        try:
            popt, pcov = curve_fit(hyperbolicTan, x, y, bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]))
            #popt, pcov = curve_fit(hyperbolicSecSq, x, y)
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            ax.plot(x, hyperbolicTan(x, *popt), color='k')
            #ax.plot(x, hyperbolicSecSq(x, *popt), color='k')
            width[d,2] = popt[3]
            print("fitting parameters - a, b, x0, w:", popt)
        #width[d,2] = utils.computeInterfaceWidth(x, y)
        data = np.loadtxt(dirSample + "interfaceWidth.dat")
        width[d,0] = np.mean(data[:,1])
        width[d,1] = np.std(data[:,1])
        if not(os.path.exists(dirSample + "clusterTemperature.dat")):
            interface.computeClusterTemperatureVSTime(dirSample)
        data = np.loadtxt(dirSample + "clusterTemperature.dat")
        temp[d] = np.mean(data[:,1])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.set_xlabel("$x/L_x$", fontsize=16)
    ax.set_ylabel("$Profile,$ $\\varphi(x)$", fontsize=16)
    figure1Name = "/home/francesco/Pictures/soft/mips/profileVSTemp-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        x = temp
        xlabel = "$Temperature,$ $T/ \\varepsilon$"
    elif(which=='active'):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p / \\tau_i$"
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Interface$ $width,$ $w/L_x$", fontsize=16)
    ax.errorbar(x, width[:,0], width[:,1], lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', capsize=3, label='$Average$ $of$ $fits$')
    #ax.plot(x, width[:,2], lw=1.2, marker='v', markersize=8, color='g', markeredgecolor='k', fillstyle='full', label='$Fit$ $after$ $average$')
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/widthVSTemp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPActiveEnergyVSTemp(dirName, figureName):
    fig, ax = plt.subplots(figsize=(7.5,5.5), dpi = 120)
    # plot passive sample
    tempList = np.array(['0.20', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26', '0.27', '0.28', '0.29', 
                         '0.30', '0.31', '0.32', '0.33', '0.34', '0.35', '0.36', '0.37', '0.38', '0.39', 
                         '0.40', '0.41', '0.42', '0.43', '0.44', '0.45'])
    temp = np.zeros((tempList.shape[0],2))
    epot = np.zeros((tempList.shape[0],2))
    for d in range(tempList.shape[0]):
        dirSample = dirName + "box/0.30compress/langevin-lj/T" + tempList[d] + "/dynamics/"
        data = np.loadtxt(dirSample + "energy.dat")
        temp[d,0] = np.mean(data[:,3])
        temp[d,1] = np.std(data[:,3])
        epot[d,0] = np.mean(data[:,2])
        epot[d,1] = np.std(data[:,2])
    ax.errorbar(temp[:,0], epot[:,0], epot[:,1], temp[:,1], lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', capsize=3, label='$Passive$')
    # plot active samples
    tempList = np.array(['0.30'])
    markerList = ['s', 'v', 'd', 'D', '^']
    f0List = np.array(['1', '2', '4'])
    colorList = ['c', 'g', 'b', 'y', 'r']
    ratio = np.zeros((tempList.shape[0], f0List.shape[0]))
    ftList = np.zeros(tempList.shape[0])
    for i in range(tempList.shape[0]):
        ftList[i] = np.sqrt(2*np.sqrt(10)*float(tempList[i]))
        for j in range(f0List.shape[0]):
            ratio[i,j] = ftList[i] / float(f0List[j])
            #print(ftList[i], f0List[j], ratio[i,j])
    tpList = np.array(['1e-03', '1e-02', '2e-02', '4e-02', '6e-02', '8e-02', '1e-01', '2e-01', '4e-01', '6e-01', '8e-01', #11
                        '1', '2', '4', '6', '8', '10', '20', '40', '60', '80', '100', '200', '400', '600', '800', '1000', #27
                        '2000', '4000', '6000', '8000', '10000'])
    temp = np.zeros((tpList.shape[0],2))
    epot = np.zeros((tpList.shape[0],2))
    taup = np.zeros(tpList.shape[0])
    for t in range(tempList.shape[0]):
        for f in range(f0List.shape[0]):
            scale = float(f0List[f])
            labelName = "$\\frac{\\sqrt{2\\gamma k_B T}}{f_0} =$" + str(np.format_float_positional(ratio[t,f],2))
            labelName = "$f_T=$" + str(np.format_float_positional(ftList[t],2)) + ", $f_0=$" + f0List[f]
            for d in range(tpList.shape[0]):
                #print("T:", tempList[t], "f0:", f0List[f], "tp:", tpList[d], "Peclet:", float(f0List[f])*float(tpList[d]) / np.sqrt(10))
                dirSample = dirName + "box/0.30compress/langevin-lj/T" + tempList[t] + "/active-lj/T" + tempList[t] + "-tp" + tpList[d] + "-f0" + f0List[f] + "/"
                taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
                data = np.loadtxt(dirSample + "energy.dat")
                temp[d,0] = np.mean(data[:,3])
                temp[d,1] = np.std(data[:,3])
                epot[d,0] = np.mean(data[:,2])
                epot[d,1] = np.std(data[:,2])
                fill = 'none'
                edge = colorList[f]
                if((t==2 and f==3) and (d>3 and d<20)):
                    fill = 'full'
                    edge = 'k'
                if((t==2 and f==2) and (d>6 and d<21)):
                    fill = 'full'
                    edge = 'k'
                if((t==2 and f==1) and (d>11 and d<22)):
                    fill = 'full'
                    edge = 'k'
                if((t==1 and f==3) and (d>4 and d<20)):
                    fill = 'full'
                    edge = 'k'
                if((t==1 and f==2) and (d>7 and d<21)):
                    fill = 'full'
                    edge = 'k'
                if((t==1 and f==1) and (d>12 and d<22)):
                    fill = 'full'
                    edge = 'k'
                if((t==0 and f==3) and (d>6 and d<20)):
                    fill = 'full'
                    edge = 'k'
                if((t==0 and f==2) and (d>9 and d<21)):
                    fill = 'full'
                    edge = 'k'
                if((t==0 and f==1) and (d>14 and d<22)):
                    fill = 'full'
                    edge = 'k'
                if(d==0):
                    ax.errorbar(temp[d,0], epot[d,0], epot[d,1], temp[d,1], marker=markerList[t], markersize=10, color=colorList[f], markeredgecolor=edge, fillstyle=fill, capsize=3, label=labelName)
                else:
                    ax.errorbar(temp[d,0], epot[d,0], epot[d,1], temp[d,1], marker=markerList[t], markersize=10, color=colorList[f], markeredgecolor=edge, fillstyle=fill, capsize=3)
            ax.errorbar(temp[:,0], epot[:,0], epot[:,1], temp[:,1], lw=1.2, color=colorList[f], capsize=3)
    ax.legend(fontsize=11, loc='lower right', ncol=2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=16)
    ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/energyVSTemp-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterISF(dirName, figureName):
    fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    timeStep = utils.readFromParams(dirName, "dt")
    data = np.loadtxt(dirName + "/clusterLogCorr.dat")
    ax.semilogx(data[:,0]*timeStep, data[:,2], linewidth=1.2, linestyle='dashdot', color='k', label="$q = \\frac{2\\pi}{\\sigma}$")
    ax.semilogx(data[:,0]*timeStep, data[:,3], linewidth=1.2, color='g', label="$q = \\frac{2\\pi}{L}$")
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Elapsed$ $time,$ $t$", fontsize=16)
    ax.set_ylabel("$ISF(t)$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/relaxISF-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPWallForceVSTime(dirName, figureName, which='average', index=4, numSamples=20):
    fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    count = np.sqrt(np.mean(np.loadtxt(dirName + "/../count/energy.dat")[:,4]))
    if(os.path.exists(dirName + "/../count/singleProfile.dat")):
        data = np.loadtxt(dirName + "/../count/singleProfile.dat")
        x = data[:,0]
        y = data[:,1]
        xFluid = x[np.argwhere(y>0.5)[:,0]]
        width = xFluid[-1] - xFluid[0]
    else:
        width, _,_ = interface.computeLinearDensityProfile(dirName + "/../count/")
    boxSize = np.loadtxt(dirName + "/../count/boxSize.dat")
    boxWidth = boxSize[0]
    if(which=="single"):
        data = np.loadtxt(dirName + "/energy.dat")
        ax.plot(data[:,1], data[:,index], linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k')
    elif(which=="average"):
        data = np.loadtxt(dirName + "/0/energy.dat")
        time = data[:,1]
        wall = np.zeros((numSamples, data.shape[0]))
        wall[0] = data[:,index]
        for i in range(1,numSamples):
            data = np.loadtxt(dirName + "/" + str(i) + "/energy.dat")
            wall[i] = data[:,index]#*width/(count*boxWidth)
        mean = np.mean(wall, axis=0)
        error = np.std(wall, axis=0)
        ax.errorbar(time, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_xscale('log')
    if(index==2):
        ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        figureName = "/home/francesco/Pictures/soft/mips/wallEnergyVSTime-" + figureName
    elif(index==4):
        ax.set_ylabel("$Force$ $on$ $wall,$ $F$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallForceVSTime-" + figureName
    elif(index==5):
        ax.set_ylabel("$Pressure,$ $p$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallPressureVSTime-" + figureName
    plt.tight_layout()
    if(which=='average'):
        figureName += "-average"
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPWallForceVSStrain(dirName, figureName, which='average', index=4, limit=130):
    fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    numParticles = utils.readFromParams(dirName, "numParticles")
    #count = np.sqrt(np.mean(np.loadtxt(dirName + "/../count/energy.dat")[:,4]))
    count = np.sqrt(16384)
    print(count, np.sqrt(numParticles))
    dirList, strain = utils.getOrderedStrainDirectories(dirName)
    mean = np.zeros(dirList.shape[0])
    error = np.zeros(dirList.shape[0])
    length = np.zeros(dirList.shape[0])
    width = np.zeros(dirList.shape[0])
    boxWidth = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d]
        if(which=='average'):
            dirSample += "/simple/"
        if(os.path.exists(dirSample)):
            #print(dirSample)
            if(os.path.exists(dirSample + os.sep + "singleProfile.dat")):
                data = np.loadtxt(dirSample + os.sep + "singleProfile.dat")
                x = data[:,0]
                y = data[:,1]
                xFluid = x[np.argwhere(y>0.5)[:,0]]
                width[d] = xFluid[-1] - xFluid[0]
            else:
                width[d], _,_ = interface.computeLinearDensityProfile(dirSample)
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
            boxWidth[d] = boxSize[0]
            length[d] = boxSize[1]
            data = np.loadtxt(dirSample + "/energy.dat")
            if(index==2):
                mean[d] = np.mean(data[:,index] + 0.5 * data[:,index+1])
                error[d] = np.std(data[:,index] + 0.5 * data[:,index+1])
            else:
                mean[d] = np.mean(data[:,index]*width[d]/(count*boxWidth[d]))
                error[d] = np.std(data[:,index]*width[d]/(count*boxWidth[d]))/10
    if(which=='average'):
        mean = mean[mean!=0]
        error = error[error!=0]
        strain = strain[length!=0]
        length = length[length!=0]
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Strain,$ $\\epsilon$", fontsize=16)
    #ax.set_ylim(-0.04, 0.64)
    ax.errorbar(strain, mean, error, lw=1, marker='s', markersize=5, color='k', capsize=3, markeredgecolor='k', fillstyle='left', label='$From$ $simulation$')
    if(index==2):
        ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        figure1Name = "/home/francesco/Pictures/soft/mips/wallEnergyVSStrain-" + figureName
        # fit line to energy versus length or strain
        energy = mean
        failed = False
        try:
            #popt, pcov = curve_fit(lineFit, strain[strain<limit], energy[strain<limit])
            popt, pcov = curve_fit(lineFit, length[length<limit], energy[length<limit])
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            #ax.plot(strain, lineFit(strain, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax^2 + bx + c$")
            ax.plot(length, lineFit(length, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax^2 + bx + c$")
            print("Energy: a, b:", popt, "line tension from fit:", popt[0])
    elif(index==4):
        ax.set_ylabel("$\\frac{F \\sigma}{\\varepsilon_{LJ}}$", fontsize=24, rotation='horizontal', labelpad=20)
        figure1Name = "/home/francesco/Pictures/soft/mips/wallForceVSStrain-" + figureName
    elif(index==5):
        ax.set_ylabel("$Pressure,$ $p$", fontsize=16)
        figure1Name = "/home/francesco/Pictures/soft/mips/wallPressureVSStrain-" + figureName
    plt.tight_layout()
    if(which=='average'):
        figure1Name += "-average"
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    if(index==4):
        fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
        dirSample = dirName + "/../simple/"
        boxSize = np.loadtxt(dirSample + "boxSize.dat")
        if(os.path.exists(dirSample + "singleProfile.dat")):
            data = np.loadtxt(dirSample + "singleProfile.dat")
            x = data[:,0]
            y = data[:,1]
            xFluid = x[np.argwhere(y>0.5)[:,0]]
            width = xFluid[-1] - xFluid[0]
        else:
            width, _,_ = interface.computeLinearDensityProfile(dirSample)
        data = np.loadtxt(dirSample + "energy.dat")
        temp = np.mean(data[:,3])
        ertemp = np.std(data[:,3])
        data[:,4] *= width / (count * boxSize[0])
        force0 = np.array([np.mean(data[:,4]), np.std(data[:,4])])
        force = mean
        length0 = boxSize[1]
        print("average force at zero strain:", force0[0], "+-", force0[1], "height: ", boxSize[1])
        print("average delta force:", np.mean(mean[1:]), "+-", np.std(mean[1:]))
        work = np.zeros(force.shape[0])
        #delta = strain[1] - strain[0]
        delta = length[1] - length[0]
        #width = np.mean(width[:-1] - width[1:])
        work = np.zeros(force.shape[0])
        for i in range(work.shape[0]):
            #work[i] = force[i] * strain[i] * width[i]
            #work[i] += force[i] * delta
            #work[i] = 0.5 * force[i] * delta
            work[i] = 0.5 * np.sum(force[:i]) * delta
            #work[i] = 0.5 * np.sum(force[:i] - force0[0]) * delta
            #work[i] = np.sum(force0[0] - force[:i]) * delta
        failed = False
        try:
            #popt, pcov = curve_fit(lineFit, strain[strain<limit], work[strain<limit])
            popt, pcov = curve_fit(lineFit, length[length<limit], work[length<limit])
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            #ax.plot(strain, lineFit(strain, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax^2 + bx + c$")
            ax.plot(length/length0 - 1, lineFit(length, *popt), color='b', lw=1.2, linestyle='dashdot', label="$Linear$ $fit$")
            print("Work: a, b:", popt, "line tension from fit:", popt[0], popt[0]/temp)
            print("Temperature:", temp, ertemp)
        #ax.plot(strain, work, marker='o', fillstyle='none', color='k', lw=1.2)
        ax.plot(length/length0 - 1, work, marker='o', markersize=8, fillstyle='none', color='k', lw=1, markeredgewidth=1.2)
        ax.tick_params(axis='both', labelsize=14)
        #ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
        ax.set_xlabel("$Strain,$ $\\epsilon$", fontsize=16)
        ax.set_ylabel("$\\frac{1}{2}\\int_{H^0}^{H}$ $F(H')$ $dH'$", fontsize=16)
        #ax.set_ylabel("$W(L_y) - W(L_y^0)$", fontsize=16)
        plt.tight_layout()
        figure2Name = "/home/francesco/Pictures/soft/mips/workVSlength-" + figureName
        fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPWidthVSStrain(dirName, figureName, which='average', limit=130):
    fig1, ax1 = plt.subplots(figsize=(6,5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize=(6,5), dpi = 120)
    dirList, strain = utils.getOrderedStrainDirectories(dirName)
    width = np.zeros(dirList.shape[0])
    liquid = np.zeros(dirList.shape[0])
    vapor = np.zeros(dirList.shape[0])
    strain = np.zeros(dirList.shape[0])
    length = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d]
        if(which=='average'):
            dirSample += "/simple/"
        if(os.path.exists(dirSample)):
            #print(dirSample)
            if(os.path.exists(dirSample + os.sep + "singleProfile.dat")):
                data = np.loadtxt(dirSample + os.sep + "singleProfile.dat")
                x = data[:,0]
                y = data[:,1]
                xFluid = x[np.argwhere(y>0.5)[:,0]]
                width[d] = xFluid[-1] - xFluid[0]
                liquid[d] = np.mean(y[y>0.5])
                vapor[d] = np.mean(y[y<0.5])
            else:
                width[d], liquid[d], vapor[d] = interface.computeLinearDensityProfile(dirSample)
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
            length[d] = boxSize[1]
    if(which=='average'):
        width = width[length!=0]
        liquid = liquid[length!=0]
        vapor = vapor[length!=0]
        strain = strain[length!=0]
        length = length[length!=0]
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax1.set_ylabel("$Liquid$ $width,$ $w$", fontsize=16)
    ax1.plot(length, width * length, lw=1, marker='v', markersize=5, color='k', markeredgecolor='k', fillstyle='none')
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax2.set_ylabel("$Density,$ $\\varphi$", fontsize=16)
    ax2.plot(length, liquid, lw=1, marker='o', markersize=8, color='b', fillstyle='none', label='$Liquid$')
    ax2.plot(length, vapor, lw=1, marker='s', markersize=8, color='g', fillstyle='none', label='$Vapor$')
    figure1Name = "/home/francesco/Pictures/soft/mips/widthVSStrain-" + figureName
    figure2Name = "/home/francesco/Pictures/soft/mips/densityVSStrain-" + figureName
    fig1.tight_layout()
    fig2.tight_layout()
    if(which=='average'):
        figure1Name += "-average"
        figure2Name += "-average"
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig2.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotForcePDF(dirName, figureName, index=4, numBins=20):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    data = np.loadtxt(dirName + "/energy.dat")[:,index]
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax.plot(edges[pdf>0], pdf[pdf>0], marker='o', markersize=8, lw=1, fillstyle='none', markeredgewidth=1, color='k')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$F_{wall}$", fontsize=16)
    ax.set_ylabel("$PDF(F_{wall})$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/forcePDF-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotForcePDFVSSystemSize(dirName, figureName, type='passive', index=4, numBins=20):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    dirList = np.array(['8192', '16384', '32768'])
    colorList = ['b', 'g', 'c']
    numParticles = np.zeros(dirList.shape[0])
    mean = np.zeros(dirList.shape[0])
    error = np.zeros(dirList.shape[0])
    if(type=='passive'):
        dirType = "-2d/box/0.33/langevin-lj/T0.42/dynamics/"
    elif(type=='active'):
        dirType = "-2d/box/0.33/langevin-lj/T0.42/active-lj/T0.42-tp1e-04-f01/dynamics/"
    labelName = "$N=$"
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d] + dirType
        if(os.path.exists(dirSample)):
            print(dirSample)
            numParticles[d] = utils.readFromParams(dirSample, "numParticles")
            data = np.loadtxt(dirSample + "/energy.dat")[:,index]/np.sqrt(numParticles[d])
            mean[d] = np.mean(data)
            error[d] = np.std(data)
            print(np.mean(data), np.std(data))
            pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), numBins), density=True)
            edges = (edges[1:] + edges[:-1])/2
            ax.plot(edges[pdf>0], pdf[pdf>0], marker='o', markersize=8, lw=1, fillstyle='none', markeredgewidth=1, color=colorList[d], label=labelName + dirList[d])
    #ax.errorbar(numParticles, mean, error, marker='o', markersize=8, lw=1, fillstyle='none', capsize=3, color='k')
    #ax.set_ylim(-0.4,0.84)
    #ax.set_xscale('log', base=2)
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=12)
    #ax.set_ylabel("$F_{wall}$", fontsize=16)
    #ax.set_xlabel("$Particle$ $number,$ $N$", fontsize=16)
    ax.set_xlabel("$F_{wall} / \\sqrt{N}$", fontsize=16)
    ax.set_ylabel("$PDF(F_{wall} / \\sqrt{N})$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/forcePDFvsN-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotForcePDFVSStrain(dirName, figureName, which="average", index=4, numBins=20):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    dirList = np.array(['0.0020', '0.0080', '0.0120', '0.0160', '0.0200', '0.02400', '0.0280', '0.0300'])
    #dirList = np.array(['0.0001', '0.0002', '0.0003', '0.0004', '0.0005', '0.0006', '0.0007', '0.0008', '0.0009', '0.0010'])
    #dirList = np.array(['0.0030', '0.0060', '0.0090', '0.0120', '0.0150'])
    #dirList = np.array(['0.0030', '0.0120', '0.0240', '0.0360', '0.0480', '0.0570'])#, '0.0720', '0.0750', '0.0780', '0.0810', '0.0840'])
    colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
    labelName = "$\\gamma=$"
    # unstrained system
    #if(which=="average"):
    #    data = np.loadtxt(dirName + "../simple/energy.dat")[:,index]
    #else:
    #    data = np.loadtxt(dirName + "../energy.dat")[:,index]
    #pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), numBins), density=True)
    #edges = (edges[1:] + edges[:-1])/2
    #ax.plot(edges[pdf>0], pdf[pdf>0], marker='s', markersize=8, lw=1, fillstyle='left', markeredgewidth=1, color='k', label=labelName + "0")
    for d in range(dirList.shape[0]):
        if(which=="average"):
            dirSample = dirName + "strain" + dirList[d] + "/simple/"
        else:
            dirSample = dirName + "strain" + dirList[d]
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "/energy.dat")[:,index]
            print(dirList[d], np.mean(data), np.std(data))
            pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), numBins), density=True)
            edges = (edges[1:] + edges[:-1])/2
            ax.plot(edges[pdf>0], pdf[pdf>0], marker='o', markersize=8, lw=1, fillstyle='none', markeredgewidth=1, color=colorList(d/dirList.shape[0]), label=labelName + dirList[d])
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlabel("$F_{wall}$", fontsize=16)
    ax.set_ylabel("$PDF(F_{wall})$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/forcePDFvsStrain-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSampleWallForceVSTime(dirName, figureName, type='active', which='average', index=4, numSamples=30):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(type=='active'):
        dirType = "langevin-lj/T0.30/active-lj/T0.30-tp1e-03-f02/biaxial4e-02/"
    elif(type=='passive'):
        dirType = "langevin-lj/T0.30/biaxial4e-02/"
    else:
        print("please specify sample type, active or passive")
    if(which=="single"):
        data = np.loadtxt(dirName + "/energy.dat")
        ax.plot(data[:,1], data[:,index], linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k')
    elif(which=="average"):
        data = np.loadtxt(dirName + "-0/" + dirType + "energy.dat")
        time = data[:,1]
        wall = np.zeros((numSamples, data.shape[0]))
        wall[0] = data[:,index]
        for i in range(1,numSamples):
            data = np.loadtxt(dirName + "-" + str(i) + "/" + dirType + "energy.dat")
            wall[i] = data[:,index]
        mean = np.mean(wall, axis=0)
        error = np.std(wall, axis=0)
        ax.errorbar(time, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_xscale('log')
    if(index==2):
        ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        figureName = "/home/francesco/Pictures/soft/mips/wallEnergyVSTime-" + figureName
    elif(index==4):
        ax.set_ylabel("$Force$ $on$ $wall,$ $F_{wall}$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallForceVSTime-" + figureName
    elif(index==5):
        ax.set_ylabel("$Pressure,$ $p$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallPressureVSTime-" + figureName
    plt.tight_layout()
    if(which=='average'):
        figureName += "-average"
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def readForceFileAndLength(dirName, dirList, strain, index=4, which=0):
    force = np.zeros(dirList.shape[0])
    Ly = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d]
        if(which=='average'):
                dirSample += "/dynamics/"
        if(os.path.exists(dirSample)):
            #print(dirSample)
            data = np.loadtxt(dirSample + "/energy.dat")
            force[d] = np.mean(data[:,index])
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
            Ly[d] = boxSize[1]
    if(which=='average'):
        force = force[force!=0]
        strain = strain[Ly!=0]
        Ly = Ly[Ly!=0]
    return force, Ly, strain

def readForceFile(dirName, dirList, index=4, which=0):
    force = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d]
        if(which=='average'):
                dirSample += "/dynamics/"
        if(os.path.exists(dirSample)):
            #print(dirSample)
            data = np.loadtxt(dirSample + "/energy.dat")
            force[d] = np.mean(data[:,index])
    if(which=='average'):
        force = force[force!=0]
    return force

def plotSampleWallForceVSStrain(dirPath, figureName, type='active', which='average', index=4, numSamples=30, limit=130, maxStrain=0.6, temp="T0.30", taupf0="tp1e-03-f01"):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(type=='active'):
        dirType = "langevin-lj/T0.30/active-lj/T0.30-" + taupf0 + "/biaxial1e-03-tmax2e05/"
    elif(type=='passive'):
        dirType = "langevin-lj/" + temp + "/biaxial1e-03-tmax2e05/"
    else:
        print("please specify sample type, active or passive")
    # initialize force and energy with first sample
    dirName = dirPath + "-0/" + dirType
    dirList, strain = utils.getOrderedStrainDirectories(dirName)
    dirList = dirList[strain<maxStrain]
    strain = strain[strain<maxStrain]
    gamma = np.zeros(numSamples)
    force, Ly, strain = readForceFileAndLength(dirName, dirList, strain, index, which)
    wall = np.zeros((numSamples, force.shape[0]))
    # read other samples and average
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax.set_ylabel("$\\int_{L_y^0}^{L_y} \\left[ F(L_y^0) - F(L_y') \\right] dL_y'$", fontsize=16)
    for s in range(numSamples):
        dirName = dirPath + "-" + str(s) + "/" + dirType
        dirList, strainList = utils.getOrderedStrainDirectories(dirName)
        dirList = dirList[strainList<maxStrain]
        strainList = strainList[strainList<maxStrain]
        force = readForceFile(dirName, dirList, index, which)
        wall[s] = force
        work = np.zeros(force.shape[0]-1)
        length = Ly[1:]
        width = Ly[1] - Ly[0]
        for i in range(work.shape[0]):
            work[i] = np.sum((force[0]-force[1:i+1])*width)
        failed = False
        try:
            popt, pcov = curve_fit(lineFit, length[length<limit], work[length<limit])
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            ax.plot(length, lineFit(length, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax^2 + bx + c$")
            #print("Sample", s, "a, b:", popt, "line tension from fit:", popt[0])
            ax.plot(length, work, marker='o', fillstyle='none', color='k', lw=1.2)
            gamma[s] = popt[0]
            plt.tight_layout()
            plt.pause(0.2)
    gamma = gamma[gamma>0]
    print("Average line tension:", np.mean(gamma), "+-", np.std(gamma)/np.sqrt(gamma.shape[0]))
    mean = np.mean(wall, axis=0)
    error = np.std(wall, axis=0)
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
    #ax.errorbar(strain, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax.errorbar(Ly, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    #ax.set_xlim(-0.04, 0.84)
    if(index==2):
        ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        print("change in energy:", mean[-1] - mean[0], "change in strain:", strain[-1] - strain[0], "slope:", (mean[-1] - mean[0])/(strain[-1] - strain[0]))
        figure1Name = "/home/francesco/Pictures/soft/mips/sampleWallEnergyVSStrain-" + figureName
    elif(index==4):
        ax.set_ylabel("$\\frac{F_{wall} \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=28)
        figure1Name = "/home/francesco/Pictures/soft/mips/sampleWallPressureVSStrain-" + figureName
    elif(index==5):
        ax.set_ylabel("$Pressure,$ $p$", fontsize=16)
        figure1Name = "/home/francesco/Pictures/soft/mips/sampleWallPressureVSStrain-" + figureName
    if(which=='average'):
        figure1Name += "-average"
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    if(index==4):
        midForce = (mean[1:] + mean[:-1])*0.5
        #width = strain[1] - strain[0]
        midLength = (Ly[1:] + Ly[:-1])*0.5
        #width = strain[1] - strain[0]
        width = Ly[1] - Ly[0]
        work = np.zeros(midForce.shape[0]-1)
        for i in range(midForce.shape[0]-1):
            #work[i] = np.sum((midForce[0]-midForce[:i+1])*width)
            work[i] = np.sum((midForce[0]-midForce[:i+1])*width)
        #midStrain = midStrain[:-1]
        midLength = midLength[:-1]
        fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
        failed = False
        try:
            #popt, pcov = curve_fit(lineFit, midStrain[midStrain<limit], work[midStrain<limit])
            popt, pcov = curve_fit(lineFit, midLength[midLength<limit], work[midLength<limit])
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            #ax.plot(midStrain, lineFit(midStrain, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax^2 + bx + c$")
            ax.plot(midLength, lineFit(midLength, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax^2 + bx + c$")
            #print((lineFit(midStrain, *popt))/midStrain)
            print("From average force:", popt[0])
        #ax.plot(midStrain, work, marker='o', fillstyle='none', color='k', lw=1.2)
        ax.plot(midLength, work, marker='o', fillstyle='none', color='k', lw=1.2)
        ax.tick_params(axis='both', labelsize=14)
        #ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
        ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
        ax.set_ylabel("$\\int_{L_y^0}^{L_y} \\left[ F(L_y^0) - F(L_y') \\right] dL_y'$", fontsize=16)
        #ax.set_ylim(-0.6,11.6)
        figure2Name = "/home/francesco/Pictures/soft/mips/sampleTensionVSStrain-" + figureName
        plt.tight_layout()
        fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotGammaVSTemperature(fileName, figureName):
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    data = np.loadtxt(fileName)
    #ax.errorbar(data[:,0], data[:,1], data[:,2], color='k', marker='o', markersize=8, lw=1, fillstyle='none', capsize=3)
    ax.errorbar(data[:,0], data[:,1]/data[:,0], data[:,2], color='k', marker='o', markersize=8, lw=1, fillstyle='none', capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature,$ $T/\\varepsilon$", fontsize=16)
    #ax.set_ylabel("$\\gamma \\sigma$", fontsize=16)
    ax.set_ylabel("$\\frac{\\gamma \\sigma}{k_B T}$", rotation='horizontal', fontsize=24, labelpad=24)
    figureName = "/home/francesco/Pictures/soft/mips/gammaVStemp-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotGammaVSActivity(fileName, figureName):
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    data = np.loadtxt(fileName)
    ax.semilogx(data[:,0], data[:,1]/0.3, color='k', marker='o', markersize=8, lw=1, fillstyle='none')
    ax.set_ylim(-0.02,4.92)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    ax.set_ylabel("$\\frac{\\gamma \\sigma}{k_B T}$", rotation='horizontal', fontsize=24, labelpad=24)
    figureName = "/home/francesco/Pictures/soft/mips/gammaVSactivity-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPWallForceVSTemp(dirName, figureName, sample='passive', which='average'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(sample=='passive'):
        tempList = np.array(['0.30', '0.35', '0.40', '0.45'])
    elif(sample=='active'):
        tempList = np.array(['1e-03', '1e-02', '1e-01'])
        taup = tempList.astype(np.float64)
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', tempList.shape[0]+2)
    markerList = ['s', 'v', 'd', '*']
    temp = np.zeros((tempList.shape[0],2))
    gamma = np.zeros((tempList.shape[0],2))
    for t in range(tempList.shape[0]):
        if(sample=='passive'):
            dirPath = dirName + "/T" + tempList[t] + "/biaxial2e-03-tmax2e05/"
            labelName = "$T=$"
        elif(sample=='active'):
            dirPath = dirName + "T0.30-tp" + tempList[t] + "-f02/biaxial2e-03-tmax2e05/"
            labelName = "$\\tau_p=$"
        boxSize = np.loadtxt(dirPath + "/boxSize.dat")
        Ly0 = boxSize[1]
        dirList, strainList = utils.getOrderedStrainDirectories(dirPath)
        force = np.zeros(dirList.shape[0])
        error = np.zeros(dirList.shape[0])
        Ly = np.zeros(dirList.shape[0])
        for d in range(dirList.shape[0]):
            dirSample = dirPath + dirList[d]
            if(which=='average'):
                dirSample += "/dynamics/"
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "/energy.dat")
                force[d] = np.mean(data[:,4])
                error[d] = np.std(data[:,4])/np.sqrt(data[:,4].shape[0])
                boxSize = np.loadtxt(dirSample + "/boxSize.dat")
                #Ly[d] = (boxSize[1] - Ly0)/Ly0
                Ly[d] = boxSize[1]
        if(which=='average'):
            force = force[Ly>0]
            error = error[Ly>0]
            Ly = Ly[Ly>0]
        ax.errorbar(Ly, force, error, lw=1.2, marker=markerList[t], markersize=8, color=colorList(t/tempList.shape[0]), capsize=3, fillstyle='none', label=labelName + tempList[t])
        midForce = (force[1:] + force[:-1])*0.5
        height = (Ly[1:] + Ly[:-1])*0.5
        gamma[t,0] = np.sum(midForce*height)/(height[-1]-height[0])
        gamma[t,1] = np.mean(error*Ly)/(height[-1]-height[0])
        data = np.loadtxt(dirPath + "../energy.dat")
        temp[t,0] = np.mean(data[:,3])
        temp[t,1] = np.std(data[:,3])/np.sqrt(data[:,3].shape[0])
        print("Temperature:", temp[t,0], "line tension:", gamma[t,0])
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax.set_xlabel("$Strain,$ $\\Delta L_y / {L_y}_0$", fontsize=16)
    ax.set_ylabel("$Force$ $on$ $wall,$ $F_{wall}$", fontsize=16)
    figure1Name = "/home/francesco/Pictures/soft/mips/wallForceVSTemp-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    xlabel = "$Temperature,$ $T/ \\varepsilon$"
    #if(which=='passive'):
    #    x = temp[:,0]
    #    xlabel = "$Temperature,$ $T/ \\varepsilon$"
    #elif(which=='active'):
    #    x = taup
    #    xlabel = "$Persistence$ $time,$ $\\tau_p / \\tau_i$"
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Line$ $tension,$ $\\gamma = W / \\Delta L_y$", fontsize=16)
    ax.errorbar(temp[:,0], gamma[:,0], gamma[:,1], temp[:,1], lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', capsize=3)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/wallWorkVSTemp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPProfileVSSystemSize(dirName, figureName):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    dirList = np.array(['8192', '9192', '10192'])#, '11192', '12192'])#, '13192'])
    phiList = np.array(['0.30', '0.34-strain0.136', '0.36-strain0.246'])#, '0.38', '0.40'])#, '0.42'])
    colorList = cm.get_cmap('viridis', dirList.shape[0]+4)
    for i in range(dirList.shape[0]):
        dirSample = dirName + dirList[i] + "-2d/box/" + phiList[i]
        labelName = "$N=$"
        if not(os.path.exists(dirSample + "/singleProfile.dat")):
            interface.computeLinearDensityProfile(dirSample)
            print('computing')
        data = np.loadtxt(dirSample + "/singleProfile.dat")
        ax.plot(data[:,0], data[:,1], lw=1, marker='o', markersize=6, color=colorList(i/dirList.shape[0]), fillstyle='none', label=labelName + dirList[i])
        #center = np.mean(data[:,0])
        #x = np.abs(data[:,0]-center)
        #y = data[np.argsort(x),1]
        #yerr = data[np.argsort(x),2]
        #x = np.sort(x)
        #ax.errorbar(x, y, yerr, lw=1, marker='o', markersize=6, color=colorList(i/dirList.shape[0]), fillstyle='none', label=labelName + dirList[i])
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$\\varphi(x)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/profileVSSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPEvaporationRatePDF(dirName, figureName, which='passive', numSamples=30, numBins=20):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='passive'):
        dirList = np.array(['0.30', '0.32', '0.34', '0.36', '0.38', '0.40', '0.42'])
    elif(which=='active'):
        dirList = np.array(['1e-04', '1e-03', '1e-02', '1e-01'])
    elif(which=='compare'):
        dirList = np.array(['0.30', '1e-03'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirType = "/langevin-lj/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirType = "/langevin-lj/T0.30/active-lj/T0.30-tp" + dirList[d] + "-f01/dynamics/"
            labelName = "$\\tau_p=$"
        elif(which=='compare'):
            if(d==0):
                dirType = "/langevin-lj/T" + dirList[d] + "/dynamics/"
                labelName = "$passive,$ $T=$"
            else:
                dirType = "/langevin-lj/T0.30/active-lj/T0.30-tp" + dirList[d] + "-f01/dynamics/"
                labelName = "$active,$ $T=0.30$ $f_0=1$ $\\tau_p=$"
        rate = np.empty(0)
        for i in range(numSamples):
            dirSample = dirName + "-" + str(i) + dirType
            if(os.path.exists(dirSample)):
                rate = np.append(rate, np.loadtxt(dirSample + "/evaporationRate.dat"))
        if(rate.shape[0] != 0):
            pdf, edges = np.histogram(rate, bins=np.linspace(np.min(rate), np.max(rate), numBins), density=True)
            edges = (edges[1:] + edges[:-1])/2
            ax.plot(edges[pdf>0], pdf[pdf>0], marker='o', markersize=8, lw=1, fillstyle='none', markeredgewidth=1.5, color=colorList(d/dirList.shape[0]), label=labelName + dirList[d])
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlabel("$Evaporation$ $rate,$ $R_e = N_e / \\tau_{ISF}$", fontsize=16)
    ax.set_ylabel("$PDF(R_e)$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/evaporatePDF-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterExchangeVSTemp(dirName, figureName, which='lj', Ly='1', numBins=20):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        dirList = np.array(['0.35', '0.36', '0.37', '0.38', '0.39', '0.40', '0.41'])
    elif(which=='active'):
        dirList = np.array(['1e-02', '9e-03', '8e-03', '7e-03', '6e-03', '5.5e-03', '5e-03', '4.5e-03', '4e-03', '3.5e-03', '3.3e-03', '3e-03',
                            '2.8e-03', '2.6e-03', '2.4e-03', '2.2e-03', '2e-03'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    temp = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    condense = np.zeros((dirList.shape[0],2))
    evaporate = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(which=='lj'):
            dirSample = dirName + "box6-" + Ly + "/0.238/langevin-lj/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "box/0.30/active-lj/T0.35-Dr" + dirList[d] + "-f08/dynamics/"
            labelName = "$D_r=$"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "condensationTime.dat")):
            interface.computeExchangeTimes(dirSample)
        data = np.loadtxt(dirSample + "condensationTime.dat")
        condense[d,0] = np.mean(data)
        condense[d,1] = np.std(data)
        pdf, edges = np.histogram(data, bins=np.geomspace(np.min(data), np.max(data), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        ax.loglog(edges[pdf>0], pdf[pdf>0], marker='$C$', markersize=6, lw=0.8, fillstyle='none', color=colorList(d/dirList.shape[0]), label=labelName + dirList[d])
        data = np.loadtxt(dirSample + "evaporationTime.dat")
        evaporate[d,0] = np.mean(data)
        evaporate[d,1] = np.std(data)
        pdf, edges = np.histogram(data, bins=np.geomspace(np.min(data), np.max(data), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        ax.loglog(edges[pdf>0], pdf[pdf>0], marker='$E$', markersize=6, lw=0.8, fillstyle='full', color=colorList(d/dirList.shape[0]))
        if not(os.path.exists(dirSample + "clusterTemperature.dat")):
            interface.computeClusterTemperatureVSTime(dirSample)
        data = np.loadtxt(dirSample + "clusterTemperature.dat")
        temp[d] = np.mean(data[:,1])
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    #ax.legend(loc='best', fontsize=10, ncol=2)
    ax.set_xlabel("$time$", fontsize=16)
    ax.set_ylabel("$Distribution$", fontsize=16)
    figure1Name = "/home/francesco/Pictures/soft/mips/exchangeVSTemp-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        x = temp
        xlabel = "$Temperature,$ $T$"
    elif(which=='active'):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Average$ $time$", fontsize=16)
    ax.errorbar(x, condense[:,0], condense[:,1], lw=1.2, marker='$C$', markersize=15, color='k', fillstyle='none', capsize=3, label="$Condensation$")
    ax.errorbar(x, evaporate[:,1], evaporate[:,1], lw=1.2, marker='$E$', markersize=15, color='k', fillstyle='none', capsize=3, label="$Evaporation$")
    ax.legend(loc='best', fontsize=12, ncol=2)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/exchangeTimeVSTemp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPairCorrelation(dirName, figureName, which='passive'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='passive'):
        dirList = np.array(['0.20', '0.24', '0.28', '0.32', '0.36', '0.40', '0.44'])
    elif(which=='active'):
        dirList = np.array(['1', '2', '4', '8'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirSample = dirName + "/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "/T0.30-tp1e-03-f0" + dirList[d] + "/dynamics/"
            labelName = "$f_0=$"
        if not(os.path.exists(dirSample + "densePairCorr.dat")):
            cluster.averageDensePairCorr(dirSample, 0.3, 1)
        data = np.loadtxt(dirSample + "densePairCorr.dat")
        ax.errorbar(data[:,0], data[:,1], data[:,2], marker='o', markersize=6, lw=1, fillstyle='none', color=colorList(d/dirList.shape[0]), label=labelName + dirList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best', ncol=2)
    ax.set_xlabel("$Interparticle$ $distance,$ $r/\\sigma$", fontsize=16)
    ax.set_ylabel("$g(r/\\sigma)$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/pairCorr-" + figureName + "-" + which
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPTemperatureLV(dirName, figureName, which='passive', numBins=50):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='passive'):
        dirList = np.array(['0.20', '0.24', '0.28', '0.32', '0.36', '0.40', '0.44'])
    elif(which=='active'):
        dirList = np.array(['1', '2', '4', '8'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    temp = np.zeros((dirList.shape[0], 2))
    templiquid = np.zeros((dirList.shape[0], 2))
    tempvapor = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirSample = dirName + "/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "/T0.30-tp1e-03-f0" + dirList[d] + "/dynamics/"
            labelName = "$f_0=$"
        if not(os.path.exists(dirSample + "clusterTemperature.dat")):
            interface.computeClusterTemperatureVSTime(dirSample)
        data = np.loadtxt(dirSample + "clusterTemperature.dat")
        temp[d,0] = np.mean(data[:,0])
        temp[d,1] = np.std(data[:,0])
        templiquid[d,0] = np.mean(data[:,1])
        templiquid[d,1] = np.std(data[:,1])
        tempvapor[d,0] = np.mean(data[:,2])
        tempvapor[d,1] = np.std(data[:,2])
    ax.errorbar(temp[:,0], templiquid[:,0], templiquid[:,1], marker='o', markersize=10, lw=1, fillstyle='none', color='b', capsize=3, label="Liquid")
    ax.errorbar(temp[:,0], tempvapor[:,0], tempvapor[:,1], marker='^', markersize=10, lw=1, fillstyle='none', color='g', capsize=3, label="Vapor")
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Temperature,$ $T$", fontsize=16)
    ax.set_ylabel("$T_{Liquid},$ $T_{Vapor}$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/tempLV-" + figureName + "-" + which
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

    if(whichPlot == "heightflu"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        qmax = float(sys.argv[5])
        plotSPClusterHeightFluctuations(dirName, figureName, which, qmax)

    elif(whichPlot == "heighttemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        qmax = float(sys.argv[5])
        plotSPClusterHeightVSTemp(dirName, figureName, which, qmax)

    elif(whichPlot == "heightcorr"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterHeightCorrelation(dirName, figureName, which)

    elif(whichPlot == "profile"):
        figureName = sys.argv[3]
        plotSPDensityProfile(dirName, figureName)

    elif(whichPlot == "width"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        param = sys.argv[5]
        plotSPClusterWidth(dirName, figureName, which, param)

    elif(whichPlot == "widthtemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterWidthVSTemp(dirName, figureName, which)

    elif(whichPlot == "energytemp"):
        figureName = sys.argv[3]
        plotSPActiveEnergyVSTemp(dirName, figureName)

    elif(whichPlot == "isf"):
        figureName = sys.argv[3]
        plotSPClusterISF(dirName, figureName)

    elif(whichPlot == "walltime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        index = int(sys.argv[5])
        numSamples = int(sys.argv[6])
        plotSPWallForceVSTime(dirName, figureName, which, index, numSamples)

    elif(whichPlot == "wallstrain"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        index = int(sys.argv[5])
        limit = float(sys.argv[6])
        plotSPWallForceVSStrain(dirName, figureName, which, index, limit)

    elif(whichPlot == "widthstrain"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPWidthVSStrain(dirName, figureName, which)

    elif(whichPlot == "forcepdf"):
        figureName = sys.argv[3]
        index = int(sys.argv[4])
        numBins = int(sys.argv[5])
        plotForcePDF(dirName, figureName, index, numBins)

    elif(whichPlot == "pdfsize"):
        figureName = sys.argv[3]
        type = sys.argv[4]
        index = int(sys.argv[5])
        numBins = int(sys.argv[6])
        plotForcePDFVSSystemSize(dirName, figureName, type, index, numBins)

    elif(whichPlot == "pdfstrain"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        index = int(sys.argv[5])
        numBins = int(sys.argv[6])
        plotForcePDFVSStrain(dirName, figureName, which, index, numBins)

    elif(whichPlot == "sampletime"):
        figureName = sys.argv[3]
        type = sys.argv[4]
        which = sys.argv[5]
        index = int(sys.argv[6])
        numSamples = int(sys.argv[7])
        plotSampleWallForceVSTime(dirName, figureName, type, which, index, numSamples)

    elif(whichPlot == "samplestrain"):
        figureName = sys.argv[3]
        type = sys.argv[4]
        which = sys.argv[5]
        index = int(sys.argv[6])
        numSamples = int(sys.argv[7])
        limit = float(sys.argv[8])
        maxStrain = float(sys.argv[9])
        temp = sys.argv[10]
        taupf0 = sys.argv[11]
        plotSampleWallForceVSStrain(dirName, figureName, type, which, index, numSamples, limit, maxStrain, temp, taupf0)

    elif(whichPlot == "gammatemp"):
        figureName = sys.argv[3]
        plotGammaVSTemperature(dirName, figureName)

    elif(whichPlot == "gammatau"):
        figureName = sys.argv[3]
        plotGammaVSActivity(dirName, figureName)

    elif(whichPlot == "walltemp"):
        figureName = sys.argv[3]
        sample = sys.argv[4]
        which = sys.argv[5]
        plotSPWallForceVSTemp(dirName, figureName, sample, which)

    elif(whichPlot == "profilesize"):
        figureName = sys.argv[3]
        plotSPProfileVSSystemSize(dirName, figureName)

    elif(whichPlot == "rate"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        numSamples = int(sys.argv[5])
        numBins = int(sys.argv[6])
        plotSPEvaporationRatePDF(dirName, figureName, which, numSamples, numBins)

    elif(whichPlot == "exchange"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        Ly = sys.argv[5]
        numBins = int(sys.argv[6])
        plotSPClusterExchangeVSTemp(dirName, figureName, which, Ly, numBins)

    elif(whichPlot == "pcdense"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPPairCorrelation(dirName, figureName, which)

    elif(whichPlot == "templv"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPTemperatureLV(dirName, figureName, which)