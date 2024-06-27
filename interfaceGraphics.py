'''
Created by Francesco
14 July 2023
'''
#functions for clustering visualization
import numpy as np
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
import warnings
import utils
import spCorrelation as corr
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
def plotInterfaceFluctuations(dirName, figureName, which='active', qmax=1):
    sigma = utils.readFromDynParams(dirName, 'sigma')
    fig, ax = plt.subplots(2, 1, figsize=(7,7), dpi = 120)
    if not(os.path.exists(dirName + os.sep + "heightFluctuations.dat")):
        if(which=='lj'):
            interface.averageInterfaceFluctuations(dirName, 0.3)
        elif(which=='active'):
            interface.averageInterfaceFluctuations(dirName, 0.78)
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

def plotInterfaceVSTemp(dirName, figureName, which='active', qmax=0.4):
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
                interface.averageInterfaceFluctuations(dirSample, 0.3)
            elif(which=='active'):
                interface.averageInterfaceFluctuations(dirSample, 0.78)
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

def plotInterfaceCorrelation(dirName, figureName, which='Dr'):
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

def plot2InterfaceFluctuations(dirName, figureName, num1=0, limit=0.7):
    fig, ax = plt.subplots(2, 1, figsize=(7,8), dpi = 120)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    temp = utils.readFromParams(dirName, "temperature")
    print(temp)
    if not(os.path.exists(dirName + os.sep + "heightFluctuations.dat")):
        interface.average2InterfaceFluctuations(dirName, num1)
    data = np.loadtxt(dirName + os.sep + "heightFluctuations.dat")
    print(data.shape)
    ax[0].errorbar(data[:,0], data[:,1], data[:,2], lw=1, marker='o', markersize=6, color='g', fillstyle='none', capsize=3, elinewidth=0.8)
    ax[0].errorbar(data[:,0], data[:,3], data[:,4], lw=1, marker='s', markersize=6, color='b', fillstyle='none', capsize=3, elinewidth=0.8)
    ax[1].errorbar(data[2:,5], data[2:,6], data[2:,7], lw=1, marker='o', markersize=6, color='k', fillstyle='none', capsize=3, elinewidth=0.8)
    x = data[2:,5]
    y = data[2:,6]
    failed = False
    try:
        popt, pcov = curve_fit(powerLaw, x[x<limit], y[x<limit], bounds=([-np.inf, -2.1], [np.inf, -1.9]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        label = "$Fit,$ $a \, q^b:$ $a=$" + str(np.around(popt[0],3)) + ", $b=$" + str(np.around(popt[1],3)) + "$\, \\pm \,$" + str(np.around(np.sqrt(np.diag(pcov))[1],3))
        ax[1].plot(x[x<limit], powerLaw(x[x<limit], *popt), color='b', lw=3, ls='dashed', alpha=0.5, label=label)
        print("fitting parameters - amplitude:", popt[0], "slope:", popt[1], "+-", np.sqrt(np.diag(pcov))[1])
        print("line tension:", 2*temp/(popt[0]), "+-", 2*temp/(np.sqrt(np.diag(pcov))[0]))
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].legend(fontsize=12, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$q$", fontsize=16)
    ax[1].set_ylabel("$\\langle |\delta h(q)|^2 \\rangle \\sigma$", fontsize=16)
    ax[0].set_xlabel("$y$", fontsize=16)
    ax[0].set_ylabel("$\\langle |\delta h(y)|^2 \\rangle$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/interfaceFlu-" + figureName + ".png"
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPDensityProfile(dirName, figureName):
    #print("taup:", 1/(utils.readFromDynParams(dirName, "Dr")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if not(os.path.exists(dirName + "densityProfile.dat")):
        interface.averageLinearDensityProfile(dirName)
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
                interface.averageInterfaceFluctuations(dirSample, 0.3)
            elif(which=='active'):
                interface.averageInterfaceFluctuations(dirSample, 0.78)
        energyLength = np.loadtxt(dirSample + "energyLength.dat")
        energyLength[:,1] /= sigma
        length[d,0] = np.mean(energyLength[:,1])
        length[d,1] = np.std(energyLength[:,1])
        if not(os.path.exists(dirSample + "densityProfile.dat")):
            if(which=='lj'):
                interface.averageLinearDensityProfile(dirSample, 0.3)
            elif(which=='active'):
                interface.averageLinearDensityProfile(dirSample, 0.78)
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

def plotSPPairCorrelation(dirName, figureName, which='temp'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='temp'):
        dirList = np.array(['0.80', '0.90', '1.00', '1.10', '1.20', '1.30', '1.40', '1.50', '1.60', '1.70', '1.80', '1.90', '2.00', '2.10', '2.00'])
    elif(which=='active'):
        dirList = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    peak = np.zeros((dirList.shape[0],2))
    temp = np.zeros((dirList.shape[0],2))
    if(which=='active'):
        dirSample = dirName + "../../../nh2/T0.80/dynamics-nve/"
        if(os.path.exists(dirSample)):
            if not(os.path.exists(dirSample + "pairCorr.dat")):
                corr.averagePairCorr(dirSample, "time", 5e04)
            data0 = np.loadtxt(dirSample + "pairCorr.dat")
            ax.errorbar(data0[:,0], data0[:,1], data0[:,2], marker='o', markersize=6, lw=1, fillstyle='none', color='k', capsize=3, label="$T=0.80$")
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirSample = dirName + "/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "/tp1e-01-f0" + dirList[d] + "/dynamics/"
            labelName = "$f_0=$"
        if(os.path.exists(dirSample)):
            if not(os.path.exists(dirSample + "pairCorr.dat")):
                corr.averagePairCorr(dirSample, "time", 5e04)
            energy = np.loadtxt(dirSample + "energy.dat")
            temp[d,0] = np.mean(energy[:,3])
            temp[d,1] = np.std(energy[:,3])
            data = np.loadtxt(dirSample + "pairCorr.dat")
            peak[d,0] = data[np.argmax(data[:,1]),0]
            peak[d,1] = data[np.argmax(data[:,1]),2]
            ax.errorbar(data[:,0], data[:,1], data[:,2], marker='o', markersize=6, lw=1, fillstyle='none', color=colorList(d/dirList.shape[0]), capsize=3, label=labelName + dirList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best', ncol=2)
    #ax.set_xlim(0.45,2.55)
    ax.set_xlabel("$Interparticle$ $distance,$ $r/\\sigma$", fontsize=16)
    ax.set_ylabel("$g(r/\\sigma)$", fontsize=16)
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/pCorr-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    temp = temp[peak[:,0]!=0]
    #peak[:,0] /= 2**(1/6)
    ax.errorbar(temp[:,0], peak[:,0], peak[:,1], color='k', markersize=10, fillstyle='none', marker='^', lw=1, capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=16)
    ax.set_ylabel("$r_{max}(T)$", fontsize=24, rotation='horizontal', labelpad=40)
    #ax.set_ylabel("$\\frac{r_{max}(\\tau_p)}{2^{1/6} \\sigma}$", fontsize=24, rotation='horizontal', labelpad=40)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/peakVS" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotProfileVSTemp(dirName, figureName, which='passive'):
    fig, ax = plt.subplots(figsize=(7,5.5), dpi = 120)
    if(which=='passive'):
        dirList = np.array(['0.42', '0.43', '0.44', '0.45', '0.46', '0.47'])
        temp = dirList.astype(np.float64)
    elif(which=='active'):
        dirList = np.array(['1e-04', '1e-03', '1e-02', '1e-01', '1', '1e01', '1e02', '1e03'])
        taup = dirList.astype(np.float64)
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    width = np.zeros((dirList.shape[0],2))
    fluidWidth = np.zeros((dirList.shape[0],2))
    density = np.zeros((dirList.shape[0],2))
    if(which=='active'):
        dirSample = dirName + "../dynamics-test/"
        if(os.path.exists(dirSample)):
            if not(os.path.exists(dirSample + "densityProfile.dat")):
                interface.averageLinearDensityProfile(dirSample, 0.3)
            data = np.loadtxt(dirSample + "densityProfile.dat")
            ax.errorbar(data[:,0], data[:,1], data[:,2], marker='s', markersize=6, lw=1, fillstyle='none', color='k', capsize=3, label="$T=0.42$")
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirSample = dirName + "/T" + dirList[d] + "/dynamics-test/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "/T0.42-tp" + dirList[d] + "-f04/dynamics-test/"
            labelName = "$\\tau_p=$"
        if(os.path.exists(dirSample)):
            print(dirSample)
            if not(os.path.exists(dirSample + "densityProfile.dat")):
                interface.averageLinearDensityProfile(dirSample, 0.3)
            data = np.loadtxt(dirSample + "densityProfile.dat")
            x = data[:,0]
            y = data[:,1]
            xFluid = x[np.argwhere(y>0.5)[:,0]]
            if(xFluid.shape[0] != 0):
                fluidWidth[d,0] = xFluid[-1] - xFluid[0]
                fluidWidth[d,1] = np.std(data[data[:,1]>0.5,1])
            density[d,0] = np.mean(data[data[:,1]>0.5,1])
            density[d,1] = np.std(data[data[:,1]>0.5,1])
            ax.errorbar(data[:,0], data[:,1], data[:,2], marker='o', markersize=6, lw=1, fillstyle='none', color=colorList(d/dirList.shape[0]), capsize=3, label=labelName + dirList[d])
            data = np.loadtxt(dirSample + "interfaceWidth.dat")
            width[d,0] = np.mean(data[:,1])
            width[d,1] = np.std(data[:,1])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$\\varphi(x)$", fontsize=16)
    plt.tight_layout()
    if(which=='passive'):
        figureName = "Temp-" + figureName
    elif(which=='active'):
        figureName = "Taup-" + figureName
    figure1Name = "/home/francesco/Pictures/soft/mips/profileVS" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    if(which=='passive'):
        x = temp 
        xlabel = "$Temperature,$ $T$"
    elif(which=='active'):
        x = taup
        ax.set_xscale('log')
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    ax.errorbar(x, width[:,0], width[:,1], color='k', markersize=10, fillstyle='none', marker='^', lw=1, capsize=3)
    #ax.errorbar(x, density[:,0], density[:,1], color='k', markersize=10, fillstyle='none', marker='^', lw=1, capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Interface$ $width,$ $w$", fontsize=16)
    #ax.set_ylabel("$Liquid$ $density,$ $\\varphi_{Liquid}$", fontsize=16)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/widthVS" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotWallForceVSTemperature(dirName, figureName, which='passive'):
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    if(which=='passive'):
        dirList = np.array(['0.42', '0.43', '0.44', '0.45', '0.46', '0.47'])
        temp = dirList.astype(np.float64)
    elif(which=='active'):
        dirList = np.array(['1e-04', '1e-03', '1e-02', '1e-01', '1', '1e01', '1e02', '1e03'])
        taup = dirList.astype(np.float64)
    else:
        print("Please specify the sample type")
    force = np.zeros((dirList.shape[0],2))
    force1 = np.zeros((dirList.shape[0],2))
    force2 = np.zeros((dirList.shape[0],2))
    force3 = np.zeros((dirList.shape[0],2))
    force4 = np.zeros((dirList.shape[0],2))
    force5 = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirSample = dirName + "/T" + dirList[d] + "/dynamics-test/"
        elif(which=='active'):
            dirSample = dirName + "/T0.42-tp" + dirList[d] + "-f04/dynamics-test/"
        if(os.path.exists(dirSample)):
            width = 0
            if(os.path.exists(dirSample + "/densityProfile.dat")):
                data = np.loadtxt(dirSample + "/densityProfile.dat")
                x = data[:,0]
                y = data[:,1]
                xFluid = x[np.argwhere(y>0.5)[:,0]]
                if(xFluid.shape[0] != 0):
                    width = xFluid[-1] - xFluid[0]
            else:
                width, _, _ = interface.averageLinearDensityProfile(dirSample)
            #print(dirSample)
            if(width!=0):
                data = np.loadtxt(dirSample + "/energy.dat")
                force[d,0] = np.mean(data[:,4])/(2*width)
                force[d,1] = np.std(data[:,4])/(2*width)
                data = np.loadtxt(dirSample + "/wallForce-size0.0.dat")
                force1[d,0] = np.mean(data[:,2])/(2*width)
                force1[d,1] = np.std(data[:,2])/(2*width)
                data = np.loadtxt(dirSample + "/wallForce-size0.25.dat")
                force2[d,0] = np.mean(data[:,2])/(2*width)
                force2[d,1] = np.std(data[:,2])/(2*width)
                data = np.loadtxt(dirSample + "/wallForce-size0.5.dat")
                force3[d,0] = np.mean(data[:,2])/(2*width)
                force3[d,1] = np.std(data[:,2])/(2*width)
                data = np.loadtxt(dirSample + "/wallForce-size0.75.dat")
                force4[d,0] = np.mean(data[:,2])/(2*width)
                force4[d,1] = np.std(data[:,2])/(2*width)
                data = np.loadtxt(dirSample + "/wallForce-size1.0.dat")
                force5[d,0] = np.mean(data[:,2])/(2*width)
                force5[d,1] = np.std(data[:,2])/(2*width)
    if(which=='passive'):
        x = temp 
        xlabel = "$Temperature,$ $T$"
        figureName = "Temp-" + figureName
    elif(which=='active'):
        x = taup
        ax.set_xscale('log')
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "Taup-" + figureName
    #ax.errorbar(x, force[:,0], force[:,1], marker='o', markersize=8, fillstyle='none', color='k', label="$Simulation$")
    ax.errorbar(x, force1[:,0], force1[:,1], marker='o', lw=1, markersize=8, fillstyle='none', color='g', capsize=3, label="$d_{wall} = 0$")
    ax.errorbar(x, force2[:,0], force2[:,1], marker='s', lw=1, markersize=8, fillstyle='none', color='b', capsize=3, label="$d_{wall} = 0.25 \\langle \\sigma_i \\rangle$")
    ax.errorbar(x, force3[:,0], force3[:,1], marker='v', lw=1, markersize=8, fillstyle='none', color=[1,0.5,0], capsize=3, label="$d_{wall} = 0.5 \\langle \\sigma_i \\rangle$")
    ax.errorbar(x, force4[:,0], force4[:,1], marker='^', lw=1, markersize=8, fillstyle='none', color='r', capsize=3, label="$d_{wall} = 0.75 \\langle \\sigma_i \\rangle$")
    ax.errorbar(x, force5[:,0], force5[:,1], marker='D', lw=1, markersize=8, fillstyle='none', color='k', capsize=3, label="$d_{wall} = \\langle \\sigma_i \\rangle$")
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    ax.set_ylabel("$\\frac{F_{wall} \\sigma}{w}$", fontsize=24, rotation="horizontal", labelpad=30)
    if(which=='passive'):
        figureName = "Temp-" + figureName
    elif(which=='active'):
        figureName = "Taup-" + figureName
    figureName = "/home/francesco/Pictures/soft/mips/wallForceVS-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotForcePDF(dirName, figureName, numBins=20):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(os.path.exists(dirName + "/singleProfile.dat")):
        data = np.loadtxt(dirName + "/singleProfile.dat")
        x = data[:,0]
        y = data[:,1]
        xFluid = x[np.argwhere(y>0.5)[:,0]]
        width = xFluid[-1] - xFluid[0]
    else:
        width, _, _ = interface.computeLinearDensityProfile(dirName)
    data = np.loadtxt(dirName + "/energy.dat")
    #data[:,4] /= width
    pdf, edges = np.histogram(data[:,4], bins=np.linspace(np.min(data[:,4]), np.max(data[:,4]), numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax.plot(edges[pdf>0], pdf[pdf>0], marker='o', markersize=8, lw=1, fillstyle='none', markeredgewidth=1, color='k')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$F_{wall}$", fontsize=16)
    ax.set_ylabel("$PDF(F_{wall})$", fontsize=16)
    print("Average force:", np.mean(data[:,4]), np.std(data[:,4]))
    figureName = "/home/francesco/Pictures/soft/mips/forcePDF-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotForcePDFVSTemperature(dirName, figureName, which='passive', numBins=20):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(which=='passive'):
        dirList = np.array(['0.42', '0.43', '0.44', '0.45', '0.46', '0.47'])
    elif(which=='active'):
        dirList = np.array(['1e-04', '1e-03', '1e-02', '1e-01', '1', '1e01', '1e02', '1e03'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    numParticles = np.zeros(dirList.shape[0])
    mean = np.zeros(dirList.shape[0])
    error = np.zeros(dirList.shape[0])
    labelName = "$T=$"
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirSample = dirName + "/T" + dirList[d] + "/dynamics-test/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "/T0.42-tp" + dirList[d] + "-f04/dynamics-test/"
            labelName = "$f_0=$"
        if(os.path.exists(dirSample)):
            width = 0
            if(os.path.exists(dirSample + "/densityProfile.dat")):
                data = np.loadtxt(dirSample + "/densityProfile.dat")
                x = data[:,0]
                y = data[:,1]
                xFluid = x[np.argwhere(y>0.5)[:,0]]
                if(xFluid.shape[0] != 0):
                    width = xFluid[-1] - xFluid[0]
            else:
                width, _, _ = interface.averageLinearDensityProfile(dirSample)
            #print(dirSample)
            numParticles[d] = utils.readFromParams(dirSample, "numParticles")
            data = np.loadtxt(dirSample + "/energy.dat")
            if(width!=0):
                #data = data[:,4]/(2*width)
                data = data[:,5]
                mean[d] = np.mean(data)
                error[d] = np.std(data)
                print("sample:", dirList[d], "width:", width, "line tension:", np.mean(data), np.std(data))
                pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), numBins), density=True)
                edges = (edges[1:] + edges[:-1])/2
                ax.plot(edges[pdf>0], pdf[pdf>0], marker='o', markersize=8, lw=1, fillstyle='none', markeredgewidth=1, color=colorList(d/dirList.shape[0]), label=labelName + dirList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlabel("$F_{wall} \\sigma / w$", fontsize=16)
    ax.set_ylabel("$PDF(F_{wall} \\sigma / w)$", fontsize=16)
    if(which=='passive'):
        figureName = "Temp-" + figureName
    elif(which=='active'):
        figureName = "Taup-" + figureName
    figureName = "/home/francesco/Pictures/soft/mips/forcePDFVS-" + figureName
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

def plot2FluidsISF(dirName, figureName, T1='0.80', T2='0.40', decade=7, which='short'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='short'):
        index1 = 2
        index2 = 7
    else:
        index1 = 3
        index2 = 8
    if not(os.path.exists(dirName + "/2logCorr.dat")):
        interface.compute2FluidsCorr(dirName, 0, decade, decade-1)
    data = np.loadtxt(dirName + "/2logCorr.dat")
    ax.semilogx(data[:,0], data[:,index1], color='g', marker='o', markersize=8, fillstyle='none', lw=1, label="$T_A=$"+T1)
    ax.semilogx(data[:,0], data[:,index2], color='b', marker='v', markersize=8, fillstyle='none', lw=1, label="$T_B=$"+T2)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t / \\sqrt{m/\\varepsilon} \\sigma$", fontsize=16)
    ax.set_ylabel("$ISF$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/2fluidsISF-" + figureName + "-T1-" + T1 + "-T2-" + T2
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compare2FluidsISF(dirName, figureName, which='short'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    dirList = np.array(['0.40', '0.50', '0.60', '0.70', '0.80', '0.90'])
    temp = dirList.astype(float)
    colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
    markerList = ['o', 's', 'v', '^', 'P', 'd']
    tau1 = np.zeros(dirList.shape[0])
    tau2 = np.zeros(dirList.shape[0])
    if(which=='short'):
        index1 = 2
        index2 = 7
    else:
        index1 = 3
        index2 = 8
    for d in range(dirList.shape[0]):
        dirSample = dirName + "T" + dirList[d] + "/nve/dynamics-log/"
        if not(os.path.exists(dirSample + "/2logCorr.dat")):
            interface.compute2FluidsCorr(dirSample, 0, 7, 6)
        data = np.loadtxt(dirSample + "/2logCorr.dat")
        tauData = np.column_stack((data[:,0], data[:,index1-1], data[:,index1]))
        tau1[d] = utils.computeTau(tauData)
        ax.semilogx(data[:,0], data[:,index1], color=colorList(d/dirList.shape[0]), marker=markerList[d], markersize=8, fillstyle='none', lw=1, label="$T_A=$" + dirList[d])
    for d in range(dirList.shape[0]):
        dirSample = dirName + "T" + dirList[d] + "/nve/dynamics-log/"
        data = np.loadtxt(dirSample + "/2logCorr.dat")
        tauData = np.column_stack((data[:,0], data[:,index2-1], data[:,index2]))
        tau2[d] = utils.computeTau(tauData)
        ax.semilogx(data[:,0], data[:,index2], color=colorList(d/dirList.shape[0]), lw=5, alpha=0.5, ls='dashed', label="$T_B=$" + dirList[d])
    ax.legend(fontsize=11, loc='best', ncol=2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t \\; / \\sqrt{\\frac{m}{\\varepsilon}} \\sigma$", fontsize=16)
    ax.set_ylabel("$ISF$", fontsize=16)
    figure1Name = "/home/francesco/Pictures/soft/mips/compareISF-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.plot(temp, tau1, color='g', marker='o', markersize=8, fillstyle='none', lw=1, label="$Fluid$ $A$")
    ax.plot(temp, tau2, color='b', marker='v', markersize=8, fillstyle='none', lw=1, label="$Fluid$ $B$")
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature,$ $T/\\varepsilon$", fontsize=16)
    ax.set_ylabel("$Relaxation$ $time,$ $\\tau \; / \\sqrt{\\frac{m}{\\varepsilon}} \\sigma$", fontsize=16)
    figure2Name = "/home/francesco/Pictures/soft/mips/compareTau-" + figureName
    plt.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPEnergyVSStrain(dirName, figureName, which='total', simple=False, limit=1.01, plot=True):
    # read initial energies
    energy = np.loadtxt(dirName + "../energy.dat")
    etot0 = np.mean(energy[:,4])
    dirList, strain = utils.getOrderedStrainDirectories(dirName)
    mean = np.zeros((dirList.shape[0],5))
    error = np.zeros((dirList.shape[0],5))
    length = np.zeros(dirList.shape[0])
    if(which=='time'):
        etot = []
        epot = []
        ekin = []
        time = []
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d]
        if(simple=='simple'):
            dirSample += "/simple/"
        if(os.path.exists(dirSample)):
            length[d] = np.loadtxt(dirSample + "/boxSize.dat")[1]
            width = np.loadtxt(dirSample + "/boxSize.dat")[0]
            data = np.loadtxt(dirSample + "/energy.dat")
            if(which=='time' and d!=(dirList.shape[0]-1)):
                etot.append(data[:,4])
                epot.append(data[:,2])
                ekin.append(data[:,3])
                time.append(data[:,1]+np.ones(data[:,1].shape[0])*2*d)
            data[:,4] -= etot0 
            mean[d,0] = np.mean(data[:,4])
            error[d,0] = np.std(data[:,4])
            mean[d,1] = np.mean(data[:,2])
            error[d,1] = np.std(data[:,2])
            mean[d,2] = np.mean(data[:,3])
            error[d,2] = np.std(data[:,3])
            if(which == 'force'):
                mean[d,3] = np.mean(data[:,5])
                error[d,3] = np.std(data[:,5])
    if(which=='time'):
        etot = np.array(etot).flatten()
        epot = np.array(epot).flatten()
        ekin = np.array(ekin).flatten()
        time = np.array(time).flatten()
    mean = mean[length!=0]
    error = error[length!=0]
    strain = strain[length!=0]
    length = length[length!=0]
    length /= length[0]
    np.savetxt(dirName + "/energyStrain.dat", np.column_stack((strain, length, mean[:,0], error[:,0], mean[:,1], error[:,1], mean[:,2], error[:,2], mean[:,3], error[:,3])))
    if(plot == True):
        fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
        if(which=='total'):
            figureName = "etotgamma-" + figureName
            ax.errorbar(length, mean[:,0], error[:,0], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3, label="$Total$")
            if(limit != 0):
                failed = False
                try:
                    popt, pcov = curve_fit(lineFit, length[length<limit], mean[length<limit,0])
                except RuntimeError:
                    print("Error - curve_fit failed")
                    failed = True
                if(failed == False):
                    ax.plot(length, lineFit(length, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax + b$")
                    print("Energy: a, b:", popt, "line tension from fit:", popt[0])
        elif(which=='potential'):
            ax.errorbar(length, mean[:,1], error[:,1], color='g', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3, label="$Potential$")
            figureName = "epotgamma-" + figureName
        elif(which=='kinetic'):
            ax.errorbar(length, mean[:,2], error[:,2], color='r', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3, label="$Kinetic$")
            figureName = "ekingamma-" + figureName
        elif(which=='force'):
            ax.errorbar(length, mean[:,3], error[:,3], color='c', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3, label="$Force$")
            figureName = "forcegamma-" + figureName
        elif(which=='time'):
            ax.plot(time, etot, color='b', marker='o', fillstyle='none', lw=1, label='$Total$')
        else:
            figureName = "egamma-" + figureName
            ax.errorbar(length, mean[:,0], error[:,0], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3, label="$Total$")
            ax.errorbar(length, mean[:,1], error[:,1], color='g', marker='v', markersize=8, fillstyle='none', lw=1, capsize=3, label="$Potential$")
            ax.errorbar(length, mean[:,2], error[:,2], color='r', marker='s', markersize=8, fillstyle='none', lw=1, capsize=3, label="$Kinetic$")
        ax.legend(fontsize=12, loc='best')
        ax.tick_params(axis='both', labelsize=14)
        ax.locator_params(axis='x', nbins=5)
        if(which=='time'):
            ax.set_xlabel("$Time,$ $t$", fontsize=16)
        else:
            ax.set_xlabel("$Box$ $height,$ $L_y/L_y^0$", fontsize=16)
        #ax.set_xlabel("$Strain,$ $\\epsilon$", fontsize=16)
        if(which=='force'):
            ax.set_ylabel("$Force$ $across$ $wall,$ $F_{wall}$", fontsize=16)
        else:
            if(which=='total'):
                ax.set_ylabel("$\\frac{\\Delta E}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
            else:
                ax.set_ylabel("$\\frac{E}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        figureName = "/home/francesco/Pictures/soft/mips/" + figureName
        plt.tight_layout()
        fig.savefig(figureName + ".png", transparent=True, format = "png")
        plt.show()

def plotSPStrainForceVSTime(dirName, figureName, strainStep=5e-06, compext='ext', slope='slope', plot=False):
    # read initial energies
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    data = np.loadtxt(dirName + "/energy.dat")
    maxStrain = (data.shape[0] // 10) * strainStep
    print(maxStrain)
    strain = np.arange(strainStep, maxStrain + strainStep, strainStep)
    freq = 10
    every = freq * 40
    force = np.zeros((strain.shape[0],2))
    for i in range(0,strain.shape[0]-every,every):
        #print(strain[i])
        stepForce = []
        for j in range(every):
            #print(i+j, strain[i+j])
            stepForce.append(np.mean(data[(i+j)*freq:(i+j+1)*freq,5]))
        # two interfaces
        force[i,0] = np.mean(stepForce)
        force[i,1] = np.std(stepForce)
    strain = strain[force[:,0]!=0]
    force = force[force[:,0]!=0]
    if(compext == 'ext' or compext == 'ext-wall' or compext == 'ext-eq'):
        otherStrain = -strain / (1 + strain)
        width = (np.ones(strain.shape[0]) + otherStrain) * boxSize[0]
        height = (np.ones(strain.shape[0]) + strain) * boxSize[1]
    elif(compext == 'comp' or compext == 'comp-wall' or compext == 'comp-eq'):
        otherStrain = -strain / (1 + strain)
        width = (np.ones(strain.shape[0]) + strain) * boxSize[0]
        height = (np.ones(strain.shape[0]) + otherStrain) * boxSize[1]
    np.savetxt(dirName + "/forceTime.dat", np.column_stack((strain, force, height, width)))
    if(plot == True):
        fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
        height -= height[0]
        if(compext == 'ext' or compext == 'ext-wall' or compext == 'ext-eq'):
            mean = np.mean(force[height<10,0])
            error = np.std(force[height<10,0])
        elif(compext == 'comp' or compext == 'comp-wall' or compext == 'comp-eq'):
            mean = np.mean(force[height>-10,0])
            error = np.std(force[height>-10,0])
        if(slope == 'slope'):
            print("average force:", mean, "+-", error)
            ax.plot(np.linspace(height[0],height[-1],100), mean*np.ones(100), color='b', ls='--', lw=1)
            ax.errorbar(height, force[:,0], force[:,1], color='k', marker='o', markersize=4, fillstyle='none', lw=1, capsize=3)
            failed = False
            try:
                popt, pcov = curve_fit(lineFit, height, force[:,0])
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == False):
                ax.plot(height, lineFit(height, *popt), color='g', lw=3, linestyle='dashdot', label="$ax + b$", alpha=1)
                print("Energy: a, b:", popt, "slope:", popt[0])
            ax.set_ylabel("$\\frac{F_{wall}\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=25)
        else:
            force[:,0] /= width
            force[:,1] /= width
            mean = np.mean(force[:,0])
            error = np.std(force[:,0])
            print("average force:", mean, "+-", error)
            ax.plot(np.linspace(height[0],height[-1],100), mean*np.ones(100), color='b', ls='--', lw=1)
            ax.errorbar(height, force[:,0], force[:,1], color='k', marker='o', markersize=4, fillstyle='none', lw=1, capsize=3, label='$F_{wall}/L_x$')
            ax.set_ylabel("$\\frac{F_{wall} \\sigma^2}{L_x\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=25)
        ax.tick_params(axis='both', labelsize=14)
        ax.locator_params(axis='x', nbins=5)
        ax.set_xlabel("$L_y - L_y^0$", fontsize=16)
        #ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
        if(slope == 'slope'):
            figureName = "/home/francesco/Pictures/soft/mips/forceSlope-" + figureName
        else:
            figureName = "/home/francesco/Pictures/soft/mips/forceTime-" + figureName
        plt.tight_layout()
        fig.savefig(figureName + ".png", transparent=True, format = "png")
        plt.show()

def compareForceVSTimeStrain(dirName, figureName, compext='comp-wall', which='temp', method='slope', compare=False):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which == 'temp'):
        dirList = np.array(['0.60', '0.70', '0.80', '0.90', '1.00', '1.10', '1.20', '1.30', '1.40',
                            '1.50', '1.60', '1.70', '1.80', '1.90', '2.00'])
        strain = np.ones(dirList.shape[0])*5e-06
    elif(which == 'strain'):
        dirList = np.array(['2e-06', '3e-06', '5e-06', '1e-05', '2e-05', '3e-05'])
        strain = dirList.astype(np.float32)
    tension = np.zeros((dirList.shape[0],2))
    temp = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(which == 'temp'):
            dirSample = dirName + '/nh2/T' + dirList[d] + '/nve-biaxial-' + compext + '5e-06-tmax2e03/'
        elif(which == 'strain'):
            dirSample = dirName + '/nh2/T0.80/nve-biaxial-' + compext + dirList[d] + '-tmax1e03/'
        if(os.path.exists(dirSample)):
            boxSize = np.loadtxt(dirSample + '/boxSize.dat')
            numParticles = utils.readFromParams(dirSample, 'numParticles')
            if not(os.path.exists(dirSample + "/energyTime.dat")):
                plotSPStrainEnergyVSTime(dirSample, 'temp', 'total', strain[d], compext=compext)
            data = np.loadtxt(dirSample + "/energyTime.dat")
            temp[d] = np.mean(data[:,5])
            if not(os.path.exists(dirSample + "/forceTime.dat")):
                plotSPStrainForceVSTime(dirSample, figureName, strain[d], compext=compext, slope=False)
            data = np.loadtxt(dirSample + "/forceTime.dat")
            force = data[:,1]
            height = data[:,-2]
            height -= height[0]
            if(method=='slope'):
                if(compext == 'ext' or compext == 'ext-wall' or compext == 'ext-eq'):
                    mean = np.mean(force[height<10])
                    error = np.std(force[height<10])
                elif(compext == 'comp' or compext == 'comp-wall' or compext == 'comp-eq'):
                    mean = np.mean(force[height>-10])
                    error = np.std(force[height>-10])
                failed = False
                try:
                    popt, pcov = curve_fit(lineFit, height, force)
                except RuntimeError:
                    print("Error - curve_fit failed")
                    failed = True
                if(failed == False):
                    tension[d,0] = popt[0]
                    tension[d,1] = np.sqrt(np.diag(pcov))[0]
                    if(which == 'temp'):
                        print("average tension:", tension[d], "temp:", 2*temp[d], dirList[d])
                    elif(which == 'strain'):
                        print("average tension:", tension[d], "strain:", strain[d])
            else:
                force /= data[:,-1]
                tension[d,0] = np.mean(force)
                tension[d,1] = np.std(force)
    if(which == 'temp'):
        x = temp
        xlabel = "$Temperature,$ $T/\\varepsilon$"
        if(method == 'slope'):
            np.savetxt(dirName + "/nh2/slopeTension.dat", np.column_stack((temp, tension)))
        else:
            np.savetxt(dirName + "/nh2/forceTension.dat", np.column_stack((temp, tension)))
    elif(which == 'strain'):
        x = strain
        xlabel = "$Strain$ $step,$ $\\Delta \\gamma$"
    x = x[tension[:,0]!=0]
    tension = tension[tension[:,0]!=0]
    ax.errorbar(x, tension[:,0], tension[:,1], color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    if(compare == 'compare'):
        if(method == 'slope'):
            data = np.loadtxt(dirName + '../../lj/0.64/nh2/slopeTension.dat')
        else:
            data = np.loadtxt(dirName + '../../lj/0.64/nh2/forceTension.dat')
        data = data[data[:,1]!=0]
        ax.errorbar(data[:,0], data[:,1], data[:,2], color='b', marker='s', markersize=8, fillstyle='none', capsize=3)
        ax.legend(('$Diatomic$', '$Monoatomic$'),fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.locator_params(axis='x', nbins=5)
    ax.set_xlabel(xlabel, fontsize=16)
    if(method == 'slope'):
        ax.set_ylabel("$\\frac{d F_{wall}\\sigma^2}{d L_y\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=40)
    else:
        ax.set_ylabel("$\\frac{F_{wall}\\sigma^2}{L_x\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=30)
    if(which == 'temp'):
        figureName = 'vsTemp-' + figureName
    elif(which == 'strain'):
        ax.set_xscale('log')
        figureName = 'vsStrain-' + figureName
    if(method == 'slope'):
        figureName = "/home/francesco/Pictures/soft/mips/forceSlope-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/mips/force-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPStrainEnergyVSTime(dirName, figureName, which='total', strainStep=5e-06, compext='ext', plot=False):
    # read initial energies
    numParticles = utils.readFromParams(dirName, 'numParticles')
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    data = np.loadtxt(dirName + "/energy.dat")
    if(np.sum(np.isnan(data))!=0):
        print("There are NaNs in the file")
    maxStrain = (data.shape[0] // 10) * strainStep
    print("max strain:", maxStrain)
    strain = np.arange(strainStep, maxStrain + strainStep, strainStep)
    freq = 10
    every = freq*20
    etot = np.zeros((strain.shape[0],2))
    epot = np.zeros((strain.shape[0],2))
    ekin = np.zeros((strain.shape[0],2))
    for i in range(0,strain.shape[0]-every,every):
        #print(strain[i])
        stepEtot = []
        stepEpot = []
        stepEkin = []
        for j in range(every):
            #print(i+j, strain[i+j])
            stepEtot.append(np.mean(data[(i+j)*freq:(i+j+1)*freq,4]))
            stepEpot.append(np.mean(data[(i+j)*freq:(i+j+1)*freq,2]))
            stepEkin.append(np.mean(data[(i+j)*freq:(i+j+1)*freq,3]))
        # two interfaces
        etot[i,0] = np.mean(stepEtot)/4
        etot[i,1] = np.std(stepEtot)/4
        epot[i,0] = np.mean(stepEpot)/2
        epot[i,1] = np.std(stepEpot)/2
        ekin[i,0] = np.mean(stepEkin)/2
        ekin[i,1] = np.std(stepEkin)/2
    strain = strain[etot[:,0]!=0]
    epot = epot[etot[:,0]!=0]
    ekin = ekin[etot[:,0]!=0]
    etot = etot[etot[:,0]!=0]
    otherStrain = -strain/(1 + strain)
    if(compext == 'ext' or compext == 'ext-wall' or compext == 'ext-eq'):
        height = (1 + strain)*boxSize[1]
        width = (1 + otherStrain)*boxSize[0]
    elif(compext == 'comp' or compext == 'comp-wall' or compext == 'comp-eq'):
        height = (1 + otherStrain)*boxSize[1]
        width = (1 + strain)*boxSize[0]
    np.savetxt(dirName + "/energyTime.dat", np.column_stack((strain, etot, epot, ekin, height, width)))
    if(plot == True):
        fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
        if(which=='total'):
            height -= boxSize[1]
            etot *= numParticles
            mean = np.mean(etot[strain<0.01,0])
            failed = False
            try:
                popt, pcov = curve_fit(lineFit, height[strain<0.4], etot[strain<0.4,0])
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == False):
                noise = np.sqrt(np.mean((lineFit(height, *popt) - etot[:,0])**2))
                offset = mean# + noise
                print("offset:", offset, "curve noise:", noise)
                ax.plot(height, lineFit(height, *popt)-offset, color='g', lw=3, linestyle='dashdot', label="$ax + b$", alpha=0.4)
                print("Energy: a, b:", popt, "slope:", popt[0], np.sqrt(np.diag(pcov))[0])
            ax.plot(height, etot[:,0]-offset, color='k', fillstyle='none', lw=1)
            ax.set_ylabel("$\\frac{\\Delta E}{2\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=20)
        elif(which=='potential'):
            ax.plot(width/boxSize[0], epot[:,0], color='g', fillstyle='none', lw=1)
            ax.set_ylabel("$Potential$ $energy,$ $U/N$", fontsize=16)
        elif(which=='kinetic'):
            ax.plot(width/boxSize[0], ekin[:,0], color='r', fillstyle='none', lw=1)
            ax.set_ylabel("$Kinetic$ $energy,$ $K/N$", fontsize=16)
        else:
            print("please specify the type of plot between total, potential, kinetic and force")
        ax.tick_params(axis='both', labelsize=14)
        ax.locator_params(axis='x', nbins=5)
        #ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
        ax.set_xlabel("$L_y-L_y^0$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/" + figureName
        plt.tight_layout()
        fig.savefig(figureName + ".png", transparent=True, format = "png")
        plt.show()

def compareEnergyVSTimeStrain(dirName, figureName, type='nh2', dirType='ext', versus='strain', which='time', compare=False):
    if(versus == 'temp'):
        fig, ax = plt.subplots(figsize=(8,5), dpi = 120)
    else:
        fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(versus == 'strain'):
        strainList = np.array([1e-05, 2e-05, 5e-05])
        dirList = np.array([type + '/T0.80/nve-biaxial-' + dirType + '1e-05-tmax1e03/', type + '/T0.80/nve-biaxial-' + dirType + '2e-05-tmax1e03/', 
                            type + '/T0.80/nve-biaxial-' + dirType + '5e-05-tmax1e03/'])
        labelList = ['$\\Delta \\gamma = 5 \\times 10^{-6}$', '$\\Delta \\gamma = 10^{-5}$', '$\\Delta \\gamma = 2 \\times 10^{-5}$', '$\\Delta \\gamma = 5\\times 10^{-5}$']
    elif(versus == 'tmax'):
        strainList = np.array([5e-06, 5e-06])
        dirList = np.array([type + '/T1.20/nve-biaxial-' + dirType + '5e-06-tmax1e03/', type + '/T1.20/nve-biaxial-' + dirType + '5e-06-tmax1e04/'])
        labelList = ['$t_{step} = 0.1$', '$t_{step} = 0.01$']
    elif(versus == 'phi'):
        dirList = np.array(['0.56', '0.57', '0.58', '0.59', '0.61', '0.62', '0.63', '0.64'])
        phi = dirList.astype(np.float32)
        strainList = np.ones(dirList.shape[0])*5e-06
        labelList = ['$\\varphi = 0.56$', '$\\varphi = 0.57$', '$\\varphi = 0.58$', '$\\varphi = 0.59$',
                    '$\\varphi = 0.61$', '$\\varphi = 0.62$', '$\\varphi = 0.63$', '$\\varphi = 0.64$']
    elif(versus == 'temp'):
        dirList = np.array(['0.60', '0.70', '0.80', '0.90', '1.10', '1.20', '1.30', '1.40', '1.50',
                            '1.60', '1.70', '1.80', '1.90', '2.00', '2.10', '2.20', '2.40', '2.60', '2.80', '3.00'])
        strainList = np.ones(dirList.shape[0])*5e-06
        labelList = ['$T = 0.6$', '$T = 0.7$', '$T = 0.8$', '$T = 0.9$', '$T = 1.1$', '$T = 1.2$', '$T = 1.3$', '$T = 1.4$', '$T = 1.5$',
                     '$T = 1.6$', '$T = 1.7$', '$T = 1.8$', '$T = 1.9$', '$T = 2.0$', '$T = 2.1$', '$T = 2.2$', '$T = 2.4$', '$T = 2.6$', '$T = 2.8$', '$T = 3.0$']
    elif(versus == "active"):
        dirList = np.array(['1', '2', '3', '4', '5', '6', '7', '8'])
        strainList = np.ones(dirList.shape[0])*5e-06
        labelList = ['$f_0 = 1$', '$f_0 = 2$', '$f_0 = 3$', '$f_0 = 4$', '$f_0 = 5$', '$f_0 = 6$', '$f_0 = 7$', '$f_0 = 8$']
    elif(versus == 'compext'):
        strainList = np.array([1e-05, 1e-05])
        #dirList = np.array(['box13/2lj/0.64/' + type + '/T0.80/nve-biaxial-comp1e-05-tmax1e03/', 'box31/2lj/0.64/' + type + '/T0.80/nve-biaxial-ext1e-05-tmax1e03/'])
        dirList = np.array(['box31/2lj/0.64/' + type + '/T0.80/nve-biaxial-ext1e-05-tmax1e03/', 
                            'box31/2lj/0.64/' + type + '/T0.80/nve-biaxial-ext1e-05-tmax1e03/strain2.0000/nve-biaxial-comp1e-05-tmax1e03/'])
        labelList = ['$Extension$', '$Compression$']
        typeList = ['ext', 'comp']
    elif(versus == 'ens'):
        strainList = np.array([5e-06, 5e-06, 5e-06, 5e-06, 5e-06])
        dirList = np.array(['nh2/T0.80/nve-biaxial-' + dirType + '-wall5e-06-tmax1e03/', 'nh2/T0.80/nh-biaxial-' + dirType + '-wall5e-06-tmax1e03/',
                            'nh2/T0.80/nvt-biaxial-' + dirType + '-eq5e-06-tmax1e03/', 'langevin2/T0.80/nvt-biaxial-' + dirType + '-eq5e-06-tmax1e03/', 
                            'langevin2/T0.80/active2/tp1e-01-f01/active-biaxial-' + dirType + '-eq5e-06-tmax1e03/'])
        labelList = ['$NVE$', '$NH$', '$NVT(NH)$', '$NVT$', '$Active$']
    elif(versus == 'check'):
        strainList = np.array([5e-06, 5e-06])
        dirList = np.array(['nh2/T0.80/nve-biaxial-' + dirType + '5e-06-tmax1e03/', 'nh2/T0.80/nve-biaxial-' + dirType + '-wall5e-06-tmax1e03/'])
        labelList = ['$Run$ $1$', '$Run$ $2$']
    elif(versus == 'monodia'):
        strainList = np.array([5e-06, 5e-06])
        dirList = np.array(['nh2/T1.00/nve-biaxial-' + dirType + '5e-06-tmax1e03/', '../../lj/0.64/nh2/T1.00/nve-biaxial-' + dirType + '5e-06-tmax1e03/'])
        labelList = ['$Diatomic$', '$Monoatomic$']
    else:
        dirList = np.array(['nh-ljmp/T0.80/nve-biaxial-' + dirType + '1e-05-tmax1e03/', 'nh-ljwca/T0.80/nve-biaxial-' + dirType + '1e-05-tmax1e03/', 
                            'nh2/T0.80/nve-biaxial-' + dirType + '1e-05-tmax1e03/'])
        strainList = np.ones(dirList.shape[0])*1e-05
        labelList = ['$LJ^{\\pm}$', '$LJ-WCA$', '$LJ, \\varepsilon_{AA} = \\varepsilon_{BB} = 2, \\varepsilon_{AB} = 0.5$']
    if(versus == 'temp' or versus == 'active'):
        colorList = cm.get_cmap('viridis', dirList.shape[0])
    else:
        colorList = ['g', 'b', 'r', 'k', 'c', [1,0.5,0], [0,0.5,1], [0.5,0,1]]
    markerList = ['o', 's', 'v', 'd', 'x', '+', 'D', '^']
    lsList = ['solid', 'dotted', 'dashdot', 'dashdot', '-.', ':', '--', 'solid']
    e0 = 0
    tension = np.zeros((dirList.shape[0],2))
    temp = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(versus == 'phi'):
            dirSample = dirName + dirList[d] + '/' + type + '/T1.00/nve-biaxial-' + dirType + '5e-06-tmax2e03/'
        elif(versus == 'temp'):
            dirSample = dirName + type + '/T' + dirList[d] + '/nve-biaxial-' + dirType + '5e-06-tmax1e03/'
        elif(versus == 'active'):
            dirSample = dirName + type + '/tp1e-01-f0' + dirList[d] + '/active-biaxial-' + dirType + '5e-06-tmax1e03/'
        else:
            dirSample = dirName + dirList[d]
        if(os.path.exists(dirSample)):
            numParticles = utils.readFromParams(dirSample, 'numParticles')
            boxSize = np.loadtxt(dirSample + '/boxSize.dat')
            if(which == 'time'):
                if not(os.path.exists(dirSample + "/energyTime.dat")):
                    if(versus == 'compext'):
                        plotSPStrainEnergyVSTime(dirSample, 'temp', 'total', strainList[d], compext=typeList[d])
                    else:
                        plotSPStrainEnergyVSTime(dirSample, 'temp', 'total', strainList[d], compext=dirType)
                data = np.loadtxt(dirSample + "/energyTime.dat")
                strain = data[:,0]
                if(versus == 'compext'):
                    if(d==0):
                        e0 = data[0,1]
                    energy = (data[:,1]-e0)
                    height = data[:,-2]
                else:
                    energy = data[:,1]*numParticles
                    mean = np.mean(energy[strain<0.01])
                    height = data[:,-2] - boxSize[1]
                    if(versus == 'temp' or versus == 'active'):
                        energy = energy[strain<0.4]
                        height = height[strain<0.4]
                    elif(versus == 'ens'):
                        energy = energy[strain<0.3]
                        height = height[strain<0.3] 
                failed = False
                try:
                    popt, pcov = curve_fit(lineFit, height, energy)
                except RuntimeError:
                    print("Error - curve_fit failed")
                    failed = True
                if(failed == False):
                    noise = np.sqrt(np.mean((lineFit(height, *popt) - energy)**2))
                    offset = mean + noise
                    print("offset:", offset, "curve noise:", noise)
                    # plotting fit
                    label = "$slope: $" + str(np.format_float_positional(popt[0], precision=3))
                    if(versus == 'temp' or versus == 'active'):
                        ax.plot(height, lineFit(height, *popt)-offset, color=colorList(d/dirList.shape[0]), lw=2.5, linestyle='dashed', alpha=0.4)
                    elif(versus == 'ens'):
                        ax.plot(height, lineFit(height, *popt), color=colorList[d], lw=2.5, linestyle=lsList[d], alpha=0.4)
                    else:
                        ax.plot(height, lineFit(height, *popt)-offset, color=colorList[d], lw=2.5, linestyle=lsList[d], alpha=0.4)
                    if(versus == 'temp' or versus == 'ens' or versus == 'active' or versus == 'phi'):
                        tension[d,0] = popt[0]
                        tension[d,1] = np.sqrt(np.diag(pcov))[0]
                        temp[d] = np.mean(data[:,5])
                        print("tension from fit:", tension[d], "temp:", 2*temp[d])
                # plotting data
                if(versus == 'temp' or versus == 'active'):
                    ax.plot(height, energy-offset, color=colorList(d/dirList.shape[0]), ls='solid', label=labelList[d])
                elif(versus == 'ens'):
                    ax.plot(height, energy, color=colorList[d], ls=lsList[d], lw=0.9, label=labelList[d])
                else:
                    ax.plot(height, energy-offset, color=colorList[d], ls=lsList[d], lw=0.9, label=labelList[d])
            else:
                if not(os.path.exists(dirSample + "/energyStrain.dat")):
                    plotSPEnergyVSStrain(dirSample, 'temp', 'total', 0, 0, plot=False)
                data = np.loadtxt(dirSample + "/energyStrain.dat")
                ax.errorbar(data[:,1], data[:,2], data[:,3], color=colorList[d], marker=markerList[d], markersize=8, fillstyle='none', lw=1, capsize=3, label=labelList[d])
    if(versus == 'temp'):
        colorBar = cm.ScalarMappable(cmap=colorList)
        cb = plt.colorbar(colorBar)
        cb.set_label(label='$T$', fontsize=18, labelpad=20, rotation='horizontal')
        cb.set_ticks([0,0.2,0.4,0.6,0.8,1])
        labels = np.linspace(temp[0], temp[-1], 6)
        for i in range(labels.shape[0]):
            labels[i] = str(np.format_float_positional(labels[i],1))
        cb.set_ticklabels(labels)
        cb.ax.tick_params(labelsize=14, size=0)
    else:
        ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    #ax.locator_params(axis='x', nbins=5)
    #ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
    if(dirType == "comp" or dirType == "comp-wall" or dirType == "comp-eq"):
        ax.set_xlabel("$L_y^0-L_y$", fontsize=16)
    elif(dirType == "ext" or dirType == "ext-wall" or dirType == "ext-eq"):
        ax.set_xlabel("$L_y-L_y^0$", fontsize=16)
    if(versus == 'ens'):
        ax.set_ylabel("$\\frac{E}{2\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=20)
    else:
        ax.set_ylabel("$\\frac{\\Delta E}{2\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=20)
        #ax.set_ylabel("$\\frac{K}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
    figureName = dirType + "-" + figureName
    figure1Name = "/home/francesco/Pictures/soft/mips/energy-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    if(versus == 'phi'):
        fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
        phi = phi[tension[:,0]!=0]
        tension = tension[tension[:,0]!=0]
        ax.errorbar(phi, tension[:,0], tension[:,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=16)
        ax.set_ylabel("$\\frac{\\gamma \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=20)
    if(versus == 'temp'):
        np.savetxt(dirName + type + "/energyTension.dat", np.column_stack((temp, tension)))
        fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
        temp = temp[tension[:,0]!=0]
        tension = tension[tension[:,0]!=0]
        ax.errorbar(temp, tension[:,0], tension[:,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none')
        if(compare == 'compare'):
            data = np.loadtxt(dirName + '../../lj/0.64/nh2/energyTension.dat')
            data = data[data[:,1]!=0]
            ax.errorbar(data[:,0], data[:,1], data[:,2], color='b', marker='s', markersize=8, capsize=3, fillstyle='none')
            ax.legend(('$Diatomic$', '$Monoatomic$'),fontsize=12, loc='best')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel("$Temperature,$ $T / \\varepsilon$", fontsize=16)
        ax.set_ylabel("$\\frac{\\gamma \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=20)
    elif(versus == 'active'):
        np.savetxt(dirName + type + "/energyTension.dat", np.column_stack((temp, tension)))
        fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
        temp = temp[tension[:,0]!=0]
        tension = tension[tension[:,0]!=0]
        ax.errorbar(temp, tension[:,0], tension[:,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none')
        if(compare == 'compare'):
            data = np.loadtxt(dirName + '../../lj/0.64/langevin2/T0.80/active2/energyTension.dat')
            data = data[data[:,1]!=0]
            ax.errorbar(data[:,0], data[:,1], data[:,2], color='b', marker='s', markersize=8, capsize=3, fillstyle='none')
            ax.legend(('$Diatomic$', '$Monoatomic$'),fontsize=12, loc='best')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel("$Temperature,$ $T / \\varepsilon$", fontsize=16)
        ax.set_ylabel("$\\frac{\\gamma \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=20)
    elif(versus == "ens"):
        fig, ax = plt.subplots(figsize=(6,4.5), dpi = 120)
        ens = np.array([1, 2, 3, 4, 5])
        ens = ens[tension[:,0]!=0]
        tension = tension[tension[:,0]!=0]
        ax.errorbar(ens, tension[:,0], tension[:,1], color='k', marker='o', markersize=10, capsize=3, lw=0.9, fillstyle='none')
        ax.set_ylim(-0.05, 1.85)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xticks(ens)
        ax.set_xticklabels(["$NVE$", "$NH$", "$NVT(NH)$", "$NVT$", "$Active$"])
        ax.set_ylabel("$\\frac{\\gamma \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=20)
    figure2Name = "/home/francesco/Pictures/soft/mips/tension-" + figureName
    plt.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotTensionVSPhi(dirName, figureName):
    dirList = np.array(['0.60', '0.64'])
    colorList = ['g', 'b']
    markerList = ['o', 's']
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    for d in range(dirList.shape[0]):
        tension = np.loadtxt(dirName + dirList[d] + "/nh2/energyTension.dat")
        ax.errorbar(tension[tension[:,1]!=0,0], tension[tension[:,1]!=0,1], tension[tension[:,1]!=0,2], 
                    color=colorList[d], marker=markerList[d], markersize=8, capsize=3, fillstyle='none', lw=1, label="$\\varphi=$" + dirList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Temperature,$ $T / \\varepsilon$", fontsize=16)
    ax.set_ylabel("$\\frac{\\gamma \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=20)
    figureName = "/home/francesco/Pictures/soft/mips/tensionVSphi-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotEnergyVSTempPhi(dirName, figureName, which='temp'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    phiList = np.array(['0.66', '0.68', '0.70', '0.72'])
    tempList = np.array(['0.40', '0.60', '0.80', '1.00'])
    phi = phiList.astype(np.float64)
    temp = tempList.astype(np.float64)
    colorList = ['k', 'b', 'c', 'g']
    markerList = ['o', 's', 'v', 'd']
    epot = np.zeros((phiList.shape[0],tempList.shape[0],2))
    ekin = np.zeros((phiList.shape[0],tempList.shape[0],2))
    etot = np.zeros((phiList.shape[0],tempList.shape[0],2))
    for d in range(phiList.shape[0]):
        for t in range(tempList.shape[0]):
            dirSample = dirName + phiList[d] + "/nve/T" + tempList[t] + "/dynamics/"
            if(os.path.exists(dirSample + "/energy.dat")): 
                data = np.loadtxt(dirSample + "/energy.dat")
                epot[d,t,0] = np.mean(data[:,2])
                epot[d,t,1] = np.std(data[:,2])
                ekin[d,t,0] = np.mean(data[:,3])
                ekin[d,t,1] = np.std(data[:,3]) 
                etot[d,t,0] = np.mean(data[:,4])
                etot[d,t,1] = np.std(data[:,4])
    if(which=='temp'):
        xlabel = "$Temperature,$ $T$"
        figureName = "temp-" + figureName
        for d in range(phiList.shape[0]):
            ax.errorbar(temp, epot[d,:,0], epot[d,:,1], color=colorList[d], marker='o', markersize=10, fillstyle='none', capsize=3, lw=1.2, 
                        label="$\\varphi=$" + phiList[d])
            ax.errorbar(temp, ekin[d,:,0], ekin[d,:,1], color=colorList[d], marker='v', markersize=10, fillstyle='none', capsize=3, lw=1.2)
            ax.errorbar(temp, etot[d,:,0], etot[d,:,1], color=colorList[d], marker='s', markersize=10, fillstyle='none', capsize=3, lw=1.2)
    elif(which=='phi'):
        xlabel = "$Packing$ $fraction,$ $\\varphi$"
        figureName = "phi-" + figureName
        for d in range(tempList.shape[0]):
            ax.errorbar(phi, epot[:,d,0], epot[:,d,1], color=colorList[d], marker='o', markersize=10, fillstyle='none', capsize=3, lw=1.2, 
                        label="$T=$" + tempList[d])
            ax.errorbar(phi, ekin[:,d,0], ekin[:,d,1], color=colorList[d], marker='v', markersize=10, fillstyle='none', capsize=3, lw=1.2)
            ax.errorbar(phi, etot[:,d,0], etot[:,d,1], color=colorList[d], marker='s', markersize=10, fillstyle='none', capsize=3, lw=1.2)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Energy$ $per$ $particle$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/energyVS-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plot2DensityProfile(dirName, figureName, num1=0):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    tempList = np.array(['0.40', '0.60', '0.80', '1.00'])
    colorList = ['k', 'b', 'c', 'g']
    markerList = ['o', 's', 'v', 'd']
    for t in range(tempList.shape[0]):
        dirSample = dirName + "/nve/T" + tempList[t] + "/dynamics/"
        if not(os.path.exists(dirSample + "/densityProfile.dat")): 
            interface.average2DensityProfile(dirSample, num1)
        data = np.loadtxt(dirSample + "densityProfile.dat")
        ax.errorbar(data[:,0], data[:,1], data[:,2], color=colorList[t], marker=markerList[t], markersize=8, fillstyle='none', capsize=3, lw=1.2, 
                    label="$T=$" + tempList[t])
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$Density$ $profile,$ $\\varphi_1(x)$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/2profile-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plot2PhaseWidthVSTime(dirName, figureName, num1=0):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    tempList = np.array(['0.40', '0.60', '0.80', '1.00'])
    colorList = ['k', 'b', 'c', 'g']
    markerList = ['o', 's', 'v', 'd']
    for t in range(tempList.shape[0]):
        dirSample = dirName + "/nve/T" + tempList[t] + "/dynamics/"
        boxWidth = np.loadtxt(dirSample + "boxSize.dat")[0]
        timeStep = utils.readFromParams(dirSample, "dt")
        if not(os.path.exists(dirSample + "/phaseWidth.dat")): 
            interface.average2DensityProfile(dirSample, num1)
        data = np.loadtxt(dirSample + "phaseWidth.dat")
        data[:,0] *= timeStep
        plt.plot(data[:,0], data[:,1]/boxWidth, color=colorList[t], marker=markerList[t], markersize=8, fillstyle='none', lw=1.2)
    ax.set_ylim(-0.02,0.57)
    x = np.linspace(np.min(data[:,0]), np.max(data[:,0]),100)
    plt.plot(x, 0.5 * np.ones(100), ls='dashed', color='k')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $time,$ $t$", fontsize=16)
    ax.set_ylabel("$Width,$ $w_1/L_x$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/widthtime" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plot2CorrelationVSTime(dirName, figureName, which='temp', num1=0):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='temp'):
        dirList = np.array(['0.60', '0.80', '1.00', '1.20'])
        labelList = np.array(['$T=0.60$', '$T=0.80$', '$T=1.00$', '$T=1.20$'])
    else:
        dirList = np.array(['nh-ljmp', 'nh-ljwca', 'nh2'])
        labelList = np.array(['$LJ^{\\pm}$', '$LJ-WCA$', '$LJ, \\varepsilon_{AA} = \\varepsilon_{BB} = 2, \\varepsilon_{AB} = 0.5$'])
    colorList = ['k', 'b', 'c', 'g']
    markerList = ['o', 's', 'v', 'd']
    for d in range(dirList.shape[0]):
        if(which=='temp'):
            dirSample = dirName + "/nh2/T" + dirList[d] + "/dynamics-log/"
        else:
            dirSample = dirName + "/" + dirList[d] + "/T0.80/dynamics-log/"
        if(os.path.exists(dirSample)):
            if not(os.path.exists(dirSample + "/2logCorr.dat")): 
                interface.compute2FluidsCorr(dirSample, 0, 7, 6, num1)
            data = np.loadtxt(dirSample + "2logCorr.dat")
            plt.plot(data[:,0], data[:,3], color=colorList[d], lw=1.2, ls='dashed')
            plt.plot(data[:,0], data[:,7], color=colorList[d], marker=markerList[d], markersize=8, fillstyle='none', lw=1.2, label=labelList[d])
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $time,$ $t$", fontsize=16)
    ax.set_ylabel("$ISF$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/corrtime" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plot2PhaseWidthVSStrain(dirName, figureName, num1=0):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if not(os.path.exists(dirName + "/phaseWidth.dat")): 
        interface.compute2PhaseWidthVSStrain(dirName, num1)
    data = np.loadtxt(dirName + "phaseWidth.dat")
    data[:,1] /= data[0,1]
    x = np.linspace(np.min(data[:,1]), np.max(data[:,1]),100)
    plt.plot(x, 0.5 * np.ones(100), ls='dashed', color='k')
    plt.plot(data[:,1], data[:,3]/data[:,2], color='g', lw=1.2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Box$ $height,$ $L_y/L_y^0$", fontsize=16)
    ax.set_ylabel("$Width,$ $w_1/L_x$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/widthgamma" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()
    
def plotSPEnergyVSStrainVStmax(dirName, figureName, simple=False, limit=0):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    dirtmax = np.array(['2e05', '4e05', '6e05', '8e05', '2e06'])
    colorList = ['k', 'b', 'g', 'c', 'r']
    markerList = ['o', 's', 'v', '^', 'd']
    labelName = "$t_{max}=$"
    for t in range(dirtmax.shape[0]):
        dirPath = dirName + "/biaxial1e-03-tmax" + dirtmax[t] +  "/"
        dirList, strain = utils.getOrderedStrainDirectories(dirPath)
        energy = np.zeros((dirList.shape[0],3))
        error = np.zeros((dirList.shape[0],3))
        length = np.zeros(dirList.shape[0])
        for d in range(dirList.shape[0]):
            dirSample = dirPath + dirList[d]
            if(simple=='simple'):
                dirSample += "/simple/"
            if(os.path.exists(dirSample)):
                length[d] = np.loadtxt(dirSample + "/boxSize.dat")[1]
                data = np.loadtxt(dirSample + "/energy.dat")
                energy[d,0] = np.mean(data[:,2] + data[:,3])
                error[d,0] = np.std(data[:,2] + data[:,3])
        energy = energy[length!=0]
        error = error[length!=0]
        strain = strain[length!=0]
        length = length[length!=0]
        if(limit!=0):
            energy = energy[length<limit]
            error = error[length<limit]
            length = length[length<limit]
        plt.errorbar(length/length[0], energy[:,0], error[:,0], marker=markerList[t], markersize=8, color=colorList[t], lw=1, capsize=3, fillstyle='none', label=labelName + dirtmax[t])
    ax.set_yscale('log')
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Box$ $height,$ $L_y/L_y^0$", fontsize=16)
    ax.set_ylabel("$Energy$ $per$ $particle$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/energyVStmax-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPEnergyVSStrainVSTemp(dirName, figureName, simple=False, limit=0, tmax='1e06'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    dirTemp = np.array(['0.4', '0.6'])
    colorList = ['k', 'b', 'g', 'c', 'r']
    markerList = ['o', 's', 'v', '^', 'd']
    labelName = "$T=$"
    for t in range(dirTemp.shape[0]):
        dirPath = dirName + "T" + dirTemp[t] + "/0.72/nve/biaxial1e-03-tmax" + tmax +  "/"
        dirList, strain = utils.getOrderedStrainDirectories(dirPath)
        energy = np.zeros((dirList.shape[0],3))
        error = np.zeros((dirList.shape[0],3))
        length = np.zeros(dirList.shape[0])
        for d in range(dirList.shape[0]):
            dirSample = dirPath + dirList[d]
            if(simple=='simple'):
                dirSample += "/simple/"
            if(os.path.exists(dirSample)):
                length[d] = np.loadtxt(dirSample + "/boxSize.dat")[1]
                data = np.loadtxt(dirSample + "/energy.dat")
                energy[d,0] = np.mean(data[:,2] + data[:,3])
                error[d,0] = np.std(data[:,2] + data[:,3])
                energy[d,1] = np.mean(data[:,2])
                error[d,1] = np.std(data[:,2])
                energy[d,2] = np.mean(data[:,3])
                error[d,2] = np.std(data[:,3])
        energy = energy[length!=0]
        error = error[length!=0]
        strain = strain[length!=0]
        length = length[length!=0]
        if(limit!=0):
            energy = energy[length<limit]
            error = error[length<limit]
            length = length[length<limit]
        plt.errorbar(length/length[0], energy[:,0]/energy[0,0], error[:,0], marker=markerList[t], markersize=8, color=colorList[t], lw=1, capsize=3, fillstyle='none', label=labelName + dirTemp[t])
    #ax.set_yscale('log')
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Box$ $height,$ $L_y/L_y^0$", fontsize=16)
    ax.set_ylabel("$Energy$ $per$ $particle$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/energyVSTemp-" + figureName
    plt.tight_layout()
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
                mean[d] = np.mean(data[:,index]/width[d])#*width[d]/(count*boxWidth[d]))
                error[d] = np.std(data[:,index]/width[d])#*width[d]/(count*boxWidth[d]))/10
    if(which=='average'):
        mean = mean[mean!=0]
        error = error[error!=0]
        strain = strain[length!=0]
        length = length[length!=0]
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    #ax.set_xlabel("$Strain,$ $\\epsilon$", fontsize=16)
    #ax.set_ylim(-0.04, 0.64)
    ax.errorbar(length, mean, error, lw=1, marker='s', markersize=5, color='k', capsize=3, markeredgecolor='k', fillstyle='left', label='$From$ $simulation$')
    #ax.errorbar(strain, mean, error, lw=1, marker='s', markersize=5, color='k', capsize=3, markeredgecolor='k', fillstyle='left', label='$From$ $simulation$')
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
            #ax.plot(strain, lineFit(strain, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax + b$")
            ax.plot(length, lineFit(length, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax + b$")
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
            #ax.plot(strain, lineFit(strain, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax + b$")
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
            ax.plot(length, lineFit(length, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax + b$")
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
            #ax.plot(midStrain, lineFit(midStrain, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax + b$")
            ax.plot(midLength, lineFit(midLength, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax + b$")
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

    if(whichPlot == "interface"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        qmax = float(sys.argv[5])
        plotInterfaceFluctuations(dirName, figureName, which, qmax)

    elif(whichPlot == "2interface"):
        figureName = sys.argv[3]
        num1 = int(sys.argv[4])
        limit = float(sys.argv[5])
        plot2InterfaceFluctuations(dirName, figureName, num1, limit)

    elif(whichPlot == "interfacetemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        qmax = float(sys.argv[5])
        plotInterfaceVSTemp(dirName, figureName, which, qmax)

    elif(whichPlot == "interfacecorr"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotInterfaceCorrelation(dirName, figureName, which)

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

    elif(whichPlot == "activetemp"):
        figureName = sys.argv[3]
        plotSPActiveEnergyVSTemp(dirName, figureName)

    elif(whichPlot == "isf"):
        figureName = sys.argv[3]
        plotSPClusterISF(dirName, figureName)

    elif(whichPlot == "paircorr"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPPairCorrelation(dirName, figureName, which)

    elif(whichPlot == "profiletemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotProfileVSTemp(dirName, figureName, which)

    elif(whichPlot == "walltime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        index = int(sys.argv[5])
        numSamples = int(sys.argv[6])
        plotSPWallForceVSTime(dirName, figureName, which, index, numSamples)

    elif(whichPlot == '2isf'):
        figureName = sys.argv[3]
        T1 = sys.argv[4]
        T2 = sys.argv[5]
        decade = int(sys.argv[6])
        which = sys.argv[7]
        plot2FluidsISF(dirName, figureName, T1, T2, decade, which)

    elif(whichPlot == 'compare2isf'):
        figureName = sys.argv[3]
        which = sys.argv[4]
        compare2FluidsISF(dirName, figureName, which)

    elif(whichPlot == "energystrain"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        simple = sys.argv[5]
        limit = float(sys.argv[6])
        plotSPEnergyVSStrain(dirName, figureName, which, simple, limit, plot=True)

    elif(whichPlot == "forcetime"):
        figureName = sys.argv[3]
        strainStep = float(sys.argv[4])
        compext = sys.argv[5]
        slope = sys.argv[6]
        plotSPStrainForceVSTime(dirName, figureName, strainStep, compext, slope, plot=True)

    elif(whichPlot == "forcecompare"):
        figureName = sys.argv[3]
        compext = sys.argv[4]
        which = sys.argv[5]
        method = sys.argv[6]
        compare = sys.argv[7]
        compareForceVSTimeStrain(dirName, figureName, compext, which, method, compare)

    elif(whichPlot == "energytime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        strainStep = float(sys.argv[5])
        compext = sys.argv[6]
        plotSPStrainEnergyVSTime(dirName, figureName, which, strainStep, compext, plot=True)

    elif(whichPlot == "energycompare"):
        figureName = sys.argv[3]
        type = sys.argv[4]
        dirType = sys.argv[5]
        versus = sys.argv[6]
        which = sys.argv[7]
        compare = sys.argv[8]
        compareEnergyVSTimeStrain(dirName, figureName, type, dirType, versus, which, compare)

    elif(whichPlot == "tensionphi"):
        figureName = sys.argv[3]
        plotTensionVSPhi(dirName, figureName)

    elif(whichPlot == "2energy"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotEnergyVSTempPhi(dirName, figureName, which)

    elif(whichPlot == "2profile"):
        figureName = sys.argv[3]
        num1 = int(sys.argv[4])
        plot2DensityProfile(dirName, figureName, num1)

    elif(whichPlot == "2widthtime"):
        figureName = sys.argv[3]
        num1 = int(sys.argv[4])
        plot2PhaseWidthVSTime(dirName, figureName, num1)

    elif(whichPlot == "2corrtime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        num1 = int(sys.argv[5])
        plot2CorrelationVSTime(dirName, figureName, which, num1)

    elif(whichPlot == "2widthgamma"):
        figureName = sys.argv[3]
        num1 = int(sys.argv[4])
        plot2PhaseWidthVSStrain(dirName, figureName, num1)

    elif(whichPlot == "energytmax"):
        figureName = sys.argv[3]
        simple = sys.argv[4]
        limit = float(sys.argv[5])
        plotSPEnergyVSStrainVStmax(dirName, figureName, simple, limit)

    elif(whichPlot == "energytemp"):
        figureName = sys.argv[3]
        simple = sys.argv[4]
        limit = float(sys.argv[5])
        tmax = sys.argv[6]
        plotSPEnergyVSStrainVSTemp(dirName, figureName, simple, limit, tmax)

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

    elif(whichPlot == "forcetemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotWallForceVSTemperature(dirName, figureName, which)

    elif(whichPlot == "forcepdf"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
        plotForcePDF(dirName, figureName, numBins)

    elif(whichPlot == "pdftemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        numBins = int(sys.argv[5])
        plotForcePDFVSTemperature(dirName, figureName, which, numBins)

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

    elif(whichPlot == "templv"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPTemperatureLV(dirName, figureName, which)