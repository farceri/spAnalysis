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

def linearFit(x, a, b):
    return a*x + b

def quadraticFit(x, a, b, c):
    return a*x**2 + b*x + c

def polyFit(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

################################ Shear graphics ################################
def plotStressStrain(dirName, figureName, which='lj', strain=0):
    sigma = utils.readFromDynParams(dirName, 'sigma')
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    if not(os.path.exists(dirName + os.sep + 'strainStress.dat')):
        if(which=='lj'):
            cluster.computeLJStressVSStrain(dirName, strain=strain)
        elif(which=='active'):
            cluster.computeStressVSStrain(dirName, strain=strain, active='active')
        else:
            print("Please specify the sample type")
    data = np.loadtxt(dirName + os.sep + "strainStress.dat")
    if(which=='lj'):
        ax.plot(data[:,0], data[:,2]+data[:,4], linewidth=1.1, color='k')
    elif(which=='active'):
        ax.plot(data[:,0], data[:,2]+data[:,4]+data[:,6], linewidth=1.1, color='k')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
    ax.set_ylabel("$Stress,$ $\\sigma$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/shearStrain-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotStressTime(dirName, figureName, which='lj', strain=0, logx=False):
    sigma = utils.readFromDynParams(dirName, 'sigma')
    fig, ax = plt.subplots(figsize = (6.5,4), dpi = 120)
    if not(os.path.exists(dirName + os.sep + 'timeStress.dat')):
        if(which=='lj'):
            cluster.computeLJStressVSTime(dirName, strain=strain)
        elif(which=='active'):
            cluster.computeStressVSTime(dirName, strain=strain, active='active')
        else:
            print("Please specify the sample type")
    data = np.loadtxt(dirName + os.sep + "timeStress.dat")
    data[:,0] *= sigma
    data = data[data[:,0]<50,:]
    if(which=='lj'):
        ax.plot(data[:,0], data[:,2]+data[:,4], linewidth=1.2, color='k')
    elif(which=='active'):
        ax.plot(data[:,0], data[:,2]+data[:,4]+data[:,6], linewidth=1.2, color='k')
    if(logx=='logx'):
        ax.set_xscale('log')
    ax.plot(data[:,0], np.zeros(data[:,0].shape[0]), ls='dotted', color='k')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_ylabel("$Stress,$ $\\sigma$", fontsize=16)
    plt.tight_layout()
    if(logx=='logx'):
        figureName = "/home/francesco/Pictures/soft/mips/shearTimeLog-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/mips/shearTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotStressTimeVSStrain(dirName, figureName, which='lj', limit=1e02):
    fig, ax = plt.subplots(figsize = (6.5,4), dpi = 120)
    strainList = np.array(['1.2e-01', '1.4e-01', '1.6e-01', '1.8e-01', '2e-01', '2.2e-01', '2.4e-01'])
    colorList = cm.get_cmap('viridis', strainList.shape[0])
    strain = strainList.astype(np.float64)
    stressLong = np.zeros((strain.shape[0],2))
    stressShort = np.zeros((strain.shape[0],2))
    for d in range(strainList.shape[0]):
        dirSample = dirName + "/dynamicsLE" + strainList[d] + "-log/"
        if not(os.path.exists(dirSample + os.sep + 'timeStress.dat')):
            if(which=='lj'):
                cluster.computeLJStressVSTime(dirSample, strain=strain[d])
            elif(which=='active'):
                cluster.computeStressVSTime(dirSample, strain=strain[d], active='active')
        data = np.loadtxt(dirSample + os.sep + "timeStress.dat")
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        print(strainList[d])
        data[:,0] *= sigma
        #data = data[data[:,0]<80,:]
        if(which=='lj'):
            stressData = data[:,3]+data[:,6]#data[:,2]+data[:,4] # old version
            stressShort[d,0] = np.mean(stressData[data[:,0]<2e-02])
            stressShort[d,1] = np.std(stressData[data[:,0]<2e-02])
        elif(which=='active'):
            stressData = data[:,3]+data[:,6]-data[:,9]
            stressShort[d,0] = np.mean(stressData[data[:,0]<1e-02])
            stressShort[d,1] = np.std(stressData[data[:,0]<1e-02])
        stressLong[d,0] = np.mean(stressData[data[:,0]>limit])
        stressLong[d,1] = np.std(stressData[data[:,0]>limit])
        print("Stress - short-time:", stressShort[d,0], "steady-state:", stressLong[d,0])
        ax.plot(data[:,0], stressData, linewidth=1.2, color=colorList(d/strainList.shape[0]), label="$\\gamma=$" + strainList[d], marker='o', markersize=4, fillstyle='none')
    ax.set_xscale('log')
    ax.plot(data[:,0], np.zeros(data[:,0].shape[0]), ls='dotted', color='k')
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_ylabel("$Stress,$ $\\sigma$", fontsize=16)
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/shearTimeVSstrain-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6,5), dpi = 120)
    plt.errorbar(strain, stressShort[:,0], stressShort[:,1], color='g', marker='s', markersize=8, capsize=3, lw=1.2, fillstyle='none', label='$Short-time$ $response$')
    plt.errorbar(strain, stressLong[:,0], stressLong[:,1], color='k', marker='o', markersize=8, capsize=3, lw=1.2, fillstyle='none', label='$Steady$ $state$')
    print("average steady-state stress:", np.mean(stressLong[:,0]), "+-", np.std(stressLong[:,0]))
    x = strain[:4,]
    y = stressShort[:4,0]
    failed = False
    try:
        popt, pcov = curve_fit(linearFit, x, y)
        #popt, pcov = curve_fit(quadraticFit, x, y)
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        ax.plot(strain, linearFit(strain, *popt), color='b', ls='--', lw=1.5, label='$Fit,$ $ax+b:$ $a = $' + str(np.around(popt[0],3)) + ", $b = $" + str(np.around(popt[1],2)))
        #ax.plot(x, quadraticFit(x, *popt), color='b', ls='--', lw=1.5, label='$Fit,$ $ax^2+bx+c:$ $a = $' + str(np.around(popt[0],3)) + ", $b = $" + str(np.around(popt[1],2)) + ", $c = $" + str(np.around(popt[2],2)))
        print("slope (shear modulus):", popt[0], "intersect:", popt[1])
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
    ax.set_ylabel("$Stress,$ $\\sigma_{xy}$", fontsize=16)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/shearMod-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotStressTimeVSExtension(dirName, figureName, which='lj', limit=1e02):
    fig, ax = plt.subplots(figsize = (6.5,4), dpi = 120)
    strainList = np.array(['1e-02', '1.2e-02', '1.4e-02', '1.6e-02', '1.8e-02', '2e-02', '2.2e-02'])
    colorList = cm.get_cmap('viridis', strainList.shape[0])
    strain = strainList.astype(np.float64)
    stressLong = np.zeros((strain.shape[0],2))
    stressShort = np.zeros((strain.shape[0],2))
    for d in range(strainList.shape[0]):
        dirSample = dirName + "/dynamicsLy" + strainList[d] + "-log/"
        if not(os.path.exists(dirSample + os.sep + 'timeStress.dat')):
            if(which=='lj'):
                cluster.computeLJStressVSTime(dirSample)
            elif(which=='active'):
                cluster.computeStressVSTime(dirSample, active='active')
        data = np.loadtxt(dirSample + os.sep + "timeStress.dat")
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        print(strainList[d])
        data[:,0] *= sigma
        #data = data[data[:,0]<80,:]
        if(which=='lj'):
            stressData = data[:,2]+data[:,5]
            stressShort[d,0] = np.mean(stressData[data[:,0]<2e-02])
            stressShort[d,1] = np.std(stressData[data[:,0]<2e-02])
        elif(which=='active'):
            stressData = data[:,2]+data[:,5]+data[:,8]
            stressShort[d,0] = np.mean(stressData[data[:,0]<1e-02])
            stressShort[d,1] = np.std(stressData[data[:,0]<1e-02])
        stressLong[d,0] = np.mean(stressData[data[:,0]>limit])
        stressLong[d,1] = np.std(stressData[data[:,0]>limit])
        print("Stress - short-time:", stressShort[d,0], "steady-state:", stressLong[d,0])
        ax.plot(data[:,0], stressData, linewidth=1.2, color=colorList(d/strainList.shape[0]), label="$\\gamma=$" + strainList[d], marker='o', markersize=4, fillstyle='none')
    ax.set_xscale('log')
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_ylabel("$Stress,$ $\\sigma_{yy}$", fontsize=16)
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/extendTimeVSstrain-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6.5,5), dpi = 120)
    plt.errorbar(strain, stressShort[:,0], stressShort[:,1], color='g', marker='s', markersize=8, capsize=3, lw=1.2, fillstyle='none', label='$Short-time$ $response$')
    plt.errorbar(strain, stressLong[:,0], stressLong[:,1], color='k', marker='o', markersize=8, capsize=3, lw=1.2, fillstyle='none', label='$Steady$ $state$')
    print("average steady-state stress:", np.mean(stressLong[:,0]), "+-", np.std(stressLong[:,0]))
    x = strain[:4]
    y = stressShort[:4,0]
    failed = False
    try:
        popt, pcov = curve_fit(linearFit, x, y, bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        ax.plot(strain, linearFit(strain, *popt), color='b', ls='--', lw=1.5, label='$Fit,$ $ax+b:$ $a = $' + str(np.around(popt[0],3)) + ", $b = $" + str(np.around(popt[1],2)))
        print("slope (bulk modulus):", popt[0], "intersect:", popt[1])
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
    ax.set_ylabel("$Stress,$ $\\sigma_{yy}$", fontsize=16)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/bulkMod-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotStressStrainVSTemp(dirName, figureName, which='lj', deformation='LE'):
    fig, ax = plt.subplots(figsize = (6.5,5), dpi = 120)
    if(which=='lj'):
        tempList = np.array(['0.35', '0.36', '0.37', '0.38', '0.39', '0.40', '0.41'])
        labelName = "$T=$"
    elif(which=='active'):
        tempList = np.array(['4e-04', '3e-04', '2e-04', '1.5e-04', '1.2e-04', '1e-04', '9e-05', '7e-05', '5e-05', '4e-05'])
        #tempList = np.array(['4e-04', '3e-04', '2e-04', '1.5e-04', '1.2e-04', '1e-04'])
        labelName = "$D_r=$"
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('plasma', tempList.shape[0])
    markerList = np.array(['o', 's', 'v', '^', 'x', '+', 'd', 'D', '*', '.'])
    if(deformation=='extend'):
        strainList = np.array(['1e-02', '1.2e-02', '1.4e-02', '1.6e-02', '1.8e-02', '2e-02', '2.2e-02'])
        defName = "Ly"
    elif(deformation=='shear'):
        strainList = np.array(['1.2e-01', '1.4e-01', '1.6e-01', '1.8e-01', '2e-01', '2.2e-01', '2.4e-01'])
        defName = "LE"
    strain = strainList.astype(np.float64)
    for t in range(tempList.shape[0]):
        stressShort = np.zeros((strain.shape[0],2))
        stressLong = np.zeros((strain.shape[0],2))
        for d in range(strainList.shape[0]):
            if(which=='lj'):
                dirSample = dirName + "/T" + tempList[t] + "/dynamics" + defName + strainList[d] + "-log/"
            elif(which=='active'):
                dirSample = dirName + "/Dr" + tempList[t] + "/dynamics" + defName + strainList[d] + "-log/"
            if not(os.path.exists(dirSample + os.sep + 'timeStress.dat')):
                if(which=='lj'):
                    cluster.computeLJStressVSTime(dirSample, strain=strain[d])
                elif(which=='active'):
                    cluster.computeStressVSTime(dirSample, strain=strain[d], active='active')
            data = np.loadtxt(dirSample + os.sep + "timeStress.dat")
            sigma = utils.readFromDynParams(dirSample, 'sigma')
            data[:,0] *= sigma
            if(which=='lj'):
                if(deformation=='extend'):
                    stressData = data[:,2]+data[:,5]
                elif(deformation=='shear'):
                    stressData = data[:,3]+data[:,6]
            elif(which=='active'):
                if(deformation=='extend'):
                    stressData = data[:,2]+data[:,5]+data[:,8]
                elif(deformation=='shear'):
                    stressData = data[:,3]+data[:,6]+data[:,9]
            stressShort[d,0] = np.mean(stressData[data[:,0]<1e-02])
            stressShort[d,1] = np.std(stressData[data[:,0]<1e-02])
            stressLong[d,0] = np.mean(stressData[data[:,0]>100])
            stressLong[d,1] = np.std(stressData[data[:,0]>100])
        ax.errorbar(strain, stressShort[:,0], stressShort[:,1], color=colorList(t/tempList.shape[0]), marker=markerList[t], markersize=8, capsize=3, lw=1, fillstyle='none', label=labelName + tempList[t])
        ax.errorbar(strain, stressLong[:,0], stressLong[:,1], color=colorList(t/tempList.shape[0]), marker=markerList[t], markersize=8, capsize=3, lw=1)
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
    ax.set_ylabel("$Stress,$ $\\sigma$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/stressVSTemp" + defName + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotStressEnergyTime(dirName, figureName, logx=False):
    sigma = utils.readFromDynParams(dirName, 'sigma')
    fig, ax = plt.subplots(2, 1, sharex=True, figsize = (6.5,7), dpi = 120)
    strainList = np.array(['1e-01', '1.2e-01', '1.4e-01', '1.6e-01', '1.8e-01', '2e-01'])
    strain = strainList.astype(np.float64)
    for i in range(strainList.shape[0]):
        dirSample = dirName + "/dynamicsLE" + strainList[i]
        data = np.loadtxt(dirSample + "/energy.dat")
        ax[0].plot(data[:,0], data[:,2]+data[:,3], linewidth=1.2, color='k')
        ax[1].plot(data[:,0], data[:,6], linewidth=1.2, color='k')
    if(logx=='logx'):
        ax[0].set_xscale('log')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$Time,$ $t$", fontsize=16)
    ax[0].set_ylabel("$Energy,$ $E$", fontsize=16)
    ax[1].set_ylabel("$Stress,$ $\\sigma$", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    if(logx=='logx'):
        figureName = "/home/francesco/Pictures/soft/mips/stressEnergyLog-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/mips/stressEnergy-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotEnergyVSStrain(dirName, figureName, which='lj', deformation='extend', limit=1e02):
    sigma = utils.readFromDynParams(dirName + "dynamics", 'sigma')
    fig, ax = plt.subplots(figsize = (6.5,4), dpi = 120)
    if(deformation=='extend'):
        strainList = np.array(['1e-02', '1.2e-02', '1.4e-02', '1.6e-02', '1.8e-02', '2e-02', '2.2e-02'])
        defName = "Ly"
    elif(deformation=='shear'):
        strainList = np.array(['1.2e-01', '1.4e-01', '1.6e-01', '1.8e-01', '2e-01', '2.2e-01', '2.4e-01'])
        defName = "LE"
    colorList = cm.get_cmap('viridis', strainList.shape[0])
    strain = strainList.astype(np.float64)
    energyShort = np.zeros((strain.shape[0],2))
    energyLong = np.zeros((strain.shape[0],2))
    dirSample = dirName + "/dynamics/"
    boxSize = np.loadtxt(dirSample + "/boxSize.dat").astype(np.float64)
    Ly = boxSize[1]
    for d in range(strainList.shape[0]):
        dirSample = dirName + "/dynamics" + defName + strainList[d] + "-log/"
        data = np.loadtxt(dirSample + os.sep + "energy.dat")
        data[:,0] *= sigma
        energyData = data[:,2]+data[:,3]
        energyShort[d,0] = np.mean(energyData[data[:,0]<1e-01])
        energyShort[d,1] = np.std(energyData[data[:,0]<1e-01])
        energyLong[d,0] = np.mean(energyData[data[:,0]>limit])
        energyLong[d,1] = np.std(energyData[data[:,0]>limit])
        ax.plot(data[:,0], energyData, linewidth=1.2, color=colorList(d/strainList.shape[0]), label="$\\gamma=$" + strainList[d], marker='o', markersize=4, fillstyle='none')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_ylabel("$Energy,$ $E$", fontsize=16)
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/energyTime" + defName + "-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6.5,5), dpi = 120)
    strain += np.ones(strain.shape[0])
    plt.errorbar(strain, energyShort[:,0], energyShort[:,1], color='g', marker='s', markersize=8, capsize=3, lw=1.2, fillstyle='none', label='$Short-time$ $response$')
    plt.errorbar(strain, energyLong[:,0], energyLong[:,1], color='k', marker='o', markersize=8, capsize=3, lw=1.2, fillstyle='none', label='$Steady$ $state$')
    print("average steady-state stress:", np.mean(energyLong[:,0]), "+-", np.std(energyLong[:,0]))
    x = strain[:5]
    y = energyShort[:5,0]
    failed = False
    try:
        popt, pcov = curve_fit(linearFit, x, y, bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        ax.plot(strain, linearFit(strain, *popt), color='b', ls='--', lw=1.5, label='$Fit,$ $ax+b:$ $a = $' + str(np.around(popt[0],3)) + ", $b = $" + str(np.around(popt[1],2)))
        print("slope (line tension):", popt[0], "intersect:", popt[1])
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Length,$ $L_y$", fontsize=16)
    ax.set_ylabel("$Energy,$ $E$", fontsize=16)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/energyStrain" + defName + "-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotEnergyStrainVSTemp(dirName, figureName, which='lj', deformation='extend'):
    fig, ax = plt.subplots(figsize = (6.5,5), dpi = 120)
    if(which=='lj'):
        tempList = np.array(['0.35', '0.36', '0.37', '0.38', '0.39', '0.40', '0.41'])
        labelName = "$T=$"
    elif(which=='active'):
        tempList = np.array(['4e-04', '3e-04', '2e-04', '1.5e-04', '1.2e-04', '1e-04', '9e-05', '7e-05', '5e-05', '4e-05'])
        #tempList = np.array(['4e-04', '3e-04', '2e-04', '1.5e-04', '1.2e-04', '1e-04'])
        labelName = "$D_r=$"
    else:
        print("Please specify the sample type")
    if(deformation=='extend'):
        strainList = np.array(['1e-02', '1.2e-02', '1.4e-02', '1.6e-02', '1.8e-02', '2e-02', '2.2e-02'])
        defName = "Ly"
    elif(deformation=='shear'):
        strainList = np.array(['1.2e-01', '1.4e-01', '1.6e-01', '1.8e-01', '2e-01', '2.2e-01', '2.4e-01'])
        defName = "LE"
    else:
        print("Please specify the deformation type")
    colorList = cm.get_cmap('viridis', tempList.shape[0])
    strain = strainList.astype(np.float64)
    temp = tempList.astype(np.float64)
    bulkShort = tempList.astype(np.float64)
    bulkLong = tempList.astype(np.float64)
    for t in range(tempList.shape[0]):
        energyShort = np.zeros((strain.shape[0],2))
        energyLong = np.zeros((strain.shape[0],2))
        for d in range(strainList.shape[0]):
            if(which=='lj'):
                dirSample = dirName + "/T" + tempList[t] + "/dynamics" + defName + strainList[d] + "-log/"
            elif(which=='active'):
                dirSample = dirName + "/Dr" + tempList[t] + "/dynamics" + defName + strainList[d] + "-log/"
            data = np.loadtxt(dirSample + os.sep + "energy.dat")
            sigma = utils.readFromDynParams(dirSample, "sigma")
            data[:,0] *= sigma
            energyData = data[:,2]+data[:,3]
            energyShort[d,0] = np.mean(energyData[data[:,0]<1e-01])
            energyShort[d,1] = np.std(energyData[data[:,0]<1e-01])
            energyLong[d,0] = np.mean(energyData[data[:,0]>1e02])
            energyLong[d,1] = np.std(energyData[data[:,0]>1e02])
        ax.errorbar(strain, energyShort[:,0], energyShort[:,1], color=colorList(t/tempList.shape[0]), marker='s', markersize=6, capsize=3, lw=1.2, fillstyle='none', label=labelName + tempList[t])
        ax.errorbar(strain, energyLong[:,0], energyLong[:,1], color=colorList(t/tempList.shape[0]), marker='s', markersize=6, capsize=3, lw=1.2)
        x = strain[:5]
        y = np.zeros((x.shape[0],2))
        y[:,0] = energyShort[:5,0]
        y[:,1] = energyLong[:5,0]
        color = ['g', 'b']
        for i in range(2):
            failed = False
            try:
                popt, pcov = curve_fit(linearFit, x, y[:,i], bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == False):
                if(i==0):
                    print("short-time bulk modulus:", popt[0], "intersect:", popt[1])
                    bulkShort[t] = popt[0]
                else:
                    print("steady state bulk modulus:", popt[0], "intersect:", popt[1])
                    bulkLong[t] = popt[0]
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Length,$ $L_y$", fontsize=16)
    ax.set_ylabel("$Energy,$ $E$", fontsize=16)
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/energyStrainTemp" + defName + "-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6,4), dpi = 120)
    ax.plot(temp, bulkShort, color='g', marker='s', markersize=8, lw=1.2, fillstyle='none', label='$Short-time$ $response$')
    ax.plot(temp, bulkLong, color='k', marker='o', markersize=8, lw=1.2, fillstyle='none', label='$Steady$ $state$')
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=16)
    ax.set_ylabel("$Line$ $tension,$ $\\gamma$", fontsize=16)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/tensionStrain" + defName + "-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotClusterEnergyVSTime(dirName, figureName, which='lj', strain=0):
    sigma = utils.readFromDynParams(dirName, 'sigma')
    fig, ax = plt.subplots(figsize = (6.5,4), dpi = 120)
    if not(os.path.exists(dirName + os.sep + 'timeEnergy.dat')):
        if(which=='lj'):
            cluster.computeClusterEnergy(dirName, threshold=0.3, strain=strain)
        elif(which=='active'):
            cluster.computeClusterEnergy(dirName, threshold=0.78, active='active', strain=strain)
        else:
            print("Please specify the sample type")
    data = np.loadtxt(dirName + os.sep + "timeEnergy.dat")
    data[:,0] *= sigma
    data[:,1:] *=sigma**2
    if(which=='lj'):
        ax.plot(data[:,0], data[:,1]+data[:,2], linewidth=1.2, color='b', label='$dense$')
        ax.plot(data[:,0], data[:,3]+data[:,4], linewidth=1.2, color='g', ls='dotted', label='$dilute$')
    elif(which=='active'):
        ax.plot(data[:,0], data[:,1]+data[:,2]+data[:,3], linewidth=1.2, color='b', label='$dense$')
        ax.plot(data[:,0], data[:,4]+data[:,5]+data[:,6], linewidth=1.2, color='g', ls='dotted', label='$dilute$')
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_ylabel("$Work,$ $w$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/workTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotClusterEnergyVSStrain(dirName, figureName, which='lj', deformation='extend', strain=0, limit=1e02):
    sigma = utils.readFromDynParams(dirName + "dynamics", 'sigma')
    fig, ax = plt.subplots(figsize = (6.5,4), dpi = 120)
    if(deformation=='extend'):
        strainList = np.array(['1e-02', '1.2e-02', '1.4e-02', '1.6e-02', '1.8e-02', '2e-02', '2.2e-02'])
        defName = "Ly"
    elif(deformation=='shear'):
        strainList = np.array(['1.2e-01', '1.4e-01', '1.6e-01', '1.8e-01', '2e-01', '2.2e-01', '2.4e-01'])
        defName = "LE"
    colorList = cm.get_cmap('viridis', strainList.shape[0])
    strain = strainList.astype(np.float64)
    energyShort = np.zeros((strain.shape[0],2))
    energyLong = np.zeros((strain.shape[0],2))
    for d in range(strainList.shape[0]):
        dirSample = dirName + "/dynamics" + defName + strainList[d] + "-log/"
        if not(os.path.exists(dirSample + os.sep + 'timeEnergy.dat')):
            if(which=='lj'):
                cluster.computeClusterEnergy(dirSample, threshold=0.3, strain=strain)
            elif(which=='active'):
                cluster.computeClusterEnergy(dirSample, threshold=0.78, active='active', strain=strain)
        data = np.loadtxt(dirSample + os.sep + "timeEnergy.dat")
        data[:,0] *= sigma
        data = data[data[:,0]<1e03,:]
        if(which=='lj'):
            energyData = data[:,1]+data[:,2]
            energyShort[d,0] = np.mean(energyData[data[:,0]<2e-01])
            energyShort[d,1] = np.std(energyData[data[:,0]<2e-01])
        elif(which=='active'):
            energyData = data[:,1]+data[:,2]+data[:,3]
            rightEnergy = energyData[data[:,0]<0.5]
            rightTime = data[data[:,0]<0.5,0]
            energyShort[d,0] = np.mean(rightEnergy[rightTime>0.25])
            energyShort[d,1] = np.std(rightEnergy[rightTime>0.25])
        energyLong[d,0] = np.mean(energyData[data[:,0]>limit])
        energyLong[d,1] = np.std(energyData[data[:,0]>limit])
        ax.plot(data[:,0], energyData, linewidth=1.2, color=colorList(d/strainList.shape[0]), label="$\\gamma=$" + strainList[d], marker='o', markersize=4, fillstyle='none')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_ylabel("$Work,$ $w$", fontsize=16)
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/energy" + defName + "-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6,4.5), dpi = 120)
    strain += 1
    plt.errorbar(strain, energyShort[:,0], energyShort[:,1], color='g', marker='s', markersize=8, capsize=3, lw=1.2, fillstyle='none', label='$Short-time$ $response$')
    plt.errorbar(strain, energyLong[:,0], energyLong[:,1], color='k', marker='o', markersize=8, capsize=3, lw=1.2, fillstyle='none', label='$Steady-state$')
    x = strain[:5]
    y = energyShort[:5,0]
    failed = False
    try:
        popt, pcov = curve_fit(linearFit, x, y, bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        #ax.plot(strain, linearFit(strain, *popt), color='b', ls='--', lw=1.5, label='$Fit,$ $ax+b:$ $a = $' + str(np.around(popt[0],3)) + ", $b = $" + str(np.around(popt[1],2)))
        print("slope (line tension):", popt[0], "intersect:", popt[1])
    ax.legend(fontsize=11, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Length,$ $L_y$", fontsize=16)
    ax.set_ylabel("$Work,$ $w$", fontsize=16)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/tension" + defName + "-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

################################# shear graphics ###############################
    if(whichPlot == "stressstrain"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        strain = float(sys.argv[5])
        plotStressStrain(dirName, figureName, which, strain)

    elif(whichPlot == "stresstime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        strain = float(sys.argv[5])
        logx = sys.argv[6]
        plotStressTime(dirName, figureName, which, strain, logx)

    elif(whichPlot == "shearstress"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        limit = float(sys.argv[5])
        plotStressTimeVSStrain(dirName, figureName, which, limit)

    elif(whichPlot == "bulkstress"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        limit = float(sys.argv[5])
        plotStressTimeVSExtension(dirName, figureName, which, limit)

    elif(whichPlot == "stresstemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        deformation = sys.argv[5]
        plotStressStrainVSTemp(dirName, figureName, which, deformation)

    elif(whichPlot == "shearenergy"):
        figureName = sys.argv[3]
        logx = sys.argv[4]
        plotStressEnergyTime(dirName, figureName, logx)

    elif(whichPlot == "energystrain"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        deformation = sys.argv[5]
        limit = float(sys.argv[6])
        plotEnergyVSStrain(dirName, figureName, which, deformation, limit)

    elif(whichPlot == "energytemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        deformation = sys.argv[5]
        plotEnergyStrainVSTemp(dirName, figureName, which, deformation)

    elif(whichPlot == "clusterenergy"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        strain = float(sys.argv[5])
        plotClusterEnergyVSTime(dirName, figureName, which, strain)

    elif(whichPlot == "clusterestrain"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        deformation = sys.argv[5]
        strain = float(sys.argv[6])
        limit = float(sys.argv[7])
        plotClusterEnergyVSStrain(dirName, figureName, which, deformation, strain, limit)
