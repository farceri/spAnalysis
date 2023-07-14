'''
Created by Francesco
12 October 2021
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy import interpolate
from scipy.interpolate import splev, splrep
import scipy.stats as st
import itertools
import sys
import os
import spCorrelation as spCorr
import utilsCorr as ucorr
import utilsPlot as uplot
import visuals

def plotSPCorr(ax, x, y, ylabel, color, legendLabel = None, logx = True, logy = False, linestyle = 'solid', alpha=1):
    ax.plot(x, y, linewidth=1., color=color, linestyle = linestyle, label=legendLabel, alpha=alpha)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel(ylabel, fontsize=18)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')

########################## nve and langevin comparison #########################
def plotEnergy(dirName, figureName):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    energy = np.loadtxt(dirName + os.sep + "energy.dat")
    print("temperature:", np.mean(energy[:,4]), ", energy ratio:", np.mean(energy[:,2]/energy[:,3]))
    fig = plt.figure(figsize = (7, 5), dpi = 120)
    ax = fig.gca()
    ax.plot(energy[:,0], energy[:,2]/numParticles, linewidth=1.5, color='k')
    ax.plot(energy[:,0], energy[:,3]/numParticles, linewidth=1.5, color='r', linestyle='--')
    ax.plot(energy[:,0], (energy[:,2] + energy[:,3])/numParticles, linewidth=1.5, color='b', linestyle='dotted')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $step$", fontsize=16)
    ax.set_ylabel("$Energy$", fontsize=16)
    ax.legend(("$E_{pot}$", "$E_{kin}$", "$E_{tot}$"), fontsize=15, loc=(0.75, 0.45))
    #ax.set_ylim(50, 700)
    #ax.set_yscale('log')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/penergy-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## nve and langevin comparison #########################
def compareEnergy(dirName1, dirName2, figureName):
    fig, ax = plt.subplots(1, 2, sharey = True, figsize = (12, 5), dpi = 120)
    # first sample
    numParticles = int(ucorr.readFromParams(dirName1, "numParticles"))
    energy = np.loadtxt(dirName1 + os.sep + "energy.dat")
    ax[0].plot(energy[:,0], energy[:,2]/numParticles, linewidth=1.5, color='k')
    ax[0].plot(energy[:,0], energy[:,3]/numParticles, linewidth=1.5, color='r', linestyle='--')
    ax[0].plot(energy[:,0], (energy[:,2] + energy[:,3])/numParticles, linewidth=1.5, color='b', linestyle='dotted')
    # second sample
    numParticles = int(ucorr.readFromParams(dirName2, "numParticles"))
    energy = np.loadtxt(dirName2 + os.sep + "energy.dat")
    ax[1].plot(energy[:,0], energy[:,2]/numParticles, linewidth=1.5, color='k')
    ax[1].plot(energy[:,0], energy[:,3]/numParticles, linewidth=1.5, color='r', linestyle='--')
    ax[1].plot(energy[:,0], (energy[:,2] + energy[:,3])/numParticles, linewidth=1.5, color='b', linestyle='dotted')
    ax[0].set_yscale('log')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=16)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=16)
    ax[0].set_ylabel("$Energy$", fontsize=16)
    ax[0].legend(("$E_{pot}$", "$E_{kin}$", "$E_{tot}$"), fontsize=15, loc=(0.75, 0.45))
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pCompareEnergy-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## nve and langevin comparison #########################
def plotEnergyVSSystemSize(dirName, whichDir, figureName):
    dirList = np.array(["1024", "2048", "4096", "8192", "16384"])
    mean = np.zeros((dirList.shape[0], 3))
    error = np.zeros((dirList.shape[0], 3))
    num = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/" + whichDir
        data = np.loadtxt(dirSample + "/energy.dat")
        num[d] = int(ucorr.readFromParams(dirSample, "numParticles"))
        mean[d,0] = np.mean(data[:,2]/num[d])
        error[d,0] = np.std(data[:,2]/num[d])
        mean[d,1] = np.mean(data[:,3]/num[d])
        error[d,1] = np.std(data[:,3]/num[d])
        mean[d,2] = np.mean((data[:,2] + data[:,3])/num[d])
        error[d,2] = np.std((data[:,2] + data[:,3])/num[d])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    ax.errorbar(num, mean[:,0], error[:,0], linewidth=1.5, marker='o', markersize=10, fillstyle='none', capsize=3, color='k')
    ax.errorbar(num, mean[:,1], error[:,1], linewidth=1.5, marker='v', markersize=10, fillstyle='none', capsize=3, color='r', linestyle='--')
    ax.errorbar(num, mean[:,2], error[:,2], linewidth=1.5, marker='s', markersize=10, fillstyle='none', capsize=3, color='b', linestyle='dotted')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xscale('log', basex=2)
    #ax.set_yscale('log')
    ax.set_xticks([1024, 2048, 4096, 8192])
    ax.set_xticklabels(["1024", "2048", "4096", "8192", "16384"])
    ax.set_xlabel("$System$ $size,$ $N$", fontsize=16)
    ax.set_ylabel("$Energy$", fontsize=16)
    ax.legend(("$E_{pot}$", "$E_{kin}$", "$E_{tot}$"), fontsize=15, loc='best')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pEnergyVSsystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## nve and langevin comparison #########################
def plotPressure(dirName, figureName, bound=False, prop=False):
    sigma = np.mean(np.loadtxt(dirName + "/particleRad.dat"))
    pressure = np.loadtxt(dirName + os.sep + "pressure.dat")
    pressure[:,1:] *= sigma**2
    fig = plt.figure(figsize = (7, 5), dpi = 120)
    ax = fig.gca()
    if(bound == "bound"):
        ax.plot(pressure[:,0], pressure[:,1], linewidth=1.2, color='g', label="$wall$")
    ax.plot(pressure[:,0], pressure[:,2], linewidth=1.5, color='k', linestyle='dashdot', label="$virial$")
    ax.plot(pressure[:,0], pressure[:,3], linewidth=1.5, color='r', linestyle='--', label="$thermal$")
    ptot = pressure[:,2] + pressure[:,3]
    if(prop == "prop"):
        ax.plot(pressure[:,0], pressure[:,4], linewidth=1.5, color=[1,0.5,0], label="$active$")
        ptot += pressure[:,4]
    ax.plot(pressure[:,0], ptot, linewidth=1.5, color='b', linestyle='dotted', label="$total$")
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $step$", fontsize=18)
    ax.set_ylabel("$Pressure,$ $P \\sigma^2$", fontsize=18)
    ax.legend(fontsize=12, loc='best')
    ax.set_ylim(-0.00058, 0.082)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pPressure-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## nve and langevin comparison #########################
def comparePressure(dirName1, dirName2, figureName):
    fig, ax = plt.subplots(1, 2, sharey = True, figsize = (12, 5), dpi = 120)
    # first sample
    energy = np.loadtxt(dirName1 + os.sep + "pressure.dat")
    ax[0].plot(energy[:,0], energy[:,2], linewidth=1.5, color='k')
    ax[0].plot(energy[:,0], energy[:,3], linewidth=1.5, color='r', linestyle='--')
    ax[0].plot(energy[:,0], energy[:,4], linewidth=1.5, color=[1,0.5,0], linestyle='dashdot')
    ax[0].plot(energy[:,0], (energy[:,2] + energy[:,3] + energy[:,4]), linewidth=1.5, color='b', linestyle='dotted')
    # second sample
    energy = np.loadtxt(dirName2 + os.sep + "pressure.dat")
    ax[1].plot(energy[:,0], energy[:,1], linewidth=1.5, color='k')
    ax[1].plot(energy[:,0], energy[:,2], linewidth=1.5, color='r', linestyle='--')
    ax[1].plot(energy[:,0], (energy[:,1] + energy[:,2]), linewidth=1.5, color='b', linestyle='dotted')
    ax[0].set_yscale('log')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=16)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=16)
    ax[0].set_ylabel("$Stress$", fontsize=16)
    ax[0].legend(("$S_{int}$", "$S_{kin}$", "$S_{active}$", "$S_{tot}$"), fontsize=15, loc=(0.7, 0.4))
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pComparePressure-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotAveragePressure(dirName, figureName, bound=False, prop=False, fixed='Dr'):
    if(fixed == 'temp'):
        dirList = np.array(["0.75", "0.25", "0.1", "0.06", "0.03", "0.01", "0.006", "0.003", "0.001"])
    elif(fixed == 'Dr'):
        dirList = np.array(["3", "6", "10", "30", "60", "100", "300", "600", "1000"])
    elif(fixed == 'f0'):
        dirList = np.array(["10000", "1000", "100", "10", "1", "1e-01", "1e-02", "1e-03", "1e-04"])
    T = np.zeros((dirList.shape[0],2))
    if(bound == "bound"):
        wall = np.zeros((dirList.shape[0],2))
    virial = np.zeros((dirList.shape[0],2))
    thermal = np.zeros((dirList.shape[0],2))
    total = np.zeros((dirList.shape[0],2))
    if(prop == "prop"):
        active = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(fixed == 'temp'):
            dirSample = dirName + os.sep + "T" + dirList[d] + "/dynamics/"
        elif(fixed == 'Dr'):
            dirSample = dirName + os.sep + "Dr1-f0" + dirList[d] + "/dynamics/"
        temp = np.loadtxt(dirSample + "energy.dat")[:,4]
        sigma = np.mean(np.loadtxt(dirSample + "/particleRad.dat"))
        T[d,0] = np.mean(temp)
        T[d,1] = np.std(temp)
        data = np.loadtxt(dirSample + "pressure.dat")
        data[:,1:] *= sigma**2
        if(bound == "bound"):
            wall[d,0] = np.mean(data[:,1])
            wall[d,1] = np.std(data[:,1])#/np.sqrt(data.shape[0])
        virial[d,0] = np.mean(data[:,2])
        virial[d,1] = np.std(data[:,2])
        thermal[d,0] = np.mean(data[:,3])
        thermal[d,1] = np.std(data[:,3])
        if(prop == "prop"):
            active[d,0] = np.mean(data[:,4])
            active[d,1] = np.std(data[:,4])
            total[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4])
            total[d,1] = np.std(data[:,2] + data[:,3] + data[:,4])
        else:
            total[d,0] = np.mean(data[:,2] + data[:,3])
            total[d,1] = np.std(data[:,2] + data[:,3])
    fig = plt.figure(figsize = (7, 5), dpi = 120)
    ax = fig.gca()
    if(bound == "bound"):
        ax.errorbar(T[:,0], wall[:,0], wall[:,1], color='g', marker='s', markersize=12, fillstyle='none', capsize=3, label="$wall$")
    ax.errorbar(T[:,0], virial[:,0], virial[:,1], color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label="$virial$")
    ax.errorbar(T[:,0], thermal[:,0], thermal[:,1], color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label="$thermal$")
    if(prop == "prop"):
        ax.errorbar(T[:,0], active[:,0], active[:,1], color=[1,0.5,0], marker='v', markersize=8, fillstyle='none', capsize=3, label="$active$")
    ax.errorbar(T[:,0], total[:,0], total[:,1], color='b', marker='d', markersize=8, fillstyle='none', capsize=3, label="$total$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Propulsion,$ $f_0$", fontsize=18)
    ax.set_ylabel("$Pressure,$ $P \\sigma^2$", fontsize=18)
    #ax.set_ylim(4.1e-05, 0.28)
    ax.set_ylim(5.1e-06, 1.6)
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pAvPressure-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## nve and langevin comparison #########################
def plotDropletPressureVSTime(dirName, figureName):
    pressure = np.loadtxt(dirName + os.sep + "pressure.dat")
    fig, ax = plt.subplots(1, 2, sharey=True, figsize = (11, 5), dpi = 120)
    # droplet particles
    ax[0].plot(pressure[:,0], pressure[:,1], linewidth=1.5, color='k', label="$virial$")
    ax[0].plot(pressure[:,0], pressure[:,2], linewidth=1.5, color='r', label="$thermal$")
    ptot = (pressure[:,1] + pressure[:,2])
    ax[0].plot(pressure[:,0], ptot, linewidth=1.5, color='b', ls='dotted', label="$total$")
    # dilute particles
    ax[1].plot(pressure[:,0], pressure[:,4], linewidth=1.5, color='k', ls='--', label="$virial$")
    ax[1].plot(pressure[:,0], pressure[:,5], linewidth=1.5, color='r', ls='--', label="$thermal$")
    ptot = (pressure[:,4] + pressure[:,5])
    ax[1].plot(pressure[:,0], ptot, linewidth=1.5, color='b', ls='dotted', label="$total$")
    # plotting settings
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[0].set_ylabel("$Pressure$", fontsize=15)
    ax[0].legend(fontsize=12, loc='best')
    ax[0].set_yscale('log')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    #figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterPressure-" + figureName
    #fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## nve and langevin comparison #########################
def plotClusterPressureVSTime(dirName, figureName, bound=False, prop=False):
    #numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    pressure = np.loadtxt(dirName + os.sep + "clusterPressure.dat")
    fig, ax = plt.subplots(1, 2, sharey=True, figsize = (11, 5), dpi = 120)
    if(bound == "bound"):
        ax[0].plot(pressure[:,0], pressure[:,1], linewidth=1.5, color='g', label="$wall$")
        ax[1].plot(pressure[:,0], pressure[:,1], linewidth=1.5, color='g', label="$wall$")
    # dense particles
    ax[0].plot(pressure[:,0], pressure[:,2], linewidth=1.5, color='k', label="$virial$")
    ax[0].plot(pressure[:,0], pressure[:,3], linewidth=1.5, color='r', label="$thermal$")
    ptot = pressure[:,2] + pressure[:,3]
    if(prop == "prop"):
        ax[0].plot(pressure[:,0], pressure[:,4], linewidth=1.5, color=[1,0.5,0], label="$active$")
        ptot += pressure[:,4]
    ax[0].plot(pressure[:,0], ptot, linewidth=1.5, color='b', ls='dotted', label="$total$")
    # dilute particles
    ax[1].plot(pressure[:,0], pressure[:,5], linewidth=1.5, color='k', ls='--', label="$virial$")
    ax[1].plot(pressure[:,0], pressure[:,6], linewidth=1.5, color='r', ls='--', label="$thermal$")
    ptot = pressure[:,5] + pressure[:,6]
    if(prop == "prop"):
        ax[1].plot(pressure[:,0], pressure[:,7], linewidth=1.5, color=[1,0.5,0], ls='--', label="$active$")
        ptot += pressure[:,7]
    ax[1].plot(pressure[:,0], ptot, linewidth=1.5, color='b', ls='dotted', label="$total$")
    # plotting settings
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[0].set_ylabel("$Pressure$", fontsize=15)
    ax[0].legend(fontsize=12, loc='best')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    #figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterPressure-" + figureName
    #fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSimplexDensity(dirName, figureName, pad = 1, logy=False):
    if(os.path.exists(dirName + os.sep + 'simplexDensity.dat')):
        simplexDensity = np.loadtxt(dirName + os.sep + 'simplexDensity.dat')
    else:
        _, simplexDensity = spCorr.computeDelaunayCluster(dirName, threshold=0.84, filter=True)
    denseSimplexList = np.loadtxt(dirName + os.sep + 'denseSimplexList.dat')
    #simplexDensity = simplexDensity[denseSimplexList==1]
    fig, ax = plt.subplots(1, 2, figsize=(9,4), dpi=150)
    ax[0].plot(np.arange(1, simplexDensity.shape[0]+1, 1), np.sort(simplexDensity), color='k', marker='.', markersize=8, lw=0.8, fillstyle='none')
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel('$Simplex$ $index$', fontsize=16)
    ax[0].set_ylabel('$\\varphi^{Simplex}$', fontsize=16)
    numBins = 100
    pdf, edges = np.histogram(simplexDensity, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax[1].plot(edges, pdf, color='k', marker='.', markersize=8, lw=0.8, fillstyle='none')
    y = np.linspace(np.min(pdf)-pad, np.max(pdf)+pad, 100)
    if(logy == 'logy'):
        ax[1].set_yscale('log')
        #ax[1].set_ylim(np.min(pdf), np.max(pdf)+pad/2)
        ax[1].set_ylim(6.4e-03, 50.6)
        y = np.linspace(1e-04, 100, 100)
    else:
        #ax[1].set_ylim(np.min(pdf)-pad/2, np.max(pdf)+pad/2)
        ax[1].set_ylim(-2.8, 43.2)
        y = np.linspace(-5, 50, 100)
    ax[1].plot(np.ones(100)*0.906899682, y, ls='dotted', color='k', lw=1, label='$Triangular$ $lattice$')
    ax[1].plot(np.ones(100)*0.785398163, y, ls='dashdot', color='k', lw=1, label='$Square$ $lattice$')
    ax[1].legend(fontsize=10, loc='best')
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].set_ylabel('$PDF(\\varphi^{Simplex})$', fontsize=16)
    ax[1].set_xlabel('$\\varphi^{Simplex}$', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    plt.savefig("/home/francesco/Pictures/soft/DelaunayDensityPDF" + figureName + ".png", transparent=True, format="png")
    plt.show()

def plotParticleForces(dirName, index0, index1, index2, dim):
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    force0 = []
    force1 = []
    force2 = []
    for d in range(dirList.shape[0]):
        force = np.loadtxt(dirName + os.sep + dirList[d] + "/particleForces.dat")
        force0.append(force[index0,dim])
        force1.append(force[index1,dim])
        force2.append(force[index2,dim])
    fig = plt.figure(figsize = (7, 5), dpi = 120)
    ax = fig.gca()
    ax.plot(timeList, force0, linewidth=1, color='k', marker='o', fillstyle='none', label="$vertex$" + " " + str(index0))
    ax.plot(timeList, force1, linewidth=1, color='b', marker='o', fillstyle='none', label="$vertex$" + " " + str(index1))
    ax.plot(timeList, force2, linewidth=1, color='g', marker='o', fillstyle='none', label="$vertex$" + " " + str(index2))
    ax.legend(fontsize=10, loc='lower right')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax.set_ylabel("$Forces$", fontsize=15)
    plt.tight_layout()
    plt.show()

def plotActiveEnergy(dirName, figureName):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    energy = np.loadtxt(dirName + os.sep + "energy.dat")
    fig = plt.figure(figsize = (7, 5), dpi = 120)
    ax = fig.gca()
    energy[:,5] -= numParticles*energy[:,4]
    #ax.plot(energy[:,0], (energy[:,4]-np.mean(energy[:,4]))**2/np.mean(energy[:,4])**2, linewidth=1, color='k', marker='o', fillstyle='none')
    #ax.plot(energy[:,0], (energy[:,5]-np.mean(energy[:,5]))**2/np.mean(energy[:,5])**2, linewidth=1, color='b', marker='s', fillstyle='none')
    #ax.plot(energy[:,0], (energy[:,6]-np.mean(energy[:,6]))**2/np.mean(energy[:,6])**2, linewidth=1, color='g', marker='v', fillstyle='none')
    ax.plot(energy[:,0], energy[:,4], linewidth=1, color='k', marker='o', fillstyle='none')
    ax.plot(energy[:,0], energy[:,5], linewidth=1, color='b', marker='s', fillstyle='none')
    ax.plot(energy[:,0], energy[:,6], linewidth=1, color='g', marker='v', fillstyle='none')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    #ax.set_ylabel("$(p - \\langle p \\rangle)^2 \\, / \\, \\langle p \\rangle^2$", fontsize=15)
    ax.set_ylabel("$Pressure$", fontsize=15)
    ax.legend(("$\\rho k_B T$", "$p_{virial}$", "$p_{active}$"), fontsize=12, loc='lower left')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pactiveEnergy-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareEnergyVSPhi(dirName, sampleName, figureName):
    phiList = np.array(['45', '58', '67', '72', '78', '83', '85', '88'])
    phi = np.array([0.45, 0.58, 0.67, 0.72, 0.78, 0.83, 0.85, 0.88])
    fig = plt.figure(figsize = (7, 5), dpi = 120)
    ax = fig.gca()
    colorList = cm.get_cmap('winter', dirList.shape[0])
    for d in range(dirList.shape[0]):
        deltapot = []
        deltakin = []
        ratio = []
        stdratio = []
        for p in phiList:
            dirSample = dirName + "/thermal" + p + "/langevin/T" + sampleName + "/" + dirList[d] + "/dynamics/"
            if(os.path.exists(dirSample + os.sep + "energy.dat")):
                energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                #ax.plot(energy[:,0], energy[:,2], linewidth=1.5, color='k')
                #ax.plot(energy[:,0], energy[:,3], linewidth=1.5, color='r', linestyle='--')
                #plt.pause(0.5)
                deltapot.append(np.mean((energy[:,2]-np.mean(energy[:,2]))**2)/np.mean(energy[:,2])**2)
                deltakin.append(np.mean((energy[:,3]-np.mean(energy[:,3]))**2)/np.mean(energy[:,3])**2)
                ratio.append(np.mean(energy[:,2]/energy[:,3]))
                stdratio.append(np.std(energy[:,2]/energy[:,3]))
        if(d==dirList.shape[0]-1):
            color = 'k'
        else:
            color = colorList((dirList.shape[0]-d)/dirList.shape[0])
        ax.errorbar(phi, ratio, stdratio, color=color, label=labelList[d], marker='o', fillstyle='none', capsize=3)
        #ax.plot(phi, deltakin, color=color, label=labelList[d], marker='o', fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=18)
    ax.set_ylabel("$Energy$ $ratio,$ $E_{pot}/E_{kin}$", fontsize=18)
    ax.legend(fontsize=12, loc='best')
    ax.set_yscale('log')
    ax.set_ylim(0.0033, 42)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/penergyPhi-" + figureName + "-T" + sampleName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareEnergyVSActivity(dirName, sampleName, figureName):
    damping = 623
    sigma = 2*np.mean(np.array(np.loadtxt(dirName + os.sep + "../particleRad.dat")))
    f0List = np.array(['1e-01', '2e-01', '3e-01', '5e-01', '1', '2', '3', '5', '10', '20', '30', '50', '100', '200', '300', '500', '1000'])
    Pe = []
    epot = []
    stdpot = []
    ekin = []
    stdkin = []
    fig = plt.figure(figsize = (7, 5), dpi = 120)
    ax = fig.gca()
    for i in range(f0List.shape[0]):
        dirSample = dirName + "/Dr1-f0" + f0List[i] + "/dynamics/"
        if(os.path.exists(dirSample + os.sep + "energy.dat")):
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            epot.append(np.mean(energy[:,2]))
            stdpot.append(np.std(energy[:,2]))
            ekin.append(np.mean(energy[:,3]))
            stdkin.append(np.std(energy[:,3]))
            Pe.append(float(f0List[i])/(damping*sigma))
    ax.errorbar(Pe, epot, stdpot, color='k', label='$Potential$ $energy$', marker='o', fillstyle='none', capsize=3)
    ax.errorbar(Pe, ekin, stdkin, color='r', label='$Kinetic$ $energy$', marker='o', fillstyle='none', capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Peclet$ $number,$ $Pe = v_0/D_r \\sigma$", fontsize=18)
    ax.set_ylabel("$Energy$", fontsize=18)
    ax.legend(fontsize=12, loc='best')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/penergyf0-" + figureName + "-T" + sampleName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPFourierEnergy(dirName, fileName, figureName, dyn = "nve", fixed = "temp", which = "0.001"):
    fig, ax = plt.subplots(2, 2, sharey='row', sharex='col', figsize = (9, 6), dpi = 120)
    if(fixed == "temp"):
        dirList = np.array(['thermal45', 'thermal58', 'thermal67', 'thermal72', 'thermal78', 'thermal83', 'thermal85', 'thermal88'])
        labelList = np.array(['$\\varphi=0.45$', '$\\varphi=0.58$', '$\\varphi=0.67$','$\\varphi=0.72$', '$\\varphi=0.78$', '$\\varphi=0.83$', '$\\varphi=0.85$', '$\\varphi=0.88$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
    elif(fixed == "phi"):
        dirList = np.array(['T0.001', 'T0.005', 'T0.01', 'T0.05', 'T0.1'])
        labelList = np.array(['$T=0.001$', '$T=0.005$','$T=0.01$', '$T=0.05$', '$T=0.1$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    else:
        dirList = np.array(['iod1000', 'iod100', 'iod10', 'iod1', 'iod0.1', 'iod0.01', 'iod0.001', 'nve'])
        labelList = np.array(['$\\beta \\sigma = 1000$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 1$', '$\\beta \\sigma = 0.1$', '$\\beta \\sigma = 0.01$', '$\\beta \\sigma = 0.001$', '$NVE$'])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed == "temp"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T" + which + "/" + dyn + "/dynamics/"
        elif(fixed == "phi"):
            dirSample = dirName + os.sep + "thermal" + which + "/langevin/" + dirList[d] + "/" + dyn + "/dynamics/"
        else:
            dirSample = dirName + os.sep + dirList[d] + "/dynamics/"
        if(os.path.exists(dirSample + fileName + "FourierEnergy.dat")):
            data = np.loadtxt(dirSample + fileName + "FourierEnergy.dat")
            #if(d==dirList.shape[0]-1):
            #    color = 'k'
            #else:
            color = colorList((dirList.shape[0]-d)/dirList.shape[0])
            cumkin = 2*np.sum(data[data[:,0]>0,0]*data[data[:,0]>0,1])
            cumpot = 2*np.sum(data[data[:,0]>0,0]*data[data[:,0]>0,2])
            ax[0,0].semilogy(data[:,0], data[:,1]/cumkin, color=color, lw=1, label=labelList[d])
            ax[1,0].semilogy(data[:,0], data[:,2]/cumpot, color=color, lw=1, label=labelList[d])
            ax[0,1].loglog(data[data[:,0]>0,0], data[data[:,0]>0,1]/cumkin, color=color, lw=1, label=labelList[d])
            ax[1,1].loglog(data[data[:,0]>0,0], data[data[:,0]>0,2]/cumpot, color=color, lw=1, label=labelList[d])
    ax[1,1].legend(fontsize=10, loc='best')
    x = np.linspace(-2e03, 2e03, 1000)
    gamma = 1
    y = gamma**2/(np.pi*gamma*(x**2+gamma**2))
    ax[0,0].plot(x, y, color='k', lw=1.2, linestyle='dashed')
    ax[0,1].plot(x[x>0], y[x>0], color='k', lw=1.2, linestyle='dashed')
    #ax[1,0].plot(x, y, color='k', lw=1.2, linestyle='dashed')
    #ax[1,1].plot(x[x>0], y[x>0], color='k', lw=1.2, linestyle='dashed')
    ax[0,0].tick_params(axis='both', labelsize=14)
    ax[0,1].tick_params(axis='both', labelsize=14)
    ax[1,0].tick_params(axis='both', labelsize=14)
    ax[1,1].tick_params(axis='both', labelsize=14)
    ax[1,0].set_xlabel("$\\omega$", fontsize=18)
    ax[1,1].set_xlabel("$\\omega$", fontsize=18)
    ax[0,0].set_ylabel("$\\tilde{K}(\\omega)/N$", fontsize=18)
    ax[1,0].set_ylabel("$\\tilde{U}(\\omega)/N$", fontsize=18)
    #ax[0,0].set_ylabel("$C_{KK}(\\omega)$", fontsize=18)
    #ax[1,0].set_ylabel("$C_{UU}(\\omega)$", fontsize=18)
    #ax[1,0].set_xlim(-1.2, 1.2)
    #ax[1,0].set_xlim(-22, 22)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pcompare-" + fileName + "Fourier-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPFourierCorr(dirName, fileName, figureName, dyn = "nve", fixed = "temp", which = "0.001"):
    fig, ax = plt.subplots(2, 1, sharey='row', sharex='col', figsize = (7, 5), dpi = 120)
    if(fixed == "temp"):
        dirList = np.array(['thermal45', 'thermal58', 'thermal67', 'thermal72', 'thermal78', 'thermal83', 'thermal85', 'thermal88'])
        labelList = np.array(['$\\varphi=0.45$', '$\\varphi=0.58$', '$\\varphi=0.67$','$\\varphi=0.72$', '$\\varphi=0.78$', '$\\varphi=0.83$', '$\\varphi=0.85$', '$\\varphi=0.88$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
    elif(fixed == "phi"):
        dirList = np.array(['T0.001', 'T0.005', 'T0.01', 'T0.05', 'T0.1'])
        labelList = np.array(['$T=0.001$', '$T=0.005$','$T=0.01$', '$T=0.05$', '$T=0.1$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    else:
        dirList = np.array(['iod1000', 'iod100', 'iod10', 'iod1', 'iod0.1', 'iod0.01', 'iod0.001', 'nve'])
        labelList = np.array(['$\\beta \\sigma = 1000$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 1$', '$\\beta \\sigma = 0.1$', '$\\beta \\sigma = 0.01$', '$\\beta \\sigma = 0.001$', '$NVE$'])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed == "temp"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T" + which + "/" + dyn + "/dynamics/"
        elif(fixed == "phi"):
            dirSample = dirName + os.sep + "thermal" + which + "/langevin/" + dirList[d] + "/" + dyn + "/dynamics/"
        else:
            dirSample = dirName + os.sep + dirList[d] + "/dynamics/"
        if(os.path.exists(dirSample + fileName + "FourierEnergy.dat")):
            minRad = np.min(np.loadtxt(dirSample + "/particleRad.dat"))
            timeStep = ucorr.readFromParams(dirSample, "dt")
            data = np.loadtxt(dirSample + fileName + "FourierEnergy.dat")
            color = colorList((dirList.shape[0]-d)/dirList.shape[0])
            qmax = 2*np.pi/minRad
            ax[0].loglog(data[data[:,0]>0,0]/qmax, data[data[:,0]>0,4], color=color, lw=1, label=labelList[d])
            ax[1].loglog(data[data[:,0]>0,0]/qmax, data[data[:,0]>0,5], color=color, lw=1, label=labelList[d])
    ax[0].legend(fontsize=10, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$q$", fontsize=18)
    ax[0].set_ylabel("$C_{vv}^{\\parallel}(q)$", fontsize=18)
    ax[1].set_ylabel("$C_{vv}^{\\perp}(q)$", fontsize=18)
    #ax[1,0].set_xlim(-1.2, 1.2)
    #ax[1,0].set_xlim(-22, 22)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pcompareCorr-" + fileName + "Fourier-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def curvePL(x, a, b):
    return (a*x)**b

def curveCvv(x, a, b):
    return a * np.exp(-b*x)

def curveCvvOsc(x, a, b, c, d):
    return a * np.exp(-b*x) * np.cos(c*x + d)

def plotSPCollision(dirName, figureName, scaled, dyn = "nve", fixed = "temp", which = "0.001"):
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    if(fixed == "temp"):
        dirList = np.array(['thermal45', 'thermal58', 'thermal67', 'thermal72', 'thermal78', 'thermal80', 'thermal83', 'thermal85', 'thermal88', 'thermal94', 'thermal1'])
        labelList = np.array(['$\\varphi=0.45$', '$\\varphi=0.58$', '$\\varphi=0.67$', '$\\varphi=0.72$', '$\\varphi=0.78$', '$\\varphi=0.80$', '$\\varphi=0.83$', '$\\varphi=0.85$', '$\\varphi=0.88$', '$\\varphi=0.94$', '$\\varphi=1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = []
    elif(fixed == "phi"):
        dirList = np.array(['T0.001', 'T0.002', 'T0.003', 'T0.005', 'T0.01', 'T0.02', 'T0.03', 'T0.05', 'T0.1'])
        labelList = np.array(['$T=0.001$', '$T=0.002$', '$T=0.003$', '$T=0.005$','$T=0.01$', '$T=0.02$', '$T=0.03$', '$T=0.05$', '$T=0.1$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        Temp = []
    elif(fixed=="f0"):
        f0 = which
        iod = '10'
        dirList = np.array(['1e-04', '5e-04', '1e-03', '5e-03', '1e-02', '5e-02', '1e-01', '5e-01', '1', '2', '5', '10', '50', '100', '1000'])
        labelList = np.array(['$D_r = 0.0001$', '$D_r = 0.0005$', '$D_r = 0.001$', '$D_r = 0.005$', '$D_r = 0.01$', '$D_r = 0.05$', '$D_r = 0.1$', '$D_r = 0.5$', '$D_r = 1$', '$D_r = 2$', '$D_r = 5$', '$D_r = 10$', '$D_r = 50$', '$D_r = 100$', '$D_r = 1000$'])
        Dr = []
        taup = []
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    else:
        dirList = np.array(['nve', 'iod0.001', 'iod0.01', 'iod0.1', 'iod1', 'iod10', 'iod100', 'iod1000'])
        iod = np.array([0, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
        meanRad = np.mean(np.loadtxt(dirName + "/particleRad.dat"))
        labelList = np.array(['$NVE$', '$\\beta \\sigma = 0.001$', '$\\beta \\sigma = 0.01$', '$\\beta \\sigma = 0.1$', '$\\beta \\sigma = 1$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 1000$'])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = []
    tauc = []
    for d in range(dirList.shape[0]):
        if(fixed == "temp"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T" + which + "/" + dyn + "/active-langevin/Dr1e-03-f0200/dynamics-col/"
            color = colorList(d/dirList.shape[0])
        elif(fixed == "phi"):
            dirSample = dirName + os.sep + "thermal" + which + "/langevin/" + dirList[d] + "/" + dyn + "/dynamics/"
            color = colorList(d/dirList.shape[0])
        elif(fixed == "f0"):
            if(dirList[d] == dirList[-1]):
                dirSample = dirName + dirList[d] + "/dynamics-col/"
            else:
                dirSample = dirName + "/Dr" + dirList[d] + "-f0" + f0 + "/dynamics-col/"
                color = colorList(d/dirList.shape[0])
        else:
            dirSample = dirName + os.sep + dirList[d] + "/dynamics/"
        if(os.path.exists(dirSample + "/contactCollision.dat")):
            if(dirList[d] == dirList[-1]):
                data = np.loadtxt(dirSample + "/contactCollision.dat")
            else:
                data = np.loadtxt(dirSample + "/contactCollision.dat")#cluster
            data = data[data[:,1]>0]
            if((fixed != "temp" and fixed != "phi") and fixed != "f0"):
                if(d == 0):
                    color = 'k'
            failed = False
            try:
                popt, pcov = curve_fit(curvePL, data[:10,0], data[:10,1])
            except RuntimeError:
                print("Error - Power law fit failed")
                failed = True
            if(failed == False):
                tauc.append(1/popt[0])
                ax.semilogy(data[:-20,0], curvePL(data[:-20,0], *popt), color=color, lw=1, linestyle='--')
                print("Power law fit ", dirList[d], " timescale: ", tauc[-1])
            else:
                failed = False
                try:
                    popt, pcov = curve_fit(curveCvv, data[:10,0], data[:10,1])
                except RuntimeError:
                    print("Error - Exponential fit failed")
                    failed = True
                if(failed == False):
                    tauc.append(1/popt[1])
                    ax.semilogy(data[:-20,0], curveCvv(data[:-20,0], *popt), color=color, lw=1, linestyle='--')
                    print("Exponential fit ", dirList[d], " timescale: ", tauc[-1])
                else:
                    print("No fitting converged - tauc is set to zero")
                    tauc.append(0)
            #if(dirList[d] == dirList[-1]):
            #    ax.loglog(data[:,0], data[:,1], color='k', lw=1, marker='*', markersize=6, markeredgewidth = 1.2, label=labelList[d], fillstyle='none')
            #else:
            #ax.semilogy(data[:,0], data[:,1], color=color, lw=1, marker='o', markersize=4, label=labelList[d], fillstyle='none')
            ax.loglog(data[:,0], data[:,1], color=color, lw=1, marker='o', markersize=4, label=labelList[d], fillstyle='none')
            if(fixed == "temp"):
                phi.append(ucorr.readFromParams(dirSample, "phi"))
            elif(fixed == "phi"):
                Temp.append(np.mean(np.loadtxt(dirSample + "/energy.dat")[:,4]))
            elif(fixed == "f0"):
                if(d < dirList.shape[0]-1):
                    taup.append(1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma')))
                    Dr.append(float(dirList[d]))
            else:
                damping.append(np.sqrt(iod[d])/meanRad)
            #ax.semilogy(data[:,0], data[:,1], color=color, lw=1, marker='o', markersize=4, label=labelList[d], fillstyle='none')
    ax.legend(fontsize=9, loc='best', ncol=2)
    #ax.set_xlim(-0.07, 2.07)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$\\Delta_c$", fontsize=18)
    ax.set_ylabel("$PDF(\\Delta_c)$", fontsize=18)
    #ax.set_ylim(2.3e-05, 162)
    fig.tight_layout()
    figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pcoll-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6.5, 5), dpi = 120)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$\\tau_c$", fontsize=18)
    if(fixed == "temp"):
        #ax.set_ylim(0.0013, 0.21)
        ax.plot(phi, tauc, color='b', marker='s', markersize=8, fillstyle='none', lw=1)
        ax.set_xlabel("$Density,$ $\\varphi$", fontsize=18)
        np.savetxt(dirName + "tauc-T" + which + "-vsPhi-" + dyn + ".dat", np.column_stack((phi, tauc)))
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/ptauc-vsPhi-" + figureName
    elif(fixed == "phi"):
        ax.set_ylim(0.0085, 0.22)
        ax.loglog(Temp, tauc, color='b', marker='s', markersize=8, fillstyle='none', lw=1)
        ax.set_xlabel("$Temperature,$ $T$", fontsize=18)
        np.savetxt(dirName + "thermal" + which + "/tauc-vsT.dat", np.column_stack((Temp, tauc)))
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/ptauc-vsTemp-" + figureName
    elif(fixed == "f0"):
        ax.loglog(Dr, tauc, color='b', marker='s', markersize=8, fillstyle='none', lw=1)
        ax.loglog(Dr, taup, color='k', marker='v', markersize=8, fillstyle='none', lw=1)
        ax.set_xlabel("$Rotational$ $diffusion,$ $D_r$", fontsize=18)
        ax.set_ylabel("$Timescales$", fontsize=18)
        ax.legend(("$Collision$ $time$ $\\tau_c$", "$Persistence$ $time$ $\\tau_p$"), fontsize=12, loc='best')
        np.savetxt(dirName + "/tauc-vsDr-f0200-iod10.dat", np.column_stack((Dr, taup, tauc)))
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/ptauc-vsDr-" + figureName
    else:
        ax.set_ylim(0.0002, 0.22)
        ax.loglog(damping[1:], tauc[1:], color='b', marker='s', markersize=8, fillstyle='none', lw=1)
        ax.set_xlabel("$Damping,$ $\\beta$", fontsize=18)
        np.savetxt(dirName + "tauc-vsDamping.dat", np.column_stack((damping, tauc)))
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/ptauc-vsDamping-" + figureName
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelCorr(dirName, figureName, scaled=False, dyn = "nve", fixed = "temp", which = "0.001"):
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    if(fixed == "temp"):
        dirList = np.array(['thermal45', 'thermal58', 'thermal67', 'thermal72', 'thermal78', 'thermal83', 'thermal85', 'thermal88', 'thermal91', 'thermal1'])
        labelList = np.array(['$\\varphi=0.45$', '$\\varphi=0.58$', '$\\varphi=0.67$', '$\\varphi=0.72$', '$\\varphi=0.78$', '$\\varphi=0.83$', '$\\varphi=0.85$', '$\\varphi=0.88$', '$\\varphi=0.91$', '$\\varphi=1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        decay = []
    elif(fixed == "phi"):
        dirList = np.array(['T0.001', 'T0.002', 'T0.003', 'T0.005', 'T0.01', 'T0.02', 'T0.03', 'T0.05', 'T0.1'])
        labelList = np.array(['$T=0.001$', '$T=0.002$', '$T=0.003$', '$T=0.005$','$T=0.01$', '$T=0.02$', '$T=0.03$', '$T=0.05$', '$T=0.1$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        decay = []
    else:
        dirList = np.array(['nve', 'iod0.001', 'iod0.01', 'iod0.1', 'iod1', 'iod10', 'iod100', 'iod1000'])
        labelList = np.array(['$NVE$', '$\\beta \\sigma = 0.001$', '$\\beta \\sigma = 0.01$', '$\\beta \\sigma = 0.1$', '$\\beta \\sigma = 1$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 1000$'])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        iod = np.array([0, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
        damping = []
    for d in range(dirList.shape[0]):
        if(fixed == "temp"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T" + which + "/" + dyn + "/dynamics/"
        elif(fixed == "phi"):
            dirSample = dirName + os.sep + "thermal" + which + "/langevin/" + dirList[d] + "/" + dyn + "/dynamics/"
        else:
            dirSample = dirName + os.sep + dirList[d] + "/dynamics/"
        if(os.path.exists(dirSample + "blockVelCorr.dat")):
            meanRad = np.mean(np.loadtxt(dirSample + "particleRad.dat"))
            phi = ucorr.readFromParams(dirSample, "phi")
            timeStep = ucorr.readFromParams(dirSample, "dt")
            Temp = np.mean(np.loadtxt(dirSample + "/energy.dat")[:,4])
            data = np.loadtxt(dirSample + "/blockVelCorr.dat")[1:]
            tmax = timeStep*data.shape[0]
            data[:,1] /= data[0,1]
            #print(timeStep, tmax, Temp)
            # fit the curve
            failed = False
            try:
                popt, pcov = curve_fit(curveCvvOsc, data[:,0], data[:,1], bounds=([0.99, -np.inf, 0, 0], [1, np.inf, np.inf, np.inf]))
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == True):
                try:
                    popt, pcov = curve_fit(curveCvv, data[:,0], data[:,1], bounds=([0, -np.inf], [np.inf, np.inf]))
                except RuntimeError:
                    print("Error - curve_fit failed")
            width = 1
            if(scaled=="scaled"):
                width = 1/popt[1]
                #width = data[np.argwhere(data[:,1] < np.exp(-1))[0,0],0]
            # choose color
            color = colorList(d/dirList.shape[0])
            if(fixed == "phi"):
                decay.append([Temp, 1/popt[1], data[np.argwhere(data[:,1] < np.exp(-1))[0,0],0]])
            elif(fixed == "temp"):
                decay.append([phi, 1/popt[1], data[np.argwhere(data[:,1] < np.exp(-1))[0,0],0]])
            else:
                if(d==0):
                    color = 'k'
                damping.append([np.sqrt(iod[d])/meanRad, popt[1], data[np.argwhere(data[:,1] < np.exp(-1))[0,0],0]])
            ax[0].semilogx(data[:,0]/width, data[:,1], color=color, lw=1, label=labelList[d])
            x = np.linspace(np.min(data[:,0]), 10*np.max(data[:,0]), 1000)
            if(failed==False):
                #ax[0].semilogx(data[:,0]/width, curveCvvOsc(data[:,0], *popt), color=color, linestyle='--', lw=0.9)
                if(fixed == "phi"):
                    print("T: ", Temp, " damping: ", popt[1], " timescale: ", 1/popt[1])
                elif(fixed == "temp"):
                    print("phi: ", phi, " damping: ", popt[1], " timescale: ", 1/popt[1])
                else:
                    print("iod: ", iod[d], " damping: ", popt[1], " input: ", damping[-1][0])
                y = curveCvvOsc(x, *popt)
            else:
                ax[0].semilogx(data[:,0]/width, curveCvv(data[:,0], *popt), color=color, linestyle='--', lw=0.9)
                print("damping: ", popt[1], popt[0])
                y = curveCvv(x, *popt)
            #ax[0].plot(x, y/y[0], color=color, lw=0.5)
            fy = fft(y/y[0])*2/x.shape[0]#data[:,1]/data[0,1]
            fx = fftfreq(x.shape[0], x[1]-x[0])
            fy = fy[np.argsort(fx)]
            fx = np.sort(fx)
            #ax[1].loglog(fx*(x[1]-x[0]), np.real(fy)*width, color=color, lw=1)
            if(os.path.exists(dirSample + "fourierVelCorr.dat")):
                data = np.loadtxt(dirSample + "fourierVelCorr.dat")
                if(d==0):
                    fy0 = data[data[:,0]>0,2]
                if(d>0):
                    ax[1].loglog(data[data[:,0]>0,0], data[data[:,0]>0,2]/Temp, color=color, lw=1, label=labelList[d])
                    #ax[1].semilogy(data[:,0], data[:,2]*width, color=color, lw=1, label=labelList[d])
    x = np.linspace(1, 4e03, 1000)
    gamma = 1e-04
    y = gamma/(np.pi*(x**2+gamma**2))
    #ax[1].plot(x, y, color='k', lw=1.2, linestyle='dashed')
    ax[0].legend(fontsize=10, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    xlabel0 = "$Time,$ $t$"
    #xlabel1 = "$\\tilde{\\omega} = \\omega \\, dt$"
    xlabel1 = "$Frequency,$ $\\omega$"
    if(scaled=="scaled"):
        xlabel0 = "$Scaled$ $time,$ $\\gamma \\, t$"
        #xlabel1 = "$Scaled$ $frequency,$ $\\gamma^{-1} \\, \\tilde{\\omega}$"
        xlabel1 = "$Scaled$ $frequency,$ $\\gamma^{-1} \\, \\omega$"
    ax[0].set_xlabel(xlabel0, fontsize=18)
    ax[1].set_xlabel(xlabel1, fontsize=18)
    ax[0].set_ylabel("$C_{vv}(t)$", fontsize=18)
    ax[1].set_ylabel("$\\langle |\\vec{v}(\\omega)|^2 \\rangle \\, / \\, T$", fontsize=18)
    fig.tight_layout()
    figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pcorrVel-" + figureName
    if(scaled=="scaled"):
        figure1Name += "-scaled"
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    if(fixed != "phi" and fixed != "temp"):
        fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
        damping = np.array(damping)
        np.savetxt(dirName + os.sep + "dampingInOut.dat", damping)
        ax.set_ylim(0.0002, 0.22)
        ax.loglog(damping[1:,0], 1/damping[1:,1], color='k', markersize=8, marker='o', fillstyle='none', lw=1)
        #ax.semilogx(damping[:,0], damping[:,1]/damping[:,0], color='k', markersize=8, marker='o', fillstyle='none')
        #ax.semilogx(damping[1:,0], damping[1:,2]/damping[1:,0], color='g', markersize=8, marker='v', fillstyle='none')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylabel("$Decay$ $time,$ $\\tau_{vv}$", fontsize=18)
        ax.set_xlabel("$Damping$ $coefficient$ $\\beta_{in}$", fontsize=18)
        #ax.set_ylabel("$\\gamma_{out}/\\gamma_{in}$", fontsize=18)
        fig.tight_layout()
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pdamping-" + figureName
        fig.savefig(figure2Name + ".png", transparent=True, format = "png")
        np.savetxt(dirName + "/damping.dat", damping)
    else:
        fig, ax = plt.subplots(figsize = (6.5, 5), dpi = 120)
        decay = np.array(decay)
        np.savetxt(dirName + os.sep + "tauvv-" + dyn + ".dat", decay)
        if(dyn != "nve"):
            nveDecay = np.loadtxt(dirName + os.sep + "tauvv-nve.dat")
            ax.plot(decay[:,0], decay[:,2]/nveDecay[:,2], color='k', markersize=8, marker='o', fillstyle='none', lw=1)
        else:
            ax.plot(decay[:,0], decay[:,2], color='k', markersize=8, marker='o', fillstyle='none', lw=1)
        #ax.plot(decay[:,0], decay[:,2], color='k', markersize=8, marker='o', fillstyle='none')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylabel("$Decay$ $time,$ $\\tau_{vv}$", fontsize=18)
        if(fixed == "phi"):
            x = np.linspace(0.0008, 0.08, 100)
            ax.plot(x, 0.004*x**(-1/2), color='g', lw=1.5)
            ax.set_xlabel("$Temperature,$ $T$", fontsize=18)
            np.savetxt(dirName + "/velTimescale-vsT-phi" + which + ".dat", decay)
            figure2Name = "/home/francesco/Pictures/nve-nvt-nva/ptauvv-vsT-phi" + which
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(0.0085, 0.22)
        elif(fixed == "temp"):
            ax.set_ylim(0.0013, 0.21)
            #ax.set_xlim(0.42, 0.95)
            x = np.linspace(0.44, 0.85, 100)
            ax.plot(x, 0.3*(0.881-x)**(1.5), color='g', lw=1.5)
            ax.set_xlabel("$Density,$ $\\varphi$", fontsize=18)
            np.savetxt(dirName + "/velTimescale-vsPhi-T" + which + ".dat", decay)
            ax.set_yscale('log')
            figure2Name = "/home/francesco/Pictures/nve-nvt-nva/ptauvv-vsPhi-T" + which
        fig.tight_layout()
        fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def plotVelocityTimescale(dirName):
    fig, ax = plt.subplots(figsize = (6.5, 5), dpi = 120)
    dirList = np.array(['nve', 'iod0.001', 'iod0.01', 'iod0.1', 'iod1', 'iod10', 'iod100', 'iod1000'])
    labelList = np.array(['$NVE$', '$\\beta \\sigma = 0.001$', '$\\beta \\sigma = 0.01$', '$\\beta \\sigma = 0.1$', '$\\beta \\sigma = 1$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 1000$'])
    colorList = cm.get_cmap('cividis', dirList.shape[0])
    markerList = np.array(['o', '>', 'v', '^', 's', 'd', 'D', '*'])
    #nveData = np.loadtxt(dirName + os.sep + "tauvv-nve.dat")
    for d in range(dirList.shape[0]):
        if(d==0):
            color = 'k'
        else:
            color = colorList(d/dirList.shape[0])
        if(os.path.exists(dirName + os.sep + "tauvv-" + dirList[d] + ".dat")):
            data = np.loadtxt(dirName + os.sep + "tauvv-" + dirList[d] + ".dat")
            ax.semilogy(data[:,0], data[:,1], color=color, markersize=8, marker=markerList[d], fillstyle='none', lw=1, label=labelList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Timescale,$ $\\tau_{vv}$", fontsize=18)
    ax.set_xlabel("$Density,$ $\\varphi$", fontsize=18)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.set_ylim(0.00028, 0.38)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/tauvv-vsDamping-vsPhi.dat"
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotDampingInOut(dirName):
    fig, ax = plt.subplots(figsize = (6.5, 5), dpi = 120)
    dirList = np.array(['thermal45', 'thermal58', 'thermal67', 'thermal78', 'thermal83', 'thermal85', 'thermal88', 'thermal91'])
    labelList = np.array(['$\\varphi=0.45$', '$\\varphi=0.58$', '$\\varphi=0.67$', '$\\varphi=0.78$', '$\\varphi=0.83$', '$\\varphi=0.85$', '$\\varphi=0.88$', '$\\varphi=0.91$'])
    colorList = cm.get_cmap('viridis', dirList.shape[0])
    markerList = np.array(['o', 'v', '^', 's', 'd', 'D', '*', '>'])
    for d in range(dirList.shape[0]):
        if(os.path.exists(dirName + os.sep + dirList[d] + "/langevin/T0.001/dampingInOut.dat")):
            data = np.loadtxt(dirName + os.sep + dirList[d] + "/langevin/T0.001/dampingInOut.dat")
            ax.loglog(data[:,0], data[:,1]/data[:,0], color=colorList(d/dirList.shape[0]), markersize=8, marker=markerList[d], fillstyle='none', lw=1, label=labelList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$\\gamma_{in}$", fontsize=18)
    ax.set_ylabel("$\\gamma_{out}/\\gamma_{in}$", fontsize=18)
    ax.legend(fontsize=12, loc='best')
    ax.plot(np.linspace(1, 4000, 100), np.ones(100), color='k', linestyle='--', lw=0.8)
    ax.set_xlim(1.36, 3800)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/dampingInOut-vsDamping-vsPhi.dat"
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPTimescales(dirName, figureName, fixed=False, which='1000'):
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    if(fixed=="phi"):
        Dr = which
        dirList = np.array(['5000', '3000', '2000', '1000', '500', '300', '200',  '100', '50', '10', '5', '1', '1e-01', '1e-02'])
        labelList = np.array(['$f_0 = 5000$', '$f_0 = 3000$', '$f_0 = 2000$', '$f_0 = 1000$', '$f_0 = 500$', '$f_0 = 300$', '$f_0 = 200$', '$f_0 = 100$', '$f_0 = 50$', '$f_0 = 10$', '$f_0 = 5$', '$f_0 = 1$', '$f_0 = 0.1$', '$f_0 = 0.01$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        temp = np.zeros(dirList.shape[0])
        f0 = np.array([5000, 3000, 2000, 1000, 500, 300, 200,  100, 50, 10, 5, 1, 1e-01, 1e-02])
    elif(fixed=="iod"):
        dirList = np.array(['thermal45', 'thermal58', 'thermal67', 'thermal1'])
        labelList = np.array(['$\\varphi = 0.45$', '$\\varphi = 0.58$', '$\\varphi = 0.67$', '$\\varphi = 1$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod10', 'iod100', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 1000$'])
        colorList = cm.get_cmap('cividis', dirList.shape[0]+1)
        iod = np.array([1, 10, 100, 1000])
        damping = np.zeros(dirList.shape[0])
    tauvv = np.zeros(dirList.shape[0])
    tauc = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    tauDr = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + "iod1000/active-langevin/Dr" + which + "-f0" + dirList[d] + "/dynamics/"
            meanRad = np.mean(np.loadtxt(dirSample + "particleRad.dat"))
            temp[d] = np.mean(np.loadtxt(dirSample + "energy.dat")[:,4])
        elif(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            meanRad = np.mean(np.loadtxt(dirSample + "particleRad.dat"))
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr1e-03-f0200/dynamics/"
            meanRad = np.mean(np.loadtxt(dirSample + "particleRad.dat"))
            damping[d] = np.sqrt(iod[d])/meanRad
        if(os.path.exists(dirSample + "logDirCorr.dat")):
            tauDr[d] = 1/ucorr.readFromDynParams(dirSample, "Dr")
            data = np.loadtxt(dirSample + "/logDirCorr.dat")
            #data[:,1] /= data[0,1]
            ax[0].semilogx(data[:,0], data[:,1], color=colorList(d/dirList.shape[0]), lw=1, label=labelList[d])
            failed = False
            try:
                popt, pcov = curve_fit(curveCvv, data[:,0], data[:,1], bounds=([0.99, -np.inf], [1, np.inf]))
            except RuntimeError:
                #print("Error - curve_fit failed")
                failed = True
            if(failed == False):
                taup[d] = 1/popt[1]
                ax[0].semilogx(data[:,0], curveCvv(data[:,0], *popt), color=colorList(d/dirList.shape[0]), linestyle='--', lw=0.9)
            else:
                taup[d] = data[np.argwhere(data[:,1] < np.exp(-1))[0,0],0]
            data = np.loadtxt(dirSample + "/logVelCorr.dat")
            ax[0].semilogx(data[:,0], data[:,1]/data[0,1], color=colorList(d/dirList.shape[0]), lw=1.2)
            tauvv[d] = data[np.argwhere(data[:,1]/data[0,1] < np.exp(-1))[0,0],0]
        if(os.path.exists(dirSample + "contactCollision.dat")):
            data = np.loadtxt(dirSample + "/contactCollision.dat")
            ax[1].loglog(data[:,0], data[:,1], color=colorList(d/dirList.shape[0]), lw=1.2, label=labelList[d])
            tauc[d] = data[np.argwhere(data[:,1] < np.exp(-1))[0,0],0]
            print("Damping timescale: ", taup[d], " ", tauDr[d], " collisional timescale: ", tauc[d])
    if(fixed=="phi"):
        ax[1].legend(fontsize=8, loc='best', ncol=3)
    else:
        ax[1].legend(fontsize=12, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$Time,$ $t$", fontsize=18)
    ax[1].set_xlabel("$Time$ $between$ $collisions,$ $\\Delta t_c$", fontsize=18)
    ax[0].set_ylabel("$C_{vv}(t)$", fontsize=18)
    ax[1].set_ylabel("$PDF(\\Delta_c)$", fontsize=18)
    fig.tight_layout()
    if(fixed=="phi"):
        x = f0
        xlabel = "$Propulsion,$ $f_0$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pCvvCollision-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTimescales-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pCvvCollision-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTimescales-vsPhi-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pCvvCollision-vsDamping-" + figureName
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTimescales-vsDamping-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.semilogy(x, taup, color='k', markersize=8, marker='o', fillstyle='none', label="$\\tau_{nn}$", lw=1)
    ax.semilogy(x, tauvv, color='b', markersize=7, marker='s', fillstyle='none', label="$\\tau_{vv}$", lw=1)
    ax.semilogy(x, tauc, color='g', markersize=10, marker='*', fillstyle='none', label="$\\tau_c$", lw=1)
    if(fixed!="iod"):
        ax.set_xscale('log')
    if(fixed=="phi"):
        ax.set_yscale('log')
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Timescales$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def curveCvvf0(x, a, b, c, d, e, f):
    return a * (np.exp(-b*x) - c * np.exp(-d*x)) * np.cos(e*x + f)

def plotSPVelCorrVSDrf0(dirName, figureName, scaled=False, fixed="Dr", which="100", iod = '100'):
    spacing = 10
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    if(fixed=="f0"):
        f0 = which
        dirList = np.array(['1e-03', '5e-03', '1e-02', '5e-02', '1e-01', '5e-01', '1', '5', '10', '100', '1000', '../../iod' + iod])
        Dr = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100, 1000, 1e05])
    elif(fixed=="Dr"):
        Dr = which
        dirList = np.array(['5000', '3000', '2000', '1000', '500', '300', '200',  '100', '50', '10', '5', '1', '5e-01', '1e-01', '1e-02', '../../iod' + iod])
        f0 = np.array([5000, 3000, 2000, 1000, 500, 300, 200, 100, 50, 10, 5, 1, 5e-01, 1e-01, 1e-02, 0])
    else:
        dirList = np.empty(0)
    colorList = cm.get_cmap('plasma', dirList.shape[0])#winter
    width = np.zeros(dirList.shape[0])
    omegac = np.zeros(dirList.shape[0])
    cvv0 = np.zeros(dirList.shape[0])
    Temp = np.zeros((dirList.shape[0],2))
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "../../particleRad.dat"))
    damping = np.sqrt(float(iod))/meanRad
    # nva data
    for d in range(dirList.shape[0]):
        if(d==dirList.shape[0]-1):
            dirSample = dirName + dirList[d] + "/dynamics/"
            color = 'k'
        else:
            if(fixed=="f0"):
                dirSample = dirName + "/Dr" + dirList[d] + "-f0" + f0 + "/dynamics/"
            elif(fixed=="Dr"):
                dirSample = dirName + "/Dr" + Dr + "-f0" + dirList[d] + "/dynamics/"
            color = colorList(d/dirList.shape[0])
        if(os.path.exists(dirSample + "logVelCorr.dat")):
            data = np.loadtxt(dirSample + "logVelCorr.dat")[1:]
            #data[:,1] /= data[0,1]
            timeStep = ucorr.readFromParams(dirSample, "dt")
            Temp[d,0] = np.mean(np.loadtxt(dirSample + "energy.dat")[:,4])
            Temp[d,1] = np.std(np.loadtxt(dirSample + "energy.dat")[:,4])
            failed = False
            try:
                popt, pcov = curve_fit(curveCvvf0, data[:,0], data[:,1], bounds=([0.99, 1e02, -np.inf, 1e-05, 0, 0], [1, 5e03, 1, np.inf, np.inf, 2*np.pi]))
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == True):
                try:
                    popt, pcov = curve_fit(curveCvvOsc, data[:,0], data[:,1], bounds=([0.99, -np.inf, 0, 0], [1, np.inf, np.inf, 2*np.pi]))
                except RuntimeError:
                    print("Error - curve_fit failed")
            width[d] = 1
            omegac[d] = popt[3]
            cvv0[d] = popt[0]
            if(scaled=="scaled"):
                width[d] = data[np.argwhere(data[:,1] < np.exp(-1))[0,0],0]
                #width[d+1] = 1/popt[1]
            ax[0].semilogx(data[:,0]/width[d], data[:,1], color=color, lw=1, label="$f_0=$"+dirList[d])
            if(failed == False):
                print("f0:", dirList[d], " gamma: ", popt[1], " omega_c: ", popt[3], " amplitude: ", popt[0])
                ax[0].semilogx(data[:,0]/width[d], curveCvvf0(data[:,0], *popt), color=color, lw=0.9, linestyle='--')
            else:
                print("gamma: ", popt[1])
                ax[0].semilogx(data[:,0]/width[d], curveCvvOsc(data[:,0], *popt), color=color, lw=0.9, linestyle='--')
            if(os.path.exists(dirSample + "../dynamics-short/fourierVelCorr.dat")):
                data = np.loadtxt(dirSample + "../dynamics-short/fourierVelCorr.dat")
                ax[1].loglog(data[data[:,0]>0,0]*width[d]/spacing, data[data[:,0]>0,2]/Temp[d,0], color=color, lw=1)
    # plot things
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    xlabel = "$Time,$ $t$"
    ylabel = "$\\langle |\\vec{v}(\\omega)|^2 \\rangle \\, / \\, T$"
    if(scaled=="scaled"):
        xlabel = "$Scaled$ $time,$ $\\gamma \\, t$"
        ylabel = "$\\gamma \\, \\langle |\\vec{v}(\\omega)|^2 \\rangle \\, / \\, T$"
    ax[0].set_xlabel(xlabel, fontsize=18)
    ax[1].set_xlabel("$Frequency,$ $\\omega$", fontsize=18)
    ax[0].set_ylabel("$C_{vv}(t)$", fontsize=18)
    ax[1].set_ylabel(ylabel, fontsize=18)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    if(fixed=="f0"):
        label = "$D_r$"
        labels = ['$10^{-3}$', '$10^3$']
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pcorrVel-vsDr-" + figureName
    elif(fixed=="Dr"):
        label = "$f_0$"
        labels = ['$10^4$', '$10^{-1}$']
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pcorrVel-vsf0-" + figureName
    cb.set_label(label=label, fontsize=18, labelpad=-20, rotation='horizontal')
    cb.set_ticks([0,1])
    cb.set_ticklabels(labels)
    cb.ax.tick_params(labelsize=14, size=0)
    fig.tight_layout()
    #fig.subplots_adjust(wspace=0.3)
    if(scaled=="scaled"):
        figure1Name += "-scaled"
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="f0"):
        Dr /= (meanRad*np.sqrt(float(iod)))
        x = Dr
        xlabel = "$Rotational$ $diffusion,$ $D_r$"
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pFit-vsDr-" + figureName
    elif(fixed=="Dr"):
        x = f0
        xlabel = "$Propulsion,$ $f_0$"
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pFit-vsf0-" + figureName
    ax.loglog(x[:-3], omegac[:-3]/x[:-3], color='k', lw=1.2, marker='o', fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$\\omega_c \\, / \\, D_r$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def plotSPDirCorrVSDrf0(dirName, figureName, fixed="Dr", which="100", iod = '100'):
    fig, ax = plt.subplots(dpi = 120)
    if(fixed=="f0"):
        f0 = which
        dirList = np.array(['1e-03', '5e-03', '1e-02', '5e-02', '1e-01', '5e-01', '1', '5', '10', '100', '1000', '../../iod' + iod])
        Dr = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100, 1000, 1e05])
    elif(fixed=="Dr"):
        Dr = which
        dirList = np.array(['5000', '3000', '2000', '1000', '500', '300', '200',  '100', '50', '10', '5', '1', '5e-01', '1e-01', '1e-02', '../../iod' + iod])
        f0 = np.array([5000, 3000, 2000, 1000, 500, 300, 200, 100, 50, 10, 5, 1, 5e-01, 1e-01, 1e-02, 0])
    else:
        print("specify fixed paramater")
        dirList = np.empty(0)
    colorList = cm.get_cmap('plasma', dirList.shape[0])#winter
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "../../particleRad.dat"))
    damping = np.sqrt(float(iod))/meanRad
    # nva data
    for d in range(dirList.shape[0]):
        if(d==dirList.shape[0]-1):
            dirSample = dirName + dirList[d] + "/dynamics/"
            color = 'k'
        else:
            if(fixed=="f0"):
                dirSample = dirName + "/Dr" + dirList[d] + "-f0" + f0 + "/dynamics/"
            elif(fixed=="Dr"):
                dirSample = dirName + "/Dr" + Dr + "-f0" + dirList[d] + "/dynamics/"
            color = colorList(d/dirList.shape[0])
        if(os.path.exists(dirSample + "logDirCorr.dat")):
            data = np.loadtxt(dirSample + "logDirCorr.dat")
            ax.semilogx(data[:,0], data[:,1], color=color, lw=1, label="$f_0=$"+dirList[d])
    # plot things
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=18)
    ax.set_ylabel("$C_{nn}(t)$", fontsize=18)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    if(fixed=="f0"):
        label = "$D_r$"
        labels = ['$10^{-3}$', '$10^3$']
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pcorrVel-vsDr-" + figureName
    elif(fixed=="Dr"):
        label = "$f_0$"
        labels = ['$10^4$', '$10^{-1}$']
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pcorrVel-vsf0-" + figureName
    cb.set_label(label=label, fontsize=18, labelpad=-20, rotation='horizontal')
    cb.set_ticks([0,1])
    cb.set_ticklabels(labels)
    cb.ax.tick_params(labelsize=14, size=0)
    fig.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPairCorrVSDrf0(dirName, figureName, fixed="Dr", which="100", iod='1000'):
    spacing = 10
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    if(fixed=="f0"):
        f0 = which
        dirList = np.array(['1e-03', '5e-03', '1e-02', '5e-02', '1e-01', '5e-01', '1', '5', '10', '50', '100', '500', '1000', '5000', '10000', '../../iod' + iod])
        Dr = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 1e05])
    elif(fixed=="Dr"):
        Dr = which
        dirList = np.array(['5000', '3000', '2000', '1000', '500', '300', '200',  '100', '50', '10', '5', '1', '5e-01', '1e-01', '1e-02', '../../iod' + iod])
        f0 = np.array([5000, 3000, 2000, 1000, 500, 300, 200, 100, 50, 10, 5, 1, 5e-01, 1e-01, 1e-02, 0])
    colorList = cm.get_cmap('plasma', dirList.shape[0])#winter
    Temp = np.zeros((dirList.shape[0],2))
    Pressure = np.zeros((dirList.shape[0],7))
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "../../particleRad.dat"))
    numParticles = ucorr.readFromParams(dirName + os.sep + "../../", "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "../../boxSize.dat")
    volume = boxSize[0]*boxSize[1]
    density = numParticles/volume
    damping = np.sqrt(float(iod))/meanRad
    peak = np.zeros((dirList.shape[0],2))
    # nva data
    for d in range(dirList.shape[0]):
        if(d==dirList.shape[0]-1):
            dirSample = dirName + dirList[d] + "/dynamics/"
            color = 'k'
        else:
            if(fixed=="f0"):
                dirSample = dirName + "/Dr" + dirList[d] + "-f0" + f0 + "/dynamics/"
            elif(fixed=="Dr"):
                dirSample = dirName + "/Dr" + Dr + "-f0" + dirList[d] + "/dynamics/"
            color = colorList(d/dirList.shape[0])
        if(os.path.exists(dirSample + "pairCorr.dat")):
            data = np.loadtxt(dirSample + "energy.dat")
            Temp[d,0] = np.mean(data[:,4])
            Temp[d,1] = np.std(data[:,4])
            Pressure[d,0] = np.mean(numParticles*data[:,4])
            Pressure[d,1] = np.std(numParticles*data[:,4])
            Pressure[d,2] = np.mean(data[:,5])
            Pressure[d,3] = np.std(data[:,5])
            if(d!=dirList.shape[0]-1):
                Pressure[d,4] = np.mean(data[:,6])
                Pressure[d,5] = np.std(data[:,6])
                #activePressure = spCorr.computeActivePressure(dirSample)
                #Pressure[d,4] = np.mean(activePressure)
                #Pressure[d,5] = np.std(activePressure)
            data = np.loadtxt(dirSample + "pairCorr.dat")
            peak[d,0] = data[np.argmax(data[:,1]),0]
            peak[d,1] = np.max(data[:,1])
            overlap = 1 - data[:,0]/meanRad
            Pressure[d,6] = density * Temp[d,0] - (np.pi/(6*meanRad**2))*density**2 * np.sum(data[:,0]*data[:,1]*overlap) * meanRad**6
            ax[0].plot(data[:,0], data[:,1], color=color, lw=1)
    # plot things
    ax[0].set_xlim(-0.005, 0.14)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$Distance,$ $r$", fontsize=18)
    ax[0].set_ylabel("$Pair$ $distribution,$ $g(r)$", fontsize=18)
    ax[0].set_ylim(-0.2, 4.4)
    if(fixed=="f0"):
        Dr = Dr/(meanRad*np.sqrt(float(iod)))
        x = float(f0)**2/(damping*Dr)
        xlabel = "$Rotational$ $timescale,$ $1/D_r$"
    elif(fixed=="Dr"):
        Dr = float(Dr)/(meanRad*np.sqrt(float(iod)))
        x = f0**2/(damping*Dr)
        xlabel = "$Propulsion,$ $f_0$"
    xlabel = "$Peclet$ $number,$ $f_0^2\\, / \\, \\gamma D_r$"
    ax[1].errorbar(x, Pressure[:,0], Pressure[:,1], color='k', lw=1.2, marker='o', fillstyle='none', capsize=4, label="$\\rho k_B T$")
    ax[1].errorbar(x, Pressure[:,2], Pressure[:,3], color='b', lw=1.2, marker='v', fillstyle='none', capsize=4, label="$p_{virial}$")
    #ax[1].errorbar(x, Pressure[:,4], Pressure[:,5], color='g', lw=1.2, marker='s', fillstyle='none', capsize=4, label="$p_{active}$")
    #ax[1].errorbar(x, Pressure[:,0]+Pressure[:,2]+Pressure[:,4], np.sqrt(Pressure[:,1]*Pressure[:,1]+Pressure[:,3]*Pressure[:,3]+Pressure[:,5]*Pressure[:,5]), color='r', lw=1.2, marker='*', fillstyle='none', capsize=4, label="$p_{tot}$")
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(xlabel, fontsize=18)
    ax[1].set_ylabel("$Pressure,$ $p$", fontsize=18)
    ax[1].legend(fontsize=14, loc='best')
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    if(fixed=="f0"):
        label = "$D_r$"
        labels = ['$10^{-3}$', '$10^3$']
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/ppairCorr-vsDr-" + figureName
    elif(fixed=="Dr"):
        label = "$f_0$"
        labels = ['$10^4$', '$10^{-1}$']
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/ppairCorr-vsf0-" + figureName
    cb.set_label(label=label, fontsize=18, labelpad=-20, rotation='horizontal')
    cb.set_ticks([0,1])
    cb.set_ticklabels(labels)
    cb.ax.tick_params(labelsize=14, size=0)
    fig.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="f0"):
        #Dr = Dr/(meanRad*np.sqrt(float(iod)))
        x = float(f0)**2/(damping*Dr)
        x = 1/Dr
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pPCPeak-vsDr-" + figureName
        xlabel = "$Persistence$ $time,$ $1/D_r$"
    elif(fixed=="Dr"):
        Dr = float(Dr)/(meanRad*np.sqrt(float(iod)))
        x = f0**2/(damping*Dr)
        x = f0
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pPCPeak-vsf0-" + figureName
        xlabel = "$Propulsion,$ $f_0$"
    #xlabel = "$Peclet$ $number,$ $f_0^2\\, / \\, \\gamma D_r$"
    ax.semilogx(x, peak[:,1], color='k', lw=1.2, marker='o', fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Peak$ $of$ $g(r)$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def plotSPVelSpaceCorrVSDrf0(dirName, figureName, fixed='Dr', which='200', iod='10'):
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    if(fixed=="f0"):
        f0 = which
        dirList = np.array(['1e-03', '5e-03', '1e-02', '5e-02', '1e-01', '5e-01', '1', '5', '10', '50', '100', '1000', '../../iod' + iod])
        labelList = np.array(['$D_r = 10^{-3}$', '$D_r = 5 \\times 10^{-3}$', '$D_r = 10^{-2}$', '$D_r = 5 \\times 10^{-2}$', '$D_r = 10^{-1}$', '$D_r = 5 \\times 10^{-1}$', '$D_r = 1$', '$D_r = 5$', '$D_r = 10$', '$D_r = 50$', '$D_r = 100$', '$D_r = 1000$', '$NVT$'])
        Dr = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 1e05])
        taup = np.zeros(dirList.shape[0])
    elif(fixed=="Dr"):
        Dr = which
        dirList = np.array(['1000', '700', '500', '300', '200',  '100', '50', '30', '20', '10', '../../iod' + iod])
        labelList = np.array(['$f_0 = 1000$', '$f_0 = 700$', '$f_0 = 500$', '$f_0 = 300$', '$f_0 = 200$', '$f_0 = 100$', '$f_0 = 50$', '$f_0 = 30$', '$f_0 = 20$', '$f_0 = 10$', '$NVT$'])
        f0 = np.array([1000, 700, 500, 300, 200, 100, 50, 30, 20, 10, 0])
    colorList = cm.get_cmap('plasma', dirList.shape[0])#winter
    phi = ucorr.readFromParams(dirName + dirList[-1], "phi")
    Temp = np.zeros((dirList.shape[0]+1,2))
    meanRad = np.mean(np.loadtxt(dirName + os.sep + "../particleRad.dat"))
    damping = np.sqrt(float(iod))/meanRad
    diff = np.zeros((dirList.shape[0],3))
    # nva data
    for d in range(dirList.shape[0]):
        if(d==dirList.shape[0]-1):
            dirSample = dirName + dirList[d] + "/dynamics/"
            color = 'k'
        else:
            if(fixed=="f0"):
                dirSample = dirName + "/Dr" + dirList[d] + "-f0" + f0 + "/dynamics/"
                taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            elif(fixed=="Dr"):
                dirSample = dirName + "/Dr" + Dr + "-f0" + dirList[d] + "/dynamics/"
            color = colorList((dirList.shape[0]-d)/dirList.shape[0])
        if(os.path.exists(dirSample + "spaceVelCorr.dat")):
            data = np.loadtxt(dirSample + "spaceVelCorr.dat")
            data[:,0] /= meanRad
            data = data[data[:,1]>0]
            Temp = np.mean(np.loadtxt(dirSample + "energy.dat")[:,4])
            ax[0].semilogy(data[:,0], data[:,1], color=color, lw=1, label=labelList[d])
            data = data[data[:,0]<80,:]
            diff[d,0] = np.sqrt(np.sum(data[:,0]*data[:,1]**2))
            diff[d,1] = np.sqrt(np.sum(data[:,0]*data[:,2]**2))
            diff[d,2] = np.sqrt(np.sum(data[:,0]*data[:,3]**2))
        if(os.path.exists(dirSample + "localDensity-N8.dat")):#N8 for 1k
            data = np.loadtxt(dirSample + "localDensity-N8.dat")
            ax[1].semilogy(data[:,0], data[:,1], color=color, lw=1, label=labelList[d])
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    #ax[1].set_xlim(-0.02, 1.12)
    ax[0].set_xlabel("$Distance,$ $r \\, / \\, \\sigma$", fontsize=18)
    ax[1].set_xlabel("$Local$ $density,$ $\\varphi$", fontsize=18)
    ax[0].set_ylabel("$C^{\\parallel}_{vv}(r)$", fontsize=18)
    ax[1].set_ylabel("$P(\\varphi)$", fontsize=18)
    if(fixed=='f0'):
        np.savetxt(dirName + os.sep + "spaceCorr-iod" + iod + ".dat", np.column_stack((Dr, taup, diff[:,0], diff[:,1], diff[:,2])))
    ax[1].legend(fontsize=10, loc='lower left', ncol=2)
    #colorBar = cm.ScalarMappable(cmap=colorList)
    #cb = plt.colorbar(colorBar)
    if(fixed=="f0"):
        label = "$D_r$"
        labels = ['$10^{-3}$', '$10^3$']
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceVelCorr-vsDr-" + figureName
    elif(fixed=="Dr"):
        label = "$f_0$"
        labels = ['$10^4$', '$10^{-1}$']
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceVelCorr-vsf0-" + figureName
    #cb.set_label(label=label, fontsize=18, labelpad=-20, rotation='horizontal')
    #cb.set_ticks([0,1])
    #cb.set_ticklabels(labels)
    #cb.ax.tick_params(labelsize=14, size=0)
    fig.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="f0"):
        #Dr = Dr/(meanRad*np.sqrt(float(iod)))
        x = float(f0)**2/(damping*Dr)
        x = taup
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceDiff-vsDr-" + figureName
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    elif(fixed=="Dr"):
        Dr = float(Dr)/(meanRad*np.sqrt(float(iod)))
        x = f0**2/(damping*Dr)
        x = f0
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceDiff-vsf0-" + figureName
        xlabel = "$Propulsion,$ $f_0$"
    #xlabel = "$Peclet$ $number,$ $f_0^2\\, / \\, \\gamma D_r$"
    #ax.plot(np.ones(100)*400, np.linspace(-0.1, 0.1, 100), color='k', linestyle='--', lw=0.7)#/0.5
    #ax.set_ylim(-0.0028, 0.0082)
    #ax.set_ylim(-0.011, 0.048)
    ax.semilogx(x[diff[:,0]>0], diff[diff[:,0]>0,0], color='r', lw=1.2, marker='s', fillstyle='none', label="$\\int |C_{vv}^{\\parallel}|^2(r) dr$")
    ax.semilogx(x[diff[:,0]>0], diff[diff[:,0]>0,1], color='g', lw=1.2, marker='v', fillstyle='none', label="$\\int |C_{vv}^{\\perp}(r)|^2 dr$")
    ax.semilogx(x[diff[:,0]>0], diff[diff[:,0]>0,2], color='k', lw=1.2, marker='o', fillstyle='none', label="$\\int |C_{vv}^{cross}(r)|^2 dr$")
    ax.legend(fontsize=12, loc='upper left')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Amount$ $of$ $correlation$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def curveCumSum(x, a, b, c):
    return 1 - c*np.exp(-(x*a)**b)

def plotSPCollisionPersistence(dirName, figureName, fixed=False, which='10'):
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67',  'thermal72', 'thermal78', 'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$\\varphi = 0.45$', '$\\varphi = 0.58$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.83$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-02', '5e-02', '1e-01', '5e-01', '1', '2', '5', '10', '50', '100', '1000'])
        labelList = np.array(['$D_r = 0.01$', '$D_r = 0.05$', '$D_r = 0.1$', '$D_r = 0.5$', '$D_r = 1$', '$D_r = 2$', '$D_r = 5$', '$D_r = 10$', '$D_r = 50$', '$D_r = 100$', '$D_r = 1000$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod700', 'iod900', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 700$', '$\\beta \\sigma = 900$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 700, 900, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    tauc = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics-col/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics-col/"
            Dr[d] = ucorr.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics-col/"
            damping[d] = ucorr.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "/contactCollision.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            if(os.path.exists(dirSample + "/inClusterCollision.dat")):
                collision = np.loadtxt(dirSample + "/outClusterCollisionIntervals.dat")
                print('cluster')
            else:
                collision = np.loadtxt(dirSample + "/contactCollisionIntervals.dat")
                print('contact')
            collision, counts = np.unique(collision, return_counts=True)
            cdf = np.cumsum(counts)/np.sum(counts)
            cdf = cdf[1:]
            collision = collision[1:]
            failed = False
            try:
                popt, pcov = curve_fit(curveCumSum, collision, cdf, bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == False):
                ax.plot(collision, curveCumSum(collision, *popt), color='k', lw=2, linestyle='--')
                tauc[d] = 1/popt[0]
            else:
                tauc[d] = collision[np.argwhere(cdf>0.9)[0,0]]
            print(dirList[d], " timescale: ", tauc[d])
            ax.semilogx(collision, cdf, color=colorList(d/dirList.shape[0]), lw=1, marker='o', markersize=4, label=labelList[d], fillstyle='none')
            #ax.plot(np.ones(50)*taup[d], np.linspace(-0.1,1.1,50), color=colorList(d/dirList.shape[0]), ls='dotted', lw=1)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.set_ylim(-0.07, 1.07)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$\\Delta_c$", fontsize=18)
    ax.set_ylabel("$CDF(\\Delta_c)$", fontsize=18)
    #ax.set_ylim(2.3e-05, 162)
    fig.tight_layout()
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pCollision-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTaus-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = Dr
        xlabel = "$Rotational$ $diffusion,$ $D_r$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pCollision-vsDr-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTaus-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pCollision-vsDamping-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTaus-vsDamping-" + figureName + "-Dr" + which
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6.5, 3.5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.semilogy(x, tauc, color='b', marker='s', markersize=8, fillstyle='none', lw=1)
    ax.semilogy(x, taup, color='k', marker='v', markersize=8, fillstyle='none', lw=1)
    if(fixed=='phi'):
        ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Timescales$", fontsize=16)
    ax.legend(("$Collision$ $rate$ $\\tau_c$", "$Persistence$ $time$ $\\tau_p$"), fontsize=12, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelTimeCorr(dirName, figureName, fixed=False, which='10', fit=False):
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72',  'thermal78',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$\\varphi = 0.45$', '$\\varphi = 0.58$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.83$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-03', '5e-03', '1e-02', '5e-02', '1e-01', '5e-01', '1', '2', '5', '10', '100', '1000'])
        labelList = np.array(['$D_r = 10^{-3}$', '$D_r = 5 \\times 10^{-3}$', '$D_r = 10^{-2}$', '$D_r = 5 \\times 10^{-2}$', '$D_r = 10^{-1}$', '$D_r = 5 \\times 10^{-1}$', '$D_r = 1$', '$D_r = 2$', '$D_r = 5$', '$D_r = 10$', '$D_r = 100$', '$D_r = 1000$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    tauc = []
    taud = []
    width = []
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = ucorr.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = ucorr.readFromDynParams(dirSample, "damping")
        color = colorList(d/dirList.shape[0])
        if(os.path.exists(dirSample + "../dynamics-col/contactCollision.dat")):
            # collision time distribution
            data = np.loadtxt(dirSample + "../dynamics-col/contactCollision.dat")
            data = data[data[:,1]>0]
            ax[0].loglog(data[:,0], data[:,1], color=color, lw=1, marker='o', markersize=4, label=labelList[d], fillstyle='none')
            # fitting
            if(fit=="fit"):
                failed = False
                try:
                    popt, pcov = curve_fit(curveCvv, data[:,0], data[:,1])
                except RuntimeError:
                    print("Error - curve_fit failed")
                    failed = True
                if(failed == True):
                    try:
                        popt, pcov = curve_fit(curvePL, data[:,0], data[:,1])
                    except RuntimeError:
                        print("Error - curve_fit failed")
                if(failed == False):
                    tauc.append(1/popt[0])
                    ax[0].semilogy(data[:,0], curvePL(data[:,0], *popt), color=color, lw=1, linestyle='--')
                else:
                    tauc.append(1/popt[1])
                    ax[0].semilogy(data[:,0], curveCvv(data[:,0], *popt), color=color, lw=1, linestyle='--')
            # velocity time correlation
        if(os.path.exists(dirSample + "/logVelCorr.dat")):#logVelCorr
            data = np.loadtxt(dirSample + "logVelCorr.dat")
            data[:,1] /= data[0,1]
            #width[d] = data[np.argwhere(data[:,1] < np.exp(-1))[0,0],0]
            ax[1].semilogx(data[:,0], data[:,1], color=color, lw=1, label=labelList[d])
            # fitting
            if(fit == "fit"):
                failed = False
                try:
                    popt, pcov = curve_fit(curveCvvf0, data[:,0], data[:,1], bounds=([0.99, 1e03, -np.inf, 1e-05, 0, 0], [1, 5e03, 1, np.inf, np.inf, 2*np.pi]))
                except RuntimeError:
                    print("Error - curve_fit failed")
                    failed = True
                if(failed == True):
                    try:
                        popt, pcov = curve_fit(curveCvvOsc, data[:,0], data[:,1], bounds=([0.99, -np.inf, 0, 0], [1, np.inf, np.inf, 2*np.pi]))
                    except RuntimeError:
                        print("Error - curve_fit failed")
                if(failed == False):
                    taud.append(1/popt[0])
                    ax[0].semilogx(data[:,0], curveCvvf0(data[:,0], *popt), color=color, lw=0.9, linestyle='--')
                else:
                    taud.append(1/popt[1])
                    ax[0].semilogx(data[:,0], curveCvvOsc(data[:,0], *popt), color=color, lw=0.9, linestyle='--')
    ax[0].legend(fontsize=10, loc='best', ncol=2)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$\\Delta_c$", fontsize=18)
    ax[0].set_ylabel("$PDF(\\Delta_c)$", fontsize=18)
    ax[1].set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax[1].set_ylabel("$C_{vv}(\\Delta t)$", fontsize=18)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    figure1Name = "/home/francesco/Pictures/nve-nvt-nva/ptimecorr-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    if(fit=="fit"):
        if(fixed=="iod"):
            x = phi
            xlabel = "$Density,$ $\\varphi$"
            figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTaus-vsPhi-" + figureName + "-iod" + which
        elif(fixed=="phi"):
            x = 1/Dr
            xlabel = "$Persistence$ $time,$ $\\tau_p$"
            figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTaus-vsDr-" + figureName + "-iod" + which
        else:
            x = damping
            xlabel = "$Damping,$ $\\gamma$"
            figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTaus-vsDamping-" + figureName + "-Dr" + which
        fig, ax = plt.subplots(figsize = (6.5, 5), dpi = 120)
        ax.plot(x, tauc, color='g', lw=1.2, marker='s', fillstyle='none', label="$\\tau_c$")
        ax.plot(x, taud, color='b', lw=1.2, marker='v', fillstyle='none', label="$\\tau_d$")
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel("$Timescales$", fontsize=18)
        ax.legend(fontsize=12, loc='best')
        fig.tight_layout()
        fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def curveCvv(x, a, b, c):
    return c*np.exp(-(x*a))/x**b

def plotSPVelSpaceCorr(dirName, figureName, fixed=False, which='10'):
    fig, ax = plt.subplots(figsize = (6.5, 5.5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72', 'thermal78', 'thermal83',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$\\varphi = 0.45$', '$\\varphi = 0.58$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.83$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-05', '1e-04', '5e-04', '1e-03', '5e-03', '7e-03', '1e-02', '1.2e-02', '1.5e-02', '5e-02', '1e-01', '5e-01', '1', '5'])
        labelList = np.array(['$D_r = 0.00001$', '$D_r = 0.0001$', '$D_r = 0.0005$', '$D_r = 0.0007$', '$D_r = 0.001$', '$D_r = 0.0012$', '$D_r = 0.0015$', '$D_r = 0.005$', '$D_r = 0.01$', '$D_r = 0.05$', '$D_r = 0.1$', '$D_r = 0.5$', '$D_r = 1$', '$D_r = 5$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod700', 'iod900', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 700$', '$\\beta \\sigma = 900$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 700, 900, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    diff = np.zeros((dirList.shape[0],3))
    corrlength = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = ucorr.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = ucorr.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "spaceVelCorrInCluster.dat")):
            meanRad = np.mean(np.loadtxt(dirSample + "/particleRad.dat"))
            data = np.loadtxt(dirSample + "/spaceVelCorrInCluster.dat")
            if(d > 9):
                data = np.loadtxt(dirSample + "/spaceVelCorrOutCluster.dat")
            data[:,0] /= meanRad
            data = data[data[:,0]<122]
            #data = data[data[:,1]>0]
            ax.plot(data[1:,0], data[1:,1], color=colorList((dirList.shape[0]-d)/dirList.shape[0]), lw=1.2, label=labelList[d])
            #ax.plot(data[1:,0], data[1:,2], color=colorList((dirList.shape[0]-d)/dirList.shape[0]), lw=1.2, ls='--')
            failed = False
            try:
                popt, pcov = curve_fit(curveCvv, data[:,0], data[:,1])
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == False and d < 12):
                corrlength[d,0] = 1/popt[0]
                corrlength[d,1] = ucorr.computeTau(data, index=1, threshold=np.exp(-1)*data[1,1], normalized=False)
                ax.plot(data[1:,0], curveCvv(data[1:,0], *popt), color=colorList((dirList.shape[0]-d)/dirList.shape[0]), lw=0.9, linestyle='--')
            #data = data[data[:,0]<80,:]
            data[:,1] /= data[0,1]
            diff[d,0] = np.sqrt(np.sum(data[:,0]*data[:,1]**2))
            diff[d,1] = np.sqrt(np.sum(data[:,0]*data[:,2]**2))
            diff[d,2] = np.sqrt(np.sum(data[:,0]*data[:,3]**2))
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Distance,$ $r \\, / \\, \\sigma$", fontsize=18)
    ax.set_ylabel("$Velocity$ $correlation,$ $C_{vv}^\\parallel(r)$", fontsize=18)
    fig.tight_layout()
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceVelCorr-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceDiff-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = 1/Dr
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceVelCorr-vsDr-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceDiff-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceVelCorr-vsDamping-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pSpaceDiff-vsDamping-" + figureName + "-Dr" + which
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    ax.plot(x, diff[:,0], color='k', lw=1.2, marker='s', markersize=8, fillstyle='none', label="$\\sqrt{\\int | C_{vv}^{\\parallel}(r) |^2 dr}$")
    #ax.plot(x, diff[:,1], color='g', lw=1.2, marker='v', markersize=8, fillstyle='none', label="$\\sqrt{\\int | C_{vv}^{\\perp}(r) |^2 dr}$")
    #ax.plot(x, diff[:,2], color='b', lw=1.2, marker='o', markersize=8, fillstyle='none', label="$\\sqrt{\\int | C_{vv}^{cross}(r) |^2 dr}$")
    if(fixed!="phi" and fixed!="iod"):
        ax.set_yscale('log')
    #ax.set_ylim(-0.12, 2.52)
    #ax.set_xlim(5.8e-06, 2.8e03)
    if(fixed=="phi"):
        ax.set_xscale('log')
    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Velocity$ $correlations$", fontsize=18, labelpad=5)
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelPhiPDF(dirName, figureName, fixed=False, which='1.5e-04'):
    fig, ax = plt.subplots(1,2, figsize=(12,5), dpi = 120)
    if(fixed=="Dr"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67',  'thermal78', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$\\varphi = 0.45$', '$\\varphi = 0.58$', '$\\varphi = 0.67$', '$\\varphi = 0.78$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        if(which == '0.45'):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod700', 'iod900', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 700$', '$\\beta \\sigma = 900$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 700, 900, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
    damping = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
            Dr[d] = ucorr.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
        if(os.path.exists(dirSample + "velPDFInCluster.dat")):
            if(d==0):
                damping[d] = ucorr.readFromDynParams(dirSample, "damping")
                f0 = ucorr.readFromDynParams(dirSample, "f0")
                ax[1].plot(np.ones(50)*f0/(2*damping[d]), np.linspace(-1,20,50), color='k', ls='dashed', lw=1)
            data = np.loadtxt(dirSample + "/velPDFInCluster.dat")
            ax[1].plot(data[:,0], data[:,1], color=colorList(d/dirList.shape[0]), lw=1.2)#, label=labelList[d])
        if(os.path.exists(dirSample + "velPDFOutCluster.dat")):
            data = np.loadtxt(dirSample + "/velPDFOutCluster.dat")
            ax[1].plot(data[:,0], data[:,1], color=colorList(d/dirList.shape[0]), lw=1.2, ls='--')
        if(os.path.exists(dirSample + "localVoroDensity-N16.dat")):
            data = np.loadtxt(dirSample + "localVoroDensity-N16.dat")
            ax[0].semilogy(data[1:,0], data[1:,1], color=colorList(d/dirList.shape[0]), lw=1.2)#, label=labelList[d])
            #ax[0].plot(data[data[:,1]>0,0], -np.log(data[data[:,1]>0,1]), color=colorList(d/dirList.shape[0]), lw=1.2, label=labelList[d])
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].set_xlabel("$Speed,$ $|\\vec{v}|$", fontsize=16)
    ax[1].set_ylabel("$Distribution,$ $P(|\\vec{v}|)$", fontsize=16)
    #ax[1].set_xscale('log')
    ax[1].set_ylim(-0.8,19.2)
    ax[0].set_xlabel("$Local$ $density,$ $\\varphi_{local}$", fontsize=16)
    ax[0].set_ylabel("$P(\\varphi_{local})$", fontsize=16)
    #ax[0].set_xlabel("$Local$ $density,$ $\\varphi$", fontsize=18)
    #ax[0].set_ylabel("$Free$ $energy,$ $F(\\varphi)$", fontsize=18)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    label = "$D_r$"
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=12, length=0)
    #cb.ax.invert_yaxis()
    ticklabels = [dirList[0], dirList[-1]]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=16, labelpad=-10, rotation='horizontal')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.22)
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pVelPhiPDF-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = 1/Dr
        xlabel = "$Persistent$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pVelPhiPDF-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pVelPhiPDF-vsDamping-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelPDF(dirName, figureName):
    fig, ax = plt.subplots(figsize=(6,3.5), dpi = 120)
    if(os.path.exists(dirName + "velPDFInCluster.dat")):
        data = np.loadtxt(dirName + "/velPDFInCluster.dat")
        ax.plot(data[:,0], data[:,1], color='b', lw=1.2, label="$Fluid$")
    if(os.path.exists(dirName + "velPDFOutCluster.dat")):
        data = np.loadtxt(dirName + "/velPDFOutCluster.dat")
        ax.plot(data[:,0], data[:,1], color='g', lw=1.2, ls='--', label="$Gas$")
        damping = ucorr.readFromDynParams(dirName, "damping")
        f0 = ucorr.readFromDynParams(dirName, "f0")
        ax.plot(np.ones(50)*f0/(2*damping), np.linspace(-1,20,50), color='k', ls='dashed', lw=1)
        print("active speed: ", f0/(2*damping))
    #ax.set_ylim(-0.8,17.2)
    #ax.set_xlim(-0.012,0.47)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Speed,$ $|\\vec{v}|$", fontsize=16)
    ax.set_ylabel("$Distribution,$ $P(|\\vec{v}|)$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pVelPDF-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPressurePDF(dirName, figureName):
    fig, ax = plt.subplots(figsize=(6,3.5), dpi = 120)
    if(os.path.exists(dirName + "pressurePDFInCluster.dat")):
        data = np.loadtxt(dirName + "/pressurePDFInCluster.dat")
        ax.plot(data[:,0], data[:,1], color='g', lw=1.2, label="$Virial$ $inside$ $cluster$")
        ax.plot(data[:,2], data[:,3], color='b', lw=1.2, label="$Thermal$ $inside$ $cluster$")
        ax.plot(data[:,4], data[:,5], color='r', lw=1.2, label="$Active$ $inside$ $cluster$")
    if(os.path.exists(dirName + "pressurePDFOutCluster.dat")):
        data = np.loadtxt(dirName + "/pressurePDFOutCluster.dat")
        ax.plot(data[:,0], data[:,1], color='g', lw=1.2, ls='--', label="$Virial$ $outside$ $cluster$")
        ax.plot(data[:,2], data[:,3], color='b', lw=1.2, ls='--', label="$Thermal$ $outside$ $cluster$")
        ax.plot(data[:,4], data[:,5], color='r', lw=1.2, ls='--', label="$Active$ $outside$ $cluster$")
    #ax.set_ylim(-0.8,17.2)
    #ax.set_xlim(-0.012,0.47)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Pressure,$ $p$", fontsize=16)
    ax.set_ylabel("$Distribution,$ $P(p)$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pPressurePDF-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def curve4Poly(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def fitPhiPDF(dirName, figureName, numBins):
    fig1, ax1 = plt.subplots(figsize=(5,4), dpi = 120)
    fig2, ax2 = plt.subplots(figsize=(6,3.5), dpi = 120)
    if(os.path.exists(dirName + "localDensity-N" + numBins + ".dat")):
        data = np.loadtxt(dirName + "localDensity-N" + numBins + ".dat")
        #data = data[1:-1]
        data = data[2:-4]
        ax2.semilogy(data[:,0], data[:,1], color='k', lw=1, marker='o', fillstyle='none')
        T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
        x = data[data[:,1]>0,0]
        y = -np.log(data[data[:,1]>0,1])
        ax1.plot(x, T*y, color='k', lw=1, marker='o', fillstyle='none', label="$data$")
        # interpolate
        spl = splrep(x, y, s=0.5)
        interX = np.linspace(np.min(x), np.max(x), 1000)
        interY = splev(interX, spl)
        #ax1.plot(interX, interY, lw=1, color='b', label="$smooth$ $interpolation$")
        # fit interpolation
        failed = False
        try:
            popt, pcov = curve_fit(curve4Poly, interX, interY)
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            ax1.plot(interX, T*curve4Poly(interX, *popt), color=[0,0.4,1], lw=2, linestyle='--', label="$4Poly$ $fit$")
        print("Fitting parameters: c4 ", popt[0], " c3 ", popt[1], " c2 ", popt[2], " c1 ", popt[3], " c0 ", popt[4])
        print("Mass: ", np.sqrt(popt[2]/(4*popt[0])))
        #y = np.linspace(-2, 8, 100)
        #ax1.plot(np.ones(100)*(-popt[1]/(4*popt[0])), y, linestyle='--', color='r', lw=1)
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    #ax1.legend(fontsize=12, loc='upper left')
    ax1.set_xlim(-0.007, 1.37)
    ax1.set_ylim(-0.0082, 0.028)
    ax2.set_xlim(0.27, 1.57)
    ax2.set_ylim(1.4e-03, 7.58)
    ax1.set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=16)
    ax1.set_ylabel("$Free$ $energy,$ $F(\\varphi_l)$", fontsize=16)
    ax2.set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=16)
    ax2.set_ylabel("$Distribution,$ $P(\\varphi_l)$", fontsize=16)
    fig1.tight_layout()
    fig2.tight_layout()
    figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pfitPhiFreeEnergy-" + figureName
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pfitPhiPDF-" + figureName
    fig2.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def fitPhiPDF2(dirName, figureName, numBins):
    fig, ax = plt.subplots(2, 1, figsize = (6,7.5), sharex=True, dpi = 120)
    if(os.path.exists(dirName + "localDensity-N" + numBins + ".dat")):
        data = np.loadtxt(dirName + "localDensity-N" + numBins + ".dat")
        #data = data[1:-1]
        data = data[2:-4]
        ax[0].semilogy(data[:,0], data[:,1], color='k', lw=1, marker='o', fillstyle='none')
        T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
        x = data[data[:,1]>0,0]
        y = -np.log(data[data[:,1]>0,1])
        ax[1].plot(x, T*y, color='k', lw=1, marker='o', fillstyle='none', label="$data$")
        # interpolate
        spl = splrep(x, y, s=0.5)
        interX = np.linspace(np.min(x), np.max(x), 1000)
        interY = splev(interX, spl)
        #ax1.plot(interX, interY, lw=1, color='b', label="$smooth$ $interpolation$")
        # fit interpolation
        failed = False
        try:
            popt, pcov = curve_fit(curve4Poly, interX, interY)
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            ax[1].plot(interX, T*curve4Poly(interX, *popt), color=[0,0.4,1], lw=2, linestyle='--', label="$4Poly$ $fit$")
        print("Fitting parameters: c4 ", popt[0], " c3 ", popt[1], " c2 ", popt[2], " c1 ", popt[3], " c0 ", popt[4])
        print("Mass: ", np.sqrt(popt[2]/(4*popt[0])))
        #y = np.linspace(-2, 8, 100)
        #ax1.plot(np.ones(100)*(-popt[1]/(4*popt[0])), y, linestyle='--', color='r', lw=1)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    #ax1.legend(fontsize=12, loc='upper left')
    ax[0].set_xlim(-0.007, 1.57)
    ax[0].set_ylim(1.4e-03, 7.58)
    ax[1].set_xlim(-0.007, 1.57)
    ax[1].set_ylim(-0.0082, 0.028)
    ax[0].set_ylabel("$Distribution,$ $P(\\varphi_l)$", fontsize=16)
    ax[1].set_ylabel("$Free$ $energy,$ $F(\\varphi_l)$", fontsize=16, labelpad=-5)
    ax[1].set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=16)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pfitPhiPDF-F-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPForceVelMagnitude(dirName, figureName, fixed=False, which='1.5e-04'):
    fig, ax = plt.subplots(figsize=(7.5,5), dpi = 120)
    if(fixed=="phi"):
        if(which == '0.45'):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    vmIn = np.zeros((dirList.shape[0], 3, 2))
    vmOut = np.zeros((dirList.shape[0], 3, 2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        if(os.path.exists(dirSample + "clusterVelMagnitude.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "clusterVelMagnitude.dat")
            # dense steric
            vmIn[d,0,0] = np.mean(data[:,1])
            vmIn[d,0,1] = np.std(data[:,1])
            # dense thermal
            vmIn[d,1,0] = np.mean(data[:,2])
            vmIn[d,1,1] = np.std(data[:,2])
            # dense active
            vmIn[d,2,0] = np.mean(data[:,3])
            vmIn[d,2,1] = np.std(data[:,3])
            # dilute steric
            vmOut[d,0,0] = np.mean(data[:,4])
            vmOut[d,0,1] = np.std(data[:,4])
            # dilute thermal
            vmOut[d,1,0] = np.mean(data[:,5])
            vmOut[d,1,1] = np.std(data[:,5])
            # dilute active
            vmOut[d,2,0] = np.mean(data[:,6])
            vmOut[d,2,1] = np.std(data[:,6])
    # interpolate to find tradeoff point
    #check = False
    #for i in range(vel.shape[0]):
    #    if(force[i,0] > vel[i,0]):
    #        check = True
    #if(check):
    #    index = np.argwhere(force[:,0]>vel[:,0])[-1,0]
    #    t = np.linspace(taup[index], taup[index+1],100)
    #    forceSlope = (force[index+1,0] - force[index,0]) / (taup[index+1] - taup[index])
    #    velSlope = (vel[index+1,0] - vel[index,0]) / (taup[index+1] - taup[index])
    #    forceInter = force[index,0] + forceSlope*(t - taup[index])
    #    velInter = vel[index,0] + velSlope*(t - taup[index])
    #    taupc = t[np.argwhere(forceInter>velInter)[-1,0]]
    #    print(taup[index], taup[index+1], taupc)
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pVelMagnitude-vsDr-" + figureName
        ax.set_xscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pVelMagnitude-vsPhi-" + figureName
    ax.errorbar(x[vmIn[:,0,0]>0], vmIn[vmIn[:,0,0]>0,0,0], vmIn[vmIn[:,0,0]>0,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
    ax.errorbar(x[vmIn[:,1,0]>0], vmIn[vmIn[:,1,0]>0,1,0], vmIn[vmIn[:,1,0]>0,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
    ax.errorbar(x[vmIn[:,2,0]>0], vmIn[vmIn[:,2,0]>0,2,0], vmIn[vmIn[:,2,0]>0,2,1], lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Active$')
    ax.errorbar(x[vmOut[:,0,0]>0], vmOut[vmOut[:,0,0]>0,0,0], vmOut[vmOut[:,0,0]>0,0,1], ls='--', lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax.errorbar(x[vmOut[:,1,0]>0], vmOut[vmOut[:,1,0]>0,1,0], vmOut[vmOut[:,1,0]>0,1,1], ls='--', lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3)
    ax.errorbar(x[vmOut[:,2,0]>0], vmOut[vmOut[:,2,0]>0,2,0], vmOut[vmOut[:,2,0]>0,2,1], ls='--', lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3)
    #if(check):
    #    ax.plot(np.ones(100)*taupc, np.linspace(np.min(force), np.max(vel), 100), ls='--', color='k')
    #    ax.plot(t, forceInter, lw=2, color='g')
    #    ax.plot(t, velInter, lw=2, color='r')
    #    phi = ucorr.readFromParams(dirName, "phi")
    #    np.savetxt(dirName + "iod" + which + "/active-langevin/forceVelTradeoff.dat", np.array([phi, taupc]))
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Velocity$ $magnitude$", fontsize=18)
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDropletPressure(dirName, figureName, fixed='temp', which='0.0023'):
    fig1, ax1 = plt.subplots(figsize = (7,4), dpi = 120)
    fig2, ax2 = plt.subplots(figsize = (7,4), dpi = 120)
    if(fixed=="phi"):
        phi = ucorr.readFromParams(dirName, "phi")
        dirList = np.array(['0.0023', '0.02', '0.2'])
        temp = np.zeros(dirList.shape[0])
    elif(fixed=="temp"):
        dirList = np.array(['thermal25', 'thermal27', 'thermal30'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    border = np.zeros((dirList.shape[0], 3, 2))
    droplet = np.zeros((dirList.shape[0], 3, 2))
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "/langevin-u/T" + dirList[d] + "-u0.035/dynamics/"
            temp[d] = np.mean(np.loadtxt(dirSample + "energy.dat")[:,4])
        elif(fixed=="temp"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin-u/T" + which + "-u0.035/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        if(os.path.exists(dirSample + "/pressure.dat")):
            data = np.loadtxt(dirSample + "/pressure.dat")
            # border
            border[d,0,0] = np.mean(data[:,1])
            border[d,0,1] = np.std(data[:,1])
            border[d,1,0] = np.mean(data[:,2])
            border[d,1,1] = np.std(data[:,2])
            border[d,2,0] = np.mean(data[:,1] + data[:,2])
            border[d,2,1] = np.std(data[:,1] + data[:,2])
            # droplet
            droplet[d,0,0] = np.mean(data[:,4])
            droplet[d,0,1] = np.std(data[:,4])
            droplet[d,1,0] = np.mean(data[:,5])
            droplet[d,1,1] = np.std(data[:,5])
            droplet[d,2,0] = np.mean(data[:,4] + data[:,5])
            droplet[d,2,1] = np.std(data[:,4] + data[:,5])
    if(fixed=="phi"):
        x = temp
        xlabel = "$Temperature,$ $T$"
        loc1 = 'lower right'
        loc2 = 'upper left'
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pDropletPressure-vsT-" + figureName + "-phi" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTotalDropletPressure-vsT-" + figureName + "-phi" + which
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        loc1 = 'lower right'
        loc2 = 'upper left'
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pDropletPressure-vsPhi-" + figureName + "-T" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pTotalDropletPressure-vsPhi-" + figureName + "-T" + which
        #ax1.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    # droplet
    ax1.errorbar(x, droplet[:,0,0], droplet[:,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
    ax1.errorbar(x, droplet[:,1,0], droplet[:,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
    # border
    ax1.errorbar(x, border[:,0,0], border[:,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, ls = 'dotted')
    ax1.errorbar(x, border[:,1,0], border[:,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, ls = 'dotted')
    # total
    ax2.errorbar(x, droplet[:,2,0], droplet[:,2,1], lw=1.2, color='b', marker='s', markersize=8, fillstyle='none', capsize=3, label='$Droplet$')
    #ax2.errorbar(x, border[:,2,0], border[:,2,1], lw=1.2, color='g', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Border$')
    #ax.set_xlim(5.8e-06, 2.8e03)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel("$Pressure,$ $P \\sigma^2$", fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("$Pressure,$ $P \\sigma^2$", fontsize=18)
    ax1.legend(fontsize=12, loc='best')
    ax2.legend(fontsize=12, loc='best')
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig2.savefig(figure1Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterDensity(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="phi"):
        phi = ucorr.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['thermal25', 'thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        dirList = np.array(['0.25', '0.26', '0.27', '0.28', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    fluidDensity = np.zeros((dirList.shape[0],2))
    gasDensity = np.zeros((dirList.shape[0],2))
    density = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            #phi[d] = ucorr.readFromParams(dirSample, "phi")
            #phi[d] = np.loadtxt(dirSample + 'localDelaunayDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
            if(d==0):
                numParticles = ucorr.readFromParams(dirSample, "numParticles")
        if(os.path.exists(dirSample + "delaunayDensity.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "delaunayDensity.dat")
            fluidDensity[d,0] = np.mean(data[:,1])
            fluidDensity[d,1] = np.std(data[:,1])
            gasDensity[d,0] = np.mean(data[:,2])
            gasDensity[d,1] = np.std(data[:,2])
            phi[d] = np.mean(data[:,3])
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterVoro-vsPhi-" + figureName
        ax.plot(x, x, color='k', lw=1.2, ls='--')
        # interpolate to find upper bound
        fluid = fluidDensity[fluidDensity[:,0]>0,0][10:]
        error = fluidDensity[fluidDensity[:,0]>0,1][10:]
        phiup = phi[fluidDensity[:,0]>0][10:]
        index = np.argwhere((fluid - error*0.5) <= phiup)[0,0]
        error = error[index]*0.5
        p = np.linspace(phiup[index-1], phiup[index],100)
        fluidSlope = (fluid[index] - fluid[index-1]) / (phiup[index] - phiup[index-1])
        phiSlope = (phiup[index] - phiup[index-1]) / (phiup[index] - phiup[index-1])
        fluidInter = fluid[index-1] + fluidSlope*(p - phiup[index-1])
        phiInter = phiup[index-1] + phiSlope*(p - phiup[index-1])
        phiupper = p[np.argwhere((fluidInter - error) <= phiInter)[0,0]]
        ax.plot(phiInter, phiInter, color='r')
        ax.plot(phiInter, fluidInter, color='r', ls='--', lw=4)
        # interpolate to find lower bound
        gas = gasDensity[gasDensity[:,0]>0,0][:10]
        philow = phi[fluidDensity[:,0]>0][:10]
        index = np.argwhere((gas + 1e-02) < philow)[0,-1]
        p = np.linspace(philow[index-1], philow[index],100)
        gasSlope = (gas[index] - gas[index-1]) / (philow[index] - philow[index-1])
        phiSlope = (philow[index] - philow[index-1]) / (philow[index] - philow[index-1])
        gasInter = gas[index-1] + gasSlope*(p - philow[index-1])
        phiInter = philow[index-1] + phiSlope*(p - philow[index-1])
        philower = p[np.argwhere((gasInter + 1e-02) < phiInter)[0,-1]]
        ax.plot(phiInter, phiInter, color='r')
        ax.plot(phiInter, gasInter, color='r', ls='--', lw=4)
        print("phiup:", phiupper, "phidown:", philower)
        np.savetxt(dirName + "MIPSBounds.dat", np.column_stack((numParticles, philower, phiupper)))
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterVoro-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Area$ $fraction$", fontsize=18)
    ax.errorbar(x[fluidDensity[:,0]>0], fluidDensity[fluidDensity[:,0]>0,0], fluidDensity[fluidDensity[:,0]>0,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Fluid$')
    ax.errorbar(x[gasDensity[:,0]>0], gasDensity[gasDensity[:,0]>0,0], gasDensity[gasDensity[:,0]>0,1], color='g', lw=1.2, marker='o', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Gas$')
    ax.legend(fontsize=14, loc='best')
    ax.set_ylim(-0.18, 1.02)
    y = np.linspace(-0.2, 1.05, 100)
    dirSample = dirName + os.sep + "0.30/active-langevin/Dr" + which + "/dynamics/"
    phi1 = np.mean(np.loadtxt(dirSample + "delaunayDensity.dat")[:,3])
    dirSample = dirName + os.sep + "0.31/active-langevin/Dr" + which + "/dynamics/"
    phi2 = np.mean(np.loadtxt(dirSample + "delaunayDensity.dat")[:,3])
    ax.plot(np.ones(100)*(phi1+phi2)/2, y, ls='dotted', color='k', lw=1)
    if(fixed!="Dr"):
        ax.set_xscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterShape(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="phi"):
        phi = ucorr.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['thermal25', 'thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        dirList = np.array(['0.25', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    meanShape = np.zeros((dirList.shape[0],3))
    errorShape = np.zeros((dirList.shape[0],3))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            #phi[d] = ucorr.readFromParams(dirSample, "phi")
            phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
            #print(phi[d], ucorr.readFromParams(dirSample, "phi"))
            if(d==0):
                numParticles = ucorr.readFromParams(dirSample, "numParticles")
        if(os.path.exists(dirSample + "shapeParameter.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "shapeParameter.dat")
            meanShape[d,0] = np.mean(data[:,1])
            errorShape[d,0] = np.std(data[:,1])
            meanShape[d,1] = np.mean(data[:,2])
            errorShape[d,1] = np.std(data[:,2])
            meanShape[d,2] = np.mean(data[:,3])
            errorShape[d,2] = np.std(data[:,3])
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterShape-vsPhi-" + figureName
        ax.plot(x, phi, color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterShape-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Shape$ $parameter$", fontsize=18)
    ax.errorbar(x[meanShape[:,2]>0], meanShape[meanShape[:,2]>0,2], errorShape[meanShape[:,2]>0,2], color='k', lw=1.2, marker='s', markersize = 8, fillstyle='none', elinewidth=1, capsize=4)
    if(fixed!="Dr"):
        ax.set_xscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotClusterShapeVSTime(dirName, figureName):
    #numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    shape = np.loadtxt(dirName + os.sep + "shapeParam.dat")
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    ax.plot(shape[:,0], shape[:,3], linewidth=1.2, color='k', ls='solid')
    # plotting settings
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax.set_ylabel("$Shape$ $parameter$", fontsize=15)
    ax.legend(fontsize=12, loc='best')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterShapeVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSurfaceTensionVSTime(dirName, figureName):
    #numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    data = np.loadtxt(dirName + os.sep + "surfaceWork.dat")
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    ax.plot(data[1:,0], (data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]), linewidth=1.2, color='k')
    # plotting settings
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax.set_ylabel("$Surface$ $tension,$ $\\Delta W / \\Delta l_c$", fontsize=15)
    ax.legend(fontsize=12, loc='best')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pSurfaceTensionVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSurfaceTension(dirName, figureName, fixed='Dr', which='2e-04'):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="phi"):
        phi = ucorr.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        taup = np.zeros(dirList.shape[0])
    elif(fixed=="Dr"):
        #dirList = np.array(['30', '35', '40', '45', '52', '58', '62', '67', '72',  '78', '82', '85', '88', '91', '94'])
        dirList = np.array(['25', '28', '29', '30', '31', '32', '35', '40', '45', '50', '55', '60', '65', '70',  '75', '80', '82', '84', '86', '88', '90', '92', '94', '96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    tension = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(ucorr.readFromParams(dirSample, "Dr") * ucorr.readFromParams(dirSample, "sigma"))
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + "0." + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
        if(os.path.exists(dirSample + "/borderEnergy.dat")):
            data = np.loadtxt(dirSample + "/surfaceWork.dat")
            tension[d,0] = np.mean((data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]))
            tension[d,1] = np.std((data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]))
    ax.tick_params(axis='both', labelsize=14)
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pSurfaceTension-vsDr-" + figureName + "-phi" + which
        ax.set_xscale('log')
        #ax.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pSurfaceTension-vsPhi-" + figureName + "-Dr" + which
    ax.errorbar(x[tension[:,0]>0], tension[tension[:,0]>0,0], tension[tension[:,0]>0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Surface$ $tension,$ $\\gamma$", fontsize=18, labelpad=15)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterMixingTime(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(fixed=="phi"):
        phi = ucorr.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        dirList = np.array(['0.26', '0.28', '0.30', '0.32', '0.35', '0.40', '0.45', '0.50'])#, '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    mixingTime = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/short/"
            #phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]
        if(os.path.exists(dirSample + "logMixingTime.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "logMixingTime.dat")
            ax.errorbar(data[:,0], data[:,1], data[:,2], color=colorList(d/dirList.shape[0]), lw=1, marker='o', capsize=3, label="$\\varphi=$" + dirList[d])
    ax.set_xscale('log')
    if(fixed=="Dr"):
        #x = phi
        #xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pMxingTime-vsPhi-" + figureName
    elif(fixed=="phi"):
        #x = taup
        #xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pMixingTime-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.set_xlabel("$Elapsed$ $time,$ $\\Delta t$", fontsize=18)
    ax.set_ylabel("$Mixing$ $ratio$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPTotalPressure(dirName, figureName, fixed='Dr', which='2e-04'):
    fig, ax = plt.subplots(figsize = (7,4), dpi = 120)
    if(fixed=="phi"):
        phi = ucorr.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['30', '35', '40', '45', '52', '58', '62', '67', '72',  '78', '82', '85', '88', '91', '94'])
        dirList = np.array(['25', '28', '29', '30', '31', '32', '35', '40', '45', '50', '55', '60', '65', '70',  '75', '80', '82', '84', '86', '88', '90', '92', '94', '96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    p = np.zeros((dirList.shape[0], 4, 2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + "thermal" + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            dirSample = dirName + os.sep + "0." + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            #phi[d] = ucorr.readFromParams(dirSample, "phi")
            phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
            #print(phi[d], ucorr.readFromParams(dirSample, "phi"))
        if(os.path.exists(dirSample + "/pressure.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "/pressure.dat")
            sigma = np.mean(np.loadtxt(dirSample + "/particleRad.dat"))
            # steric
            p[d,0,0] = np.mean(data[:,2])
            p[d,0,1] = np.std(data[:,2])
            # thermal
            p[d,1,0] = np.mean(data[:,3])
            p[d,1,1] = np.std(data[:,3])
            # active
            p[d,2,0] = np.mean(data[:,4])
            p[d,2,1] = np.std(data[:,4])
            # dense total
            p[d,3,0] = np.mean(data[:,2] + data[:,3] + data[:,4])
            p[d,3,1] = np.std(data[:,2] + data[:,3] + data[:,4])
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pTotPressure-vsDr-" + figureName + "-phi" + which
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pTotPressure-vsPhi-" + figureName + "-Dr" + which
    ax.tick_params(axis='both', labelsize=14)
    # pressure components
    ax.errorbar(x[p[:,0,0]>0], p[p[:,0,0]>0,0,0], p[p[:,0,0]>0,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
    ax.errorbar(x[p[:,1,0]>0], p[p[:,1,0]>0,1,0], p[p[:,1,0]>0,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
    ax.errorbar(x[p[:,2,0]>0], p[p[:,2,0]>0,2,0], p[p[:,2,0]>0,2,1], lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Actve$')
    #ax.errorbar(x[p[:,3,0]>0], p[p[:,3,0]>0,3,0], p[p[:,3,0]>0,3,1], lw=1.2, color='b', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Total$')
    #ax.set_xlim(5.8e-06, 2.8e03)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Pressure,$ $P \\sigma^2$", fontsize=18)
    ax.legend(fontsize=14, loc='best')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterPressure(dirName, figureName, fixed='Dr', inter=False, which='gasFluid'):
    fig1, ax1 = plt.subplots(figsize = (7,4), dpi = 120)
    fig2, ax2 = plt.subplots(figsize = (7,4), dpi = 120)
    if(fixed=="phi"):
        phi = ucorr.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['thermal30', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85', 'thermal85', 'thermal88', 'thermal94'])
        dirList = np.array(['0.25', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    pIn = np.zeros((dirList.shape[0], 3, 2))
    pOut = np.zeros((dirList.shape[0], 3, 2))
    ptotIn = np.zeros((dirList.shape[0], 2))
    ptotOut = np.zeros((dirList.shape[0], 2))
    deltap = np.zeros((dirList.shape[0], 2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr1.5e-04/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr2e-04/dynamics/"
            if(os.path.exists(dirSample + "localVoroDensity-N16-stats.dat")):
                phi[d] = np.loadtxt(dirSample + "localVoroDensity-N16-stats.dat")[0]
            else:
                phi[d] = spCorr.averageLocalVoronoiDensity(dirSample)
        if(os.path.exists(dirSample + "/delaunayPressure.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "/delaunayPressure.dat")
            # dense steric
            pIn[d,0,0] = np.mean(data[:,2])
            pIn[d,0,1] = np.std(data[:,2])
            # dense thermal
            pIn[d,1,0] = np.mean(data[:,3])
            pIn[d,1,1] = np.std(data[:,3])
            # dense active
            pIn[d,2,0] = np.mean(data[:,4])
            pIn[d,2,1] = np.std(data[:,4])
            # dense total
            ptotIn[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4])
            ptotIn[d,1] = np.std(data[:,2] + data[:,3] + data[:,4])
            # dilute steric
            pOut[d,0,0] = np.mean(data[:,5])
            pOut[d,0,1] = np.std(data[:,5])
            # thermal
            pOut[d,1,0] = np.mean(data[:,6])
            pOut[d,1,1] = np.std(data[:,6])
            # active
            pOut[d,2,0] = np.mean(data[:,7])
            pOut[d,2,1] = np.std(data[:,7])
            # dilute total
            ptotOut[d,0] = np.mean(data[:,5] + data[:,6] + data[:,7])
            ptotOut[d,1] = np.std(data[:,5] + data[:,6] + data[:,7])
            # delta pressure
            deltap[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
            deltap[d,1] = np.std(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
    # interpolate to find tradeoff point
    if(inter == 'inter'):
        check = False
        if(which == "gasFluid"):
            bottom = ptotIn[ptotIn[:,0]>0,0]
            top = ptotOut[ptotIn[:,0]>0,0]
            tau = taup[ptotIn[:,0]>0]
        elif(which == "stericActive"):
            bottom = pIn[pIn[:,2,0]>0,0,0]
            top = pOut[pOut[:,2,0]>0,2,0]
            tau = taup[pIn[:,2,0]>0]
        for i in range(top.shape[0]):
            if(top[i] > bottom[i]):
                check = True
        if(check):
            index = np.argwhere(bottom < top)[0,0]
            t = np.linspace(tau[index-1], tau[index],100)
            bottomSlope = (bottom[index] - bottom[index-1]) / (tau[index] - tau[index-1])
            topSlope = (top[index] - top[index-1]) / (tau[index] - tau[index-1])
            bottomInter = bottom[index-1] + bottomSlope*(t - tau[index-1])
            topInter = top[index-1] + topSlope*(t - tau[index-1])
            taupc = t[np.argwhere(bottomInter < topInter)[0,0]]
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[index] + "/dynamics/"
            if(os.path.exists(dirSample + "localVoroDensity-N16-stats.dat")):
                phic = np.loadtxt(dirSample + "localVoroDensity-N16-stats.dat")[0]
            else:
                phic = spCorr.averageLocalVoronoiDensity(dirSample)
            print("interpolation indices:", index-1, index, "tauc:", taupc, "density:", phic)
            np.savetxt(dirName + "iod10/active-langevin/" + which + "Tradeoff.dat", np.array([phic, taupc]))
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pGasFluid-vsDr-" + figureName
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pPressures-vsDr-" + figureName
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pGasFLuid-vsPhi-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pPressures-vsPhi-" + figureName + "-Dr" + which
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    # dense and dilute pressure
    ax1.errorbar(x[ptotIn[:,0]>0], ptotIn[ptotIn[:,0]>0,0], ptotIn[ptotIn[:,0]>0,1], color='b', marker='s', markersize=8, fillstyle='none', lw=1.2, capsize=3, label='$Fluid$')
    ax1.errorbar(x[ptotOut[:,0]>0], ptotOut[ptotOut[:,0]>0,0], ptotOut[ptotOut[:,0]>0,1], color='g', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3, label='$Gas$')
    #ax1.errorbar(x[deltap[:,0]!=0], np.abs(deltap[deltap[:,0]!=0,0]), deltap[deltap[:,0]!=0,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3, label='$\\Delta p$')
    # pressure components
    if(fixed=='phi'):
        ax2.errorbar(x[pIn[:,0,0]>0], pIn[pIn[:,0,0]>0,0,0], pIn[pIn[:,0,0]>0,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
        ax2.errorbar(x[pIn[:,1,0]>0], pIn[pIn[:,1,0]>0,1,0], pIn[pIn[:,1,0]>0,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
        ax2.errorbar(x[pIn[:,2,0]>0], pIn[pIn[:,2,0]>0,2,0], pIn[pIn[:,2,0]>0,2,1], lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Actve$')
        ax2.errorbar(x[pOut[:,0,0]>0], pOut[pOut[:,0,0]>0,0,0], pOut[pOut[:,0,0]>0,0,1], ls='--', lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax2.errorbar(x[pOut[:,1,0]>0], pOut[pOut[:,1,0]>0,1,0], pOut[pOut[:,1,0]>0,1,1], ls='--', lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3)
        ax2.errorbar(x[pOut[:,2,0]>0], pOut[pOut[:,2,0]>0,2,0], pOut[pOut[:,2,0]>0,2,1], ls='--', lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3)
    else:
        ax2.errorbar(x, pIn[:,0,0], pIn[:,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
        ax2.errorbar(x, pIn[:,1,0], pIn[:,1,1], lw=1.2, color='r', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
        ax2.errorbar(x, pIn[:,2,0], pIn[:,2,1], lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Active$')
        ax2.errorbar(x, pOut[:,0,0], pOut[:,0,1], ls='--', lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax2.errorbar(x, pOut[:,1,0], pOut[:,1,1], ls='--', lw=1.2, color='r', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax2.errorbar(x, pOut[:,2,0], pOut[:,2,1], ls='--', lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3)
    if(inter == 'inter' and check):
        if(which=='gasFluid'):
            ax1.plot(np.ones(100)*taupc, np.linspace(np.min(ptotOut[:,0]), np.max(ptotIn[:,0]), 100), lw=0.7, ls='--', color='k')
            ax1.plot(t, topInter, lw=2, color='r')
            ax1.plot(t, bottomInter, lw=2, color='k')
        else:
            ax2.plot(np.ones(100)*taupc, np.linspace(np.min(pOut[:,0]), np.max(pIn[:,0]), 100), lw=0.7, ls='--', color='k')
            ax2.plot(t, topInter, lw=2, color='r')
            ax2.plot(t, bottomInter, lw=2, color='k')
    #ax.set_xlim(5.8e-06, 2.8e03)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel("$Total$ $pressure$", fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("$Pressure$ $components$", fontsize=18)
    ax1.legend(fontsize=14, loc='best')
    ax2.legend(fontsize=14, loc='best')
    fig1.tight_layout()
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig2.tight_layout()
    fig2.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterTradeoff(dirName, figureName, which='gasFluid'):
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    dirList = np.array(['thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal83', 'thermal88'])
    taupc = np.zeros(dirList.shape[0])
    phi = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/"
        if(os.path.exists(dirSample + os.sep + which + "Tradeoff.dat")):
            data = np.loadtxt(dirSample + os.sep + which + "Tradeoff.dat")
            phi[d] = data[0]
            taupc[d] = data[1]
    ax.plot(taupc[taupc>0], phi[taupc>0], color='k', lw=1, marker='o', fillstyle='none')
    np.savetxt(dirName + os.sep + which + "Tradeoff.dat", np.column_stack((phi[taupc>0], taupc[taupc>0])))
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterTradeoff-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Density$", fontsize=18)
    ax.set_xlabel("$Tradeoff$ $time,$ $\\tau_p^*$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPhaseDiagram(dirName, numBins, figureName, which='16', log=False):
    fig, ax = plt.subplots(figsize=(7.5,6), dpi = 120)
    phi = np.array(['25', '27', '30', '35', '40', '45', '52', '58', '62', '67', '72', '78', '82', '85', '88', '91', '94'])
    Dr = np.array(['100', '10', '5', '1', '5e-01', '1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    taup = np.zeros((phi.shape[0], Dr.shape[0]))
    # load the data
    deltaPhi = np.zeros((phi.shape[0], Dr.shape[0]))
    meanPhi = np.zeros((phi.shape[0], Dr.shape[0]))
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            dirSample = dirName + 'thermal' + phi[i] + '/langevin/T0.001/iod10/active-langevin/Dr' + Dr[j] + '/dynamics/'
            deltaFile = dirSample + 'localVoroDensity-N' + which + '-stats.dat'#'localDensity-N16-stats.dat'
            #phiFile = dirSample + 'voronoiDensity.dat'
            phiFile = dirSample + 'delaunayDensity.dat'
            if(os.path.exists(deltaFile)):
                #data = np.loadtxt(phiFile)
                #meanPhi[i,j] = np.mean(data[:,3])
                data = np.loadtxt(deltaFile)
                meanPhi[i,j] = data[0]
                deltaPhi[i,j] = data[1]
                taup[i,j] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
    # assign color based on deltaPhi
    colorId = np.zeros((phi.shape[0], Dr.shape[0]))
    min = np.min(deltaPhi)
    max = np.max(deltaPhi)
    if(log=='log'):
        bins = np.geomspace(min+1e-04, max+0.001, numBins)
    else:
        bins = np.linspace(min, max+0.001, numBins)
    print('minimum intensity: ', min, ' maximum intensity: ', max)
    colorMap = cm.get_cmap('inferno', numBins)
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            for k in range(numBins-1):
                if(deltaPhi[i,j] > bins[k] and deltaPhi[i,j] < bins[k+1]):
                    ax.semilogx(taup[i,j], meanPhi[i,j], color=colorMap((numBins-k)/numBins), marker='s', markersize=15, lw=0)
    #data = np.loadtxt(dirName + "/gasFluidTradeoff.dat")
    #ax.plot(data[:,1], data[:,0], color='k', marker='o', markersize=6, markeredgewidth=1.2, fillstyle='none', lw=1)
    #data = np.loadtxt(dirName + "/stericActiveTradeoff.dat")#[0.3,0.7,0]
    #ax.plot(data[:,1], data[:,0], color='k', marker='v', markersize=8, markeredgewidth=1.2, fillstyle='none', lw=1)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Average$ $local$ $density,$ $\\langle \\varphi_l \\rangle$", fontsize=18)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=18)
    ax.set_ylim(0.22, 0.95)
    ax.plot(np.ones(50)*1e06, np.linspace(0,1.2,50), ls='dotted', color='k', lw=0.7)
    colorBar = cm.ScalarMappable(cmap=colorMap)
    cb = plt.colorbar(colorBar)
    label = "$\\Delta \\varphi_l$"#"$\\Delta \\varphi^2_{16}}$"
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=14, length=0)
    cb.ax.invert_yaxis()
    ticklabels = [np.format_float_positional(max, 2), np.format_float_positional(min, 2)]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=18, labelpad=10, rotation='horizontal')
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pPhaseDiagram-" + figureName
    if(log=='log'):
        figureName += "-" + log
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPhaseDiagramDeltaPressure(dirName, numBins, figureName, which='16'):
    figp, axp = plt.subplots(figsize=(7.5,5), dpi = 120)
    labelList = np.array(['$\\varphi=0.25$', '$\\varphi=0.27$', '$\\varphi=0.30$', '$\\varphi=0.35$', '$\\varphi=0.40$', '$\\varphi=0.45$', '$\\varphi=0.52$', '$\\varphi=0.58$', '$\\varphi=0.62$', '$\\varphi=0.67$', '$\\varphi=0.72$', '$\\varphi=0.78$', '$\\varphi=0.82$', '$\\varphi=0.85$', '$\\varphi=0.88$', '$\\varphi=0.91$', '$\\varphi=0.94$'])
    phi = np.array(['25', '27', '30', '35', '40', '45', '52', '58', '62', '67', '72', '78', '82', '85', '88', '91', '94'])
    Dr = np.array(['5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    colorList = cm.get_cmap('viridis', phi.shape[0])
    taup = np.zeros((phi.shape[0], Dr.shape[0]))
    # load the data
    voroPhi = np.zeros((phi.shape[0], Dr.shape[0]))
    deltaPressure = np.zeros((phi.shape[0], Dr.shape[0],2))
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            dirSample = dirName + 'thermal' + phi[i] + '/langevin/T0.001/iod10/active-langevin/Dr' + Dr[j] + '/dynamics/'
            fileName1 = dirSample + 'localVoroDensity-N' + which + '-stats.dat'
            fileName2 = dirSample + 'clusterPressure.dat'
            if(os.path.exists(fileName1) and os.path.exists(fileName2)):
                data = np.loadtxt(fileName1)
                voroPhi[i,j] = data[0]
                data = np.loadtxt(fileName2)
                deltaPressure[i,j,0] = np.mean(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
                deltaPressure[i,j,1] = np.std(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
                taup[i,j] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
        if(float(phi[i]) < 89 and float(phi[i]) > 28):
            if(deltaPressure[i,j,0] < 0):
                axp.errorbar(taup[i,deltaPressure[i,:,0]!=0], -deltaPressure[i,deltaPressure[i,:,0]!=0,0], deltaPressure[i,deltaPressure[i,:,0]!=0,1], marker='o', capsize=3, color=colorList(i/phi.shape[0]), lw=1.2, label=labelList[i])
            else:
                axp.errorbar(taup[i,deltaPressure[i,:,0]!=0], deltaPressure[i,deltaPressure[i,:,0]!=0,0], deltaPressure[i,deltaPressure[i,:,0]!=0,1], marker='o', capsize=3, color=colorList(i/phi.shape[0]), lw=1.2, label=labelList[i])
    axp.set_xscale('log')
    axp.set_yscale('log')
    axp.tick_params(axis='both', labelsize=14)
    axp.set_ylabel("$Pressure$ $difference,$ $|\\Delta p| \\sigma^2$", fontsize=18)
    axp.set_xlabel("$Persistence$ $time,$ $\\tau_p \\sigma$", fontsize=18)
    axp.legend(fontsize=10, loc='best', ncol=2)
    figp.tight_layout()
    figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pDeltaPressureVSPhi-" + figureName
    figp.savefig(figure1Name + ".png", transparent=True, format = "png")
    # assign color based on deltaPhi
    fig, ax = plt.subplots(figsize=(7.5,6), dpi = 120)
    ax.plot(np.ones(50)*1e06, np.linspace(0,1.2,50), ls='dotted', color='k', lw=0.7)
    colorId = np.zeros((phi.shape[0], Dr.shape[0]))
    colorMap = cm.get_cmap('inferno', numBins)
    pmagnitude = np.abs(deltaPressure[:,:,0])
    min = np.min(pmagnitude)
    max = np.max(pmagnitude)
    print('minimum intensity: ', min, ' maximum intensity: ', max)
    bins = np.geomspace(min, max, numBins)
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            for k in range(numBins-1):
                if(pmagnitude[i,j] > bins[k] and pmagnitude[i,j] < bins[k+1]):
                    if(deltaPressure[i,j,0] < 0):
                        ax.semilogx(taup[i,j], voroPhi[i,j], color=colorMap((numBins-k)/numBins), marker='s', markersize=15, lw=0, markeredgewidth=2.5, fillstyle='none')
                    else:
                        ax.semilogx(taup[i,j], voroPhi[i,j], color=colorMap((numBins-k)/numBins), marker='s', markersize=15, lw=0)
    #data = np.loadtxt(dirName + "/gasFluidTradeoff.dat")
    #ax.plot(data[:,1], data[:,0], color='k', marker='o', markersize=6, markeredgewidth=1.2, fillstyle='none', lw=1)
    #data = np.loadtxt(dirName + "/stericActiveTradeoff.dat")#[0.3,0.7,0]
    #ax.plot(data[:,1], data[:,0], color='k', marker='v', markersize=8, markeredgewidth=1.2, fillstyle='none', lw=1)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Average$ $local$ $density,$ $\\langle \\varphi_l \\rangle$", fontsize=18)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p \\sigma$", fontsize=18)
    ax.set_ylim(0.22, 0.95)
    colorBar = cm.ScalarMappable(cmap=colorMap)
    cb = plt.colorbar(colorBar)
    label = "$|\\Delta p|$"
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cb.ax.tick_params(labelsize=12, length=0)
    cb.ax.invert_yaxis()
    spacing = (max - min)/5
    ticklabels = [np.format_float_scientific(max, 2), np.format_float_scientific(min + 4*spacing, 2), np.format_float_scientific(min + 3*spacing, 2), np.format_float_scientific(min + 2*spacing, 2), np.format_float_scientific(min + spacing, 2), np.format_float_scientific(min, 2)]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=18, labelpad=10, rotation='horizontal')
    fig.tight_layout()
    figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pDeltaPressurePhaseDiagram-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDeltaPVSPhi(dirName, figureName, which='2e-04'):
    fig, ax = plt.subplots(figsize = (7,5), dpi = 120)
    #dirList = np.array(['thermal25', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85', 'thermal85', 'thermal88'])
    dirList = np.array(['0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50'])
    deltap = np.zeros((dirList.shape[0], 2))
    clusterRad = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
        dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
        if(os.path.exists(dirSample + "/clusterPressure.dat")):
            data = np.loadtxt(dirSample + "/clusterPressure.dat")
            # delta pressure
            deltap[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
            deltap[d,1] = np.std(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
        if(os.path.exists(dirSample + "/voronoiArea.dat")):
            data = np.loadtxt(dirSample + "/voronoiArea.dat")
            # fluid radius
            clusterRad[d,0] = np.mean(np.sqrt(data[:,1]/np.pi))
            clusterRad[d,1] = np.std(np.sqrt(data[:,1]/np.pi))
    ax.tick_params(axis='both', labelsize=14)
    # plot delta versus fluid radius
    ax.errorbar(clusterRad[deltap[:,0]!=0,0], deltap[deltap[:,0]!=0,0]/clusterRad[deltap[:,0]!=0,0], deltap[deltap[:,0]!=0,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3)
    ax.set_xlabel('$Fluid$ $radius,$ $R_c$', fontsize=18)
    ax.set_ylabel('$Pressure$ $difference,$ $\\Delta p \\sigma^2$', fontsize=18)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pDeltaP-vsPhi-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDeltaPVSSystemSize(dirName, figureName, which='1.5e-04'):
    fig, ax = plt.subplots(figsize = (7,5), dpi = 120)
    dirList = np.array(['1024', '2048', '4096', '8192'])
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$'])
    numParticles = np.array([1024, 2048, 4096, 8192])
    deltap = np.zeros((dirList.shape[0], 2))
    clusterRad = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/thermal45/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
        if(os.path.exists(dirSample + "/clusterPressure.dat")):
            data = np.loadtxt(dirSample + "/clusterPressure.dat")
            sigma = ucorr.readFromDynParams(dirSample, "sigma")
            # delta pressure
            deltap[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
            deltap[d,1] = np.std(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
        if(os.path.exists(dirSample + "/voronoiArea.dat")):
            data = np.loadtxt(dirSample + "/voronoiArea.dat")
            # fluid radius
            clusterRad[d,0] = np.mean(np.sqrt(data[:,1]/np.pi))
            clusterRad[d,1] = np.std(np.sqrt(data[:,1]/np.pi))
    ax.tick_params(axis='both', labelsize=14)
    # plot delta versus fluid radius
    ax.errorbar(clusterRad[deltap[:,0]!=0,0], deltap[deltap[:,0]!=0,0]*clusterRad[deltap[:,0]!=0,0], deltap[deltap[:,0]!=0,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3)
    ax.set_xlabel('$Fluid$ $radius,$ $R_c$', fontsize=18)
    ax.set_ylabel('$Surface$ $tension,$ $\\gamma \\sigma$', fontsize=18)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pDeltaP-vsSystemSize-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPMIPSBoundsVSSystemSize(dirName, figureName, which='up'):
    fig, ax = plt.subplots(figsize = (7,3.5), dpi = 120)
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    dirList = np.array(['1024', '2048', '4096', '8192', '16384'])#, '32768'])
    numParticles = np.array([1024, 2048, 4096, 8192, 16384])#, 32768])
    bounds = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/densitySweep/"
        if(os.path.exists(dirSample + "/MIPSBounds.dat")):
            data = np.loadtxt(dirSample + "/MIPSBounds.dat")
            bounds[d,0] = data[1]
            bounds[d,1] = data[2]
    if(which=='down'):
        label = '$Lower$ $bound$'
        ax.semilogx(numParticles, bounds[:,0], color='k', marker='v', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, ls='dashdot', label='$Lower$ $bound$')
        figureName += '-lower'
    elif(which=='up'):
        label = '$Upper$ $bound$'
        ax.semilogx(numParticles, bounds[:,1], color='k', marker='^', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, label='$Upper$ $bound$')
        figureName += '-upper'
    else:
        label = '$MIPS$ $bounds$'
        ax.semilogx(numParticles, bounds[:,1], color='k', marker='^', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, label='$Upper$ $bound$')
        ax.semilogx(numParticles, bounds[:,0], color='k', marker='v', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, ls='dashdot', label='$Lower$ $bound$')
        figureName += '-both'
        ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xscale('log', basex=2)
    ax.set_xlabel('$System$ $size,$ $N$', fontsize=18)
    ax.set_ylabel(label, fontsize=18)
    ax.set_xticks(numParticles)
    ax.set_xticklabels(dirList)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pBounds-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDropletTensionVSSystemSize(dirName, figureName):
    fig, ax = plt.subplots(figsize = (7,5), dpi = 120)
    labelList = np.array(['$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    dirList = np.array(['4096', '8192', '16384'])#, '32768'])
    numParticles = np.array([4096, 8192, 16384])#, 32768])
    tension = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        dirSample = dirName  + os.sep + dirList[d] + "-2d/densitySweep/0.31/active-langevin/Dr2e-04/langevin-u/T0.004-u0.1/dynamics/"
        if(os.path.exists(dirSample + "/dropletPressure.dat") and os.path.exists(dirSample + "/voronoiArea.dat")):
            pressure = np.loadtxt(dirSample + "/dropletPressure.dat")
            area = np.loadtxt(dirSample + "/voronoiArea.dat")
            tension[d,0] = np.mean((pressure[:,1] + pressure[:,2] - pressure[:,3] - pressure[:,4]) * np.sqrt(area[:,0]/np.pi))
            tension[d,1] = np.std((pressure[:,1] + pressure[:,2] - pressure[:,3] - pressure[:,4]) * np.sqrt(area[:,0]/np.pi))
    ax.errorbar(numParticles, tension[:,0], tension[:,1], color='k', marker='o', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(-0.1, 82)
    ax.set_xscale('log', basex=2)
    ax.set_xlabel('$System$ $size,$ $N$', fontsize=18)
    ax.set_ylabel('$Surface$ $tension,$ $\\gamma = \\Delta p R_c$', fontsize=18)
    ax.set_xticks(numParticles)
    ax.set_xticklabels(dirList)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pDropletTension-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterTensionVSSystemSize(dirName, figureName):
    fig, ax = plt.subplots(figsize = (7,5), dpi = 120)
    labelList = np.array(['$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    dirList = np.array(['4096', '8192', '16384'])#, '32768'])
    numParticles = np.array([4096, 8192, 16384])#, 32768])
    tension = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        dirSample = dirName  + os.sep + dirList[d] + "-2d/densitySweep/0.31/active-langevin/Dr2e-04/dynamics/"
        if(os.path.exists(dirSample + "/clusterPressure.dat") and os.path.exists(dirSample + "/voronoiArea.dat")):
            pressure = np.loadtxt(dirSample + "/clusterPressure.dat")
            area = np.loadtxt(dirSample + "/voronoiArea.dat")
            tension[d,0] = np.mean((pressure[:,2] + pressure[:,3] + pressure[:,4] - pressure[:,5] - pressure[:,6] - pressure[:,7]) * np.sqrt(area[:,0]/np.pi))
            tension[d,1] = np.std((pressure[:,2] + pressure[:,3] + pressure[:,4] - pressure[:,5] - pressure[:,6] - pressure[:,7]) * np.sqrt(area[:,0]/np.pi))
    ax.errorbar(numParticles, tension[:,0], tension[:,1], color='k', marker='o', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(-282, 2)
    ax.set_xscale('log', basex=2)
    ax.set_xlabel('$System$ $size,$ $N$', fontsize=18)
    ax.set_ylabel('$Surface$ $tension,$ $\\gamma = \\Delta p R_c$', fontsize=18)
    ax.set_xticks(numParticles)
    ax.set_xticklabels(dirList)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterTension-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPressureProfile(dirName, figureName, shift=0, which='pressure'):
    fig, ax = plt.subplots(figsize = (8,4), dpi = 120)
    if(os.path.exists(dirName + "/pressureProfile.dat")):
        data = np.loadtxt(dirName + "/pressureProfile.dat")
        sigma = float(ucorr.readFromDynParams(dirName, 'sigma'))
        data[:,1] = np.roll(data[:,1], shift)
        data[:,2] = np.roll(data[:,2], shift)
        data[:,3] = np.roll(data[:,3], shift)
        data[:,4] = np.roll(data[:,4], shift)
        data[:,5] = np.roll(data[:,5], shift)
        data[:,6] = np.roll(data[:,6], shift)
        data[:,7] = np.roll(data[:,7], shift)
        data[:,8] = np.roll(data[:,8], shift)
        half = int(data.shape[0]/2)
        print("surface tension: ", np.sum(data[:half,0] * (data[:half,2] + data[:half,6] - data[:half,3] - data[:half,7]))/sigma)
        if(which=='pressure'):
            ax.plot(data[:,0], data[:,1], lw=1.5, color='k', ls='--', label='$Steric$')
            ax.plot(data[:,0], data[:,4], lw=1.5, color='r', ls='dotted', label='$Thermal$')
            ax.plot(data[:,0], data[:,5], lw=1.5, color=[1,0.5,0], label='$Active$')
            ax.plot(data[:,0], data[:,8], lw=1.5, color='b', ls='dashdot', label='$Total$')
        elif(which=='delta'):
            ax.plot(data[:,0], data[:,2] - data[:,3], lw=1.5, color='k', ls='solid', label='$Steric$')
            ax.plot(data[:,0], data[:,6] - data[:,7], lw=1.5, color=[1,0.5,0], ls='solid', label='$Active$')
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Radial$ $distance,$ $r$", fontsize=18)
    ax.set_xlabel("$Position,$ $x$", fontsize=18)
    if(which=="pressure"):
        ax.set_ylabel("$Pressure,$ $p \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,4])-0.2, np.max(data[:,8])+0.6)
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pProfile-" + figureName + ".png"
    elif(which=="delta"):
        ax.set_ylabel("$\\Delta p^\ast = p_{xx} - p_{yy} \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,6] - data[:,7])-0.2, np.max(data[:,6] - data[:,7])+0.6)
        figureName = "/home/francesco/Pictures/nve-nvt-nva/deltaProfile-" + figureName + ".png"
    ax.legend(fontsize=13, loc='upper right', ncol=4)
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPDropletProfile(dirName, figureName, shift=0, which='pressure'):
    fig, ax = plt.subplots(figsize = (8,4), dpi = 120)
    if(os.path.exists(dirName + "/pressureProfile.dat")):
        data = np.loadtxt(dirName + "/pressureProfile.dat")
        data[:,1] = np.roll(data[:,1], shift)
        data[:,2] = np.roll(data[:,2], shift)
        data[:,3] = np.roll(data[:,3], shift)
        data[:,4] = np.roll(data[:,4], shift)
        data[:,5] = np.roll(data[:,5], shift)
        data[:,6] = np.roll(data[:,6], shift)
        data[:,7] = np.roll(data[:,7], shift)
        half = int(data.shape[0]/2)
        print("surface tension: ", np.sum(data[:half,0] * (data[:half,2] + data[:half,5] - data[:half,3] - data[:half,6])))
        if(which=='pressure'):
            ax.plot(data[:,0], data[:,1], lw=1.5, color='k', ls='solid', label='$Steric$')
            ax.plot(data[:,0], data[:,4], lw=1.5, color='r', ls='dashed', label='$Thermal$')
        elif(which=='delta'):
            ax.plot(data[:,0], data[:,2] - data[:,3], lw=1.5, color='k', ls='solid', label='$Steric$')
            ax.plot(data[:,0], data[:,5] - data[:,6], lw=1.5, color='r', ls='dashed', label='$Thermal$')
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Radial$ $distance,$ $r$", fontsize=18)
    ax.set_xlabel("$Position,$ $x$", fontsize=18)
    if(which=="pressure"):
        ax.set_ylabel("$Pressure,$ $p \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,4])-0.2, np.max(data[:,1])+0.4)
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pProfile-" + figureName + ".png"
    elif(which=="delta"):
        ax.set_ylabel("$\\Delta p^\ast = p_{xx} - p_{yy} \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,2] - data[:,3])-0.2, np.max(data[:,2] - data[:,3])+0.6)
        figureName = "/home/francesco/Pictures/nve-nvt-nva/deltaProfile-" + figureName + ".png"
    ax.legend(fontsize=13, loc='upper right', ncol=4)
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPClusterMixing(dirName, figureName, fixed='Dr', which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(fixed=="phi"):
        dirList = np.array(['5e-02', '3e-02', '2e-02', '1.5e-02', '1.2e-02', '1e-02', '8e-03', '7e-03', '6e-03', '5e-03', '3e-03', '2e-03', '1e-03', '5e-04', '3e-04', '1e-04', '5e-05', '1e-05'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal83', 'thermal85', 'thermal88', 'thermal94'])
        labelList = np.array(['$\\varphi = 0.30$', '$\\varphi = 0.35$', '$\\varphi = 0.40$', '$\\varphi = 0.45$', '$\\varphi = 0.52$', '$\\varphi = 0.58$', '$\\varphi = 0.62$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.83$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    phi = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/short/"
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "-f0200/dynamics/short/"
        if(os.path.exists(dirSample + "clusterMixing-block.dat")):
            data = np.loadtxt(dirSample + "clusterMixing-block.dat")
            timeStep = ucorr.readFromParams(dirSample, "dt")
            sigma = np.mean(np.loadtxt(dirSample + "particleRad.dat"))
            phi[d] = ucorr.readFromParams(dirSample, "phi")
            ax.plot(data[:,0]*timeStep/sigma, data[:,1], lw=1.2, color=colorList(d/dirList.shape[0]), marker='o', markersize=4, fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $time,$ $t/\\sigma$", fontsize=18)
    ax.set_ylabel("$N_{fluid}^0(t) / N_{fluid}^0$", fontsize=18)
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = plt.colorbar(colorBar, cax=cax)
    cb.ax.tick_params(labelsize=14)
    cb.set_ticks((0.5/dirList.shape[0],1-0.5/dirList.shape[0]))
    if(fixed=='phi'):
        cb.set_ticklabels(("$5 \\times 10^{-2}$", "$5 \\times 10^{-5}$"))
        cb.set_label(label="$D_r$", fontsize=16, labelpad=-30, rotation='horizontal')
        fig.savefig("/home/francesco/Pictures/nve-nvt-nva/pMixing-vsDr-" + figureName + "-Dr" + which + ".png", transparent=True, format = "png")
    elif(fixed=='Dr'):
        cb.set_ticklabels(("$0.30$", "$0.93$"))
        cb.set_label(label="$\\varphi$", fontsize=16, labelpad=-10, rotation='horizontal')
        fig.savefig("/home/francesco/Pictures/nve-nvt-nva/pMixing-vsPhi-" + figureName + "-Dr" + which + ".png", transparent=True, format = "png")
    #ax.legend(fontsize=10, loc='best')
    fig.tight_layout()
    plt.show()

def plotSPClusterLengthscale(dirName, figureName, fixed=False, which='10'):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-06', '1e-05', '5e-05', '1e-04', '3e-04', '5e-04', '1e-03', '3e-03', '5e-03', '6e-03', '7e-03', '8e-03', '1e-02', '1.2e-02', '1.5e-02', '2e-02', '3e-02', '5e-02', '1e-01', '2e-01', '5e-01', '1', '5'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    clusterLength = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    lp = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = ucorr.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "clusterRad.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            lp[d] = ucorr.readFromDynParams(dirSample, "f0") * taup[d] / ucorr.readFromDynParams(dirSample, "damping")
            data = 2*np.loadtxt(dirSample + "clusterRad.dat")
            clusterLength[d,0] = np.mean(data)
            clusterLength[d,1] = np.std(data)
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterLength-vsPhi-" + figureName + "-iod" + which
        ax.plot(np.linspace(phi[0], phi[-1], 50), np.ones(50), color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterLength-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterLength-vsDamping-" + figureName + "-Dr" + which
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Cluster$ $lengthscale,$ $L_{cluster} / L_{box}$", fontsize=18)
    ax.errorbar(x, clusterLength[:,0], clusterLength[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
    if(fixed!="iod"):
        ax.set_xscale('log')
        #ax.set_yscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterPairCorr(dirName, figureName, fixed=False, which='10'):
    fig1, ax1 = plt.subplots(figsize=(6,5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-06', '1e-05', '5e-05', '1e-04', '3e-04', '5e-04', '1e-03', '3e-03', '5e-03', '6e-03', '7e-03', '8e-03', '1e-02', '1.2e-02', '1.5e-02', '2e-02', '3e-02', '5e-02'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    interDistance = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    lp = np.zeros(dirList.shape[0])
    meanRad = np.mean(np.loadtxt(dirName + "/particleRad.dat"))
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = ucorr.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "pairCorrCluster.dat")):
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            lp[d] = ucorr.readFromDynParams(dirSample, "f0") * taup[d] / ucorr.readFromDynParams(dirSample, "damping")
            data = np.loadtxt(dirSample + "pairCorrCluster.dat")[20:]
            interDistance[d,0] = data[np.argmax(data[:,1]),0]
            interDistance[d,1] = data[np.argmax(data[:,2]),0]
            ax1.plot(data[data[:,0]<0.5,0], data[data[:,0]<0.5,1], color=colorList(d/dirList.shape[0]), lw=1)
            ax2.plot(data[data[:,0]<0.5,0], data[data[:,0]<0.5,2], color=colorList(d/dirList.shape[0]), lw=1)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel("$r$", fontsize=18)
    ax1.set_ylabel("$g_{Fluid}(r)$", labelpad=3, fontsize=18)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel("$r$", fontsize=18)
    ax2.set_ylabel("$g_{Gas}(r)$", labelpad=3, fontsize=18)
    #ax.set_ylim(-0.004, 0.036)
    fig1.tight_layout()
    fig2.tight_layout()
    fig, ax = plt.subplots(figsize = (6.5,4.5), dpi = 120)
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterLength-vsPhi-" + figureName + "-iod" + which
        ax.plot(np.linspace(phi[0], phi[-1], 50), np.ones(50), color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterLength-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterLength-vsDamping-" + figureName + "-Dr" + which
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Inter-particle$ $distance,$ $d / \\sigma$", fontsize=18)
    ax.semilogx(x[interDistance[:,0]>0], interDistance[interDistance[:,0]>0,0]/meanRad, color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none')
    ax.semilogx(x[interDistance[:,1]>0], interDistance[interDistance[:,1]>0,1]/meanRad, color='b', lw=1.2, marker='o', markersize = 7, fillstyle='none')
    ax.legend(("$Fluid$", "$Gas$"), fontsize=14, loc='best')
    if(fixed!="iod"):
        ax.set_xscale('log')
        #ax.set_yscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterFluctuations(dirName, figureName, fixed=False, which='10'):
    fig1, ax1 = plt.subplots(figsize=(6,5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78',  'thermal80',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])
        labelList = np.array(['$\\varphi = 0.30$', '$\\varphi = 0.35$', '$\\varphi = 0.40$', '$\\varphi = 0.45$', '$\\varphi = 0.52$', '$\\varphi = 0.58$', '$\\varphi = 0.62$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.80$', '$\\varphi = 0.83$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-05', '5e-05', '1e-04', '5e-04', '1e-03', '5e-03', '7e-03', '1e-02', '1.2e-02', '1.5e-02', '2e-02', '3e-02', '5e-02', '1e-01', '2e-01', '5e-01', '1', '5', '10', '50', '100', '1000'])
        labelList = np.array(['$D_r = 10^{-5}$', '$D_r = 5 \\times 10^{-5}$', '$D_r = 10^{-4}$', '$D_r = 5 \\times 10^{-4}$', '$D_r = 0.001$', '$D_r = 0.005$', '$D_r = 0.007$', '$D_r = 0.01$', '$D_r = 0.012$', '$D_r = 0.015$', '$D_r = 0.02$', '$D_r = 0.03$', '$D_r = 0.05$', '$D_r = 0.1$', '$D_r = 0.2$', '$D_r = 0.5$', '$D_r = 1$', '$D_r = 5$', '$D_r = 10$', '$D_r = 50$', '$D_r = 100$', '$D_r = 1000$'])
        #dirList = np.array(['1e-05', '1e-04', '1e-03', '1e-02', '1e-01', '1'])
        #labelList = np.array(['$D_r = 0.00001$', '$D_r = 0.0001$', '$D_r = 0.001$', '$D_r = 0.01$', '$D_r = 0.1$', '$D_r = 1$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    clusterNum = np.zeros((dirList.shape[0],2))
    clusterSize = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = ucorr.readFromDynParams(dirSample, "Dr")
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = ucorr.readFromDynParams(dirSample, "damping")
        #if(os.path.exists(dirSample + "dbAllClustersSize.dat")):
        #    data = np.loadtxt(dirSample + "dbAllClustersSize.dat")
        #    clusterSize[d,0] = np.mean(data)
        #    clusterSize[d,1] = np.std(data)
        if(os.path.exists(dirSample + "clusterFluctuations.dat")):
            data = np.loadtxt(dirSample + "clusterFluctuations.dat")
            clusterNum[d,0] = np.mean(data[:,1])
            clusterNum[d,1] = np.std(data[:,1])
            clusterSize[d,0] = np.mean(data[:,2])
            clusterSize[d,1] = np.std(data[:,2])
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pClusterNum-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pClusterSize-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = 1/Dr
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pClusterNum-vsDr-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pClusterSize-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/nve-nvt-nva/pClusterNum-vsDamping-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/nve-nvt-nva/pClusterSize-vsDamping-" + figureName + "-Dr" + which
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel("$Cluster$ $number,$ $\\langle N_c \\rangle$", fontsize=18)
    ax2.set_ylabel("$Cluster$ $area,$ $\\langle A_c \\rangle$", fontsize=18)
    ax1.errorbar(x, clusterNum[:,0], clusterNum[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
    ax2.errorbar(x, clusterSize[:,0], clusterSize[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
    if(fixed!="iod"):
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    ax2.set_xlim(5.8e-06, 2.8e03)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig2.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def plotSPNumberDensityFluctuations(dirName, figureName, fixed='Dr', which='1e-03'):
    #fig, ax = plt.subplots(2, 1, figsize = (7,8), sharex=True, dpi = 120)
    fig, ax = plt.subplots(figsize=(8,4.5), dpi = 120)
    if(fixed=="Dr"):
        dirList = np.array(['thermal94', 'thermal88', 'thermal85', 'thermal78', 'thermal67', 'thermal58', 'thermal45', 'thermal40', 'thermal35', 'thermal30', 'thermal27', 'thermal25'])
        labelList = np.array(['$\\varphi = 0.94$', '$\\varphi = 0.88$', '$\\varphi = 0.85$', '$\\varphi = 0.78$', '$\\varphi = 0.67$', '$\\varphi = 0.58$', '$\\varphi = 0.45$', '$\\varphi = 0.40$', '$\\varphi = 0.35$', '$\\varphi = 0.30$', '$\\varphi = 0.27$', '$\\varphi = 0.25$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['10', '1', '1e-01', '1.5e-02', '1e-02', '1e-03', '1e-04', '1e-05'])
        labelList = np.array(['$10$', '$1$', '$0.1$', '$0.015$', '$0.01$', '$0.001$', '$10^{-4}$', '$10^{-5}$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        taup = np.zeros(dirList.shape[0])
    else:
        print("please specify fixed parameter")
    for d in range(dirList.shape[0]):
        if(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "-f0200/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
        if(os.path.exists(dirSample + "averageLocalNumberDensity.dat")):
            if(d<8):
                data = np.loadtxt(dirSample + "averageLocalNumberDensity.dat")
                ax.errorbar(data[:,1], data[:,7], data[:,8], lw=1.2, color=colorList(d/dirList.shape[0]), marker='o', fillstyle='none', capsize=3, label=labelList[d])
                #ax.plot(data[:,1], data[:,4], lw=1.2, color=colorList((dirList.shape[0]-d-1)/dirList.shape[0]), marker='o', fillstyle='none', label=labelList[d])
    ax.set_ylim(0.57, 2886)
    ax.set_ylim(4.6e-06, 0.34)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Density$ $variance,$ $\\Delta \\varphi^2$", fontsize=18)
    ax.set_xlabel("$Number$ $of$ $particles,$ $\\langle N_{sub} \\rangle$", fontsize=18)
    if(fixed=="Dr"):
        ax.legend(fontsize=10, loc='lower left', ncol=3)
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pNumberPhiVar-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        colorBar = cm.ScalarMappable(cmap=colorList)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = plt.colorbar(colorBar, cax=cax)
        cb.ax.tick_params(labelsize=14)
        cb.set_ticks([0.5/labelList.shape[0], 1-2.5/labelList.shape[0], 1-0.5/labelList.shape[0]])
        #ticklabels = [np.format_float_scientific(taup[0], precision=2), np.format_float_scientific(taup[-1], precision=2)]
        ticklabels = ['$10^{-3}$', '$10^{1}$', '$10^3$']
        cb.set_ticklabels(ticklabels)
        cb.set_label(label="$\\tau_p$", fontsize=16, labelpad=-5, rotation='horizontal')
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pNumberPhiVar-vsDr-" + figureName + "-iod" + which
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterDistribution(dirName, figureName, fixed=False, which='10', numBins=40):
    fig, ax = plt.subplots(figsize=(6.5,4), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72',  'thermal78',  'thermal80',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$0.45$', '$0.58$', '$0.67$', '$0.72$', '$0.78$', '$0.80$', '$0.83$', '$0.85$', '$0.88$', '$0.94$', '$1.00$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1.5e-02', '5e-02', '1e-01', '2e-01', '5e-01', '1'])
        labelList = np.array(['$0.015$', '$0.05$', '$0.1$', '$0.2$', '$0.5$', '$1$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+3)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = ucorr.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = ucorr.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "clusterNumbers.dat")):
            clusterNumber = np.loadtxt(dirSample + "clusterNumbers.dat")
            clusterNumber = clusterNumber[:-100]
            pdf, edges = np.histogram(clusterNumber, bins=np.geomspace(np.min(clusterNumber), np.max(clusterNumber), numBins), density=True)
            edges = (edges[1:] + edges[:-1])/2
            data = np.column_stack((edges, pdf))
            #data = np.loadtxt(dirSample + "clusterNumberPDF.dat")[:-1,:]
            data = data[data[:,1]>0]
            if(d == 1):
                data = data[:-2]
            else:
                data = data[:-1]
            ax.loglog(data[1:,0], data[1:,1]/data[1,1], lw=1.2, color=colorList((dirList.shape[0]-d-3)/dirList.shape[0]), marker='o', fillstyle='none')
    # make color bar for legend
    if(fixed=="iod"):
        label = "$\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pNumberPhiVar-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        label = "$D_r$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pNumberPhiVar-vsDr-" + figureName + "-iod" + which
    else:
        label = "$m/\\gamma$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pNumberPhiVar-vsDamping-" + figureName + "-Dr" + which
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Number$ $of$ $particles$ $in$ $cluster,$ $N_c$", fontsize=18)
    ax.set_ylabel("$Distribution,$ $P(N_c)$", fontsize=18)
    fig.tight_layout()
    #plt.subplots_adjust(hspace=0)
    #colorBar = cm.ScalarMappable(cmap=colorList)
    #cb = fig.colorbar(colorBar, ax=ax)#, shrink=0.7)
    #cb.set_ticks(np.arange(0.5/labelList.shape[0],1,1/labelList.shape[0]))
    #cb.ax.tick_params(labelsize=12)
    #ticklabels = labelList
    #cb.set_ticklabels(ticklabels)
    #cb.set_label(label=label, fontsize=16, labelpad=10, rotation='horizontal')
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSizeVSTime(dirName, figureName, fixed=False, which='10'):
    fig, ax = plt.subplots(figsize=(7.5,5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72',  'thermal78',  'thermal80',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$0.45$', '$0.58$', '$0.67$', '$0.72$', '$0.78$', '$0.80$', '$0.83$', '$0.85$', '$0.88$', '$0.94$', '$1.00$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-05', '5e-05', '1e-04', '5e-04', '1e-03', '5e-03', '7e-03', '1e-02', '1.5e-02', '3e-02', '1e-01', '2e-01', '5e-01', '1', '10'])
        labelList = np.array(['$10^{-5}$', '$5 \\times 10^{-5}$', '$10^{-4}$', '$5 \\times 10^{-4}$', '$0.001$', '$0.005$', '$0.007$', '$0.01$', '$0.015$', '$0.03$', '$0.1$', '$0.2$', '$0.5$', '$1$', '$10$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    for d in range(4,dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = ucorr.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = ucorr.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = ucorr.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "clusterFluctuations.dat")):
            data = np.loadtxt(dirSample + "clusterFluctuations.dat")
            data = data[np.argwhere(data[:,0]%100000==0)[:,0]]
            ax.plot(data[:,0], data[:,2], lw=1.2, color=colorList(d/dirList.shape[0]))
    ax.set_ylim(-0.018, 0.406)
    # make color bar for legend
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    if(fixed=="iod"):
        label = "$\\varphi$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterTime-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        label = "$D_r$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterTime-vsDr-" + figureName + "-iod" + which
    else:
        label = "$m/\\gamma$"
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pClusterTime-vsDamping-" + figureName + "-Dr" + which
    cb.set_ticks(np.arange(0.5/labelList.shape[0],1,1/labelList.shape[0])+4/labelList.shape[0])
    cb.ax.tick_params(labelsize=12)
    ticklabels = labelList[4:]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=16, labelpad=10, rotation='horizontal')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $time,$ $t$", fontsize=18)
    ax.set_ylabel("$Cluster$ $area,$ $A_c(t)$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSystemSize(dirName, figureName, which = "tension"):
    fig, ax = plt.subplots(figsize=(7.5,4.5), dpi = 120)
    dirList = np.array(['1024', '2048', '4096', '8192', '16384', '32768'])
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    numParticles = np.array([1024, 2048, 4096, 8192, 16384, 32678])
    clusterDensity = np.zeros((dirList.shape[0],2))
    border = np.zeros((dirList.shape[0],2))
    energy = np.zeros((dirList.shape[0],2))
    tension = np.zeros((dirList.shape[0],2))
    sigma = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/thermal45/langevin/T0.001/iod10/active-langevin/Dr1e-03-f0200/dynamics/"
        if(os.path.exists(dirSample + "borderEnergy.dat")):
            sigma[d] = np.mean(np.loadtxt(dirSample + 'particleRad.dat'))
            data = np.loadtxt(dirSample + "borderEnergy.dat")
            #clusterDensity[d,0] = np.mean(data)
            #clusterDensity[d,1] = np.std(data)
            border[d,0] = np.mean(data[:,1])
            border[d,1] = np.std(data[:,1])#/np.sqrt(data[:,0].shape[0])
            energy[d,0] = np.mean(data[:,2])
            energy[d,1] = np.std(data[:,2])#/np.sqrt(data[:,0].shape[0])
            tension[d,0] = np.mean(data[:,2]/data[:,1])
            tension[d,1] = np.std(data[:,2]/data[:,1])#/np.sqrt(data[:,0].shape[0])
    if(which=='tension'):
        ax.errorbar(numParticles, tension[:,0]*sigma, tension[:,1]*sigma, color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
        ax.set_ylabel("$Surface$ $tension,$ $\\gamma_c \\sigma$", fontsize=18)
    elif(which=='border'):
        ax.errorbar(numParticles, border[:,0], border[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
        ax.set_ylabel("$Border$ $length,$ $L_c$", fontsize=18)
    elif(which=='energy'):
        ax.errorbar(numParticles, energy[:,0], energy[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
        ax.set_ylabel("$Border$ $energy,$ $E_c$", fontsize=18)
    #x = np.linspace(numParticles[0], numParticles[-1], 100)
    #ax.plot(x, 0.05*np.log(x), linestyle='--', color='b')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$System$ $size,$ $N$", fontsize=18)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_ylim(0.00082,)
    #ax.set_ylabel("$Cluster$ $density,$ $\\langle \\varphi_c \\rangle$", fontsize=18)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pCluster-vsSystemSize-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDensityVSWindowSize(dirName, figureName, weight=False):
    fig, ax = plt.subplots(figsize=(7.5,5), dpi = 120)
    fileList = np.arange(8,34,2)
    colorList = cm.get_cmap('coolwarm', fileList.shape[0])
    for d in range(fileList.shape[0]):
        if(weight == 'weight'):
            fileName = dirName + os.sep + "localVoroDensity-N" + str(fileList[d]) + "-weight.dat"
        else:
            fileName = dirName + os.sep + "localVoroDensity-N" + str(fileList[d]) + ".dat"
        if(os.path.exists(fileName)):
            data = np.loadtxt(fileName)
            plt.plot(data[data[:,1]>0,0], data[data[:,1]>0,1], lw=1.2, color=colorList(d/fileList.shape[0]))
            if(fileList[d]==16):
                plt.semilogy(data[data[:,1]>0,0], data[data[:,1]>0,1], color='k', marker='*', markersize=10, lw=0, fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=18)
    ax.set_ylabel("$PDF(\\varphi_l)$", fontsize=18)
    ax.set_xlim(-0.05, 1.05)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    label = "$N_{bins}$"
    cb.set_ticks([1/26, 7/26, 13/26, 19/26, 25/26])
    cb.ax.tick_params(labelsize=14)
    ticklabels = ['$8$', '$14$', '$20$', '$26$', '$32$']
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=18, labelpad=25, rotation='horizontal')
    fig.tight_layout()
    if(weight == 'weight'):
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pWeightedWindow-" + figureName
    else:
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pWindow-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPLocalDensity(dirName, figureName, fixed='phi', which='30'):
    fig, ax = plt.subplots(figsize=(8,4), dpi = 120)
    if(fixed=="phi"):
        dirList = np.array(['10', '1', '1e-01', '1e-02', '1e-03', '1e-04', '1e-05'])
        labelList = np.array(['$10$', '$1$', '$0.1$', '$0.01$', '$0.001$', '$10^{-4}$', '$10^{-5}$'])
        taup = np.zeros(dirList.shape[0])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['thermal25', 'thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85', 'thermal85', 'thermal88', 'thermal94', 'thermal1'])
        labelList = np.array(['$\\varphi = 0.25$', '$\\varphi = 0.30$', '$\\varphi = 0.35$', '$\\varphi = 0.40$', '$\\varphi = 0.45$', '$\\varphi = 0.52$', '$\\varphi = 0.58$', '$\\varphi = 0.62$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.83$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr1e-03-f0200/dynamics/"
        if(os.path.exists(dirSample + "/localVoroDensity-N" + which + ".dat")):
            data = np.loadtxt(dirSample + os.sep + "localVoroDensity-N" + which + ".dat")
            plt.semilogy(data[:,0], data[:,1], lw=1.2, color=colorList(d/dirList.shape[0]), label=labelList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=18)
    ax.set_ylabel("$PDF(\\varphi_l)$", fontsize=18)
    ax.set_xlim(-0.05, 1.05)
    if(fixed=="phi"):
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pLocalPhi-vsDr-" + figureName
        colorBar = cm.ScalarMappable(cmap=colorList)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = plt.colorbar(colorBar, cax=cax)
        cb.ax.tick_params(labelsize=14)
        cb.set_ticks([0.5/labelList.shape[0],1-0.5/labelList.shape[0]])
        ticklabels = [np.format_float_scientific(taup[0], precision=2), np.format_float_scientific(taup[-1], precision=2)]
        ticklabels = ['$10^{-3}$', '$10^3$']
        cb.set_ticklabels(ticklabels)
        cb.set_label(label="$\\tau_p$", fontsize=16, labelpad=-20, rotation='horizontal')
    else:
        ax.legend(loc='upper right', fontsize=12)
        figureName = "/home/francesco/Pictures/nve-nvt-nva/pLocalPhi-vsPhi-" + figureName
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPCompareLocalDensity(dirName, figureName, which='30'):
    fig, ax = plt.subplots(3,1, sharex=True, figsize=(8,10), dpi = 120)
    phiList = np.array(['thermal94', 'thermal45', 'thermal25'])
    dirList = np.array(['10', '1e-01', '1e-03', '1e-05'])
    labelList = np.array(['$10$', '$0.1$', '$0.001$', '$10^{-5}$'])
    taup = np.zeros(dirList.shape[0])
    colorList = cm.get_cmap('plasma', dirList.shape[0])
    colorBar = cm.ScalarMappable(cmap=colorList)
    for i in range(phiList.shape[0]):
        ax[i].tick_params(axis='both', labelsize=14)
        ax[i].set_ylabel("$PDF(\\varphi_l)$", fontsize=18)
        ax[i].set_ylim(0.0055, 64)
        divider = make_axes_locatable(ax[i])
        for d in range(dirList.shape[0]):
            dirSample = dirName + os.sep + phiList[i] + "/langevin/T0.001/iod10/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            taup[d] = 1/(ucorr.readFromDynParams(dirSample, 'Dr')*ucorr.readFromDynParams(dirSample, 'sigma'))
            if(os.path.exists(dirSample + "/localVoroDensity-N" + which + ".dat")):
                if(d<1):
                    data = np.loadtxt(dirSample + os.sep + "localVoroDensity-N" + which + ".dat")
                    ax[i].semilogy(data[:,0], data[:,1], lw=1.2, color=colorList(d/dirList.shape[0]), label=labelList[d])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = plt.colorbar(colorBar, cax=cax)
        cb.ax.tick_params(labelsize=14)
        cb.set_ticks([0.5/labelList.shape[0],1-2.5/labelList.shape[0], 1-1.5/labelList.shape[0], 1-0.5/labelList.shape[0]])
        #ticklabels = [np.format_float_scientific(taup[0], precision=2), np.format_float_scientific(taup[-1], precision=2)]
        ticklabels = ['$10^{-3}$', '$10^{-1}$', '$10^{1}$', '$10^3$']
        cb.set_ticklabels(ticklabels)
        cb.set_label(label="$\\tau_p$", fontsize=16, labelpad=20, rotation='horizontal')
    ax[2].set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=18)
    ax[2].set_xlim(-0.05, 1.05)
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pCompareLocalPhi-vsDr-" + figureName
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSSystemSize(dirName, figureName):
    fig, ax = plt.subplots(figsize=(7.5,4.5), dpi = 120)
    dirList = np.array(['1024', '2048', '4096', '8192'])#, '16384', '32768'])
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    numParticles = np.array([1024, 2048, 4096, 8192, 16384, 32678])
    colorList = cm.get_cmap('inferno', dirList.shape[0]+1)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/thermal45/langevin/T0.001/iod10/dynamics/"
        if(os.path.exists(dirSample + "logVelCorr.dat")):
            sigma = np.mean(np.loadtxt(dirSample + 'particleRad.dat'))
            data = np.loadtxt(dirSample + "logVelCorr.dat")
            ax.errorbar(data[:,0]/sigma, data[:,1], data[:,2], color=colorList(d/dirList.shape[0]), lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4, label=labelList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Elapsed$ $time,$ $\\Delta t / \\sigma$", fontsize=18)
    ax.set_ylabel("$C_{vv}(\\Delta t)$", fontsize=18)
    ax.set_xscale('log')
    ax.legend(fontsize=12, loc='upper right')
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/nve-nvt-nva/pDynamics-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################### plot and check compression #########################
def plotSPCompression(dirName, figureName):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), sharex = True, dpi = 120)
    # check if the compression data are already saved
    if(os.path.exists(dirName + os.sep + "compression.dat")):
        data = np.loadtxt(dirName + os.sep + "compression.dat")
        phi = data[:,0]
        pressure = data[:,1]
        numContacts = data[:,2]
    # if not saved, computed them and save them
    else:
        phi = []
        pressure = []
        numContacts = []
        for dir in os.listdir(dirName):
            dirSample = dirName + os.sep + dir
            if(os.path.isdir(dirSample)):
                phi.append(ucorr.readFromParams(dirSample, "phi"))
                p = ucorr.readFromParams(dirSample, "pressure")
                if(p == None):
                    p = ucorr.computePressure(dirSample)
                pressure.append(p)
                z = ucorr.readFromParams(dirSample, "numContacts")
                if(z == None):
                    z = ucorr.computeNumberOfContacts(dirSample)
                numContacts.append(z)
        pressure = np.array(pressure)
        numContacts = np.array(numContacts)
        phi = np.array(phi)
        pressure = pressure[np.argsort(phi)]
        numContacts = numContacts[np.argsort(phi)]
        phi = np.sort(phi)
        np.savetxt(dirName + os.sep + "compression.dat", np.column_stack((phi, pressure, numContacts)))
    # plot compression data
    ax[0].semilogy(phi, pressure, color='k', linewidth=1.5)
    ax[1].plot(phi, numContacts, color='k', linewidth=1.5)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$Packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_ylabel("$Pressure,$ $p$", fontsize=17)
    ax[1].set_ylabel("$Coordination$ $number,$ $z$", fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/comp-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPJamming(dirName, figureName):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), sharex = True, dpi = 120)
    phiJ = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            comp = np.loadtxt(dirName + os.sep + dir + os.sep + "compression.dat")
            phiJ.append(comp[np.argwhere(comp[:,1]>1e-08)[0,0],0])
    pdf, edges = np.histogram(phiJ, bins=10, density=True)
    edges = (edges[:-1] + edges[1:])*0.5
    ax[0].plot(edges, pdf, color='k', linewidth=1.5)
    #ax[1].plot(phi, zeta, color='k', linewidth=1.5)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_xlabel("$PDF(\\varphi)$", fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/jamming-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPPSI6P2Compression(dirName, figureName):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), sharex = True, dpi = 120)
    phi = []
    hop = []
    p2 = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            phi.append(ucorr.readFromParams(dirName + os.sep + dir, "phi"))
            boxSize = np.loadtxt(dirName + os.sep + dir + "/boxSize.dat")
            nv = np.loadtxt(dirName + os.sep + dir + "/numVertexInParticleList.dat", dtype=int)
            psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dir, boxSize)
            hop.append(np.mean(psi6))
            eigvmax, _ = shapeDescriptors.getShapeDirections(dirName + os.sep + dir, boxSize, nv)
            angles = np.arctan2(eigvmax[:,1], eigvmax[:,0])
            p2.append(np.mean(2 * np.cos(angles - np.mean(angles))**2 - 1))
    phi = np.array(phi)
    hop = np.array(hop)
    p2 = np.array(p2)
    hop = hop[np.argsort(phi)]
    p2 = p2[np.argsort(phi)]
    phi = np.sort(phi)
    hop = hop[phi>0.65]
    p2 = p2[phi>0.65]
    phi = phi[phi>0.65]
    np.savetxt(dirName + os.sep + "compression.dat", np.column_stack((phi, pressure, zeta)))
    ax[0].plot(phi, p2, color='k', linewidth=1.5)
    ax[1].plot(phi, hop, color='k', linewidth=1.5)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax[0].set_ylabel("$nematic$ $order,$ $\\langle p2 \\rangle$", fontsize=17)
    ax[1].set_ylabel("$hexagonal$ $order,$ $\\langle \\psi_6 \\rangle$", fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/comp-param-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPHOPCompression(dirName, figureName):
    dataSetList = np.array(os.listdir(dirName))
    phi = dataSetList.astype(float)
    dataSetList = dataSetList[np.argsort(phi)]
    phi = np.sort(phi)
    hop = np.zeros(phi.shape[0])
    err = np.zeros(phi.shape[0])
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dataSetList[i])
        hop[i] = np.mean(psi6)
        err[i] = np.sqrt(np.var(psi6)/psi6.shape[0])
    ax.errorbar(phi[hop>0], hop[hop>0], err[hop>0], marker='o', color='k', markersize=5, markeredgecolor='k', markeredgewidth=0.7, linewidth=1, elinewidth=1, capsize=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax.set_ylabel("$hexatic$ $order$ $parameter,$ $\\psi_6$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/hop-comp-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotCompressionSet(dirName, figureName):
    #dataSetList = np.array(["kb1e-03", "kb1e-02", "kb1e-01", "kb2e-01", "kb4e-01", "kb5e-01-kakl", "kb6e-01", "kb8e-01"])
    dataSetList = np.array(["A1_1-sigma17", "A1_2-sigma17", "A1_3-sigma17"])
    phiJ = np.array([0.8301, 0.8526, 0.8242, 0.8205, 0.8176, 0.7785, 0.7722, 0.7707])
    colorList = ['k', [0.5,0,1], 'b', 'g', [0.8,0.9,0.2], [1,0.5,0], 'r', [1,0,0.5]]
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        pressure = []
        phi = []
        for dir in os.listdir(dirName + dataSetList[i]):
            if(os.path.isdir(dirName + dataSetList[i] + os.sep + dir)):
                phi.append(ucorr.readFromParams(dirName + dataSetList[i] + os.sep + dir, "phi"))
                pressure.append(ucorr.readFromParams(dirName + dataSetList[i] + os.sep + dir, "pressure"))
        pressure = np.array(pressure)
        phi = np.array(phi)
        pressure = pressure[np.argsort(phi)]
        phi = np.sort(phi)
        phi = phi[pressure>0]
        pressure = pressure[pressure>0]
        np.savetxt(dirName + dataSetList[i] + os.sep + "compression.dat", np.column_stack((phi, pressure)))
        ax.semilogy(phi, pressure, color=colorList[i], linewidth=1.2, label=dataSetList[i])
    ax.legend(loc = 'best', fontsize = 12)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$packing$ $fraction,$ $\\varphi$", fontsize=17)
    ax.set_ylabel("$pressure,$ $p$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/compression-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPHOPDynamics(dirName, figureName):
    step = []
    hop = []
    err = []
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for dir in os.listdir(dirName)[::10]:
        if(os.path.isdir(dirName + os.sep + dir)):
            step.append(float(dir[1:]))
            psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dir)
            hop.append(np.mean(psi6))
            err.append(np.sqrt(np.var(psi6)/psi6.shape[0]))
    step = np.array(step)
    hop = np.array(hop)
    err = np.array(err)
    hop = hop[np.argsort(step)]
    err = err[np.argsort(step)]
    step = np.sort(step)
    plotErrorBar(ax, step, hop, err, "$simulation$ $step$", "$hexatic$ $order$ $parameter,$ $\\psi_6$")
    plt.savefig("/home/francesco/Pictures/soft/hexatic-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPPSI6P2Dynamics(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04):
    stepList = uplot.getStepList(numFrames, firstStep, stepFreq)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    nv = np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int)
    numParticles = nv.shape[0]
    hop = []
    p2 = []
    for i in stepList:
        psi6 = spCorr.computeHexaticOrder(dirName + os.sep + "t" + str(i), boxSize)
        hop.append(np.mean(psi6))
        eigvmax, _ = shapeDescriptors.getShapeDirections(dirName + os.sep + "t" + str(i), boxSize, nv)
        angles = np.arctan2(eigvmax[:,1], eigvmax[:,0])
        p2.append(np.mean(2 * np.cos(angles - np.mean(angles))**2 - 1))
    stepList -= stepList[0]
    stepList = np.array(stepList-stepList[0])/np.max(stepList-stepList[0])
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6,5), dpi=150)
    ax[0].plot(stepList, hop, linewidth=1.2, color='b')
    ax[1].plot(stepList, p2, linewidth=1.2, color='g')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$time$ $fraction,$ $t/t_{relax}$", fontsize=17)
    ax[0].set_ylabel("$\\langle \\psi_6 \\rangle$", fontsize=17)
    ax[1].set_ylabel("$\\langle P_2 \\rangle$", fontsize=17, labelpad=-5)
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/psi6-p2-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPHOPVSphi(dirName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    phi = []
    hop = []
    err = []
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    for i in range(dataSetList.shape[0]):
        phi.append(ucorr.readFromParams(dirName + os.sep + dataSetList[i], "phi"))
        psi6 = spCorr.computeHexaticOrder(dirName + os.sep + dataSetList[i])
        hop.append(np.mean(psi6))
        err.append(np.sqrt(np.var(psi6)/psi6.shape[0]))
    plotErrorBar(ax, phi, hop, err, "$packing$ $fraction,$ $\\varphi$", "$hexatic$ $order$ $parameter,$ $\\psi_6$")
    plt.savefig("/home/francesco/Pictures/soft/hop-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotDeltaEvsDeltaV(dirName, figureName):
    #dataSetList = np.array(["1e-03", "3e-03", "5e-03", "7e-03", "9e-03", "1e-02", "1.3e-02", "1.5e-02", "1.7e-02", "2e-02", "2.3e-02", "2.5e-02", "2.7e-02", "3e-02", "4e-02", "5e-02", "6e-02"])
    dataSetList = np.array(["1e-03", "3e-03", "5e-03", "7e-03", "1e-02", "3e-02", "5e-02", "7e-02", "1e-01"])
    deltaE = []
    deltaV = []
    pressure = []
    fig = plt.figure(0, dpi=120)
    ax = fig.gca()
    energy0 = np.mean(np.loadtxt(dirName + os.sep + "dynamics-test/energy.dat")[:,2])
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + os.sep + "comp-delta" + dataSetList[i] + os.sep + "energy.dat")):
            energy = np.loadtxt(dirName + os.sep + "comp-delta" + dataSetList[i] + os.sep + "energy.dat")
            deltaE.append(np.mean(energy[:,2]) - energy0)
            deltaV.append(1 - (1-float(dataSetList[i]))**2)
            if(i < 5 and i > 0):
                pressure.append((deltaE[-1] - deltaE[0]) / (deltaV[-1] - deltaV[0]))
    ax.plot(deltaV, deltaE, lw=1.2, color='k', marker='.')
    print("average pressure: ", np.mean(pressure), "+-", np.std(pressure))
    x = np.linspace(0,0.1,100)
    m = np.mean(pressure)
    q = -10
    ax.plot(x, m*x + q, lw=1.2, color='g')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$\\Delta E$", fontsize=17)
    ax.set_xlabel("$\\Delta V$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pressure-" + figureName + ".png", transparent=False, format = "png")
    plt.show()


################################# plot dynamics ################################
def plotSPDynamics(dirName, figureName):
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    # plot brownian dynamics
    data = np.loadtxt(dirName + "/corr-log-q1.dat")
    timeStep = ucorr.readFromParams(dirName, "dt")
    ax.semilogx(data[1:,0]*timeStep, data[1:,2], color='b', linestyle='--', linewidth=1.2, marker="$T$", markersize = 10, markeredgewidth = 0.2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax.set_ylabel("$ISF(\\Delta t)$", fontsize=18)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pcorr-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSActivity(dirName, sampleName, figureName, q="1"):
    damping = 1e03
    meanRad = np.mean(np.loadtxt(dirName + "../particleRad.dat"))
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    colorList = ['r', 'g', 'b']
    markerList = ['v', 's', 'd']
    fig1, ax1 = plt.subplots(figsize = (7, 5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize = (7, 5), dpi = 120)
    thermalData = np.loadtxt(dirName + "../langevin/T" + sampleName + "/dynamics/corr-log-q" + q + ".dat")
    timeStep = ucorr.readFromParams(dirName + "../langevin/T" + sampleName + "/dynamics/", "dt")
    diff = np.mean(thermalData[-5:,1]/(4 * thermalData[-5:,0] * timeStep))
    tau = timeStep*ucorr.computeTau(thermalData)
    Pe = 0
    #ax2.plot(Pe, tau, color='k', marker='o')
    #ax2.semilogy(Pe, diff, color='k', marker='o')
    for j in range(f0List.shape[0])[:3]:
        diff = []
        tau = []
        Pe = []
        for i in range(DrList.shape[0])[:3]:
            dirSample = dirName + "/Dr" + DrList[i] + "-f0" + f0List[j] + "/T" + sampleName + "/dynamics"
            if(os.path.exists(dirSample + os.sep + "corr-log-q" + q + ".dat")):
                data = np.loadtxt(dirSample + os.sep + "corr-log-q" + q + ".dat")
                timeStep = ucorr.readFromParams(dirSample, "dt")
                diff.append(np.mean(data[-5:,1]/(4 * data[-5:,0] * timeStep)))
                tau.append(timeStep*ucorr.computeTau(data))
                Pe.append(((float(f0List[j])/damping) / float(DrList[i])) / meanRad)
                #ax1.semilogx(data[:,0]*timeStep, data[:,2], marker=markerList[i], color=colorList[j], markersize=6, markeredgewidth=1, fillstyle='none')
                ax1.loglog(data[:,0]*timeStep, data[:,5], marker=markerList[i], color=colorList[j], markersize=6, markeredgewidth=1, fillstyle='none')
                ax2.loglog(data[:,0]*timeStep, data[:,1]/data[:,0]*timeStep, marker=markerList[i], color=colorList[j], markersize=6, markeredgewidth=1, fillstyle='none')
                #ax2.loglog(data[:,0]*timeStep, data[:,1], marker=markerList[i], color=colorList[j], markersize=6, markeredgewidth=1, fillstyle='none')
        Pe = np.array(Pe)
        diff = np.array(diff)
        tau = np.array(tau)
        #ax2.semilogy(Pe, diff, linewidth=1.5, color=colorList[j], marker=markerList[i])
        #ax2.plot(Pe, tau, linewidth=1.5, color=colorList[j], marker=markerList[i])
    #ax1.semilogx(thermalData[:,0]*timeStep, thermalData[:,2], color='k', linewidth=2)
    ax1.loglog(thermalData[:,0]*timeStep, thermalData[:,5], color='k', linewidth=2)
    ax2.loglog(thermalData[:-10,0]*timeStep, thermalData[:-10,1]/thermalData[:-10,0]*timeStep, color='k', linewidth=2)
    #ax2.loglog(thermalData[:,0]*timeStep, thermalData[:,1], color='k', linewidth=2)
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    #ax2.set_ylim(7.8e-07, 1.7e-05)
    #ax2.set_xlim(2.6e-04, 9170)
    ax1.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax2.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    #ax1.set_ylabel("$ISF(\\Delta t)$", fontsize=18)
    ax1.set_ylabel("$\\chi_4(\\Delta t)$", fontsize=18)
    ax2.set_ylabel("$\\frac{MSD(\\Delta t)}{\\Delta t}$", fontsize=24, labelpad=-10)
    #ax2.set_ylabel("$MSD(\\Delta t)$", fontsize=18)
    #ax2.set_xlabel("$Peclet$ $number,$ $v_0/(D_r \sigma)$", fontsize=18)
    #ax2.set_ylabel("$Diffusivity,$ $D$", fontsize=17)
    #ax2.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=18)
    #ax2.set_ylabel("$Relaxation$ $interval,$ $\\Delta_\\chi$", fontsize=18)
    fig1.tight_layout()
    fig1.savefig("/home/francesco/Pictures/soft/corrFunctions/pchi-Drf0-" + figureName + ".png", transparent=True, format = "png")
    fig2.tight_layout()
    fig2.savefig("/home/francesco/Pictures/soft/corrFunctions/pdiff-Drf0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSTemp(dirName, figureName, q="1"):
    T = []
    diff = []
    tau = []
    deltaChi = []
    dataSetList = np.array(["0.035", "0.04", "0.045", "0.05", "0.06", "0.065", "0.07", "0.08", "0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.17", "0.18", "0.19", #1e08
                            "0.2", "0.23", "0.26", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1", "1.3", "1.6", "2", "3", "4", "5", "6", "7", "8", "9", "10"]) #1e07
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/T" + dataSetList[i] + "/dynamics/corr-log-q" + q + ".dat")):
            data = np.loadtxt(dirName + "/T" + dataSetList[i] + "/dynamics/corr-log-q" + q + ".dat")
            timeStep = ucorr.readFromParams(dirName + "/T" + dataSetList[i] + "/dynamics/", "dt")
            #T.append(ucorr.readFromParams(dirName + "/T" + dataSetList[i] + "/dynamics/", "temperature"))
            energy = np.loadtxt(dirName + "/T" + dataSetList[i] + "/dynamics/energy.dat")
            if(energy[-1,3] < energy[-1,4]):
                T.append(np.mean(energy[:,3]))
            else:
                T.append(np.mean(energy[:,4]))
            print(T[-1], dataSetList[i])
            diff.append(np.mean(data[-10:,1]/(4 * data[-10:,0] * timeStep)))
            tau.append(timeStep*ucorr.computeTau(data))
            deltaChi.append(timeStep*ucorr.computeDeltaChi(data))
            #print("T: ", T[-1], " diffusity: ", Deff[-1], " relation time: ", tau[-1], " tmax:", data[-1,0]*timeStep)
            #plotSPCorr(ax, data[:,0]*timeStep, data[:,1], "$MSD(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), logy = True)
            #plotSPCorr(ax, data[:,0]*timeStep, data[:,1]/data[:,0]*timeStep, "$\\frac{MSD(\\Delta t)}{\\Delta t}$", color = colorList(i/dataSetList.shape[0]), logy = True)
            plotSPCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList(i/dataSetList.shape[0]))
            #plotSPCorr(ax, data[1:,0]*timeStep, data[1:,3], "$\\chi(\\Delta t)$", color = colorList(i/dataSetList.shape[0]))
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pdiff-vsT-" + figureName + "-q" + q + ".png", transparent=True, format = "png")
    T = np.array(T)
    diff = np.array(diff)
    tau = np.array(tau)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    #ax.semilogy(T, diff, linewidth=1.5, color='k', marker='o')
    ax.semilogy(1/T, tau, linewidth=1.5, color='k', marker='o')
    #ax.semilogy(T, diff*tau, linewidth=1.5, color='k', marker='o')
    #ax.semilogy(1/T[2:], deltaChi[2:], linewidth=1.5, color='k', marker='o')
    #ax.set_ylim(0.12, 1.34)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    #ax.set_ylabel("$Diffusivity,$ $D$", fontsize=17)
    ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    #ax.set_ylabel("$D$ $\\tau$", fontsize=17)
    #ax.set_ylabel("$Susceptibility$ $width,$ $\\Delta \\chi$", fontsize=17)
    plt.tight_layout()
    np.savetxt(dirName + "relaxationData-q" + q + ".dat", np.column_stack((T, diff, tau, deltaChi)))
    plt.savefig("/home/francesco/Pictures/soft/ptau-Tu-" + figureName + "-q" + q + ".png", transparent=True, format = "png")
    plt.show()

def plotSPSEvsTemp(dirName, figureName, q="1", indexDr="3", indexf0="3"):
    damping = 1e03
    meanRad = np.mean(np.loadtxt(dirName + "particleRad.dat"))
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    colorList = ['r', 'g', 'b']
    markerList = ['v', 's', 'd']
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), dpi = 120)
    for i in range(DrList.shape[0])[:indexDr]:
        for j in range(f0List.shape[0])[:indexf0]:
            fileName = dirName + "/active-langevin/Dr" + DrList[i] + "-f0" + f0List[j] + "/relaxationData-q" + q + ".dat"
            print(fileName)
            if(os.path.exists(fileName)):
                data = np.loadtxt(fileName)
                label = "$D_r =$" + DrList[i] + "$, f_0=$" + f0List[j]
                Pe = (float(f0List[j])/damping)/(float(DrList[i])*meanRad)
                ax[0].semilogy(1/data[:,0], data[:,2]*np.sqrt(data[:,0]), linewidth=1, color=colorList[j], marker=markerList[i], markersize=6, markeredgewidth=1, fillstyle='none')
                ax[1].semilogy(data[:,0], data[:,1]*data[:,2], linewidth=1.2, color=colorList[j], marker=markerList[i], markersize=6, markeredgewidth=1, fillstyle='none')
    thermalData = np.loadtxt(dirName + "/langevin/relaxationData-q" + q + ".dat")
    ax[0].errorbar(1/thermalData[:,0], thermalData[:,2]*np.sqrt(thermalData[:,0]), linewidth=1.2, marker='o', markersize=6, color='k', fillstyle='none')
    ax[1].errorbar(thermalData[:,0], thermalData[:,1]*thermalData[:,2], linewidth=1.2, marker='o', markersize=6, color='k', fillstyle='none')
    #attractData = np.loadtxt(dirName + "/../../attractData/12/attractive-langevin/relaxationData-q" + q + ".dat")
    #ax[0].semilogy(1/attractData[:,0], attractData[:,2], linewidth=1.2, color='k', linestyle='dotted')
    #ax[1].semilogy(attractData[:,0], attractData[:,1]*attractData[:,2], linewidth=1.2, color='k', linestyle='dotted')
    #ax[0].set_ylim(0.0024,13)
    ax[1].set_ylim(0.056,2.82)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].set_ylabel("$\\tau \\sqrt{T}$", fontsize=15)
    ax[0].set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=15)
    #ax[0].set_ylabel("$Diffusivity,$ $D$", fontsize=15)
    ax[1].set_xlabel("$Temperature,$ $T$", fontsize=15)
    ax[1].set_ylabel("$D$ $\\tau$", fontsize=15)
    fig.tight_layout()
    #plt.subplots_adjust(hspace=0)
    fig.savefig("/home/francesco/Pictures/soft/pSE-vsT" + figureName + "-Dr" + str(indexDr) + "-f0" + str(indexf0) + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelSpaceCorrVSActivity(dirName, sampleName, figureName):
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    colorList = ['r', 'g', 'b']
    markerList = ['v', 's', 'd']
    #fig, ax = plt.subplots(2, 1, sharex=True, figsize = (6, 7), dpi = 120)
    fig, ax = plt.subplots(dpi = 120)
    for i in range(DrList.shape[0]):
        for j in range(f0List.shape[0]):
            fileName = dirName + "/active-langevin/Dr" + DrList[i] + "-f0" + f0List[j] + "/T" + sampleName + "/dynamics/corr-vel-space.dat"
            if(os.path.exists(fileName)):
                print(fileName)
                #speedc, velc = spCorr.computeParticleVelSpaceCorr(dirSample, meanRad, bins)
                data = np.loadtxt(fileName)
                ax.plot(data[:,0], data[:,1], linewidth=1, color=colorList[j], marker=markerList[i], markersize=6, markeredgewidth=1, fillstyle='none')
                #ax[1].plot(data[:,0], data[:,2], linewidth=1, color=colorList[j], marker=markerList[i], markersize=6, markeredgewidth=1, fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlim(0.8,2.7)
    #ax[1].tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Distance,$ $r/\\sigma$", fontsize=17)
    #ax[1].set_xlabel("$Distance,$ $r/\\sigma$", fontsize=15)
    ax.set_ylabel("$Velocity$ $correlation,$ $C_v(r)$", fontsize=17)
    #ax[1].set_ylabel("$Speed$ $correlation,$ $C_s(r)$", fontsize=15)
    plt.tight_layout()
    #fig.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/velCorr-Drf0-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def compareSPDynamicsVSTemp(dirName1, dirName2, figureName, q="1"):
    dataSetList = np.array(["0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", #1e09
                            "0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19", "0.2", "0.3", #1e08
                            "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]) #1e07
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    dirList = np.array([dirName1, dirName2])
    markerList = ['o', 'v']
    colorList = ['k', 'b']
    for d in range(dirList.shape[0]):
        T = []
        diff = []
        tau = []
        deltaChi = []
        for i in range(dataSetList.shape[0]):
            if(d == 0):
                dirSample = dirList[d] + "/T" + dataSetList[i] + "/dynamics/"
            else:
                dirSample = dirList[d] + "/T" + dataSetList[i] + "/dynamics/"
            if(os.path.exists(dirSample + "corr-log-q" + q + ".dat")):
                data = np.loadtxt(dirSample + "corr-log-q" + q + ".dat")
                timeStep = ucorr.readFromParams(dirSample, "dt")
                if(d == 0):
                    T.append(ucorr.readFromParams(dirSample, "temperature"))
                else:
                    energy = np.loadtxt(dirSample + "energy.dat")
                    T.append(np.mean(energy[:,4]))
                diff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
                tau.append(timeStep*ucorr.computeTau(data))
                deltaChi.append(timeStep*ucorr.computeDeltaChi(data))
                #plotSPCorr(ax, data[:,0]*timeStep, data[:,1], "$MSD(\\Delta t)$", color = colorList[d], logy = True)
                #plotSPCorr(ax, data[:,0]*timeStep, data[:,1]/data[:,0]*timeStep, "$\\frac{MSD(\\Delta t)}{\\Delta t}$", color = colorList[d], logy = True)
                plotSPCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList[d])
                #plotSPCorr(ax, data[1:,0]*timeStep, data[1:,3], "$\\chi(\\Delta t)$", color = colorList[d])
        T = np.array(T)
        diff = np.array(diff)
        tau = np.array(tau)
        #ax.semilogy(1/T, diff, linewidth=1.5, color=colorList[d], marker=markerList[d])
        #ax.semilogy(1/T, tau*np.sqrt(T), linewidth=1.5, color=colorList[d], marker=markerList[d])
        #ax.semilogy(T, diff*tau, linewidth=1.5, color=colorList[d], marker=markerList[d])
        #ax.semilogx(1/T, deltaChi, linewidth=1.5, color=colorList[d], marker=markerList[d])
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    ax.set_ylabel("$Diffusivity,$ $D$", fontsize=17)
    #ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    #ax.set_ylabel("$D$ $\\tau$", fontsize=17)
    #ax.set_ylabel("$Susceptibility$ $width,$ $\\Delta \\chi$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/compareTau-vsT-" + figureName + "-q" + q + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSAttraction(dirName, figureName, q="1"):
    u = []
    diff = []
    tau = []
    deltaChi = []
    dataSetList = np.array(["1e-03", "1e-02", "1e-01", "2e-01", "3e-01", "1", "2", "3"]) #1e07
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/T10-u" + dataSetList[i] + "/dynamics/corr-log-q" + q + ".dat")):
            data = np.loadtxt(dirName + "/T10-u" + dataSetList[i] + "/dynamics/corr-log-q" + q + ".dat")
            timeStep = ucorr.readFromParams(dirName + "/T10-u" + dataSetList[i] + "/dynamics/", "dt")
            diff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
            tau.append(timeStep*ucorr.computeTau(data))
            deltaChi.append(timeStep*ucorr.computeDeltaChi(data))
            #plotSPCorr(ax, data[:,0]*timeStep, data[:,1], "$MSD(\\Delta t)$", color = colorList(i/dataSetList.shape[0]), logy = True)
            #plotSPCorr(ax, data[:,0]*timeStep, data[:,1]/data[:,0]*timeStep, "$\\frac{MSD(\\Delta t)}{\\Delta t}$", color = colorList(i/dataSetList.shape[0]), logy = True)
            plotSPCorr(ax, data[1:,0]*timeStep, data[1:,4], "$ISF(\\Delta t)$", color = colorList(i/dataSetList.shape[0]))
            #plotSPCorr(ax, data[1:,0]*timeStep, data[1:,3], "$\\chi(\\Delta t)$", color = colorList(i/dataSetList.shape[0]))
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pisf-vsu-" + figureName + "-q" + q + ".png", transparent=True, format = "png")
    u = np.array(dataSetList).astype(float)
    diff = np.array(diff)
    tau = np.array(tau)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    #ax.semilogx(u, diff, linewidth=1.5, color='k', marker='o')
    ax.semilogx(u, tau, linewidth=1.5, color='k', marker='o')
    #ax.semilogx(T, diff*tau, linewidth=1.5, color='k', marker='o')
    #ax.set_ylim(0.12, 1.34)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_ylabel("$Diffusivity,$ $D$", fontsize=17)
    ax.set_xlabel("$Attraction$ $energy,$ $u$", fontsize=17)
    ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    #ax.set_ylabel("$D$ $\\tau$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pfdt-vsu-" + figureName + "-q" + q + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSPhi(dirName, sampleName, figureName):
    phi = []
    tau = []
    Deff = []
    dirDyn = "/langevin/"
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0])
    fig, ax = plt.subplots(dpi = 150)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics/corr-log-q1.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + dirDyn  + "/T" + sampleName + "/dynamics/corr-log-q1.dat")
            timeStep = ucorr.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "dt")
            phi.append(ucorr.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "phi"))
            Deff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
            tau.append(timeStep*ucorr.computeTau(data))
            print("phi: ", phi[-1], " Deff: ", Deff[-1], " tau: ", tau[-1])
            legendlabel = "$\\varphi=$" + str(np.format_float_positional(phi[-1],4))
            #plotSPCorr(ax, data[1:,0]*timeStep, data[1:,1], "$MSD(\\Delta t)$", color = colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), legendLabel = legendlabel, logy = True)
            plotSPCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), legendLabel = legendlabel)
    #ax.plot(np.linspace(1e-03,1e10,50), np.exp(-1)*np.ones(50), linestyle='--', linewidth=1.5, color='k')
    #ax.set_ylim(3e-06,37100)#2.3e-04
    ax.legend(loc = 'best', fontsize = 11, ncol = 2)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax.set_xlim(3.8e-04, 8.13e05)
    #ax.set_ylim(7.5e-06, 8.8e03)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/pisf-vsphi-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPTauVSActivity(dirName, figureName):
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"])
    colorList = cm.get_cmap('plasma', dataSetList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/tauDiff.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/tauDiff.dat")
            ax.loglog(1/data[:,1], data[:,2], linewidth=1.5, color=colorList(i/dataSetList.shape[0]), marker='o')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    #ax[1].set_ylabel("$Diffusivity,$ $D_{eff}$", fontsize=17)
    ax.set_ylabel("$log(\\tau)$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-active-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPTauVSTemp(dirName, figureName):
    phi0 = 0.8277#0.83867
    mu = 1.1#1.1
    delta = 1.05#1.2
    dataSetList = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0]+10)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/relaxationData.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/relaxationData.dat")
            phi = ucorr.readFromParams(dirName + dataSetList[i], "phi")
            #ax.loglog(1/data[:,0], np.log(data[:,2]), linewidth=1.5, color=colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), marker='o')
            ax.loglog(np.abs(phi - phi0)**(2/mu)/data[:,0], np.abs(phi0 - phi)**(delta) * np.log(np.sqrt(data[:,0])*data[:,2]), linewidth=1.5, color=colorList((dataSetList.shape[0]-i)/dataSetList.shape[0]), marker='o')
    ax.set_xlim(7.6e-04, 4.2e02)
    ax.set_ylim(6.3e-03, 1.8)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    ax.set_xlabel("$|\\varphi - \\varphi_0|^{2/\\mu}/T$", fontsize=17)
    #ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    ax.set_ylabel("$|\\varphi - \\varphi_0|^\\delta \\log(\\tau T^{1/2})$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-vsT-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPTauVSPhi(dirName, sampleName, figureName):
    dataSetList = np.array(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    colorList = cm.get_cmap('viridis', dataSetList.shape[0]+10)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/tauDiff.dat")):
            data = np.loadtxt(dirName + dataSetList[i] + "/active-langevin/T" + sampleName + "/tauDiff.dat")
            data = data[1:,:]
            ax.loglog(data[data[:,4]>0,1], data[data[:,4]>0,4], linewidth=1.5, color=colorList(i/dataSetList.shape[0]), marker='o')
    #ax.set_xlim(1.3, 15300)
    ax.plot(np.linspace(5,100,50), 2e04/np.linspace(5,100,50)**2, linestyle='--', linewidth=1.5, color='k')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Wave$ $vector$ $magnitude,$ $q$", fontsize=17)
    ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/ptau-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPDynamicsVSQ(dirName, figureName):
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    q0 = np.pi/pRad
    qList = np.array(["0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8", "2", "2.5", "3", "4", "5", "6", "8", "10", "12", "15", "20", "30", "40", "50"])
    q = qList.astype(float)
    colorList = cm.get_cmap('viridis', qList.shape[0])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    q = []
    tau = []
    diff = []
    for i in range(qList.shape[0]):
        if(os.path.exists(dirName + "/corr-log-q" + qList[i] + ".dat")):
            data = np.loadtxt(dirName + "/corr-log-q" + qList[i] + ".dat")
            timeStep = ucorr.readFromParams(dirName, "dt")
            legendlabel = "$q=2\\pi/($" + qList[i] + "$\\times d)$"
            plotSPCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList((qList.shape[0]-i)/qList.shape[0]), legendLabel = legendlabel)
            tau.append(timeStep*ucorr.computeTau(data))
            diff.append(np.mean(data[-5:,1]/4*data[-5:,0]))
            q.append(np.pi/(float(qList[i])*pRad * q0))
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    plt.tight_layout()
    #plt.savefig("/home/francesco/Pictures/soft/pisf-vsq-" + figureName + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    tau = np.array(tau)
    diff = np.array(diff)
    np.savetxt(dirName + os.sep + "diffTauVSq.dat", np.column_stack((q, diff, tau)))
    ax.loglog(q, diff*tau, linewidth=1.5, color='b', marker='*')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Wave$ $vector$ $magnitude,$ $|\\vec{q}|$", fontsize=17)
    #ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    ax.set_ylabel("$D \\tau$", fontsize=17)
    plt.tight_layout()
    #plt.savefig("/home/francesco/Pictures/soft/ptau-vsq-" + figureName + "png", transparent=True, format = "png")
    plt.show()

def plotSPBetaVSActivity(dirName, sampleName, figureName, start):
    damping = 1e03
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    q0 = np.pi/pRad
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    qList = np.array(["0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8", "2", "2.5", "3", "4", "5", "6", "8", "10", "12", "15", "20", "30"])
    Pe = []
    beta = []
    std = []
    for Dr in DrList:
        for f0 in f0List:
            tau = []
            q = []
            Pe.append((float(f0)/damping)/(float(Dr)*2*pRad))
            dirSample = dirName + os.sep + "active-langevin/Dr" + Dr + "-f0" + f0 + "/T" + sampleName
            for i in range(qList.shape[0]):
                if(os.path.exists(dirSample + "/dynamics/corr-log-q" + qList[i] + ".dat")):
                    data = np.loadtxt(dirSample + "/dynamics/corr-log-q" + qList[i] + ".dat")
                    timeStep = ucorr.readFromParams(dirSample, "dt")
                    tau.append(timeStep*ucorr.computeTau(data, 2))
                    q.append(np.pi/(float(qList[i])*pRad * q0))
            tau = np.array(tau)
            q = np.array(q)
            ax.loglog(q, tau, linewidth=1.5, color='k', marker='o', markersize=6, markeredgewidth=1, fillstyle='none')
            plt.pause(0.5)
            beta.append(np.mean((np.log(tau[start+1:]) - np.log(tau[start:-1]))/(np.log(q[start+1:]) - np.log(q[start:-1]))))
            std.append(np.std((np.log(tau[start+1:]) - np.log(tau[start:-1]))/(np.log(q[start+1:]) - np.log(q[start:-1])))/(qList.shape[0]-start-1))
    ax.clear()
    beta = np.array(beta)
    std = np.array(std)
    Pe = np.array(Pe)
    beta = beta[np.argsort(Pe)]
    std = std[np.argsort(Pe)]
    Pe = np.sort(Pe)
    uplot.plotErrorBar(ax, Pe, beta, std, "$Pe$", "$\\beta$", logx = True, logy = False)
    ax.set_ylim(-2.18, -1.12)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/pcompareBeta-" + figureName
    plt.savefig(figureName + "-vsDrf0" + "-T" + sampleName + ".png", transparent=True, format = "png")
    plt.show()

def compareSPDynamicsVSQ(dirName, sampleName, figureName, index = 0, fixed = "f0"):
    damping = 1e03
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    q0 = np.pi/pRad
    qList = np.array(["0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8", "2", "2.5", "3", "4", "5", "6", "8", "10", "12", "15", "20", "30"])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    dirList, labelList, colorList, markerList = uplot.getDirLabelColorMarker(dirName, sampleName, index, fixed)
    Pe = []
    for Dr in DrList:
        for f0 in f0List:
            Pe.append((float(f0)/damping)/(float(Dr)*2*pRad))
    for d in range(dirList.shape[0]):
        tau = []
        q = []
        diff = []
        for i in range(qList.shape[0]):
            if(os.path.exists(dirList[d] + "/dynamics/corr-log-q" + qList[i] + ".dat")):
                data = np.loadtxt(dirList[d] + "/dynamics/corr-log-q" + qList[i] + ".dat")
                timeStep = ucorr.readFromParams(dirList[d], "dt")
                tau.append(timeStep*ucorr.computeTau(data, 2))
                diff.append(np.mean(data[-5:,1]/(4*data[-5:,0])))
                q.append(np.pi/(float(qList[i])*pRad * q0))
        tau = np.array(tau)
        diff = np.array(diff)
        ax.loglog(q, tau, linewidth=1.5, color=colorList[d], marker=markerList[d], label=labelList[d], markersize=6, markeredgewidth=1, fillstyle='none')
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$|\\vec{q}(r)|/q_\\sigma$", fontsize=17)
    #ax.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=17)
    ax.set_ylabel("$\\tau$", fontsize=17)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/pcompareTau-vsq-" + figureName
    if(fixed == "f0"):
        figureName += "-f0" + f0List[index] + "-vsDr" + "-T" + sampleName
    elif(fixed == "Dr"):
        figureName += "-Dr" + DrList[index] + "-vsf0" + "-T" + sampleName
    else:
        figureName += "-T" + sampleName
    plt.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPairCorrVSActivity(dirName, sampleName, figureName, start, end, index=0, fixed="f0"):
    fig1, ax1 = plt.subplots(2, 1, figsize = (6, 7), dpi = 120)
    #fig2, ax2 = plt.subplots(figsize = (7, 5), dpi = 120)
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    dirList, labelList, colorList, markerList = uplot.getDirLabelColorMarker(dirName, sampleName, index, fixed)
    for d in range(dirList.shape[0]):
        if(os.path.exists(dirList[d] + "/dynamics/pairCorr.dat")):
            pc = np.loadtxt(dirList[d] + os.sep + "dynamics/pairCorr.dat")
            sf = np.loadtxt(dirList[d] + os.sep + "dynamics/structureFactor.dat")
            ax1[0].plot(pc[start:end,0], pc[start:end,1], marker=markerList[d], color=colorList[d], label=labelList[d], markersize=5, fillstyle='none', lw=1)
            #ax1[1].plot(pc[start:end,0], -T*np.log(pc[start:end,1]), marker=markerList[d], color=colorList[d], label=labelList[d], markersize=5, fillstyle='none', lw=1)
            sf = sf[sf[:,0]<20]
            ax1[1].plot(sf[:,0], sf[:,1], color=colorList[d], label=labelList[d], lw=1)
    ax1[1].legend(fontsize=10, loc='best')
    ax1[0].tick_params(axis='both', labelsize=14)
    ax1[1].tick_params(axis='both', labelsize=14)
    ax1[0].set_xlabel("$r$", fontsize=18)
    ax1[0].set_ylabel("$g(r)$", fontsize=18)
    ax1[1].set_xlabel("$q$", fontsize=18)
    ax1[1].set_ylabel("$S(q)$", fontsize=18)
    fig1.tight_layout()
    figureName = "/home/francesco/Pictures/soft/pcompareStructure-" + figureName
    if(fixed == "f0"):
        figureName += "-f0" + f0List[index] + "-vsDr" + "-T" + sampleName
    elif(fixed == "Dr"):
        figureName += "-Dr" + DrList[index] + "-vsf0" + "-T" + sampleName
    else:
        figureName += "-T" + sampleName
    fig1.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelCorrVSActivity(dirName, sampleName, figureName, index=0, fixed="f0"):
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), dpi = 120)
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    dirList, labelList, colorList, markerList = uplot.getDirLabelColorMarker(dirName, sampleName, index, fixed)
    for d in range(dirList.shape[0]):
        if(os.path.exists(dirList[d] + "/dynamics/corr-vel-space.dat")):
            vsf = np.loadtxt(dirList[d] + os.sep + "dynamics/velocitySF.dat")
            vcorr = np.loadtxt(dirList[d] + os.sep + "dynamics/velTimeCorr-d1.dat")
            #ax[0].plot(vs[start:end,0], vs[start:end,1]/(2*np.pi*vs[start:end,0]), marker=markerList[d], color=colorList[d], label=labelList[d], markersize=3, fillstyle='none', lw=1)
            ax[0].semilogx(vsf[vsf[:,0]<20,0], vsf[vsf[:,0]<20,1]/np.mean(vsf[-20:,1]), marker=markerList[d], color=colorList[d], label=labelList[d], markersize=3, fillstyle='none', lw=1)
            ax[1].semilogx(vcorr[:,0], vcorr[:,4], marker=markerList[d], color=colorList[d], label=labelList[d], markersize=3, fillstyle='none', lw=1)
    ax[1].legend(fontsize=10, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel("$q$", fontsize=18)
    ax[0].set_ylabel("$C_{vv}(q)$", fontsize=18)
    ax[1].set_xlabel("$\\Delta t$", fontsize=18)
    ax[1].set_ylabel("$ISF_{vv}(\\Delta t)$", fontsize=18)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/pcompareVelCorr-" + figureName
    if(fixed == "f0"):
        figureName += "-f0" + f0List[index] + "-vsDr" + "-T" + sampleName
    elif(fixed == "Dr"):
        figureName += "-Dr" + DrList[index] + "-vsf0" + "-T" + sampleName
    else:
        figureName += "-T" + sampleName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelCorrVSQ(dirName, figureName):
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    q0 = np.pi/pRad
    qList = np.array(["0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8", "2", "2.5", "3", "4", "5", "6", "8", "10", "12", "15", "20", "30", "40", "50"])
    q = qList.astype(float)
    colorList = cm.get_cmap('viridis', qList.shape[0]+1)
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(qList.shape[0]):
        if(os.path.exists(dirName + "/velTimeCorr-d" + qList[i] + ".dat")):
            data = np.loadtxt(dirName + "/velTimeCorr-d" + qList[i] + ".dat")
            timeStep = ucorr.readFromParams(dirName, "dt")
            legendlabel = "$q=2\\pi/($" + qList[i] + "$\\; \\sigma)$"
            plotSPCorr(ax, data[:,0]*timeStep, data[:,4], "$ISF_v(\\Delta t)$", color = colorList((qList.shape[0]-i)/qList.shape[0]), legendLabel = legendlabel)
    ax.set_xlabel("$Time$ $interval,$ $\\Delta t$", fontsize=18)
    ax.legend(fontsize=10, loc='best', ncol=2)
    plt.tight_layout()
    #plt.savefig("/home/francesco/Pictures/soft/pisf-vsq-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

############################## plot dynamics FDT ###############################
def plotSPEnergyScale(dirName, sampleName, figureName):
    Dr = []
    T = []
    pressure = []
    timeStep = 3e-04
    dataSetList = np.array(["1e03", "1e02", "1e01", "1", "1e-01", "1e-02", "1e-03", "1e-04"])
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    data = np.loadtxt(dirName + "../../T1/energy.dat")
    ax[0].semilogx(1/timeStep, np.mean(data[10:,4]), color='g', marker='$B$', markersize = 10, markeredgewidth = 0.2, alpha=0.2)
    ax[1].semilogx(1/timeStep, np.mean(data[10:,6]), color='g', marker='$B$', markersize = 10, markeredgewidth = 0.2, alpha=0.2)
    for i in range(dataSetList.shape[0]):
        if(os.path.exists(dirName + "/Dr" + dataSetList[i] + "-" + sampleName + "/dynamics/")):
            data = np.loadtxt(dirName + "/Dr" + dataSetList[i] + "-" + sampleName + "/dynamics/energy.dat")
            Dr.append(float(dataSetList[i]))
            T.append(np.mean(data[10:,4]))
            pressure.append(np.mean(data[10:,6]))
    ax[0].tick_params(axis='both', labelsize=15)
    ax[1].tick_params(axis='both', labelsize=15)
    ax[0].semilogx(Dr, T, linewidth=1.2, color='k', marker='o')
    ax[1].semilogx(Dr, pressure, linewidth=1.2, color='k', marker='o')
    ax[0].set_xlabel("$Persistence$ $time,$ $1/D_r$", fontsize=18)
    ax[1].set_xlabel("$Persistence$ $time,$ $1/D_r$", fontsize=18)
    ax[0].set_xlabel("$Rotational$ $diffusion,$ $D_r$", fontsize=18)
    ax[1].set_xlabel("$Rotational$ $diffusion,$ $D_r$", fontsize=18)
    ax[0].set_ylabel("$Temperature,$ $T$", fontsize=18)
    ax[1].set_ylabel("$Pressure,$, $p$", fontsize=18)
    ax[0].set_ylim(0.98,3.8)#1.15)#
    ax[1].set_ylim(5e-05,6.4e-03)#9.6e-04)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/soft-Tp-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelPDFVSMass(dirName, firstIndex, figureName):
    #dataSetList = np.array(["1e03", "5e03", "1e04", "5e04", "1e05", "5e05", "1e06"])
    #massList = np.array([1e03, 5e03, 1e04, 5e04, 1e05, 5e05, 1e06])
    dataSetList = np.array(["5e04", "1e05", "5e05", "1e06", "5e06"])
    massList = np.array([5e04, 1e05, 5e05, 1e06, 5e06])
    colorList = cm.get_cmap('plasma', massList.shape[0] + 1)
    fig = plt.figure(0, dpi=120)
    ax = fig.gca()
    for i in range(massList.shape[0]):
        scale = np.sqrt(massList[i])
        vel = []
        dirSample = dirName + os.sep + "dynamics-mass" + dataSetList[i]
        for dir in os.listdir(dirSample):
            if(os.path.isdir(dirSample + os.sep + dir)):
                vel.append(np.loadtxt(dirSample + os.sep + dir + os.sep + "particleVel.dat")[:firstIndex])
        vel = np.array(vel).flatten()
        mean = np.mean(vel) * scale
        Temp = np.var(vel) * scale**2
        alpha2 = np.mean((vel * scale - mean)**4)/(3 * Temp**2) - 1
        velPDF, edges = np.histogram(vel, bins=np.linspace(np.min(vel), np.max(vel), 60), density=True)
        edges = 0.5 * (edges[:-1] + edges[1:])
        print("Mass:", massList[i], " variance: ", Temp, " alpha2: ", alpha2)
        ax.semilogy(edges[velPDF>0] * scale, velPDF[velPDF>0] / scale, linewidth=1.5, color=colorList(i/massList.shape[0]), label="$m =$" + dataSetList[i])
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$P(v) / m^{1/2}$", fontsize=17)
    ax.set_xlabel("$v m^{1/2}$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/velSubSet-" + figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPDensityVarVSTime(dirName, sampleName, numBins, figureName):
    dataSetList = np.array(["1", "1e-01", "1e-02"])
    if(sampleName == "10"):
        pressureList = np.array(["400", "400", "400"])
    else:
        pressureList = np.array(["485", "495", "560"])
    colorList = ['k', 'b', 'g']
    markerList = ['o', 'v', '*']
    fig, ax = plt.subplots(figsize = (8,4), dpi = 120)
    for i in range(dataSetList.shape[0]):
        var = []
        phi = []
        step = []
        dirSample = dirName + "Dr" + dataSetList[i] + "/Dr" + dataSetList[i] + "-f0" + sampleName + "/dynamics-ptot" + pressureList[i] + "/"
        #dirSample = dirName + "Dr" + dataSetList[i] + "/Dr" + dataSetList[i] + "-f0" + sampleName + "/dynamics-test/"
        for dir in os.listdir(dirSample):
            if(os.path.exists(dirSample + os.sep + dir + os.sep + "restAreas.dat")):
                if(float(dir[1:])%1e04 == 0):
                    localDensity = spCorr.computeLocalDensity(dirSample + os.sep + dir, numBins)
                    var.append(np.std(localDensity)/np.mean(localDensity))
                    phi.append(ucorr.readFromParams(dirSample + os.sep + dir, "phi"))
                    step.append(int(dir[1:]))
        var = np.array(var)
        phi = np.array(phi)
        step = np.array(step)
        var = var[np.argsort(step)]
        phi = phi[np.argsort(step)]
        step = np.sort(step)
        plt.plot(step, var, color=colorList[i], lw=1, marker=markerList[i], markersize=4)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('$Simulation$ $step$', fontsize=18)
    ax.set_ylabel('$\\Delta \\varphi / \\varphi$', fontsize=18)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/localDensity-vsPhi-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDensityPDF(dirName, sampleName, numBins, figureName):
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    colorList = ['r', 'g', 'b']
    markerList = ['v', 's', 'd']
    fig, ax = plt.subplots(dpi = 120)
    for i in range(DrList.shape[0]):
        for j in range(f0List.shape[0]):
            dirSample = dirName + "/Dr" + DrList[i] + "-f0" + f0List[j] + "/T" + sampleName + "/dynamics/"
            if(os.path.exists(dirSample + os.sep + "localDensity-N" + numBins + ".dat")):
                data = np.loadtxt(dirSample + os.sep + "localDensity-N" + numBins + ".dat")
                data = data[data[:,1]>0]
                if(i == 2 and j == 2):
                    ax.plot(data[:,0], data[:,1], linewidth=1.2, marker=markerList[i], color=colorList[j], fillstyle='none')
                else:
                    ax.plot(data[:,0], data[:,1], linewidth=1.2, marker=markerList[i], color=colorList[j], fillstyle='none')
    data = np.loadtxt(dirName + "../langevin/T" + sampleName + "/dynamics/localDensity-N" + numBins + ".dat")
    data = data[data[:,1]>0]
    ax.plot(data[1:,0], data[1:,1], linewidth=1.2, marker='*', markersize=12, color='k', fillstyle='none', markeredgewidth=1.5)
    ax.tick_params(axis='both', labelsize=15)
    ax.set_ylabel('$PDF(\\varphi)$', fontsize=18)
    ax.set_xlabel('$\\varphi$', fontsize=18)
    ax.set_yscale('log')
    #ax.set_xlim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/densityPDF-active-vsDrf0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPLocalDensityPDFvsTemp(dirName1, dirName2, sampleName, numBins, figureName):
    T = []
    deltaPhi = []
    dataSetList = np.array(["0.06", "0.07", "0.08", "0.09", "0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19",
                            "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    dirList = np.array([dirName1, dirName2])
    markerList = ['o', 'v']
    colorList = ['k', 'b']
    for d in range(dirList.shape[0]):
        T = []
        deltaPhi = []
        for i in range(dataSetList.shape[0]):
            if(d == 0):
                dirSample = dirList[d] + "/T" + dataSetList[i] + "/dynamics/"
            else:
                dirSample = dirList[d] + "/T" + dataSetList[i] + "/dynamics/"
            if(os.path.exists(dirSample + "localDensity-N" + numBins + ".dat")):
                data = np.loadtxt(dirSample + "localDensity-N" + numBins + ".dat")
                if(d == 0):
                    T.append(ucorr.readFromParams(dirSample, "temperature"))
                else:
                    energy = np.loadtxt(dirSample + "energy.dat")
                    T.append(np.mean(energy[:,4]))
                    deltaPhi.append(spCorr.computeLocalDensityPDF(dirSample, numBins, plot="plot"))
        T = np.array(T)
        deltaPhi = np.array(deltaPhi)
        ax.semilogy(1/T, deltaPhi, linewidth=1.5, color=colorList[d], marker=markerList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Inverse$ $temperature,$ $1/T$", fontsize=17)
    ax.set_ylabel("$Variance$ $of$ $local$ $density,$ $\\sigma_\\varphi$", fontsize=17)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/plocalDensityPDF-vsT-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPLocalDensityPDFvsActivity(dirName, numBins, figureName):
    damping = 1e03
    meanRad = np.mean(np.loadtxt(dirName + "../particleRad.dat"))
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    markerList = ['v', 'o', 's']
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(DrList.shape[0]):
        T = []
        Tsubset = []
        deltaPhi = []
        for j in range(f0List.shape[0]):
            dirSample = dirName + "/Dr" + DrList[i] + "-f0" + f0List[j] + "/dynamics-mass1e05"
            if(os.path.exists(dirSample)):
                t, tsubset = spCorr.computeParticleVelPDFSubSet(dirSample, firstIndex=10, mass=1e06, plot=False)
                T.append(t)
                Tsubset.append(tsubset)
                deltaPhi.append(spCorr.computeLocalDensityPDF(dirSample, numBins))
        np.savetxt(dirName + "/Dr" + DrList[i] + "/localDensityData.dat", np.column_stack((T, Tsubset, deltaPhi)))
        Tsubset = np.array(Tsubset)
        ax.semilogx(Tsubset, deltaPhi, linewidth=1.2, color='k', marker=markerList[i], fillstyle='none', markersize=8, markeredgewidth=1.5)
    thermalData = np.loadtxt(dirName + "../../glassFDT/localDensityData.dat")
    ax.semilogx(thermalData[:,0], thermalData[:,1], linewidth=1.2, color='k', linestyle='--')
    ax.legend(("$D_r = 1$", "$D_r = 0.1$", "$D_r = 0.01$", "$thermal$"), fontsize=14, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature$ $T,$ $T_{FDT}$", fontsize=18)
    ax.set_ylabel("$Variance$ $of$ $PDF(\\varphi)$", fontsize=18)
    fig.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/pPDFphi-Drf0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPFDTSusceptibility(dirName, figureName, Dr, driving):
    tmeasure = 100
    fextStr = np.array(["2", "3", "4"])
    fext = fextStr.astype(float)
    mu = np.zeros((fextStr.shape[0],2))
    T = np.zeros((fextStr.shape[0],2))
    #fig0, ax0 = plt.subplots(dpi = 120)
    fig, ax = plt.subplots(1, 2, figsize = (12.5, 5), dpi = 120)
    corr = np.loadtxt(dirName + os.sep + "dynamics/corr-log-q1.dat")
    timeStep = ucorr.readFromParams(dirName + os.sep + "dynamics/", "dt")
    #plotSPCorr(ax0, corr[1:,0]*timeStep, corr[1:,1]/(corr[1:,0]*timeStep), "$MSD(\\Delta t) / \\Delta t$", color = 'k', logy = True)
    timeStep = ucorr.readFromParams(dirName + os.sep + "dynamics", "dt")
    diff = np.mean(corr[corr[:,0]*timeStep>tmeasure,1]/(2*corr[corr[:,0]*timeStep>tmeasure,0]*timeStep))
    for i in range(fextStr.shape[0]):
        sus = np.loadtxt(dirName + os.sep + "dynamics-fext" + fextStr[i] + "/susceptibility.dat")
        sus = sus[sus[:,0]>tmeasure,:]
        mu[i,0] = np.mean(sus[:,1]/sus[:,0])
        mu[i,1] = np.std(sus[:,1]/sus[:,0])
        energy = np.loadtxt(dirName + os.sep + "dynamics-fext" + fextStr[i] + "/energy.dat")
        energy = energy[energy[:,0]>tmeasure,:]
        T[i,0] = np.mean(energy[:,4])
        T[i,1] = np.std(energy[:,4])
    ax[0].errorbar(fext, mu[:,0], mu[:,1], color='k', marker='o', markersize=8, lw=1, ls='--', capsize=3)
    ax[1].errorbar(fext, T[:,0], T[:,1], color='b', marker='D', fillstyle='none', markeredgecolor = 'b', markeredgewidth = 1.5, markersize=8, lw=1, ls='--', capsize=3)
    ax[1].errorbar(fext, diff/mu[:,0], mu[:,1], color='k', marker='o', markersize=8, lw=1, ls='--', capsize=3)
    for i in range(ax.shape[0]):
        ax[i].tick_params(axis='both', labelsize=15)
        ax[i].set_xlabel("$f_0$", fontsize=18)
    ax[0].set_ylabel("$Mobility,$ $\\chi / t = \\mu$", fontsize=18)
    ax[1].set_ylabel("$Temperature$", fontsize=18)
    ax[1].legend(("$Kinetic,$ $T$", "$FDT,$ $D/ \\mu = T_{FDT}$"), loc='best', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    #fig.savefig("/home/francesco/Pictures/soft/pFDT-" + figureName + ".png", transparent=True, format = "png")
    pSize = 2 * np.mean(np.array(np.loadtxt(dirName + os.sep + "dynamics/particleRad.dat")))
    Pe = pSize * driving / 1e-02
    Pev = ((driving / 1e03) / Dr) / pSize
    print("Pe: ", Pev, " susceptibility: ",  np.mean(mu[i,0]), " diffusivity: ", diff, " T_FDT: ", diff/np.mean(mu[i,0]))
    np.savetxt(dirName + "FDTtemp.dat", np.column_stack((Pe, Pev, np.mean(T[:,0]), np.std(T[:,0]), np.mean(mu[:,0]), np.std(mu[:,0]), diff)))
    plt.show()

def plotSPFDTdata(dirName, firstIndex, mass, figureName):
    damping = 1e03
    meanRad = np.mean(np.loadtxt(dirName + "../particleRad.dat"))
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["10", "20", "40", "60", "80", "100"])
    markerList = ['v', 'o', 's']
    fig1, ax1 = plt.subplots(figsize = (7, 5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize = (7, 5), dpi = 120)
    for i in range(DrList.shape[0]):
        Dr = []
        f0 = []
        Pe = []
        T = []
        Tsubset = []
        diff = []
        tau = []
        deltaChi = []
        Treduced = []
        for j in range(f0List.shape[0]):
            dirSample = dirName + "/Dr" + DrList[i] + "/Dr" + DrList[i] + "-f0" + f0List[j] + "/dynamics-mass1e06"
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + os.sep + "../dynamics/corr-log-q1.dat")
                timeStep = ucorr.readFromParams(dirSample + os.sep + "../dynamics", "dt")
                diff.append(np.mean(data[-10:,1]/(4 * data[-10:,0] * timeStep)))
                tau.append(timeStep*ucorr.computeTau(data))
                deltaChi.append(ucorr.computeDeltaChi(data))
                Dr.append(float(DrList[i]))
                f0.append(float(f0List[j]))
                Pe.append(((float(f0List[j])/damping) / float(DrList[i])) / (2 * meanRad))
                t, tsubset = spCorr.computeParticleVelPDFSubSet(dirSample, firstIndex, mass, plot=False)
                T.append(t)
                Tsubset.append(tsubset)
                Treduced.append(Tsubset[-1]*f0[-1]/(Dr[-1] * damping * 2 * meanRad))
        np.savetxt(dirName + "/Dr" + DrList[i] + "/FDTdata.dat", np.column_stack((Dr, f0, Pe, T, Tsubset, tau, diff, deltaChi)))
        Pe = np.array(Pe)
        Tsubset = np.array(Tsubset)
        tau = np.array(tau)
        diff = np.array(diff)
        Dr = np.array(Dr)
        f0 = np.array(f0)
        Treduced = np.array(Treduced)
        ax1.loglog(Pe, Treduced, linewidth=1.2, color='k', marker=markerList[i], fillstyle='none', markersize=10, markeredgewidth=1.5)
        #ax2.loglog(Treduced, tau*diff, linewidth=1.2, color='k', marker=markerList[i], fillstyle='none', markersize=8, markeredgewidth=1.5)
        print("energy scale: ", Tsubset/Treduced)
    thermalData = np.loadtxt(dirName + "../../thermal88/langevin/relaxationData-q1.dat")
    ax2.semilogx(thermalData[:,0], thermalData[:,1]*thermalData[:,2], linewidth=1.2, color='k', linestyle='--')
    ax2.legend(("$D_r = 1$", "$D_r = 0.1$", "$D_r = 0.01$", "$thermal$"), fontsize=14, loc='best')
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel("$Peclet$ $number,$ $v_0/(D_r \\sigma)$", fontsize=18)
    ax1.set_ylabel("$T_{FDT}/\\epsilon_A$", fontsize=18)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel("$T_{FDT}/\\epsilon_A, T/\\epsilon$", fontsize=18)
    #ax2.set_ylabel("$Diffusivity,$ $D$", fontsize=18)
    #ax2.set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=18)
    ax2.set_ylabel("$D$ $\\tau$", fontsize=18)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig("/home/francesco/Pictures/soft/pPeTfdt-Drf0-" + figureName + ".png", transparent=True, format = "png")
    fig2.savefig("/home/francesco/Pictures/soft/pfdt-Drf0-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPFDTvsTemp(dirName, figureName, q="1", index=3):
    damping = 1e03
    meanRad = np.mean(np.loadtxt(dirName + "../particleRad.dat"))
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    TList = np.array([#"0.06", "0.07", "0.08", "0.09", "0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19",
                    "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "2", "7", "8", "9", "10"])
    colorList = ['r', 'g', 'b']
    markerList = ['v', 's', 'd']
    fig, ax = plt.subplots(2, 1, figsize = (6, 7), dpi = 120)
    for i in range(DrList.shape[0])[:index]:
        Dr = DrList[i]
        for j in range(f0List.shape[0]):
            f0 = f0List[j]
            diff = []
            tau = []
            deltaChi = []
            Pe = [] #this times the energy scale for repulsive interactions is an energy scale for active forces
            Temp = []
            Tsubset = []
            Treduced = []
            for T in TList:
                dirSample = dirName + "/Dr" + Dr + "-f0" + f0 + "/T" + T + "/"
                if(os.path.exists(dirSample + "dynamics/corr-log-q" + q + ".dat")):
                    data = np.loadtxt(dirSample + "dynamics/corr-log-q" + q + ".dat")
                    timeStep = ucorr.readFromParams(dirSample + "dynamics", "dt")
                    diff.append(np.mean(data[-5:,1]/(4 * data[-5:,0] * timeStep)))
                    tau.append(timeStep*ucorr.computeTau(data))
                    deltaChi.append(ucorr.computeDeltaChi(data))
                    energy = np.loadtxt(dirSample + "dynamics/energy.dat")
                    Pe.append(float(f0)/ (damping * float(Dr) * 2 * meanRad))
                    #Temp.append(np.mean(energy[:,4]))
                    #velVar = np.array(np.loadtxt(dirSample + "dynamics-mass1e05/tracerTemp.dat"))
                    velVar = spCorr.computeParticleVelPDFSubSet(dirSample + "dynamics-mass1e05/", firstIndex=20, mass=1e05)
                    Temp.append(velVar[0])
                    Tsubset.append(velVar[1])
                    Treduced.append(Tsubset[-1]*Pe[-1])
            np.savetxt(dirName + "/Dr" + Dr + "-f0" + f0 + "/FDTdata.dat", np.column_stack((Temp, Tsubset, Pe, tau, diff, deltaChi)))
            Pe = np.array(Pe)
            Temp = np.array(Temp)
            Tsubset = np.array(Tsubset)
            Treduced = np.array(Treduced)
            tau = np.array(tau)
            diff = np.array(diff)
            label = "$D_r =$" + DrList[i] + "$, f_0=$" + f0List[j]
            ax[0].semilogy(1/Tsubset, tau, linewidth=1, color=colorList[j], marker=markerList[i], markersize=6, markeredgewidth=1, fillstyle='none')
            ax[1].semilogy(Tsubset, diff*tau, linewidth=1.2, color=colorList[j], marker=markerList[i], markersize=6, markeredgewidth=1, fillstyle='none')
    thermalData = np.loadtxt(dirName + "../langevin/relaxationData-q" + q + ".dat")
    ax[0].semilogy(1/thermalData[6:,0], thermalData[6:,2], linewidth=1.2, color='k', linestyle='--')
    ax[1].semilogy(thermalData[:,0], thermalData[:,1]*thermalData[:,2], linewidth=1.2, color='k', linestyle='--')
    attractData = np.loadtxt(dirName + "../../../attractData/12/attractive-langevin/relaxationData-q" + q + ".dat")
    ax[0].semilogy(1/attractData[:,0], attractData[:,2], linewidth=1.2, color='k', linestyle='dotted')
    ax[1].semilogy(attractData[:,0], attractData[:,1]*attractData[:,2], linewidth=1.2, color='k', linestyle='dotted')
    #ax[0].legend(fontsize=10, loc='best')
    ax[0].set_ylim(0.0024,13)
    ax[1].set_ylim(0.056,1.82)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].set_ylabel("$Relaxation$ $time,$ $\\tau$", fontsize=15)
    ax[0].set_xlabel("$Inverse$ $temperature,$ $1/T_{FDT}$", fontsize=15)
    #ax[0].set_ylabel("$Diffusivity,$ $D$", fontsize=15)
    ax[1].set_xlabel("$Temperature,$ $T_{FDT}$", fontsize=15)
    ax[1].set_ylabel("$D$ $\\tau$", fontsize=15)
    fig.tight_layout()
    #plt.subplots_adjust(hspace=0)
    fig.savefig("/home/francesco/Pictures/soft/pFDT-vsT" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotTempDensityHeatMap(dirName, numBins):
    fig, ax = plt.subplots(1, 2, figsize = (12, 5), dpi = 120)
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    visuals.setBigBoxAxes(boxSize, ax[0])
    visuals.setBigBoxAxes(boxSize, ax[1])
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    xedges = (xbin[:-1] + xbin[1:])*0.5
    yedges = (ybin[:-1] + ybin[1:])*0.5
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    localTemp = np.zeros((numBins, numBins))
    localArea = np.zeros((numBins, numBins))
    pRad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    pVel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
    Temp = np.mean(np.linalg.norm(pVel,axis=1)**2)
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    pPos[:,0] -= np.floor(pPos[:,0]/boxSize[0]) * boxSize[0]
    pPos[:,1] -= np.floor(pPos[:,1]/boxSize[1]) * boxSize[1]
    ucorr.computeLocalTempGrid(pPos, pVel, xbin, ybin, localTemp)
    ucorr.computeLocalAreaGrid(pPos, pRad, xbin, ybin, localArea)
    #localTemp /= Temp
    localArea /= localSquare
    c = ax[0].pcolormesh(xedges, yedges, localArea, cmap='Greys', vmin=np.min(localArea), vmax=np.max(localArea))
    ax[0].set_title('density map')
    # set the limits of the plot to the limits of the data
    ax[0].axis([xedges.min(), xedges.max(), yedges.min(), yedges.max()])
    fig.colorbar(c, ax=ax[0])
    c = ax[1].pcolormesh(xedges, yedges, localTemp, cmap='Greys', vmin=np.min(localTemp), vmax=np.max(localTemp))
    ax[1].set_title('temperature map')
    # set the limits of the plot to the limits of the data
    ax[1].axis([xedges.min(), xedges.max(), yedges.min(), yedges.max()])
    fig.colorbar(c, ax=ax[1])
    fig.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/density-temp-map-" + figureName + "-N" + str(numBins) + ".png", transparent=True, format = "png")
    plt.show()


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

    if(whichPlot == "energy"):
        figureName = sys.argv[3]
        plotEnergy(dirName, figureName)

    elif(whichPlot == "compare"):
        dirName1 = dirName + sys.argv[3]
        dirName2 = dirName + sys.argv[4]
        figureName = sys.argv[5]
        compareEnergy(dirName1, dirName2, figureName)

    elif(whichPlot == "energynum"):
        whichDir = sys.argv[3]
        figureName = sys.argv[4]
        plotEnergyVSSystemSize(dirName, whichDir, figureName)

    elif(whichPlot == "pressure"):
        figureName = sys.argv[3]
        bound = sys.argv[4]
        prop = sys.argv[5]
        plotPressure(dirName, figureName, bound, prop)

    elif(whichPlot == "comparep"):
        dirName1 = dirName + sys.argv[3]
        dirName2 = dirName + sys.argv[4]
        figureName = sys.argv[5]
        comparePressure(dirName1, dirName2, figureName)

    elif(whichPlot == "apressure"):
        figureName = sys.argv[3]
        bound = sys.argv[4]
        prop = sys.argv[5]
        plotAveragePressure(dirName, figureName, bound, prop)

    elif(whichPlot == "dropletptime"):
        figureName = sys.argv[3]
        plotDropletPressureVSTime(dirName, figureName)

    elif(whichPlot == "clusterptime"):
        figureName = sys.argv[3]
        bound = sys.argv[4]
        prop = sys.argv[5]
        plotClusterPressureVSTime(dirName, figureName, bound, prop)

    elif(whichPlot == "simplex"):
        figureName = sys.argv[3]
        pad = float(sys.argv[4])
        logy = sys.argv[5]
        plotSimplexDensity(dirName, figureName, pad, logy)

    elif(whichPlot == "forces"):
        index0 = int(sys.argv[3])
        index1 = int(sys.argv[4])
        index2 = int(sys.argv[5])
        dim = int(sys.argv[6])
        plotParticleForces(dirName, index0, index1, index2, dim)

    elif(whichPlot == "active"):
        figureName = sys.argv[3]
        plotActiveEnergy(dirName, figureName)

    elif(whichPlot == "energyphi"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        compareEnergyVSPhi(dirName, sampleName, figureName)

    elif(whichPlot == "energyf0"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        compareEnergyVSActivity(dirName, sampleName, figureName)

    elif(whichPlot == "collision"):
        figureName = sys.argv[3]
        scaled = sys.argv[4]
        dyn = sys.argv[5]
        fixed = sys.argv[6]
        which = sys.argv[7]
        plotSPCollision(dirName, figureName, scaled, dyn, fixed, which)

    elif(whichPlot == "fourier"):
        fileName = sys.argv[3]
        figureName = sys.argv[4]
        dyn = sys.argv[5]
        fixed = sys.argv[6]
        which = sys.argv[7]
        plotSPFourierEnergy(dirName, fileName, figureName, dyn, fixed, which)

    elif(whichPlot == "fouriercorr"):
        fileName = sys.argv[3]
        figureName = sys.argv[4]
        dyn = sys.argv[5]
        fixed = sys.argv[6]
        which = sys.argv[7]
        plotSPFourierCorr(dirName, fileName, figureName, dyn, fixed, which)

    elif(whichPlot == "velcorr"):
        figureName = sys.argv[3]
        scaled = sys.argv[4]
        dyn = sys.argv[5]
        fixed = sys.argv[6]
        which = sys.argv[7]
        plotSPVelCorr(dirName, figureName, scaled, dyn, fixed, which)

    elif(whichPlot == "timescale"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPTimescales(dirName, figureName, fixed=fixed, which=which)

    elif(whichPlot == "tauvv"):
        plotVelocityTimescale(dirName)

    elif(whichPlot == "inout"):
        plotDampingInOut(dirName)

    elif(whichPlot == "velcorrf0"):
        figureName = sys.argv[3]
        scaled = sys.argv[4]
        fixed = sys.argv[5]
        which = sys.argv[6]
        iod = sys.argv[7]
        plotSPVelCorrVSDrf0(dirName, figureName, scaled, fixed, which, iod)

    elif(whichPlot == "dircorrf0"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        iod = sys.argv[6]
        plotSPDirCorrVSDrf0(dirName, figureName, fixed, which, iod)

    elif(whichPlot == "paircorrf0"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPPairCorrVSDrf0(dirName, figureName, fixed, which)

    elif(whichPlot == "spacevelf0"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        iod = sys.argv[6]
        plotSPVelSpaceCorrVSDrf0(dirName, figureName, fixed, which, iod)

    elif(whichPlot == "colpers"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPCollisionPersistence(dirName, figureName, fixed, which)

    elif(whichPlot == "spacevel"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPVelSpaceCorr(dirName, figureName, fixed, which)

    elif(whichPlot == "timevel"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        fit = sys.argv[6]
        plotSPVelTimeCorr(dirName, figureName, fixed, which, fit)

    elif(whichPlot == "velphi"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPVelPhiPDF(dirName, figureName, fixed, which)

    elif(whichPlot == "velpdf"):
        figureName = sys.argv[3]
        plotSPVelPDF(dirName, figureName)

    elif(whichPlot == "pressurepdf"):
        figureName = sys.argv[3]
        plotSPPressurePDF(dirName, figureName)

    elif(whichPlot == "fitphi"):
        figureName = sys.argv[3]
        numBins = sys.argv[4]
        fitPhiPDF(dirName, figureName, numBins)

    elif(whichPlot == "fitphi2"):
        figureName = sys.argv[3]
        numBins = sys.argv[4]
        fitPhiPDF2(dirName, figureName, numBins)

    elif(whichPlot == "clusterflu"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterFluctuations(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterphi"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterDensity(dirName, figureName, fixed, which)

    elif(whichPlot == "shapetime"):
        figureName = sys.argv[3]
        plotClusterShapeVSTime(dirName, figureName)

    elif(whichPlot == "clustershape"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterShape(dirName, figureName, fixed, which)

    elif(whichPlot == "clustermixing"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterMixingTime(dirName, figureName, fixed, which)

    elif(whichPlot == "gammatime"):
        figureName = sys.argv[3]
        plotSPClusterSurfaceTensionVSTime(dirName, figureName)

    elif(whichPlot == "clustergamma"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterSurfaceTension(dirName, figureName, fixed, which)

    elif(whichPlot == "pprofile"):
        figureName = sys.argv[3]
        shift = int(sys.argv[4])
        which = sys.argv[5]
        plotSPPressureProfile(dirName, figureName, shift, which)

    elif(whichPlot == "dprofile"):
        figureName = sys.argv[3]
        shift = int(sys.argv[4])
        which = sys.argv[5]
        plotSPDropletProfile(dirName, figureName, shift, which)

    elif(whichPlot == "mixing"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterMixing(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterl"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterLengthscale(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterpaircorr"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterPairCorr(dirName, figureName, fixed, which)

    elif(whichPlot == "numphiflu"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPNumberDensityFluctuations(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterpdf"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        numBins = int(sys.argv[6])
        plotSPClusterDistribution(dirName, figureName, fixed, which, numBins)

    elif(whichPlot == "clustertime"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterSizeVSTime(dirName, figureName, fixed, which)

    elif(whichPlot == "forcevel"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        plotSPForceVelMagnitude(dirName, figureName, fixed)

    elif(whichPlot == "tradeoff"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterTradeoff(dirName, figureName, which)

    elif(whichPlot == "dropletp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPDropletPressure(dirName, figureName, fixed, which)

    elif(whichPlot == "totalp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPTotalPressure(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        inter = sys.argv[5]
        which = sys.argv[6]
        plotSPClusterPressure(dirName, figureName, fixed, inter, which)

    elif(whichPlot == "deltapphi"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPDeltaPVSPhi(dirName, figureName, which)

    elif(whichPlot == "deltapnum"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPDeltaPVSSystemSize(dirName, figureName, which)

    elif(whichPlot == "clustersize"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterSystemSize(dirName, figureName, which)

    elif(whichPlot == "phiwindow"):
        figureName = sys.argv[3]
        weight = sys.argv[4]
        plotSPDensityVSWindowSize(dirName, figureName, weight)

    elif(whichPlot == "localphi"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPLocalDensity(dirName, figureName, fixed, which)

    elif(whichPlot == "comparelocalphi"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPCompareLocalDensity(dirName, figureName, which)

    elif(whichPlot == "dynsize"):
        figureName = sys.argv[3]
        plotSPDynamicsVSSystemSize(dirName, figureName)

    elif(whichPlot == "bounds"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPMIPSBoundsVSSystemSize(dirName, figureName, which)

    elif(whichPlot == "droptension"):
        figureName = sys.argv[3]
        plotSPDropletTensionVSSystemSize(dirName, figureName)

    elif(whichPlot == "activetension"):
        figureName = sys.argv[3]
        plotSPClusterTensionVSSystemSize(dirName, figureName)

########################### check and plot compression #########################
    elif(whichPlot == "comp"):
        figureName = sys.argv[3]
        plotSPCompression(dirName, figureName)

    elif(whichPlot == "jam"):
        figureName = sys.argv[3]
        plotSPJamming(dirName, figureName)

    elif(whichPlot == "hexcomp"):
        figureName = sys.argv[3]
        plotSPHOPCompression(dirName, figureName)

    elif(whichPlot == "comppsi6p2"):
        figureName = sys.argv[3]
        plotSPPSI6P2Compression(dirName, figureName)

    elif(whichPlot == "compset"):
        figureName = sys.argv[3]
        plotCompressionSet(dirName, figureName)

    elif(whichPlot == "hop"):
        figureName = sys.argv[3]
        plotSPHOPDynamics(dirName, figureName)

    elif(whichPlot == "psi6p2"):
        figureName = sys.argv[3]
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        plotSPPSI6P2Dynamics(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "hopphi"):
        figureName = sys.argv[3]
        plotSPHOPVSphi(dirName, figureName)

    elif(whichPlot == "pslope"):
        figureName = sys.argv[3]
        plotDeltaEvsDeltaV(dirName, figureName)

################################# plot dynamics ################################
    elif(whichPlot == "pdyn"):
        figureName = sys.argv[3]
        plotSPDynamics(dirName, figureName)

    elif(whichPlot == "pdyndrf0"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        q = sys.argv[5]
        plotSPDynamicsVSActivity(dirName, sampleName, figureName, q)

    elif(whichPlot == "pdyntemp"):
        figureName = sys.argv[3]
        q = sys.argv[4]
        plotSPDynamicsVSTemp(dirName, figureName, q)

    elif(whichPlot == "psetemp"):
        figureName = sys.argv[3]
        q = sys.argv[4]
        indexDr = int(sys.argv[5])
        indexf0 = int(sys.argv[6])
        plotSPSEvsTemp(dirName, figureName, q, indexDr, indexf0)

    elif(whichPlot == "pvelcorrdrf0"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        plotSPVelSpaceCorrVSActivity(dirName, sampleName, figureName)

    elif(whichPlot == "pcomparedyn"):
        dirName1 = dirName + sys.argv[3]
        dirName2 = dirName + sys.argv[4]
        figureName = sys.argv[5]
        q = sys.argv[6]
        compareSPDynamicsVSTemp(dirName1, dirName2, figureName, q)

    elif(whichPlot == "pdynattract"):
        figureName = sys.argv[3]
        q = sys.argv[4]
        plotSPDynamicsVSAttraction(dirName, figureName, q)

    elif(whichPlot == "pdynphi"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        plotSPDynamicsVSPhi(dirName, sampleName, figureName)

    elif(whichPlot == "ptauactivity"):
        figureName = sys.argv[3]
        plotSPTauVSActivity(dirName, figureName)

    elif(whichPlot == "ptautemp"):
        figureName = sys.argv[3]
        plotSPTauVSTemp(dirName, figureName)

    elif(whichPlot == "ptauphi"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        plotSPTauVSPhi(dirName, sampleName, figureName)

    elif(whichPlot == "pdynq"):
        figureName = sys.argv[3]
        plotSPDynamicsVSQ(dirName, figureName)

    elif(whichPlot == "pbetadrf0"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        start = int(sys.argv[5])
        plotSPBetaVSActivity(dirName, sampleName, figureName, start)

    elif(whichPlot == "pdynqdrf0"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        index = int(sys.argv[5])
        fixed = sys.argv[6]
        compareSPDynamicsVSQ(dirName, sampleName, figureName, index, fixed)

    elif(whichPlot == "pcorrdrf0"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        start = int(sys.argv[5])
        end = int(sys.argv[6])
        index = int(sys.argv[7])
        fixed = sys.argv[8]
        plotSPPairCorrVSActivity(dirName, sampleName, figureName, start, end, index, fixed)

    elif(whichPlot == "pveldrf0"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        index = int(sys.argv[5])
        fixed = sys.argv[6]
        plotSPVelCorrVSActivity(dirName, sampleName, figureName, index, fixed)

    elif(whichPlot == "pvelq"):
        figureName = sys.argv[3]
        plotSPVelCorrVSQ(dirName, figureName)

############################## plot dynamics FDT ###############################
    elif(whichPlot == "pscale"):
        sampleName = sys.argv[3]
        figureName = sys.argv[4]
        plotSPEnergyScale(dirName, sampleName, figureName)

    elif(whichPlot == "pvelmass"):
        firstIndex = int(sys.argv[3])
        figureName = sys.argv[4]
        plotSPVelPDFVSMass(dirName, firstIndex, figureName)

    elif(whichPlot == "pdensityvstime"):
        sampleName = sys.argv[3]
        numBins = int(sys.argv[4])
        figureName = sys.argv[5]
        plotSPDensityVarVSTime(dirName, sampleName, numBins, figureName)

    elif(whichPlot == "pdensitypdf"):
        sampleName = sys.argv[3]
        numBins = sys.argv[4]
        figureName = sys.argv[5]
        plotSPDensityPDF(dirName, sampleName, numBins, figureName)

    elif(whichPlot == "pdensityvsactivity"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        plotSPLocalDensityPDFvsActivity(dirName, numBins, figureName)

    elif(whichPlot == "pdensityvstemp"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        plotSPLocalDensityPDFvsTemp(dirName, numBins, figureName)

    elif(whichPlot == "pfdtsus"):
        figureName = sys.argv[3]
        Dr = float(sys.argv[4])
        driving = float(sys.argv[5])
        plotSPFDTSusceptibility(dirName, figureName, Dr, driving)

    elif(whichPlot == "pfdtdata"):
        firstIndex = int(sys.argv[3])
        mass = float(sys.argv[4])
        figureName = sys.argv[5]
        plotSPFDTdata(dirName, firstIndex, mass, figureName)

    elif(whichPlot == "pfdttemp"):
        figureName = sys.argv[3]
        q = sys.argv[4]
        index = int(sys.argv[5])
        plotSPFDTvsTemp(dirName, figureName, q, index)

    elif(whichPlot == "heatmap"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
        plotTempDensityHeatMap(dirName, numBins)

    elif(whichPlot == "phased"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        which = sys.argv[5]
        log = sys.argv[6]
        plotSPPhaseDiagram(dirName, numBins, figureName, which, log)

    elif(whichPlot == "phasedeltap"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        which = sys.argv[5]
        plotSPPhaseDiagramDeltaPressure(dirName, numBins, figureName, which)

    else:
        print("Please specify the type of plot you want")
