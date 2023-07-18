'''
Created by Francesco
12 October 2021
'''
#functions for soft particle packing visualization
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
import utils
import visuals
import spCorrelation as spCorr

def curvePL(x, a, b):
    return (a*x)**b

def curveCvv(x, a, b):
    return a * np.exp(-b*x)

def curveCvvOsc(x, a, b, c, d):
    return a * np.exp(-b*x) * np.cos(c*x + d)

def curveCvvf0(x, a, b, c, d, e, f):
    return a * (np.exp(-b*x) - c * np.exp(-d*x)) * np.cos(e*x + f)

def curveCumSum(x, a, b, c):
    return 1 - c*np.exp(-(x*a)**b)

def curveCvv(x, a, b, c):
    return c*np.exp(-(x*a))/x**b

def curve4Poly(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def getDirLabelColorMarker(dirName, sampleName, index, fixed):
    DrList = np.array(["1", "1e-01", "1e-02"])
    f0List = np.array(["1", "40", "80"])
    dirList = []
    labelList = []
    dirList.append(dirName + os.sep + "langevin/T" + sampleName)
    labelList.append("thermal")
    if(fixed == "f0"):
        for i in range(DrList.shape[0]):
            dirList.append(dirName + os.sep + "active-langevin/Dr" + DrList[i] + "-f0" + f0List[index] + "/T" + sampleName)
            labelList.append("$D_r =$" + DrList[i] + "$, f_0=$" + f0List[index])
            if(index==0):
                color='r'
            elif(index==1):
                color='g'
            else:
                color='b'
        colorList = ['k', color, color, color]
        markerList = ['o', 'v', 's', 'd']
    elif(fixed == "Dr"):
        for i in range(f0List.shape[0]):
            dirList.append(dirName + os.sep + "active-langevin/Dr" + DrList[index] + "-f0" + f0List[i] + "/T" + sampleName)
            labelList.append("$D_r =$" + DrList[index] + "$, f_0=$" + f0List[i])
            if(index==0):
                marker='v'
            elif(index==1):
                marker='s'
            else:
                marker='d'
        colorList = ['k', 'r', 'g', 'b']
        markerList = ['o', marker, marker, marker]
    else:
        print("specify which parameter to be kept fixed")
        colorList = []
        markerList = []
    dirList = np.array(dirList, dtype=str)
    return dirList, labelList, colorList, markerList

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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
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
    figureName = "/home/francesco/Pictures/soft/penergy-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## nve and langevin comparison #########################
def compareEnergy(dirName1, dirName2, figureName):
    fig, ax = plt.subplots(1, 2, sharey = True, figsize = (12, 5), dpi = 120)
    # first sample
    numParticles = int(utils.readFromParams(dirName1, "numParticles"))
    energy = np.loadtxt(dirName1 + os.sep + "energy.dat")
    ax[0].plot(energy[:,0], energy[:,2]/numParticles, linewidth=1.5, color='k')
    ax[0].plot(energy[:,0], energy[:,3]/numParticles, linewidth=1.5, color='r', linestyle='--')
    ax[0].plot(energy[:,0], (energy[:,2] + energy[:,3])/numParticles, linewidth=1.5, color='b', linestyle='dotted')
    # second sample
    numParticles = int(utils.readFromParams(dirName2, "numParticles"))
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
    figureName = "/home/francesco/Pictures/soft/pCompareEnergy-" + figureName
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
        num[d] = int(utils.readFromParams(dirSample, "numParticles"))
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
    figureName = "/home/francesco/Pictures/soft/pEnergyVSsystemSize-" + figureName
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
    figureName = "/home/francesco/Pictures/soft/pPressure-" + figureName
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
    figureName = "/home/francesco/Pictures/soft/pComparePressure-" + figureName
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
    figureName = "/home/francesco/Pictures/soft/pAvPressure-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## nve and langevin comparison #########################
def plotParticleForces(dirName, index0, index1, index2, dim):
    dirList, timeList = utils.getOrderedDirectories(dirName)
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
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
    figureName = "/home/francesco/Pictures/soft/pactiveEnergy-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

######################## functions for fourier analysis ########################
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
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pcompare-" + fileName + "Fourier-" + figureName
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
            timeStep = utils.readFromParams(dirSample, "dt")
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
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pcompareCorr-" + fileName + "Fourier-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

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
                phi.append(utils.readFromParams(dirSample, "phi"))
            elif(fixed == "phi"):
                Temp.append(np.mean(np.loadtxt(dirSample + "/energy.dat")[:,4]))
            elif(fixed == "f0"):
                if(d < dirList.shape[0]-1):
                    taup.append(1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma')))
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
    figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pcoll-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6.5, 5), dpi = 120)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$\\tau_c$", fontsize=18)
    if(fixed == "temp"):
        #ax.set_ylim(0.0013, 0.21)
        ax.plot(phi, tauc, color='b', marker='s', markersize=8, fillstyle='none', lw=1)
        ax.set_xlabel("$Density,$ $\\varphi$", fontsize=18)
        np.savetxt(dirName + "tauc-T" + which + "-vsPhi-" + dyn + ".dat", np.column_stack((phi, tauc)))
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ptauc-vsPhi-" + figureName
    elif(fixed == "phi"):
        ax.set_ylim(0.0085, 0.22)
        ax.loglog(Temp, tauc, color='b', marker='s', markersize=8, fillstyle='none', lw=1)
        ax.set_xlabel("$Temperature,$ $T$", fontsize=18)
        np.savetxt(dirName + "thermal" + which + "/tauc-vsT.dat", np.column_stack((Temp, tauc)))
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ptauc-vsTemp-" + figureName
    elif(fixed == "f0"):
        ax.loglog(Dr, tauc, color='b', marker='s', markersize=8, fillstyle='none', lw=1)
        ax.loglog(Dr, taup, color='k', marker='v', markersize=8, fillstyle='none', lw=1)
        ax.set_xlabel("$Rotational$ $diffusion,$ $D_r$", fontsize=18)
        ax.set_ylabel("$Timescales$", fontsize=18)
        ax.legend(("$Collision$ $time$ $\\tau_c$", "$Persistence$ $time$ $\\tau_p$"), fontsize=12, loc='best')
        np.savetxt(dirName + "/tauc-vsDr-f0200-iod10.dat", np.column_stack((Dr, taup, tauc)))
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ptauc-vsDr-" + figureName
    else:
        ax.set_ylim(0.0002, 0.22)
        ax.loglog(damping[1:], tauc[1:], color='b', marker='s', markersize=8, fillstyle='none', lw=1)
        ax.set_xlabel("$Damping,$ $\\beta$", fontsize=18)
        np.savetxt(dirName + "tauc-vsDamping.dat", np.column_stack((damping, tauc)))
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ptauc-vsDamping-" + figureName
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
            phi = utils.readFromParams(dirSample, "phi")
            timeStep = utils.readFromParams(dirSample, "dt")
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
    figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pcorrVel-" + figureName
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
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pdamping-" + figureName
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
            figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ptauvv-vsT-phi" + which
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
            figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ptauvv-vsPhi-T" + which
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
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/tauvv-vsDamping-vsPhi.dat"
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
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/dampingInOut-vsDamping-vsPhi.dat"
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
            phi[d] = utils.readFromParams(dirSample, "phi")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr1e-03-f0200/dynamics/"
            meanRad = np.mean(np.loadtxt(dirSample + "particleRad.dat"))
            damping[d] = np.sqrt(iod[d])/meanRad
        if(os.path.exists(dirSample + "logDirCorr.dat")):
            tauDr[d] = 1/utils.readFromDynParams(dirSample, "Dr")
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
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pCvvCollision-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pTimescales-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pCvvCollision-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pTimescales-vsPhi-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pCvvCollision-vsDamping-" + figureName
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pTimescales-vsDamping-" + figureName
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
            timeStep = utils.readFromParams(dirSample, "dt")
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
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pcorrVel-vsDr-" + figureName
    elif(fixed=="Dr"):
        label = "$f_0$"
        labels = ['$10^4$', '$10^{-1}$']
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pcorrVel-vsf0-" + figureName
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
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pFit-vsDr-" + figureName
    elif(fixed=="Dr"):
        x = f0
        xlabel = "$Propulsion,$ $f_0$"
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pFit-vsf0-" + figureName
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
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pcorrVel-vsDr-" + figureName
    elif(fixed=="Dr"):
        label = "$f_0$"
        labels = ['$10^4$', '$10^{-1}$']
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pcorrVel-vsf0-" + figureName
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
    numParticles = utils.readFromParams(dirName + os.sep + "../../", "numParticles")
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
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ppairCorr-vsDr-" + figureName
    elif(fixed=="Dr"):
        label = "$f_0$"
        labels = ['$10^4$', '$10^{-1}$']
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ppairCorr-vsf0-" + figureName
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
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pPCPeak-vsDr-" + figureName
        xlabel = "$Persistence$ $time,$ $1/D_r$"
    elif(fixed=="Dr"):
        Dr = float(Dr)/(meanRad*np.sqrt(float(iod)))
        x = f0**2/(damping*Dr)
        x = f0
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pPCPeak-vsf0-" + figureName
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
    phi = utils.readFromParams(dirName + dirList[-1], "phi")
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
                taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
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
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pSpaceVelCorr-vsDr-" + figureName
    elif(fixed=="Dr"):
        label = "$f_0$"
        labels = ['$10^4$', '$10^{-1}$']
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pSpaceVelCorr-vsf0-" + figureName
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
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pSpaceDiff-vsDr-" + figureName
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    elif(fixed=="Dr"):
        Dr = float(Dr)/(meanRad*np.sqrt(float(iod)))
        x = f0**2/(damping*Dr)
        x = f0
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pSpaceDiff-vsf0-" + figureName
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
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = utils.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = utils.readFromDynParams(dirSample, "damping")
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
    figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/ptimecorr-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    if(fit=="fit"):
        if(fixed=="iod"):
            x = phi
            xlabel = "$Density,$ $\\varphi$"
            figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pTaus-vsPhi-" + figureName + "-iod" + which
        elif(fixed=="phi"):
            x = 1/Dr
            xlabel = "$Persistence$ $time,$ $\\tau_p$"
            figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pTaus-vsDr-" + figureName + "-iod" + which
        else:
            x = damping
            xlabel = "$Damping,$ $\\gamma$"
            figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pTaus-vsDamping-" + figureName + "-Dr" + which
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
                phi.append(utils.readFromParams(dirSample, "phi"))
                p = utils.readFromParams(dirSample, "pressure")
                if(p == None):
                    p = utils.computePressure(dirSample)
                pressure.append(p)
                z = utils.readFromParams(dirSample, "numContacts")
                if(z == None):
                    z = utils.computeNumberOfContacts(dirSample)
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
            phi.append(utils.readFromParams(dirName + os.sep + dir, "phi"))
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
                phi.append(utils.readFromParams(dirName + dataSetList[i] + os.sep + dir, "phi"))
                pressure.append(utils.readFromParams(dirName + dataSetList[i] + os.sep + dir, "pressure"))
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
    stepList = utils.getStepList(numFrames, firstStep, stepFreq)
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
        phi.append(utils.readFromParams(dirName + os.sep + dataSetList[i], "phi"))
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
    timeStep = utils.readFromParams(dirName, "dt")
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
    timeStep = utils.readFromParams(dirName + "../langevin/T" + sampleName + "/dynamics/", "dt")
    diff = np.mean(thermalData[-5:,1]/(4 * thermalData[-5:,0] * timeStep))
    tau = timeStep*utils.computeTau(thermalData)
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
                timeStep = utils.readFromParams(dirSample, "dt")
                diff.append(np.mean(data[-5:,1]/(4 * data[-5:,0] * timeStep)))
                tau.append(timeStep*utils.computeTau(data))
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
    fig1.savefig("/home/francesco/Pictures/soft/pchi-Drf0-" + figureName + ".png", transparent=True, format = "png")
    fig2.tight_layout()
    fig2.savefig("/home/francesco/Pictures/soft/pdiff-Drf0-" + figureName + ".png", transparent=True, format = "png")
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
            timeStep = utils.readFromParams(dirName + "/T" + dataSetList[i] + "/dynamics/", "dt")
            #T.append(utils.readFromParams(dirName + "/T" + dataSetList[i] + "/dynamics/", "temperature"))
            energy = np.loadtxt(dirName + "/T" + dataSetList[i] + "/dynamics/energy.dat")
            if(energy[-1,3] < energy[-1,4]):
                T.append(np.mean(energy[:,3]))
            else:
                T.append(np.mean(energy[:,4]))
            print(T[-1], dataSetList[i])
            diff.append(np.mean(data[-10:,1]/(4 * data[-10:,0] * timeStep)))
            tau.append(timeStep*utils.computeTau(data))
            deltaChi.append(timeStep*utils.computeDeltaChi(data))
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
                timeStep = utils.readFromParams(dirSample, "dt")
                if(d == 0):
                    T.append(utils.readFromParams(dirSample, "temperature"))
                else:
                    energy = np.loadtxt(dirSample + "energy.dat")
                    T.append(np.mean(energy[:,4]))
                diff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
                tau.append(timeStep*utils.computeTau(data))
                deltaChi.append(timeStep*utils.computeDeltaChi(data))
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
            timeStep = utils.readFromParams(dirName + "/T10-u" + dataSetList[i] + "/dynamics/", "dt")
            diff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
            tau.append(timeStep*utils.computeTau(data))
            deltaChi.append(timeStep*utils.computeDeltaChi(data))
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
            timeStep = utils.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "dt")
            phi.append(utils.readFromParams(dirName + dataSetList[i] + dirDyn + "/T" + sampleName + "/dynamics", "phi"))
            Deff.append(data[-1,1]/(4 * data[-1,0] * timeStep))
            tau.append(timeStep*utils.computeTau(data))
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
            phi = utils.readFromParams(dirName + dataSetList[i], "phi")
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
            timeStep = utils.readFromParams(dirName, "dt")
            legendlabel = "$q=2\\pi/($" + qList[i] + "$\\times d)$"
            plotSPCorr(ax, data[1:,0]*timeStep, data[1:,2], "$ISF(\\Delta t)$", color = colorList((qList.shape[0]-i)/qList.shape[0]), legendLabel = legendlabel)
            tau.append(timeStep*utils.computeTau(data))
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
                    timeStep = utils.readFromParams(dirSample, "dt")
                    tau.append(timeStep*utils.computeTau(data, 2))
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
    utils.plotErrorBar(ax, Pe, beta, std, "$Pe$", "$\\beta$", logx = True, logy = False)
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
    dirList, labelList, colorList, markerList = getDirLabelColorMarker(dirName, sampleName, index, fixed)
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
                timeStep = utils.readFromParams(dirList[d], "dt")
                tau.append(timeStep*utils.computeTau(data, 2))
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
    dirList, labelList, colorList, markerList = getDirLabelColorMarker(dirName, sampleName, index, fixed)
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
    dirList, labelList, colorList, markerList = getDirLabelColorMarker(dirName, sampleName, index, fixed)
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
            timeStep = utils.readFromParams(dirName, "dt")
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

def plotSPFDTSusceptibility(dirName, figureName, Dr, driving):
    tmeasure = 100
    fextStr = np.array(["2", "3", "4"])
    fext = fextStr.astype(float)
    mu = np.zeros((fextStr.shape[0],2))
    T = np.zeros((fextStr.shape[0],2))
    #fig0, ax0 = plt.subplots(dpi = 120)
    fig, ax = plt.subplots(1, 2, figsize = (12.5, 5), dpi = 120)
    corr = np.loadtxt(dirName + os.sep + "dynamics/corr-log-q1.dat")
    timeStep = utils.readFromParams(dirName + os.sep + "dynamics/", "dt")
    #plotSPCorr(ax0, corr[1:,0]*timeStep, corr[1:,1]/(corr[1:,0]*timeStep), "$MSD(\\Delta t) / \\Delta t$", color = 'k', logy = True)
    timeStep = utils.readFromParams(dirName + os.sep + "dynamics", "dt")
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
                timeStep = utils.readFromParams(dirSample + os.sep + "../dynamics", "dt")
                diff.append(np.mean(data[-10:,1]/(4 * data[-10:,0] * timeStep)))
                tau.append(timeStep*utils.computeTau(data))
                deltaChi.append(utils.computeDeltaChi(data))
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
                    timeStep = utils.readFromParams(dirSample + "dynamics", "dt")
                    diff.append(np.mean(data[-5:,1]/(4 * data[-5:,0] * timeStep)))
                    tau.append(timeStep*utils.computeTau(data))
                    deltaChi.append(utils.computeDeltaChi(data))
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
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
    utils.computeLocalTempGrid(pPos, pVel, xbin, ybin, localTemp)
    utils.computeLocalAreaGrid(pPos, pRad, xbin, ybin, localArea)
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

    elif(whichPlot == "collision"):
        figureName = sys.argv[3]
        scaled = sys.argv[4]
        dyn = sys.argv[5]
        fixed = sys.argv[6]
        which = sys.argv[7]
        plotSPCollision(dirName, figureName, scaled, dyn, fixed, which)

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

    else:
        print("Please specify the type of plot you want")
