'''
Created by Francesco
7 November 2024
'''
#functions for soft particle packing visualization
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import CloughTocher2DInterpolator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import sys
import os
import utils

#######################################################################
########################## READING FUNCTIONS ##########################
#######################################################################

def readFromParams(dirName, paramName):
    name = None
    with open(dirName + os.sep + "params.dat") as file:
        for line in file:
            name, scalarString = line.strip().split("\t")
            if(name == paramName):
                return float(scalarString)
    if(name == None):
        print("The variable", paramName, "is not saved in this file")
        return None
    
    
def readFromDynParams(dirName, paramName):
    with open(dirName + os.sep + "dynParams.dat") as file:
        for line in file:
            name, scalarString = line.strip().split("\t")
            if(name == paramName):
                return float(scalarString)
            
            
def readFromWallParams(dirName, paramName):
    with open(dirName + os.sep + "wallParams.dat") as file:
        for line in file:
            name, scalarString = line.strip().split("\t")
            if(name == paramName):
                return float(scalarString)
            

def getIndexYlabel(which):
    if(which == "epot"):
        index = 2
        ylabel = "$\\frac{U}{N}$"
    elif(which == "ekin"):
        index = 3
        ylabel = "$\\frac{K}{N}$"
    elif(which == "prad"):
        index = 4
        ylabel = "$P_r$"
    elif(which == "pphi"):
        index = 5
        ylabel = "$P_\\phi$"
    elif(which == "pos"):
        index = 6
        ylabel = "$| \\phi_r|$"
    elif(which == "vel"):
        index = 7
        ylabel = "$| \\phi_v|$"
    elif(which == "velpos"):
        index = 8
        ylabel = "$| \\alpha|$"
    elif(which == "corr"):
        index = -3
        ylabel = "$C_{vv}$"
    elif(which == "moi"):
        index = -2
        ylabel = "$\\tilde{I}$"
    else:
        index = -1
        ylabel = "$\\tilde{L}$"
    return index, ylabel


########################################################################
##################### GENERIC PLOTTING FUNCTIONS #######################
########################################################################

def plotEnergyFile(dirName, figureName, which='corr'):
    if(os.path.exists(dirName + "/energy.dat")):
        energy = np.loadtxt(dirName + os.sep + "energy.dat")
        print("potential energy:", np.mean(energy[:,2]), "+-", np.std(energy[:,2]))
        print("temperature:", np.mean(energy[:,3]), "+-", np.std(energy[:,3]))
        print("velocity alignment:", np.mean(energy[:,-2]), "+-", np.std(energy[:,-2]), "relative error:", np.std(energy[:,-2])/np.mean(energy[:,-2]))
        fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
        index, ylabel = getIndexYlabel(which)
        ax.plot(energy[::2,0], energy[::2,index], linewidth=1.2, color='k')
        ax.tick_params(axis='both', labelsize=14)
        #ax.set_ylim(0.722, 1.022)
        ax.set_xlabel("$Simulation$ $step$", fontsize=16)
        if index == 2 or index == 3: ax.set_ylabel(ylabel, fontsize=24, rotation='horizontal', labelpad=15)
        else: ax.set_ylabel(ylabel, fontsize=16, rotation='horizontal', labelpad=15)
        plt.tight_layout()
        figureName = "/home/francesco/Pictures/soft/align-" + figureName
        fig.savefig(figureName + ".png", transparent=True, format = "png")
        plt.show()
    else:
        print("no energy.dat file was found in", dirName)


def plotAlignmentVSInteraction(dirName, figureName, which, taup="0", dynamics="/"):
    dirList = np.array(["1e-03", "1e-02", "3e-02", "1e-01", "3e-01", "4e-01", "5e-01", "7e-01", "1", "1.5", "2", "3", "5", "7",
                        "1e01", "3e01", "1e02", "3e02", "1e03", "1e04", "1e05", "1e06", "1e07"])
    jvic = np.zeros(dirList.shape[0])
    align = np.zeros((dirList.shape[0], 2))
    index, ylabel = getIndexYlabel(which)
    fig, ax = plt.subplots(figsize=(5,4.5), dpi = 120)
    for d in range(dirList.shape[0]):
        dirSample = dirName + "j" + dirList[d] + "-tp" + taup + dynamics
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            if(index == -1):
                align[d,0] = np.mean(np.abs(data[:,index]))
                align[d,1] = np.std(np.abs(data[:,index]))
            else:
                align[d,0] = np.mean(data[:,index])
                align[d,1] = np.std(data[:,index])
            jvic[d] = readFromDynParams(dirSample, "Jvicsek")
            #print(dirList[d], 1/jvic[d])
            if(d == 0 and index == -2):
                noisetime = readFromDynParams(dirSample, "taup")
                ax.plot(np.ones(100)*noisetime, np.linspace(-0.3,1.3,100), ls='dotted', color='k', lw=0.8)
    ax.errorbar(1/jvic[jvic!=0], align[jvic!=0,0], align[jvic!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
    ax.set_xscale('log')
    if(index == -1):
        ax.set_yscale('log')
    if(index == 5):
        ax.set_ylim(-0.057, 1.112)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/alignVSinter-" + which + "-tp" + taup + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()


def compareAlignmentVSInteraction(dirName, figureName, which, dynamics="/"):
    noiseList = np.array(["1e-02", "1e-01", "1", "1e01", "1e02", "0"])
    labelList = np.array(["$\\tau_p = 10^{-2}$", "$\\tau_p = 10^{-1}$", "$\\tau_p = 10^0$", "$\\tau_p = 10^1$", "$\\tau_p = 10^2$", "$\\tau_p \\rightarrow \\infty$"])
    dirList = np.array(["1e-03", "1e-02", "3e-02", "1e-01", "3e-01", "4e-01", "5e-01", "7e-01", "1", "1.5", "2", "3", "5", "7",
                        "1e01", "3e01", "1e02", "3e02", "1e03", "1e04", "1e05", "1e06", "1e07"])
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    index, ylabel = getIndexYlabel(which)
    colorList = [[1,0.5,0], 'r', 'g', 'c', 'b', 'k']
    markerList = ['v', 'd', 's', 'D', '^', 'o']
    for i in range(noiseList.shape[0]):
        jvic = np.zeros(dirList.shape[0])
        align = np.zeros((dirList.shape[0], 2))
        noisetime = 0
        for d in range(dirList.shape[0]):
            dirSample = dirName + "j" + dirList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                if(index == -1):
                    align[d,0] = np.mean(np.abs(data[:,index]))
                    align[d,1] = np.std(np.abs(data[:,index]))
                else:
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])
                jvic[d] = readFromDynParams(dirSample, "Jvicsek")
                if(noisetime == 0 and index == -2):
                    noisetime = readFromDynParams(dirSample, "taup")
                    plt.plot(np.ones(100)*noisetime, np.linspace(-0.3,1.3,100), ls='dotted', color=colorList[i], lw=0.8)
        plt.errorbar(1/jvic[jvic!=0], align[jvic!=0,0], align[jvic!=0,1], color=colorList[i], marker=markerList[i], markersize=8, capsize=3, fillstyle='none', lw=1, label=labelList[i])
    ax.legend(fontsize=11, loc='best')
    ax.set_xscale('log')
    if(index == -1):
        ax.set_yscale('log')
    if(index == 5):
        ax.set_ylim(-0.057, 1.112)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/compare-" + which + "VSinter-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()


def plotAlignmentVSNoise(dirName, figureName, which, jvic="1e02", dynamics="/"):
    dirList = np.array(["1e-04", "2e-04", "3e-04", "5e-04", "1e-03", "2e-03", "3e-03", "5e-03", "1e-02", "1.5e-02", "2e-02", "2.5e-02", "3e-02", "4e-02",
                        "5e-02", "7e-02", "1e-01", "1.5e-01", "2e-01", "3e-01", "5e-01", "1", "2", "3", "5", "1e01", "2e01", "3e01", "5e01", "1e02",
                        "2e02", "3e02", "5e02", "1e03", "2e03", "3e03", "5e03", "1e04", "2e04", "3e04", "5e04", "1e05", "2e05", "3e05", "5e05",
                        "1e06", "1e07", "1e08", "1e09"])
    noise = np.zeros(dirList.shape[0])
    align = np.zeros((dirList.shape[0], 2))
    index, ylabel = getIndexYlabel(which)
    fig, ax = plt.subplots(figsize=(5,4.5), dpi = 120)
    for d in range(dirList.shape[0]):
        if(jvic == "active"):
            dirSample = dirName + "tp" + dirList[d] + dynamics
        else:
            dirSample = dirName + "j" + jvic + "-tp" + dirList[d] + dynamics
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            if(index == -1):
                align[d,0] = np.mean(np.abs(data[:,index]))
                align[d,1] = np.std(np.abs(data[:,index]))
            else:
                align[d,0] = np.mean(data[:,index])
                align[d,1] = np.std(data[:,index])
            noise[d] = readFromDynParams(dirSample, "taup")
            if(d == 0 and index == -2):
                aligntime = 1/readFromDynParams(dirSample, "Jvicsek")
                plt.plot(np.ones(100)*aligntime, np.linspace(-0.3,1.3,100), ls='dotted', color='k', lw=0.8)
    plt.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
    ax.set_xscale('log')
    if(index == -1):
        ax.set_yscale('log')
    if(index == 5):
        ax.set_ylim(-0.057, 1.112)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2\\Delta t/\\tau_p}$", fontsize=16)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    if(jvic == "active"):
        figureName = "/home/francesco/Pictures/soft/alignVSnoise-" + which + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/alignVSnoise-" + which + "j" + jvic + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()


def compareAlignmentVSNoise(dirName, figureName, which, dynamics="/"):
    interList = np.array(["1e-01", "4e-01", "1", "7", "3e01", "1e03"])
    #labelList = np.array(["$J = 0.1$", "$J = 0.4$", "$J = 1$", "$J = 7$", "$J = 30$", "$J = 10^3$"])
    labelList = np.array(["$\\tau_K = 7.1 \\times 10^1$", "$\\tau_K = 1.8 \\times 10^1$", "$\\tau_K = 7.1 \\times 10^0$", "$\\tau_K = 10^0$", "$\\tau_K = 2.3 \\times 10^{-1}$", "$\\tau_K = 7.1 \\times 10^{-3}$"])
    colorList = [[1,0.5,0], 'r', 'g', 'c', 'b', 'k']
    markerList = ['v', 'd', 's', 'D', '^', 'o']
    dirList = np.array(["1e-04", "2e-04", "3e-04", "5e-04", "1e-03", "2e-03", "3e-03", "5e-03", "1e-02", "1.5e-02", "2e-02", "2.5e-02", "3e-02", "4e-02",
                        "5e-02", "7e-02", "1e-01", "1.5e-01", "2e-01", "3e-01", "5e-01", "1", "2", "3", "5", "1e01", "2e01", "3e01", "5e01", "1e02",
                        "2e02", "3e02", "5e02", "1e03", "2e03", "3e03", "5e03", "1e04", "2e04", "3e04", "5e04", "1e05", "2e05", "3e05", "5e05",
                        "1e06", "1e07", "1e08", "1e09"])
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    index, ylabel = getIndexYlabel(which)
    for t in range(interList.shape[0]):
        noise = np.zeros(dirList.shape[0])
        align = np.zeros((dirList.shape[0], 2))
        aligntime = 0
        for d in range(dirList.shape[0]):
            dirSample = dirName + "j" + interList[t] + "-tp" + dirList[d] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                if(index == -1):
                    align[d,0] = np.mean(np.abs(data[:,index]))
                    align[d,1] = np.std(np.abs(data[:,index]))
                else:
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])
                noise[d] = readFromDynParams(dirSample, "taup")
                if(aligntime == 0 and index == -2):
                    aligntime = 1/readFromDynParams(dirSample, "Jvicsek")
                    #print(interList[t], aligntime)
                    plt.plot(np.ones(100)*aligntime, np.linspace(-0.3,1.3,100), ls='dotted', color=colorList[t], lw=0.8)
                    #labelList[t] = "$\\tau_K =$" + str(np.format_float_scientific(aligntime,1))
        plt.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color=colorList[t], marker=markerList[t], markersize=8, label=labelList[t], capsize=3, fillstyle='none', lw=1)
    ax.legend(fontsize=12, loc='best')
    ax.set_xscale('log')
    if(index == -1):
        ax.set_yscale('log')
    if(index == 5):
        ax.set_ylim(-0.057, 1.112)
    if(index == 4):
        ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=16, rotation='horizontal')
    else:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2\\Delta t/\\tau_p}$", fontsize=16)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/compare-" + which + "VSnoise-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################################################################
########################### CLUSTER ANALYSIS ###########################
########################################################################

def computeMaxClusterKuramoto(dirSample, eps=3, maxCluster=10, minFraction=0.3):
    dirList, timeList = utils.getOrderedDirectories(dirSample)
    numParticles = int(readFromParams(dirSample, "numParticles"))
    eps *= 2 * np.mean(np.loadtxt(dirSample + "particleRad.dat"))
    clusterPhi_r = np.zeros((dirList.shape[0], 2))
    numCluster = np.empty(0)
    for d in range(dirList.shape[0]):
        dirFrame = dirSample + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        angles = np.arctan2(pos[:,1], pos[:,0])
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        if uniqueLabels.shape[0] < maxCluster:
            # Do not check for largest clusters if there are many clusters
            numCluster = np.append(numCluster, uniqueLabels.shape[0])
            # if there are few clusters, restrict computation of Kuramoto order parameter for the largest cluster
            maxLabel = -1
            numMaxLabel = 0
            for label in uniqueLabels:
                numLabel = labels[labels==label].shape[0]
                #print("label", label, "num particles in cluster", numLabel)
                if numLabel > numMaxLabel:
                    maxLabel = label
                    numMaxLabel = numLabel
            fraction = labels[labels==maxLabel].shape[0] / numParticles
            if fraction > minFraction and maxLabel != -1:
                #print("largest cluster:", maxLabel, numMaxLabel, "particles", fraction*numParticles)
                clusterIndices = np.where(labels == maxLabel)[0]
                angles = angles[clusterIndices]
        numParticles = angles.shape[0]
        # compute Kuramoto order parameter for the cluster
        sumReal = 0
        sumImag = 0
        for i in range(numParticles):
            sumReal += np.cos(angles[i])
            sumImag += np.sin(angles[i])
        phi_r = np.sqrt(sumReal**2 + sumImag**2) / numParticles
        clusterPhi_r[d,0] = timeList[d]
        clusterPhi_r[d,1] = phi_r
    if numCluster.shape[0] > 0:
        print("Average number of clusters:", np.mean(numCluster), "number of used particles:", numParticles)
    np.savetxt(dirSample + "/clusterKuramoto.dat", clusterPhi_r)


def computeClusterKuramoto(dirSample, eps=1.5):
    dirList, timeList = utils.getOrderedDirectories(dirSample)
    numParticles = int(readFromParams(dirSample, "numParticles"))
    eps *= 2 * np.mean(np.loadtxt(dirSample + "particleRad.dat"))
    clusterPhi_r = np.zeros((dirList.shape[0], 2))
    numLabels = np.empty(0)
    for d in range(dirList.shape[0]):
        dirFrame = dirSample + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        angles = np.arctan2(pos[:,1], pos[:,0])
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        numLabels = np.append(numLabels, uniqueLabels.shape[0])
        phi_r = np.empty(0)
        for label in uniqueLabels:
            if label != -1:
                fraction = labels[labels==label].shape[0] / numParticles
                #print(dirList[d], "label", label, "num particles in cluster", fraction*numParticles)
                clusterAngles = angles[labels==label]
                numCluster = clusterAngles.shape[0]
                # compute Kuramoto order parameter for the cluster
                sumReal = 0
                sumImag = 0
                for i in range(numCluster):
                    sumReal += np.cos(clusterAngles[i])
                    sumImag += np.sin(clusterAngles[i])
                phi_r = np.append(phi_r, np.sqrt(sumReal**2 + sumImag**2) / numCluster)
        clusterPhi_r[d,0] = timeList[d]
        clusterPhi_r[d,1] = np.mean(phi_r)
        #print("Time:", timeList[d], "Num clusters:", uniqueLabels.shape[0], "Average phi_r:", clusterPhi_r[d,1])
        #print(dirFrame)
    if(numLabels.shape[0] != 0): print("Average number of clusters:", np.mean(numLabels))
    np.savetxt(dirSample + "/clusterKuramoto.dat", clusterPhi_r)


def plotClustersCOM(dirName, figureName, eps=1.5, maxCluster=12):
    fig, ax = plt.subplots(2, 1, figsize=(7,6), sharex=True,dpi = 120)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    boxRadius = np.loadtxt(dirName + "boxSize.dat")
    eps *= 2.5 * np.mean(np.loadtxt(dirName + "particleRad.dat"))
    cluster_com1 = np.empty((0,2))
    cluster_com2 = np.empty((0,2))
    times = np.array([])
    dt = float(readFromParams(dirName, "dt"))
    for d in range(dirList.shape[0]):
        dirFrame = dirName + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        # transform positions to polar coordinates
        angles = np.arctan2(pos[:,1], pos[:,0])
        radial = np.sqrt(pos[:,0]**2 + pos[:,1]**2) / boxRadius
        polarPos = np.column_stack((radial, angles))
        if d == 0:
            prevPos = polarPos
        else:
            # account for periodicity in theta
            deltaTheta = polarPos[:,1] - prevPos[:,1]
            deltaTheta = (deltaTheta + np.pi) % (2 * np.pi) - np.pi
            polarPos[:,1] = prevPos[:,1] + deltaTheta
            prevPos = polarPos
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        #print("Time:", timeList[d], "Num clusters:", uniqueLabels.shape[0])
        if uniqueLabels.shape[0] < maxCluster:
            # find largest cluster and plot its COM
            numCluster = np.empty(0)
            clusterLabel = np.empty(0)
            for label in uniqueLabels:
                numCluster = np.append(numCluster, labels[labels==label].shape[0])
                clusterLabel = np.append(clusterLabel, label)
            maxIndex = np.argmax(numCluster)
            maxLabel = clusterLabel[maxIndex]
            clusterIndices1 = np.where(labels == maxLabel)[0]
            clusterPos1 = polarPos[clusterIndices1]
            com1 = np.mean(clusterPos1, axis=0)
            cluster_com1 = np.vstack((cluster_com1, com1))
            #print("largest cluster:", maxLabel, numCluster[maxIndex], "particles", com1)
            if uniqueLabels.shape[0] > 2:
                # find second largest cluster and plot its COM
                numCluster[maxIndex] = -1
                secondMaxIndex = np.argmax(numCluster)
                secondMaxLabel = clusterLabel[secondMaxIndex]
                clusterIndices2 = np.where(labels == secondMaxLabel)[0]
                clusterPos2 = polarPos[clusterIndices2]
                com2 = np.mean(clusterPos2, axis=0)
                cluster_com2 = np.vstack((cluster_com2, com2))
                #print("second largest cluster:", secondMaxLabel, numCluster[secondMaxIndex], "particles", com2)
            times = np.append(times, timeList[d] * dt)
    ax[0].plot(times, cluster_com1[:,0], color='b', lw=1)
    ax[1].plot(times, cluster_com1[:,1], color='b', lw=1, label="$\\theta_1$")
    #ax[1].plot(times[1:], (cluster_com1[1:,1]-cluster_com1[:-1,1])/(times[1:]-times[:-1]), color='b', lw=1, label="$Cluster$ $1$")
    if cluster_com2.shape[0] == times.shape[0]: 
        ax[0].plot(times, cluster_com2[:,0], color='g', lw=1)
        ax[1].plot(times, cluster_com2[:,1], color='g', lw=1, label="$\\theta_2$")
        #ax[1].plot(times[1:], (cluster_com2[1:,1]-cluster_com2[:-1,1])/(times[1:]-times[:-1]), color='g', lw=1, label="$Cluster$ $2$")
        delta_theta = cluster_com1[:,1] - cluster_com2[:,1]
        ax[1].plot(times, delta_theta, color='k', lw=1.1, ls='dashed', label="$\\theta_1 - \\theta_2$")
    ax[1].legend(fontsize=12, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$Time,$ $t$", fontsize=16)
    ax[0].set_ylabel("$\\frac{r(t)}{R}$", fontsize=22, rotation='horizontal', labelpad=20)
    #ax[1].set_ylabel("$\\theta(t)$", fontsize=16, rotation='horizontal', labelpad=20)
    #ax[1].set_ylabel("$\\Delta \\theta(t)$", fontsize=16, rotation='horizontal', labelpad=10)
    ax[1].set_yticks([-np.pi, 0, np.pi])
    ax[1].set_yticklabels(['$-\\pi$', '$0$', '$\\pi$'])
    ax[1].set_yticks([-np.pi, np.pi, 5*np.pi, 10*np.pi, 15*np.pi, 20*np.pi])
    ax[1].set_yticklabels(['$-\\pi$', '$\\pi$', '$5\\pi$', '$10\\pi$', '$15\\pi$', '$20\\pi$'])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/soft/clustersCOM-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()


def plotClusterDensity(dirName, figureName, eps=1.5, maxCluster=5):
    fig, ax = plt.subplots(2, 1, figsize=(7,6), sharex=True,dpi = 120)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    numParticles = int(readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    eps *= 2.5 * np.mean(rad)
    rgyr = np.empty(0)
    density = np.empty(0)
    times = np.empty(0)
    theta = np.empty(0)
    dt = float(readFromParams(dirName, "dt"))
    for d in range(dirList.shape[0]):
        dirFrame = dirName + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        #print("Time:", timeList[d], "Num clusters:", uniqueLabels.shape[0])
        if uniqueLabels.shape[0] < maxCluster:
            # find largest cluster and plot its COM
            numCluster = np.empty(0)
            clusterLabel = np.empty(0)
            for label in uniqueLabels:
                numCluster = np.append(numCluster, labels[labels==label].shape[0])
                clusterLabel = np.append(clusterLabel, label)
            maxIndex = np.argmax(numCluster)
            maxLabel = clusterLabel[maxIndex]
            clusterIndices = np.where(labels == maxLabel)[0]
            clusterPos = pos[clusterIndices]
            clusterRad = rad[clusterIndices]
            com = np.mean(clusterPos, axis=0)
            clusterPos -= com
            rgyr = np.append(rgyr, np.mean(np.linalg.norm(clusterPos,axis=1)))
            density = np.append(density, np.sum(clusterRad**2) / (2 * rgyr[-1]**2)) # cluster radius is sqrt(2) * rgyr
            times = np.append(times, timeList[d] * dt)
            #print("time:", times[-1], "radius of gyration:", rgyr[-1], "density:", density[-1])
    ax[0].plot(times, rgyr, color='k', lw=1)
    ax[1].plot(times, density, color='k', lw=1, label="$Cluster$ $1$")
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$Time,$ $t$", fontsize=16)
    ax[0].set_ylabel("$Radius$ $of$ $gyration,$ $R_g$", fontsize=16)
    ax[1].set_ylabel("$Cluster$ $density,$ $\\varphi_c$", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/soft/clustePhi-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()


def computeNumClusterVSTime(dirSample, minNum=2, eps=1.5):
    dirList, timeList = utils.getOrderedDirectories(dirSample)
    rad = np.loadtxt(dirSample + "particleRad.dat")
    numParticles = rad.shape[0]
    eps *= 2 * np.mean(rad)
    numLabels = np.zeros(dirList.shape[0])
    numInCluster = np.zeros(dirList.shape[0])
    fraction = np.zeros(dirList.shape[0])
    free = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirFrame = dirSample + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        numClusters = 0 # number of clusters with more than minNum particles
        fracCluster = 0
        for label in uniqueLabels:
            if label != -1 and labels[labels==label].shape[0] > minNum:
                numClusters += 1
                fracCluster += labels[labels==label].shape[0]
        numLabels[d] = numClusters
        if numClusters > 0:
            numInCluster[d] = fracCluster / numClusters
        fraction[d] = fracCluster / numParticles
        free[d] = labels[labels==-1].shape[0] / numParticles
        #print("Time:", timeList[d], "Num clusters:", numClusters, "fraction in clusters:", fraction[d], "fraction free:", free[d])
    np.savetxt(dirSample + "/numCluster.dat", np.column_stack((timeList, numLabels, numInCluster, fraction, free)))


def compareNumClusterVSTime(dirName, figureName, versus="inter", which="num", dynamics="/", minNum=2):
    fig, ax = plt.subplots(1, 2, figsize=(10,4), dpi = 120)
    dt = float(readFromParams(dirName + "j1e03-tp1e03", "dt"))
    if versus == "inter":
        dirList = np.array(["1e-04", "3e-04", "1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "2e-01", "3e-01", "5e-01", 
                            "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
        cblabel = "$J_K$"
        xlabel = "$Alignment$ $time,$ $\\tau_K$"
    else:
        dirList = np.array(["1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
        cblabel = "$\\tau_p$"
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    colorList = cm.get_cmap('plasma')
    aligntime = np.zeros(dirList.shape[0])
    noisetime = np.zeros(dirList.shape[0])
    numCluster1 = np.zeros((dirList.shape[0],2))
    numCluster2 = np.zeros((dirList.shape[0],2))
    numCluster3 = np.zeros((dirList.shape[0],2))
    if which == "num":
        index = 1
        ylabel1 = "$N_C$"
        ylabel2 = "$\\langle N_C \\rangle$"
    elif which == "numin":
        index = 2
        ylabel1 = "$\\bar{N}_p$"
        ylabel2 = "$\\langle \\bar{N}_p \\rangle$"
    elif which == "frac":
        index = 3
        ylabel1 = "$f_C$"
        ylabel2 = "$\\langle f_C \\rangle$"
    else:
        index = -1
        ylabel1 = "$f_0$"
        ylabel2 = "$\\langle f_0 \\rangle$"
    for d in range(dirList.shape[0]):
        if versus == "inter":
            dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/" + dynamics
        else:
            dirSample = dirName + "j1e-01-tp" + dirList[d] + "/dynamics-vel/" + dynamics
        aligntime[d] = 1/readFromDynParams(dirSample, "Jvicsek")
        noisetime[d] = readFromDynParams(dirSample, "taup")
        if(os.path.exists(dirSample + "/t0/")):
            if not(os.path.exists(dirSample + "/numCluster.dat")):
                computeNumClusterVSTime(dirSample, minNum)
            clusterData = np.loadtxt(dirSample + "/numCluster.dat")
            ax[1].plot(clusterData[:,0]*dt, clusterData[:,index], linewidth=1, color=colorList(d/dirList.shape[0]), label ="$J_K=$" + dirList[d])
            numCluster1[d,0] = np.mean(clusterData[-20:,index])
            numCluster1[d,1] = np.std(clusterData[-20:,index])
            numCluster2[d,0] = np.mean(clusterData[-50:,index])
            numCluster2[d,1] = np.std(clusterData[-50:,index])
            numCluster3[d,0] = np.mean(clusterData[:,index])
            numCluster3[d,1] = np.std(clusterData[:,index])
    if which == 'num':
        upper_lim = 92
        ax[1].set_ylim(-3,upper_lim)
        ax[0].set_ylim(-3,upper_lim)
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.)
    #cax = fig.add_subplot(gs[2])
    cbar = fig.colorbar(colorBar, cax)
    cbar.set_label(cblabel, rotation='horizontal', fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=14, length=0)
    if versus == 'inter':
        cbar.set_ticks(np.linspace(0,1,4))
        cbar.set_ticklabels(['$10^{-4}$', '$10^{-2}$', '$10^2$', '$10^4$'])
        x = aligntime
    else:
        cbar.set_ticks(np.linspace(0,1,3))
        cbar.set_ticklabels(['$1$', '$10^4$', '$10^8$'])
        x = noisetime
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$Time,$ $t$", fontsize=14)
    ax[1].set_ylabel(ylabel1, fontsize=14, rotation='horizontal', labelpad=10)
    ax[0].errorbar(x[numCluster1[:,0]!=0], numCluster1[numCluster1[:,0]!=0,0], numCluster1[numCluster1[:,0]!=0,1], 
                   lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label="$t > 0.8 t_{max}$")
    ax[0].errorbar(x[numCluster2[:,0]!=0], numCluster2[numCluster2[:,0]!=0,0], numCluster2[numCluster2[:,0]!=0,1], 
                   lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label="$t > 0.5 t_{max}$", alpha=0.4)
    ax[0].errorbar(x[numCluster3[:,0]!=0], numCluster3[numCluster3[:,0]!=0,0], numCluster3[numCluster3[:,0]!=0,1], 
                   lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label="$t > 0$", alpha=0.2)
    ax[0].legend(fontsize=11, loc='best')
    ax[0].set_xscale('log')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel(xlabel, fontsize=14)
    ax[0].set_ylabel(ylabel2, fontsize=14, rotation='horizontal', labelpad=15)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/numCluster-" + figureName + "-" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################################################################
############################ PHASE DIAGRAMS ############################
########################################################################

def phaseDiagramNoiseAlignment(dirName, figureName, dynamics="/", which="vcorr", interpolate=False):
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    aligntime = np.array([])
    noisetime = np.array([])
    corr = np.array([])
    if(which == "pos"):
        cbar_label = "$\\phi_r$"
        index = 6
    elif(which == "vel"):
        cbar_label = "$\\phi_v$"
        index = 7
    elif(which == "velpos"):
        cbar_label = "$\\phi_\\alpha$"
        index = 8
    else:
        cbar_label = "$C_{vv}$"
        index = -2
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08", "0"])
    alignList = np.array(["3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    for i in range(noiseList.shape[0]):
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                corr = np.append(corr, np.mean(data[:,index]))
                aligntime = np.append(aligntime, 1/readFromDynParams(dirSample, "Jvicsek"))
                #print(noiseList[i], noiseDirList[d], aligntime[-1])
                tp = readFromDynParams(dirSample, "taup")
                if(tp == 0):
                    tp = 1e09
                noisetime = np.append(noisetime, tp)
    if(interpolate == 'interpolate'):
        # Add a small random noise to avoid numerical issues
        noisetime += np.random.uniform(0, 0.01, size=noisetime.shape)
        aligntime += np.random.uniform(0, 0.01, size=aligntime.shape)
        # Convert to log-space for interpolation
        log_noisetime = np.log10(noisetime)
        log_aligntime = np.log10(aligntime)
        # Define a log-spaced grid
        log_tp_lin = np.linspace(log_noisetime.min(), log_noisetime.max(), 100)
        log_tk_lin = np.linspace(log_aligntime.min(), log_aligntime.max(), 100)
        grid_tp, grid_tk = np.meshgrid(log_tp_lin, log_tk_lin)
        grid_corr = griddata((log_noisetime, log_aligntime), corr, (grid_tp, grid_tk), method='linear')
        # Convert grid back to linear scale for plotting
        grid_tp_lin = 10**grid_tp
        grid_tk_lin = 10**grid_tk
        contour = plt.contourf(grid_tp_lin, grid_tk_lin, grid_corr, levels=20, cmap='plasma')
        cbar = plt.colorbar(contour, ax=ax, pad=0, aspect=20)
    else:
        vmin = np.min(corr)
        if vmin < 0: vmin = 0
        vmax = np.max(corr)
        sc = plt.scatter(noisetime, aligntime, c=corr, cmap='plasma', s=200, edgecolors='k', marker='s', linewidths=0.5, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax, pad=0, aspect=20)
    cbar.set_label(cbar_label, rotation='horizontal', fontsize=22, labelpad=20)
    cbar.ax.tick_params(labelsize=14, length=0)
    min = np.min(corr)
    max = np.max(corr)
    cbar.set_ticks(np.linspace(min,max,5))
    #cbar.set_ticks(np.linspace(0,1,5))
    if which == "vcorr" or which == "velpos":
        cbar.set_ticklabels(["$0.00$", "$0.25$", "$0.50$", "$0.75$", "$1.00$"])
    # Set log scales for proper visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    ax.set_ylabel("$Alignment$ $time,$ $\\tau_K$", fontsize=22)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/phaseDiagram-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()


def phaseDiagrams3(dirName, figureName, dynamics="/", cluster=False, maxCluster=6, interpolate=False):
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(11,3), dpi = 150)
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])#, "0"])
    alignList = np.array(["3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    aligntime = np.array([])
    noisetime = np.array([])
    corr1 = np.array([])
    corr2 = np.array([])
    corr3 = np.array([])
    interCut = 0
    noiseCut = 1e03
    boxSize = np.atleast_1d(np.loadtxt(dirName + "j1e03-tp1e03/boxSize.dat"))
    maxI = boxSize**2
    f0 = float(readFromDynParams(dirName + "j1e03-tp1e03", "f0"))
    gamma = float(readFromDynParams(dirName + "j1e03-tp1e03", "damping"))
    maxL = boxSize * (f0/gamma)
    colorMap1 = 'plasma'
    colorMap2 = 'plasma'
    colorMap3 = 'plasma'
    if figureName == 'smooth':
        points = np.array(['1e-01', '3', '1e04'])
        color1 = 'darkgrey'#[0.6,0.6,0.6]
        color2 = 'darkviolet'
        color3 = [1,0.7,0]
    elif figureName == 'rough':
        points = np.array(['1e-01', '3', '1e02'])
        color1 = 'forestgreen'#[0,0.6,0.4]
        color2 = 'dodgerblue'#[0.5,0.5,1]
        color3 = [0.6,0,0.8]
    for i in range(noiseList.shape[0]):
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                aligntime = np.append(aligntime, 1/readFromDynParams(dirSample, "Jvicsek"))
                tp = readFromDynParams(dirSample, "taup")
                tauk = 1/readFromDynParams(dirSample, "Jvicsek")
                if i == 0 and alignList[d] == "1e-01":
                    interCut = tauk
                if noiseList[i] == '1e03':
                    if alignList[d] == points[0]: point1 = np.array([tp, tauk])
                    elif alignList[d] == points[1]: point2 = np.array([tp, tauk])
                    if alignList[d] == points[2]: point3 = np.array([tp, tauk])
                data = np.loadtxt(dirSample + "energy.dat")
                if cluster:
                    if(os.path.exists(dirSample + "/t0/")):
                        if not(os.path.exists(dirSample + "/clusterKuramoto!.dat")):
                            print("tp =", noiseList[i], "j =", alignList[d])
                            computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster, minFraction=0.3)
                        corr1 = np.append(corr1, np.mean(np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]))
                    else:
                        corr1 = np.append(corr1, np.mean(data[:,6]))
                else:
                    corr1 = np.append(corr1, np.mean(data[:,6]))
                corr2 = np.append(corr2, np.mean(data[:,-2] / maxI))
                corr3 = np.append(corr3, np.mean(np.abs(data[:,-1]) / maxL))
                if(tp == 0):
                    tp = 1e09
                noisetime = np.append(noisetime, tp)
    vmin = np.min(corr3)
    if vmin < 0: vmin = 0
    vmax = np.max(corr3)
    if vmax > 1: vmax = 1
    if(interpolate == 'interpolate'):
        # Convert to log-space for interpolation
        log_noisetime = np.log10(noisetime)
        log_aligntime = np.log10(aligntime)
        # Remove duplicate points
        points = np.column_stack((log_noisetime, log_aligntime))
        points_unique, idx = np.unique(points, axis=0, return_index=True)
        corr1_unique = corr1[idx]
        corr2_unique = corr2[idx]
        corr3_unique = corr3[idx]
        # Define log grid
        log_tp_lin = np.linspace(points_unique[:,0].min(), points_unique[:,0].max(), 200)
        log_tk_lin = np.linspace(points_unique[:,1].min(), points_unique[:,1].max(), 200)
        grid_tp, grid_tk = np.meshgrid(log_tp_lin, log_tk_lin)
        inter1 = CloughTocher2DInterpolator(points_unique, corr1_unique)
        grid_corr1 = inter1(grid_tp, grid_tk)
        grid_corr1 = gaussian_filter(grid_corr1, sigma=2)
        grid_corr1 = np.clip(grid_corr1, 0, 1)
        inter2 = CloughTocher2DInterpolator(points_unique, corr2_unique)
        grid_corr2 = inter2(grid_tp, grid_tk)
        grid_corr2 = gaussian_filter(grid_corr2, sigma=2)
        grid_corr2 = np.clip(grid_corr2, 0, 1)
        inter3 = CloughTocher2DInterpolator(points_unique, corr3_unique)
        grid_corr3 = inter3(grid_tp, grid_tk)
        grid_corr3 = gaussian_filter(grid_corr3, sigma=2)
        grid_corr3 = np.clip(grid_corr3, 0, 1)
        # Convert grid back to linear scale for plotting
        grid_tp_lin = 10**grid_tp
        grid_tk_lin = 10**grid_tk
        ax[0].contourf(grid_tp_lin, grid_tk_lin, grid_corr1, levels=150, cmap=colorMap1, vmin=vmin, vmax=vmax)
        ax[1].contourf(grid_tp_lin, grid_tk_lin, grid_corr2, levels=150, cmap=colorMap2, vmin=vmin, vmax=vmax)
        contour = ax[2].contourf(grid_tp_lin, grid_tk_lin, grid_corr3, levels=150, cmap=colorMap3, vmin=vmin, vmax=vmax)
        # create a floating inset for the colorbar, relative to the figure
        cax = inset_axes(ax[2], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0.0, 1, 1),  # position outside right edge
                        bbox_transform=ax[2].transAxes, borderpad=0.0)
        cbar = plt.colorbar(contour, cax=cax)
    else:
        ax[0].scatter(noisetime, aligntime, c=corr1, cmap=colorMap1, s=200, edgecolors='k', marker='s', linewidths=0.4, vmin=vmin, vmax=vmax)
        ax[1].scatter(noisetime, aligntime, c=corr2, cmap=colorMap2, s=200, edgecolors='k', marker='s', linewidths=0.4, vmin=vmin, vmax=vmax)
        sc3 = ax[2].scatter(noisetime, aligntime, c=corr3, cmap=colorMap3, s=200, edgecolors='k', marker='s', linewidths=0.4, vmin=vmin, vmax=vmax)
        # create a floating inset for the colorbar, relative to the figure
        cax = inset_axes(ax[2], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0.0, 1, 1),  # position outside right edge
                        bbox_transform=ax[2].transAxes, borderpad=0.0)
        cbar = plt.colorbar(sc3, cax=cax)
    cbar.ax.tick_params(labelsize=12, length=0)
    cbar.set_ticks(np.linspace(vmin,vmax,3))
    cbar.set_ticklabels(["$0.0$", "$0.5$", "$1.0$"])
    # Set log scales for proper visualization
    for i in range(3):
        ax[i].axvline(x=noiseCut, color='aqua', linestyle='solid', lw=3)
        ax[i].axhline(y=interCut, color='chartreuse', linestyle='solid', lw=3)
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xticks([1e-02, 1e01, 1e4, 1e7])
        ax[i].tick_params(axis='both', labelsize=12)
        ax[i].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=14)
    for i in range(3):
        ax[i].plot(point1[0], point1[1], marker='s', color=color1, markersize=12, markeredgecolor='k')
        ax[i].plot(point2[0], point2[1], marker='s', color=color2, markersize=12, markeredgecolor='k')
        if figureName == 'smooth': ax[i].plot(point3[0], point3[1], marker='s', color=color3, markersize=12, markeredgecolor='k', clip_on=False, zorder=10)
        else: ax[i].plot(point3[0], point3[1], marker='s', color=color3, markersize=12, markeredgecolor='k')
    ax[0].set_ylabel("$Alignment$ $time,$ $\\tau_K$", fontsize=14)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.2)
    plt.subplots_adjust(wspace=0.05)
    figureName = "/home/francesco/Pictures/soft/3diagrams-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

########################################################################
############### PLOTTING FUNCTIONS FOR ORDER PARAMETERS ################
########################################################################

def plotOrderParamsVSInteraction(dirName, figureName, cluster=False, maxCluster=32): # for maximum number of clusters to consider
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,7), dpi = 120)
    alignList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "2e-01", "3e-01", "5e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    dirList = np.array(["/reflect/", "/rough/dynamics/"])
    for a in range(dirList.shape[0]):
        ax_twin = ax[a].twinx()
        tauk = np.zeros(alignList.shape[0])
        corr1 = np.zeros((alignList.shape[0], 2))
        corr2 = np.zeros((alignList.shape[0], 2))
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-" + figureName + "/dynamics-vel" + dirList[a]
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                tauk[d] = 1/readFromDynParams(dirSample, "Jvicsek")
                if alignList[d] == "1e02" and dirList[a] == "/rough/dynamics/":
                    print(tauk[d])
                if cluster == 'cluster':
                    if(os.path.exists(dirSample + "/t0/")):
                        computed = False
                        if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                            computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster)
                            computed = True
                        corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                        if computed:
                            print(dirList[a], alignList[d], tauk[d], "computed cluster kuramoto", np.mean(corrCluster))
                        corr1[d,0] = np.mean(corrCluster)
                        corr1[d,1] = np.std(corrCluster)
                    else:
                        corr1[d,0] = np.mean(data[:,6])
                        corr1[d,1] = np.std(data[:,6])
                else:
                    corr1[d,0] = np.mean(data[:,6])
                    corr1[d,1] = np.std(data[:,6])
                corr2[d,0] = np.mean(data[:,8])
                corr2[d,1] = np.std(data[:,8])
        ax[a].errorbar(tauk[tauk!=0], corr1[tauk!=0,0], corr1[tauk!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax_twin.errorbar(tauk[tauk!=0], corr2[tauk!=0,0], corr2[tauk!=0,1], color='b', marker='s', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax[a].set_xscale('log')
        ax[a].set_ylim(-0.08, 1.08)
        ax_twin.set_ylim(ax[a].get_ylim())
        ax[a].tick_params(axis='both', labelsize=14)
        ax_twin.tick_params(axis='y', colors='b', labelsize=14)
        ax[a].set_ylabel("$\\phi_r^C$", fontsize=18, rotation='horizontal', labelpad=5)
        ax_twin.set_ylabel("$\\phi_\\alpha$", fontsize=18, color='b', rotation='horizontal', labelpad=5)
        # Align labels vertically at center (x is position from the axis)
        ax[a].yaxis.set_label_coords(-0.15, 0.46)
        ax_twin.yaxis.set_label_coords(1.15, 0.57)
    ax[1].tick_params(axis='x', which='both', labeltop=False, top=True)
    ax[1].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    if cluster == 'cluster':
        figureName = "/home/francesco/Pictures/soft/interParamsCluster-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/interParams-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()


def plotOrderParamsVSNoise(dirName, figureName, cluster=False, maxCluster=32): # for maximum number of clusters to consider
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,7), dpi = 120)
    noiseList = np.array(["1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
    dirList = np.array(["/reflect/", "/rough/dynamics/"])
    for a in range(dirList.shape[0]):
        ax_twin = ax[a].twinx()
        taup = np.zeros(noiseList.shape[0])
        corr1 = np.zeros((noiseList.shape[0], 2))
        corr2 = np.zeros((noiseList.shape[0], 2))
        for d in range(noiseList.shape[0]):
            dirSample = dirName + figureName + "-tp" + noiseList[d] + "/dynamics-vel" + dirList[a]
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                taup[d] = readFromDynParams(dirSample, "taup")
                if cluster == 'cluster':
                    if(os.path.exists(dirSample + "/t0/")):
                        computed = False
                        if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                            computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster)
                            computed = True
                        corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                        if computed:
                            print(dirList[a], noiseList[d], taup[d], "computed cluster kuramoto", np.mean(corrCluster))
                        corr1[d,0] = np.mean(corrCluster)
                        corr1[d,1] = np.std(corrCluster)
                    else:
                        corr1[d,0] = np.mean(data[:,6])
                        corr1[d,1] = np.std(data[:,6])
                else:
                    corr1[d,0] = np.mean(data[:,6])
                    corr1[d,1] = np.std(data[:,6])
                corr2[d,0] = np.mean(data[:,8])
                corr2[d,1] = np.std(data[:,8])
        ax[a].errorbar(taup[taup!=0], corr1[taup!=0,0], corr1[taup!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax_twin.errorbar(taup[taup!=0], corr2[taup!=0,0], corr2[taup!=0,1], color='b', marker='s', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax[a].set_xscale('log')
        ax[a].set_ylim(-0.08, 1.08)
        ax_twin.set_ylim(ax[a].get_ylim())
        ax[a].tick_params(axis='both', labelsize=14)
        ax_twin.tick_params(axis='y', colors='b', labelsize=14)
        ax[a].set_ylabel("$\\phi_r^C$", fontsize=18, rotation='horizontal', labelpad=5)
        ax_twin.set_ylabel("$\\phi_\\alpha$", fontsize=18, color='b', rotation='horizontal', labelpad=5)
        # Align labels vertically at center (x is position from the axis)
        ax[a].yaxis.set_label_coords(-0.15, 0.46)
        ax_twin.yaxis.set_label_coords(1.15, 0.57)
    ax[1].tick_params(axis='x', which='both', labeltop=False, top=True)
    ax[1].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    if cluster == 'cluster':
        figureName = "/home/francesco/Pictures/soft/noiseParamsCluster-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/noiseParams-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()


def plotMomentsVSBoundary(dirName, figureName):
    fig, ax = plt.subplots(2, 2, figsize=(9,7), dpi = 200)
    ax[0,1].sharey(ax[0,0])   # top row
    ax[1,1].sharey(ax[1,0])   # bottom row
    boxSize = np.atleast_1d(np.loadtxt(dirName + "j1e03-tp1e03/boxSize.dat"))
    maxI = boxSize[0]**2
    f0 = float(readFromDynParams(dirName + "j1e03-tp1e03", "f0"))
    gamma = float(readFromDynParams(dirName + "j1e03-tp1e03", "damping"))
    maxL = boxSize[0] * (f0/gamma)
    interList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1",
                          "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    noiseList = np.array(["1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
    aligntime = np.zeros(interList.shape[0])
    noisetime = np.zeros(noiseList.shape[0])
    boundary = np.array(["reflect/dynamics/", "rough/dynamics/"])#, "fixed/dynamics/"])
    labelList = np.array(["$Reflective$", "$Rough$", "$Fixed$"])
    colorList = [[0.6,0.6,0.6], 'k', [0.6,0.6,1]]
    markerList = ['o', 's', 'D']
    index1 = -2
    ylabel1 = "$\\langle \\tilde{I} \\rangle$" #"$\\frac{\\langle I \\rangle}{M R^2}$"
    index2 = -1
    ylabel2 = "$\\langle \\tilde{L} \\rangle$" #"$\\frac{\\langle |L| \\rangle}{M v_0 R}$"
    ax[0,0].tick_params(axis='both', labelsize=17)
    ax[0,1].tick_params(axis='both', labelsize=17)
    ax[1,0].tick_params(axis='both', labelsize=17)
    ax[1,1].tick_params(axis='both', labelsize=17)
    ax[0,0].set_ylabel(ylabel1, fontsize=22, rotation='horizontal', labelpad=25)
    ax[1,0].set_ylabel(ylabel2, fontsize=22, rotation='horizontal', labelpad=25)
    ax[1,0].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=22)
    ax[1,1].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    # collect data vs interaction strength
    for b in range(boundary.shape[0]):
        plotline = True
        MOI = np.zeros((interList.shape[0],2))
        angMom = np.zeros((interList.shape[0],2))
        for d in range(interList.shape[0]):
            dirSample = dirName + "j" + interList[d] + "-tp1e03/dynamics-vel/" + boundary[b]
            if os.path.exists(dirSample + "/energy.dat"):
                aligntime[d] = 1/readFromDynParams(dirSample, "Jvicsek")
                energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                MOI[d,0] = np.abs(np.mean(energy[:,index1])) / maxI
                MOI[d,1] = np.std(energy[:,index1]) / maxI
                angMom[d,0] = np.abs(np.mean(energy[:,index2])) / maxL
                angMom[d,1] = np.std(energy[:,index2]) / maxL
                if interList[d] == "1e-01":
                    interCut = aligntime[d]
                    print("angular momentum at inter cut:", angMom[d])
                    if plotline:
                        ax[0,0].axhline(y=0.93, color='k', linestyle='dashed', lw=1)
                        ax[0,1].axhline(y=0.93, color='k', linestyle='dashed', lw=1)
                        plotline = False
                if interList[d] == "1e04":
                    print("boundary:", boundary[b], "ball MOI:", MOI[d])
        ax[0,0].errorbar(aligntime[MOI[:,0]!=0], MOI[MOI[:,0]!=0,0], MOI[MOI[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        ax[1,0].errorbar(aligntime[angMom[:,0]!=0], angMom[angMom[:,0]!=0,0], angMom[angMom[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
    print("Interaction cut tau_K =", interCut, ", J_K =", 1/interCut, "K =", np.pi * 1.5**2/interCut)
    # collect data vs noise strength
    for b in range(boundary.shape[0]):
        plotline = True
        MOI = np.zeros((noiseList.shape[0],2))
        angMom = np.zeros((noiseList.shape[0],2))
        for d in range(noiseList.shape[0]):
            dirSample = dirName + "j1e-01-tp" + noiseList[d] + "/dynamics-vel/" + boundary[b]
            if os.path.exists(dirSample + "/energy.dat"):
                noisetime[d] = readFromDynParams(dirSample, "taup")
                energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                MOI[d,0] = np.abs(np.mean(energy[:,index1])) / maxI
                MOI[d,1] = np.std(energy[:,index1]) / maxI
                angMom[d,0] = np.abs(np.mean(energy[:,index2])) / maxL
                angMom[d,1] = np.std(energy[:,index2]) / maxL
                if noiseList[d] == "1e03":
                    noiseCut = noisetime[d]
                    print("angular momentum at noise cut:", angMom[d])
                    if plotline:
                        plotline = False
        ax[0,1].errorbar(noisetime[MOI[:,0]!=0], MOI[MOI[:,0]!=0,0], MOI[MOI[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        ax[1,1].errorbar(noisetime[angMom[:,0]!=0], angMom[angMom[:,0]!=0,0], angMom[angMom[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
    print("Noise cut tau_p =", noiseCut)
    ax[0,0].set_xscale('log')
    ax[1,0].set_xscale('log')
    ax[0,1].set_xscale('log')
    ax[1,1].set_xscale('log')
    ax[1,0].xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax[1,0].xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax[1,1].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax[1,1].xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax[0,1].tick_params(labelleft=False)
    ax[1,1].tick_params(labelleft=False)
    ax[1,0].tick_params(top=True)
    ax[1,1].tick_params(top=True)
    ax[0,0].tick_params(labelbottom=False, bottom=True)
    ax[0,1].tick_params(labelbottom=False, bottom=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    figureName = "/home/francesco/Pictures/soft/moments-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()


def plotClustersVSBoundary(dirName, figureName, minNum=2):
    fig, ax = plt.subplots(2, 2, figsize=(9,7), dpi = 200)
    ax[0,1].sharey(ax[0,0])   # top row
    ax[1,1].sharey(ax[1,0])   # bottom row
    numParticles = float(readFromParams(dirName + "j1e03-tp1e03", "numParticles"))
    interList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1",
                          "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    noiseList = np.array(["1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
    aligntime = np.zeros(interList.shape[0])
    noisetime = np.zeros(noiseList.shape[0])
    boundary = np.array(["reflect/dynamics/", "rough/dynamics/", "fixed/dynamics/"])
    labelList = np.array(["$Reflective$", "$Rough$", "$Fixed$"])
    colorList = [[0.6,0.6,0.6], 'k', [0.6,0.6,1]]
    markerList = ['o', 's', 'D']
    index1 = 1
    ylabel1 = "$\\langle N_c \\rangle$"
    index2 = 2
    ylabel2 = "$\\frac{\\langle N_p \\rangle}{N}$"
    ax[0,0].tick_params(axis='both', labelsize=17)
    ax[0,1].tick_params(axis='both', labelsize=17)
    ax[1,0].tick_params(axis='both', labelsize=17)
    ax[1,1].tick_params(axis='both', labelsize=17)
    ax[0,0].set_ylabel(ylabel1, fontsize=24, rotation='horizontal', labelpad=35)
    ax[1,0].set_ylabel(ylabel2, fontsize=34, rotation='horizontal', labelpad=30)
    ax[1,0].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=22)
    ax[1,1].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    # collect data vs interaction strength
    for b in range(boundary.shape[0]):
        numCluster = np.zeros((interList.shape[0],2))
        clusterSize = np.zeros((interList.shape[0],2))
        for d in range(interList.shape[0]):
            dirSample = dirName + "j" + interList[d] + "-tp1e03/dynamics-vel/" + boundary[b]
            aligntime[d] = 1/readFromDynParams(dirSample, "Jvicsek")
            if interList[d] == "1e-01":
                interCut = aligntime[d]
            if(os.path.exists(dirSample + "/t0/")):
                if not(os.path.exists(dirSample + "/numCluster.dat")):
                    computeNumClusterVSTime(dirSample, minNum)
                clusterData = np.loadtxt(dirSample + "/numCluster.dat")
                numCluster[d,0] = np.mean(clusterData[:,index1])
                numCluster[d,1] = np.std(clusterData[:,index1])
                clusterSize[d,0] = np.mean(clusterData[:,index2])/numParticles
                clusterSize[d,1] = np.std(clusterData[:,index2])/numParticles
                if interList[d] == "1e-01":
                    numCluster_same = numCluster[d]
                    clusterSize_same = clusterSize[d]
                    print("clusterSize at inter cut:", clusterSize[d])
                if interList[d] == "3e03":
                    print("tauk:", aligntime[d])
        ax[0,0].errorbar(aligntime[numCluster[:,0]!=0], numCluster[numCluster[:,0]!=0,0], numCluster[numCluster[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        ax[1,0].errorbar(aligntime[clusterSize[:,0]!=0], clusterSize[clusterSize[:,0]!=0,0], clusterSize[clusterSize[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])

    # collect data vs noise strength
    for b in range(boundary.shape[0]):
        numCluster = np.zeros((noiseList.shape[0],2))
        clusterSize = np.zeros((noiseList.shape[0],2))
        for d in range(noiseList.shape[0]):
            dirSample = dirName + "j1e-01-tp" + noiseList[d] + "/dynamics-vel/" + boundary[b]
            noisetime[d] = readFromDynParams(dirSample, "taup")
            if noiseList[d] == "1e03":
                noiseCut = noisetime[d]
            if(os.path.exists(dirSample + "/t0/")):
                if not(os.path.exists(dirSample + "/numCluster.dat")):
                    computeNumClusterVSTime(dirSample, minNum)
                clusterData = np.loadtxt(dirSample + "/numCluster.dat")
                numCluster[d,0] = np.mean(clusterData[:,index1])
                numCluster[d,1] = np.std(clusterData[:,index1])
                clusterSize[d,0] = np.mean(clusterData[:,index2])/numParticles
                clusterSize[d,1] = np.std(clusterData[:,index2])/numParticles
                if noiseList[d] == "1e03":
                    numCluster_same = numCluster[d]
                    clusterSize_same = clusterSize[d]
                    print("clusterSize at noise cut:", clusterSize[d])
        ax[0,1].errorbar(noisetime[numCluster[:,0]!=0], numCluster[numCluster[:,0]!=0,0], numCluster[numCluster[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        ax[1,1].errorbar(noisetime[clusterSize[:,0]!=0], clusterSize[clusterSize[:,0]!=0,0], clusterSize[clusterSize[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
    print("Interaction cut tau_K =", interCut, ", J_K =", 1/interCut, "K =", np.pi * 1.5**2/interCut)
    print("Noise cut tau_p =", noiseCut)
    ax[1,0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[1,1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[1,0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[0,0].set_xscale('log')
    ax[1,0].set_xscale('log')
    ax[0,1].set_xscale('log')
    ax[1,1].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[0,1].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[1,0].xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax[1,0].xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax[1,1].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax[1,1].xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax[0,1].set_ylim(0.12,)
    ax[1,1].set_ylim(0.00034,)
    ax[0,1].tick_params(labelleft=False)
    ax[1,1].tick_params(labelleft=False)
    ax[1,0].tick_params(top=True)
    ax[1,1].tick_params(top=True)
    ax[0,0].tick_params(labelbottom=False, bottom=True)
    ax[0,1].tick_params(labelbottom=False, bottom=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    figureName = "/home/francesco/Pictures/soft/cluster-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()


def comparePhaseRoughness(dirName, figureName, versus='size', which1='moi', which2='angmom', dynamics='/'):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5,5.5), dpi = 120)
    alignList = np.array(["1e-01", "3", "1e02", "1e04"])
    labelList = np.array(["$7.1 \\times 10^1$", "$2.4$", "$7.1 \\times 10^{-2}$", "$7.1 \\times 10^{-4}$"])
    labelList = np.array(["$TG$", "$PDC+G$", "$PDC$", "$LC$"])
    colorList = ['forestgreen', 'dodgerblue', [0.6,0,0.8], [1,0.7,0]]
    markerList = ['v', 'o', 's', '^']
    if versus == 'size': dirList = np.array(["0", "0.1", "0.2", "0.4", "0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8"])
    else: 
        dirList = np.array(["1e-01", "1", "1e01", "1e02"])
        ew = dirList.astype(float)
    index1, ylabel1 = getIndexYlabel(which1)
    index2, ylabel2 = getIndexYlabel(which2)
    start = 1
    for a in range(alignList.shape[0]):
        obs1 = np.zeros((dirList.shape[0],2))
        obs2 = np.zeros((dirList.shape[0],2))
        dirPath = dirName + "j" + alignList[a] + "-tp1e03/dynamics-vel/"
        boxRadius = np.loadtxt(dirPath + "/boxSize.dat")
        f0 = float(readFromDynParams(dirPath, "f0"))
        gamma = float(readFromDynParams(dirPath, "damping"))
        maxI = boxRadius**2
        maxL = boxRadius * (f0/gamma)
        if versus == 'size':
            roughness = np.zeros(dirList.shape[0])
            for d in range(start, dirList.shape[0]):
                if dirList[d] == '0': dirSample = dirPath + "fixed" + dynamics
                else: dirSample = dirPath + "rough" + dirList[d] + dynamics
                if os.path.exists(dirSample):
                    if dirList[d] != '0': roughness[d] = 2 * readFromWallParams(dirPath + "rough" + dirList[d], "wallRad")
                    if(os.path.exists(dirSample + "/energy.dat")):
                        energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                        if which1 == "moi" and which2 == "angmom":
                            energy[:,index1] /= maxI
                            energy[:,index2] /= maxL
                        obs1[d,0] = np.mean(np.abs(energy[:,index1]))
                        obs1[d,1] = np.std(np.abs(energy[:,index1]))
                        obs2[d,0] = np.mean(np.abs(energy[:,index2]))
                        obs2[d,1] = np.std(np.abs(energy[:,index2]))
            if which1 == "moi": colorId = 1 - (np.mean(energy[:,-2]) - 0.7) / 0.3
            else: colorId = 1 - (np.mean(energy[:,-2]) / maxI - 0.7) / 0.3
            print("color index:", colorId)
            ax[0].errorbar(roughness[1:], obs1[1:,0], obs1[1:,1], lw=1.2, color=colorList[a], marker=markerList[a], 
                        markersize=8, fillstyle='none', capsize=3, label="$\\tau_K \\approx$" + labelList[a])
            ax[1].errorbar(roughness[1:], obs2[1:,0], obs2[1:,1], lw=1.2, color=colorList[a], marker=markerList[a], 
                        markersize=8, fillstyle='none', capsize=3, label="$\\tau_K \\approx$" + labelList[a])
            if start == 0:
                ax[0].errorbar(roughness[0], obs1[0,0], obs1[0,1], lw=1.2, color=colorList[a], marker=markerList[a], markersize=8, fillstyle='none', capsize=3)
                ax[1].errorbar(roughness[0], obs2[0,0], obs2[0,1], lw=1.2, color=colorList[a], marker=markerList[a], markersize=8, fillstyle='none', capsize=3)
        else:
            for d in range(dirList.shape[0]):
                if dirList[d] == "1e01": dirSample = dirPath + "fixed" + dynamics
                else: dirSample = dirPath + "fixed-ew" + dirList[d] + dynamics
                if os.path.exists(dirSample):
                    if(os.path.exists(dirSample + "/energy.dat")):
                        energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                        if which1 == "moi" and which2 == "angmom":
                            energy[:,index2] /= maxI
                            energy[:,index1] /= maxL
                        obs1[d,0] = np.mean(np.abs(energy[:,index1]))
                        obs1[d,1] = np.std(np.abs(energy[:,index1]))
                        obs2[d,0] = np.mean(np.abs(energy[:,index2]))
                        obs2[d,1] = np.std(np.abs(energy[:,index2]))
            # get color from value of velpos parameter
            if which1 == "moi": colorId = 1 - (np.mean(energy[:,-2]) - 0.7) / 0.3
            else: colorId = 1 - (np.mean(energy[:,-2]) / maxI - 0.7) / 0.3
            print("color index:", colorId)
            ax[0].errorbar(ew, obs1[:,0], obs1[:,1], lw=1.2, color=colorList(colorId), marker=markerList[a], 
                        markersize=8, fillstyle='none', capsize=3, label="$\\tau_K \\approx$" + labelList[a])
            ax[1].errorbar(ew, obs2[:,0], obs2[:,1], lw=1.2, color=colorList(colorId), marker=markerList[a], 
                        markersize=8, fillstyle='none', capsize=3, label="$\\tau_K \\approx$" + labelList[a])
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    if which1 == "moi" and which2 == "angmom":
        ax[0].set_ylabel("$\\frac{\\langle I \\rangle}{M R^2}$", fontsize=24, rotation='horizontal', labelpad=25)
        ax[1].set_ylabel("$\\frac{\\langle |L| \\rangle}{M v_0 R}$", fontsize=24, rotation='horizontal', labelpad=30)
        ax[0].set_ylabel("$\\tilde{ I }$", fontsize=18, rotation='horizontal', labelpad=25)
        ax[1].set_ylabel("$\\tilde{ L }$", fontsize=18, rotation='horizontal', labelpad=30)
        ax[0].set_ylim(0.47,1.06)
        ax[1].set_ylim(-0.12,1.12)
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[0].legend(loc='lower right', fontsize=14, ncols=2)
    elif which1 == "pphi":
        ax[0].set_ylabel("$|P_\\phi|$", fontsize=18, rotation='horizontal', labelpad=30)
        ax[1].set_ylabel(ylabel2, fontsize=18, rotation='horizontal', labelpad=30)
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].legend(loc='lower right', fontsize=12, ncols=2)
    else:
        ax[0].set_ylabel(ylabel1, fontsize=18, rotation='horizontal', labelpad=30)
        ax[1].set_ylabel(ylabel2, fontsize=18, rotation='horizontal', labelpad=30)
        ax[1].legend(loc='lower right', fontsize=12, ncols=2)
    if versus == 'size': ax[1].set_xlabel("$Roughness,$ $\\sigma_m / \\sigma$", fontsize=16)
    else: 
        ax[1].set_xlabel("$Relative$ $strength,$ $\\epsilon_w / \\epsilon$", fontsize=16)
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
    ax[1].tick_params(top=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figure1Name = "/home/francesco/Pictures/soft/phaseRough-" + which1 + "-" + which2 + "-" + figureName
    fig.savefig(figure1Name + ".png", transparent=False, format = "png")
    plt.show()


def compareAngMomRoughness(dirName, figureName, dynamics='/'):
    fig, ax = plt.subplots(figsize=(7.2,2.8), dpi = 200)
    alignList = np.array(["1e-01", "3", "1e02", "1e04"])
    labelList = np.array(["$7.1 \\times 10^1$", "$2.4$", "$7.1 \\times 10^{-2}$", "$7.1 \\times 10^{-4}$"])
    labelList = np.array(["$TG$", "$PDC+G$", "$PDC$", "$LC$"])
    colorList = ['forestgreen', 'dodgerblue', [0.6,0,0.8], [1,0.7,0]]
    markerList = ['v', 'o', 's', '^']
    dirList = np.array(["0.1", "0.2", "0.4", "0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8"])
    index, ylabel = getIndexYlabel('angmom')
    for a in range(alignList.shape[0]):
        obs = np.zeros((dirList.shape[0],2))
        dirPath = dirName + "j" + alignList[a] + "-tp1e03/dynamics-vel/"
        boxRadius = np.loadtxt(dirPath + "/boxSize.dat")
        f0 = float(readFromDynParams(dirPath, "f0"))
        gamma = float(readFromDynParams(dirPath, "damping"))
        maxL = boxRadius * (f0/gamma)
        roughness = np.zeros(dirList.shape[0])
        for d in range(dirList.shape[0]):
            dirSample = dirPath + "rough" + dirList[d] + dynamics
            if os.path.exists(dirSample):
                roughness[d] = 2 * readFromWallParams(dirPath + "rough" + dirList[d], "wallRad")
                if(os.path.exists(dirSample + "/energy.dat")):
                    energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                    energy[:,index] /= maxL
                    obs[d,0] = np.mean(np.abs(energy[:,index]))
                    obs[d,1] = np.std(np.abs(energy[:,index]))
        ax.errorbar(roughness, obs[:,0], obs[:,1], lw=1.2, color=colorList[a], marker=markerList[a], 
                    markersize=10, fillstyle='none', capsize=3, label=labelList[a])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$\\langle \\tilde{L} \\rangle$", fontsize=18, rotation='horizontal', labelpad=20)
    ax.set_xlabel("$Roughness,$ $\\sigma_m / \\sigma$", fontsize=18)
    ax.set_ylim(-0.12,1.42)
    ax.legend(loc='upper right', fontsize=14, ncols=4, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figure1Name = "/home/francesco/Pictures/soft/angmomRough-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    plt.show()



if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

    if(whichPlot == "energy"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotEnergyFile(dirName, figureName, which)

    elif(whichPlot == "aligninter"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        taup = sys.argv[5]
        dynamics = sys.argv[6]
        plotAlignmentVSInteraction(dirName, figureName, which, taup, dynamics)

    elif(whichPlot == "compareinter"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareAlignmentVSInteraction(dirName, figureName, which, dynamics)

    elif(whichPlot == "alignnoise"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        jvic = sys.argv[5]
        dynamics = sys.argv[6]
        plotAlignmentVSNoise(dirName, figureName, which, jvic, dynamics)

    elif(whichPlot == "comparenoise"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareAlignmentVSNoise(dirName, figureName, which, dynamics)

    elif(whichPlot == "clustercom"):
        figureName = sys.argv[3]
        plotClustersCOM(dirName, figureName)

    elif(whichPlot == "clusterphi"):
        figureName = sys.argv[3]
        plotClusterDensity(dirName, figureName)

    elif(whichPlot == "numcluster"):
        figureName = sys.argv[3]
        versus = sys.argv[4]
        which = sys.argv[5]
        dynamics = sys.argv[6]
        minNum = float(sys.argv[7])
        compareNumClusterVSTime(dirName, figureName, versus, which, dynamics, minNum)

    elif(whichPlot == "phasediagram"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        which = sys.argv[5]
        interpolate = sys.argv[6]
        phaseDiagramNoiseAlignment(dirName, figureName, dynamics, which, interpolate)

    elif(whichPlot == "3diagrams"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        cluster = sys.argv[5]
        if cluster == 'cluster':
            cluster = True
        else:
            cluster = False
        maxCluster = int(sys.argv[6])
        interpolate = sys.argv[7]
        phaseDiagrams3(dirName, figureName, dynamics, cluster, maxCluster, interpolate)

    elif(whichPlot == "interparams"):
        figureName = sys.argv[3]
        cluster = sys.argv[4]
        maxCluster = int(sys.argv[5])
        plotOrderParamsVSInteraction(dirName, figureName, cluster, maxCluster)

    elif(whichPlot == "noiseparams"):
        figureName = sys.argv[3]
        cluster = sys.argv[4]
        maxCluster = int(sys.argv[5])
        plotOrderParamsVSNoise(dirName, figureName, cluster, maxCluster)

    elif(whichPlot == "clusterbound"):
        figureName = sys.argv[3]
        minNum = float(sys.argv[4])
        plotClustersVSBoundary(dirName, figureName, minNum)

    elif(whichPlot == "momsbound"):
        figureName = sys.argv[3]
        plotMomentsVSBoundary(dirName, figureName)

    elif(whichPlot == "phaserough"):
        figureName = sys.argv[3]
        versus = sys.argv[4]
        which1 = sys.argv[5]
        which2 = sys.argv[6]
        dynamics = sys.argv[7]
        comparePhaseRoughness(dirName, figureName, versus, which1, which2, dynamics)

    elif(whichPlot == "angmomrough"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        compareAngMomRoughness(dirName, figureName, dynamics)

    else:
        print("Please specify the type of plot you want")
