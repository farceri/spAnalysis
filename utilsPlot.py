'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import os

def setAxes2D(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def setPackingAxes(boxSize, ax):
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def plotPacking(boxSize, pos, rad, labels=np.empty([])):
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    setPackingAxes(boxSize, ax)
    numParticles = rad.shape[0]
    colorId = np.zeros((pos.shape[0], 4))
    if(labels.size != 1):
        numLabels = np.max(labels)+1
        colorList = cm.get_cmap('prism', numLabels)
        for i in range(numParticles):
            if(labels[i]==-1): # particles not in a cluster
                colorId[i] = [1,1,1,1]
            else:
                colorId[i] = colorList(labels[i]/numLabels)
    else:
        for i in range(numParticles):
            colorId[i] = [0,0,1,0]
    for particleId in range(numParticles):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=0.6, linewidth=0.5))
    plt.pause(0.5)

#################################### plotting ##################################
def plotErrorBar(ax, x, y, err, xlabel, ylabel, logx = False, logy = False):
    ax.errorbar(x, y, err, marker='o', color='k', markersize=7, markeredgecolor='k', markeredgewidth=0.7, linewidth=1.2, elinewidth=1, capsize=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')
    plt.tight_layout()

def plotCorrWithError(x, y, err, ylabel, xlabel = "$Time$ $interval,$ $\\Delta t$", logx = False, logy = False, color = 'k', show = True):
    fig = plt.figure(0, dpi = 120)
    ax = fig.gca()
    ax.errorbar(x, y, err, marker='o', color=color, markersize=4, markeredgecolor='k', markeredgewidth=0.7, linewidth=1.2, elinewidth=1, capsize=4)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    if(logx == True):
        ax.set_xscale('log')
    if(logy == True):
        ax.set_yscale('log')
    plt.tight_layout()
    if(show == True):
        plt.pause(0.5)

def plotCorrelation(x, y, ylabel, xlabel = "$Distance,$ $r$", logy = False, logx = False, color = 'k', markersize = 4, lw = 1.2, ls='solid', show = True):
    fig = plt.figure(0, dpi = 120)
    ax = fig.gca()
    ax.plot(x, y, color=color, marker='o', markersize=markersize, lw=lw, ls=ls)
    if(logy == True):
        ax.set_yscale('log')
    if(logx == True):
        ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    plt.tight_layout()
    if(show == True):
        plt.pause(0.5)

def getStepList(numFrames, firstStep, stepFreq):
    maxStep = int(firstStep + stepFreq * numFrames)
    stepList = np.arange(firstStep, maxStep, stepFreq, dtype=int)
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    return stepList

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


if __name__ == '__main__':
    print("library for plotting utilities")
