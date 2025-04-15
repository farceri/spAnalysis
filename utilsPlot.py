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

def plotCorrWithError(x, y, err, ylabel, xlabel = "$Time$ $interval,$ $\\Delta t$", logx = False, logy = False, color = 'k', show = True, marker='o', lw = 1.2):
    fig = plt.figure(0, dpi = 120)
    ax = fig.gca()
    ax.errorbar(x, y, err, marker=marker, fillstyle='none', color=color, markersize=6, markeredgewidth=1, linewidth=1.2, elinewidth=1, capsize=4)
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

def plotCorrelation(x, y, ylabel, xlabel = "$Distance,$ $r$", logy = False, logx = False, color = 'k', markersize = 4, lw = 1.2, ls='solid', show = True, marker='o', fs='full'):
    fig = plt.figure(0, dpi = 120)
    ax = fig.gca()
    ax.plot(x, y, color=color, marker=marker, markersize=markersize, lw=lw, ls=ls, fillstyle=fs)
    #ax.plot(np.linspace(np.min(x), np.max(x), 100), np.ones(100)*1e07, lw=1.2, ls='dotted', color='k')
    if(logy == True):
        ax.set_yscale('log')
    if(logx == True):
        ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    if(show == True):
        plt.pause(0.5)


if __name__ == '__main__':
    print("library for plotting utilities")
