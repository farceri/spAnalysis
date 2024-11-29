'''
Created by Francesco
12 October 2021
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import animation
from matplotlib import cm
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import sys
import os
import utils
import spCluster as cluster

def setAxes3D(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

def set3DPackingAxes(boxSize, ax):
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    zBounds = np.array([0, boxSize[2]])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_ylim(zBounds[0], zBounds[1])
    #ax.set_box_aspect(aspect = (1,1,1))
    #ax.set_aspect('equal', adjustable='box')
    setAxes3D(ax)

def plot3DPacking(dirName, figureName):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + 'boxSize.dat')
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    zBounds = np.array([0, boxSize[2]])
    rad = np.array(np.loadtxt(dirName + sep + 'particleRad.dat'))
    pos = np.array(np.loadtxt(dirName + os.sep + 'particlePos.dat'))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    pos[:,2] -= np.floor(pos[:,2]/boxSize[2]) * boxSize[2]
    fig = plt.figure(dpi=200)
    ax = Axes3D(fig)
    set3DPackingAxes(boxSize, ax)
    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0, np.pi, 120)
    colorId = getRadColorList(rad)
    for i in range(pos.shape[0]):
        x = pos[i,0] + rad[i]*np.outer(np.cos(u), np.sin(v))
        y = pos[i,1] + rad[i]*np.outer(np.sin(u), np.sin(v))
        z = pos[i,2] + rad[i]*np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x,y,z, color=colorId[i], rstride=4, cstride=4, alpha=1)
    plt.savefig('/home/francesco/Pictures/soft/packings/3d-' + figureName + '.png', transparent=True, format = 'png')
    plt.show()

def setAxes2D(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def setPackingAxes(boxSize, ax):
    if(boxSize.shape[0] == 1):
        ax.set_xlim(-boxSize, boxSize)
        ax.set_ylim(-boxSize, boxSize)
    else:
        ax.set_xlim(0, boxSize[0])
        ax.set_ylim(0, boxSize[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def setInvisiblePackingAxes(boxSize, ax):
    setPackingAxes(boxSize, ax)
    for spine in ax.spines.values():
        spine.set_visible(False)

def setCenteredPackingAxes(boxSize, frameSize, ax):
    # Centering the frame
    x_center = boxSize[0] / 2
    y_center = boxSize[1] / 2
    ax.set_xlim([x_center - frameSize[0] / 2, x_center + frameSize[0] / 2])
    ax.set_ylim([y_center - frameSize[1] / 2, y_center + frameSize[1] / 2])
    ax.set_aspect('equal', anchor='C')
    setAxes2D(ax)

def setAutoPackingAxes(boxSize, ax):
    ax.set_xlim(0, boxSize[0])
    ax.set_ylim(0, boxSize[1])
    ax.set_aspect('equal', anchor='C')
    setAxes2D(ax)

def setZoomPackingAxes(xBounds, yBounds, ax):
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def setGridAxes(bins, ax):
    xBounds = np.array([bins[0], bins[-1]])
    yBounds = np.array([bins[0], bins[-1]])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def setBigBoxAxes(boxSize, ax, delta=1.1):
    if(boxSize.shape[0] == 1):
        ax.set_xlim(-boxSize, boxSize)
        ax.set_ylim(-boxSize, boxSize)
        xBounds = np.array([-boxSize*delta, boxSize*delta])
        yBounds = np.array([-boxSize*delta, boxSize*delta])
    else:
        xBounds = np.array([boxSize[0]*(1-delta), boxSize[0]*delta])
        yBounds = np.array([boxSize[1]*(1-delta), boxSize[1]*delta])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def getRadColorList(rad):
    colorList = cm.get_cmap('viridis', rad.shape[0])
    colorId = np.zeros((rad.shape[0], 4))
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    return colorId

def getAngleColorList(vel):
    colorMap = cm.get_cmap('hsv')  # Set the color map
    angle = ((utils.getVelocityAngles(vel) + 2.*np.pi) % (2. * np.pi)) / (2*np.pi)
    # Create color array based on normalized angles
    colorId = colorMap(angle)
    return colorId

def getForceColorList(force):
    force = np.linalg.norm(force,axis=1)
    colorList = cm.get_cmap('bwr', force.shape[0])
    colorId = np.zeros((force.shape[0], 4))
    count = 0
    for particleId in np.argsort(force):
        colorId[particleId] = colorList(count/force.shape[0])
        count += 1
    return colorId

def getEkinColorList(ekin):
    colorList = cm.get_cmap('viridis', ekin.shape[0])
    colorId = np.zeros((ekin.shape[0], 4))
    count = 0
    for particleId in np.argsort(ekin):
        colorId[particleId] = colorList(count/ekin.shape[0])
        count += 1
    return colorId

def getColorListFromLabels(labels):
    numLabels = np.unique(labels).shape[0]-1
    colorList = cm.get_cmap('tab20', numLabels)
    colorId = np.zeros((labels.shape[0], 4))
    for particleId in range(labels.shape[0]):
        if(labels[particleId]==-1 or labels[particleId]==0): # particles not in a cluster
            colorId[particleId] = [1,1,1,1]
        else:
            colorId[particleId] = colorList(labels[particleId]/numLabels)
    return colorId

def getDenseColorList(denseList):
    colorId = np.zeros((denseList.shape[0], 4))
    for particleId in range(denseList.shape[0]):
        if(denseList[particleId]==1):
            colorId[particleId] = [0.2,0.2,0.2,0.2]
        else:
            colorId[particleId] = [1,1,1,1]
    return colorId

def getDoubleColorList(rad, num1=0, tag=False):
    colorId = np.zeros((rad.shape[0], 4))
    colorId[:num1] = [0,1,0,0.2]
    colorId[num1:] = [0,0,1,0.2]
    if(tag == True):
        tagList = np.linspace(500, 8000, 30).astype(int)
        colorId[tagList] = [0,0,0,0]
        tagList = np.linspace(8500, 16000, 30).astype(int)
        colorId[tagList] = [1,1,1,1]
    return colorId

def plotSPPacking(dirName, figureName, fixed=False, shear=False, lj=False, ekmap=False, forcemap=False, quiver=False, dense=False, border=False, 
                  threshold=0.62, filter='filter', strain=0, shiftx=0, shifty=0, center=False, double=False, num1=0, alpha=0.6):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.atleast_1d(np.loadtxt(dirName + sep + 'boxSize.dat'))
    print(boxSize)
    roundBox = False
    if(boxSize.shape[0] == 1):
        print("Setting circular box geometry")
        roundBox = True
    if fixed or roundBox:
        pos = np.array(np.loadtxt(dirName + os.sep + 'particlePos.dat'))
        if roundBox:
            outSideIdx = utils.checkParticlesInCircle(pos, boxSize)
            if(outSideIdx.shape[0] != 0):
                vel = np.array(np.loadtxt(dirName + os.sep + 'particleVel.dat'))
    elif shear:
        figureName = '/home/francesco/Pictures/soft/packings/shear-' + figureName + '.png'
        pos = utils.getLEPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize, strain)
    else:
        pos = utils.getPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize)
    rad = np.array(np.loadtxt(dirName + sep + 'particleRad.dat'))
    if lj:
        rad *= 2**(1/6)
    if(shiftx != 0 or shifty != 0):
        pos = utils.shiftPositions(pos, boxSize, shiftx, shifty)
    else:
        if(center == 'center'):
            pos = utils.centerCOM(pos, rad, boxSize)
        elif(center == 'centercluster'):
            eps = 1.8*np.max(rad)
            labels, maxLabel = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold)
            pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
            print('maxLabel:', maxLabel, 'number of particles in biggest cluster:', labels[labels==maxLabel].shape[0])
    print('Center of mass:', np.mean(pos, axis=0))
    if not roundBox:
        print('BoxRatio Ly / Lx:', boxSize[1] / boxSize[0])
    # make figure
    fig, ax = plt.subplots(dpi=200)
    if fixed or roundBox:
        setBigBoxAxes(boxSize, ax, 1.05)
        if roundBox:
            fig.patch.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_artist(plt.Circle([0, 0], boxSize, edgecolor='k', facecolor=[1,1,1], linewidth=0.5)) 
    else:
        setPackingAxes(boxSize, ax)
    if dense:
        figureName = '/home/francesco/Pictures/soft/packings/dense-' + figureName + '.png'
        if not(os.path.exists(dirName + os.sep + 'particleList.dat')):
            cluster.computeDelaunayCluster(dirName, threshold, filter=filter)
        denseList = np.loadtxt(dirName + os.sep + 'particleList.dat')[:,0]
        colorId = getDenseColorList(denseList)
    elif border:
        figureName = '/home/francesco/Pictures/soft/packings/border-' + figureName + '.png'
        if not(os.path.exists(dirName + os.sep + 'particleList!.dat')):
            cluster.computeDelaunayCluster(dirName, threshold, filter=filter)
        borderList = np.loadtxt(dirName + os.sep + 'particleList.dat')[:,1]
        colorId = getDenseColorList(borderList)
    elif ekmap:
        vel = np.array(np.loadtxt(dirName + os.sep + 'particleVel.dat'))
        ekin = 0.5*np.linalg.norm(vel, axis=1)**2
        colorId = getEkinColorList(ekin)
    elif forcemap:
        force = np.array(np.loadtxt(dirName + os.sep + 'particleForces.dat'))
        colorId = getForceColorList(force)
    elif double:
        colorId = getDoubleColorList(rad, num1)
    else:
        colorId = getRadColorList(rad)
    if quiver:
        vel = np.array(np.loadtxt(dirName + os.sep + 'particleVel.dat'))
        colorId = getAngleColorList(vel)
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        #print('particle', particleId, 'position:', x, y)
        if quiver:
            #ax.add_artist(plt.Circle([x, y], r, edgecolor=colorId[particleId], facecolor='none', alpha=alpha, linewidth=0.7))
            ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=0.3))
            vx = vel[particleId,0]
            vy = vel[particleId,1]
            ax.quiver(x, y, vx, vy, facecolor='k', linewidth=0.1, width=0.001, scale=80, headlength=5, headaxislength=5, headwidth=5, alpha=0.6)
        else:
            ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=0.3))
            if roundBox:
                if(outSideIdx.shape[0] != 0):
                    if(np.isin(particleId, outSideIdx)):
                        ax.add_artist(plt.Circle([x, y], 10*r, edgecolor='k', facecolor='k', alpha=alpha, linewidth=0.3))
                        vx = vel[particleId,0]
                        vy = vel[particleId,1]
                        ax.quiver(x, y, vx, vy, facecolor='k', width=0.002, scale=10)
            #print(particleId)
            #plt.pause(0.5)
        #label = ax.annotate(str(particleId), xy=(x, y), fontsize=4, verticalalignment='center', horizontalalignment='center')
    if ekmap:
        colorBar = cm.ScalarMappable(cmap='viridis')
        cb = plt.colorbar(colorBar)
        label = '$E_{kin}$'
        cb.set_ticks([0, 1])
        cb.ax.tick_params(labelsize=12, size=0)
        ticklabels = [np.format_float_scientific(np.min(ekin), precision=2), np.format_float_scientific(np.max(ekin), precision=2)]
        cb.set_ticklabels(ticklabels)
        cb.set_label(label=label, fontsize=14, labelpad=-20, rotation='horizontal')
        figureName = '/home/francesco/Pictures/soft/packings/ekmap-' + figureName + '.png'
    elif quiver:
        figureName = '/home/francesco/Pictures/soft/packings/velmap-' + figureName + '.png'
    else:
        figureName = '/home/francesco/Pictures/soft/packings/' + figureName + '.png'
    plt.tight_layout()
    plt.savefig(figureName, transparent=False)
    plt.show()

def plotMobilePacking(dirName, figureName, lj=False, colorAngle=False, alpha=0.6):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.atleast_1d(np.loadtxt(dirName + sep + 'boxSize.dat'))
    print(boxSize)
    if(boxSize.shape[0] == 1):
        print("Setting circular box geometry")
        pos = np.array(np.loadtxt(dirName + os.sep + 'particlePos.dat'))
        wallPos = np.array(np.loadtxt(dirName + os.sep + 'wallPos.dat'))
    else:
        print("plotMobilePacking is designed for round boxes only")
        exit()
    rad = np.array(np.loadtxt(dirName + sep + 'particleRad.dat'))
    wallRad = utils.readFromWallParams(dirName, 'wallRad')
    if lj:
        rad *= 2**(1/6)
    print('Center of mass:', np.mean(pos, axis=0))
    # make figure
    fig, ax = plt.subplots(dpi=200)
    setInvisiblePackingAxes(boxSize, ax)
    # plot wall particles
    for wallId in range(wallPos.shape[0]):
        x = wallPos[wallId,0]
        y = wallPos[wallId,1]
        r = wallRad
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=[0.9,0.9,0.9], alpha=alpha, linewidth=0.3))
    # choose color map and plot particles
    if colorAngle:
        vel = np.array(np.loadtxt(dirName + os.sep + 'particleVel.dat'))
        colorId = getAngleColorList(vel)
    else:
        colorId = getRadColorList(rad)
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=0.3))
    figureName = '/home/francesco/Pictures/soft/packings/wall-' + figureName + '.png'
    plt.tight_layout()
    plt.savefig(figureName, transparent=False)
    plt.show()

def getStressColorList(stress, which='total', potential='lj'):
    colorList = cm.get_cmap('viridis', stress.shape[0])
    colorId = np.zeros((stress.shape[0], 4))
    count = 0
    if(which=='total'):
        if(potential=='lj' or potential=='ra'):
            p = np.sum(stress[:,0:2], axis=1)
        else:
            p = np.sum(stress[:,0:3], axis=1)
    elif(which=='steric'):
        p = stress[:,0]
    elif(which=='thermal'):
        p = stress[:,1]
    elif(which=='active'):
        p = stress[:,2]
    elif(which=='etot'):
        p = stress[:,3]
    for particleId in np.argsort(p):
        colorId[particleId] = colorList(count/p.shape[0])
        count += 1
    return colorId, colorList

def plotSPStressMapPacking(dirName, figureName, which='total', potential='lj', lcut=5.5, shiftx=0, shifty=0, alpha=0.7):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + 'boxSize.dat')
    rad = np.array(np.loadtxt(dirName + sep + 'particleRad.dat'))
    pos = utils.getPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize)
    pos = utils.shiftPositions(pos, boxSize, shiftx, shifty)
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    setPackingAxes(boxSize, ax)
    if not(os.path.exists(dirName + os.sep + 'particleStress!.dat')):
        if(potential == 'lj'):
            stress = cluster.computeLJParticleStress(dirName, lcut)
        elif(potential == 'ra'):
            stress = cluster.computeRAParticleStress(dirName, lcut)
        else:
            stress = cluster.computeParticleStress(dirName)
    stress = np.loadtxt(dirName + os.sep + 'particleStress.dat')
    colorId, colorList = getStressColorList(stress, which, potential)
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=0.3))
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cb = plt.colorbar(colorBar, cax=cax)
    cb.set_ticks(np.linspace(0, 1, 5))
    cb.ax.tick_params(labelsize=10)
    if(which=='total'):
        mintick = np.min(stress[:,0] + stress[:,1] + stress[:,2])
        maxtick = np.max(stress[:,0] + stress[:,1] + stress[:,2])
        label = '$ Total$\n$stress$'
    elif(which=='steric'):
        mintick = np.min(stress[:,0])
        maxtick = np.max(stress[:,0])
        label = '$ Steric$\n$stress$'
    elif(which=='thermal'):
        mintick = np.min(stress[:,1])
        maxtick = np.max(stress[:,1])
        label = '$ Thermal$\n$stress$'
    elif(which=='active'):
        mintick = np.min(stress[:,2])
        maxtick = np.max(stress[:,2])
        label = '$ Active$\n$stress$'
    elif(which=='etot'):
        mintick = np.min(stress[:,3])
        maxtick = np.max(stress[:,3])
        label = '$E_{tot}$'
    tickList = np.linspace(mintick, maxtick, 5)
    for i in range(tickList.shape[0]):
        #tickList[i] = np.format_float_positional(tickList[i], precision=0)
        tickList[i] = np.format_float_scientific(tickList[i], precision=0)
    cb.set_ticklabels(tickList)
    cb.set_label(label=label, fontsize=12, labelpad=25, rotation='horizontal')
    plt.tight_layout()
    figureName = '/home/francesco/Pictures/soft/packings/pmap-' + figureName + '.png'
    plt.savefig(figureName, transparent=True, format = 'png')
    plt.show()

def getDenseSimplexColorList(denseList):
    colorId = np.ones(denseList.shape[0])
    for simplexId in range(denseList.shape[0]):
        if(denseList[simplexId]==0):
            colorId[simplexId] = 0
    return colorId

def computeDenseSimplexColorList(densityList):
    colorId = np.ones(densityList.shape[0])
    for simplexId in range(densityList.shape[0]):
        if(densityList[simplexId] > 0.78):
            colorId[simplexId] = 0
        if(densityList[simplexId] < 0.78 and densityList[simplexId] > 0.453):
            colorId[simplexId] = 0.5
    return colorId

def getDenseBorderColorList(denseList, borderList):
    colorId = np.zeros((denseList.shape[0], 4))
    for particleId in range(denseList.shape[0]):
        if(borderList[particleId]==1):
            colorId[particleId] = [0.5,0.5,0.5,0.5]
        else:
            if(denseList[particleId]==1):
                colorId[particleId] = [0,0,0.8,1]#[0.2,0.2,0.2,0.2]
            else:
                colorId[particleId] = [0,0.8,0,1]
    return colorId

def getBorderColorList(borderList):
    colorId = np.zeros((borderList.shape[0], 4))
    for particleId in range(borderList.shape[0]):
        if(borderList[particleId]==1):
            colorId[particleId] = [1,0,0,1]#[0.2,0.2,0.2,0.2]
        else:
            colorId[particleId] = [1,1,1,1]
    return colorId

def plotSPDelaunayPacking(dirName, figureName, dense=False, border=False, threshold=0.58, filter='filter', alpha=0.8, colored=False, shiftx=0, shifty=0, lj=False):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + 'boxSize.dat')
    rad = np.array(np.loadtxt(dirName + sep + 'particleRad.dat'))
    if lj:
        rad *= 2**(1/6)
    pos = utils.getPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize)
    pos = utils.shiftPositions(pos, boxSize, shiftx, shifty) # for 4k and 16k, -0.3, 0.1 for 8k 0 -0.2
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    setPackingAxes(boxSize, ax)
    #setBigBoxAxes(boxSize, ax, 1.1)
    colorId = getRadColorList(rad)
    if(dense==True):
        if not(os.path.exists(dirName + os.sep + 'particleList!.dat')):
            cluster.computeDelaunayCluster(dirName, threshold, filter=filter)
        denseList = np.loadtxt(dirName + os.sep + 'particleList.dat')[:,0]
        colorId = getDenseColorList(denseList)
    if(border==True):
        if not(os.path.exists(dirName + os.sep + 'particleList!.dat')):
            cluster.computeDelaunayCluster(dirName, threshold, filter=filter)
        borderList = np.loadtxt(dirName + os.sep + 'particleList.dat')[:,1]
        denseList = np.loadtxt(dirName + os.sep + 'particleList.dat')[:,0]
        colorId = getBorderColorList(borderList)
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=0.5, linewidth=0.3))
    if(colored == 'colored'):
        newPos, simplices, colorId, borderColorId = cluster.computeAugmentedDelaunayCluster(dirName, threshold, filter, shiftx, shifty) # colorId is 0 for dense and 1 for dilute
        if(dense==True):
            plt.tripcolor(newPos[:,0], newPos[:,1], simplices, lw=0.3, facecolors=colorId, edgecolors='k', alpha=0.5, cmap='bwr')
            #plt.tripcolor(newPos[:,0], newPos[:,1], simplices[colorId==0], lw=0.3, facecolors=colorId[colorId==0], edgecolors='k', alpha=0.5, cmap='bwr')
        if(border==True):
            plt.tripcolor(newPos[:,0], newPos[:,1], simplices[borderColorId==0], lw=0.3, facecolors=borderColorId[borderColorId==0], edgecolors='k', alpha=0.5, cmap='bwr')
        plt.triplot(newPos[:,0], newPos[:,1], simplices, lw=0.2, color='k')
        if(filter == 'filter'):
            figureName = 'filter-' + figureName
        else:
            figureName = 'cluster-' + figureName
    else:
        newPos, newRad, newIndices = utils.augmentPacking(pos, rad, lx=boxSize[0], ly=boxSize[1])
        simplices = Delaunay(newPos).simplices
        simplices = np.unique(np.sort(simplices, axis=1), axis=0)
        insideIndex = utils.getInsideBoxDelaunaySimplices(simplices, newPos, boxSize)
        plt.triplot(newPos[:,0], newPos[:,1], simplices, lw=0.2, color='k')
    if(dense==True):
        figureName = '/home/francesco/Pictures/soft/packings/deldense-' + figureName + '.png'
    elif(border==True):
        figureName = '/home/francesco/Pictures/soft/packings/delborder-' + figureName + '.png'
    else:
        figureName = '/home/francesco/Pictures/soft/packings/del-' + figureName + '.png'
    plt.tight_layout()
    plt.savefig(figureName, transparent=False, format = 'png')
    plt.show()

def plotSPDelaunayLabels(dirName, figureName, dense=False, threshold=0.78, filter=False, alpha=0.8, label='dense3dilute'):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + 'boxSize.dat')
    rad = np.array(np.loadtxt(dirName + sep + 'particleRad.dat'))
    pos = utils.getPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize)
    pos = utils.shiftPositions(pos, boxSize, 0, -0.2)
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    setPackingAxes(boxSize, ax)
    # plot simplices belonging to a certain label
    if(filter == 'filter'):
        checkFile = 'denseParticleList-filter.dat'
    else:
        checkFile = 'denseParticleList.dat'
    if(os.path.exists(dirName + os.sep + 'augmented/' + checkFile)):
        denseList = np.loadtxt(dirName + os.sep + 'augmented/' + checkFile)
    else:
        cluster.computeAugmentedDelaunayCluster(dirName, threshold, filter, 0, -0.2, label='label')
        denseList = np.loadtxt(dirName + os.sep + 'augmented/' + checkFile)
    newRad = np.loadtxt(dirName + os.sep + 'augmented/augmentedRad.dat')
    newPos = np.loadtxt(dirName + os.sep + 'augmented/augmentedPos.dat')
    simplices = np.array(np.loadtxt(dirName + os.sep + 'augmented/simplices.dat'), dtype=int)
    # first plot packing with dense / dilute particle labels
    colorId = getDenseColorList(denseList)
    for particleId in range(rad.shape[0]):
        x = newPos[particleId,0]
        y = newPos[particleId,1]
        r = newRad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=0.5, linewidth=0.3))
    # then plot labels on simplices
    if(filter == 'filter'):
        labelList = np.loadtxt(dirName + os.sep + 'augmented/filterDelaunayLabels/' + label + '.dat')
        allNeighborList = np.loadtxt(dirName + os.sep + 'augmented/filterDelaunayLabels/' + label + 'AllNeighbors.dat')
        neighborList = np.loadtxt(dirName + os.sep + 'augmented/filterDelaunayLabels/' + label + 'Neighbors.dat')
    else:
        labelList = np.loadtxt(dirName + os.sep + 'augmented/delaunayLabels/' + label + '.dat')
        allNeighborList = np.loadtxt(dirName + os.sep + 'augmented/delaunayLabels/' + label + 'AllNeighbors.dat')
        neighborList = np.loadtxt(dirName + os.sep + 'augmented/delaunayLabels/' + label + 'Neighbors.dat')
    plt.tripcolor(newPos[:,0], newPos[:,1], simplices[labelList==1], lw=0.3, facecolors=labelList[labelList==1], edgecolors='k', alpha=1)
    plt.tripcolor(newPos[:,0], newPos[:,1], simplices[neighborList==1], lw=0.3, facecolors=neighborList[neighborList==1], edgecolors='k', alpha=0.5)
    plt.tripcolor(newPos[:,0], newPos[:,1], simplices[allNeighborList==1], lw=0.3, facecolors=allNeighborList[allNeighborList==1], edgecolors='k', alpha=0.2)
    plt.tight_layout()
    if(filter == 'filter'):
        figureName = '/home/francesco/Pictures/soft/packings/filter2Labels-' + label + '-' + figureName + '.png'
    else:
        figureName = '/home/francesco/Pictures/soft/packings/labels-' + label + '-' + figureName + '.png'
    plt.savefig(figureName, transparent=False, format = 'png')
    plt.show()

def plotSPDelaunayParticleClusters(dirName, figureName, threshold=0.62, compute=False, alpha=0.7, paused=False, lj=False):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + 'boxSize.dat')
    rad = np.array(np.loadtxt(dirName + sep + 'particleRad.dat'))
    if lj:
        rad *= 2**(1/6)
    eps = 1.8*np.max(rad)
    pos = utils.getPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize)
    labels, maxLabel = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold=threshold, compute=compute)
    colorId = getColorListFromLabels(labels)
    print('maxLabel:', maxLabel, 'number of particles in biggest cluster:', labels[labels==maxLabel].shape[0])
    #pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
    # make figure
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    setPackingAxes(boxSize, ax)
    #setBigBoxAxes(boxSize, ax, 2)
    if(paused=='paused'):
        #i = 0
        for label in np.unique(labels):
            #print(label)
            for particleId in np.argwhere(labels==label)[:,0]:
                x = pos[particleId,0]
                y = pos[particleId,1]
                r = rad[particleId]
                ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=0.3))
            plt.tight_layout()
            #if(label!=-1):
            #    print(label, np.mean(pos[np.argwhere(labels==label)], axis=0), 'computed:', dropletPos[i])
            #    i += 1
            plt.pause(0.5)
    else:
        for particleId in range(rad.shape[0]):
            x = pos[particleId,0]
            y = pos[particleId,1]
            r = rad[particleId]
            ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=0.3))
            if(labels[particleId]==maxLabel):
                ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=alpha, linewidth=0.3))
    figureName = '/home/francesco/Pictures/soft/packings/clusters-' + figureName + '.png'
    plt.tight_layout()
    plt.savefig(figureName, transparent=True, format = 'png')
    plt.show()
    #plt.pause(0.5)

def getColorListFromSimplexLabels(labels):
    numLabels = np.unique(labels).shape[0]-1
    colorId = np.zeros(labels.shape[0])
    colors = utils.getUniqueRandomList(0, np.max(labels), numLabels)
    for label in range(numLabels):
        colorId[labels==label] = colors[label]
    return colorId

def plotSPDelaunaySimplexClusters(dirName, figureName, threshold=0.76, filter='filter', alpha=0.7, paused=False, shiftx=0, shifty=0):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + 'boxSize.dat')
    rad = np.array(np.loadtxt(dirName + sep + 'particleRad.dat'))
    eps = np.max(rad)
    pos = utils.getPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize)
    pos = utils.shiftPositions(pos, boxSize, shiftx, shifty) # for 4k and 16k, 0, -0.2 for 8k -0.4 0.05
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    setPackingAxes(boxSize, ax)
    dirAugment = dirName + os.sep + 'augmented'
    if not(os.path.exists(dirAugment + '!')):
        cluster.computeAugmentedDelaunayCluster(dirName, threshold, filter, shiftx, shifty)
    newPos = np.loadtxt(dirAugment + os.sep + 'augmentedPos.dat')
    newRad = np.loadtxt(dirAugment + os.sep + 'augmentedRad.dat')
    simplices = np.loadtxt(dirAugment + os.sep + 'simplices.dat').astype(np.int64)
    denseSimplexList = np.loadtxt(dirAugment + os.sep + 'denseSimplexList-filter.dat')
    # compute simplex positions for clustering algorithm
    if not(os.path.exists(dirAugment + os.sep + 'simplexLabels.dat')):
        simplexPos = utils.computeSimplexPos(simplices, newPos)
        labels = utils.getDBClusterLabels(simplexPos, boxSize*1.1, eps, min_samples=1, denseList=denseSimplexList)
        labels = labels + np.ones(labels.shape[0])
        allLabels = -1*np.ones(denseSimplexList.shape[0])
        allLabels[denseSimplexList==1] = labels
        labels = allLabels.astype(np.int64)
        np.savetxt(dirAugment + os.sep + 'simplexLabels.dat', labels)
    else:
        labels = np.loadtxt(dirAugment + os.sep + 'simplexLabels.dat').astype(np.int64)
    #print(np.unique(labels))
    colorId = getColorListFromSimplexLabels(labels)
    # plot particles
    for particleId in range(newRad.shape[0]):
        x = newPos[particleId,0]
        y = newPos[particleId,1]
        r = newRad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='none', alpha=alpha, linewidth=0.3))
    # plot simplex clusters
    if(paused=='paused'):
        for label in np.unique(labels):
            if(label!=-1):
                plt.tripcolor(newPos[:,0], newPos[:,1], simplices[labels==label], facecolors=labels[labels==label], lw=0.3, edgecolors='k', alpha=alpha, cmap='tab20')
                plt.tight_layout()
                plt.pause(0.5)
    else:
        plt.tripcolor(newPos[:,0], newPos[:,1], simplices[labels!=-1], facecolors=colorId[labels!=-1], lw=0.2, edgecolors='k', alpha=0.9, cmap='tab20c')
    figureName = '/home/francesco/Pictures/soft/packings/simplexClusters-' + figureName + '.png'
    plt.tight_layout()
    plt.savefig(figureName, transparent=False, format = 'png')
    plt.show()

def plotSoftParticles(ax, pos, rad, colorMap = True, alpha = 0.6, lw = 0.3):
    colorId = np.zeros((rad.shape[0], 4))
    if(colorMap == True):
        colorList = cm.get_cmap('viridis', rad.shape[0])
    else:
        colorList = cm.get_cmap('Reds', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=lw))

def plotSoftParticlesSubSet(axFrame, pos, rad, maxIndex, alpha = 0.6, lw = 0.3):
    colorId = np.zeros((rad.shape[0], 4))
    colorList = cm.get_cmap('viridis', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=lw))
        if(particleId < maxIndex):
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=alpha, linewidth=lw))

def plotSoftParticleQuiverVel(axFrame, pos, vel, rad, tagList = np.array([]), alpha = 0.6, lw = 0.3):
    colorId = np.zeros((rad.shape[0], 4))
    colorList = cm.get_cmap('viridis', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    if(tagList.size > 0):
        color = np.array(['g', 'b', 'k'])
        d = 0
        for particleId in range(tagList.shape[0]):
            if(tagList[particleId]==1):
                x = pos[particleId,0]
                y = pos[particleId,1]
                vx = vel[particleId,0]
                vy = vel[particleId,1]
                axFrame.quiver(x, y, vx, vy, facecolor=color[d], width=0.008, minshaft=3, scale=3, headwidth=5)
                d += 1
    else:
        speed = np.mean(np.linalg.norm(vel, axis=1))
        vel /= speed
        for particleId in range(pos.shape[0]):
            x = pos[particleId,0]
            y = pos[particleId,1]
            r = rad[particleId]
            vx = vel[particleId,0]
            vy = vel[particleId,1]
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor=colorId[particleId], facecolor='none', alpha=alpha, linewidth=lw))
            axFrame.quiver(x, y, vx, vy, facecolor=[0.2,0.4,1], edgecolor='k', linewidth=0.1, width=0.001, scale=80, headlength=4, headaxislength=4, headwidth=4, alpha=0.6)#width=0.003, scale=1, headwidth=5)

def plotSoftParticleCircleTangentVel(axFrame, pos, vel, rad, tagList = np.array([]), alpha = 0.6, lw = 0.3):
    colorId = np.zeros((rad.shape[0], 4))
    colorList = cm.get_cmap('viridis', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    speed = np.mean(np.linalg.norm(vel, axis=1))
    vel /= speed
    vel, angle = utils.calcTangentialVelocity(pos, vel)
    velColorId = np.zeros((angle.shape[0], 3))
    blue = [0.2,0.2,1]
    red = [1,0,0]
    for particleId in range(angle.shape[0]):
        if(angle[particleId] > 0 and angle[particleId] < np.pi * 0.5):
            if(vel[particleId,0] > 0 and vel[particleId,1] < 0):
                velColorId[particleId] = blue
            else:
                velColorId[particleId] = red
        if(angle[particleId] > np.pi * 0.5 and angle[particleId] < np.pi):
            if(vel[particleId,0] > 0 and vel[particleId,1] > 0):
                velColorId[particleId] = blue
            else:
                velColorId[particleId] = red
        if(angle[particleId] > -np.pi and angle[particleId] < -np.pi * 0.5):
            if(vel[particleId,0] < 0 and vel[particleId,1] > 0):
                velColorId[particleId] = blue
            else:
                velColorId[particleId] = red
        if(angle[particleId] > -np.pi * 0.5 and angle[particleId] < 0):
            if(vel[particleId,0] < 0 and vel[particleId,1] < 0):
                velColorId[particleId] = blue
            else:
                velColorId[particleId] = red
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        vx = vel[particleId,0]
        vy = vel[particleId,1]
        axFrame.add_artist(plt.Circle([x, y], r, edgecolor=colorId[particleId], facecolor='none', alpha=alpha, linewidth=lw))
        axFrame.quiver(x, y, vx, vy, facecolor=velColorId[particleId], edgecolor='k', linewidth=0.2, width=0.0012, scale=60, headlength=5, headaxislength=5, headwidth=5, alpha=0.7)#width=0.003, scale=1, headwidth=5)


def plotSoftParticleStressMap(axFrame, pos, stress, rad, potential='lj', alpha = 0.6, lw = 0.3):
    colorId = np.zeros((rad.shape[0], 4))
    colorList = cm.get_cmap('bwr', rad.shape[0])
    count = 0
    if(potential=='lj' or potential=='ra'):
        p = np.sum(stress[:,0:2], axis=1)
    else:
        p = np.sum(stress[:,0:3], axis=1)
    for particleId in np.argsort(p):
        colorId[particleId] = colorList(count/p.shape[0])
        count += 1
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=lw))

def plotSoftParticleShearStressMap(axFrame, pos, stress, rad, potential='lj', alpha = 0.6, lw = 0.3):
    colorId = np.zeros((rad.shape[0], 4))
    colorList = cm.get_cmap('bwr', rad.shape[0])
    count = 0
    if(potential=='lj' or potential=='ra'):
        p = stress[:,3]
    else:
        p = stress[:,4]
    for particleId in np.argsort(p):
        colorId[particleId] = colorList(count/p.shape[0])
        count += 1
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=lw))

def plotSoftParticleCluster(axFrame, pos, rad, denseList, alpha = 0.6, lw = 0.3):
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        if(denseList[particleId] == 1):
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=alpha, linewidth=lw))
        else:
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=[1,1,1], alpha=alpha, linewidth=lw))

def plotSoftParticlePerturb(axFrame, pos, rad, movedLabel, alpha = 0.6, lw = 0.3):
    colorId = np.zeros((rad.shape[0], 4))
    colorList = cm.get_cmap('viridis', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        if(movedLabel[particleId] == 1):
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=alpha, linewidth=lw))
        else:
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=lw))

def plotSoftParticleDouble(ax, pos, rad, num1, tag=False, alpha = 0.6, lw = 0.3):
    colorId = np.zeros((rad.shape[0], 4))
    colorId[:num1] = [0,1,0,0.2]
    colorId[num1:] = [0,0,1,0.2]
    alphaId = np.ones(colorId.shape[0])*alpha
    if(tag == True):
        tagList = np.linspace(500, 8000, 30).astype(int)
        colorId[tagList] = [0,0,0,0]
        alphaId[tagList] = 1
        tagList = np.linspace(8500, 16000, 30).astype(int)
        colorId[tagList] = [1,1,1,1]
        alphaId[tagList] = 1
    for i, (x, y) in enumerate(pos):
        circle = plt.Circle((x, y), rad[i], edgecolor='k', facecolor=colorId[i], alpha=alphaId[i], linewidth=lw)
        ax.add_patch(circle)

def makeSoftParticleClusterFrame(dirName, rad, boxSize, figFrame, frames, clusterList):
    pos = np.array(np.loadtxt(dirName + os.sep + 'particlePos.dat'))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setPackingAxes(boxSize, axFrame)
    plotSoftParticleCluster(axFrame, pos, rad, clusterList)
    figFrame.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def plotSoftParticlesWithAngles(axFrame, pos, vel, rad, colorMap, alpha = 0.6, lw = 0.3):
    angle = ((utils.getVelocityAngles(vel) + 2.*np.pi) % (2. * np.pi)) / (2*np.pi)
    # Create color array based on normalized angles
    colorId = colorMap(angle)
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        vx = vel[particleId,0]
        vy = vel[particleId,1]
        if(pos.shape[0] < 3e03):
            axFrame.quiver(x, y, vx, vy, facecolor='k', edgecolor='k', linewidth=0.1, width=0.001, scale=120, headlength=5, headaxislength=5, headwidth=5, alpha=0.5)#width=0.003, scale=1, headwidth=5)
        axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=lw))
        #axFrame.annotate(str(particleId), xy=(x, y), fontsize=4, verticalalignment='center', horizontalalignment='center')

def makeSPPackingClusterMixingVideo(dirName, figureName, numFrames = 20, firstStep = 0, stepFreq = 1e04):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    stepList = utils.getStepList(numFrames, firstStep, stepFreq)
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=200)
    fig = plt.figure(dpi=200)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + 'boxSize.dat')
    setPackingAxes(boxSize, ax)
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    if(os.path.exists(dirName + os.sep + 't' + str(stepList[0]) + '/denseList.dat')):
        denseList = np.loadtxt(dirName + os.sep + 't' + str(stepList[0]) + '/denseList.dat')
    # the first configuration gets two frames for better visualization
    makeSoftParticleClusterFrame(dirName + os.sep + 't' + str(stepList[0]), rad, boxSize, figFrame, frames, denseList)
    for i in stepList:
        dirSample = dirName + os.sep + 't' + str(i)
        makeSoftParticleClusterFrame(dirSample, rad, boxSize, figFrame, frames, denseList)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    anim.save('/home/francesco/Pictures/soft/packings/clustermix-' + figureName + '.gif', writer='imagemagick', dpi=plt.gcf().dpi)

def makeSoftParticleFrame(ax, dirName, rad, boxSize, angle=False, quiver=False, veltang=False, dense=False, 
                          perturb=False, pmap=False, potential=False, lcut=4, double=False, num1=0, colorMap=None):
    if boxSize.shape[0] == 1:
        pos = np.array(np.loadtxt(dirName + os.sep + 'particlePos.dat'))
        utils.checkParticlesInCircle(pos, boxSize)
    else:
        pos = utils.getPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize)
    #pos = np.loadtxt(dirName + os.sep + 'particlePos.dat')
    if double:
        plotSoftParticleDouble(ax, pos, rad, num1, tag=False)
    else:
        plotSoftParticles(ax, pos, rad)

    if angle:
        vel = np.array(np.loadtxt(dirName + os.sep + 'particleVel.dat'))
        plotSoftParticlesWithAngles(ax, pos, vel, rad, colorMap)

    if quiver:
        vel = np.array(np.loadtxt(dirName + os.sep + 'particleVel.dat'))
        if veltang == 'veltang':
            plotSoftParticleCircleTangentVel(ax, pos, vel, rad)
        else:
            plotSoftParticleQuiverVel(ax, pos, vel, rad)

    if dense:
        if not(os.path.exists(dirName + os.sep + 'particleList.dat')):
            cluster.computeDelaunayCluster(dirName)
        denseList = np.loadtxt(dirName + os.sep + 'particleList.dat')[:,0]
        plotSoftParticleCluster(ax, pos, rad, denseList)

    if perturb:
        movedLabel = np.loadtxt(dirName + '/../movedLabel.dat')
        plotSoftParticlePerturb(ax, pos, rad, movedLabel)

    if pmap:
        if not(os.path.exists(dirName + os.sep + 'particleStress.dat')):
            if(potential == 'lj'):
                stress = cluster.computeLJParticleStress(dirName, lcut)
            elif(potential == 'ra'):
                stress = cluster.computeRAParticleStress(dirName, lcut)
            else:
                stress = cluster.computeParticleStress(dirName)
        stress = np.loadtxt(dirName + os.sep + 'particleStress.dat')
        plotSoftParticleShearStressMap(ax, pos, stress, rad, potential)

def makeCircularColorBar(ax_cb, color_map):
    # Create an array of angles (theta) and radial distances (r) for the circular colorbar
    theta = np.linspace(0, 2*np.pi, 100)

    # Create a meshgrid for colors, where the color values are determined by theta
    theta_grid, r_grid = np.meshgrid(theta, [0,1])

    # Plot the circular colorbar using grid
    ax_cb.grid(False)
    c = ax_cb.pcolormesh(theta_grid, r_grid, theta_grid, cmap=color_map, shading='auto')

    # Remove radial ticks and labels
    ax_cb.set_yticklabels([])

    # Set the angular ticks and labels
    ax_cb.set_xticks([0, np.pi])
    ax_cb.set_xticklabels(['0', r'$\pi$'], fontsize=9)

    # Set aspect ratio to be equal for a circular look
    ax_cb.set_aspect(1)
    # Return the colorbar object if you want to modify or access it later
    return c

def makeSPPackingVideo(dirName, figureName, numFrames=20, firstStep=0, stepFreq=1e04, logSpaced=False, fixed=False, angle=False, quiver=False,
                       veltang=False, dense=False, perturb=False, pmap=False, potential=False, lcut=4, double=False, num1=0, lj=False, colorMap=None):
    def animate(i):
        ax.clear()  # Clear the previous frame
        if boxSize.shape[0] == 1:
            setBigBoxAxes(boxSize, ax, 1.05)
            fig.patch.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_artist(plt.Circle([0, 0], boxSize, edgecolor='k', facecolor=[1,1,1], linewidth=0.5)) 
        else:
            setPackingAxes(boxSize, ax)
        dirSample = dirName + os.sep + 't' + str(stepList[i])
        makeSoftParticleFrame(ax, dirSample, rad, boxSize, angle, quiver, veltang, dense, perturb, pmap, potential, lcut, double, num1, colorMap)
        plt.tight_layout()
        return ax.artists
    
    frameTime = 120
    if(logSpaced == False):
        stepList = utils.getStepList(numFrames, firstStep, stepFreq)
    else:
        print("entering log spaced")
        stepList = utils.getLogSpacedStepList(minDecade=5, maxDecade=9)
        numFrames = stepList.shape[0]
    print('Time list:', stepList)

    boxSize = np.atleast_1d(np.loadtxt(dirName + os.sep + 'boxSize.dat'))
    if(boxSize.shape[0] == 1):
        print("Setting circular box geometry")
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    if lj:
        rad *= 2**(1/6)

    # Initialize figure and axis
    fig, ax = plt.subplots(dpi=200)
    # Set background to transparent
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create a polar axis for the circular colorbar
    if angle:
        ax_cb = fig.add_axes([0.8, 0.6, 0.1, 0.6], polar=True)  # Position for the colorbar
        colorMap = cm.get_cmap('hsv')  # Set the color map
        makeCircularColorBar(ax_cb, colorMap)  # Create the colorbar once
    
    # Create animation
    numFrames = len(stepList) # One extra frame for the repeated first image
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    
    # Set figure background to transparent
    fig.patch.set_facecolor('none')

    # Save the animation
    anim.save(f'/home/francesco/Pictures/soft/packings/{figureName}.gif', writer='pillow', dpi=fig.dpi)
    #anim.save(f'/home/francesco/Pictures/soft/packings/{figureName}.mov', writer='ffmpeg', dpi=fig.dpi)

def makeSPCompressionVideo(dirName, figureName, quiver=False, fixed='fixed', lj=False):
    def animate(i):
        ax.clear()  # Clear the previous frame
        dirSample = dirName + os.sep + dirList[i]
        boxSize = np.atleast_1d(np.loadtxt(dirSample + os.sep + 'boxSize.dat'))
        if fixed == 'fixed':
            pos = np.array(np.loadtxt(dirSample + os.sep + 'particlePos.dat'))
        else:
            pos = utils.getPBCPositions(dirSample + os.sep + 'particlePos.dat', boxSize)
        rad = np.array(np.loadtxt(dirSample + os.sep + 'particleRad.dat'))
        if lj:
            rad *= 2**(1/6)
        if boxSize.shape[0] == 1:
            setBigBoxAxes(boxSize, ax, 1.05)
            fig.patch.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.add_artist(plt.Circle([0, 0], boxSize, edgecolor='k', facecolor=[1,1,1], linewidth=0.5)) 
        else:
            setPackingAxes(boxSize, ax)
        plotSoftParticles(ax, pos, rad)
        if quiver:
            vel = np.array(np.loadtxt(dirSample + os.sep + 'particleVel.dat'))
            plotSoftParticleQuiverVel(ax, pos, vel, rad)
        # Add title to the frame
        ax.set_title(f'$\\varphi=${dirList[i]}', fontsize=12)
        plt.tight_layout()
        return ax.artists
    
    frameTime = 350
    dirList, phiList = utils.getOrderedPhiDirectories(dirName)
    numFrames = dirList.shape[0]
    print('Phi list:', phiList)

    # Initialize figure and axis
    fig, ax = plt.subplots(dpi=200)
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)

    # Save the animation
    anim.save(f'/home/francesco/Pictures/soft/packings/comp-{figureName}.gif', writer='pillow', dpi=fig.dpi)
    #anim.save(f'/home/francesco/Pictures/soft/packings/comp-{figureName}.mov', writer='ffmpeg', dpi=fig.dpi)

def makeSPExtendPackingVideo(dirName, figureName, maxStrain = 0.0300, strainFreq = 2, which = 'extend', centered = 'centered', dynamics = 'nve', lj = False, double = False, num1 = 0):
    def animate(i):
        ax.clear()  # Clear the previous frame
        dirSample = dirName + os.sep + dirList[i]
        if(dynamics == 'nve'):
            dirSample += '/dynamics'
        elif(dynamics == '1e-15'):
            dirSample += '/damping1e-15'
        elif(dynamics == '1e-12'):
            dirSample += '/damping1e-12'
        elif(dynamics == '1e-10'):
            dirSample += '/damping1e-10'
        elif(dynamics == '1e-08'):
            dirSample += '/damping1e-08'
        elif(dynamics == '1e-05'):
            dirSample += '/damping1e-05'
        elif(dynamics == '1e-03'):
            dirSample += '/damping1e-03'
        elif(dynamics == '1e-01'):
            dirSample += '/damping1e-01'
        elif(dynamics == '1e01'):
            dirSample += '/damping1e01'
        boxSize = np.loadtxt(dirSample + os.sep + 'boxSize.dat')
        pos = utils.getPBCPositions(dirSample + os.sep + 'particlePos.dat', boxSize)
        if centered == 'centered':
            setCenteredPackingAxes(boxSize, frameSize, ax)
        else:
            setAutoPackingAxes(frameSize, ax)
        if double:
            plotSoftParticleDouble(ax, pos, rad, num1, tag=False)
        else:
            plotSoftParticles(ax, pos, rad)
        # Add title to the frame
        ax.set_title(f'Strain: {strainList[i]:.4f}', fontsize=12)
        plt.tight_layout()
        return ax.artists
    
    frameTime = 350
    if(which == 'extend' or which == 'compress'):
        dirList, strainList = utils.getOrderedStrainDirectories(dirName)
    elif(which == 'ext-rev' or which == 'comp-rev'):
        dirList, strainList = utils.getFrontBackStrainDirectories(dirName)
    if(dynamics != 0):
        dirList = dirList[1:]
        strainList = strainList[1:]
    dirList = dirList[strainList < maxStrain]
    strainList = strainList[strainList < maxStrain]
    dirList = dirList[::strainFreq]
    strainList = strainList[::strainFreq]
    print('Strain list:', strainList)
    frameSize = np.zeros(2)
    if which == 'extend':
        frameSize[1] = np.loadtxt(dirName + dirList[-1] + os.sep + 'boxSize.dat')[1]
        frameSize[0] = np.loadtxt(dirName + dirList[0] + os.sep + 'boxSize.dat')[0]
    elif which == 'compress':
        frameSize[1] = np.loadtxt(dirName + dirList[0] + os.sep + 'boxSize.dat')[1]
        frameSize[0] = np.loadtxt(dirName + dirList[-1] + os.sep + 'boxSize.dat')[0]
    elif which == 'ext-rev':
        frameSize[1] = 1.05*np.loadtxt(dirName + dirList[int(dirList.shape[0] / 2)] + os.sep + 'boxSize.dat')[1]
        frameSize[0] = 1.05*np.loadtxt(dirName + dirList[0] + os.sep + 'boxSize.dat')[0]
    elif which == 'comp-rev':
        frameSize[1] = 1.05*np.loadtxt(dirName + dirList[0] + os.sep + 'boxSize.dat')[1]
        frameSize[0] = 1.05*np.loadtxt(dirName + dirList[int(dirList.shape[0] / 2)] + os.sep + 'boxSize.dat')[0]
    else:
        frameSize = np.loadtxt(dirName + os.sep + 'boxSize.dat')
    print('Frame size:', frameSize, 'initial boxSize:', np.loadtxt(dirName + dirList[0] + os.sep + 'boxSize.dat'))

    # Load initial data
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    if lj:
        rad *= 2**(1/6)

    # Initialize figure and axis
    fig, ax = plt.subplots(dpi=200)
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create animation
    numFrames = len(dirList) # One extra frame for the repeated first image
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)

    # Save the animation
    anim.save(f'/home/francesco/Pictures/soft/packings/{which}-{figureName}.gif', writer='pillow', dpi=fig.dpi)
    #anim.save(f'/home/francesco/Pictures/soft/packings/{which}-{figureName}.mov', writer='ffmpeg', dpi=fig.dpi)

def makeSPShearPackingVideo(dirName, figureName, maxStrain = 0.0300, strainFreq = 2, lj = False):
    def animate(i):
        ax.clear()  # Clear the previous frame
        setPackingAxes(boxSize, ax)
        dirSample = dirName + os.sep + dirList[i]
        pos = utils.getLEPBCPositions(dirName + os.sep + 'particlePos.dat', boxSize, strainList[i])
        plotSoftParticles(ax, pos, rad)
        plt.tight_layout()
        return ax.artists

    frameTime = 350
    dirList, strainList = utils.getOrderedStrainDirectories(dirName)
    dirList = dirList[strainList < maxStrain]
    dirList = dirList[::strainFreq]
    strainList = strainList[::strainFreq]
    print('Strain list:', strainList)

    # Load initial data
    boxSize = np.loadtxt(dirName + os.sep + 'boxSize.dat')
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    if lj:
        rad *= 2**(1/6)

    # Initialize figure and axis
    fig, ax = plt.subplots(dpi=200)
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Create animation
    numFrames = len(dirList) # One extra frame for the repeated first image
    anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)

    # Save the animation
    anim.save(f'/home/francesco/Pictures/soft/packings/shear-{figureName}.gif', writer='imagemagick', dpi=fig.dpi)
    #anim.save(f'/home/francesco/Pictures/soft/packings/shear-{figureName}.mov', writer='ffmpeg', dpi=fig.dpi)

def plotSoftParticleDroplet(axFrame, pos, rad, labels, maxLabel, alpha = 0.7):
    colorList = getColorListFromLabels(labels)
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        if(labels[particleId]==maxLabel):
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=alpha, linewidth=0.3))
        else:
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorList[particleId], alpha=alpha, linewidth=0.3))

def makeSoftParticleDropletFrame(pos, rad, boxSize, figFrame, frames, labels, maxLabel):
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setPackingAxes(boxSize, axFrame)
    plotSoftParticleDroplet(axFrame, pos, rad, labels, maxLabel)
    figFrame.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def makeSPPackingDropletVideo(dirName, figureName, numFrames = 20, firstStep = 0, stepFreq = 1e04, threshold=0.78, lj=False, shiftx=0, shifty=0):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    stepList = utils.getStepList(numFrames, firstStep, stepFreq)
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=200)
    fig = plt.figure(dpi=200)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + 'boxSize.dat')
    setPackingAxes(boxSize, ax)
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    eps = 1.8*np.max(rad)
    if lj:
        rad *= 2**(1/6)
    pos = utils.getPBCPositions(dirName + os.sep + 't' + str(stepList[0]) + '/particlePos.dat', boxSize)
    pos = utils.shiftPositions(pos, boxSize, shiftx, shifty)
    labels, maxLabel = cluster.getParticleClusterLabels(dirName + os.sep + 't' + str(stepList[0]), boxSize, eps, threshold=threshold)
    makeSoftParticleDropletFrame(pos, rad, boxSize, figFrame, frames, labels, maxLabel)
    for i in stepList:
        dirSample = dirName + os.sep + 't' + str(i)
        pos = utils.getPBCPositions(dirSample + os.sep + 'particlePos.dat', boxSize)
        pos = utils.shiftPositions(pos, boxSize, shiftx, shifty)
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold=threshold)
        makeSoftParticleDropletFrame(pos, rad, boxSize, figFrame, frames, labels, maxLabel)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    anim.save('/home/francesco/Pictures/soft/packings/droplet-' + figureName + '.gif', writer='imagemagick', dpi=plt.gcf().dpi)

def makeVelFieldFrame(dirName, numBins, bins, boxSize, numParticles, figFrame, frames):
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setGridAxes(bins, axFrame)
    grid, field = cluster.computeVelocityField(dirName, numBins, plot=False, boxSize=boxSize, numParticles=numParticles)
    axFrame.quiver(grid[:,0], grid[:,1], field[:,0], field[:,1], facecolor='k', width=0.002, scale=3)
    figFrame.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def makeSPVelFieldVideo(dirName, figureName, numFrames = 20, firstStep = 0, stepFreq = 1e04, numBins=20):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    _, stepList = utils.getOrderedDirectories(dirName)
    #timeList = timeList.astype(int)
    stepList = stepList[np.argwhere(stepList%stepFreq==0)[:,0]]
    stepList = stepList[:numFrames]
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=200)
    fig = plt.figure(dpi=200)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + 'boxSize.dat')
    bins = np.linspace(-0.5*boxSize[0],0, numBins)
    bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
    bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
    setGridAxes(bins, ax)
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    numParticles = int(utils.readFromParams(dirName, 'numParticles'))
    # the first configuration gets two frames for better visualization
    makeVelFieldFrame(dirName + os.sep + 't0', numBins, bins, boxSize, numParticles, figFrame, frames)
    vel = []
    for i in stepList:
        dirSample = dirName + os.sep + 't' + str(i)
        makeVelFieldFrame(dirSample, numBins, bins, boxSize, numParticles, figFrame, frames)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    anim.save('/home/francesco/Pictures/soft/packings/velfield-' + figureName + '.gif', writer='imagemagick', dpi=plt.gcf().dpi)

def makeInterfaceVideo(dirName, figureName, numFrames = 20, firstStep = 0, stepFreq = 1e04):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    stepList = utils.getStepList(numFrames, firstStep, stepFreq)
    # interface vertical coordinates
    boxSize = np.array(np.loadtxt(dirName + os.sep + 'boxSize.dat'))
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    spacing = 3*np.mean(rad)
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    #frame figure
    figFrame = plt.figure(dpi=200)
    fig = plt.figure(dpi=200)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    heightvstime = np.loadtxt(dirName + os.sep + 'heightVStime.dat')
    for i in range(heightvstime.shape[0]):
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        axFrame.plot(heightvstime[i], centers, lw=1, marker='o', color='k')
        axFrame.tick_params(axis='both', labelsize=14)
        axFrame.set_xlabel('$Height$', fontsize=16)
        axFrame.set_ylim(0,1)
        axFrame.set_xlim(boxSize[0]*0.8, boxSize[0]*0.5)
        axFrame.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        axFrame.set_xticks((boxSize[0]*0.5, boxSize[0]*0.6, boxSize[0]*0.7, boxSize[0]*0.8))
        figFrame.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save('/home/francesco/Pictures/soft/mips/' + figureName + '.gif', writer='imagemagick', dpi=plt.gcf().dpi)


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]
    figureName = sys.argv[3]

    if(whichPlot == 'sp'):
        plotSPPacking(dirName, figureName, shiftx=float(sys.argv[4]), shifty=float(sys.argv[5]))

    elif(whichPlot == 'lj'):
        plotSPPacking(dirName, figureName, lj=True, shiftx=float(sys.argv[4]), shifty=float(sys.argv[5]), center=sys.argv[6])

    elif(whichPlot == 'ljvel'):
        plotSPPacking(dirName, figureName, lj=True, quiver=True, shiftx=float(sys.argv[4]), shifty=float(sys.argv[5]), center=sys.argv[6])

    elif(whichPlot == 'wall'):
        plotMobilePacking(dirName, figureName, lj=True, colorAngle=False)

    elif(whichPlot == '2lj'):
        plotSPPacking(dirName, figureName, lj=True, shiftx=float(sys.argv[4]), shifty=float(sys.argv[5]), center=sys.argv[6], double=True, num1=int(sys.argv[7]))

    elif(whichPlot == 'spshear'):
        plotSPPacking(dirName, figureName, shear=True, strain=float(sys.argv[4]), shiftx=float(sys.argv[5]), shifty=float(sys.argv[6]))

    elif(whichPlot == 'ljshear'):
        plotSPPacking(dirName, figureName, lj=True, shear=True, strain=float(sys.argv[4]), shiftx=float(sys.argv[5]), shifty=float(sys.argv[6]))

    elif(whichPlot == 'sp3d'):
        plot3DPacking(dirName, figureName)

    elif(whichPlot == 'spvel'):
        plotSPPacking(dirName, figureName, quiver=True)

    elif(whichPlot == 'spdense'):
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        plotSPPacking(dirName, figureName, dense=True, threshold=threshold, filter=filter, shiftx=float(sys.argv[6]), shifty=float(sys.argv[7]))

    elif(whichPlot == 'ljdense'):
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        plotSPPacking(dirName, figureName, dense=True, threshold=threshold, filter=filter, shiftx=float(sys.argv[6]), shifty=float(sys.argv[7]), lj=True)

    elif(whichPlot == 'spborder'):
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        plotSPPacking(dirName, figureName, border=True, threshold=threshold, filter=filter, shiftx=float(sys.argv[6]), shifty=float(sys.argv[7]))

    elif(whichPlot == 'ljborder'):
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        plotSPPacking(dirName, figureName, border=True, threshold=threshold, filter=filter, shiftx=float(sys.argv[6]), shifty=float(sys.argv[7]), lj=True)

    elif(whichPlot == 'spekin'):
        alpha = float(sys.argv[4])
        plotSPPacking(dirName, figureName, ekmap=True, alpha=alpha)

    elif(whichPlot == 'ljforce'):
        alpha = float(sys.argv[4])
        plotSPPacking(dirName, figureName, forcemap=True, alpha=alpha, lj=True)

    elif(whichPlot == 'stress'):
        which = sys.argv[4]
        potential = sys.argv[5]
        lcut = float(sys.argv[6])
        plotSPStressMapPacking(dirName, figureName, which, potential, lcut, shiftx=float(sys.argv[7]), shifty=float(sys.argv[8]))

    elif(whichPlot == 'spdel'):
        plotSPDelaunayPacking(dirName, figureName, shiftx=float(sys.argv[4]), shifty=float(sys.argv[5]))

    elif(whichPlot == 'ljdel'):
        plotSPDelaunayPacking(dirName, figureName, shiftx=float(sys.argv[4]), shifty=float(sys.argv[5]), lj=True)

    elif(whichPlot == 'spdeldense'):
        np.seterr(divide='ignore', invalid='ignore')
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        colored = sys.argv[6]
        plotSPDelaunayPacking(dirName, figureName, dense=True, threshold=threshold, filter=filter, colored=colored, shiftx=float(sys.argv[7]), shifty=float(sys.argv[8]))

    elif(whichPlot == 'ljdeldense'):
        np.seterr(divide='ignore', invalid='ignore')
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        colored = sys.argv[6]
        plotSPDelaunayPacking(dirName, figureName, dense=True, threshold=threshold, filter=filter, colored=colored, shiftx=float(sys.argv[7]), shifty=float(sys.argv[8]), lj=True)

    elif(whichPlot == 'ljdelborder'):
        np.seterr(divide='ignore', invalid='ignore')
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        colored = sys.argv[6]
        plotSPDelaunayPacking(dirName, figureName, border=True, threshold=threshold, filter=filter, colored=colored, shiftx=float(sys.argv[7]), shifty=float(sys.argv[8]), lj=True)

    elif(whichPlot == 'spdelborder'):
        np.seterr(divide='ignore', invalid='ignore')
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        colored = sys.argv[6]
        plotSPDelaunayPacking(dirName, figureName, border=True, threshold=threshold, filter=filter, colored=colored, shiftx=float(sys.argv[7]), shifty=float(sys.argv[8]))

    elif(whichPlot == 'spdellabel'):
        np.seterr(divide='ignore', invalid='ignore')
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        label = sys.argv[6]
        plotSPDelaunayLabels(dirName, figureName, dense=False, threshold=threshold, filter=filter, label=label)

    elif(whichPlot == 'spdelparticle'):
        np.seterr(divide='ignore', invalid='ignore')
        threshold = float(sys.argv[4])
        compute = sys.argv[5]
        if(compute == 'compute'):
            compute = True
        paused = sys.argv[6]
        plotSPDelaunayParticleClusters(dirName, figureName, threshold=threshold, compute=compute, paused=paused)

    elif(whichPlot == 'ljdelparticle'):
        np.seterr(divide='ignore', invalid='ignore')
        threshold = float(sys.argv[4])
        compute = sys.argv[5]
        if(compute == 'compute'):
            compute = True
        paused = sys.argv[6]
        plotSPDelaunayParticleClusters(dirName, figureName, threshold=threshold, compute=compute, paused=paused, lj=True)

    elif(whichPlot == 'spdelsimplex'):
        np.seterr(divide='ignore', invalid='ignore')
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        paused = sys.argv[6]
        plotSPDelaunaySimplexClusters(dirName, figureName, threshold=threshold, filter=filter, paused=paused, shiftx=float(sys.argv[7]), shifty=float(sys.argv[8]))

################################################################################################################################
########################################################## VIDEOS ##############################################################
################################################################################################################################
    elif(whichPlot == 'spvideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        fixed = sys.argv[7]
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, fixed=fixed)

    elif(whichPlot == 'spvelvideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        fixed = sys.argv[7]
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, fixed=fixed, quiver=True)

    elif(whichPlot == 'spcompvideo'):
        fixed = sys.argv[4]
        makeSPCompressionVideo(dirName, figureName, fixed)

    elif(whichPlot == 'ljvideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        fixed = sys.argv[7]
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, fixed=fixed, lj=True)

    elif(whichPlot == 'ljvelvideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        fixed = sys.argv[7]
        veltang = sys.argv[8]
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, fixed=fixed, quiver=True, veltang=veltang, lj=True)

    elif(whichPlot == 'ljanglevideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        fixed = sys.argv[7]
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, fixed=fixed, angle=True, lj=True)

    elif(whichPlot == '2ljvideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, lj=True, double=True, num1=int(sys.argv[7]), logSpaced=False)

    elif(whichPlot == 'spperturb'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, perturb=True)

    elif(whichPlot == 'ljperturb'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, lj=True, perturb=True)

    elif(whichPlot == 'spstressvideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, pmap=True)

    elif(whichPlot == 'ljstressvideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        lcut = float(sys.argv[7])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, pmap=True, potential='lj', lcut=lcut)

    elif(whichPlot == 'clustervideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, dense=True)

    elif(whichPlot == 'shearvideo'):
        maxStrain = float(sys.argv[4])
        strainFreq = int(sys.argv[5])
        makeSPShearPackingVideo(dirName, figureName, maxStrain, strainFreq)

    elif(whichPlot == 'shearljvideo'):
        maxStrain = float(sys.argv[4])
        strainFreq = int(sys.argv[5])
        makeSPShearPackingVideo(dirName, figureName, maxStrain, strainFreq, lj=True)

    elif(whichPlot == 'extendljvideo'):
        maxStrain = float(sys.argv[4])
        strainFreq = int(sys.argv[5])
        which = sys.argv[6]
        centered = sys.argv[7]
        dynamics = sys.argv[8]
        makeSPExtendPackingVideo(dirName, figureName, maxStrain, strainFreq, which, centered=centered, dynamics=dynamics, lj=True)

    elif(whichPlot == 'extend2ljvideo'):
        maxStrain = float(sys.argv[4])
        strainFreq = int(sys.argv[5])
        which = sys.argv[6]
        centered = sys.argv[7]
        dynamics = sys.argv[8]
        num1 = int(sys.argv[9])
        makeSPExtendPackingVideo(dirName, figureName, maxStrain, strainFreq, which, centered=centered, dynamics=dynamics, lj=True, double=True, num1=num1)

    elif(whichPlot == 'velfield'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        numBins = int(sys.argv[7])
        makeSPVelFieldVideo(dirName, figureName, numFrames, firstStep, stepFreq, numBins=numBins)

    elif(whichPlot == 'dropletvideo'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        threshold = float(sys.argv[7])
        makeSPPackingDropletVideo(dirName, figureName, numFrames, firstStep, stepFreq, shiftx=float(sys.argv[8]), shifty=float(sys.argv[9]))

    elif(whichPlot == 'clustermix'):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingClusterMixingVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == 'interfacevideo'):
        figureName = sys.argv[3]
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeInterfaceVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    else:
        print('Please specify the type of plot you want')
