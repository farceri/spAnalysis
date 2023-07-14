'''
Created by Francesco
12 October 2021
'''
#functions and script to visualize a 2d dpm packing
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import animation
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyvoro
import itertools
import sys
import os
import shapeDescriptors
import shapeGraphics
import spCorrelation as spCorr
import utilsCorr as ucorr
import utilsPlot as uplot


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
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    zBounds = np.array([0, boxSize[2]])
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    pos[:,2] -= np.floor(pos[:,2]/boxSize[2]) * boxSize[2]
    fig = plt.figure(dpi=100)
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
    plt.savefig("/home/francesco/Pictures/soft/packings/3d-" + figureName + ".png", transparent=True, format = "png")
    plt.show()

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

def setGridAxes(bins, ax):
    xBounds = np.array([bins[0], bins[-1]])
    yBounds = np.array([bins[0], bins[-1]])
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setAxes2D(ax)

def setBigBoxAxes(boxSize, ax, delta=0.1):
    xBounds = np.array([-delta, boxSize[0]+delta])
    yBounds = np.array([-delta, boxSize[1]+delta])
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
        if(labels[particleId]==-1): # particles not in a cluster
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

def plotSPPacking(dirName, figureName, ekmap=False, quiver=False, dense=False, border=False, threshold=0.65, filter='filter', alpha = 0.6):
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    #pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    #denseList = np.loadtxt(dirName + os.sep + "denseList.dat")
    pos = ucorr.shiftPositions(pos, boxSize, 0, 0.2)
    #pos = ucorr.shiftPositions(pos, boxSize, 0.6, -0.35)
    pos = ucorr.centerPositions(pos, rad, boxSize)
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setPackingAxes(boxSize, ax)
    #setBigBoxAxes(boxSize, ax, 0.05)
    if(dense==True):
        if(os.path.exists(dirName + os.sep + "delaunayList!.dat")):
            denseList = np.loadtxt(dirName + os.sep + "delaunayList.dat")
        else:
            denseList,_ = spCorr.computeDelaunayCluster(dirName, threshold, filter=filter)
        #if(os.path.exists(dirName + os.sep + "denseList!.dat")):
        #    denseList = np.loadtxt(dirName + os.sep + "denseList.dat")
        #else:
        #    denseList,_ = spCorr.computeVoronoiCluster(dirName, threshold, filter=filter)
        colorId = getDenseColorList(denseList)
    elif(border==True):
        if(os.path.exists(dirName + os.sep + "delaunayBorderList!.dat")):
            borderList = np.loadtxt(dirName + os.sep + "delaunayBorderList.dat")
        else:
            borderList,_ = spCorr.computeDelaunayBorder(dirName, threshold, filter=filter)
        #if(os.path.exists(dirName + os.sep + "borderList!.dat")):
        #    borderList = np.loadtxt(dirName + os.sep + "borderList!.dat")
        #else:
        #    borderList,_ = spCorr.computeVoronoiBorder(dirName, threshold, filter=filter)
        colorId = getDenseColorList(borderList)
    elif(ekmap==True):
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        ekin = 0.5*np.linalg.norm(vel, axis=1)**2
        colorId = getEkinColorList(ekin)
    else:
        colorId = getRadColorList(rad)
    if(quiver==True):
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        #vel *= 5
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        if(quiver==True):
            ax.add_artist(plt.Circle([x, y], r, edgecolor=colorId[particleId], facecolor='none', alpha=alpha, linewidth = 0.7))
            vx = vel[particleId,0]
            vy = vel[particleId,1]
            ax.quiver(x, y, vx, vy, facecolor='k', width=0.002, scale=10)#width=0.002, scale=3)20
        else:
            ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth='0.3'))
    if(border==True):
        plt.tight_layout()
        plt.pause(1)
        borderPos = np.loadtxt(dirName + os.sep + "borderPos.dat")
        for particleId in range(1,borderPos.shape[0]):
            ax.plot(borderPos[particleId,0], borderPos[particleId,1], marker='*', markeredgecolor='k', color=[0.5,0.5,1], markersize=12, markeredgewidth=0.5)
            #slope = (borderPos[particleId,1] - borderPos[particleId-1,1]) / (borderPos[particleId,0] - borderPos[particleId-1,0])
            #intercept = borderPos[particleId-1,1] - borderPos[particleId-1,0] * slope
            #x = np.linspace(borderPos[particleId-1,0], borderPos[particleId,0])
            #ax.plot(x, slope*x+intercept, lw=0.7, ls='dashed', color='r')
            #plt.pause(0.05)
    #for particleId in range(rad.shape[0]):
    #    x = pos[particleId,0]
    #    y = pos[particleId,1]
    #    r = rad[particleId]
    #    ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=0.2, linewidth='0.3'))
    #    print(particleId)
    #    plt.pause(1)
    if(dense==True):
        figureName = "/home/francesco/Pictures/soft/packings/dense-" + figureName + ".png"
    elif(border==True):
        figureName = "/home/francesco/Pictures/soft/packings/border-" + figureName + ".png"
    elif(ekmap==True):
        colorBar = cm.ScalarMappable(cmap='viridis')
        cb = plt.colorbar(colorBar)
        label = "$E_{kin}$"
        cb.set_ticks([0, 1])
        cb.ax.tick_params(labelsize=12)
        ticklabels = [np.format_float_scientific(np.min(ekin), precision=2), np.format_float_scientific(np.max(ekin), precision=2)]
        cb.set_ticklabels(ticklabels)
        cb.set_label(label=label, fontsize=14, labelpad=-20, rotation='horizontal')
        figureName = "/home/francesco/Pictures/soft/packings/ekmap-" + figureName + ".png"
    elif(quiver==True):
        figureName = "/home/francesco/Pictures/soft/packings/velmap-" + figureName + ".png"
    else:
        figureName = "/home/francesco/Pictures/soft/packings/" + figureName + ".png"
    plt.tight_layout()
    plt.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPFixedBoundaryPacking(dirName, figureName, onedim=False, quiver=False, alpha = 0.6):
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    #if(onedim == "onedim"):
    #    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setPackingAxes(boxSize, ax)
    #setBigBoxAxes(boxSize, ax, 0.05)
    colorId = getRadColorList(rad)
    if(quiver==True):
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        #vel *= 5
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        if(quiver==True):
            ax.add_artist(plt.Circle([x, y], r, edgecolor=colorId[particleId], facecolor='none', alpha=alpha, linewidth = 0.7))
            vx = vel[particleId,0]
            vy = vel[particleId,1]
            ax.quiver(x, y, vx, vy, facecolor='k', width=0.002, scale=10)#width=0.002, scale=3)20
        else:
            ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth='0.3'))
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/packings/fb-" + figureName + ".png"
    plt.savefig(figureName, transparent=True, format = "png")
    plt.show()

def getPressureColorList(pressure, which='total'):
    colorList = cm.get_cmap('viridis', pressure.shape[0])
    colorId = np.zeros((pressure.shape[0], 4))
    count = 0
    if(which=='total'):
        p = pressure[:,0] + pressure[:,1] + pressure[:,2]
    elif(which=='virial'):
        p = pressure[:,0]
    elif(which=='thermal'):
        p = pressure[:,1]
    elif(which=='active'):
        p = pressure[:,2]
    elif(which=='epot'):
        p = pressure[:,3]
    for particleId in np.argsort(p):
        colorId[particleId] = colorList(count/p.shape[0])
        count += 1
    return colorId, colorList

def plotSPStressMapPacking(dirName, figureName, which='total', droplet=False, l1=0.035, alpha=0.7):
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    pos = ucorr.shiftPositions(pos, boxSize, 0.2, -0.3)
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setPackingAxes(boxSize, ax)
    if(os.path.exists(dirName + os.sep + "particleStress!.dat")):
        pressure = np.loadtxt(dirName + os.sep + "particleStress.dat")
    else:
        if(droplet == 'droplet'):
            pressure = spCorr.computeDropletParticleStress(dirName, l1)
        else:
            pressure = spCorr.computeParticleStress(dirName)
    colorId, colorList = getPressureColorList(pressure, which)
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth='0.3'))
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = plt.colorbar(colorBar, cax=cax)
    cb.set_ticks(np.linspace(0, 1, 5))
    cb.ax.tick_params(labelsize=10)
    if(which=='total'):
        mintick = np.min(pressure[:,0] + pressure[:,1] + pressure[:,2])
        maxtick = np.max(pressure[:,0] + pressure[:,1] + pressure[:,2])
        label = "$ Total$\n$stress$"
    elif(which=='virial'):
        mintick = np.min(pressure[:,0])
        maxtick = np.max(pressure[:,0])
        label = "$ Steric$\n$stress$"
    elif(which=='thermal'):
        mintick = np.min(pressure[:,1])
        maxtick = np.max(pressure[:,1])
        label = "$ Thermal$\n$stress$"
    elif(which=='active'):
        mintick = np.min(pressure[:,2])
        maxtick = np.max(pressure[:,2])
        label = "$ Active$\n$stress$"
    elif(which=='epot'):
        mintick = np.min(pressure[:,3])
        maxtick = np.max(pressure[:,3])
        label = "$E_{pot}$"
    tickList = np.linspace(mintick, maxtick, 5)
    for i in range(tickList.shape[0]):
        tickList[i] = np.format_float_positional(tickList[i], precision=0)
        #tickList[i] = np.format_float_scientific(tickList[i], precision=0)
    cb.set_ticklabels(tickList)
    cb.set_label(label=label, fontsize=14, labelpad=20, rotation='horizontal')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/packings/pmap-" + figureName + ".png"
    plt.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPVoronoiPacking(dirName, figureName, dense=False, threshold=0.84, filter=True, alpha=0.7):
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    pos = ucorr.shiftPositions(pos, boxSize, 0.1, -0.3)
    # compute voronoi tessellation of particle positions
    #vor = Voronoi(pos)
    #fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False, line_colors='k', line_width=0.5, line_alpha=1, point_size=2)
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    #fig.set_dpi(150)
    ax.set_xlim(xBounds[0], xBounds[1])
    ax.set_ylim(yBounds[0], yBounds[1])
    ax.set_aspect('equal', adjustable='box')
    setPackingAxes(boxSize, ax)
    #setBigBoxAxes(boxSize, ax, 0.1)
    newPos, newRad, newIndices = ucorr.augmentPacking(pos, rad)
    colorId = getRadColorList(newRad)
    if(dense==True):
        if(os.path.exists(dirName + os.sep + "delaunayList!.dat")):
            denseList = np.loadtxt(dirName + os.sep + "delaunayList.dat")
        else:
            denseList,_ = spCorr.computeDelaunayCluster(dirName, threshold, filter=filter)
        #if(os.path.exists(dirName + os.sep + "denseList!.dat")):
        #    denseList = np.loadtxt(dirName + os.sep + "denseList.dat")
        #else:
        #    denseList,_ = spCorr.computeVoronoiCluster(dirName, threshold, filter=filter)
        colorId = getDenseColorList(denseList)
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=0.3))
        #if(particleId > rad.shape[0]):
        #    plt.plot(pos[newIndices[particleId],0], pos[newIndices[particleId],1], '*', markersize=8, markeredgecolor='k', markeredgewidth = 0.6, color='skyblue')
    #cells = pyvoro.compute_2d_voronoi(pos, [[0, boxSize[0]], [0, boxSize[1]]], 1, radii=rad)
    #for i, cell in enumerate(cells):
    #    polygon = cell['vertices']
    #    ax.fill(*zip(*polygon), facecolor = 'none', edgecolor='k', lw=0.2)
    delaunay = Delaunay(newPos)
    insideIndex = ucorr.getInsideBoxDelaunaySimplices(delaunay.simplices, newPos, boxSize)
    plt.triplot(newPos[:,0], newPos[:,1], delaunay.simplices[insideIndex==1], lw=0.2, color='k')
    #plt.plot(pos[:,0], pos[:,1], 'o', markersize=0.2, markeredgecolor='k', color='r')
    #plt.plot(pos[14730,0], pos[14730,1], '*', markersize=10, markeredgecolor='k', color='b')
    #plt.plot(pos[595,0], pos[595,1], '*', markersize=10, markeredgecolor='k', color='r')
    #plt.plot(pos[11881,0], pos[11881,1], '*', markersize=10, markeredgecolor='k', color='k')
    #plt.plot(pos[7929,0], pos[7929,1], '*', markersize=10, markeredgecolor='k', color='b')
    #plt.plot(pos[6831,0], pos[6831,1], '*', markersize=10, markeredgecolor='k', color='r')
    #plt.plot(pos[7895,0], pos[7895,1], '*', markersize=10, markeredgecolor='k', color='k')
    # particles for 0.30 8k
    #plt.plot(pos[2169,0], pos[2169,1], '*', markersize=10, markeredgecolor='k', color='b')
    #plt.plot(pos[4153,0], pos[4153,1], '*', markersize=10, markeredgecolor='k', color='r')
    #plt.plot(pos[1474,0], pos[1474,1], '*', markersize=10, markeredgecolor='k', color='k')
    #plt.plot(pos[6358,0], pos[6358,1], '*', markersize=10, markeredgecolor='k', color=[0.5,0.5,0.5])
    #plt.plot(pos[57,0], pos[57,1], '*', markersize=10, markeredgecolor='k', color='b')
    #plt.plot(pos[5377,0], pos[5377,1], '*', markersize=10, markeredgecolor='k', color='r')
    #plt.plot(pos[3995,0], pos[3995,1], '*', markersize=10, markeredgecolor='k', color='k')
    #plt.plot(pos[297,0], pos[297,1], '*', markersize=10, markeredgecolor='k', color='b')
    #plt.plot(pos[2061,0], pos[2061,1], '*', markersize=10, markeredgecolor='k', color='r')
    #plt.plot(pos[2713,0], pos[2713,1], '*', markersize=10, markeredgecolor='k', color='k')
    # particle for 0.96 8k
    #plt.plot(pos[7612,0], pos[7612,1], '*', markersize=10, markeredgecolor='k', color='b')
    #plt.plot(pos[5307,0], pos[5307,1], '*', markersize=10, markeredgecolor='k', color='r')
    #plt.plot(pos[4719,0], pos[4719,1], '*', markersize=10, markeredgecolor='k', color='b')
    #plt.plot(pos[0,0], pos[0,1], '*', markersize=10, markeredgecolor='k', color='r')
    #plt.plot(pos[7961,0], pos[7961,1], '*', markersize=10, markeredgecolor='k', color='g')
    #x = np.linspace(0,1,1000)
    #slope = -0.11838938050442274
    #intercept = 0.9852218251735015
    #plt.plot(x, slope*x + intercept, ls='dashed', color='r', lw=1.2)
    #xp = 0.41924684666399464
    #yp = 0.9355874507185185
    #plt.plot(xp, yp, marker='s', markersize=8, markeredgecolor='k', color=[1,0.5,0])
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/packings/voronoi-" + figureName + ".png"
    plt.savefig(figureName, transparent=False, format = "png")
    plt.show()

def plotSoftParticles(ax, pos, rad, alpha = 0.6, colorMap = True, lw = 0.5):
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
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth = lw))

def plotSoftParticlesSubSet(ax, pos, rad, tagList, alpha = 0.6, colorMap = True, lw = 0.5):
    colorId = np.zeros((rad.shape[0], 4))
    if(colorMap == True):
        colorList = cm.get_cmap('viridis', rad.shape[0])
    else:
        colorList = cm.get_cmap('Reds', rad.shape[0])
    count = 0
    for particleId in np.argsort(rad):
        colorId[particleId] = colorList(count/rad.shape[0])
        count += 1
    colorId[tagList==1] = [0,0,0,1]
    alphaId = np.ones(colorId.shape[0])
    alphaId[tagList==1] = alpha
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alphaId[particleId], linewidth = lw))

def plotSoftParticleQuiverVel(axFrame, pos, vel, rad, alpha = 0.6, maxVelList = []):#122, 984, 107, 729, 59, 288, 373, 286, 543, 187, 6, 534, 104, 347]):
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
        vx = vel[particleId,0]
        vy = vel[particleId,1]
        axFrame.add_artist(plt.Circle([x, y], r, edgecolor=colorId[particleId], facecolor='none', alpha=alpha, linewidth = 0.7))
        axFrame.quiver(x, y, vx, vy, facecolor='k', width=0.002, scale=10)#width=0.003, scale=1, headwidth=5)
        #for j in range(13):
        #    if(particleId == maxVelList[j]):
        #        axFrame.quiver(x, y, vx, vy, facecolor='k', width=0.003, scale=1, headwidth=5)

def plotSoftParticlePressureMap(axFrame, pos, pressure, rad, alpha = 0.7):
    colorId = np.zeros((rad.shape[0], 4))
    colorList = cm.get_cmap('viridis', rad.shape[0])
    count = 0
    p = pressure[:,0] + pressure[:,1] + pressure[:,2]
    for particleId in np.argsort(p):
        colorId[particleId] = colorList(count/p.shape[0])
        count += 1
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth='0.3'))

def plotSoftParticleCluster(axFrame, pos, rad, clusterList, alpha = 0.4):
    for particleId in range(pos.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        if(clusterList[particleId] == 1):
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='b', alpha=alpha, linewidth = 0.7))
        else:
            axFrame.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=[1,1,1], alpha=alpha, linewidth = 0.7))

def makeSoftParticleClusterFrame(dirName, rad, boxSize, figFrame, frames, clusterList):
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setPackingAxes(boxSize, axFrame)
    plotSoftParticleCluster(axFrame, pos, rad, clusterList)
    plt.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def makeSPPackingClusterMixingVideo(dirName, figureName, numFrames = 20, firstStep = 0, stepFreq = 1e04):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    stepList = uplot.getStepList(numFrames, firstStep, stepFreq)
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    if(os.path.exists(dirName + os.sep + "t" + str(stepList[0]) + "/denseList!.dat")):
        denseList = np.loadtxt(dirName + os.sep + "t" + str(stepList[0]) + "/denseList.dat")
    else:
        denseList,_ = spCorr.computeVoronoiCluster(dirName + os.sep + "t" + str(stepList[0]))
    # the first configuration gets two frames for better visualization
    makeSoftParticleClusterFrame(dirName + os.sep + "t" + str(stepList[0]), rad, boxSize, figFrame, frames, denseList)
    for i in stepList:
        dirSample = dirName + os.sep + "t" + str(i)
        makeSoftParticleClusterFrame(dirSample, rad, boxSize, figFrame, frames, denseList)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/soft/packings/clustermix-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def makeSoftParticleFrame(dirName, rad, boxSize, figFrame, frames, subSet = False, firstIndex = 10, npt = False, quiver = False, cluster = False, pmap = False, droplet = False, l1=0.035):
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    pos = ucorr.shiftPositions(pos, boxSize, 0, 0.2)
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setPackingAxes(boxSize, axFrame)
    if(subSet == "subset"):
        tagList = np.zeros(rad.shape[0])
        tagList[:firstIndex] = 1
    elif(quiver == "quiver"):
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        plotSoftParticleQuiverVel(axFrame, pos, vel, rad)
    elif(pmap == "pmap"):
        if(os.path.exists(dirName + os.sep + "particleStress!.dat")):
            pressure = np.loadtxt(dirName + os.sep + "particleStress.dat")
        else:
            if(droplet == 'droplet'):
                pressure = spCorr.computeDropletParticleStress(dirName, l1)
            else:
                pressure = spCorr.computeParticleStress(dirName)
        plotSoftParticlePressureMap(axFrame, pos, pressure, rad)
    elif(cluster == "cluster"):
        if(os.path.exists(dirName + os.sep + "clusterLabels.dat")):
            clusterList = np.loadtxt(dirName + os.sep + "clusterLabels.dat")[:,0]
        else:
            clusterList,_,_ = spCorr.searchClusters(dirName, numParticles=rad.shape[0])
        plotSoftParticleCluster(axFrame, pos, rad, clusterList)
    else:
        if(npt == "npt"):
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
        plotSoftParticles(axFrame, pos, rad)
    figFrame.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def makeSPPackingVideo(dirName, figureName, numFrames = 20, firstStep = 0, stepFreq = 1e04, logSpaced = False, subSet = False, firstIndex = 0, npt = False, quiver = False, cluster = False, pmap = False, droplet = False):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    if(logSpaced == False):
        stepList = uplot.getStepList(numFrames, firstStep, stepFreq)
    else:
        stepList = []
        for dir in os.listdir(dirName):
            if(os.path.isdir(dirName + os.sep + dir) and dir != "dynamics"):
                stepList.append(int(dir[1:]))
        stepList = np.array(np.sort(stepList))
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    # the first configuration gets two frames for better visualization
    makeSoftParticleFrame(dirName + os.sep + "t" + str(stepList[0]), rad, boxSize, figFrame, frames, subSet, firstIndex, npt, quiver, cluster, pmap, droplet)
    vel = []
    for i in stepList:
        dirSample = dirName + os.sep + "t" + str(i)
        makeSoftParticleFrame(dirSample, rad, boxSize, figFrame, frames, subSet, firstIndex, npt, quiver, cluster, pmap, droplet)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    if(quiver=="quiver"):
        figureName = "velmap-" + figureName
    if(pmap=="pmap"):
        figureName = "pmap-" + figureName
    anim.save("/home/francesco/Pictures/soft/packings/" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def makeVelFieldFrame(dirName, numBins, bins, boxSize, numParticles, figFrame, frames):
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setGridAxes(bins, axFrame)
    grid, field = spCorr.computeVelocityField(dirName, numBins, plot=False, boxSize=boxSize, numParticles=numParticles)
    axFrame.quiver(grid[:,0], grid[:,1], field[:,0], field[:,1], facecolor='k', width=0.002, scale=3)
    plt.tight_layout()
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
    _, stepList = ucorr.getOrderedDirectories(dirName)
    #timeList = timeList.astype(int)
    stepList = stepList[np.argwhere(stepList%stepFreq==0)[:,0]]
    stepList = stepList[:numFrames]
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    bins = np.linspace(-0.5*boxSize[0],0, numBins)
    bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
    bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
    setGridAxes(bins, ax)
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    # the first configuration gets two frames for better visualization
    makeVelFieldFrame(dirName + os.sep + "t0", numBins, bins, boxSize, numParticles, figFrame, frames)
    vel = []
    for i in stepList:
        dirSample = dirName + os.sep + "t" + str(i)
        makeVelFieldFrame(dirSample, numBins, bins, boxSize, numParticles, figFrame, frames)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/soft/packings/velfield-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def plotDeformableParticles(ax, pos, rad, nv, faceColor = [0,0.5,1], edgeColor = [0.3,0.3,0.3], colorMap = False, edgeColorMap = False, alpha = 0.7, ls = '-', lw = 0.2):
    start = 0
    colorList = cm.get_cmap('viridis', nv.shape[0])
    colorId = np.zeros((nv.shape[0], 4))
    count = 0
    for particleId in np.argsort(nv):
        colorId[particleId] = colorList(count/nv.shape[0])
        count += 1
    for particleId in range(nv.shape[0]):
        for vertexId in range(nv[particleId]):
            x = pos[start + vertexId,0]
            y = pos[start + vertexId,1]
            r = rad[start + vertexId]
            if(colorMap == True):
                if(faceColor == [0,0.5,1]):
                    ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = colorId[particleId], alpha = alpha, linestyle = ls, linewidth = lw))
                elif(edgeColorMap == True):
                    ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor[particleId], facecolor = faceColor[particleId], alpha = alpha, linestyle = ls, linewidth = lw))
                else:
                    ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = faceColor(particleId/nv.shape[0]), alpha = alpha, linestyle = ls, linewidth = lw))
            else:
                ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = faceColor, alpha = alpha, linestyle = ls, linewidth = lw))
            #label = ax.annotate(str(start + vertexId), xy=(x, y), fontsize=5, verticalalignment="center", horizontalalignment="center")
        start += nv[particleId]

def trackDeformableParticles(ax, pos, rad, nv, edgeColor = [0.3,0.3,0.3], alpha = 0.5, ls = '-', lw = 0.5, trackList = [], highlightList = []):
    start = 0
    trackList = np.array(trackList)
    highlightList = np.array(highlightList)
    for particleId in range(nv.shape[0]):
        for vertexId in range(nv[particleId]):
            x = pos[start + vertexId,0]
            y = pos[start + vertexId,1]
            r = rad[start + vertexId]
            if(np.isin(particleId, trackList)):
                ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = [0.2,1,0.2], alpha = 1, linestyle = ls, linewidth = lw))
            elif(np.isin(particleId, highlightList)):
                ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = 'k', alpha = 1, linestyle = ls, linewidth = lw))
            else:
                ax.add_artist(plt.Circle([x, y], r, edgecolor = edgeColor, facecolor = [0.9,0.9,0.9], alpha = alpha, linestyle = ls, linewidth = lw))
        start += nv[particleId]

def plotDPMPacking(dirName, figureName, faceColor = [0,0.5,1], edgeColor = [0.3,0.3,0.3], colorMap = False, edgeColorMap = False, alpha = 0.7, save = True, plot = True):
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    if(os.path.exists(dirName + os.sep + "radii.dat")):
        sep = "/"
    else:
        sep = "../"
    rad = np.array(np.loadtxt(dirName + sep + "radii.dat"))
    nv = np.array(np.loadtxt(dirName + sep + "numVertexInParticleList.dat"), dtype=int)
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    #setBigBoxAxes(boxSize, ax)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    plotDeformableParticles(ax, pos, rad, nv, faceColor, edgeColor, colorMap, edgeColorMap, alpha)
    if(save == True):
        plt.savefig("/home/francesco/Pictures/dpm/packings/" + figureName + ".png", transparent=True, format = "png")
    if(plot == True):
        plt.show()
    else:
        return ax

def compareDPMPackings(dirName1, dirName2, figureName):
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    boxSize = np.loadtxt(dirName1 + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax) # packings need to have the same boxSize
    dirNameList = np.array([dirName1, dirName2])
    colorList = ['r', 'b']
    for i in range(dirNameList.shape[0]):
        pos = np.array(np.loadtxt(dirNameList[i] + os.sep + "positions.dat"))
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        rad = np.array(np.loadtxt(dirNameList[i] + os.sep + "radii.dat"))
        nv = np.array(np.loadtxt(dirNameList[i] + os.sep + "numVertexInParticleList.dat"), dtype=int)
        plotDeformableParticles(ax, pos, rad, nv, colorList[i])
    plt.savefig("/home/francesco/Pictures/dpm/packings/" + figureName + ".png", transparent=True, format = "png")
    plt.show()

def makeDeformablePackingFrame(pos, rad, nv, boxSize, figFrame, frames):
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    gcfFrame = plt.gcf()
    gcfFrame.clear()
    axFrame = figFrame.gca()
    setPackingAxes(boxSize, axFrame)
    plotDeformableParticles(axFrame, pos, rad, nv, colorMap = True)
    plt.tight_layout()
    axFrame.remove()
    frames.append(axFrame)

def makeDPMPackingVideo(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04, logSpaced = False):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 300
    frames = []
    if(logSpaced == False):
        _, stepList = ucorr.getOrderedDirectories(dirName)
        stepList = stepList[np.argwhere(stepList%stepFreq==0)[:,0]]
        stepList = stepList[:numFrames]
    else:
        stepList = []
        for dir in os.listdir(dirName):
            if(os.path.isdir(dirName + os.sep + dir) and dir != "dynamics"):
                stepList.append(int(dir[1:]))
        stepList = np.array(np.sort(stepList))
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    rad = np.array(np.loadtxt(dirName + os.sep + "radii.dat"))
    # the first configuration gets two frames for better visualization
    pos = np.array(np.loadtxt(dirName + os.sep + "t10000/positions.dat"))
    makeDeformablePackingFrame(pos, rad, nv, boxSize, figFrame, frames)
    numVertices = rad.shape[0]
    for i in stepList:
        pos = np.array(np.loadtxt(dirName + os.sep + "t" + str(i) + "/positions.dat"))
        makeDeformablePackingFrame(pos, rad, nv, boxSize, figFrame, frames)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames+1, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def compareDPMPackingsVideo(dirName, fileName, figureName):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    maxStep = 1001
    numFrames = int((maxStep -1)/ 10)
    frameTime = 200
    frames = []
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    rad = np.loadtxt(dirName + os.sep + "radii.dat")
    rad = np.array(rad)
    numVertices = rad.shape[0]
    for i in range(1, maxStep, 10):
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        setPackingAxes(boxSize, axFrame)
        # second packing
        pos = np.loadtxt(fileName, skiprows=(numVertices*i), max_rows = numVertices, usecols = (0,1))
        pos = np.array(pos)
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        plotDeformableParticles(axFrame, pos, rad, nv, faceColor = [1,1,1], alpha = 1, ls = '--', lw = 1.2)
        # first packing
        pos = np.loadtxt(dirName + os.sep + "pos_step" + str(i) + ".dat")
        pos = np.array(pos)
        plotDeformableParticles(axFrame, pos, rad, nv)
        plt.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def plotDPMcolorHOP(dirName, figureName, colorMap = True, alpha = 0.5, save = False):
    psi6 = spCorrelation.computeHexaticOrder(dirName)
    colorList = [[1-np.abs(x), 1-np.abs(x), 1] for x in psi6]
    plotDPMPacking(dirName, figureName, colorList, colorMap = colorMap, alpha = alpha, save = save)

def makeCompressionVideo(dirName, figureName, numFrames = 50):
    phiList = np.sort(os.listdir(dirName))[60::4]
    phiList = phiList[-numFrames:]
    print(phiList)
    numFrames -= 1
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 200
    frames = []
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + phiList[0] + "/boxSize.dat")
    setPackingAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + phiList[0] + "/numVertexInParticleList.dat", dtype=int))
    numVertices = np.sum(nv)
    colorList = cm.get_cmap('viridis', nv.shape[0])
    for phi in phiList:
        pos = np.array(np.loadtxt(dirName + os.sep + phi + "/positions.dat"))
        rad = np.array(np.loadtxt(dirName + os.sep + phi +"/radii.dat"))
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        setPackingAxes(boxSize, axFrame)
        plotDeformableParticles(axFrame, pos, rad, nv, faceColor = colorList, colorMap = True)
        plt.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/comp-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def makeRearrengementsVideo(dirName, figureName, numFrames = 20, firstStep = 1e07, stepFreq = 1e04):
    def animate(i):
        frames[i].figure=fig
        fig.axes.append(frames[i])
        fig.add_axes(frames[i])
        return gcf.artists
    frameTime = 200
    frames = []
    stepList = uplot.getStepList(numFrames, firstStep, stepFreq)
    print(stepList)
    #frame figure
    figFrame = plt.figure(dpi=150)
    fig = plt.figure(dpi=150)
    gcf = plt.gcf()
    gcf.clear()
    ax = fig.gca()
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    setPackingAxes(boxSize, ax)
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat", dtype=int))
    numParticles = nv.shape[0]
    rad = np.loadtxt(dirName + os.sep + "radii.dat")
    rad = np.array(rad)
    contactdiff = shapeGraphics.getContactDiff(dirName, numParticles, stepList)
    rearrangeList = np.argwhere(contactdiff==2)[1][0]
    initialContacts = np.loadtxt(dirName + os.sep + "t" + str(stepList[0]) + "/contacts.dat", dtype=int)+1
    initialContacts = np.flip(np.sort(initialContacts, axis=1), axis=1)
    highlightList = (initialContacts[rearrangeList, initialContacts[rearrangeList]>0]-1).tolist()
    rearrangeList = [rearrangeList]
    print(rearrangeList, highlightList)
    pos = np.loadtxt(dirName + os.sep + "t" + str(stepList[0]) + "/positions.dat")
    print(pos[rearrangeList], boxSize)
    for i in stepList:
        pos = np.loadtxt(dirName + os.sep + "t" + str(i) + "/positions.dat")
        pos = np.array(pos)
        #pos[:,0] += 0.2*boxSize[0]
        pos[:,1] -= 0.2*boxSize[1]
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        gcfFrame = plt.gcf()
        gcfFrame.clear()
        axFrame = figFrame.gca()
        setPackingAxes(boxSize, axFrame)
        trackDeformableParticles(axFrame, pos, rad, nv, trackList = rearrangeList, highlightList = highlightList)
        plt.tight_layout()
        axFrame.remove()
        frames.append(axFrame)
        anim = animation.FuncAnimation(fig, animate, frames=numFrames, interval=frameTime, blit=False)
    anim.save("/home/francesco/Pictures/dpm/packings/rearrange-" + figureName + ".gif", writer='imagemagick', dpi=plt.gcf().dpi)

def plotSPDPMPacking(dirName, figureName, faceColor = [0,0.5,1], edgeColor = [0.3,0.3,0.3], colorMap = False, edgeColorMap = False, alpha = 0.7, save = True):
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    pos = np.array(np.loadtxt(dirName + os.sep + "positions.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "radii.dat"))
    nv = np.array(np.loadtxt(dirName + os.sep + "numVertexInParticleList.dat"), dtype=int)
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    area = np.array(np.loadtxt(dirName + os.sep + "restAreas.dat"))
    setPackingAxes(boxSize, ax)
    #pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    plotDeformableParticles(ax, pos, rad, nv, faceColor, edgeColor, colorMap, edgeColorMap, alpha)
    rad = np.array(np.loadtxt(dirName + os.sep + "softRad.dat"))
    pos = np.array(np.loadtxt(dirName + os.sep + "softPos.dat"))
    #pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    plotSoftParticles(ax, pos, rad, alpha = 0.8, colorMap = False, lw = 0.5)
    plt.tight_layout()
    if(save == True):
        plt.savefig("/home/francesco/Pictures/spdpm/" + figureName + ".png", transparent=False, format = "png")
    plt.show()


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]
    figureName = sys.argv[3]

    if(whichPlot == "ss"):
        plotSPPacking(dirName, figureName)

    elif(whichPlot == "ss3d"):
        plot3DPacking(dirName, figureName)

    elif(whichPlot == "ssfixed"):
        onedim = sys.argv[4]
        plotSPFixedBoundaryPacking(dirName, figureName, onedim)

    elif(whichPlot == "ssvel"):
        plotSPPacking(dirName, figureName, quiver=True)

    elif(whichPlot == "ssdense"):
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        plotSPPacking(dirName, figureName, dense=True, threshold=threshold, filter=filter)

    elif(whichPlot == "ssborder"):
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        plotSPPacking(dirName, figureName, border=True, threshold=threshold, filter=filter)

    elif(whichPlot == "ssekin"):
        alpha = float(sys.argv[4])
        plotSPPacking(dirName, figureName, ekmap=True, alpha=alpha)

    elif(whichPlot == "ssstress"):
        which = sys.argv[4]
        droplet = sys.argv[5]
        l1 = float(sys.argv[6])
        plotSPStressMapPacking(dirName, figureName, which, droplet, l1)

    elif(whichPlot == "ssvoro"):
        plotSPVoronoiPacking(dirName, figureName)

    elif(whichPlot == "ssdeldense"):
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        plotSPVoronoiPacking(dirName, figureName, dense=True, threshold=threshold, filter=filter)

    elif(whichPlot == "ssvideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "velfield"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        numBins = int(sys.argv[7])
        makeSPVelFieldVideo(dirName, figureName, numFrames, firstStep, stepFreq, numBins=numBins)

    elif(whichPlot == "velvideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, quiver = "quiver")

    elif(whichPlot == "pvideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, pmap = "pmap")

    elif(whichPlot == "dropvideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        l1 = float(sys.argv[7])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, pmap = "pmap", droplet = "droplet", l1=l1)

    elif(whichPlot == "clustervideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, cluster = "cluster")

    elif(whichPlot == "clustermix"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingClusterMixingVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "ssvideosubset"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        firstIndex = int(sys.argv[7])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, subSet = "subset", firstIndex = firstIndex)

    elif(whichPlot == "ssvideonpt"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeSPPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq, npt = "npt")

    elif(whichPlot == "dpm"):
        plotDPMPacking(dirName, figureName, colorMap = True)

    elif(whichPlot == "dpmvideo"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeDPMPackingVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "comparedpm"):
        dirSample = dirName + os.sep + sys.argv[4]
        compareDPMPackings(dirName, dirSample, figureName)

    elif(whichPlot == "comparedpmvideo"):
        fileName = dirName + os.sep + sys.argv[4]
        dirName = dirName + os.sep + sys.argv[5]
        compareDPMPackingsVideo(dirName, fileName, figureName)

    elif(whichPlot == "dpmhop"):
        plotDPMcolorHOP(dirName, figureName)

    elif(whichPlot == "compvideo"):
        numFrames = int(sys.argv[4])
        makeCompressionVideo(dirName, figureName, numFrames)

    elif(whichPlot == "dpmrearrange"):
        numFrames = int(sys.argv[4])
        firstStep = float(sys.argv[5])
        stepFreq = float(sys.argv[6])
        makeRearrengementsVideo(dirName, figureName, numFrames, firstStep, stepFreq)

    elif(whichPlot == "spdpm"):
        plotSPDPMPacking(dirName, figureName)

    else:
        print("Please specify the type of plot you want")
