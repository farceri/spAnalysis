'''
Created by Francesco
14 July 2023
'''
#functions and script to compute cluster correlations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
from scipy.fftpack import rfft, rfftfreq, fft, fftfreq
import pyvoro
import sys
import os
import utils
import utilsPlot as uplot

############################## Voronoi clustering ##############################
def computeVoronoiCluster(dirName, threshold=0.65, filter='filter', plot=False):
    sep = utils.getDirSep(dirName, "boxSize")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    localDensity = np.zeros(numParticles)
    denseList = np.zeros(numParticles)
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
    # need to center the cluster for voronoi border detection
    pos = utils.centerPositions(pos, rad, boxSize)
    cells = pyvoro.compute_2d_voronoi(pos, [[0, boxSize[0]], [0, boxSize[1]]], 1, radii=rad)
    for i in range(numParticles):
        localDensity[i] = np.pi*rad[i]**2 / np.abs(cells[i]['volume'])
        if(localDensity[i] > threshold):
                denseList[i] = 1
    #print("average local density:", np.mean(localDensity))
    #print("Number of dense particles: ", denseList[denseList==1].shape[0])
    if(filter=='filter'):
        connectList = np.zeros(numParticles)
        for i in range(numParticles):
            if(np.sum(contacts[i]!=-1)>2):
                denseContacts = 0
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    if(denseList[c] == 1):
                        denseContacts += 1
                if(denseContacts > 1):
                    # this is at least a four particle cluster
                    connectList[i] = 1
        denseList[connectList==0] = 0
        # label contacts and contacts of contacts of dense particles as dense
        for times in range(2):
            for i in range(numParticles):
                if(denseList[i] == 1):
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(denseList[c] != 1):
                            denseList[c] = 1
        #print("Number of dense particles after contact filter: ", denseList[denseList==1].shape[0])
    # look for rattlers in the fluid and label them as dense particles
    neighborCount = np.zeros(numParticles)
    denseNeighborCount = np.zeros(numParticles)
    for i in range(numParticles):
        if(denseList[i]==0):
            for j in range(len(cells[i]['faces'])):
                index = cells[i]['faces'][j]['adjacent_cell']
                neighborCount[i] += 1
                if(denseList[index] == 1):
                    denseNeighborCount[i] += 1
    rattlerList = np.zeros(numParticles)
    for i in range(numParticles):
        if(denseList[i]==0):
            if(neighborCount[i] == denseNeighborCount[i]):
                rattlerList[i] = 1
    denseList[rattlerList==1] = 1
    #print("Number of dense particles after rattler correction: ", denseList[denseList==1].shape[0])
    np.savetxt(dirName + "/denseList.dat", denseList)
    np.savetxt(dirName + "/voroDensity.dat", localDensity)
    if(plot=='plot'):
        uplot.plotCorrelation(np.arange(1, numParticles+1, 1), np.sort(localDensity), "$\\varphi^{Voronoi}$", xlabel = "$Particle$ $index$", color='k')
        plt.show()
        numBins = 100
        pdf, edges = np.histogram(localDensity, bins=np.linspace(0, 1, numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        uplot.plotCorrelation(edges, pdf, "$PDF(\\varphi^{Voronoi})$", xlabel = "$\\varphi^{Voronoi}$", color='r')
        plt.show()
    return denseList, localDensity

######################## Compute voronoi cluster border ########################
def computeVoronoiBorder(dirName, threshold=0.65, filter='filter'):
    sep = utils.getDirSep(dirName, "boxSize")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    phi = utils.readFromParams(dirName + sep, "phi")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    meanRad = np.mean(rad)
    localDensity = np.zeros(numParticles)
    denseList = np.zeros(numParticles)
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    # need to center the cluster for voronoi border detection
    pos = utils.centerPositions(pos, rad, boxSize)
    cells = pyvoro.compute_2d_voronoi(pos, [[0, boxSize[0]], [0, boxSize[1]]], 1, radii=rad)
    # check if denseList already exists
    if(os.path.exists(dirName + os.sep + "denseList.dat")):
        denseList = np.loadtxt(dirName + os.sep + "denseList.dat")
    else:
        denseList,_ = computeVoronoiCluster(dirName, threshold, filter=filter)
    borderList = np.zeros(numParticles)
    for i in range(numParticles):
        if(denseList[i]==1):
            for j in range(len(cells[i]['faces'])):
                index = cells[i]['faces'][j]['adjacent_cell']
                edgeIndex = cells[i]['faces'][j]['vertices']
                if(denseList[index] == 0 and index>0):
                    borderList[i] = 1
    # compute border length from sorted border segments
    borderLength = 0
    borderPos = pos[borderList==1]
    borderPos = utils.sortBorderPos(borderPos, borderList, boxSize)
    np.savetxt(dirName + os.sep + "borderPos.dat", borderPos)
    # compute border length by summing over segments on the border
    borderLength = 0
    for i in range(1,borderPos.shape[0]):
        borderLength += np.linalg.norm(utils.pbcDistance(borderPos[i], borderPos[i-1], boxSize))
    #print("Number of dense particles at the interface: ", borderList[borderList==1].shape[0])
    print("Border length from voronoi edges: ", borderLength)
    np.savetxt(dirName + os.sep + "borderList.dat", borderList)
    np.savetxt(dirName + os.sep + "borderLength.dat", np.array([borderLength]))
    return borderList, borderLength

######################## Average voronoi local density #########################
def averageLocalVoronoiDensity(dirName, numBins=16, plot=False, dirSpacing=1):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    localDensity = np.empty(0)
    globalDensity = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "voroDensity.dat")):
            voroDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            _, voroDensity = computeVoronoiCluster(dirSample)
        voroArea = np.pi * rad**2 / voroDensity
        pos = utils.getPBCPositions(dirSample + os.sep + "particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + os.sep + "particleContacts.dat").astype(np.int64)
        localVoroDensity, density = utils.computeLocalVoronoiDensityGrid(pos, rad, contacts, boxSize, voroArea, xbin, ybin)
        localDensity = np.append(localDensity, localVoroDensity)
        globalDensity = np.append(globalDensity, density)
    localDensity = np.sort(localDensity).flatten()
    localDensity = localDensity[localDensity>0]
    alpha2 = np.mean(localDensity**4)/(2*(np.mean(localDensity**2)**2)) - 1
    pdf, edges = np.histogram(localDensity, bins=np.linspace(np.min(localDensity), np.max(localDensity), 100), density=True)
    edges = (edges[:-1] + edges[1:])/2
    np.savetxt(dirName + os.sep + "localVoroDensity-N" + str(numBins) + ".dat", np.column_stack((edges, pdf)))
    data = np.column_stack((np.mean(localDensity), np.var(localDensity), alpha2, np.mean(globalDensity), np.var(globalDensity)))
    np.savetxt(dirName + os.sep + "localVoroDensity-N" + str(numBins) + "-stats.dat", data)
    print("average local density: ", np.mean(localDensity), " +- ", np.var(localDensity))
    if(plot == 'plot'):
        uplot.plotCorrelation(edges, pdf, "$Local$ $density$ $distribution$", "$Local$ $density$", color='k')
        plt.pause(0.5)
        #plt.show()
    return np.mean(localDensity)

########################### Compute voronoi density ############################
def computeClusterVoronoiDensity(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    voronoiDensity = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        if(os.path.exists(dirSample + os.sep + "denseList.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
            voroDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            denseList, voroDensity = computeVoronoiCluster(dirSample)
        voroArea = np.pi*rad**2 / voroDensity
        for i in range(numParticles):
            # remove the overlaps from the particle area
            overlapArea = 0
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                overlapArea += utils.computeOverlapArea(pos[i], pos[c], rad[i], rad[c], boxSize)
            if(denseList[i]==1):
                voronoiDensity[d,0] += (np.pi * rad[i]**2 - overlapArea) / voroArea[i]
            else:
                voronoiDensity[d,1] += (np.pi * rad[i]**2 - overlapArea) / voroArea[i]
            voronoiDensity[d,2] += (np.pi * rad[i]**2 - overlapArea) / voroArea[i]
        voronoiDensity[d,0] /= numParticles
        voronoiDensity[d,1] /= numParticles
        voronoiDensity[d,2] /= numParticles
    np.savetxt(dirName + os.sep + "voronoiDensity.dat", np.column_stack((timeList, voronoiDensity)))
    print("Density in the fluid: ", np.mean(voronoiDensity[:,0]), " +- ", np.std(voronoiDensity[:,0]))
    print("Density outside the fluid: ", np.mean(voronoiDensity[:,1]), " +- ", np.std(voronoiDensity[:,1]))
    print("Density in the whole system: ", np.mean(voronoiDensity[:,2]), " +- ", np.std(voronoiDensity[:,2]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, voronoiDensity[:,0], "$\\varphi^{Voronoi}$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, voronoiDensity[:,1], "$\\varphi^{Voronoi}$", xlabel = "$Time,$ $t$", color='g')
        uplot.plotCorrelation(timeList, voronoiDensity[:,2], "$\\varphi^{Voronoi}$", xlabel = "$Time,$ $t$", color='k')
        plt.pause(0.5)

########################### Compute voronoi density ############################
def computeClusterVoronoiArea(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    voronoiArea = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        if(os.path.exists(dirSample + os.sep + "denseList.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
            voroDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            denseList, voroDensity = computeVoronoiCluster(dirSample)
        voroArea = np.pi*rad**2 / voroDensity
        for i in range(numParticles):
            # remove the overlaps from the particle area
            overlapArea = 0
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                overlapArea += utils.computeOverlapArea(pos[i], pos[c], rad[i], rad[c], boxSize)
            if(denseList[i]==1):
                voronoiArea[d,0] += (voroArea[i] - overlapArea)
            else:
                voronoiArea[d,1] += (voroArea[i] - overlapArea)
            voronoiArea[d,2] += (voroArea[i] - overlapArea)
    np.savetxt(dirName + os.sep + "voronoiArea.dat", np.column_stack((timeList, voronoiArea)))
    print("Fluid area: ", np.mean(voronoiArea[:,0]), " +- ", np.std(voronoiArea[:,0]), " fluid radius: ", np.mean(np.sqrt(voronoiArea[:,0]/np.pi)))
    print("Gas area: ", np.mean(voronoiArea[:,1]), " +- ", np.std(voronoiArea[:,1]))
    print("Occupied area in the whole system: ", np.mean(voronoiArea[:,2]), " +- ", np.std(voronoiArea[:,2]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, voronoiArea[:,0], "$A^{Voronoi}$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, voronoiArea[:,1], "$A^{Voronoi}$", xlabel = "$Time,$ $t$", color='g')
        uplot.plotCorrelation(timeList, voronoiArea[:,2], "$A^{Voronoi}$", xlabel = "$Time,$ $t$", color='k')
        plt.pause(0.5)
    return voronoiArea

def setAxes2D(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

def setPackingAxes(boxSize, ax):
    ax.set_xlim(0, boxSize[0])
    ax.set_ylim(0, boxSize[1])
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

def getDenseColorList(denseList):
    colorId = np.zeros((denseList.shape[0], 4))
    for particleId in range(denseList.shape[0]):
        if(denseList[particleId]==1):
            colorId[particleId] = [0.2,0.2,0.2,0.2]
        else:
            colorId[particleId] = [1,1,1,1]
    return colorId

def plotSPVoronoiPacking(dirName, figureName, dense=False, threshold=0.84, filter=True, alpha=0.7, shiftx=0, shifty=0, lj=False):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    if(lj==True):
        rad *= 2**(1/6)
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    pos = utils.shiftPositions(pos, boxSize, shiftx, shifty)
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    setPackingAxes(boxSize, ax)
    colorId = getRadColorList(rad)
    if(dense==True):
        if(os.path.exists(dirName + os.sep + "denseList!.dat")):
            denseList = np.loadtxt(dirName + os.sep + "denseList.dat")
        else:
            denseList,_ = computeVoronoiCluster(dirName, threshold, filter=filter)
        colorId = getDenseColorList(denseList)
    for particleId in range(rad.shape[0]):
        x = pos[particleId,0]
        y = pos[particleId,1]
        r = rad[particleId]
        ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=alpha, linewidth=0.3))
    cells = pyvoro.compute_2d_voronoi(pos, [[0, boxSize[0]], [0, boxSize[1]]], 1, radii=rad)
    for i, cell in enumerate(cells):
        polygon = cell['vertices']
        ax.fill(*zip(*polygon), facecolor = 'none', edgecolor='k', lw=0.2)
    plt.plot(pos[0,0], pos[0,1], marker='*', markersize=20, color='k')
    plt.plot(pos[cells[0]['faces'][0]['adjacent_cell'],0], pos[cells[0]['faces'][0]['adjacent_cell'],1], marker='*', markersize=20, color='r')
    plt.plot(pos[cells[0]['faces'][1]['adjacent_cell'],0], pos[cells[0]['faces'][1]['adjacent_cell'],1], marker='*', markersize=20, color='b')
    plt.plot(pos[cells[0]['faces'][2]['adjacent_cell'],0], pos[cells[0]['faces'][2]['adjacent_cell'],1], marker='*', markersize=20, color='g')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/packings/voronoi-" + figureName + ".png"
    plt.savefig(figureName, transparent=False, format = "png")
    plt.show()

if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

   if(whichCorr == "vorocluster"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        filter = sys.argv[5]
        computeVoronoiCluster(dirName, threshold, filter, plot)

    elif(whichCorr == "vorodensity"):
        plot = sys.argv[3]
        computeClusterVoronoiDensity(dirName, plot)

    elif(whichCorr == "voroarea"):
        plot = sys.argv[3]
        computeClusterVoronoiArea(dirName, plot)

    elif(whichCorr == "vorold"):
        np.seterr(divide='ignore', invalid='ignore')
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        averageLocalVoronoiDensity(dirName, numBins, plot)

    elif(whichCorr == "voroborder"):
        threshold = float(sys.argv[3])
        computeVoronoiBorder(dirName, threshold)

    elif(whichCorr == "ssvoro"):
        plotSPVoronoiPacking(dirName, "soft", shiftx=float(sys.argv[4]), shifty=float(sys.argv[5]))

    elif(whichCorr == "ljvoro"):
        plotSPVoronoiPacking(dirName, "lj", shiftx=float(sys.argv[4]), shifty=float(sys.argv[5]), lj=True)

    else:
        print("Please specify the correlation you want to compute")
