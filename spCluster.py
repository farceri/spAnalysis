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

################################################################################
############################## Cluster properties ##############################
################################################################################
def averageLocalDensity(dirName, numBins=30, weight=False, plot=False, dirSpacing=1):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    timeStep = utils.readFromParams(dirName, "dt")
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    cutoff = 2*xbin[1]
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    localDensity = np.empty(0)
    globalDensity = np.empty(0)
    for dir in dirList:
        localArea = np.zeros((numBins, numBins))
        pos = utils.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
        contacts = np.array(np.loadtxt(dirName + os.sep + dir + "/particleContacts.dat")).astype(int)
        if(weight == 'weight'):
            globalDensity = np.append(globalDensity, utils.computeWeightedLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea, cutoff))
        else:
            globalDensity = np.append(globalDensity, utils.computeLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea))
        localDensity = np.append(localDensity, localArea/localSquare)
    localDensity = np.sort(localDensity).flatten()
    localDensity = localDensity[localDensity>0]
    alpha2 = np.mean(localDensity**4)/(2*(np.mean(localDensity**2)**2)) - 1
    pdf, edges = np.histogram(localDensity, bins=np.linspace(np.min(localDensity), np.max(localDensity), 80), density=True)
    edges = (edges[:-1] + edges[1:])/2
    if(weight == 'weight'):
        np.savetxt(dirName + os.sep + "localDensity-N" + str(numBins) + "-weight.dat", np.column_stack((edges, pdf)))
        np.savetxt(dirName + os.sep + "localDensity-N" + str(numBins) + "-stats-weight.dat", np.column_stack((np.mean(localDensity), np.var(localDensity), np.mean(globalDensity), np.std(globalDensity))))
    else:
        np.savetxt(dirName + os.sep + "localDensity-N" + str(numBins) + ".dat", np.column_stack((edges, pdf)))
        np.savetxt(dirName + os.sep + "localDensity-N" + str(numBins) + "-stats.dat", np.column_stack((np.mean(localDensity), np.var(localDensity), alpha2, np.mean(globalDensity), np.std(globalDensity))))
    print("average global density: ", np.mean(globalDensity), " +- ", np.var(globalDensity))
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Local$ $density$ $distribution$", "$Local$ $density$")
        plt.show()

def computeLocalDensityAndNumberVSTime(dirName, numBins=12, plot=False):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    pRad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    localDensityVar = np.empty(0)
    localNumberVar = np.empty(0)
    for dir in dirList:
        localArea = np.zeros((numBins, numBins))
        localNumber = np.zeros((numBins, numBins))
        pos = utils.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
        contacts = np.array(np.loadtxt(dirName + os.sep + dir + "/particleContacts.dat"), dtype=int)
        utils.computeLocalAreaAndNumberGrid(pos, pRad, contacts, boxSize, xbin, ybin, localArea, localNumber)
        localDensity = localArea/localSquare
        localDensityVar = np.append(localDensityVar, np.var(localDensity))
        localNumberVar = np.append(localNumberVar, np.var(localNumber))
    if(plot=="plot"):
        np.savetxt(dirName + "localDensityAndNumberVarVSTime-N" + str(numBins) + ".dat", np.column_stack((timeList, localDensityVar, localNumberVar)))
        uplot.plotCorrelation(timeList, localDensityVar, "$Variance$ $of$ $local$ $density$", "$Time,$ $t$", color='k')
        uplot.plotCorrelation(timeList, localNumberVar, "$Variance$ $of$ $local$ $number$", "$Time,$ $t$", color='g')
        plt.show()

################################################################################
############################## Clustering algorithms ###########################
################################################################################
def searchClusters(dirName, numParticles=None, plot=False, cluster="cluster"):
    if(numParticles==None):
        numParticles = int(utils.readFromParams(dirName, "numParticles"))
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat"), dtype=int)
    particleLabel = np.zeros(numParticles)
    connectLabel = np.zeros(numParticles)
    noClusterList = np.zeros(numParticles)
    clusterLabel = 0
    for i in range(1,numParticles):
        if(np.sum(contacts[i]!=-1)>2):
            if(particleLabel[i] == 0): # this means it hasn't been checked yet
                # check that it is not a contact of contacts of previously checked particles
                belongToCluster = False
                for j in range(i):
                    for c in contacts[j, np.argwhere(contacts[j]!=-1)[:,0]]:
                        if(i==c):
                            # a contact of this particle already belongs to a cluster
                            particleLabel[i] = particleLabel[j]
                            belongToCluster = True
                            break
                if(belongToCluster == False):
                    newCluster = False
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(np.sum(contacts[c]!=-1)>2 and newCluster == False):
                            newCluster = True
                    if(newCluster == True):
                        clusterLabel += 1
                        particleLabel[i] = clusterLabel
                        particleLabel[contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]] = particleLabel[i]
        else:
            particleLabel[i] = 0
    # more stringent condition on cluster belonging
    connectLabel[np.argwhere(particleLabel > 0)] = 1
    for i in range(numParticles):
        connected = False
        for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
            if(particleLabel[c] != 0 and connected == False):
                connectLabel[i] = 1
                connected = True
        #connectLabel[contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]] = 1
    # get cluster lengthscale, center
    if(os.path.exists(dirName + os.sep + "boxSize.dat")):
        sep = "/"
    else:
        sep = "/../"
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    NpInCluster = connectLabel[connectLabel!=0].shape[0]
    clusterSize = np.sqrt(NpInCluster) * np.mean(rad[connectLabel!=0])
    pos = np.loadtxt(dirName + os.sep + "particlePos.dat")
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    clusterPos = np.mean(pos[connectLabel!=0], axis=0)
    # get list of particles within half of the lengthscale from the center
    #deepList = np.zeros(numParticles)
    for i in range(numParticles):
        if(particleLabel[i] == 0 and np.sum(contacts[i]) < 2):
            noClusterList[i] = 1
    #    delta = utils.pbcDistance(pos[i], clusterPos, boxSize)
    #    distance = np.linalg.norm(delta)
    #    if(distance < clusterSize * 0.5 and connectLabel[i] == 1):
    #        deepList[i] = 1
    np.savetxt(dirName + "/clusterLabels.dat", np.column_stack((connectLabel, noClusterList, particleLabel)))
    if(plot=="plot"):
        print("Cluster position, x: ", clusterPos[0], " y: ", clusterPos[1])
        print("Cluster size: ", clusterSize)
        print("Number of clusters: ", clusterLabel)
        print("Number of particles in clusters: ", connectLabel[connectLabel!=0].shape[0])
        # plot packing
        boxSize = np.loadtxt(dirName + os.sep + "../boxSize.dat")
        pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
        rad = np.array(np.loadtxt(dirName + os.sep + "../particleRad.dat"))
        xBounds = np.array([0, boxSize[0]])
        yBounds = np.array([0, boxSize[1]])
        pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
        fig = plt.figure(0, dpi = 150)
        ax = fig.gca()
        uplot.setPackingAxes(boxSize, ax)
        colorList = cm.get_cmap('prism', clusterLabel)
        colorId = np.zeros((pos.shape[0], 4))
        for i in range(numParticles):
            if(particleLabel[i]==0):
                colorId[i] = [1,1,1,1]
            else:
                colorId[i] = colorList(particleLabel[i]/clusterLabel)
        for particleId in range(numParticles):
            x = pos[particleId,0]
            y = pos[particleId,1]
            r = rad[particleId]
            ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=[1,1,1], alpha=0.6, linewidth=0.5))
            if(cluster=="cluster"):
                ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=colorId[particleId], alpha=0.6, linewidth=0.5))
                if(connectLabel[particleId] == 1):
                    ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=0.3, linewidth=0.5))
        plt.show()
    return connectLabel, noClusterList, particleLabel

def searchDBClusters(dirName, eps=0, min_samples=6, plot=False, contactFilter='contact'):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat"), dtype=int)
    # use 0.03 as typical distance
    if(eps == 0):
        eps = 2 * np.max(np.loadtxt(dirName + sep + "particleRad.dat"))
    labels = utils.getDBClusterLabels(pos, boxSize, eps, min_samples, contacts, contactFilter)
    clusterLabels = np.zeros(pos.shape[0])
    noClusterLabels = np.zeros(pos.shape[0])
    clusterLabels[labels!=-1] = 1
    noClusterLabels[labels==-1] = 1
    np.savetxt(dirName + os.sep + "dbClusterLabels.dat", np.column_stack((clusterLabels, noClusterLabels, labels)))
    #print("Found", np.unique(labels).shape[0]-1, "clusters") # zero is a label
    # plotting
    if(plot=="plot"):
        rad = np.loadtxt(dirName + sep + "particleRad.dat")
        uplot.plotPacking(boxSize, pos, rad, labels)
        plt.show()
    return clusterLabels, noClusterLabels, labels

def averageDBClusterSize(dirName, dirSpacing, eps=0.03, min_samples=6, plot=False, contactFilter=False):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    #cutoff = 2 * np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    print("number of samples: ", dirList.shape[0])
    clusterSize = []
    allClustersSize = []
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "/"
        if(os.path.exists(dirSample + os.sep + "dbClusterLabels.dat")):
            labels = np.loadtxt(dirSample + os.sep + "dbClusterLabels.dat")[:,2]
            if(plot=="plot"):
                pos = utils.getPBCPositions(dirSample + os.sep + "particlePos.dat", boxSize)
                uplot.plotPacking(boxSize, pos, rad, labels)
        else:
            _,_,labels = searchDBClusters(dirSample, eps=0, min_samples=min_samples, plot=plot, contactFilter=contactFilter)
        plt.clf()
        # get area of particles in clusters and area of individual clusters
        numLabels = np.unique(labels).shape[0]-1
        for i in range(numLabels):
            clusterSize.append(np.pi * np.sum(rad[labels==i]**2))
        allClustersSize.append(np.pi * np.sum(rad[labels!=-1]**2))
    np.savetxt(dirName + "dbClusterSize.dat", clusterSize)
    np.savetxt(dirName + "dbAllClustersSize.dat", allClustersSize)
    print("area of all particles in a cluster: ", np.mean(allClustersSize), " += ", np.std(allClustersSize))
    print("typical cluster size: ", np.mean(clusterSize), " += ", np.std(clusterSize))

def getDBClusterLabels(pos, boxSize, eps, min_samples, contacts, contactFilter=False):
    numParticles = pos.shape[0]
    distance = utils.omputeDistances(pos, boxSize)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance)
    labels = db.labels_
    if(contactFilter == 'contact'):
        connectLabel = np.zeros(numParticles)
        for i in range(numParticles):
            if(np.sum(contacts[i]!=-1)>1):
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(np.sum(contacts[c]!=-1)>1):
                            # this is at least a three particle cluster
                            connectLabel[i] = 1
        labels[connectLabel==0] = -1
    return labels

def getNoClusterLabel(labels, contacts):
    noLabels = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        if(labels[i] != -1 and np.sum(contacts[i]) < 2):
            noLabels[i] = 1
    return noLabels

########################## Cluster border calculation ##########################
def computeDBClusterBorder(dirName, plot='plot'):
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    xBounds = np.array([0, boxSize[0]])
    yBounds = np.array([0, boxSize[1]])
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    if(os.path.exists(dirName + os.sep + "dbClusterLabels.dat")):
        clusterList = np.loadtxt(dirName + os.sep + "dbClusterLabels.dat")[:,0]
    else:
        clusterList, _,_ = searchDBClusters(dirName, eps=0, min_samples=8, contactFilter=False)
        #clusterList, _,_ = searchClusters(dirName, numParticles)
    clusterParticles = np.argwhere(clusterList==1)[:,0]
    # compute particle neighbors with a distance cutoff
    distances = utils.computeDistances(pos, boxSize)
    maxNeighbors = 40
    neighbors = -1 * np.ones((numParticles, maxNeighbors), dtype=int)
    cutoff = 2 * np.max(rad)
    for i in clusterParticles:
        index = 0
        for j in clusterParticles:
            if(i != j and distances[i,j] < cutoff):
                neighbors[i, index] = j
                index += 1
    # initilize probe position
    numBins = 16
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    localArea = np.zeros((numBins, numBins))
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
    utils.computeLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea)
    localDensity = localArea/localSquare
    ri, ci = localDensity.argmin()//localDensity.shape[1], localDensity.argmin()%localDensity.shape[1]
    probe = np.array([xbin[ri], ybin[ci]])
    sigma = np.min(rad)*0.2
    multiple = 10
    # move it diagonally until hitting a particle in a cluster
    step = np.ones(2)*sigma/10
    contact = False
    while contact == False:
        probe += step
        for i in clusterParticles:
            distance = np.linalg.norm(utils.pbcDistance(pos[i], probe, boxSize))
            if(distance < (rad[i] + sigma)):
                contact = True
                firstId = i
                break
    # find the closest particle to the initial contact
    minDistance = 1
    for i in clusterParticles:
        distance = np.linalg.norm(utils.pbcDistance(pos[i], pos[firstId], boxSize))
        if(distance < minDistance and distance > 0):
            minDistance = distance
            closestId = i
    contactId = closestId
    currentParticles = clusterParticles[clusterParticles!=contactId]
    currentParticles = currentParticles[currentParticles!=firstId]
    print("Starting from particle: ", contactId, "last particle: ", firstId)
    # rotate the probe around cluster particles and check when they switch
    step = 1e-04
    borderLength = 0
    fig = plt.figure(0, dpi = 150)
    ax = fig.gca()
    uplot.setPackingAxes(boxSize, ax)
    if(plot=='plot'):
        for particleId in range(numParticles):
            x = pos[particleId,0]
            y = pos[particleId,1]
            r = rad[particleId]
            ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor=[1,1,1], alpha=0.6, linewidth=0.5))
            if(clusterList[particleId] == 1):
                ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=0.6, linewidth=0.5))
        ax.add_artist(plt.Circle(probe, multiple*sigma, edgecolor='k', facecolor=[0.2,0.8,0.5], alpha=0.9, linewidth=0.5))
        ax.add_artist(plt.Circle(pos[contactId], rad[contactId], edgecolor='k', facecolor='b', alpha=0.9, linewidth=0.5))
        ax.add_artist(plt.Circle(pos[closestId], rad[closestId], edgecolor='k', facecolor='g', alpha=0.9, linewidth=0.5))
    nCheck = 0
    previousNeighbors = []
    previousNeighbors.append(firstId)
    checkedParticles = np.zeros(numParticles, dtype=int)
    checkedParticles[contactId] += 1
    # save particles at the border of the cluster
    borderParticles = np.array([firstId], dtype=int)
    while contactId != firstId:
        borderParticles = np.append(borderParticles, contactId)
        delta = utils.pbcDistance(probe, pos[contactId], boxSize)
        theta0 = utils.checkAngle(np.arctan2(delta[1], delta[0]))
        #print("contactId: ", contactId, " contact angle: ", theta0)
        director = np.array([np.cos(theta0), np.sin(theta0)])
        probe = pos[contactId] + utils.polarPos(rad[contactId], theta0)
        theta = utils.checkAngle(theta0 + step)
        currentNeighbors = neighbors[contactId]
        currentNeighbors = np.setdiff1d(currentNeighbors, previousNeighbors)
        while theta > theta0:
            newProbe = pos[contactId] + utils.polarPos(rad[contactId], theta)
            distance = np.linalg.norm(utils.pbcDistance(newProbe, probe, boxSize))
            borderLength += distance
            theta = utils.checkAngle(theta + step)
            # loop over neighbors of the current cluster particle and have not been traveled yet
            for i in currentNeighbors[currentNeighbors!=-1]:
                distance = np.linalg.norm(utils.pbcDistance(pos[i], newProbe, boxSize))
                if(distance < (rad[i] + sigma)):
                    contact = True
                    #print("Found the next particle: ", i, " previous particle: ", contactId)
                    theta = theta0
                    previousNeighbors.append(contactId)
                    contactId = i
                    checkedParticles[contactId] += 1
                    if(plot=='plot'):
                        ax.add_artist(plt.Circle(newProbe, multiple*sigma, edgecolor='k', facecolor=[0.2,0.8,0.5], alpha=0.9, linewidth=0.5))
                        plt.pause(0.1)
            probe = newProbe
        previousNeighbors.append(contactId)
        if(theta < theta0):
            minDistance = 1
            for i in currentNeighbors[currentNeighbors!=-1]:
                if(checkedParticles[i] == 0):
                    distance = np.linalg.norm(utils.pbcDistance(pos[i], pos[contactId], boxSize))
                    if(distance < minDistance):
                        minDistance = distance
                        nextId = i
            if(minDistance == 1):
                #print("couldn't find another close particle within the distance cutoff - check all the particles in the cluster")
                for i in currentParticles:
                    if(checkedParticles[i] == 0):
                        distance = np.linalg.norm(utils.pbcDistance(pos[i], pos[contactId], boxSize))
                        if(distance < minDistance):
                            minDistance = distance
                            nextId = i
            contactId = nextId
            checkedParticles[contactId] += 1
            currentParticles = currentParticles[currentParticles!=contactId]
            #print("finished loop - switch to closest unchecked particle: ", contactId)
        nCheck += 1
        if(nCheck > 50):
            # check if the loop around the cluster is closed
            distance = np.linalg.norm(utils.pbcDistance(pos[contactId], pos[firstId], boxSize))
            if(distance < 4 * cutoff):
                contactId = firstId
    NpInCluster = clusterList[clusterList!=0].shape[0]
    clusterSize = np.sqrt(NpInCluster) * np.mean(rad[clusterList!=0])
    clusterArea = np.sum(np.pi*rad[clusterList!=0]**2)
    np.savetxt(dirName + os.sep + "clusterSize.dat", np.column_stack((borderLength, clusterSize, clusterArea)))
    np.savetxt(dirName + os.sep + "clusterBorderList.dat", borderParticles)
    print("border length: ", borderLength, " cluster size: ", clusterSize, " cluster density: ", clusterArea)
    if(plot=='plot'):
        plt.show()

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

############################## Delaunay inclusivity ############################
def checkDelaunay(dirName):
    sep = utils.getDirSep(dirName, "boxSize")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    delaunay = Delaunay(pos)
    utils.checkDelaunayInclusivity(delaunay.simplices, pos, rad, boxSize)

################### Cluster augmented packing for plotting #####################
def computeAugmentedDelaunayCluster(dirName, threshold=0.76, filter=False, shiftx=0, shifty=0, label=False):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    pos = utils.shiftPositions(pos, boxSize, shiftx, shifty)
    newPos, newRad, newIndices = utils.augmentPacking(pos, rad, 0.1, lx=boxSize[0], ly=boxSize[1])
    numParticles = newRad.shape[0]
    simplices = Delaunay(newPos).simplices
    simplices = np.unique(np.sort(simplices, axis=1), axis=0)
    insideIndex = utils.getInsideBoxDelaunaySimplices(simplices, newPos, boxSize)
    simplexDensity, _ = utils.computeDelaunayDensity(simplices, newPos, newRad, boxSize*1.1)
    denseSimplexList = np.zeros(simplexDensity.shape[0])
    # first find simplices above square lattice density 0.785
    for i in range(simplexDensity.shape[0]):
        if(simplexDensity[i] > threshold):
            denseSimplexList[i] = 1
    # save augmented packing
    dirAugment = dirName + os.sep + "augmented/"
    if(os.path.isdir(dirAugment)==False):
        os.mkdir(dirAugment)
    np.savetxt(dirAugment + os.sep + "augmentedPos.dat", newPos)
    np.savetxt(dirAugment + os.sep + "augmentedRad.dat", newRad)
    np.savetxt(dirAugment + os.sep + "simplices.dat", simplices)
    np.savetxt(dirAugment + os.sep + "insideIndex.dat", insideIndex)
    if(label == 'label'):
        dirLabel = dirAugment + os.sep + "delaunayLabels/"
        if(os.path.isdir(dirLabel)==False):
            os.mkdir(dirLabel)
        utils.labelDelaunaySimplices(dirLabel, simplices, denseSimplexList)
    # apply filters on triangles
    if(filter == 'filter'):
        for i in range(denseSimplexList.shape[0]):
            if(denseSimplexList[i] == 1 and insideIndex[i] == 1):
                indices = utils.findNeighborSimplices(simplices, i)
                if(np.sum(denseSimplexList[indices]) >= 1):
                    for j in indices:
                        if(denseSimplexList[j] == 1 and insideIndex[j] == 1):
                            secondIndices = utils.findNeighborSimplices(simplices, j)
                            if(np.sum(denseSimplexList[secondIndices]) <= 1):
                                denseSimplexList[i] = 0
        for times in range(3):
            for i in range(denseSimplexList.shape[0]):
                if(denseSimplexList[i] == 0 and insideIndex[i] == 1):
                    indices = utils.findNeighborSimplices(simplices, i)
                    if(np.sum(denseSimplexList[indices]) >= 2):
                        denseSimplexList[i] = 1
        if(label == 'label'):
            dirLabel = dirAugment + os.sep + "filterDelaunayLabels/"
            if(os.path.isdir(dirLabel)==False):
                os.mkdir(dirLabel)
            utils.labelDelaunaySimplices(dirLabel, simplices, denseSimplexList)
    # if one dense simplex is touching a particle then the particle is dense
    denseList = np.zeros(numParticles)
    for i in range(denseSimplexList.shape[0]):
        if(denseSimplexList[i]==1):
            denseList[simplices[i]] = 1
    # find simplices at the interface between dense and non-dense
    borderSimplexList = np.zeros(denseSimplexList.shape[0])
    for i in range(denseSimplexList.shape[0]):
        if(denseSimplexList[i]==1):
            indices = utils.findNeighborSimplices(simplices, i)
            if(np.sum(denseSimplexList[indices]) <= 2 and simplexDensity[i]):
                borderSimplexList[i] = 1
    for i in range(borderSimplexList.shape[0]):
        if(borderSimplexList[i] == 1):
            indices = utils.findNeighborSimplices(simplices, i)
            for j in indices:
                if(simplexDensity[j] > threshold):
                    borderSimplexList[j] = 1
    np.savetxt(dirAugment + os.sep + "borderSimplexList.dat", borderSimplexList)
    # label border particles if they belong to a border simplex
    borderList = np.zeros(numParticles)
    for i in range(borderSimplexList.shape[0]):
        if(borderSimplexList[i]==1):
            borderList[simplices[i]] = 1
    np.savetxt(dirAugment + os.sep + "borderParticleList.dat", borderList)
    if(filter == 'filter'):
        np.savetxt(dirAugment + os.sep + "denseSimplexList-filter.dat", denseSimplexList)
        np.savetxt(dirAugment + os.sep + "denseParticleList-filter.dat", denseList)
    else:
        np.savetxt(dirAugment + os.sep + "denseSimplexList.dat", denseSimplexList)
        np.savetxt(dirAugment + os.sep + "denseParticleList.dat", denseList)
    print("total # of simplices:", denseSimplexList.shape[0])
    print("# of dilute simplices:", denseSimplexList[denseSimplexList==0].shape[0])
    print("# of dense simplices:", denseSimplexList[denseSimplexList==1].shape[0])
    #return newPos, simplices, denseSimplexList, simplexDensity
    colorId = -(denseSimplexList - np.ones(denseSimplexList.shape[0])).astype(int)
    borderColorId = -(borderSimplexList - np.ones(borderSimplexList.shape[0])).astype(int)
    #return newPos, simplices, colorId, borderColorId
    return newPos, simplices[insideIndex==1], colorId[insideIndex==1], borderColorId[insideIndex==1]

############################## Delaunay clustering #############################
def computeDelaunayCluster(dirName, threshold=0.78, filter='filter', plot=False, label=False, save=False, LE=False, strain=0):
    sep = utils.getDirSep(dirName, "boxSize")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    if(LE=='LE'):
        pos = utils.getLEPBCPositions(dirName + "/particlePos.dat", boxSize, strain)
    else:
        pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    #delaunay = Delaunay(pos)
    #simplices = delaunay.simplices
    simplices = utils.getPBCDelaunay(pos, rad, boxSize)
    # compute delaunay densities
    simplexDensity, simplexArea = utils.computeDelaunayDensity(simplices, pos, rad, boxSize)
    #if(np.argwhere(simplexDensity<0)[:,0].shape[0] > 0):
    #    print(dirName)
    #    print("There are", np.argwhere(simplexDensity<0)[:,0].shape[0], "negative simplex densities")
    #    print(np.argwhere(simplexDensity<0)[:,0], simplices[np.argwhere(simplexDensity<0)[:,0]])
    denseSimplexList = np.zeros(simplexDensity.shape[0])
    # first find simplices above square lattice density 0.785
    for i in range(simplexDensity.shape[0]):
        if(simplexDensity[i] > threshold):
            denseSimplexList[i] = 1
    if(label == 'label'):
        dirLabel = dirName + os.sep + "delaunayLabels/"
        if(os.path.isdir(dirLabel)==False):
            os.mkdir(dirLabel)
        utils.labelDelaunaySimplices(dirLabel, simplices, denseSimplexList)
    # apply filters on triangles
    if(filter == 'filter'):
        for i in range(denseSimplexList.shape[0]):
            if(denseSimplexList[i] == 1):
                indices = utils.findNeighborSimplices(simplices, i)
                if(np.sum(denseSimplexList[indices]) >= 1):
                    for j in indices:
                        if(denseSimplexList[j] == 1):
                            secondIndices = utils.findNeighborSimplices(simplices, j)
                            if(np.sum(denseSimplexList[secondIndices]) <= 1):
                                denseSimplexList[i] = 0
        for times in range(3):
            for i in range(denseSimplexList.shape[0]):
                if(denseSimplexList[i] == 0):
                    indices = utils.findNeighborSimplices(simplices, i)
                    if(np.sum(denseSimplexList[indices]) >= 2):
                        denseSimplexList[i] = 1
        # first filter - label dilute simplices surrounded by dense simplices as dense
        #for i in range(denseSimplexList.shape[0]):
        #    if(denseSimplexList[i] == 0):
        #        indices = utils.findNeighborSimplices(simplices, i)
        #        if(np.sum(denseSimplexList[indices]) >= 2 and simplexDensity[i] > 0.45): # all are dense
        #            for j in indices:
        #                secondIndices = utils.findNeighborSimplices(simplices, j)
        #                if(np.sum(denseSimplexList[secondIndices]) >= 2 and denseSimplexList[i] == 0):
        #                    denseSimplexList[i] = 1
        # second filter - label dense simplices surrounded by dilute simplices as dilute - this filter is confirmed!
        #for i in range(denseSimplexList.shape[0]):
        #    if(denseSimplexList[i] == 1):
        #        indices = utils.findNeighborSimplices(simplices, i)
        #        if(np.sum(denseSimplexList[indices]) <= 1 and simplexDensity[i] < 0.84): # all are dilute, phiJ
        #            for j in indices:
        #                secondIndices = utils.findNeighborSimplices(simplices, j)
        #                if(np.sum(denseSimplexList[secondIndices]) <= 1 and denseSimplexList[i] == 1):
        #                    denseSimplexList[i] = 0
        if(label == 'label'):
            dirLabel = dirName + os.sep + "filterDelaunayLabels/"
            if(os.path.isdir(dirLabel)==False):
                os.mkdir(dirLabel)
            utils.labelDelaunaySimplices(dirLabel, simplices, denseSimplexList)
        #print("Fraction of dense simplices: ", denseSimplexList[denseSimplexList==1].shape[0]/denseSimplexList.shape[0])
    # if one dense simplex is touching a particle then the particle is dense
    denseList = np.zeros(numParticles)
    for i in range(denseSimplexList.shape[0]):
        if(denseSimplexList[i]==1):
            denseList[simplices[i]] = 1
    # find simplices at the interface between dense and non-dense
    borderSimplexList = np.zeros(denseSimplexList.shape[0])
    for i in range(denseSimplexList.shape[0]):
        if(denseSimplexList[i]==1):
            indices = utils.findNeighborSimplices(simplices, i)
            if(np.sum(denseSimplexList[indices]) <= 2):
                borderSimplexList[i] = 1
    for i in range(borderSimplexList.shape[0]):
        if(borderSimplexList[i] == 1):
            indices = utils.findNeighborSimplices(simplices, i)
            for j in indices:
                if(simplexDensity[j] > threshold):
                    borderSimplexList[j] = 1
    #denseSimplexList[borderSimplexList==1] = 0 # do not include border simplices in dense simplices
    # label border particles if they belong to a border simplex
    borderList = np.zeros(numParticles)
    for i in range(borderSimplexList.shape[0]):
        if(borderSimplexList[i]==1):
            borderList[simplices[i]] = 1
    #print("Fraction of dense particles: ", denseList[denseList==1].shape[0]/denseList.shape[0])
    if(filter=='particle-filter'):
        contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
        denseList, denseSimplexList = utils.applyParticleFilters(contacts, denseList, simplices, denseSimplexList)
    particleList = np.column_stack((denseList, borderList)).astype(np.int64)
    simplexList = np.column_stack((denseSimplexList, borderSimplexList, simplexArea, simplexDensity)).astype(np.int64)
    np.savetxt(dirName + "/particleList.dat", particleList)
    np.savetxt(dirName + "/simplexList.dat", simplexList)
    if(save=='save'):
        np.savetxt(dirName + os.sep + 'simplices.dat', simplices)
    #print("average density of dense simplices:", np.mean(simplexDensity[denseSimplexList==1]), np.min(simplexDensity[denseSimplexList==1]), np.max(simplexDensity[denseSimplexList==1]))
    if(plot=='plot'):
        uplot.plotCorrelation(np.arange(1, simplexDensity.shape[0]+1, 1), np.sort(simplexDensity), "$\\varphi^{Simplex}$", xlabel = "$Simplex$ $index$", color='k')
        plt.show()
        numBins = 100
        pdf, edges = np.histogram(simplexDensity, bins=np.linspace(0, 1, numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        uplot.plotCorrelation(edges, pdf, "$PDF(\\varphi^{Simplex})$", xlabel = "$\\varphi^{Simplex}$", color='b')
        #plt.yscale('log')
        plt.show()
    return particleList, simplexList

######################## Average delaunay local density #########################
def averageLocalDelaunayDensity(dirName, plot=False, dirSpacing=1):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    localDensity = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample)
        simplexDensity = np.loadtxt(dirSample + os.sep + "simplexList.dat")[:,3]
        localDensity = np.append(localDensity, simplexDensity)
    localDensity = np.sort(localDensity).flatten()
    localDensity = localDensity[localDensity>0]
    alpha2 = np.mean(localDensity**4)/(2*(np.mean(localDensity**2)**2)) - 1
    pdf, edges = np.histogram(localDensity, bins=np.linspace(np.min(localDensity), np.max(localDensity), 100), density=True)
    edges = (edges[:-1] + edges[1:])/2
    np.savetxt(dirName + os.sep + "localDelaunayDensity.dat", np.column_stack((edges, pdf)))
    data = np.column_stack((np.mean(localDensity), np.var(localDensity), alpha2))
    np.savetxt(dirName + os.sep + "localDelaunayDensity-stats.dat", data)
    print("average local density: ", np.mean(localDensity), " +- ", np.var(localDensity))
    if(plot == 'plot'):
        uplot.plotCorrelation(edges, pdf, "$Local$ $density$ $distribution$", "$Local$ $density$", color='k')
        plt.pause(0.5)
    return np.mean(localDensity)

########################### Compute delaunay density ###########################
def computeClusterDelaunayDensity(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    delaunayDensity = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        simplexArea = simplexList[:,2]
        simplexDensity = simplexList[:,3]
        occupiedArea = simplexDensity * simplexArea
        fluidArea = 0
        occupiedFluidArea = 0
        gasArea = 0
        occupiedGasArea = 0
        for i in range(simplexDensity.shape[0]):
            if(simplexDensity[i] > 0 and simplexDensity[i] < 1):
                if(denseSimplexList[i]==1):
                    fluidArea += simplexArea[i]
                    occupiedFluidArea += occupiedArea[i]
                else:
                    gasArea += simplexArea[i]
                    occupiedGasArea += occupiedArea[i]
        if(fluidArea > 0):
            delaunayDensity[d,0] = occupiedFluidArea / fluidArea
        else:
            delaunayDensity[d,0] = 0
        if(gasArea > 0):
            delaunayDensity[d,1] = occupiedGasArea / gasArea
        else:
            delaunayDensity[d,1] = 0
        delaunayDensity[d,2] = (occupiedFluidArea + occupiedGasArea) / (fluidArea + gasArea)
    np.savetxt(dirName + os.sep + "delaunayDensity.dat", np.column_stack((timeList, delaunayDensity)))
    print("Density in the fluid: ", np.mean(delaunayDensity[:,0]), " +- ", np.std(delaunayDensity[:,0]))
    print("Density in the gas: ", np.mean(delaunayDensity[:,1]), " +- ", np.std(delaunayDensity[:,1]))
    print("Density in the whole system: ", np.mean(delaunayDensity[:,2]), " +- ", np.std(delaunayDensity[:,2]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, delaunayDensity[:,0], "$\\varphi^{Delaunay}$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, delaunayDensity[:,1], "$\\varphi^{Delaunay}$", xlabel = "$Time,$ $t$", color='g')
        uplot.plotCorrelation(timeList, delaunayDensity[:,2], "$\\varphi^{Delaunay}$", xlabel = "$Time,$ $t$", color='k')
        plt.pause(0.5)
        #plt.show()
    return np.column_stack((timeList, delaunayDensity))

######################## Compute cluster shape parameter #######################
def computeClusterDelaunayArea(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    area = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        # area
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        borderSimplexList = simplexList[:,1]
        simplexArea = simplexList[:,2]
        simplexDensity = simplexList[:,3]
        occupiedArea = simplexDensity * simplexArea
        for i in range(simplexDensity.shape[0]):
            if(denseSimplexList[i]==1 and borderSimplexList[i]==0):
                area[d,0] += simplexArea[i]
            elif(denseSimplexList[i]==0 and borderSimplexList[i]==0):
                area[d,1] += simplexArea[i]
    np.savetxt(dirName + os.sep + "delaunayArea.dat", np.column_stack((timeList, area)))
    print("Fluid area: ", np.mean(area[:,0]), " +- ", np.std(area[:,0]))
    print("Gas area: ", np.mean(area[:,1]), " +- ", np.std(area[:,1]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, area[:,0], "$Area$ $fraction$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, area[:,1], "$Area$ $fraction$", xlabel = "$Time,$ $t$", color='g')
        plt.pause(0.5)
    return area

######################## Compute delaunay cluster border #######################
def computeDelaunayBorderLength(dirName, threshold=0.76, filter='filter'):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    phi = utils.readFromParams(dirName + sep, "phi")
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    # need to center the cluster for voronoi border detection
    if not(os.path.exists(dirName + os.sep + "particleList.dat")):
        computeDelaunayCluster(dirName)
    particleList = np.loadtxt(dirName + os.sep + "particleList.dat")
    borderList = particleList[:,1]
    #borderPos = pos[borderList==1]
    #borderPos = utils.sortBorderPos(borderPos, borderList, boxSize)
    #np.savetxt(dirName + os.sep + "borderPos.dat", borderPos)
    # compute border length by summing over segments on the border
    #borderLength = 0
    #for i in range(1,borderPos.shape[0]):
    #    borderLength += np.linalg.norm(utils.pbcDistance(borderPos[i], borderPos[i-1], boxSize))
    borderRad = rad[borderList==1]
    borderLength = np.sum(borderRad)
    #print("Number of dense particles at the interface: ", borderList[borderList==1].shape[0])
    #print("Border length from delaunay edges: ", borderLength)
    return borderLength

######################## Compute cluster shape parameter #######################
def computeClusterDelaunayShape(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    eps = np.max(rad) # different from eps for particles
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    clusterShape = np.empty(0)
    for d in range(dirList.shape[0]):
        #print(dirList[d])
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + os.sep + "simplices.dat")):
            computeDelaunayCluster(dirSample, save='save')
        simplices = np.loadtxt(dirSample + os.sep + 'simplices.dat').astype(np.int64)
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        simplexArea = simplexList[:,2]
        simplexArea = simplexArea[denseSimplexList==1]
        # compute simplex positions for clustering algorithm
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if not(os.path.exists(dirSample + os.sep + "simplexLabels.dat")):
            simplexPos = utils.computeSimplexPos(simplices, pos)
            labels = utils.getDBClusterLabels(simplexPos, boxSize, eps, min_samples=1, denseList=denseSimplexList)
            numLabels = np.unique(labels).shape[0]-1
            allLabels = -1*np.ones(denseSimplexList.shape[0], dtype=np.int64)
            allLabels[denseSimplexList==1] = labels
            np.savetxt(dirSample + os.sep + "simplexLabels.dat", allLabels)
        else:
            allLabels = np.loadtxt(dirSample + os.sep + "simplexLabels.dat")
            labels = allLabels[denseSimplexList==1]
        uniqueLabels = np.unique(labels).astype(np.int64)
        for label in uniqueLabels:
            clusterArea = np.sum(simplexArea[labels==label])
            # find cluster perimeter
            clusterPerimeter = 0
            for sIndex in np.argwhere(allLabels==label)[:,0]:
                indices = utils.findNeighborSimplices(simplices, sIndex)
                for idx in indices:
                    if(allLabels[idx] != label): # find the common edge and count it as part of the perimeter
                        edgeIndices = np.intersect1d(simplices[sIndex], simplices[idx])
                        clusterPerimeter += utils.pbcDistance(pos[edgeIndices[0]], pos[edgeIndices[1]], boxSize)
        clusterShape = np.append(clusterShape, clusterPerimeter**2/(4*np.pi*clusterArea))
        #print(np.min(clusterShape), np.max(clusterShape))
    clusterShape = clusterShape[clusterShape>0]
    clusterShape = clusterShape[np.argsort(clusterShape)]
    np.savetxt(dirName + os.sep + "clusterShape.dat", clusterShape)
    print("Cluster shape: ", np.mean(clusterShape), " +- ", np.std(clusterShape))
    if(plot=='plot'):
        # time plots
        uplot.plotCorrelation(np.arange(1, clusterShape.shape[0]+1, 1), np.sort(clusterShape), "$Cluster$ $shape$", xlabel = "$index$", color='k', logy=True)
        plt.pause(0.5)
        #plt.show()

########################### Compute delaunay density ###########################
def computeDelaunayClusterVel(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    clusterVel = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        if not(os.path.exists(dirSample + os.sep + "particleList.dat")):
            computeDelaunayCluster(dirSample)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        clusterVel[d,0] = np.mean(np.linalg.norm(vel, axis=1))
        clusterVel[d,1] = np.mean(np.linalg.norm(vel[denseList==1], axis=1))
        clusterVel[d,2] = np.mean(np.linalg.norm(vel[denseList==0], axis=1))
    np.savetxt(dirName + os.sep + "clusterVel.dat", np.column_stack((timeList, clusterVel)))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, clusterVel[:,2], "$l_p$", xlabel = "$Time,$ $t$", color='k')
        plt.pause(0.5)
    return np.column_stack((timeList, clusterVel))

def getParticleClusterLabels(dirSample, boxSize, eps, threshold=0.3, compute=False, save='save'):
    if(compute==True or compute=='cluster'):
        computeDelaunayCluster(dirSample, threshold=threshold, save=save)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
    if(compute==True or compute=='label'):
        if not(os.path.exists(dirSample + os.sep + "particleList.dat")):
            computeDelaunayCluster(dirSample, threshold=threshold, save=save)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = utils.getDBClusterLabels(pos, boxSize, eps, min_samples=2, denseList=denseList)
        allLabels = -1*np.ones(denseList.shape[0], dtype=np.int64)
        allLabels[denseList==1] = labels
        allLabels = allLabels.astype(np.int64)
        np.savetxt(dirSample + os.sep + "clusterLabels.dat", allLabels)
    else:
        if not(os.path.exists(dirSample + os.sep + "particleList.dat")):
            computeDelaunayCluster(dirSample, threshold=threshold, save=save)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        if not(os.path.exists(dirSample + os.sep + "clusterLabels.dat")):
            # compute simplex positions for clustering algorithm
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            labels = utils.getDBClusterLabels(pos, boxSize, eps, min_samples=2, denseList=denseList)
            allLabels = -1*np.ones(denseList.shape[0])
            allLabels[np.argwhere(denseList==1)[:,0]] = labels
            allLabels = allLabels.astype(np.int64)
            np.savetxt(dirSample + os.sep + "clusterLabels.dat", allLabels)
        allLabels = np.loadtxt(dirSample + os.sep + "clusterLabels.dat")
    return allLabels

def getParticleDenseLabel(dirSample, threshold=0.3, compute=False, save='save'):
    if(compute==True):
        computeDelaunayCluster(dirSample, threshold=threshold, save=save)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        borderList = particleList[:,1]
    else:
        if not(os.path.exists(dirSample + os.sep + "particleList.dat")):
            computeDelaunayCluster(dirSample, threshold=threshold, save=save)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        borderList = particleList[:,1]
    return denseList, borderList

def getLEParticleClusterLabels(dirSample, boxSize, eps, threshold=0.76, compute=False, save='save', strain=0):
    if(compute==True):
        computeDelaunayCluster(dirSample, threshold=threshold, save=save, LE='LE', strain=strain)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        pos = utils.getLEPBCPositions(dirSample + "/particlePos.dat", boxSize, strain)
        labels = utils.getDBClusterLabels(pos, boxSize, eps, min_samples=2, denseList=denseList)
        allLabels = -1*np.ones(denseList.shape[0], dtype=np.int64)
        allLabels[denseList==1] = labels
        np.savetxt(dirSample + os.sep + "clusterLabels.dat", allLabels)
    else:
        if not(os.path.exists(dirSample + os.sep + "particleList.dat")):
            computeDelaunayCluster(dirSample, threshold=threshold, save=save, LE='LE', strain=strain)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        if not(os.path.exists(dirSample + os.sep + "clusterLabels.dat")):
            # compute simplex positions for clustering algorithm
            pos = utils.getLEPBCPositions(dirSample + "/particlePos.dat", boxSize, strain)
            labels = utils.getDBClusterLabels(pos, boxSize, eps, min_samples=2, denseList=denseList)
            allLabels = -1*np.ones(denseList.shape[0], dtype=np.int64)
            allLabels[denseList==1] = labels
            np.savetxt(dirSample + os.sep + "clusterLabels.dat", allLabels)
        allLabels = np.loadtxt(dirSample + os.sep + "clusterLabels.dat")
    return allLabels

def getLEParticleDenseLabel(dirSample, threshold=0.76, compute=False, save='save', strain=0):
    if(compute==True):
        computeDelaunayCluster(dirSample, threshold=threshold, save=save, LE='LE', strain=strain)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        borderList = particleList[:,1]
    else:
        if not(os.path.exists(dirSample + os.sep + "particleList.dat")):
            computeDelaunayCluster(dirSample, threshold=threshold, save=save, LE='LE', strain=strain)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        borderList = particleList[:,1]
    return denseList, borderList

############################# Cluster distribution #############################
def computeDelaunayClusterDistribution(dirName, threshold=0.76, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    phi = int(utils.readFromParams(dirName, "phi"))
    rad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    eps = 1.8*np.max(rad)
    boxSize = np.loadtxt(dirName + "/boxSize.dat")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    clusterNumber = np.empty(0)
    clusterArea = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        uniqueLabels = np.unique(labels).astype(np.int64)
        for label in uniqueLabels:
            if(label!=-1):
                clusterNumber = np.append(clusterNumber, labels[labels==label].shape[0] / labels.shape[0])
                clusterArea = np.append(clusterArea, np.sum(rad[labels==label]**2)*np.pi)
    # in cluster
    clusterArea = clusterArea[clusterNumber>0]
    clusterNumber = clusterNumber[clusterNumber>0]
    clusterArea = clusterArea[np.argsort(clusterNumber)]
    clusterNumber = np.sort(clusterNumber)
    np.savetxt(dirName + os.sep + "clusterDistribution.dat", np.column_stack((clusterNumber, clusterArea)))
    weight = clusterArea / np.sum(clusterArea)
    print("Average fraction of particles in cluster: ", np.sum(clusterNumber * weight), " +- ", np.std(clusterNumber) * np.sqrt(np.sum(weight)))
    if(plot == 'plot'):
        numBins = 50
        pdf, edges = np.histogram(clusterArea, bins=np.geomspace(np.min(clusterArea), np.max(clusterArea), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        uplot.plotCorrelation(edges[pdf>0], pdf[pdf>0], "$PDF(A_c)$", xlabel = "$A_c$", color='k', logx=True, logy=True)
        #plt.show()
        plt.pause(0.5)

def getSimplexClusterLabels(dirSample, boxSize, eps, threshold=0.76, compute=False, save='save'):
    if(compute==True):
        computeDelaunayCluster(dirSample, threshold=threshold, save=save)
        simplices = np.loadtxt(dirSample + os.sep + 'simplices.dat').astype(np.int64)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        simplexArea = simplexList[:,2]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        simplexPos = utils.computeSimplexPos(simplices, pos)
        labels = utils.getDBClusterLabels(simplexPos, boxSize, eps, min_samples=1, denseList=denseSimplexList)
        allLabels = -1*np.ones(denseSimplexList.shape[0], dtype=np.int64)
        allLabels[denseSimplexList==1] = labels
        np.savetxt(dirSample + os.sep + "simplexLabels.dat", allLabels)
    else:
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample, threshold=threshold, save=save)
        simplices = np.loadtxt(dirSample + os.sep + 'simplices.dat').astype(np.int64)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        simplexArea = simplexList[:,2]
        if not(os.path.exists(dirSample + os.sep + "simplexLabels.dat")):
            # compute simplex positions for clustering algorithm
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            simplexPos = utils.computeSimplexPos(simplices, pos)
            labels = utils.getDBClusterLabels(simplexPos, boxSize, eps, min_samples=1, denseList=denseSimplexList)
            allLabels = -1*np.ones(denseSimplexList.shape[0], dtype=np.int64)
            allLabels[denseSimplexList==1] = labels
            np.savetxt(dirSample + os.sep + "simplexLabels.dat", allLabels)
        allLabels = np.loadtxt(dirSample + os.sep + "simplexLabels.dat")
    return allLabels, simplexArea

######################### Simplex cluster distribution #########################
def computeSimplexClusterDistribution(dirName, threshold=0.76, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    eps = np.max(rad) # different from eps for particles
    boxSize = np.loadtxt(dirName + "/boxSize.dat")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    clusterNumber = np.empty(0)
    clusterArea = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        labels, simplexArea = getSimplexClusterLabels(dirSample, boxSize, eps, threshold, compute=True)
        uniqueLabels = np.unique(labels).astype(np.int64)
        for label in uniqueLabels:
            if(label!=-1):
                clusterNumber = np.append(clusterNumber, labels[labels==label].shape[0] / labels.shape[0])
                clusterArea = np.append(clusterArea, np.sum(simplexArea[labels==label]))
    # in cluster
    clusterArea = clusterArea[clusterNumber>0]
    clusterNumber = clusterNumber[clusterNumber>0]
    clusterArea = clusterArea[np.argsort(clusterNumber)]
    clusterNumber = np.sort(clusterNumber)
    np.savetxt(dirName + os.sep + "simplexDistribution.dat", np.column_stack((clusterNumber, clusterArea)))
    print("Average cluster area: ", np.mean(clusterArea), " +- ", np.std(clusterArea), np.std(clusterArea) / np.mean(clusterArea))
    if(plot == 'plot'):
        numBins = 20
        pdf, edges = np.histogram(clusterNumber, bins=np.geomspace(np.min(clusterNumber), np.max(clusterNumber), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        uplot.plotCorrelation(edges[pdf>0], pdf[pdf>0], "$PDF(N_c)$", xlabel = "$N_c$", color='k', logx=True, logy=True)
        plt.show()
        #plt.pause(0.5)

######################### Simplex cluster distribution #########################
def computeClusterSizeVSTime(dirName, threshold=0.76, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    eps = np.max(rad) # different from eps for particles
    boxSize = np.loadtxt(dirName + "/boxSize.dat")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    clusterNumber = np.zeros((dirList.shape[0],2))
    clusterArea = np.zeros((dirList.shape[0],2))
    clusterSize = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        labels, simplexArea = getSimplexClusterLabels(dirSample, boxSize, eps, threshold)
        uniqueLabels = np.unique(labels).astype(np.int64)
        number = np.empty(0)
        area = np.empty(0)
        size = np.empty(0)
        for label in uniqueLabels:
            if(label!=-1):
                number = np.append(number, labels[labels==label].shape[0] / labels.shape[0])
                area = np.append(area, np.sum(simplexArea[labels==label]))
                size = np.append(size, np.sqrt(np.sum(simplexArea[labels==label]) / np.pi))
        clusterNumber[d,0] = np.mean(number)
        clusterNumber[d,1] = np.std(number)
        clusterArea[d,0] = np.mean(area)
        clusterArea[d,1] = np.std(area)
        clusterSize[d,0] = np.mean(size)
        clusterSize[d,1] = np.std(size)
    np.savetxt(dirName + os.sep + "clusterSizeVSTime.dat", np.column_stack((timeList, clusterSize, clusterArea, clusterNumber)))
    if(plot == 'plot'):
        uplot.plotCorrWithError(timeList, clusterSize[:,0], clusterSize[:,1], "$\\langle l_c \\rangle$", xlabel = "$Time,$ $t$", color='k', logx=True, logy=True)
        #uplot.plotCorrWithError(timeList, clusterArea[:,0], clusterArea[:,1], "$\\langle A_c \\rangle$", xlabel = "$Time,$ $t$", color='k')
        #uplot.plotCorrWithError(timeList, clusterNumber[:,0], clusterNumber[:,1], "$\\langle f_c \\rangle$", xlabel = "$Time,$ $t$", color='k')
        plt.show()
        #plt.pause(0.5)

########################### Pair Correlation Function ##########################
def computeClusterPairCorr(dirName, boxSize, bins, labels, maxLabel, plot="plot", which="dense", save="save"):
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    # select cluster positions
    pos1 = pos[labels==maxLabel]
    pos2 = pos[labels!=maxLabel]
    distance = utils.computeDistances(pos1, boxSize)
    pairCorr1, edges = np.histogram(distance, bins=bins, density=True)
    distance = utils.computeDistances(pos2, boxSize)
    pairCorr2, edges = np.histogram(distance, bins=bins, density=True)
    binCenter = 0.5 * (edges[:-1] + edges[1:])
    pairCorr1 /= (2 * np.pi * binCenter)
    pairCorr2 /= (2 * np.pi * binCenter)
    firstPeak = binCenter[np.argmax(pairCorr1)]
    if(save == "save"):
        #print("First peak of pair corr is at distance:", firstPeak)
        if(which=="dense"):
            np.savetxt(dirName + os.sep + "densePairCorr.dat", np.column_stack((binCenter, pairCorr1, pairCorr2)))
        else:
            np.savetxt(dirName + os.sep + "clusterPairCorr.dat", np.column_stack((binCenter, pairCorr1, pairCorr2)))
    else:
        return pairCorr1, pairCorr2
    if(plot == "plot"):
        uplot.plotCorrelation(binCenter, pairCorr1, "$g(r)$", color='b')
        uplot.plotCorrelation(binCenter, pairCorr2, "$g(r)$", color='g')
        plt.pause(0.5)
    else:
        return firstPeak

def averageClusterPairCorr(dirName, threshold, lj='lj', dirSpacing=1):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat").astype(np.float64)
    eps = 1.8*np.max(rad)
    meanRad = np.mean(rad)
    bins = np.linspace(0.1*meanRad, 10*meanRad, 50)
    if(lj=='lj'):
        rad *= 2**(1/6)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    labels = getParticleClusterLabels(dirName, boxSize, eps, threshold)
    maxLabel = utils.findLargestParticleCluster(rad, labels)
    pcorr = np.zeros((dirList.shape[0], bins.shape[0]-1, 2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + os.sep + "clusterPairCorr!.dat")):
            computeClusterPairCorr(dirSample, boxSize, bins, labels, maxLabel, plot=False)
        data = np.loadtxt(dirSample + os.sep + "clusterPairCorr.dat")
        pcorr[d,:,0] = data[:,1]
        pcorr[d,:,1] = data[:,2]
    pcorr1 = np.column_stack((np.mean(pcorr[:,:,0], axis=0), np.std(pcorr[:,:,0], axis=0)))
    pcorr2 = np.column_stack((np.mean(pcorr[:,:,1], axis=0), np.std(pcorr[:,:,1], axis=0)))
    firstPeak = bins[np.argmax(pcorr1[:,0])]
    print("First peak of pair corr in cluster is at:", firstPeak)
    binCenter = 0.5 * (bins[:-1] + bins[1:])
    np.savetxt(dirName + os.sep + "clusterPairCorr.dat", np.column_stack((binCenter, pcorr1, pcorr2)))
    uplot.plotCorrWithError(binCenter, pcorr1[:,0], pcorr1[:,1], "$g(r/\\sigma)$", "$r/\\sigma$", color='b')
    uplot.plotCorrWithError(binCenter, pcorr2[:,0], pcorr2[:,1], "$g(r/\\sigma)$", "$r/\\sigma$", color='g')
    plt.pause(0.5)
    #plt.show()
    
def averageDensePairCorr(dirName, threshold=0.3, dirSpacing=1):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat").astype(np.float64)
    meanRad = np.mean(rad)
    bins = np.linspace(0.1*meanRad, 10*meanRad, 50)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    denseList,borderList = getParticleDenseLabel(dirName, threshold)
    pcorr = np.zeros((dirList.shape[0], bins.shape[0]-1,2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + os.sep + "densePairCorr!.dat")):
            computeClusterPairCorr(dirSample, boxSize, bins, denseList, 1, plot=False, which="dense")
        data = np.loadtxt(dirSample + os.sep + "densePairCorr.dat")
        pcorr[d,:,0] = data[:,1]
        pcorr[d,:,1] = data[:,2]
    pcorr1 = np.column_stack((np.mean(pcorr[:,:,0], axis=0), np.std(pcorr[:,:,0], axis=0)))
    pcorr2 = np.column_stack((np.mean(pcorr[:,:,1], axis=0), np.std(pcorr[:,:,1], axis=0)))
    firstPeak = bins[np.argmax(pcorr1[:,0])]
    print("First peak of pair corr in cluster is at:", firstPeak)
    binCenter = 0.5 * (bins[:-1] + bins[1:])
    np.savetxt(dirName + os.sep + "densePairCorr.dat", np.column_stack((binCenter, pcorr1, pcorr2)))
    uplot.plotCorrWithError(binCenter, pcorr1[:,0], pcorr1[:,1], "$g(r/\\sigma)$", "$r/\\sigma$", color='b')
    uplot.plotCorrWithError(binCenter, pcorr2[:,0], pcorr2[:,1], "$g(r/\\sigma)$", "$r/\\sigma$", color='g')
    plt.pause(0.5)
    #plt.show()

def getClusterContactCollisionIntervalPDF(dirName, check=False, numBins=40):
    timeStep = utils.readFromParams(dirName, "dt")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    if(os.path.exists(dirName + "/contactCollisionIntervals.dat") and check=="check"):
        print("loading already existing file")
        intervalInCluster = np.loadtxt(dirName + os.sep + "inClusterCollisionIntervals.dat")
        intervalOutCluster = np.loadtxt(dirName + os.sep + "outClusterCollisionIntervals.dat")
    else:
        intervalInCluster = np.empty(0)
        intervalOutCluster = np.empty(0)
        previousTime = np.zeros(numParticles)
        previousContacts = np.array(np.loadtxt(dirName + os.sep + "t0/particleContacts.dat"))
        for d in range(1,dirList.shape[0]):
            dirSample = dirName + os.sep + dirList[d]
            if(os.path.exists(dirSample + os.sep + "clusterLabels.dat")):
                clusterLabels = np.loadtxt(dirSample + os.sep + "clusterLabels.dat")[:,0]
                noClusterLabels = np.loadtxt(dirSample + os.sep + "clusterLabels.dat")[:,1]
            else:
                #clusterLabels, noClusterLabels,_ = searchDBClusters(dirSample, eps=0, min_samples=10)
                clusterLabels, noClusterLabels,_ = searchClusters(dirSample, numParticles)
            particlesInClusterIndex = np.argwhere(clusterLabels==1)[:,0]
            particlesOutClusterIndex = np.argwhere(noClusterLabels==1)[:,0]
            currentTime = timeList[d]
            currentContacts = np.array(np.loadtxt(dirSample + "/particleContacts.dat"), dtype=np.int64)
            colIndex = np.unique(np.argwhere(currentContacts!=previousContacts)[:,0])
            # in cluster collisions
            colIndexInCluster = np.intersect1d(colIndex, particlesInClusterIndex)
            currentInterval = currentTime-previousTime[colIndexInCluster]
            intervalInCluster = np.append(intervalInCluster, currentInterval[currentInterval>1])
            previousTime[colIndexInCluster] = currentTime
            # out cluster collisions
            colIndexOutCluster = np.intersect1d(colIndex, particlesOutClusterIndex)
            currentInterval = currentTime-previousTime[colIndexOutCluster]
            intervalOutCluster = np.append(intervalOutCluster, currentInterval[currentInterval>1])
            previousTime[colIndexOutCluster] = currentTime
            previousContacts = currentContacts
        intervalInCluster = np.sort(intervalInCluster)
        intervalInCluster *= timeStep
        np.savetxt(dirName + os.sep + "inClusterCollisionIntervals.dat", intervalInCluster)
        intervalOutCluster = np.sort(intervalOutCluster)
        intervalOutCluster *= timeStep
        np.savetxt(dirName + os.sep + "outClusterCollisionIntervals.dat", intervalOutCluster)
    # in cluster collision distribution
    bins = np.arange(np.min(intervalInCluster), np.max(intervalInCluster), 5*np.min(intervalInCluster))
    pdf, edges = np.histogram(intervalInCluster, bins=bins, density=True)
    centers = (edges[1:] + edges[:-1])/2
    print("average collision time in cluster:", np.mean(intervalInCluster), " standard deviation: ", np.std(intervalInCluster))
    np.savetxt(dirName + os.sep + "inClusterCollision.dat", np.column_stack((centers, pdf)))
    uplot.plotCorrelation(centers, pdf, "$PDF(\\Delta_c)$", "$Time$ $between$ $collisions,$ $\\Delta_c$", logy=True, color='g')
    # out cluster collision distribution
    bins = np.arange(np.min(intervalOutCluster), np.max(intervalOutCluster), 5*np.min(intervalOutCluster))
    pdf, edges = np.histogram(intervalOutCluster, bins=bins, density=True)
    centers = (edges[1:] + edges[:-1])/2
    print("average collision time in cluster:", np.mean(intervalOutCluster), " standard deviation: ", np.std(intervalOutCluster))
    np.savetxt(dirName + os.sep + "outClusterCollision.dat", np.column_stack((centers, pdf)))
    uplot.plotCorrelation(centers, pdf, "$PDF(\\Delta_c)$", "$Time$ $between$ $collisions,$ $\\Delta_c$", logy=True, color='k')
    print("max time: ", timeList[-1]*timeStep, " max interval: ", np.max([np.max(intervalInCluster), np.max(intervalOutCluster)]))

def computeParticleVelSpaceCorrCluster(dirName):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    bins = np.arange(np.min(rad), np.sqrt(2)*boxSize[0]/2, 2*np.mean(rad))
    velCorrInCluster = np.zeros((bins.shape[0]-1,4))
    countsInCluster = np.zeros(bins.shape[0]-1)
    velCorrOutCluster = np.zeros((bins.shape[0]-1,4))
    countsOutCluster = np.zeros(bins.shape[0]-1)
    if(os.path.exists(dirName + os.sep + "dbClusterLabels.dat")):
        clusterLabels = np.loadtxt(dirName + os.sep + "dbClusterLabels.dat")[:,0]
        noClusterLabels = np.loadtxt(dirName + os.sep + "dbClusterLabels.dat")[:,1]
    else:
        clusterLabels, noClusterLabels,_ = searchDBClusters(dirName, eps=0, min_samples=10)
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    distance = utils.computeDistances(pos, boxSize)
    vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
    velNorm = np.linalg.norm(vel, axis=1)
    vel[:,0] /= velNorm
    vel[:,1] /= velNorm
    velNormSquared = np.mean(velNorm**2)
    for i in range(distance.shape[0]):
        for j in range(i):
                for k in range(bins.shape[0]-1):
                    if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                        # parallel
                        delta = utils.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
                        parProj1 = np.dot(vel[i],delta)
                        parProj2 = np.dot(vel[j],delta)
                        # perpendicular
                        deltaPerp = np.array([-delta[1], delta[0]])
                        perpProj1 = np.dot(vel[i],deltaPerp)
                        perpProj2 = np.dot(vel[j],deltaPerp)
                        # correlations
                        if(clusterLabels[i]==1):
                            velCorrInCluster[k,0] += parProj1 * parProj2
                            velCorrInCluster[k,1] += perpProj1 * perpProj2
                            velCorrInCluster[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                            velCorrInCluster[k,3] += np.dot(vel[i],vel[j])
                            countsInCluster[k] += 1
                        if(noClusterLabels[i]==1):
                            velCorrOutCluster[k,0] += parProj1 * parProj2
                            velCorrOutCluster[k,1] += perpProj1 * perpProj2
                            velCorrOutCluster[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                            velCorrOutCluster[k,3] += np.dot(vel[i],vel[j])
                            countsOutCluster[k] += 1
    binCenter = (bins[1:] + bins[:-1])/2
    for i in range(velCorrInCluster.shape[1]):
        velCorrInCluster[countsInCluster>0,i] /= countsInCluster[countsInCluster>0]
        velCorrOutCluster[countsOutCluster>0,i] /= countsOutCluster[countsOutCluster>0]
    np.savetxt(dirName + os.sep + "spaceVelCorrInCluster1.dat", np.column_stack((binCenter, velCorrInCluster, countsInCluster)))
    np.savetxt(dirName + os.sep + "spaceVelCorrOutCluster1.dat", np.column_stack((binCenter, velCorrOutCluster, countsOutCluster)))

def averageParticleVelSpaceCorrCluster(dirName, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    bins = np.arange(np.min(rad)/2, np.sqrt(2)*boxSize[0]/2, 2*np.mean(rad))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    #dirList = np.array([dirName])
    velCorrInCluster = np.zeros((bins.shape[0]-1,4))
    countsInCluster = np.zeros(bins.shape[0]-1)
    velCorrOutCluster = np.zeros((bins.shape[0]-1,4))
    countsOutCluster = np.zeros(bins.shape[0]-1)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "denseList.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
        else:
            denseList = computeVoronoiCluster(dirSample)
        pos = np.array(np.loadtxt(dirSample + os.sep + "particlePos.dat"))
        distance = utils.computeDistances(pos, boxSize)
        vel = np.array(np.loadtxt(dirSample + os.sep + "particleVel.dat"))
        velNorm = np.linalg.norm(vel, axis=1)
        vel[:,0] /= velNorm
        vel[:,1] /= velNorm
        velNormSquared = np.mean(velNorm**2)
        for i in range(distance.shape[0]):
            for j in range(i):
                    for k in range(bins.shape[0]-1):
                        if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                            # parallel
                            delta = utils.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
                            parProj1 = np.dot(vel[i],delta)
                            parProj2 = np.dot(vel[j],delta)
                            # perpendicular
                            deltaPerp = np.array([-delta[1], delta[0]])
                            perpProj1 = np.dot(vel[i],deltaPerp)
                            perpProj2 = np.dot(vel[j],deltaPerp)
                            # correlations
                            if(denseList[i]==1):
                                velCorrInCluster[k,0] += parProj1 * parProj2
                                velCorrInCluster[k,1] += perpProj1 * perpProj2
                                velCorrInCluster[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                                velCorrInCluster[k,3] += np.dot(vel[i],vel[j])
                                countsInCluster[k] += 1
                            else:
                                velCorrOutCluster[k,0] += parProj1 * parProj2
                                velCorrOutCluster[k,1] += perpProj1 * perpProj2
                                velCorrOutCluster[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                                velCorrOutCluster[k,3] += np.dot(vel[i],vel[j])
                                countsOutCluster[k] += 1
    binCenter = (bins[1:] + bins[:-1])/2
    for i in range(velCorrInCluster.shape[1]):
        velCorrInCluster[countsInCluster>0,i] /= countsInCluster[countsInCluster>0]
        velCorrOutCluster[countsOutCluster>0,i] /= countsOutCluster[countsOutCluster>0]
    np.savetxt(dirName + os.sep + "spaceVelCorrInCluster.dat", np.column_stack((binCenter, velCorrInCluster, countsInCluster)))
    np.savetxt(dirName + os.sep + "spaceVelCorrOutCluster.dat", np.column_stack((binCenter, velCorrOutCluster, countsOutCluster)))
    uplot.plotCorrelation(binCenter, velCorrInCluster[:,0], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'r')
    uplot.plotCorrelation(binCenter, velCorrInCluster[:,1], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'g')
    uplot.plotCorrelation(binCenter, velCorrInCluster[:,2], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'k')
    #plt.show()

################################## Velocity field ##############################
def computeVelocityField(dirName, numBins=100, plot=False, figureName=None, read=False):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    bins = np.linspace(-0.5*boxSize[0],0, numBins)
    bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
    bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
    if(read=='read' and os.path.exists(dirName + os.sep + "velocityField.dat")):
        grid = np.loadtxt(dirName + os.sep + "velocityGrid.dat")
        field = np.loadtxt(dirName + os.sep + "velocityField.dat")
    else:
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
        delta = utils.computeDeltas(pos, boxSize)
        bins = np.linspace(-0.5*boxSize[0],0, numBins)
        bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
        bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
        numBins = bins.shape[0]
        field = np.zeros((numBins, numBins, 2))
        grid = np.zeros((numBins, numBins, 2))
        for k in range(numBins-1):
            for l in range(numBins-1):
                grid[k,l,0] = bins[k]
                grid[k,l,1] = bins[l]
        for i in range(numParticles):
            rotation = np.array([[vel[i,0], -vel[i,1]], [vel[i,1], vel[i,0]]]) / np.linalg.norm(vel[i])
            for j in range(numParticles):
                for k in range(numBins-1):
                    if(delta[i,j,0] > bins[k] and delta[i,j,0] <= bins[k+1]):
                        for l in range(numBins-1):
                            if(delta[i,j,1] > bins[l] and delta[i,j,1] <= bins[l+1]):
                                field[k,l] += np.matmul(rotation, vel[j])
        field = field.reshape(numBins*numBins, 2)
        field /= np.max(np.linalg.norm(field,axis=1))
        grid = grid.reshape(numBins*numBins, 2)
        np.savetxt(dirName + os.sep + "velocityGrid.dat", grid)
        np.savetxt(dirName + os.sep + "velocityField.dat", field)
    if(plot=="plot"):
        xBounds = np.array([bins[0], bins[-1]])
        yBounds = np.array([bins[0], bins[-1]])
        fig = plt.figure(0, dpi = 150)
        ax = fig.gca()
        ax.set_xlim(xBounds[0], xBounds[1])
        ax.set_ylim(yBounds[0], yBounds[1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.quiver(grid[:,0], grid[:,1], field[:,0], field[:,1], facecolor='k', width=0.004, headlength=3, headaxislength=3)
        plt.savefig("/home/francesco/Pictures/soft/packings/vfield-" + figureName + ".png", transparent=False, format = "png")
        plt.show()
    return grid, field

######################### Cluster Velocity Correlation #########################
def computeVelocityFieldCluster(dirName, numBins=100, plot=False, figureName=None, read=False):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    bins = np.linspace(-0.5*boxSize[0],0, numBins)
    bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
    bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
    if(read=='read' and os.path.exists(dirName + os.sep + "dbClusterLabels.dat")):
        clusterLabels = np.loadtxt(dirName + os.sep + "dbClusterLabels.dat")[:,0]
    else:
        clusterLabels,_,_ = searchDBClusters(dirName, eps=0, min_samples=10)
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
        delta = utils.computeDeltas(pos, boxSize)
        bins = np.linspace(-0.5*boxSize[0],0, numBins)
        bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
        bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
        numBins = bins.shape[0]
        inField = np.zeros((numBins, numBins, 2))
        outField = np.zeros((numBins, numBins, 2))
        grid = np.zeros((numBins, numBins, 2))
        for k in range(numBins-1):
            for l in range(numBins-1):
                grid[k,l,0] = bins[k]
                grid[k,l,1] = bins[l]
        for i in range(numParticles):
            rotation = np.array([[vel[i,0], -vel[i,1]], [vel[i,1], vel[i,0]]]) / np.linalg.norm(vel[i])
            for j in range(numParticles):
                for k in range(numBins-1):
                    if(delta[i,j,0] > bins[k] and delta[i,j,0] <= bins[k+1]):
                        for l in range(numBins-1):
                            if(delta[i,j,1] > bins[l] and delta[i,j,1] <= bins[l+1]):
                                if(clusterLabels[i]!=-1):
                                    inField[k,l] += np.matmul(rotation, vel[j])
                                else:
                                    outField[k,l] += np.matmul(rotation, vel[j])
        inField = inField.reshape(numBins*numBins, 2)
        inField /= np.max(np.linalg.norm(inField,axis=1))
        outField = outField.reshape(numBins*numBins, 2)
        outField /= np.max(np.linalg.norm(outField,axis=1))
        grid = grid.reshape(numBins*numBins, 2)
        np.savetxt(dirName + os.sep + "velocityGridInCluster.dat", grid)
        np.savetxt(dirName + os.sep + "velocityFieldInCluster.dat", inField)
        np.savetxt(dirName + os.sep + "velocityFieldOutCluster.dat", outField)
    if(plot=="plot"):
        xBounds = np.array([bins[0], bins[-1]])
        yBounds = np.array([bins[0], bins[-1]])
        fig = plt.figure(0, dpi = 150)
        ax = fig.gca()
        ax.set_xlim(xBounds[0], xBounds[1])
        ax.set_ylim(yBounds[0], yBounds[1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.quiver(grid[:,0], grid[:,1], inField[:,0], inField[:,1], facecolor='k', width=0.004, headlength=3, headaxislength=3)
        plt.savefig("/home/francesco/Pictures/soft/packings/vfieldCluster-" + figureName + ".png", transparent=False, format = "png")
        plt.show()
    return grid, inField

######################### Cluster Velocity Correlation #########################
def averageVelocityFieldCluster(dirName, dirSpacing=1000, numBins=100, plot=False, figureName=None):
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    #dirList = dirList[-10:]
    bins = np.linspace(-0.5*boxSize[0],0, numBins)
    bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
    bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
    numBins = bins.shape[0]
    field = np.zeros((numBins*numBins, 2))
    grid = np.zeros((numBins*numBins, 2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        gridTemp, fieldTemp = computeVelocityFieldCluster(dirSample, numBins)
        grid += gridTemp
        field += fieldTemp
    grid /= dirList.shape[0]
    field /= dirList.shape[0]
    np.savetxt(dirName + os.sep + "averageVelocityGridInCluster.dat", grid)
    np.savetxt(dirName + os.sep + "averageVelocityFieldInCluster.dat", field)
    if(plot=="plot"):
        xBounds = np.array([bins[0], bins[-1]])
        yBounds = np.array([bins[0], bins[-1]])
        fig = plt.figure(0, dpi = 150)
        ax = fig.gca()
        ax.set_xlim(xBounds[0], xBounds[1])
        ax.set_ylim(yBounds[0], yBounds[1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.quiver(grid[:,0], grid[:,1], field[:,0], field[:,1], facecolor='k', width=0.004, headlength=3, headaxislength=3)
        plt.savefig("/home/francesco/Pictures/soft/packings/avfield-" + figureName + ".png", transparent=False, format = "png")
        plt.show()
    return grid, field

############################# Cluster fluctuations #############################
def averageClusterFluctuations(dirName, dirSpacing=10000):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    phi = int(utils.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    #dirList = dirList[-50:]
    numberCluster = np.zeros(dirList.shape[0])
    densityCluster = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "dbClusterLabels.dat")):
            clusterLabels = np.loadtxt(dirSample + os.sep + "dbClusterLabels.dat")[:,0]
        else:
            clusterLabels, _,_ = searchDBClusters(dirSample, eps=0, min_samples=8)
            #clusterLabels, _,_ = searchClusters(dirSample, numParticles)
        numberCluster[d] = clusterLabels[clusterLabels==1].shape[0]
        densityCluster[d] = np.sum(np.pi*particleRad[clusterLabels==1]**2)
    # in cluster
    np.savetxt(dirName + os.sep + "clusterFluctuations.dat", np.column_stack((timeList, numberCluster, densityCluster)))
    print("Number of particles in cluster: ", np.mean(numberCluster), " +- ", np.std(numberCluster))
    print("Cluster area: ", np.mean(densityCluster), " +- ", np.std(densityCluster))
    uplot.plotCorrelation(timeList, densityCluster, "$A_p$", xlabel = "$Time,$ $t$", color='k')
    #plt.show()

############################# Cluster distribution #############################
def averageClusterDistribution(dirName, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    phi = int(utils.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    clusterNumber = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "dbClusterLabels.dat")):
            labels = np.loadtxt(dirSample + os.sep + "dbClusterLabels.dat")[:,2]
        else:
            _,_, labels = searchDBClusters(dirSample, eps=0, min_samples=8)
            #clusterLabels, _,_ = searchClusters(dirSample, numParticles)
        numLabels = np.unique(labels).shape[0]-1
        for i in range(numLabels):
            clusterNumber = np.append(clusterNumber, labels[labels==i].shape[0])
    # in cluster
    clusterNumber = clusterNumber[clusterNumber>0]
    clusterNumber = np.sort(clusterNumber)
    print(clusterNumber)
    np.savetxt(dirName + os.sep + "clusterNumbers.dat", clusterNumber)
    print("Average number in cluster: ", np.mean(clusterNumber), " +- ", np.std(clusterNumber))
    pdf, edges = np.histogram(clusterNumber, bins=np.geomspace(np.min(clusterNumber), np.max(clusterNumber), numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    uplot.plotCorrelation(edges, pdf, "$PDF(N_c)$", xlabel = "$N_c$", color='k', logx=True, logy=True)
    np.savetxt(dirName + os.sep + "clusterNumberPDF.dat", np.column_stack((edges, pdf)))
    plt.plot()

def averageLocalDensityAndNumberFluctuations(dirName, plot=False, dirSpacing=1000000):
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    numBins = np.arange(2,101)
    meanNumList = np.zeros((dirList.shape[0], numBins.shape[0]))
    deltaNumList = np.zeros((dirList.shape[0], numBins.shape[0]))
    meanPhiList = np.zeros((dirList.shape[0], numBins.shape[0]))
    deltaPhiList = np.zeros((dirList.shape[0], numBins.shape[0]))
    colorList = cm.get_cmap('inferno', dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(os.path.exists(dirName + os.sep + "localNumberDensity.dat")):
            data = np.loadtxt(dirName + os.sep + "localNumberDensity.dat")
            meanNumList[d] = data[:,1]
            deltaNumList[d] = data[:,2]
            meanPhiList[d] = data[:,3]
            deltaPhiList[d] = data[:,4]
        else:
            meanNumList[d], deltaNumList[d], meanPhiList[d], deltaPhiList[d] = utils.computeLocalDensityAndNumberFluctuations(dirName + os.sep + dirList[d], plot=False, color=colorList(d/dirList.shape[0]))
    meanNum = np.mean(meanNumList, axis=0)
    stdMeanNum = np.std(meanNumList, axis=0)
    deltaNum = np.mean(deltaNumList, axis=0)
    stdDeltaNum = np.std(deltaNumList, axis=0)
    meanPhi = np.mean(meanPhiList, axis=0)
    stdMeanPhi = np.std(meanPhiList, axis=0)
    deltaPhi = np.mean(deltaPhiList, axis=0)
    stdDeltaPhi = np.std(deltaPhiList, axis=0)
    np.savetxt(dirName + os.sep + "averageLocalNumberDensity.dat", np.column_stack((numBins, meanNum, stdMeanNum, deltaNum, stdDeltaNum, meanPhi, stdMeanPhi, deltaPhi, stdDeltaPhi)))
    if(plot=="plot"):
        uplot.plotCorrWithError(meanNum, deltaPhi, stdDeltaPhi, "$Variance$ $of$ $local$ $number,$ $\\Delta N^2$", "$Local$ $number,$ $N_s$", color='k', logx=True)
        plt.pause(0.5)

############################ Velocity distribution #############################
def averageClusterVelPDF(dirName, threshold=0.76, plot=False, dirSpacing=1000000):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    #dirList = dirList[-50:]
    velInCluster = np.empty(0)
    velOutCluster = np.empty(0)
    velTotal = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + "/particleList.dat")):
            computeDelaunayCluster(dirSample, threshold=threshold)
        particleList = np.loadtxt(dirSample + "/particleList.dat")
        denseList = particleList[:,0]
        vel = np.loadtxt(dirSample + os.sep + "particleVel.dat")
        velNorm = np.linalg.norm(vel, axis=1)
        velInCluster = np.append(velInCluster, velNorm[denseList==1].flatten())
        velOutCluster = np.append(velOutCluster, velNorm[denseList!=1].flatten())
        velTotal = np.append(velTotal, velNorm.flatten())
    # in cluster
    velInCluster = velInCluster[velInCluster>0]
    mean = np.mean(velInCluster)
    tempIn = np.var(velInCluster)
    skewness = np.mean((velInCluster - mean)**3)/tempIn**(3/2)
    kurtosis = np.mean((velInCluster - mean)**4)/tempIn**2
    data = velInCluster# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity inside the cluster: ", tempIn, " kurtosis: ", kurtosis, " skewness: ", skewness)
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='b')
    np.savetxt(dirName + os.sep + "velPDFInCluster.dat", np.column_stack((edges, pdf)))
    # out of cluster
    velOutCluster = velOutCluster[velOutCluster>0]
    mean = np.mean(velOutCluster)
    tempOut = np.var(velOutCluster)
    skewness = np.mean((velOutCluster - mean)**3)/tempOut**(3/2)
    kurtosis = np.mean((velOutCluster - mean)**4)/tempOut**2
    data = velOutCluster# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity outside the cluster: ", tempOut, " kurtosis: ", kurtosis, " skewness: ", skewness)
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='g')
    np.savetxt(dirName + os.sep + "velPDFOutCluster.dat", np.column_stack((edges, pdf)))
    # total
    velTotal = velTotal[velTotal>0]
    mean = np.mean(velTotal)
    temp = np.var(velTotal)
    skewness = np.mean((velTotal - mean)**3)/temp**(3/2)
    kurtosis = np.mean((velTotal - mean)**4)/temp**2
    data = velTotal# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity in the whole system: ", temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='k')
    np.savetxt(dirName + os.sep + "velPDF.dat", np.column_stack((edges, pdf)))
    if(plot == "plot"):
        plt.pause(0.5)
        #plt.show()
    return temp, tempIn, tempOut

############################ Cluster residence pdf #############################
def computeClusterResidence(dirName, numBlocks, blockPower, spacing='log', plot=False, dirSpacing=1):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]*timeStep
    # measure residence length in blocks
    if(spacing=='log'):
        blockSize = 10 + (numBlocks-1)*(blockPower-1)
    elif(spacing=='linear'):
        blockFreq = 1e05
        blockSize = int((10**(blockPower+1) / blockFreq) / numBlocks)
    else:
        print("Specify saving spacing type: log or linear")
        return 0
    gasTime = []
    liquidTime = []
    gasLength = []
    liquidLength = []
    for n in range(numBlocks):
        print("block index", n)
        dirBlockList = dirList[n*blockSize:(n+1)*blockSize]
        # first get cluster at initial condition
        pos0 = np.loadtxt(dirName + dirBlockList[0] + os.sep + "particlePos.dat")
        boxMultiple0 = np.floor(pos0/boxSize)
        pos0 -= boxMultiple0 * boxSize
        if not(os.path.exists(dirName + os.sep + dirBlockList[0] + "/particleList.dat")):
            computeDelaunayCluster(dirName + os.sep + dirBlockList[0])
        particleList = np.loadtxt(dirName + os.sep + dirBlockList[0] + "/particleList.dat")
        denseList0 = particleList[:,0]
        #switched = np.zeros(numParticles)
        time0 = np.ones(numParticles)*timeList[n*blockSize]
        for d in range(1,dirBlockList.shape[0]):
            #print(dirBlockList[0], dirBlockList[d])
            pos = np.loadtxt(dirName + dirBlockList[d] + os.sep + "particlePos.dat")
            pos -= boxMultiple0 * boxSize
            if not(os.path.exists(dirName + os.sep + dirBlockList[d] + "/particleList.dat")):
                computeDelaunayCluster(dirName + os.sep + dirBlockList[d])
            particleList = np.loadtxt(dirName + os.sep + dirBlockList[d] + "/particleList.dat")
            denseList = particleList[:,0]
            for i in range(numParticles):
                #if(switched[i] == 0):
                if(denseList0[i]==0 and denseList[i]==1):
                    # this particle has switched - save residence time in gas
                    gasTime.append(timeList[n*blockSize + d] - time0[i])
                    #gasLength.append(np.linalg.norm(utils.pbcDistance(pos[i], pos0[i], boxSize)))
                    gasLength.append(np.linalg.norm(pos[i] - pos0[i]))
                    denseList0[i] = 1
                    time0[i] = timeList[n*blockSize + d]
                    #switched[i] = 1
                if(denseList0[i]==1 and denseList[i]==0):
                    # this particle has switched - save residence time in liquid
                    liquidTime.append(timeList[n*blockSize + d] - time0[i])
                    #liquidLength.append(np.linalg.norm(utils.pbcDistance(pos[i], pos0[i], boxSize)))
                    liquidLength.append(np.linalg.norm(pos[i] - pos0[i]))
                    denseList0[i] = 0
                    time0[i] = timeList[n*blockSize + d]
                    #switched[i] = 1
    gasTime = np.array(gasTime)
    liquidTime = np.array(liquidTime)
    gasLength = np.array(gasLength)
    liquidLength = np.array(liquidLength)
    print("time spent in gas:", np.mean(gasTime), "+-", np.std(gasTime))
    print("time spent in liquid:", np.mean(liquidTime), "+-", np.std(liquidTime))
    print("distance traveled in gas:", np.mean(gasLength), "+-", np.std(gasLength))
    print("distance traveled in liquid:", np.mean(liquidLength), "+-", np.std(liquidLength))
    np.savetxt(dirName + os.sep + "gasResidence.dat", np.column_stack((gasTime, gasLength)))
    np.savetxt(dirName + os.sep + "liquidResidence.dat", np.column_stack((liquidTime, liquidLength)))
    if(plot=='plot'):
        # length plots
        #uplot.plotCorrelation(np.arange(1, liquidLength.shape[0]+1, 1), np.sort(liquidLength), "$Liquid$ $residence$ $length$", xlabel = "$index$", color='k', logy=True)
        uplot.plotCorrelation(np.arange(1, gasLength.shape[0]+1, 1), np.sort(gasLength), "$Gas$ $residence$ $length$", xlabel = "$index$", color='k', logy=True)
        # time plots
        #uplot.plotCorrelation(np.arange(1, liquidTime.shape[0]+1, 1), np.sort(liquidTime), "$Liquid$ $residence$ $time$", xlabel = "$index$", color='k', logy=True)
        #uplot.plotCorrelation(np.arange(1, gasTime.shape[0]+1, 1), np.sort(gasTime), "$Gas$ $residence$ $time$", xlabel = "$index$", color='k', logy=True)
        plt.pause(0.5)
        #plt.show()

######################### Cluster residence correlation ########################
def computeClusterVelCorr(dirName, numBlocks, blockPower, spacing='log', plot=False, dirSpacing=1):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]*timeStep
    # measure residence correlation in blocks
    if(spacing=='log'):
        blockSize = 10 + (numBlocks-1)*(blockPower-1)
    elif(spacing=='linear'):
        blockFreq = 1e05
        blockSize = int((10**(blockPower+1) / blockFreq) / numBlocks)
    else:
        print("Specify saving spacing type: log or linear")
        return 0
    blockTimeList = timeList[:blockSize]
    gasSpeed = []
    liquidSpeed = []
    gasCorr = np.zeros((numBlocks, blockSize))
    liquidCorr = np.zeros((numBlocks, blockSize))
    for n in range(numBlocks):
        print("block index", n)
        blockGasCorr = np.zeros((blockSize, numParticles))
        blockLiquidCorr = np.zeros((blockSize, numParticles))
        dirBlockList = dirList[n*blockSize:(n+1)*blockSize]
        # first get cluster at initial condition
        if not(os.path.exists(dirName + os.sep + dirBlockList[0] + "/particleList.dat")):
            computeDelaunayCluster(dirName + os.sep + dirBlockList[0])
        particleList = np.loadtxt(dirName + os.sep + dirBlockList[0] + "/particleList.dat")
        denseList0 = particleList[:,0]
        vel0 = np.loadtxt(dirName + dirBlockList[0] + os.sep + "particleVel.dat")
        speed0 = np.linalg.norm(vel0, axis=1)
        for i in range(numParticles):
            if(denseList0[i] == 0):
                blockGasCorr[0,i] = speed0[i]**2
            else:
                blockLiquidCorr[0,i] = speed0[i]**2
        switched = np.zeros(numParticles)
        for d in range(1,dirBlockList.shape[0]):
            #print(dirBlockList[0], dirBlockList[d])
            if not(os.path.exists(dirName + os.sep + dirBlockList[d] + "/particleList.dat")):
                computeDelaunayCluster(dirName + os.sep + dirBlockList[d])
            particleList = np.loadtxt(dirName + os.sep + dirBlockList[d] + "/particleList.dat")
            denseList = particleList[:,0]
            vel = np.loadtxt(dirName + dirBlockList[d] + os.sep + "particleVel.dat")
            speed = np.linalg.norm(vel, axis=1)
            for i in range(numParticles):
                if(switched[i] == 0):
                    if(denseList0[i]==0):
                        if(denseList[i]==0):
                            gasSpeed.append(speed[i])
                            blockGasCorr[d,i] = np.sum(vel[i]*vel0[i])
                        else:
                            switched[i] = 1
                    if(denseList0[i]==1):
                        if(denseList[i]==1):
                            liquidSpeed.append(speed[i])
                            blockLiquidCorr[d,i] = np.sum(vel[i]*vel0[i])
                        else:
                            switched[i] = 1
        # average correlations over particles
        gasCorr[n] = np.mean(blockGasCorr[:,denseList0==0], axis=1)
        liquidCorr[n] = np.mean(blockLiquidCorr[:,denseList0==1], axis=1)
    # average over blocks
    gasCorr = np.column_stack((np.mean(gasCorr, axis=0), np.std(gasCorr, axis=0)))
    liquidCorr = np.column_stack((np.mean(liquidCorr, axis=0), np.std(liquidCorr, axis=0)))
    print("average gas speed:", np.mean(gasSpeed), "+-", np.std(gasSpeed))
    print("average liquid speed:", np.mean(liquidSpeed), "+-", np.std(liquidSpeed))
    gasDecay = utils.computeDecay(blockTimeList, gasCorr[:,0], threshold=np.exp(-1), normalize=True)
    liquidDecay = utils.computeDecay(blockTimeList, liquidCorr[:,0], threshold=np.exp(-1), normalize=True)
    print("gas velocity decay time:", gasDecay)
    print("liquid velocity decay time:", liquidDecay)
    gasSpeed = np.array(gasSpeed)
    liquidSpeed = np.array(liquidSpeed)
    print("typical gas persistence length:", np.mean(gasSpeed)*gasDecay, "+-", np.std(gasSpeed)*gasDecay)
    print("typical liquid persistence length:", np.mean(liquidSpeed)*liquidDecay, "+-", np.std(liquidSpeed)*liquidDecay)
    np.savetxt(dirName + os.sep + "clusterVelCorr.dat", np.column_stack((blockTimeList, gasCorr, liquidCorr)))
    np.savetxt(dirName + os.sep + "clusterDecayLength.dat", np.column_stack((np.mean(gasSpeed)*gasDecay, np.std(gasSpeed)*gasDecay, np.mean(liquidSpeed)*liquidDecay, np.std(liquidSpeed)*liquidDecay)))
    if(plot=='plot'):
        # speed plots
        #data = liquidSpeed
        #pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
        #edges = (edges[1:] + edges[:-1])/2
        #uplot.plotCorrelation(edges, pdf, "$Liquid$ $speed$ $PDF$", xlabel = "$speed$", color='k')
        # corr plots
        #uplot.plotCorrWithError(blockTimeList, liquidCorr[:,0], liquidCorr[:,1], ylabel="$C_{vv}^{liquid}(\\Delta t)$", xlabel="$Elapsed$ $time,$ $\\Delta t$", color = 'k', logx=True)
        uplot.plotCorrWithError(blockTimeList, gasCorr[:,0], gasCorr[:,1], ylabel="$C_{vv}^{gas}(\\Delta t)$", xlabel="$Elapsed$ $time,$ $\\Delta t$", color = 'k', logx=True)
        plt.pause(0.5)
        #plt.show()

#################### Time-averaged cluster mixing correlation ##################
def computeClusterMixing(dirName, numBlocks, blockPower, spacing='log', plot=False, dirSpacing=1):
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]*timeStep
    # measure mixing time in blocks
    if(spacing=='log'):
        blockSize = 10 + (numBlocks-1)*(blockPower-1)
    elif(spacing=='linear'):
        blockFreq = 1e05
        blockSize = int((10**(blockPower+1) / blockFreq) / numBlocks)
    else:
        print("Specify saving spacing type: log or linear")
        return 0
    blockTimeList = timeList[:blockSize]
    denseLabelCorr = np.zeros((numBlocks, blockSize, numParticles))
    for n in range(numBlocks):
        print("block index", n)
        blockDenseLabel = np.zeros((blockSize, numParticles))
        blockDenseLabelSq = np.zeros((blockSize, numParticles))
        blockDenseLabelCorr = np.zeros((blockSize, numParticles))
        dirBlockList = dirList[n*blockSize:(n+1)*blockSize]
        # first get cluster at initial condition
        if not(os.path.exists(dirName + os.sep + dirBlockList[0] + "/particleList.dat")):
            computeDelaunayCluster(dirName + os.sep + dirBlockList[0])
        particleList = np.loadtxt(dirName + os.sep + dirBlockList[0] + "/particleList.dat")
        denseList0 = particleList[:,0]
        denseList0 *= 2
        denseList0 -= 1
        blockDenseLabel[0] = denseList0
        blockDenseLabelSq[0] = denseList0**2
        blockDenseLabelCorr[0] = denseList0**2
        for d in range(1,dirBlockList.shape[0]):
            #print(dirBlockList[0], dirBlockList[d])
            if not(os.path.exists(dirName + os.sep + dirBlockList[d] + "/particleList.dat")):
                computeDelaunayCluster(dirName + os.sep + dirBlockList[d])
            particleList = np.loadtxt(dirName + os.sep + dirBlockList[d] + "/particleList.dat")
            denseList0 = particleList[:,0]
            denseList *= 2
            denseList -= 1
            blockDenseLabel[d] = denseList
            blockDenseLabelSq[d] = denseList**2
            blockDenseLabelCorr[d] = denseList * denseList0
        denseLabelCorr[n] = (blockDenseLabelCorr - np.mean(blockDenseLabel)**2) / (np.mean(blockDenseLabelSq) - np.mean(blockDenseLabel)**2)
    denseLabelCorr = np.mean(denseLabelCorr, axis=2) # average over particles
    print(denseLabelCorr.shape)
    denseLabelCorr = np.column_stack((np.mean(denseLabelCorr, axis=0), np.std(denseLabelCorr, axis=0))) # average over blocks
    np.savetxt(dirName + os.sep + "clusterMixing.dat", np.column_stack((blockTimeList, denseLabelCorr)))
    if(plot=='plot'):
        uplot.plotCorrWithError(blockTimeList, denseLabelCorr[:,0], denseLabelCorr[:,1], ylabel="$C_{mix}(\\Delta t)$", logx = True, color = 'k')
        plt.pause(0.5)
        #plt.show()

############### Cluster evaporation time averaged in time blocks ###############
def computeClusterRateBySize(dirName, numBlocks, blockPower, blockFreq=1e05, spacing='log', threshold=0.76, plot=False, dirSpacing=1):
    numBins = 20
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + "/particleRad.dat"), dtype=np.float64)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # measure cluster rate in blocks
    if(spacing=='log'):
        blockSize = 10 + (numBlocks-1)*(blockPower-1)
    elif(spacing=='linear'):
        blockSize = int((10**(blockPower+1) / blockFreq) / numBlocks)
    else:
        print("Specify saving spacing type: log or linear")
    blockTimeList = timeList[:blockSize]*timeStep
    # first get range of cluster size distribution
    if not(os.path.exists(dirName + os.sep + "/clusterDistribution.dat")):
        computeDelaunayClusterDistribution(dirName)
    clusterArea = np.loadtxt(dirName + os.sep + "/clusterDistribution.dat")[:,1]
    pdf, edges = np.histogram(clusterArea, bins=np.geomspace(np.min(clusterArea), np.max(clusterArea), numBins+1), density=True)
    condensationTime = np.zeros((edges.shape[0]-1,3))
    evaporationTime = np.zeros((edges.shape[0]-1,3))
    # measure rate in blocks
    for n in range(numBlocks):
        #print("block index", n)
        dirBlockList = dirList[n*blockSize:(n+1)*blockSize]
        # re-label particles by size of cluster they belong to
        labels0 = np.loadtxt(dirName + os.sep + dirBlockList[0] + "/clusterLabels.dat")
        sizeLabels0 = -1*np.ones(labels0.shape[0], dtype=np.int64)
        for label in np.unique(labels0):
            if(label!=-1):
                clusterArea = np.sum(rad[labels0==label]**2)*np.pi
                #clusterArea = labels0[labels0==label].shape[0] / labels0.shape[0]
                for k in range(edges.shape[0]-1):
                    if(clusterArea > edges[k] and clusterArea < edges[k+1]):
                        sizeLabels0[labels0==label] = k
        #print(np.unique(sizeLabels))
        time0 = np.ones(numParticles)*blockTimeList[0]
        for d in range(1,dirBlockList.shape[0]):
            if not(os.path.exists(dirName + os.sep + dirBlockList[d] + "/particleList.dat")):
                computeDelaunayCluster(dirName + os.sep + dirBlockList[d], threshold=threshold)
            particleList = np.loadtxt(dirName + os.sep + dirBlockList[d] + "/particleList.dat")
            denseList = particleList[:,0]
            borderList = particleList[:,1]
            #print(dirBlockList[0], dirBlockList[d])
            labels = np.loadtxt(dirName + os.sep + dirBlockList[d] + "/clusterLabels.dat")
            sizeLabels = -1*np.ones(labels.shape[0], dtype=np.int64)
            for label in np.unique(labels):
                if(label!=-1):
                    clusterArea = np.sum(rad[labels==label]**2)*np.pi
                    #clusterArea = labels[labels==label].shape[0] / labels.shape[0]
                    for k in range(edges.shape[0]-1):
                        if(clusterArea > edges[k] and clusterArea < edges[k+1]):
                            sizeLabels[labels==label] = k
            #print(np.unique(sizeLabels))
            for i in range(numParticles):
                if(borderList[i]==1 or denseList[i]==0):
                    if(sizeLabels[i]!=sizeLabels0[i]):
                        if(sizeLabels0[i]==-1):
                            if(time0[i] != blockTimeList[0]):
                                condensationTime[sizeLabels[i],0] += blockTimeList[d] - time0[i]
                                condensationTime[sizeLabels[i],1] += (blockTimeList[d] - time0[i])**2
                                condensationTime[sizeLabels[i],2] += 1
                            sizeLabels0[i] = sizeLabels[i]
                            time0[i] = blockTimeList[d]
                        else:
                            if(time0[i] != blockTimeList[0]):
                                evaporationTime[sizeLabels0[i],0] += blockTimeList[d] - time0[i]
                                evaporationTime[sizeLabels0[i],1] += (blockTimeList[d] - time0[i])**2
                                evaporationTime[sizeLabels0[i],2] += 1
                            sizeLabels0[i] = sizeLabels[i]
                            time0[i] = blockTimeList[d]
    for i in range(condensationTime.shape[0]):
        if(condensationTime[i,2]>0):
            condensationTime[i,0] /= condensationTime[i,2]
            condensationTime[i,1] /= condensationTime[i,2]
        if(evaporationTime[i,2]>0):
            evaporationTime[i,0] /= evaporationTime[i,2]
            evaporationTime[i,1] /= evaporationTime[i,2]
    condensationTime = np.column_stack((condensationTime[:,0], np.sqrt(condensationTime[:,1] - condensationTime[:,0]**2)))
    evaporationTime = np.column_stack((evaporationTime[:,0], np.sqrt(evaporationTime[:,1] - evaporationTime[:,0]**2)))
    clusterSize = (edges[1:] + edges[:-1])/2
    # average correlations over particles
    np.savetxt(dirName + os.sep + "rateBySize.dat", np.column_stack((clusterSize, condensationTime, evaporationTime)))
    print("Orange, condensation time:", np.mean(condensationTime[condensationTime[:,1]>0,0]), "+-", np.std(condensationTime[condensationTime[:,1]>0,0]))
    print("Blue, evaporation time:", np.mean(evaporationTime[evaporationTime[:,1]>0,0]), "+-", np.std(evaporationTime[evaporationTime[:,1]>0,0]))
    if(plot=='plot'):
        uplot.plotCorrWithError(clusterSize, condensationTime[:,0], condensationTime[:,1], ylabel="$Timescale$", xlabel="$Cluster$ $size$", logx = True, color = [1,0.5,0], marker='o')
        uplot.plotCorrWithError(clusterSize, evaporationTime[:,0], evaporationTime[:,1], ylabel="$Timescale$", xlabel="$Cluster$ $size$", logx = True, color = 'b', marker='s')
        plt.pause(0.5)
        #plt.show()

############### Cluster evaporation time averaged in time blocks ###############
def computeClusterMSD(dirName, numBlocks, blockPower, blockFreq=1e04, spacing='log', threshold=0.76, plot=False, dirSpacing=1):
    numBins = 20
    boxSize = np.loadtxt(dirName + "/boxSize.dat")
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + "/particleRad.dat"), dtype=np.float64)
    sizeTh = 10 * np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]*timeStep
    # measure cluster rate in blocks
    if(spacing=='log'):
        blockSize = 10 + (numBlocks-1)*(blockPower-1)
    elif(spacing=='linear'):
        blockSize = int((10**(blockPower+1) / blockFreq) / numBlocks)
    else:
        print("Specify saving spacing type: log or linear")
    blockTimeList = timeList[:blockSize]
    corrDroplet = np.zeros((numBlocks, blockSize))
    corrGas = np.zeros((numBlocks, blockSize))
    for n in range(numBlocks):
        #print("block index", n)
        dirBlockList = dirList[n*blockSize:(n+1)*blockSize]
        # re-label particles by size of cluster they belong to
        labels0 = np.loadtxt(dirName + os.sep + dirBlockList[0] + "/clusterLabels.dat")
        uniqueLabels0 = np.unique(labels0).astype(np.int64)
        pos0 = utils.getPBCPositions(dirName + os.sep + dirBlockList[0] + os.sep + "particlePos.dat", boxSize)
        dropletPos0, dropletRad0 = utils.getDropletPosRad(pos0, rad, boxSize, labels0)
        blockCorrDroplet = np.zeros((blockSize, dropletPos0.shape[0]))
        blockCorrGas = np.zeros((blockSize, labels0[labels0==-1].shape[0]))
        if not(os.path.exists(dirName + os.sep + dirBlockList[0] + "/particleList.dat")):
            computeDelaunayCluster(dirName + os.sep + dirBlockList[0])
        particleList = np.loadtxt(dirName + os.sep + dirBlockList[0] + "/particleList.dat")
        denseList0 = particleList[:,0]
        switched = np.zeros(numParticles)
        for d in range(1,dirBlockList.shape[0]):
            pos = utils.getPBCPositions(dirName + os.sep + dirBlockList[d] + os.sep + "particlePos.dat", boxSize)
            if not(os.path.exists(dirName + os.sep + dirBlockList[d] + "/particleList.dat")):
                computeDelaunayCluster(dirName + os.sep + dirBlockList[d])
            particleList = np.loadtxt(dirName + os.sep + dirBlockList[d] + "/particleList.dat")
            denseList = particleList[:,0]
            labelIndex = 0
            for label in uniqueLabels0:
                if(label!=-1):
                    dropletList = np.argwhere(labels0==label)[:,0].astype(np.int64)
                    msdList = np.empty(0, dtype=np.int64)
                    for index in dropletList:
                        if(switched[index]==0):
                            if(denseList0[index]==1 and denseList[index]==1):
                                msdList = np.append(msdList, index)
                            elif(denseList0[index]==1 and denseList[index]==0):
                                switched[index] = 1
                    if(msdList.shape[0]>0):
                        dropletPos = utils.getSingleDropletPosFromReference(pos, dropletRad0[labelIndex], boxSize, msdList)
                        blockCorrDroplet[d,labelIndex] += np.linalg.norm(dropletPos - dropletPos0[labelIndex])**2
                    labelIndex += 1
            gasIndex = 0
            for index in range(numParticles):
                if(labels0[index]==-1):
                    if(switched[index]==0):
                        if(denseList0[index]==0 and denseList[index]==0):
                            blockCorrGas[d,gasIndex] = np.linalg.norm(pos[index] - pos0[index])**2
                        elif(denseList0[index]==0 and denseList[index]==1):
                            switched[index] = 1
                    gasIndex += 1
        corrDroplet[n] = np.mean(blockCorrDroplet, axis=1)
        corrGas[n] = np.mean(blockCorrGas, axis=1)
    corrDroplet = np.column_stack((np.mean(corrDroplet, axis=0), np.std(corrDroplet, axis=0)))
    corrGas = np.column_stack((np.mean(corrGas, axis=0), np.std(corrGas, axis=0)))
    np.savetxt(dirName + os.sep + "clusterMSD.dat", np.column_stack((blockTimeList, corrDroplet, corrGas)))
    if(plot=='plot'):
        # MSD plots
        uplot.plotCorrelation(blockTimeList, corrDroplet[:,0], ylabel="$Cluster$ $MSD$", xlabel="$Time$", logx = True, logy = True, color = [1,0.5,0], marker='o')
        uplot.plotCorrelation(blockTimeList, corrGas[:,0], ylabel="$Cluster$ $MSD$", xlabel="$Time$", logx = True, logy = True, color = 'b', marker='s')
        #plt.pause(0.5)
        plt.show()

def computeClusterISF(dirName, startBlock, maxPower, freqPower, threshold=0.3, plot=False):
    boxSize = np.loadtxt(dirName + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + "particleRad.dat").astype(np.float64)
    sigma = 2 * np.mean(rad)
    bins = np.linspace(0.1*sigma, 10*sigma, 50)
    timeStep = utils.readFromParams(dirName, "dt")
    #labels,_ = getParticleDenseLabel(dirName, threshold)
    #labels = np.ones(numParticles)
    #if not(os.path.exists(dirName + os.sep + "pairCorr.dat")):
    #    computeClusterPairCorr(dirName, boxSize, bins, labels, 1, plot=False)
    #pcorr = np.loadtxt(dirName + os.sep + "pairCorr.dat")
    #firstPeak = pcorr[np.argmax(pcorr[:,1]),0]
    longWave = 2 * np.pi / sigma
    shortWave = 2 * np.pi / boxSize[1]
    particleCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            #print(stepRange, spacing*spacingDecade)
            stepParticleCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pos1, pos2 = utils.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        # select cluster positions
                        #pos1 = pos1[labels==1]
                        #pos2 = pos2[labels==1]
                        stepParticleCorr.append(utils.computeLongShortWaveCorr(pos1, pos2, boxSize, longWave, shortWave, sigma))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList) * timeStep
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],5))
    particleCorr = particleCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "clusterLogCorr.dat", np.column_stack((stepList, particleCorr)))
    data = np.column_stack((stepList, particleCorr))
    tau = utils.computeTau(data)
    print("Relaxation time:", tau)
    if(plot=="plot"):
        #uplot.plotCorrelation(stepList, particleCorr[:,0]/(stepList*timeStep), "$MSD(\\Delta t)/\\Delta t$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color = 'k')
        uplot.plotCorrelation(stepList, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k')
        uplot.plotCorrelation(stepList, particleCorr[:,2], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
        #plt.pause(0.5)
        plt.show()

################## Cluster mixing time for fixed cluster size ##################
def computeClusterMixingByLabel(dirName, numBlocks, blockPower, spacing='log', plot=False, dirSpacing=1):
    numBins = 20
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + "/particleRad.dat"), dtype=np.float64)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # measure cluster rate in blocks
    if(spacing=='log'):
        blockSize = 10 + (numBlocks-1)*(blockPower-1)
    elif(spacing=='linear'):
        blockFreq = 1e05
        blockSize = int((10**(blockPower+1) / blockFreq) / numBlocks)
    else:
        print("Specify saving spacing type: log or linear")
    blockTimeList = timeList[:blockSize]*timeStep
    # first get range of cluster size distribution
    if not(os.path.exists(dirName + os.sep + "/clusterDistribution.dat")):
        computeDelaunayClusterDistribution(dirName)
    clusterArea = np.loadtxt(dirName + os.sep + "/clusterDistribution.dat")[:,1]
    pdf, edges = np.histogram(clusterArea, bins=np.geomspace(np.min(clusterArea), np.max(clusterArea), numBins+1), density=True)
    # select range of cluster size
    areaTh = edges[-2]
    print("correlation is computed for clusters of size: ", areaTh)
    denseLabelCorr = np.zeros((numBlocks, blockSize))
    # measure mixing in blocks
    for n in range(numBlocks):
        print("block index", n)
        dirBlockList = dirList[n*blockSize:(n+1)*blockSize]
        # re-label particles by size of cluster they belong to
        labels0 = np.loadtxt(dirName + os.sep + dirBlockList[0] + "/clusterLabels.dat").astype(np.int64)
        labelList = np.empty(0, dtype=np.int64)
        for label in np.unique(labels0):
            if(label!=-1):
                clusterArea = np.sum(rad[labels0==label]**2)*np.pi
                if(clusterArea >= areaTh):
                    print("label:", label, "cluster size: ", clusterArea)
                    labelList = np.append(labelList, np.argwhere(labels0==label)[:,0])
        labelList = labelList.flatten()
        print(labelList)
        blockDenseLabel = np.zeros((blockSize, labelList.shape[0]))
        blockDenseLabelSq = np.zeros((blockSize, labelList.shape[0]))
        blockDenseLabelCorr = np.zeros((blockSize, labelList.shape[0]))
        # load denseList0 to compute correlation
        if not(os.path.exists(dirName + os.sep + dirBlockList[0] + "/particleList.dat")):
            computeDelaunayCluster(dirName + os.sep + dirBlockList[0])
        particleList = np.loadtxt(dirName + os.sep + dirBlockList[0] + "/particleList.dat")
        denseList0 = particleList[:,0]
        denseList0 *= 2
        denseList0 -= 1
        blockDenseLabel[0] = denseList0[labelList]
        blockDenseLabelSq[0] = denseList0[labelList]**2
        blockDenseLabelCorr[0] = denseList0[labelList]**2
        for d in range(1,dirBlockList.shape[0]):
            if not(os.path.exists(dirName + os.sep + dirBlockList[d] + "/particleList.dat")):
                computeDelaunayCluster(dirName + os.sep + dirBlockList[d])
            particleList = np.loadtxt(dirName + os.sep + dirBlockList[d] + "/particleList.dat")
            denseList = particleList[:,0]
            denseList *= 2
            denseList -= 1
            blockDenseLabel[d] = denseList[labelList]
            blockDenseLabelSq[d] = denseList[labelList]**2
            blockDenseLabelCorr[d] = denseList[labelList] * denseList0[labelList]
        blockDenseLabelCorr = (blockDenseLabelCorr - np.mean(blockDenseLabel)**2) / (np.mean(blockDenseLabelSq) - np.mean(blockDenseLabel)**2)
        denseLabelCorr[n] = np.mean(blockDenseLabelCorr, axis=1)
    print(denseLabelCorr.shape)
    denseLabelCorr = np.column_stack((np.mean(denseLabelCorr, axis=0), np.std(denseLabelCorr, axis=0))) # average over blocks
    np.savetxt(dirName + os.sep + "mixingBySize.dat", np.column_stack((blockTimeList, denseLabelCorr)))
    if(plot=='plot'):
        uplot.plotCorrWithError(blockTimeList, denseLabelCorr[:,0], denseLabelCorr[:,1], ylabel="$C_{mix}(\\Delta t)$", logx = True, color = 'k')
        #plt.pause(0.5)
        plt.show()


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    if(whichCorr == "averageld"):
        numBins = int(sys.argv[3])
        weight = sys.argv[4]
        plot = sys.argv[5]
        averageLocalDensity(dirName, numBins, weight, plot)

    elif(whichCorr == "nphitime"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        computeLocalDensityAndNumberVSTime(dirName, numBins, plot)

############################ clustering algorithms #############################
    elif(whichCorr == "cluster"):
        numParticles = int(sys.argv[3])
        plot = sys.argv[4]
        cluster = sys.argv[5]
        searchClusters(dirName, numParticles, plot=plot, cluster=cluster)

    elif(whichCorr == "dbcluster"):
        eps = float(sys.argv[3])
        min_samples = int(sys.argv[4])
        plot = sys.argv[5]
        contactFilter = sys.argv[6]
        searchDBClusters(dirName, eps=eps, min_samples=min_samples, plot=plot, contactFilter=contactFilter)

    elif(whichCorr == "dbsize"):
        dirSpacing = int(sys.argv[3])
        eps = float(sys.argv[4])
        min_samples = int(sys.argv[5])
        plot = sys.argv[6]
        contactFilter = sys.argv[7]
        averageDBClusterSize(dirName, dirSpacing, eps=eps, min_samples=min_samples, plot=plot, contactFilter=contactFilter)

    elif(whichCorr == "border"):
        plot = sys.argv[3]
        computeDBClusterBorder(dirName, plot)

    elif(whichCorr == "vorocluster"):
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

    elif(whichCorr == "checkdelaunay"):
        checkDelaunay(dirName)

    elif(whichCorr == "delcluster"):
        threshold = float(sys.argv[3])
        filter = sys.argv[4]
        plot = sys.argv[5]
        computeDelaunayCluster(dirName, threshold, filter, plot)

########################### Cluster structure #############################
    elif(whichCorr == "delld"):
        np.seterr(divide='ignore', invalid='ignore')
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        averageLocalDelaunayDensity(dirName, numBins, plot)

    elif(whichCorr == "deldensity"):
        plot = sys.argv[3]
        computeClusterDelaunayDensity(dirName, plot)

    elif(whichCorr == "delarea"):
        plot = sys.argv[3]
        computeClusterDelaunayArea(dirName, plot)

    elif(whichCorr == "delshape"):
        plot = sys.argv[3]
        computeClusterDelaunayShape(dirName, plot)

    elif(whichCorr == "delpers"):
        plot = sys.argv[3]
        computeDelaunayClusterVel(dirName, plot)

    elif(whichCorr == "deldistro"):
        plot = sys.argv[3]
        computeDelaunayClusterDistribution(dirName, plot)

    elif(whichCorr == "simplexdistro"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        computeSimplexClusterDistribution(dirName, threshold, plot)

    elif(whichCorr == "sizetime"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        computeClusterSizeVSTime(dirName, threshold, plot)

########################### Cluster dynamics #############################
    elif(whichCorr == "pccluster"):
        threshold = float(sys.argv[3])
        lj = sys.argv[4]
        dirSpacing = int(sys.argv[5])
        averageClusterPairCorr(dirName, threshold, lj, dirSpacing)

    elif(whichCorr == "pcdense"):
        threshold = float(sys.argv[3])
        dirSpacing = int(sys.argv[4])
        averageDensePairCorr(dirName, threshold, dirSpacing)

    elif(whichCorr == "clustercol"):
        check = sys.argv[3]
        numBins = int(sys.argv[4])
        getClusterContactCollisionIntervalPDF(dirName, check, numBins)

    elif(whichCorr == "vccluster"):
        computeParticleVelSpaceCorrCluster(dirName)

    elif(whichCorr == "averagevccluster"):
        averageParticleVelSpaceCorrCluster(dirName)

    elif(whichCorr == "vfield"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        figureName = sys.argv[5]
        computeVelocityField(dirName, numBins=numBins, plot=plot, figureName=figureName)

    elif(whichCorr == "vfcluster"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        figureName = sys.argv[5]
        computeVelocityFieldCluster(dirName, numBins=numBins, plot=plot, figureName=figureName)

    elif(whichCorr == "avfcluster"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        figureName = sys.argv[5]
        averageVelocityFieldCluster(dirName, numBins=numBins, plot=plot, figureName=figureName)

    elif(whichCorr == "clusterflu"):
        averageClusterFluctuations(dirName)

    elif(whichCorr == "clusterpdf"):
        numBins = int(sys.argv[3])
        averageClusterDistribution(dirName, numBins)

    elif(whichCorr == "averagephinum"):
        plot = sys.argv[3]
        averageLocalDensityAndNumberFluctuations(dirName, plot)

    elif(whichCorr == "velpdf"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageClusterVelPDF(dirName, threshold, plot)
        
    elif(whichCorr == "residence"):
        numBlocks = int(sys.argv[3])
        blockPower = int(sys.argv[4])
        spacing = sys.argv[5]
        plot = sys.argv[6]
        computeClusterResidence(dirName, numBlocks, blockPower, spacing, plot)

    elif(whichCorr == "velcorr"):
        numBlocks = int(sys.argv[3])
        blockPower = int(sys.argv[4])
        spacing = sys.argv[5]
        plot = sys.argv[6]
        computeClusterVelCorr(dirName, numBlocks, blockPower, spacing, plot)

    elif(whichCorr == "mixing"):
        numBlocks = int(sys.argv[3])
        blockPower = int(sys.argv[4])
        spacing = sys.argv[5]
        plot = sys.argv[6]
        computeClusterMixing(dirName, numBlocks, blockPower, spacing, plot)

    elif(whichCorr == "ratebysize"):
        numBlocks = int(sys.argv[3])
        blockPower = int(sys.argv[4])
        blockFreq = float(sys.argv[5])
        spacing = sys.argv[6]
        threshold = float(sys.argv[7])
        plot = sys.argv[8]
        computeClusterRateBySize(dirName, numBlocks, blockPower, blockFreq, spacing, threshold, plot)

    elif(whichCorr == "clustermsd"):
        numBlocks = int(sys.argv[3])
        blockPower = int(sys.argv[4])
        blockFreq = float(sys.argv[5])
        spacing = sys.argv[6]
        threshold = float(sys.argv[7])
        plot = sys.argv[8]
        computeClusterMSD(dirName, numBlocks, blockPower, blockFreq, spacing, threshold, plot)

    elif(whichCorr == "clusterisf"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = float(sys.argv[5])
        threshold = float(sys.argv[6])
        plot = sys.argv[7]
        computeClusterISF(dirName, startBlock, maxPower, freqPower, threshold, plot)

    elif(whichCorr == "mixingbylabel"):
        numBlocks = int(sys.argv[3])
        blockPower = int(sys.argv[4])
        spacing = sys.argv[5]
        plot = sys.argv[6]
        computeClusterMixingByLabel(dirName, numBlocks, blockPower, spacing, plot)

    else:
        print("Please specify the correlation you want to compute")
