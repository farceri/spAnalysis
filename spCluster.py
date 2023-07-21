'''
Created by Francesco
14 July 2023
'''
#functions and script to compute cluster correlations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
import pyvoro
import sys
import os
import utils
import utilsPlot as uplot

################################################################################
############################## Cluster properties ##############################
################################################################################
def computeLocalDensity(dirName, numBins, plot = False):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    localArea = np.zeros((numBins, numBins))
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    utils.computeLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea)
    localDensity = localArea/localSquare
    localDensity = np.sort(localDensity.flatten())
    cdf = np.arange(len(localDensity))/len(localDensity)
    pdf, edges = np.histogram(localDensity, bins=np.linspace(np.min(localDensity), np.max(localDensity), 30), density=True)
    edges = (edges[1:] + edges[:-1])/2
    if(plot=="plot"):
        #print("subBoxSize: ", xbin[1]-xbin[0], ybin[1]-ybin[0], " lengthscale: ", np.sqrt(np.mean(area)))
        print("data stats: ", np.min(localDensity), np.max(localDensity), np.mean(localDensity), np.std(localDensity))
        fig = plt.figure(dpi=120)
        ax = plt.gca()
        ax.plot(edges[1:], pdf[1:], linewidth=1.2, color='k')
        #ax.plot(localDensity, cdf, linewidth=1.2, color='k')
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylabel('$P(\\varphi)$', fontsize=18)
        ax.set_xlabel('$\\varphi$', fontsize=18)
        #ax.set_xlim(-0.02, 1.02)
        #plt.tight_layout()
        #plt.show()
    else:
        return localDensity

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
        utils.computeLocalAreaAndNumberGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea, localNumber)
        localDensity = localArea/localSquare
        localDensityVar = np.append(localDensityVar, np.var(localDensity))
        localNumberVar = np.append(localNumberVar, np.var(localNumber))
    if(plot=="plot"):
        np.savetxt(dirName + "localDensityAndNumberVarVSTime-N" + str(numBins) + ".dat", np.column_stack((timeList, localDensityVar, localNumberVar)))
        uplot.plotCorrelation(timeList, localDensityVar, "$Variance$ $of$ $local$ $density$", "$Time,$ $t$", color='k')
        uplot.plotCorrelation(timeList, localNumberVar, "$Variance$ $of$ $local$ $number$", "$Time,$ $t$", color='g')
        plt.show()

def averagePairCorrCluster(dirName, dirSpacing=1000000):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    phi = utils.readFromParams(dirName, "phi")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    particleRad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    minRad = np.mean(particleRad)
    rbins = np.arange(0, np.sqrt(2)*boxSize[0]/2, 0.02*minRad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    pcorrInCluster = np.zeros(rbins.shape[0]-1)
    pcorrOutCluster = np.zeros(rbins.shape[0]-1)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "denseList.dat")):
            clusterLabels = np.loadtxt(dirSample + os.sep + "denseList.dat")
            localDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            clusterLabels, density = computeVoronoiCluster(dirSample)
        #if(os.path.exists(dirSample + os.sep + "clusterLabels.dat")):
        #    clusterLabels = np.loadtxt(dirSample + os.sep + "clusterLabels.dat")[:,0]
        #    noClusterLabels = np.loadtxt(dirSample + os.sep + "clusterLabels.dat")[:,1]
        #else:
            #clusterLabels, noClusterLabels,_ = searchDBClusters(dirSample, eps=0, min_samples=10)
        #    clusterLabels, noClusterLabels,_ = searchClusters(dirSample, numParticles)
        phiInCluster = np.sum(localDensity[clusterLabels==1])
        phiOutCluster = np.sum(localDensity[clusterLabels==0])
        NpInCluster = clusterLabels[clusterLabels==1].shape[0]
        NpOutCluster = clusterLabels[clusterLabels==0].shape[0]
        #pos = np.array(np.loadtxt(dirSample + os.sep + "particlePos.dat"))
        pos = utils.getPBCPositions(dirSample + os.sep + "particlePos.dat", boxSize)
        pcorrInCluster += utils.getPairCorr(pos[clusterLabels==1], boxSize, rbins, minRad)/(NpInCluster * phiInCluster)
        pcorrOutCluster += utils.getPairCorr(pos[clusterLabels==0], boxSize, rbins, minRad)/(NpOutCluster * phiOutCluster)
    pcorrInCluster[pcorrInCluster>0] /= dirList.shape[0]
    pcorrOutCluster[pcorrOutCluster>0] /= dirList.shape[0]
    binCenter = (rbins[:-1] + rbins[1:])*0.5
    np.savetxt(dirName + os.sep + "pairCorrCluster.dat", np.column_stack((binCenter, pcorrInCluster, pcorrOutCluster)))
    firstPeakInCluster = binCenter[np.argmax(pcorrInCluster)]
    firstPeakOutCluster = binCenter[np.argmax(pcorrOutCluster)]
    print("First peak of pair corr in cluster is at:", firstPeakInCluster, "equal to", firstPeakInCluster/minRad, "times the min radius:", minRad)
    print("First peak of pair corr out cluster is at:", firstPeakOutCluster, "equal to", firstPeakOutCluster/minRad, "times the min radius:", minRad)
    uplot.plotCorrelation(binCenter[:200]/minRad, pcorrInCluster[:200], "$g(r/\\sigma)$", "$r/\\sigma$", color='k')
    uplot.plotCorrelation(binCenter[:200]/minRad, pcorrOutCluster[:200], "$g(r/\\sigma)$", "$r/\\sigma$", color='r')
    plt.pause(0.5)
    #plt.show()
    return firstPeakInCluster, firstPeakOutCluster

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

def averageParticleVelSpaceCorrCluster(dirName, dirSpacing=1000000):
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
        ax.quiver(grid[:,0], grid[:,1], field[:,0], field[:,1], facecolor='k', width=0.004, headlength=3, headaxislength=3)
        plt.savefig("/home/francesco/Pictures/soft/packings/vfieldCluster-" + figureName + ".png", transparent=False, format = "png")
        plt.show()
    return grid, field

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
def averageClusterDistribution(dirName, numBins=40, dirSpacing=1):
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

############################## Number fluctuations #############################
def computeLocalDensityAndNumberFluctuations(dirName, plot=False, color='k'):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
    numBins = np.arange(2,101)
    meanNum = np.zeros(numBins.shape[0])
    deltaNum = np.zeros(numBins.shape[0])
    meanPhi = np.zeros(numBins.shape[0])
    deltaPhi = np.zeros(numBins.shape[0])
    for i in range(numBins.shape[0]):
        xbin = np.linspace(0, boxSize[0], numBins[i]+1)
        ybin = np.linspace(0, boxSize[1], numBins[i]+1)
        localSquare = (boxSize[0]/numBins[i])*(boxSize[1]/numBins[i])
        localArea = np.zeros((numBins[i], numBins[i]))
        localNumber = np.zeros((numBins[i], numBins[i]))
        utils.computeLocalAreaAndNumberGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea, localNumber)
        localDensity = (localArea/localSquare).reshape(numBins[i]*numBins[i])
        localNumber = localNumber.reshape(numBins[i]*numBins[i])
        meanNum[i] = np.mean(localNumber)
        deltaNum[i] = np.var(localNumber)
        meanPhi[i] = np.mean(localDensity)
        deltaPhi[i] = np.var(localDensity)
    np.savetxt(dirName + os.sep + "localNumberDensity.dat", np.column_stack((numBins, meanNum, deltaNum, meanPhi, deltaPhi)))
    if(plot=="plot"):
        uplot.plotCorrelation(meanNum, deltaPhi, "$Variance$ $of$ $local$ $number,$ $\\Delta N^2$", "$Local$ $number,$ $N_s$", color=color, logx=True, logy=True)
        plt.pause(0.5)
    return meanNum, deltaNum, meanPhi, deltaPhi

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
            meanNumList[d], deltaNumList[d], meanPhiList[d], deltaPhiList[d] = computeLocalDensityAndNumberFluctuations(dirName + os.sep + dirList[d], plot=False, color=colorList(d/dirList.shape[0]))
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
        uplot.plotCorrWithError(meanNum, deltaNum, stdDeltaNum, "$Variance$ $of$ $local$ $number,$ $\\Delta N^2$", "$Local$ $number,$ $N_s$", color='k', logx=True, logy=True)
        plt.pause(0.5)

############################# Cluster mixing time ##############################
def computeClusterMixingTime(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    phi = int(utils.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # first get cluster at initial condition
    if(os.path.exists(dirName + os.sep + "t0/delaunayList.dat")):
        initDenseList = np.loadtxt(dirName + os.sep + "t0/delaunayList.dat")
    else:
        initDenseList,_ = computeVoronoiCluster(dirName + os.sep + "t0/")
    initParticlesInCluster = initDenseList[initDenseList==1].shape[0]
    fraction = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        sharedParticles = 0
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "delaunayList.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "delaunayList.dat")
        else:
            denseList,_ = computeVoronoiCluster(dirSample)
        # check whether the particles in the cluster have changed by threshold
        for i in range(numParticles):
            if(initDenseList[i] == 1 and denseList[i] == 1):
                sharedParticles += 1
        fraction[d] = sharedParticles / initParticlesInCluster
        #print(timeList[d], fraction[d])
    np.savetxt(dirName + os.sep + "mixingTime.dat", np.column_stack((timeList, fraction)))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList[fraction>0], fraction[fraction>0], "$N_c^0(t) / N_c^0$", xlabel = "$Simulation$ $time$", color='k')
        plt.show()

################## Cluster mixing time averaged in time blocks #################
def computeClusterBlockMixingTime(dirName, numBlocks, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    phi = int(utils.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    blockFreq = dirList.shape[0]//numBlocks
    timeList = timeList[:blockFreq]
    fraction = np.zeros((blockFreq, numBlocks))
    for block in range(numBlocks):
        print(block)
        # first get cluster at initial condition
        if(os.path.exists(dirName + os.sep + dirList[block*blockFreq] + "/delaunayList.dat")):
            initDenseList = np.loadtxt(dirName + os.sep + dirList[block*blockFreq] + "/delaunayList.dat")
        else:
            initDenseList,_ = computeDelaunayCluster(dirName + os.sep + dirList[block*blockFreq])
        initParticlesInCluster = initDenseList[initDenseList==1].shape[0]
        for d in range(blockFreq):
            sharedParticles = 0
            dirSample = dirName + os.sep + dirList[block*blockFreq + d]
            if(os.path.exists(dirSample + os.sep + "delaunayList.dat")):
                denseList = np.loadtxt(dirSample + os.sep + "delaunayList.dat")
            else:
                denseList,_ = computeDelaunayCluster(dirSample)
            # check whether the particles in the cluster have changed by threshold
            for i in range(numParticles):
                if(initDenseList[i] == 1 and denseList[i] == 1):
                    sharedParticles += 1
            fraction[d, block] = sharedParticles / initParticlesInCluster
    blockFraction = np.column_stack((np.mean(fraction, axis=1), np.std(fraction, axis=1)))
    np.savetxt(dirName + os.sep + "blockMixingTime.dat", np.column_stack((timeList, blockFraction)))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, blockFraction[:,0], "$N_c^0(t) / N_c^0$", xlabel = "$Simulation$ $time$", color='k')
        plt.show()

############ Time-averaged cluster mixing in log-spaced time window ############
def computeClusterLogMixingTime(dirName, startBlock, maxPower, freqPower):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    timeStep = utils.readFromParams(dirName, "dt")
    phi = int(utils.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    fraction = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            stepFraction = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        denseList1, denseList2 = utils.readDenseListPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        sharedParticles = 0
                        initClusterParticles = denseList1[denseList1==1].shape[0]
                        if(initClusterParticles > 0):
                            for i in range(numParticles):
                                if(denseList1[i] == 1 and denseList2[i] == 1):
                                    sharedParticles += 1
                            stepFraction.append(sharedParticles / initClusterParticles)
                            numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                fraction.append([np.mean(stepFraction), np.std(stepFraction)])
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    fraction = np.array(fraction).reshape((stepList.shape[0],2))
    fraction = fraction[np.argsort(stepList)]
    stepList = np.sort(stepList)
    np.savetxt(dirName + os.sep + "logMixingTime.dat", np.column_stack((stepList*timeStep, fraction)))
    uplot.plotCorrWithError(stepList*timeStep, fraction[:,0], fraction[:,1], ylabel="$C_{vv}(\\Delta t)$", logx = True, color = 'k')
    plt.pause(0.5)

############### Cluster evaporation time averaged in time blocks ###############
def computeClusterBlockEvaporationTime(dirName, numBlocks, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    timeStep = float(utils.readFromParams(dirName, "dt"))
    phi = int(utils.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    blockFreq = dirList.shape[0]//numBlocks
    timeList = timeList[:blockFreq]
    time = np.zeros((blockFreq, numBlocks))
    evaporationTime = []
    for block in range(numBlocks):
        print(block)
        # first get cluster at initial condition
        if(os.path.exists(dirName + os.sep + dirList[block*blockFreq] + "/delaunayBorderList.dat")):
            initBorderList = np.loadtxt(dirName + os.sep + dirList[block*blockFreq] + "/delaunayBorderList.dat")
        else:
            initBorderList,_ = computeDelaunayBorder(dirName + os.sep + dirList[block*blockFreq])
        for d in range(blockFreq):
            sharedParticles = 0
            dirSample = dirName + os.sep + dirList[block*blockFreq + d]
            if(os.path.exists(dirSample + os.sep + "delaunayBorderList.dat")):
                borderList = np.loadtxt(dirSample + os.sep + "delaunayBorderList.dat")
            else:
                borderList,_ = computeDelaunayBorder(dirSample)
            # check whether the particles in the cluster have changed by threshold
            for i in range(numParticles):
                if(initBorderList[i] == 1 and borderList[i] == 0):
                    evaporationTime.append(d*timeStep)
                    initBorderList[i] = 0
    print("average evaporation time:", np.mean(evaporationTime), "+-", np.std(evaporationTime))
    np.savetxt(dirName + os.sep + "evaporationTime.dat", np.column_stack((evaporationTime)))
    if(plot=='plot'):
        pdf, edges = np.histogram(evaporationTime, np.linspace(np.min(evaporationTime), np.max(evaporationTime), 50), density=True)
        edges = (edges[1:] + edges[-1]) * 0.5
        uplot.plotCorrelation(edges, pdf, "$PDF(t_{vapor})$", xlabel = "$Evaporation$ $time,$ $t_{vapor}$", color='k')
        plt.show()

############################ Velocity distribution #############################
def averageParticleVelPDFCluster(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    #dirList = dirList[-50:]
    velInCluster = np.empty(0)
    velOutCluster = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "delaunayList!.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "delaunayList.dat")
        else:
            denseList,_ = computeDelaunayCluster(dirSample, threshold=0.78, filter=False)
        vel = np.loadtxt(dirSample + os.sep + "particleVel.dat")
        velNorm = np.linalg.norm(vel, axis=1)
        velInCluster = np.append(velInCluster, velNorm[denseList==1].flatten())
        velOutCluster = np.append(velOutCluster, velNorm[denseList!=1].flatten())
    # in cluster
    velInCluster = velInCluster[velInCluster>0]
    mean = np.mean(velInCluster)
    Temp = np.var(velInCluster)
    skewness = np.mean((velInCluster - mean)**3)/Temp**(3/2)
    kurtosis = np.mean((velInCluster - mean)**4)/Temp**2
    data = velInCluster# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity inside the cluster: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Velocity$ $distribution,$ $P(v)$", xlabel = "$Velocity,$ $v$", color='b')
    np.savetxt(dirName + os.sep + "velPDFInCluster.dat", np.column_stack((edges, pdf)))
    # out of cluster
    velOutCluster = velOutCluster[velOutCluster>0]
    mean = np.mean(velOutCluster)
    Temp = np.var(velOutCluster)
    skewness = np.mean((velOutCluster - mean)**3)/Temp**(3/2)
    kurtosis = np.mean((velOutCluster - mean)**4)/Temp**2
    data = velOutCluster# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity outside the cluster: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Velocity$ $distribution,$ $P(v)$", xlabel = "$Velocity,$ $v$", color='g')
    np.savetxt(dirName + os.sep + "velPDFOutCluster.dat", np.column_stack((edges, pdf)))
    if(plot == "plot"):
        #plt.pause(0.5)
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
            if(cluster=="deep" and deepList[particleId] == 1):
                ax.add_artist(plt.Circle([x, y], r, edgecolor='k', facecolor='k', alpha=0.6, linewidth=0.5))
        plt.show()
    return connectLabel, noClusterList, particleLabel

def searchDBClusters(dirName, eps=0, min_samples=8, plot=False, contactFilter='contact'):
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

def averageDBClusterSize(dirName, dirSpacing, eps=0.03, min_samples=10, plot=False, contactFilter=False):
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
    distance = computeDistances(pos, boxSize)
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

######################## Compute cluster shape parameter #######################
def computeClusterVoronoiShape(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    shapeParam = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        # perimeter
        if(os.path.exists(dirSample + os.sep + "borderLength.dat")):
            shapeParam[d,0] = np.loadtxt(dirSample + os.sep + "borderLength.dat")
        else:
            _, shapeParam[d,0] = computeVoronoiBorder(dirSample)
        # area
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
                shapeParam[d,1] += (voroArea[i] - overlapArea)
        # shape parameter
        shapeParam[d,2] = shapeParam[d,0]**2 / (4 * np.pi * shapeParam[d,1])
    np.savetxt(dirName + os.sep + "voronoiShape.dat", np.column_stack((timeList, shapeParam)))
    print("Cluster perimeter: ", np.mean(shapeParam[:,0]), " +- ", np.std(shapeParam[:,0]))
    print("Cluster area: ", np.mean(shapeParam[:,1]), " +- ", np.std(shapeParam[:,1]))
    print("Cluster shape parameter: ", np.mean(shapeParam[:,2]), " +- ", np.std(shapeParam[:,2]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, shapeParam[:,0], "$Perimeter,$ $Area,$ $Shape$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, shapeParam[:,1], "$Perimeter,$ $Area,$ $Shape$", xlabel = "$Time,$ $t$", color='g')
        uplot.plotCorrelation(timeList, shapeParam[:,2], "$Perimeter,$ $Area,$ $Shape$", xlabel = "$Time,$ $t$", color='k')
        plt.pause(0.5)
    return shapeParam

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
    checkDelaunayInclusivity(delaunay.simplices, pos, rad, boxSize)

################### Cluster augmented packing for plotting #####################
def computeAugmentedDelaunayCluster(dirName, threshold1=0.78, threshold2=0.45, shiftx=0, shifty=0):
    sep = utils.getDirSep(dirName, "boxSize")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    pos = utils.shiftPositions(pos, boxSize, shiftx, shifty)
    newPos, newRad, newIndices = utils.augmentPacking(pos, rad)
    delaunay = Delaunay(newPos)
    simplices = delaunay.simplices
    simplexDensity, _ = utils.computeDelaunayDensity(simplices, newPos, newRad, boxSize*1.5)
    insideIndex = utils.getInsideBoxDelaunaySimplices(simplices, newPos, boxSize)
    simplexLabelList = np.ones(simplexDensity.shape[0])
    for simplexId in range(simplexLabelList.shape[0]):
        if(simplexDensity[simplexId] > threshold1):
            simplexLabelList[simplexId] = 0
        if(simplexDensity[simplexId] < threshold1 and simplexDensity[simplexId] > threshold2):
            simplexLabelList[simplexId] = 0.5
    insideIndex = utils.getInsideBoxDelaunaySimplices(simplices, newPos, boxSize)
    return newPos, simplices, simplexLabelList, simplexDensity

############################## Delaunay clustering #############################
def computeDelaunayCluster(dirName, threshold=0.78, filter=False, plot=False):
    sep = utils.getDirSep(dirName, "boxSize")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    denseList = np.zeros(numParticles)
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    #delaunay = Delaunay(pos)
    #simplices = delaunay.simplices
    simplices = utils.getPBCDelaunay(pos, rad, boxSize)
    # compute delaunay densities
    simplexDensity, simplexArea = utils.computeDelaunayDensity(simplices, pos, rad, boxSize)
    if(np.argwhere(simplexDensity<0)[:,0].shape[0] > 0):
        print("There are", np.argwhere(simplexDensity<0)[:,0].shape[0], "negative simplex densities")
        print(simplices[np.argwhere(simplexDensity<0)[:,0]])
    denseSimplexList = np.zeros(simplexDensity.shape[0])
    #print("average local density:", np.mean(simplexDensity))
    # first find dense simplices
    for i in range(simplexDensity.shape[0]):
        if(simplexDensity[i] > threshold):
            denseSimplexList[i] = 1
    #print("Fraction of dense simplices: ", denseSimplexList[denseSimplexList==1].shape[0]/denseSimplexList.shape[0])
    # if all the simplices touching a particle are dense then the particle is dense
    for i in range(numParticles):
        indices = np.argwhere(simplices==i)[:,0]
        #count = 0
        #maxCount = indices.shape[0]
        for sIndex in indices:
            if(denseSimplexList[sIndex] == 1):
                denseList[i] = 1
        #        count += 1
        #if(count > int(maxCount / 1.5)):
        #    denseList[i] = 1
    #print("Fraction of dense particles: ", denseList[denseList==1].shape[0]/denseList.shape[0])
    if(filter=='filter'):
        connectList = np.zeros(numParticles)
        for i in range(numParticles):
            if(np.sum(contacts[i]!=-1)>5):
                denseContacts = 0
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    if(denseList[c] == 1):
                        denseContacts += 1
                if(denseContacts > 1):
                # this is at least a four particle cluster
                    connectList[i] = 1
        denseList[connectList==0] = 0
        # this is to include contacts of particles belonging to the cluster
        for times in range(5):
            for i in range(numParticles):
                if(denseList[i] == 1):
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(denseList[c] != 1):
                            denseList[c] = 1
        # look for rattles inside the fluid and label them as dense particles
        neighborCount = np.zeros(numParticles)
        denseNeighborCount = np.zeros(numParticles)
        for i in range(numParticles):
            if(denseList[i]==0):
                for sIndex in np.argwhere(simplices==i)[:,0]:
                    indices = np.delete(simplices[sIndex], np.argwhere(simplices[sIndex]==i)[0,0])
                    for index in indices:
                        neighborCount[i] += 1
                        if(denseList[index] == 1):
                            denseNeighborCount[i] += 1
        rattlerList = np.zeros(numParticles)
        for i in range(numParticles):
            if(denseList[i]==0):
                if(neighborCount[i] == denseNeighborCount[i]):
                    rattlerList[i] = 1
        denseList[rattlerList==1] = 1
        #print("Number of dense particles after contact filter: ", denseList[denseList==1].shape[0])
        # need to update denseSimplexList after the applied filters
        for sIndex in range(simplices.shape[0]):
            indexCount = 0
            for pIndex in range(simplices[sIndex].shape[0]):
                indexCount += denseList[simplices[sIndex][pIndex]]
            if(indexCount==3):
                denseSimplexList[sIndex] = 1
            else:
                denseSimplexList[sIndex] = 0
    np.savetxt(dirName + "/delaunayList.dat", denseList)
    np.savetxt(dirName + "/denseSimplexList.dat", denseSimplexList)
    np.savetxt(dirName + "/simplexDensity.dat", simplexDensity)
    np.savetxt(dirName + "/simplexArea.dat", simplexArea)
    # find simplices at the interface between dense and non-dense
    borderSimplexList = np.zeros(denseSimplexList.shape[0])
    for i in range(numParticles):
        if(denseList[i]==1):
            for sIndex in np.argwhere(simplices==i)[:,0]:
                indices = np.delete(simplices[sIndex], np.argwhere(simplices[sIndex]==i)[0,0])
                for index in indices:
                    if(denseList[index] == 0 and index>=0):
                        borderSimplexList[sIndex] = 1
    np.savetxt(dirName + "/borderSimplexList.dat", borderSimplexList)
    #print("average density of dense simplices:", np.mean(simplexDensity[denseSimplexList==1]), np.min(simplexDensity[denseSimplexList==1]), np.max(simplexDensity[denseSimplexList==1]))
    if(plot=='plot'):
        print("there are", simplices[simplexDensity<0].shape[0], "simplices with negative density")
        print(simplices[simplexDensity<0.02])
        uplot.plotCorrelation(np.arange(1, simplexDensity.shape[0]+1, 1), np.sort(simplexDensity), "$\\varphi^{Simplex}$", xlabel = "$Simplex$ $index$", color='k')
        plt.savefig("/home/francesco/Pictures/soft/delaunayDensityCDF.png", transparent=True, format="png")
        plt.show()
        numBins = 100
        pdf, edges = np.histogram(simplexDensity, bins=np.linspace(0, 1, numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        uplot.plotCorrelation(edges, pdf, "$PDF(\\varphi^{Simplex})$", xlabel = "$\\varphi^{Simplex}$", color='b')
        #plt.yscale('log')
        plt.savefig("/home/francesco/PictureSoftDelaunayDensityPDF.png", transparent=True, format="png")
        plt.show()
    return denseList, simplexDensity

######################## Average delaunay local density #########################
def averageLocalDelaunayDensity(dirName, numBins=16, plot=False, dirSpacing=1):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    localDensity = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "simplexDensity.dat")):
            simplexDensity = np.loadtxt(dirSample + os.sep + "simplexDensity.dat")
        else:
            _, simplexDensity = computeDelaunayCluster(dirSample)
        pos = utils.getPBCPositions(dirSample + os.sep + "particlePos.dat", boxSize)
        simplexPos = utils.getDelaunaySimplexPos(pos, rad, boxSize)
        delaunayDensity = utils.computeLocalDelaunayDensityGrid(simplexPos, simplexDensity, xbin, ybin)
        localDensity = np.append(localDensity, delaunayDensity)
    localDensity = np.sort(localDensity).flatten()
    localDensity = localDensity[localDensity>0]
    alpha2 = np.mean(localDensity**4)/(2*(np.mean(localDensity**2)**2)) - 1
    pdf, edges = np.histogram(localDensity, bins=np.linspace(np.min(localDensity), np.max(localDensity), 100), density=True)
    edges = (edges[:-1] + edges[1:])/2
    np.savetxt(dirName + os.sep + "localDelaunayDensity-N" + str(numBins) + ".dat", np.column_stack((edges, pdf)))
    data = np.column_stack((np.mean(localDensity), np.var(localDensity), alpha2))
    np.savetxt(dirName + os.sep + "localDelaunayDensity-N" + str(numBins) + "-stats.dat", data)
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
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        if(os.path.exists(dirSample + os.sep + "denseSimplexList!.dat")):
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexDensity = np.loadtxt(dirSample + os.sep + "simplexDensity.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        else:
            _, simplexDensity = computeDelaunayCluster(dirSample, threshold=0.78, filter='filter')
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
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
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        # area
        if(os.path.exists(dirSample + os.sep + "denseSimplexList.dat")):
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexDensity = np.loadtxt(dirSample + os.sep + "simplexDensity.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        else:
            _, simplexDensity = computeDelaunayCluster(dirSample)
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        occupiedArea = simplexDensity * simplexArea
        for i in range(simplexDensity.shape[0]):
            if(denseSimplexList[i]==1):
                area[d,0] += simplexArea[i]
            else:
                area[d,1] += simplexArea[i]
    np.savetxt(dirName + os.sep + "delaunayArea.dat", np.column_stack((timeList, area)))
    print("Fluid area: ", np.mean(area[:,0]), " +- ", np.std(area[:,0]))
    print("Gas area: ", np.mean(area[:,1]), " +- ", np.std(area[:,1]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, area[:,0], "$Area$ $fraction$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, area[:,1], "$Area$ $fraction$", xlabel = "$Time,$ $t$", color='g')
        plt.pause(0.5)

######################## Compute delaunay cluster border #######################
def computeDelaunayBorder(dirName, threshold=0.85, filter='filter'):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    phi = utils.readFromParams(dirName + sep, "phi")
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    # need to center the cluster for voronoi border detection
    pos = utils.centerPositions(pos, rad, boxSize)
    delaunay = Delaunay(pos)
    # check if denseList already exists
    if(os.path.exists(dirName + os.sep + "delaunayList.dat")):
        denseList = np.loadtxt(dirName + os.sep + "delaunayList.dat")
    else:
        denseList,_ = computeDelaunayCluster(dirName, threshold, filter=filter)
    borderList = np.zeros(numParticles)
    for i in range(numParticles):
        if(denseList[i]==1):
            for sIndex in np.argwhere(delaunay.simplices==i)[:,0]:
                indices = np.delete(delaunay.simplices[sIndex], np.argwhere(delaunay.simplices[sIndex]==i)[0,0])
                for index in indices:
                    if(denseList[index] == 0 and index>=0):
                        borderList[i] = 1
    # compute angles with respect to center of cluster to sort border particles
    borderPos = pos[borderList==1]
    borderPos = utils.sortBorderPos(borderPos, borderList, boxSize)
    np.savetxt(dirName + os.sep + "borderPos.dat", borderPos)
    # compute border length by summing over segments on the border
    borderLength = 0
    for i in range(1,borderPos.shape[0]):
        borderLength += np.linalg.norm(utils.pbcDistance(borderPos[i], borderPos[i-1], boxSize))
    #print("Number of dense particles at the interface: ", borderList[borderList==1].shape[0])
    #print("Border length from delaunay edges: ", borderLength)
    np.savetxt(dirName + os.sep + "delaunayBorderList.dat", borderList)
    np.savetxt(dirName + os.sep + "delaunayBorderLength.dat", np.array([borderLength]))
    return borderList, borderLength

######################## Compute cluster shape parameter #######################
def computeClusterDelaunayShape(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    shapeParam = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        # perimeter
        if(os.path.exists(dirSample + os.sep + "delaunayBorderLength.dat")):
            shapeParam[d,0] = np.loadtxt(dirSample + os.sep + "delaunayBorderLength.dat")
        else:
            _, shapeParam[d,0] = computeDelaunayBorder(dirSample)
        # area
        if(os.path.exists(dirSample + os.sep + "denseSimplexList.dat")):
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexDensity = np.loadtxt(dirSample + os.sep + "simplexDensity.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        else:
            _, simplexDensity = computeDelaunayCluster(dirSample)
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        occupiedArea = simplexDensity * simplexArea
        fluidArea = 0
        for i in range(simplexDensity.shape[0]):
            if(denseSimplexList[i]==1):
                shapeParam[d,1] += simplexArea[i]
        # shape parameter
        shapeParam[d,2] = shapeParam[d,0]**2 / (4 * np.pi * shapeParam[d,1])
    np.savetxt(dirName + os.sep + "delaunayShape.dat", np.column_stack((timeList, shapeParam)))
    print("Cluster perimeter: ", np.mean(shapeParam[:,0]), " +- ", np.std(shapeParam[:,0]))
    print("Cluster area: ", np.mean(shapeParam[:,1]), " +- ", np.std(shapeParam[:,1]))
    print("Cluster shape parameter: ", np.mean(shapeParam[:,2]), " +- ", np.std(shapeParam[:,2]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, shapeParam[:,0], "$Perimeter,$ $Area,$ $Shape$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, shapeParam[:,1], "$Perimeter,$ $Area,$ $Shape$", xlabel = "$Time,$ $t$", color='g')
        uplot.plotCorrelation(timeList, shapeParam[:,2], "$Perimeter,$ $Area,$ $Shape$", xlabel = "$Time,$ $t$", color='k')
        plt.pause(0.5)
    return shapeParam

######################### Cluster pressure components ##########################
def computeDelaunayClusterPressureVSTime(dirName, dirSpacing=1):
    dim = 2
    gamma = float(utils.readFromDynParams(dirName, "damping"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    wallPressure = np.zeros(dirList.shape[0])
    fluidPressure = np.zeros((dirList.shape[0],3))
    gasPressure = np.zeros((dirList.shape[0],3))
    boxLength = 2 * (boxSize[0] + boxSize[1])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "delaunayList!.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "delaunayList.dat")
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        else:
            denseList, _ = computeDelaunayCluster(dirSample)
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        # first fill out the area occupied by fluid and the gas
        fluidArea = 0
        gasArea = 0
        for i in range(simplexArea.shape[0]):
            if(denseSimplexList[i]==1):
                fluidArea += simplexArea[i]
            else:
                gasArea += simplexArea[i]
        # compute stress components
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        angle = np.mod(angle, 2*np.pi)
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        virialIn = 0
        thermalIn = 0
        activeIn = 0
        virialOut = 0
        thermalOut = 0
        activeOut = 0
        for i in range(numParticles):
            # wall pressure
            isWall, wallPos = utils.isNearWall(pos[i], rad[i], boxSize)
            if(isWall == True):
                delta = pos[i] - wallPos
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / rad[i]
                if(overlap > 0):
                    gradMultiple = ec * overlap / rad[i]
                    wallForce = gradMultiple * delta / distance
                    wallPressure[d] += np.linalg.norm(wallForce) / boxLength
            # particle pressure components
            if(denseList[i]==1):
                thermalIn += np.linalg.norm(vel[i])**2
                activeIn += np.sum(vel[i] * director[i])
            else:
                thermalOut += np.linalg.norm(vel[i])**2
                activeOut += np.sum(vel[i] * director[i])
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    if(denseList[i]==1):
                        virialIn += 0.5 * np.sum(force * delta)
                    else:
                        virialOut += 0.5 * np.sum(force * delta)
        if(fluidArea > 0):
            fluidPressure[d,0] = virialIn / (dim * fluidArea) # double counting
            fluidPressure[d,1] = thermalIn / fluidArea # dim k_B T / dim, dim cancels out
            fluidPressure[d,2] = driving * activeIn / (dim * 2*Dr * fluidArea)
        if(gasArea > 0):
            gasPressure[d,0] = virialOut / (dim * gasArea) # double counting
            gasPressure[d,1] = thermalOut / gasArea # dim k_B T / dim, dim cancels out
            gasPressure[d,2] = driving * activeOut / (dim * 2*Dr * gasArea)
    wallPressure *= sigma**2
    fluidPressure *= sigma**2
    gasPressure *= sigma**2
    #borderPressure *= sigma**2
    np.savetxt(dirName + os.sep + "delaunayPressure.dat", np.column_stack((timeList, wallPressure, fluidPressure, gasPressure)))
    # pressure components in the fluid
    print("dense pressure: ", np.mean(fluidPressure[:,0] + fluidPressure[:,1] + fluidPressure[:,2]), " +/- ", np.std(fluidPressure[:,0] + fluidPressure[:,1] + fluidPressure[:,2]))
    print("dense virial pressure: ", np.mean(fluidPressure[:,0]), " +/- ", np.std(fluidPressure[:,0]))
    print("dense thermal pressure: ", np.mean(fluidPressure[:,1]), " +/- ", np.std(fluidPressure[:,1]))
    print("dense active pressure: ", np.mean(fluidPressure[:,2]), " +/- ", np.std(fluidPressure[:,2]))
    # pressure components in the gas
    print("\ndilute pressure: ", np.mean(gasPressure[:,0] + gasPressure[:,1] + gasPressure[:,2]), " +/- ", np.std(gasPressure[:,0] + gasPressure[:,1] + gasPressure[:,2]))
    print("dilute virial pressure: ", np.mean(gasPressure[:,0]), " +/- ", np.std(gasPressure[:,0]))
    print("dilute thermal pressure: ", np.mean(gasPressure[:,1]), " +/- ", np.std(gasPressure[:,1]))
    print("dilute active pressure: ", np.mean(gasPressure[:,2]), " +/- ", np.std(gasPressure[:,2]), "\n")

################################################################################
############################### Cluster pressure ###############################
################################################################################
def computeParticleStress(dirName, prop='prop'):
    dim = 2
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    if(prop == 'prop'):
        Dr = float(utils.readFromDynParams(dirName + sep, "Dr"))
        gamma = float(utils.readFromDynParams(dirName + sep, "damping"))
        driving = float(utils.readFromDynParams(dirName + sep, "f0"))
    stress = np.zeros((numParticles,4))
    #pos = np.loadtxt(dirName + "/particlePos.dat")
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    vel = np.loadtxt(dirName + "/particleVel.dat")
    if(prop == 'prop'):
        #angle = np.loadtxt(dirName + "/particleAngles.dat")
        angle = utils.getMOD2PIAngles(dirName + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
    contacts = np.loadtxt(dirName + "/particleContacts.dat").astype(np.int64)
    for i in range(numParticles):
        virial = 0
        #active = v0 - np.sum(vel[i]*director[i])
        energy = 0
        for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
            radSum = rad[i] + rad[c]
            delta = utils.pbcDistance(pos[i], pos[c], boxSize)
            distance = np.linalg.norm(delta)
            overlap = 1 - distance / radSum
            if(overlap > 0):
                gradMultiple = ec * overlap / radSum
                force = gradMultiple * delta / distance
                virial += 0.5 * np.sum(force * delta) # double counting
                #active += np.sum(force * director[i]) / gamma
                energy += 0.5 * ec * overlap**2 * 0.5 # double counting and e = k/2
        stress[i,0] = virial
        stress[i,1] = np.linalg.norm(vel[i])**2
        if(prop == 'prop'):
            stress[i,2] = driving * np.sum(vel[i] * director[i]) / (2*Dr)
        stress[i,3] = energy
    np.savetxt(dirName + os.sep + "particleStress.dat", stress)
    print('potential energy: ', np.mean(stress[:,3]), ' +- ', np.std(stress[:,3]))
    print('kinetic energy: ', np.mean(stress[:,1]*0.5), ' +- ', np.std(stress[i,1]*0.5))
    print('energy: ', np.mean(stress[:,1]*0.5 + stress[:,3]), ' +- ', np.std(stress[i,1]*0.5 + stress[:,3]))
    return stress

def computeParticleStressVSTime(dirName, dirSpacing=1):
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        computeParticleStress(dirSample)

########################## Total pressure components ###########################
def computePressureVSTime(dirName, bound = False, prop = False, dirSpacing=1):
    dim = 2
    if(prop == "prop"):
        gamma = float(utils.readFromDynParams(dirName, "damping"))
        driving = float(utils.readFromDynParams(dirName, "f0"))
        Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    wallPressure = np.zeros(dirList.shape[0])
    pressure = np.zeros((dirList.shape[0],3))
    boxLength = 2 * (boxSize[0] + boxSize[1])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        if(prop == "prop"):
            angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
            director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        wall = 0
        virial = 0
        thermal = 0
        active = 0
        for i in range(numParticles):
            # wall pressure
            isWall, wallPos = utils.isNearWall(pos[i], rad[i], boxSize)
            if(isWall == True):
                delta = pos[i] - wallPos
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / rad[i]
                if(overlap > 0):
                    gradMultiple = ec * overlap / rad[i]
                    wallForce = gradMultiple * delta / distance
                    if(bound == "bound"):
                        wall -= np.sum(wallForce * pos[i]) / dim
                        #if(prop == "prop"):
                        #    active += np.sum(wallForce * director[i]) / gamma # wall director
                    else:
                        wall += np.linalg.norm(wallForce) / boxLength
            # particle pressure components
            thermal += np.linalg.norm(vel[i])**2
            if(prop == "prop"):
                #active += v0 - np.sum(vel[i] * director[i])
                active += np.sum(vel[i] * director[i])
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    virial += 0.5 * np.sum(force * delta) # double counting
                    #if(prop == "prop"):
                    #    active += 0.5 * np.sum(force * director[i]) / gamma # double counting
        wallPressure[d] = wall
        pressure[d,0] = virial / dim
        pressure[d,1] = thermal # dim k_B T / dim, dim cancels out
        if(prop == "prop"):
            pressure[d,2] = driving * active / (dim * 2*Dr)
    pressure *= sigma**2
    wallPressure *= sigma**2
    np.savetxt(dirName + os.sep + "pressure.dat", np.column_stack((timeList, wallPressure, pressure)))
    print("bulk pressure: ", np.mean(pressure[:,0] + pressure[:,1] + pressure[:,2]), " +/- ", np.std(pressure[:,0] + pressure[:,1] + pressure[:,2]))
    print("virial pressure: ", np.mean(pressure[:,0]), " +/- ", np.std(pressure[:,0]))
    print("thermal pressure: ", np.mean(pressure[:,1]), " +/- ", np.std(pressure[:,1]))
    if(prop == "prop"):
        print("active pressure: ", np.mean(pressure[:,2]), " +/- ", np.std(pressure[:,2]))
    print("pressure on the wall: ", np.mean(wallPressure), " +/- ", np.std(wallPressure))

######################### Cluster pressure components ##########################
def computeClusterPressureVSTime(dirName, bound = False, prop = False, dirSpacing=1):
    dim = 2
    if(prop == "prop"):
        gamma = float(utils.readFromDynParams(dirName, "damping"))
        driving = float(utils.readFromDynParams(dirName, "f0"))
        Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    wallPressure = np.zeros(dirList.shape[0])
    pressureIn = np.zeros((dirList.shape[0],3))
    pressureOut = np.zeros((dirList.shape[0],3))
    boxLength = 2 * (boxSize[0] + boxSize[1])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "voroDensity.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
            voroDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            denseList, voroDensity = computeVoronoiCluster(dirSample)
        #if(os.path.exists(dirSample + os.sep + "borderList!.dat")):
        #    borderList = np.loadtxt(dirSample + os.sep + "borderList.dat")
        #else:
        #    borderList,_ = computeVoronoiBorder(dirSample)
        voroVolume = np.pi*rad**2/voroDensity
        #pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        if(prop == "prop"):
            angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
            angle = np.mod(angle, 2*np.pi)
            director = np.array([np.cos(angle), np.sin(angle)]).T
            #activeForce = driving * np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        wall = 0
        volumeIn = 0
        virialIn = 0
        thermalIn = 0
        activeIn = 0
        volumeOut = 0
        virialOut = 0
        thermalOut = 0
        activeOut = 0
        for i in range(numParticles):
            # wall pressure
            isWall, wallPos = utils.isNearWall(pos[i], rad[i], boxSize)
            if(isWall == True):
                delta = pos[i] - wallPos
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / rad[i]
                if(overlap > 0):
                    gradMultiple = ec * overlap / rad[i]
                    wallForce = gradMultiple * delta / distance
                    if(bound == "bound"):
                        wall -= np.sum(wallForce * pos[i]) / dim
                        #if(prop == "prop"):
                        #    if(denseList[i] == 1):
                        #        activeIn += np.sum(wallForce * director[i]) / gamma
                        #    else:
                        #        activeOut += np.sum(wallForce * director[i]) / gamma
                    else:
                        wall += np.linalg.norm(wallForce) / boxLength
            #if(borderList[i] == 0):
            # particle pressure components
            if(denseList[i] == 1):
                volumeIn += voroVolume[i]
                thermalIn += np.linalg.norm(vel[i])**2
                if(prop == "prop"):
                #    activeIn += (v0 - np.sum(vel[i] * director[i]))
                    activeIn += np.sum(vel[i] * director[i])
            else:
                volumeOut += voroVolume[i]
                thermalOut += np.linalg.norm(vel[i])**2
                if(prop == "prop"):
                #    activeOut += (v0 - np.sum(vel[i] * director[i]))
                    activeOut += np.sum(vel[i] * director[i])
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    if(denseList[i] == 1):
                        virialIn += 0.5 * np.sum(force * delta)
                        #if(prop == "prop"):
                        #    activeIn += 0.5 * np.sum(force * director[i]) / gamma
                    else:
                        virialOut += 0.5 * np.sum(force * delta)
                        #if(prop == "prop"):
                        #    activeOut += 0.5 * np.sum(force * director[i]) / gamma
        wallPressure[d] = wall
        if(volumeIn > 0):
            pressureIn[d,0] = virialIn / (dim * volumeIn) # double counting
            pressureIn[d,1] = thermalIn / volumeIn # dim k_B T / dim, dim cancels out
            if(prop == "prop"):
                pressureIn[d,2] = driving * activeIn / (dim * 2*Dr * volumeIn)
        if(volumeOut > 0):
            pressureOut[d,0] = virialOut / (dim * volumeOut) # double counting
            pressureOut[d,1] = thermalOut / volumeOut # dim k_B T / dim, dim cancels out
            if(prop == "prop"):
                pressureOut[d,2] = driving * activeOut / (dim * 2*Dr * volumeOut)
    pressureIn *= sigma**2
    pressureOut *= sigma**2
    wallPressure *= sigma**2
    np.savetxt(dirName + os.sep + "clusterPressure.dat", np.column_stack((timeList, wallPressure, pressureIn, pressureOut)))
    # pressure components in the fluid
    print("dense pressure: ", np.mean(pressureIn[:,0] + pressureIn[:,1] + pressureIn[:,2]), " +/- ", np.std(pressureIn[:,0] + pressureIn[:,1] + pressureIn[:,2]))
    print("dense virial pressure: ", np.mean(pressureIn[:,0]), " +/- ", np.std(pressureIn[:,0]))
    print("dense thermal pressure: ", np.mean(pressureIn[:,1]), " +/- ", np.std(pressureIn[:,1]))
    if(prop == "prop"):
        print("dense active pressure: ", np.mean(pressureIn[:,2]), " +/- ", np.std(pressureIn[:,2]))
    # pressure components in the gas
    print("\ndilute pressure: ", np.mean(pressureOut[:,0] + pressureOut[:,1] + pressureOut[:,2]), " +/- ", np.std(pressureOut[:,0] + pressureOut[:,1] + pressureOut[:,2]))
    print("dilute virial pressure: ", np.mean(pressureOut[:,0]), " +/- ", np.std(pressureOut[:,0]))
    print("dilute thermal pressure: ", np.mean(pressureOut[:,1]), " +/- ", np.std(pressureOut[:,1]))
    if(prop == "prop"):
        print("dilute active pressure: ", np.mean(pressureOut[:,2]), " +/- ", np.std(pressureOut[:,2]), "\n")
    if(bound == "bound"):
        print("pressure on the wall: ", np.mean(wallPressure), " +/- ", np.std(wallPressure))

######################### Average radial surface work ##########################
def averageRadialSurfaceWork(dirName, plot=False, dirSpacing=1):
    dim = 2
    ec = 240
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # work components
    work = np.zeros((dirList.shape[0],4))
    border = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "delaunayBorderList.dat")):
            borderList = np.loadtxt(dirSample + os.sep + "delaunayBorderList.dat")
            border[d] = np.loadtxt(dirSample + os.sep + "delaunayBorderLength.dat")
        else:
            borderList, border[d] = computeDelaunayBorder(dirSample)
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            compute = False
            if(borderList[i] == 1):
                compute = True
            else:
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    if(borderList[c] == 1):
                        compute = True
            if(compute == True):
                virial = 0
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    radSum = rad[i] + rad[c]
                    delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                    distance = np.linalg.norm(delta)
                    overlap = 1 - distance / radSum
                    if(overlap > 0):
                        gradMultiple = ec * overlap / radSum
                        force = gradMultiple * delta / distance
                        virial += 0.5 * np.sum(force * delta)
                work[d,0] += virial
                work[d,1] += np.linalg.norm(vel[i])**2
                work[d,2] += driving * np.sum(vel[i] * director[i]) / (2*Dr)
                work[d,3] += (virial + np.linalg.norm(vel[i])**2 + driving * np.sum(vel[i] * director[i]) / (2*Dr))
    np.savetxt(dirName + os.sep + "surfaceWork.dat", np.column_stack((timeList, work, border)))
    print("average surface tension:", np.mean((work[1:,3] - work[:-1,3]) / (border[1:] - border[:-1]))*sigma, "+-", np.std((work[1:,3] - work[:-1,3]) / (border[1:] - border[:-1]))*sigma)
    if(plot=='plot'):
        uplot.plotCorrelation(timeList[1:], sigma*(work[1:,0] - work[:-1,0])/(border[1:] - border[:-1]), "$Surface$ $tension$", xlabel = "$Simulation$ $time$", color='k', lw=1.5)
        uplot.plotCorrelation(timeList[1:], sigma*(work[1:,1] - work[:-1,1])/(border[1:] - border[:-1]), "$Surface$ $tension$", xlabel = "$Simulation$ $time$", color='r', lw=1.5)
        uplot.plotCorrelation(timeList[1:], sigma*(work[1:,2] - work[:-1,2])/(border[1:] - border[:-1]), "$Surface$ $tension$", xlabel = "$Simulation$ $time$", color=[1,0.5,0], lw=1.5)
        uplot.plotCorrelation(timeList[1:], sigma*(work[1:,3] - work[:-1,3])/(border[1:] - border[:-1]), "$Surface$ $tension$", xlabel = "$Simulation$ $time$", color='b', lw=1)
        #plt.pause(0.5)
        plt.show()

###################### Average radial pressure profile #########################
def averageRadialPressureProfile(dirName, shiftx, shifty, dirSpacing=1):
    dim = 2
    ec = 240
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0]/2, 2*np.mean(rad))
    binArea = np.pi * (bins[1:]**2 - bins[:-1]**2)
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    virial = np.zeros(bins.shape[0]-1)
    thermal = np.zeros(bins.shape[0]-1)
    active = np.zeros(bins.shape[0]-1)
    total = np.zeros(bins.shape[0]-1)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "denseList.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
            voroDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            denseList, voroDensity = computeVoronoiCluster(dirSample)
        # CENTER CLUSTER BY BRUTE FORCE - NEED TO IMPLEMENT SOMETHING BETTER
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if(shiftx != 0 or shiftx != 0):
            pos = utils.shiftPositions(pos, boxSize, -shiftx, -shifty)
        #pos = utils.centerPositions(pos, rad, boxSize, denseList)
        centerOfMass = np.mean(pos[denseList==1], axis=0)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            distanceToCOM = np.linalg.norm(utils.pbcDistance(pos[i], centerOfMass, boxSize))
            work = 0
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    work += 0.5 * np.sum(force * delta)
            for j in range(bins.shape[0]-1):
                if(distanceToCOM > bins[j] and distanceToCOM < bins[j+1]):
                    virial[j] += work / dim
                    thermal[j] += np.linalg.norm(vel[i])**2 / dim
                    active[j] += driving * np.sum(vel[i] * director[i]) / (4*Dr * dim)
                    total[j] += (work + np.linalg.norm(vel[i])**2 + driving * np.sum(vel[i] * director[i]) / (4*Dr)) / dim
    virial *= sigma**2/(binArea * dirList.shape[0])
    thermal *= sigma**2/(binArea * dirList.shape[0])
    active *= sigma**2/(binArea * dirList.shape[0])
    total *= sigma**2/(binArea * dirList.shape[0])
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((centers, virial, thermal, active, total)))
    uplot.plotCorrelation(centers, virial, "$Pressure$ $profile$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(centers, thermal, "$Pressure$ $profile$", xlabel = "$Distance$", color='r', lw=1.5)
    uplot.plotCorrelation(centers, active, "$Pressure$ $profile$", xlabel = "$Distance$", color=[1,0.5,0], lw=1.5)
    uplot.plotCorrelation(centers, total, "$Pressure$ $profile$", xlabel = "$Distance$", color='b', lw=1)
    plt.show()

############################ Linear surface tension ############################
def averageRadialTension(dirName, shiftx, shifty, dirSpacing=1):
    dim = 2
    ec = 240
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0]/2, 2*np.mean(rad))
    binArea = np.pi * (bins[1:]**2 - bins[:-1]**2)
    binWidth = bins[1:] - bins[:-1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    virial = np.zeros((bins.shape[0]-1,2))
    active = np.zeros((bins.shape[0]-1,2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "denseList.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
            voroDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            denseList, voroDensity = computeVoronoiCluster(dirSample)
        # CENTER CLUSTER BY BRUTE FORCE - NEED TO IMPLEMENT SOMETHING BETTER
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if(shiftx != 0 or shiftx != 0):
            pos = utils.shiftPositions(pos, boxSize, -shiftx, -shifty)
        #pos = utils.centerPositions(pos, rad, boxSize, denseList)
        centerOfMass = np.mean(pos[denseList==1], axis=0)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        workn = 0
        workt = 0
        for i in range(numParticles):
            # Convert pressure components into normal and tangential components
            relPos = utils.pbcDistance(pos[i], centerOfMass, boxSize)
            relAngle = np.arctan2(relPos[1], relPos[0])
            unitVector = np.array([np.cos(relAngle), np.sin(relAngle)])
            velNT = utils.projectToNormalTangentialComp(vel[i], unitVector)
            directorNT = utils.projectToNormalTangentialComp(director[i], unitVector)
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    # normal minus tangential component
                    forceNT = utils.projectToNormalTangentialComp(force, unitVector)
                    deltaNT = utils.projectToNormalTangentialComp(delta, unitVector)
                    workn += 0.5 * np.dot(forceNT[0], deltaNT[0])
                    workt += 0.5 * np.dot(forceNT[1], deltaNT[1])
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] < bins[j+1]):
                    virial[j,0] += workn * binWidth[j] / (dim * binArea[j])
                    virial[j,1] += workt * binWidth[j] / (dim * binArea[j])
                    active[j,0] += driving * np.dot(velNT[0], directorNT[0]) * binWidth[j] / (2*Dr * dim * binArea[j])
                    active[j,1] += driving * np.dot(velNT[1], directorNT[1]) * binWidth[j] / (2*Dr * dim * binArea[j])
    for i in range(virial.shape[1]):
        virial[:,i] *= sigma/dirList.shape[0]
        active[:,i] *= sigma/dirList.shape[0]
    np.savetxt(dirName + os.sep + "surfaceTension.dat", np.column_stack((centers, virial, active)))
    print("surface tension: ", np.sum(centers/sigma * (virial[:,0] + active[:,0] - virial[:,1] - active[:,1])))
    uplot.plotCorrelation(centers, virial[:,0] - virial[:,1], "$\\Delta p$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(centers, active[:,0] - active[:,1], "$\\Delta p$", xlabel = "$Distance$", color=[1,0.5,0], lw=1.5)
    #plt.yscale('log')
    plt.show()

####################### Average linear pressure profile ########################
def averageLinearPressureProfile(dirName, shiftx=0, dirSpacing=1):
    dim = 2
    ec = 240
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 0.01)
    binArea = (bins[1] - bins[0])*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros(bins.shape[0]-1)
    virial = np.zeros((bins.shape[0]-1,3))
    active = np.zeros((bins.shape[0]-1,3))
    total = np.zeros(bins.shape[0]-1)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = utils.shiftPositions(pos, boxSize, shiftx, 0)
        #pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            work = 0
            workx = 0
            worky = 0
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    work += 0.5 * np.sum(force * delta)
                    # diagonal components of work tensor
                    workx += 0.5 * force[0] * delta[0]
                    worky += 0.5 * force[1] * delta[1]
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] < bins[j+1]):
                    thermal[j] += np.linalg.norm(vel[i])**2 / dim
                    virial[j,0] += work / dim
                    active[j,0] += driving * np.sum(vel[i] * director[i]) / (2*Dr * dim)
                    # normal and tangential components
                    virial[j,1] += workx
                    virial[j,2] += worky
                    active[j,1] += driving * vel[i,0] * director[i,0] / (2*Dr)
                    active[j,2] += driving * vel[i,1] * director[i,1] / (2*Dr)
                    total[j] += (work + np.linalg.norm(vel[i])**2 + driving * np.sum(vel[i] * director[i]) / (2*Dr)) / dim
    virial *= sigma**2/(binArea * dirList.shape[0])
    thermal *= sigma**2/(binArea * dirList.shape[0])
    active *= sigma**2/(binArea * dirList.shape[0])
    total *= sigma**2/(binArea * dirList.shape[0])
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((centers, virial, thermal, active, total)))
    print("average pressure: ", np.mean(total), "+-", np.std(total))
    uplot.plotCorrelation(centers, virial[:,0], "$Pressure$ $profile$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(centers, thermal, "$Pressure$ $profile$", xlabel = "$Distance$", color='r', lw=1.5)
    uplot.plotCorrelation(centers, active[:,0], "$Pressure$ $profile$", xlabel = "$Distance$", color=[1,0.5,0], lw=1.5)
    uplot.plotCorrelation(centers, total, "$Pressure$ $profile$", xlabel = "$Distance$", color='b', lw=1)
    #plt.yscale('log')
    plt.show()

############################ Linear surface tension ############################
def averageLinearTension(dirName, shiftx=0, dirSpacing=1):
    dim = 2
    ec = 240
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 0.01)
    binArea = (bins[1] - bins[0])*boxSize[1]
    binWidth = bins[1] - bins[0]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    virial = np.zeros((bins.shape[0]-1,2))
    active = np.zeros((bins.shape[0]-1,2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = utils.shiftPositions(pos, boxSize, shiftx, 0)
        #pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            workx = 0
            worky = 0
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    # diagonal components of work tensor
                    workx += 0.5 * force[0] * delta[0]
                    worky += 0.5 * force[1] * delta[1]
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] < bins[j+1]):
                    # normal and tangential components
                    virial[j,0] += workx
                    virial[j,1] += worky
                    active[j,0] += driving * vel[i,0] * director[i,0] / (2*Dr)
                    active[j,1] += driving * vel[i,1] * director[i,1] / (2*Dr)
    virial *= sigma**2/(binArea * dirList.shape[0])
    active *= sigma**2/(binArea * dirList.shape[0])
    np.savetxt(dirName + os.sep + "surfaceTension.dat", np.column_stack((centers, virial, active)))
    print("surface tension: ", np.sum(centers/sigma*(virial[:,0] + active[:,0] - virial[:,1] - active[:,1])))
    uplot.plotCorrelation(centers, virial[:,0] - virial[:,1], "$\\Delta p$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(centers, active[:,0] - active[:,1], "$\\Delta p$", xlabel = "$Distance$", color=[1,0.5,0], lw=1.5)
    #plt.yscale('log')
    plt.show()


########################## Total velocity components ###########################
def computeVelMagnitudeVSTime(dirName, plot=False, dirSpacing=1):
    dim = 2
    ec = 240
    gamma = float(utils.readFromDynParams(dirName, "damping"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    velMagnitude = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        steric = 0
        thermal = 0
        active = 0
        for i in range(numParticles):
            thermal += np.linalg.norm(vel[i])
            active += driving * np.sum(director[i] * vel[i]) / gamma
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = pos[i] - pos[c]
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    steric += 0.5 * np.linalg.norm(force) / gamma
        velMagnitude[d,0] = steric
        velMagnitude[d,1] = thermal
        velMagnitude[d,2] = active
        np.savetxt(dirName + os.sep + "velMagnitude.dat", np.column_stack((timeList, velMagnitude)))
    print("steric velocity: ", np.mean(velMagnitude[:,0]), " +/- ", np.std(velMagnitude[:,0]))
    print("thermal velocity: ", np.mean(velMagnitude[:,1]), " +/- ", np.std(velMagnitude[:,1]))
    print("active velocity: ", np.mean(velMagnitude[:,2]), " +/- ", np.std(velMagnitude[:,2]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, velMagnitude[:,0], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color='k')
        uplot.plotCorrelation(timeList, velMagnitude[:,1], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color='r')
        uplot.plotCorrelation(timeList, velMagnitude[:,2], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color=[1,0.5,0])
        plt.show()

########################## Cluster velocity components ###########################
def computeClusterVelMagnitudeVSTime(dirName, plot=False, dirSpacing=1):
    dim = 2
    ec = 240
    gamma = float(utils.readFromDynParams(dirName, "damping"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    velMagnitudeIn = np.zeros((dirList.shape[0],3))
    velMagnitudeOut = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "denseList.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
            voroDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            denseList, voroDensity = computeVoronoiCluster(dirSample)
        voroVolume = np.pi * rad**2 / voroDensity
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        volumeIn = 0
        stericIn = 0
        thermalIn = 0
        activeIn = 0
        volumeOut = 0
        stericOut = 0
        thermalOut = 0
        activeOut = 0
        for i in range(numParticles):
            if(denseList[i] == 1):
                volumeIn += voroVolume[i]
                thermalIn += np.linalg.norm(vel[i])
                activeIn += driving * np.sum(director[i] * vel[i]) / gamma
            else:
                volumeOut += voroVolume[i]
                thermalOut += np.linalg.norm(vel[i])
                activeOut += driving * np.sum(director[i] * vel[i]) / gamma
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = pos[i] - pos[c]
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    if(denseList[i] == 1):
                        stericIn += 0.5 * np.linalg.norm(force) / gamma
                    else:
                        stericOut += 0.5 * np.linalg.norm(force) / gamma
        if(volumeIn > 0):
            velMagnitudeIn[d,0] = stericIn / volumeIn
            velMagnitudeIn[d,1] = thermalIn / volumeIn
            velMagnitudeIn[d,2] = activeIn / volumeIn
        if(volumeOut > 0):
            velMagnitudeOut[d,0] = stericOut / volumeOut
            velMagnitudeOut[d,1] = thermalOut / volumeOut
            velMagnitudeOut[d,2] = activeOut / volumeOut
        np.savetxt(dirName + os.sep + "clusterVelMagnitude.dat", np.column_stack((timeList, velMagnitudeIn, velMagnitudeOut)))
    print("dense steric velocity: ", np.mean(velMagnitudeIn[:,0]), " +/- ", np.std(velMagnitudeIn[:,0]))
    print("dense thermal velocity: ", np.mean(velMagnitudeIn[:,1]), " +/- ", np.std(velMagnitudeIn[:,1]))
    print("dense active velocity: ", np.mean(velMagnitudeIn[:,2]), " +/- ", np.std(velMagnitudeIn[:,2]))
    print("\ndilute steric velocity: ", np.mean(velMagnitudeOut[:,0]), " +/- ", np.std(velMagnitudeOut[:,0]))
    print("dilute thermal velocity: ", np.mean(velMagnitudeOut[:,1]), " +/- ", np.std(velMagnitudeOut[:,1]))
    print("dilute active velocity: ", np.mean(velMagnitudeOut[:,2]), " +/- ", np.std(velMagnitudeOut[:,1]), "\n")
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, velMagnitudeIn[:,0], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color='k')
        uplot.plotCorrelation(timeList, velMagnitudeIn[:,1], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color='r')
        uplot.plotCorrelation(timeList, velMagnitudeIn[:,2], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color=[1,0.5,0])
        uplot.plotCorrelation(timeList, velMagnitudeOut[:,0], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color='k', ls='--')
        uplot.plotCorrelation(timeList, velMagnitudeOut[:,1], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color='r', ls='--')
        uplot.plotCorrelation(timeList, velMagnitudeOut[:,2], "$Steric,$ $thermal,$ $active$", "$Time,$ $t$", color=[1,0.5,0], ls='--')
        plt.show()

################################################################################
############################### Droplet pressure ###############################
################################################################################
def computeDropletParticleStress(dirName, l1 = 0.04):
    dim = 2
    ec = 240
    l2 = 0.2
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    stress = np.zeros((numParticles,3))
    #pos = np.loadtxt(dirName + "/particlePos.dat")
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    vel = np.loadtxt(dirName + "/particleVel.dat")
    neighbors = np.loadtxt(dirName + "/particleNeighbors.dat").astype(np.int64)
    for i in range(numParticles):
        virial = 0
        energy = 0
        for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
            radSum = rad[i] + rad[c]
            delta = utils.pbcDistance(pos[i], pos[c], boxSize)
            distance = np.linalg.norm(delta)
            overlap = 1 - distance / radSum
            if(distance < (1 + l1) * radSum):
                gradMultiple = ec * overlap / radSum
                energy += 0.5 * ec * (overlap**2 - l1 * l2) * 0.5
            elif((distance >= (1 + l1) * radSum) and (distance < (1 + l2) * radSum)):
                gradMultiple = (ec * l1 / (l2 - l1)) * (overlap + l2) / radSum
                energy += -(0.5 * (ec * l1 / (l2 - l1)) * (overlap + l2)**2) * 0.5
            else:
                gradMultiple = 0
            force = gradMultiple * delta / distance
            virial += 0.5 * np.sum(force * delta) # double counting
        stress[i,0] = virial
        stress[i,1] = np.linalg.norm(vel[i])**2
        stress[i,2] = energy
    np.savetxt(dirName + os.sep + "particleStress.dat", stress)
    print('potential energy: ', np.mean(stress[:,2]), ' +- ', np.std(stress[:,2]))
    print('kinetic energy: ', np.mean(stress[:,1]*0.5), ' +- ', np.std(stress[i,1]*0.5))
    print('energy: ', np.mean(stress[:,1]*0.5 + stress[:,2]), ' +- ', np.std(stress[i,1]*0.5 + stress[:,2]))
    return stress

def computeDropletParticleStressVSTime(dirName, dirSpacing=1):
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        computeDropletParticleStress(dirSample)

def computeDropletPressureVSTime(dirName, l1=0.1, dirSpacing=1):
    dim = 2
    ec = 240
    l2 = 0.2
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    pressureIn = np.zeros((dirList.shape[0],2))
    pressureOut = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "voroDensity.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
            voroDensity = np.loadtxt(dirSample + os.sep + "voroDensity.dat")
        else:
            denseList, voroDensity = computeVoronoiCluster(dirSample)
        voroVolume = np.pi*rad**2/voroDensity
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        volumeIn = 0
        virialIn = 0
        thermalIn = 0
        volumeOut = 0
        virialOut = 0
        thermalOut = 0
        for i in range(numParticles):
            if(denseList[i] == 1):
                volumeIn += voroVolume[i]
                thermalIn += np.linalg.norm(vel[i])**2
            else:
                volumeOut += voroVolume[i]
                thermalOut += np.linalg.norm(vel[i])**2
            # compute virial stress
            for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(distance < (1 + l1) * radSum):
                    gradMultiple = ec * overlap / radSum
                elif((distance >= (1 + l1) * radSum) and (distance < (1 + l2) * radSum)):
                    gradMultiple = (ec * l1 / (l2 - l1)) * (overlap + l2) / radSum
                else:
                    gradMultiple = 0
                force = gradMultiple * delta / distance
                if(denseList[i] == 1):
                    virialIn += 0.5 * np.sum(force * delta) # double counting
                else:
                    virialOut += 0.5 * np.sum(force * delta) # double counting
        if(volumeIn > 0):
            pressureIn[d,0] = virialIn / (dim * volumeIn) # double counting
            pressureIn[d,1] = thermalIn / volumeIn # dim k_B T / dim, dim cancels out
        if(volumeOut > 0):
            pressureOut[d,0] = virialOut / (dim * volumeOut) # double counting
            pressureOut[d,1] = thermalOut / volumeOut # dim k_B T / dim, dim cancels out
    pressureIn *= sigma**2
    pressureOut *= sigma**2
    np.savetxt(dirName + os.sep + "dropletPressure.dat", np.column_stack((timeList, pressureIn, pressureOut)))
    # pressure components in the fluid
    print("dense pressure: ", np.mean(pressureIn[:,0] + pressureIn[:,1]), " +/- ", np.std(pressureIn[:,0] + pressureIn[:,1]))
    print("dense virial pressure: ", np.mean(pressureIn[:,0]), " +/- ", np.std(pressureIn[:,0]))
    print("dense thermal pressure: ", np.mean(pressureIn[:,1]), " +/- ", np.std(pressureIn[:,1]))
    # pressure components in the gas
    print("\ndilute pressure: ", np.mean(pressureOut[:,0] + pressureOut[:,1]), " +/- ", np.std(pressureOut[:,0] + pressureOut[:,1]))
    print("dilute virial pressure: ", np.mean(pressureOut[:,0]), " +/- ", np.std(pressureOut[:,0]))
    print("dilute thermal pressure: ", np.mean(pressureOut[:,1]), " +/- ", np.std(pressureOut[:,1]), "\n")

############################ Radial droplet tension ############################
def averageDropletRadialTension(dirName, shiftx, shifty, l1=0.04, dirSpacing=1):
    dim = 2
    ec = 240
    l2 = 0.2
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0]/2, 2*np.mean(rad))
    binArea = np.pi * (bins[1:]**2 - bins[:-1]**2)
    binWidth = bins[1:] - bins[:-1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((bins.shape[0]-1,2))
    virial = np.zeros((bins.shape[0]-1,2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "denseList.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
        else:
            denseList,_ = computeVoronoiCluster(dirSample)
        # CENTER CLUSTER BY BRUTE FORCE - NEED TO IMPLEMENT SOMETHING BETTER
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if(shiftx != 0 or shiftx != 0):
            pos = utils.shiftPositions(pos, boxSize, -shiftx, -shifty)
        #pos = utils.centerPositions(pos, rad, boxSize, denseList)
        centerOfMass = np.mean(pos[denseList==1], axis=0)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        workn = 0
        workt = 0
        for i in range(numParticles):
            # Convert pressure components into normal and tangential components
            relPos = utils.pbcDistance(pos[i], centerOfMass, boxSize)
            relAngle = np.arctan2(relPos[1], relPos[0])
            unitVector = np.array([np.cos(relAngle), np.sin(relAngle)])
            for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(distance < (1 + l1) * radSum):
                    gradMultiple = ec * overlap / radSum
                elif((distance >= (1 + l1) * radSum) and (distance < (1 + l2) * radSum)):
                    gradMultiple = (ec * l1 / (l2 - l1)) * (overlap + l2) / radSum
                else:
                    gradMultiple = 0
                force = gradMultiple * delta / distance
                # normal minus tangential component
                forceNT = utils.projectToNormalTangentialComp(force, unitVector)
                deltaNT = utils.projectToNormalTangentialComp(delta, unitVector)
                workn += np.dot(forceNT[0], deltaNT[0])
                workt += np.dot(forceNT[1], deltaNT[1])
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] < bins[j+1]):
                    thermal[j,0] += vel[i,0]**2 * binWidth[j] / (dim * binArea[j])
                    thermal[j,1] += vel[i,1]**2 * binWidth[j] / (dim * binArea[j])
                    virial[j,0] = workn * binWidth[j] / (dim * binArea[j])
                    virial[j,1] = workt * binWidth[j] / (dim * binArea[j])
    for i in range(virial.shape[1]):
        virial[:,i] *= sigma/dirList.shape[0]
        thermal[:,i] *= sigma/dirList.shape[0]
    np.savetxt(dirName + os.sep + "surfaceTension.dat", np.column_stack((centers, virial, thermal)))
    print("surface tension: ", np.sum(centers * (virial[:,0] + thermal[:,0] - virial[:,1] - thermal[:,1])))
    uplot.plotCorrelation(centers, virial[:,0] - virial[:,1], "$\\Delta p$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(centers, thermal[:,0] - thermal[:,1], "$\\Delta p$", xlabel = "$Distance$", color='r', lw=1.5)
    #plt.yscale('log')
    plt.show()

####################### Average linear pressure profile ########################
def averageDropletLinearPressureProfile(dirName, l1 = 0.04, dirSpacing=1):
    dim = 2
    ec = 240
    l2 = 0.2
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 0.01)
    binArea = (bins[1] - bins[0])*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    virial = np.zeros((bins.shape[0]-1,3))
    thermal = np.zeros((bins.shape[0]-1,3))
    total = np.zeros(bins.shape[0]-1)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        for i in range(numParticles):
            work = 0
            workx = 0
            worky = 0
            for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(distance < (1 + l1) * radSum):
                    gradMultiple = ec * overlap / radSum
                elif((distance >= (1 + l1) * radSum) and (distance < (1 + l2) * radSum)):
                    gradMultiple = (ec * l1 / (l2 - l1)) * (overlap + l2) / radSum
                else:
                    gradMultiple = 0
                force = gradMultiple * delta / distance
                work += 0.5 * np.sum(force * delta)
                workx += 0.5 * force[0] * delta[0]
                worky += 0.5 * force[1] * delta[1]
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] < bins[j+1]):
                    virial[j,0] += work / dim
                    virial[j,1] += workx
                    virial[j,2] += worky
                    thermal[j,0] += np.linalg.norm(vel[i])**2 / dim
                    thermal[j,1] += vel[i,0]**2
                    thermal[j,2] += vel[i,1]**2
                    total[j] += (work + np.linalg.norm(vel[i])**2) / dim
    for i in range(virial.shape[1]):
        virial[:,i] *= sigma**2/(binArea * dirList.shape[0])
        thermal[:,i] *= sigma**2/(binArea * dirList.shape[0])
    total *= sigma**2/(binArea * dirList.shape[0])
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((centers, virial, thermal, total)))
    print("average pressure: ", np.mean(total), "+-", np.std(total))
    uplot.plotCorrelation(centers, virial[:,0], "$Pressure$ $profile$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(centers, thermal, "$Pressure$ $profile$", xlabel = "$Distance$", color='r', lw=1.5)
    uplot.plotCorrelation(centers, total, "$Pressure$ $profile$", xlabel = "$Distance$", color='b', lw=1)
    #plt.yscale('log')
    plt.show()

############################ Linear surface tension ############################
def averageDropletLinearTension(dirName, l1=0.04, dirSpacing=1):
    dim = 2
    ec = 240
    l2 = 0.2
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 0.01)
    binArea = (bins[1] - bins[0])*boxSize[1]
    binWidth = bins[1] - bins[0]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    virial = np.zeros((bins.shape[0]-1,2))
    thermal = np.zeros((bins.shape[0]-1,2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        #pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        workx = 0
        worky = 0
        for i in range(numParticles):
            for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(distance < (1 + l1) * radSum):
                    gradMultiple = ec * overlap / radSum
                elif((distance >= (1 + l1) * radSum) and (distance < (1 + l2) * radSum)):
                    gradMultiple = (ec * l1 / (l2 - l1)) * (overlap + l2) / radSum
                else:
                    gradMultiple = 0
                force = gradMultiple * delta / distance
                # normal minus tangential component
                workx += 0.5 * (force[0] * delta[0])
                worky += 0.5 * (force[1] * delta[1])
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] < bins[j+1]):
                    thermal[j,0] += vel[i,0]**2
                    thermal[j,1] += vel[i,1]**2
                    virial[j,0] += workx
                    virial[j,1] += worky
    thermal *= sigma*binWidth/(binArea * dirList.shape[0])
    virial *= sigma*binWidth/(binArea * dirList.shape[0])
    np.savetxt(dirName + os.sep + "surfaceTension.dat", np.column_stack((centers, virial, thermal)))
    print("surface tension: ", np.sum(centers * (virial[:,0] + thermal[:,0] - virial[:,1] - thermal[:,1])))
    uplot.plotCorrelation(centers, virial[:,0] - virial[:,1], "$\\Delta p$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(centers, thermal[:,0] - thermal[:,1], "$\\Delta p$", xlabel = "$Distance$", color='r', lw=1.5)
    #plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    if(whichCorr == "density"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        computeLocalDensity(dirName, numBins, plot)

    elif(whichCorr == "averageld"):
        numBins = int(sys.argv[3])
        weight = sys.argv[4]
        plot = sys.argv[5]
        averageLocalDensity(dirName, numBins, weight, plot)

    elif(whichCorr == "nphitime"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        computeLocalDensityAndNumberVSTime(dirName, numBins, plot)

    elif(whichCorr == "pccluster"):
        dirSpacing = int(sys.argv[3])
        averagePairCorrCluster(dirName, dirSpacing)

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

    elif(whichCorr == "phinum"):
        plot = sys.argv[3]
        computeLocalDensityAndNumberFluctuations(dirName, plot)

    elif(whichCorr == "averagephinum"):
        plot = sys.argv[3]
        averageLocalDensityAndNumberFluctuations(dirName, plot)

    elif(whichCorr == "mixing"):
        plot = sys.argv[3]
        computeClusterMixingTime(dirName, plot)

    elif(whichCorr == "bmixing"):
        numBlocks = int(sys.argv[3])
        plot = sys.argv[4]
        computeClusterBlockMixingTime(dirName, numBlocks, plot)

    elif(whichCorr == "lmixing"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeClusterLogMixingTime(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "bvapor"):
        numBlocks = int(sys.argv[3])
        plot = sys.argv[4]
        computeClusterBlockEvaporationTime(dirName, numBlocks, plot)

    elif(whichCorr == "velpdfcluster"):
        plot = sys.argv[3]
        averageParticleVelPDFCluster(dirName, plot)

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

    elif(whichCorr == "voroshape"):
        plot = sys.argv[3]
        computeClusterVoronoiShape(dirName, plot)

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

    elif(whichCorr == "clusterdptime"):
        computeDelaunayClusterPressureVSTime(dirName)

############################### Cluster pressure ###############################
    elif(whichCorr == "stress"):
        prop = sys.argv[3]
        computeParticleStress(dirName, prop)

    elif(whichCorr == "stresstime"):
        computeParticleStressVSTime(dirName)

    elif(whichCorr == "surfacework"):
        plot = sys.argv[3]
        averageRadialSurfaceWork(dirName, plot)

    elif(whichCorr == "radialprofile"):
        shiftx = float(sys.argv[3])
        shifty = float(sys.argv[4])
        averageRadialPressureProfile(dirName, shiftx, shifty)

    elif(whichCorr == "radialtension"):
        shiftx = float(sys.argv[3])
        shifty = float(sys.argv[4])
        averageRadialTension(dirName, shiftx, shifty)

    elif(whichCorr == "linearprofile"):
        shiftx = float(sys.argv[3])
        averageLinearPressureProfile(dirName, shiftx)

    elif(whichCorr == "lineartension"):
        shiftx = float(sys.argv[3])
        averageLinearTension(dirName, shiftx)

    elif(whichCorr == "ptime"):
        bound = sys.argv[3]
        prop = sys.argv[4]
        computePressureVSTime(dirName, bound, prop)

    elif(whichCorr == "clusterptime"):
        bound = sys.argv[3]
        prop = sys.argv[4]
        computeClusterPressureVSTime(dirName, bound, prop)

    elif(whichCorr == "veltime"):
        plot = sys.argv[3]
        computeVelMagnitudeVSTime(dirName, plot)

    elif(whichCorr == "clusterveltime"):
        plot = sys.argv[3]
        computeClusterVelMagnitudeVSTime(dirName, plot)

############################### Droplet pressure ###############################
    elif(whichCorr == "dropstress"):
        l1 = float(sys.argv[3])
        computeDropletParticlePressureVSTime(dirName, l1)

    elif(whichCorr == "dropptime"):
        l1 = float(sys.argv[3])
        computeDropletPressureVSTime(dirName, l1)

    elif(whichCorr == "droptension"):
        shiftx = float(sys.argv[3])
        shifty = float(sys.argv[4])
        l1 = float(sys.argv[5])
        averageDropletRadialTension(dirName, shiftx, shifty, l1)

    elif(whichCorr == "slabprofile"):
        l1 = float(sys.argv[3])
        averageDropletLinearPressureProfile(dirName, l1)

    elif(whichCorr == "slabtension"):
        l1 = float(sys.argv[3])
        averageDropletLinearTension(dirName, l1)

    else:
        print("Please specify the correlation you want to compute")
