'''
Created by Francesco
14 July 2023
'''
#functions and script to compute cluster correlations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.fftpack import rfft, rfftfreq, fft, fftfreq
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
        uplot.plotCorrelation(meanNum, deltaPhi, "$Variance$ $of$ $local$ $number,$ $\\Delta N^2$", "$Local$ $number,$ $N_s$", color=color, logx=True)
        plt.pause(0.5)
        #plt.show()
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
        uplot.plotCorrWithError(meanNum, deltaPhi, stdDeltaPhi, "$Variance$ $of$ $local$ $number,$ $\\Delta N^2$", "$Local$ $number,$ $N_s$", color='k', logx=True)
        plt.pause(0.5)

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
            denseList0 = particleList[:,0]
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
            denseList0 = particleList[:,0]
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
def computeClusterRate(dirName, numBlocks, blockPower, blockFreq=1e04, spacing='log', threshold=0.76, plot=False, dirSpacing=1):
    numBins = 20
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + "/particleRad.dat"), dtype=np.float64)
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
    blockTimeList = timeList[:blockSize]# first get range of cluster size distribution
    if not(os.path.exists(dirName + os.sep + "/clusterDistribution.dat")):
        computeDelaunayClusterDistribution(dirName)
    clusterArea = np.loadtxt(dirName + os.sep + "/clusterDistribution.dat")[:,1]
    pdf, edges = np.histogram(clusterArea, bins=np.geomspace(np.min(clusterArea), np.max(clusterArea), numBins+1), density=True)
    numCondensed = np.zeros(blockSize)
    numEvaporated = np.zeros(blockSize)
    condensationTime = np.empty(0)
    evaporationTime = np.empty(0)
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
                                numCondensed[d] += 1
                                condensationTime = np.append(condensationTime, blockTimeList[d] - time0[i])
                            sizeLabels0[i] = sizeLabels[i]
                            time0[i] = blockTimeList[d]
                        else:
                            if(time0[i] != blockTimeList[0]):
                                numEvaporated[d] += 1
                                evaporationTime = np.append(evaporationTime, blockTimeList[d] - time0[i])
                            sizeLabels0[i] = sizeLabels[i]
                            time0[i] = blockTimeList[d]
    numCondensed /= numParticles
    numEvaporated /= numParticles
    # average correlations over particles
    np.savetxt(dirName + os.sep + "clusterRate.dat", np.column_stack((blockTimeList, numCondensed, numEvaporated)))
    print("Orange, condensation time:", np.mean(condensationTime), "+-", np.std(condensationTime))
    print("Blue, evaporation time:", np.mean(evaporationTime), "+-", np.std(evaporationTime))
    if(plot=='plot'):
        # rate plots
        uplot.plotCorrelation(blockTimeList[1:], numCondensed[1:]/blockTimeList[1:], ylabel="$Cluster$ $rate$", xlabel="$Time$", logx = True, color = [1,0.5,0], marker='o')
        uplot.plotCorrelation(blockTimeList[1:], numEvaporated[1:]/blockTimeList[1:], ylabel="$Cluster$ $rate$", xlabel="$Time$", logx = True,color = 'b', marker='s')
        # time plots
        #uplot.plotCorrelation(np.arange(1, condensationTime.shape[0]+1, 1), np.sort(condensationTime), "$Condensation$ $time$", xlabel = "$index$", color='k', logy=True)
        #uplot.plotCorrelation(np.arange(1, evaporationTime.shape[0]+1, 1), np.sort(evaporationTime), "$Evaporation$ $time$", xlabel = "$index$", color='k', logy=True)
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

def computeClusterTemperatureVSTime(dirName, threshold=0.76, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    temp = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + "/particleList.dat")):
            computeDelaunayCluster(dirSample, threshold=threshold)
        particleList = np.loadtxt(dirSample + "/particleList.dat")
        denseList = particleList[:,0]
        vel = np.loadtxt(dirSample + os.sep + "particleVel.dat")
        velNorm = np.linalg.norm(vel, axis=1)
        temp[d,0] = np.mean(velNorm**2)
        temp[d,1] = np.mean(velNorm[denseList==1]**2)
        temp[d,2] = np.mean(velNorm[denseList==0]**2)
    print("temperature:", np.mean(temp[:,0]))
    print("liquid:", np.mean(temp[:,1]))
    print("vapor:", np.mean(temp[:,2]))
    np.savetxt(dirName + os.sep + "clusterTemperature.dat", temp)
    return temp

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
    skewness = np.mean((velInCluster - mean)**3)/Temp**(3/2)
    kurtosis = np.mean((velInCluster - mean)**4)/Temp**2
    data = velInCluster# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity inside the cluster: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='b')
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
        uplot.plotCorrelation(edges, pdf, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='g')
    np.savetxt(dirName + os.sep + "velPDFOutCluster.dat", np.column_stack((edges, pdf)))
    # total
    velTotal = velTotal[velTotal>0]
    mean = np.mean(velTotal)
    Temp = np.var(velTotal)
    skewness = np.mean((velTotal - mean)**3)/Temp**(3/2)
    kurtosis = np.mean((velTotal - mean)**4)/Temp**2
    data = velTotal# / np.sqrt(2*Temp)
    pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity in the whole system: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='k')
    np.savetxt(dirName + os.sep + "velPDF.dat", np.column_stack((edges, pdf)))
    if(plot == "plot"):
        plt.pause(0.5)
        #plt.show()
    return temp, tempIn, tempOut

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
        if(denseSimplexList[i] == 1 and insideIndex[i] == 1):
            indices = utils.findNeighborSimplices(simplices, i)
            if(np.sum(denseSimplexList[indices]) <= 2):
                borderSimplexList[i] = 1
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
    particleList = np.column_stack((denseList, borderList))
    simplexList = np.column_stack((denseSimplexList, borderSimplexList, simplexArea, simplexDensity))
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
    if not(os.path.exists(dirSample + os.sep + "particleList.dat")):
        computeDelaunayCluster(dirSample)
    particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
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

def getParticleClusterLabels(dirSample, boxSize, eps, threshold=0.76, compute=False, save='save'):
    if(compute==True):
        computeDelaunayCluster(dirSample, threshold=threshold, save=save)
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        denseList = particleList[:,0]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = utils.getDBClusterLabels(pos, boxSize, eps, min_samples=2, denseList=denseList)
        allLabels = -1*np.ones(denseList.shape[0], dtype=np.int64)
        allLabels[denseList==1] = labels
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
            allLabels = -1*np.ones(denseList.shape[0], dtype=np.int64)
            allLabels[denseList==1] = labels
            np.savetxt(dirSample + os.sep + "clusterLabels.dat", allLabels)
        else:
            allLabels = np.loadtxt(dirSample + os.sep + "clusterLabels.dat")
    return allLabels

def getParticleDenseLabel(dirSample, boxSize, eps, threshold=0.76, compute=False, save='save'):
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
        else:
            allLabels = np.loadtxt(dirSample + os.sep + "clusterLabels.dat")
    return allLabels

def getLEParticleDenseLabel(dirSample, boxSize, eps, threshold=0.76, compute=False, save='save', strain=0):
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
        else:
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

######################### Simplex cluster distribution #########################
def computeDropletSizeVSTime(dirName, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    eps = np.max(rad) # different from eps for particles
    boxSize = np.loadtxt(dirName + "/boxSize.dat")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # first define three droplets to follow the dynamics of
    dirSample = dirName + os.sep + dirList[0]
    pos0 = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
    labels0 = np.loadtxt(dirSample + "/clusterLabels.dat").astype(np.int64)
    dropletPos0, dropletRad0 = utils.getDropletPosRad(pos0, rad, boxSize, labels0)
    # plot the cluster number and area for three droplets
    thisLabels = np.argsort(dropletRad0)[-3:]
    clusterNumber = np.zeros((dirList.shape[0],3))
    clusterArea = np.zeros((dirList.shape[0],3))
    print("Droplet labels:", thisLabels)
    labelIndex = 0
    for label in thisLabels:
        clusterNumber[0,labelIndex] = labels0[labels0==label].shape[0]
        clusterArea[0, labelIndex] = np.sum(rad[labels0==label]**2)*np.pi
        labelIndex += 1
    for d in range(1,dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = np.loadtxt(dirSample + "/clusterLabels.dat").astype(np.int64)
        dropletPos, dropletRad = utils.getDropletPosRad(pos, rad, boxSize, labels)
        # find the droplets
        for j in range(thisLabels.shape[0]):
            found = False
            for label in range(dropletPos.shape[0]):
                if(found == False and np.linalg.norm(utils.pbcDistance(dropletPos[label], dropletPos0[thisLabels[j]], boxSize)) < dropletRad0[thisLabels[j]]):
                    clusterNumber[d, j] = labels[labels==label].shape[0]
                    clusterArea[d, j] = np.sum(rad[labels==label]**2)*np.pi
                    dropletPos0[thisLabels[j]] = dropletPos[label]
                    dropletRad0[thisLabels[j]] = dropletRad[label]
                    found = True
    np.savetxt(dirName + os.sep + "dropletSizeVSTime.dat", np.column_stack((timeList, clusterArea, clusterNumber)))
    if(plot == 'plot'):
        #uplot.plotCorrelation(timeList, clusterArea[:,0], "$A_c$", xlabel = "$Time,$ $t$", color='b')
        #uplot.plotCorrelation(timeList, clusterArea[:,1], "$A_c$", xlabel = "$Time,$ $t$", color='g')
        #uplot.plotCorrelation(timeList, clusterArea[:,2], "$A_c$", xlabel = "$Time,$ $t$", color='c')
        uplot.plotCorrelation(timeList, clusterNumber[:,0], "$N_c$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, clusterNumber[:,1], "$N_c$", xlabel = "$Time,$ $t$", color='g')
        uplot.plotCorrelation(timeList, clusterNumber[:,2], "$N_c$", xlabel = "$Time,$ $t$", color='c')
        plt.show()
        #plt.pause(0.5)


################################################################################
############################### Cluster pressure ###############################
################################################################################
def computeParticleStress(dirName, nDim=2):
    ec = 240
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
    Dr = float(utils.readFromDynParams(dirName + sep, "Dr"))
    gamma = float(utils.readFromDynParams(dirName + sep, "damping"))
    driving = float(utils.readFromDynParams(dirName + sep, "f0"))
    stress = np.zeros((numParticles,5))
    #pos = np.loadtxt(dirName + "/particlePos.dat")
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    vel = np.loadtxt(dirName + "/particleVel.dat")
    angle = utils.getMOD2PIAngles(dirName + "/particleAngles.dat")
    director = np.array([np.cos(angle), np.sin(angle)]).T
    contacts = np.loadtxt(dirName + "/particleContacts.dat").astype(np.int64)
    for i in range(numParticles):
        virial = 0
        shear = 0
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
                shear -= 0.5 * (force[0] * delta[1] + force[1] * delta[0])
                #active += np.sum(force * director[i]) / gamma
                energy += 0.5 * ec * overlap**2 * 0.5 # double counting and e = k/2
        stress[i,0] = virial*sigma**2/ec
        stress[i,1] = np.linalg.norm(vel[i])**2 / (nDim*sigma**2/ec)
        stress[i,2] = driving * np.sum(vel[i] * director[i]) / (2*Dr*sigma**2/ec)
        stress[i,3] = energy + 0.5 * np.linalg.norm(vel[i])**2/ec
        stress[i,4] = shear*sigma**2/ec
    np.savetxt(dirName + os.sep + "particleStress.dat", stress)
    print('virial: ', np.mean(stress[:,0]), ' +- ', np.std(stress[:,0]))
    print('shear: ', np.mean(stress[:,3]), ' +- ', np.std(stress[:,4]))
    print('thermal: ', np.mean(stress[:,1]), ' +- ', np.std(stress[:,1]))
    print('active: ', np.mean(stress[:,2]), ' +- ', np.std(stress[:,2]))
    return stress

########################## Total pressure components ###########################
def computePressureVSTime(dirName, dirSpacing=1, nDim=2):
    ec = 240
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    boxArea = boxSize[0]*boxSize[1]
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
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
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
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
                        wallPressure[d] -= np.sum(wallForce * pos[i]) / nDim
                    else:
                        wallPressure[d] += np.linalg.norm(wallForce) / boxLength
            # particle pressure components
            pressure[d,0] += np.linalg.norm(vel[i])**2
            pressure[d,2] += driving * np.sum(vel[i] * director[i]) / (2*Dr*nDim)
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    pressure[d,1] += 0.5 * np.sum(force * delta) / nDim # double counting
    pressure /= (ec / (boxArea * sigma**2))
    wallPressure /= (ec / (boxArea * sigma**2))
    np.savetxt(dirName + os.sep + "pressure.dat", np.column_stack((timeList, pressure, wallPressure)))
    print("bulk pressure: ", np.mean(np.sum(pressure, axis=1)), " +/- ", np.std(np.sum(pressure, axis=1)))
    print("thermal pressure: ", np.mean(pressure[:,0]), " +/- ", np.std(pressure[:,0]))
    print("virial pressure: ", np.mean(pressure[:,1]), " +/- ", np.std(pressure[:,1]))
    print("active pressure: ", np.mean(pressure[:,2]), " +/- ", np.std(pressure[:,2]))
    print("pressure on the wall: ", np.mean(wallPressure), " +/- ", np.std(wallPressure))

############################ Total stress components ###########################
def computeStressVSTime(dirName, strain=0, dirSpacing=1, nDim=2):
    ec = 240
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    boxArea = boxSize[0]*boxSize[1]
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    stress = np.zeros((dirList.shape[0],9))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(strain > 0):
            pos = utils.getLEPBCPositions(dirSample + "/particlePos.dat", boxSize, strain)
        else:
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            # particle pressure components
            stress[d,3] += vel[i,0]**2
            stress[d,4] += vel[i,1]**2
            stress[d,5] -= (vel[i,0] * vel[i,1])
            stress[d,6] += driving * vel[i,0] * director[i,0] / (2*Dr)
            stress[d,7] += driving * vel[i,0] * director[i,1] / (2*Dr)
            stress[d,8] += driving * (vel[i,0] * director[i,1] + vel[i,1] * director[i,0]) / (4*Dr)
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = 0.5 * gradMultiple * delta / distance # double counting
                    stress[d,0] += force[0] * delta[0]
                    stress[d,1] += force[1] * delta[1]
                    stress[d,2] -= (force[0] * delta[1] + force[1] * delta[0]) * 0.5
        #print(dirList[d], stress[d,0], stress[d,1])
    stress /= (ec / (boxArea * sigma**2))
    np.savetxt(dirName + os.sep + "timeStress.dat", np.column_stack((timeList, stress)))
    bulk = stress[:,0]+stress[:,1]+stress[:,3]+stress[:,4]+stress[:,6]+stress[:,7]
    print("bulk stress: ", np.mean(bulk), " +/- ", np.std(bulk))
    shear = stress[:,2]+stress[:,5]+stress[:,8]
    print("shear stress: ", np.mean(shear), " +/- ", np.std(shear))
    uplot.plotCorrelation(timeList, bulk, color='k', ylabel='$\\sigma$', xlabel='$\\gamma$', logx=True)
    uplot.plotCorrelation(timeList, shear, color='k', ylabel='$\\sigma$', xlabel='$\\gamma$', logx=True)
    plt.pause(0.5)

############################ Total stress components ###########################
def computeStressVSStrain(dirName, strain=0, dirSpacing=1, nDim=2):
    ec = 240
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    sep = utils.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    boxArea = boxSize[0]*boxSize[1]
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
    dirList, strainList = utils.getShearDirectories(dirName)
    stress = np.zeros((dirList.shape[0],9))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(strain > 0):
            pos = utils.getLEPBCPositions(dirSample + "/particlePos.dat", boxSize, strain)
        else:
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            # particle pressure components
            stress[d,3] += vel[i,0]**2
            stress[d,4] += vel[i,1]**2
            stress[d,5] -= (vel[i,0] * vel[i,1])
            stress[d,6] += driving * vel[i,0] * director[i,0] / (2*Dr)
            stress[d,7] += driving * vel[i,0] * director[i,1] / (2*Dr)
            stress[d,8] -= driving * (vel[i,0] * director[i,1] + vel[i,1] * director[i,0]) / (4*Dr)
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = 0.5 * gradMultiple * delta / distance # double counting
                    stress[d,0] += force[0] * delta[0]
                    stress[d,1] += force[1] * delta[1]
                    stress[d,2] -= (force[0] * delta[1] + force[1] * delta[0]) * 0.5
    stress /= (ec / (boxArea * sigma**2))
    np.savetxt(dirName + os.sep + "strainStress.dat", np.column_stack((strainList, stress)))
    bulk = stress[:,0]+stress[:,1]+stress[:,3]+stress[:,4]+stress[:,6]+stress[:,7]
    print("bulk stress: ", np.mean(bulk), " +/- ", np.std(bulk))
    shear = stress[:,2]+stress[:,5]+stress[:,8]
    print("shear stress: ", np.mean(shear), " +/- ", np.std(shear))
    uplot.plotCorrelation(timeList, bulk, color='k', ylabel='$\\sigma$', xlabel='$\\gamma$')
    uplot.plotCorrelation(timeList, shear, color='k', ylabel='$\\sigma$', xlabel='$\\gamma$')
    plt.pause(0.5)

######################### Cluster pressure components ##########################
def computeClusterPressureVSTime(dirName, dirSpacing=1, nDim=2):
    ec = 240
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    pressureIn = np.zeros((dirList.shape[0],3))
    pressureOut = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample, threshold=0.76, save='save')
        simplices = np.loadtxt(dirSample + os.sep + "simplices.dat").astype(np.int64)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        borderSimplexList = simplexList[:,1]
        simplexArea = simplexList[:,2]
        numSharedSimplices = np.zeros(numParticles)
        for p in range(numParticles):
            numSharedSimplices[p] = np.argwhere(simplices==p)[:,0].shape[0]
        pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        angle = np.mod(angle, 2*np.pi)
        director = np.array([np.cos(angle), np.sin(angle)]).T
        # particle pressure components
        areaIn = np.sum(simplexArea[denseSimplexList==1])
        areaOut = np.sum(simplexArea[denseSimplexList==0])
        for i in range(simplices.shape[0]):
            if(denseSimplexList[i] == 1):
                for p in simplices[i]:
                    pressureIn[d,0] += np.linalg.norm(vel[p])**2
                    pressureIn[d,2] += driving * np.sum(vel[p] * director[p]) / (2*Dr*nDim)
            else:
                for p in simplices[i]:
                    pressureOut[d,0] += np.linalg.norm(vel[p])**2
                    pressureOut[d,2] += driving * np.sum(vel[p] * director[p]) / (2*Dr*nDim)
            simplexNeighbor = utils.findNeighborSimplices(simplices, i)
            for neighbor in simplexNeighbor:
                edge = np.intersect1d(simplices[i], simplices[neighbor])
                radSum = rad[edge[0]] + rad[edge[1]]
                delta = utils.pbcDistance(pos[edge[0]], pos[edge[1]], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = 0.5 * gradMultiple * delta / distance
            if(denseSimplexList[i] == 1):
                pressureIn[d,1] += np.sum(force * delta) / nDim
            else:
                pressureOut[d,1] += np.sum(force * delta) / nDim
        if(areaIn > 0):
            pressureIn *= sigma**2 / areaIn
        if(areaOut > 0):
            pressureOut *= sigma**2 / areaOut
    np.savetxt(dirName + os.sep + "clusterPressure.dat", np.column_stack((timeList, pressureIn, pressureOut)))
    # pressure components in the fluid
    print("dense pressure: ", np.mean(np.sum(pressureIn,axis=1)), " +/- ", np.std(np.sum(pressureIn,axis=1)))
    print("dense thermal pressure: ", np.mean(pressureIn[:,0]), " +/- ", np.std(pressureIn[:,0]))
    print("dense virial pressure: ", np.mean(pressureIn[:,1]), " +/- ", np.std(pressureIn[:,1]))
    print("dense active pressure: ", np.mean(pressureIn[:,2]), " +/- ", np.std(pressureIn[:,2]))
    # pressure components in the gas
    print("\ndilute pressure: ", np.mean(np.sum(pressureOut,axis=1)), " +/- ", np.std(np.sum(pressureOut,axis=1)))
    print("dilute thermal pressure: ", np.mean(pressureOut[:,0]), " +/- ", np.std(pressureOut[:,0]))
    print("dilute virial pressure: ", np.mean(pressureOut[:,1]), " +/- ", np.std(pressureOut[:,1]))
    print("dilute active pressure: ", np.mean(pressureOut[:,2]), " +/- ", np.std(pressureOut[:,2]), "\n")

######################### Simplex pressure components ##########################
def computeSimplexPressureVSTime(dirName, dirSpacing=1, nDim=2):
    ec = 240
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    borderPressure = np.zeros((dirList.shape[0],3))
    fluidPressure = np.zeros((dirList.shape[0],3))
    gasPressure = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        angle = np.mod(angle, 2*np.pi)
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        # load simplices
        if not(os.path.exists(dirSample + os.sep + "simplices.dat")):
            computeDelaunayCluster(dirSample, threshold=0.76, save='save')
        simplices = np.loadtxt(dirSample + os.sep + 'simplices.dat').astype(np.int64)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        borderSimplexList = simplexList[:,1]
        simplexArea = simplexList[:,2]
        numSharedSimplices = np.zeros(numParticles)
        for p in range(numParticles):
            numSharedSimplices[p] = np.argwhere(simplices==p)[:,0].shape[0]
        # first fill out the area occupied by fluid and the gas
        borderArea = 0
        fluidArea = 0
        gasArea = 0
        for i in range(simplexArea.shape[0]):
            if(borderSimplexList[i] == 1):
                borderArea += simplexArea[i]
            else:
                if(denseSimplexList[i]==1):
                    fluidArea += simplexArea[i]
                else:
                    gasArea += simplexArea[i]
        # compute stress components
        for i in range(simplexArea.shape[0]):
            if(borderSimplexList[i]==1):
                for p in simplices[i]:
                    borderPressure[d,0] += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
                    borderPressure[d,2] += driving * np.sum(vel[p] * director[p]) / (2*Dr*numSharedSimplices[p])
            else:
                if(denseSimplexList[i]==1):
                    for p in simplices[i]:
                        fluidPressure[d,0] += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
                        fluidPressure[d,2] += driving * np.sum(vel[p] * director[p]) / (2*Dr*numSharedSimplices[p])
                else:
                    for p in simplices[i]:
                        gasPressure[d,0] += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
                        gasPressure[d,2] += driving * np.sum(vel[p] * director[p]) / (2*Dr*numSharedSimplices[p])
            simplexNeighbor = utils.findNeighborSimplices(simplices, i)
            for neighbor in simplexNeighbor:
                edge = np.intersect1d(simplices[i], simplices[neighbor])
                radSum = rad[edge[0]] + rad[edge[1]]
                delta = utils.pbcDistance(pos[edge[0]], pos[edge[1]], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = 0.5 * gradMultiple * delta / distance
                    if(borderSimplexList[i]==1):
                        borderPressure[d,1] += np.sum(force * delta)
                    else:
                        if(denseSimplexList[i]==1):
                            fluidPressure[d,1] += np.sum(force * delta)
                        else:
                            gasPressure[d,1] += np.sum(force * delta)
        if(borderArea > 0):
            borderPressure[d,0] /= (nDim * borderArea)
            borderPressure[d,1] /= borderArea # dim k_B T / dim, dim cancels out
            borderPressure[d,2] /= (nDim * borderArea)
        if(fluidArea > 0):
            fluidPressure[d,0] /= (dim * fluidArea)
            fluidPressure[d,1] /= fluidArea # dim k_B T / dim, dim cancels out
            fluidPressure[d,2] /= (dim * fluidArea)
        if(gasArea > 0):
            gasPressure[d,0] /= (dim * gasArea)
            gasPressure[d,1] /= gasArea # dim k_B T / dim, dim cancels out
            gasPressure[d,2] /= (dim * gasArea)
        #print(d, np.sum(fluidPressure[d])*sigma**2, np.sum(gasPressure[d])*sigma**2, np.sum(borderPressure[d])*sigma**2)
    borderPressure *= sigma**2
    fluidPressure *= sigma**2
    gasPressure *= sigma**2
    np.savetxt(dirName + os.sep + "simplexPressure.dat", np.column_stack((timeList, fluidPressure, gasPressure, borderPressure)))
    # pressure components in the fluid
    print("dense pressure: ", np.mean(np.sum(fluidPressure,axis=1)), " +/- ", np.std(np.sum(fluidPressure,axis=1)))
    print("dense thermal pressure: ", np.mean(fluidPressure[:,0]), " +/- ", np.std(fluidPressure[:,0]))
    print("dense virial pressure: ", np.mean(fluidPressure[:,1]), " +/- ", np.std(fluidPressure[:,1]))
    print("dense active pressure: ", np.mean(fluidPressure[:,2]), " +/- ", np.std(fluidPressure[:,2]))
    # pressure components in the gas
    print("\ndilute pressure: ", np.mean(np.sum(gasPressure,axis=1)), " +/- ", np.std(np.sum(gasPressure,axis=1)))
    print("dilute thermal pressure: ", np.mean(gasPressure[:,0]), " +/- ", np.std(gasPressure[:,0]))
    print("dilute virial pressure: ", np.mean(gasPressure[:,1]), " +/- ", np.std(gasPressure[:,1]))
    print("dilute active pressure: ", np.mean(gasPressure[:,2]), " +/- ", np.std(gasPressure[:,2]), "\n")

###################### Average radial pressure profile #########################
def averageRadialPressureProfile(dirName, dirSpacing=1, nDim=2):
    ec = 240
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    sigma = float(utils.readFromDynParams(dirName, "sigma"))
    driving = float(utils.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, np.sqrt(boxSize[0]**2 + boxSize[1]**2)/2, 2*np.max(rad))
    binArea = np.pi * (bins[1:]**2 - bins[:-1]**2)
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((dirList.shape[0],bins.shape[0]-1))
    virial = np.zeros((dirList.shape[0],bins.shape[0]-1))
    active = np.zeros((dirList.shape[0],bins.shape[0]-1))
    normal = np.zeros((dirList.shape[0],bins.shape[0]-1))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = getParticleClusterLabels(dirSample, boxSize, eps)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        #print(dirList[d], "maxLabel", maxLabel, "clusterPos", centerOfMass)
        #centerOfMass = np.mean(pos[labels==maxLabel], axis=0)
        #pos = utils.shiftPositions(pos, boxSize, 0.5-centerOfMass[0], 0.5-centerOfMass[1])
        pos = utils.centerDroplet(pos, rad, boxSize, labels, maxLabel)
        centerOfMass = np.mean(pos[labels==maxLabel], axis=0)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            deltaCOM = utils.pbcDistance(pos[i], centerOfMass, boxSize)
            distanceCOM = np.linalg.norm(deltaCOM)
            # tangential unit vector
            #tanCOM = np.array([-deltaCOM[1], deltaCOM[0]])
            #tanCOM /= np.linalg.norm(tanCOM)
            # normal unit vector
            normCOM = deltaCOM / distanceCOM
            normalForce = 0
            #tangentialForce = 0
            #normalWork = 0
            #work = np.zeros(nDim)
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    normalForce += np.abs(np.sum(normCOM*delta)) * gradMultiple / distance
                    #force = gradMultiple * delta / distance
                    #work += 0.5 * force * delta
                    #normalWork += np.dot(deltaCOM, work)
            activeWork = driving * vel[i] * director[i] / (2*Dr)
            activeWork = np.dot(normCOM, activeWork)
            #normalWork = np.dot(deltaCOM,work)
            for j in range(bins.shape[0]-1):
                if(distanceCOM > bins[j] and distanceCOM < bins[j+1]):
                    virial[d,j] += normalForce / (2*np.pi*distanceCOM)
                    #virial[d,j] += normalWork / (np.pi*distanceCOM**2)
                    thermal[d,j] += np.linalg.norm(vel[i])**2 / (np.pi*distanceCOM**2)
                    #active[d,j] += driving * np.sum(vel[i] * director[i]) / (np.pi*distanceCOM**2)
                    active[d,j] += activeWork / (np.pi*distanceCOM**2)
                    #normal[d,j] += normalForce / (2*np.pi*distanceCOM) + (np.linalg.norm(vel[i])**2 + driving * np.sum(vel[i] * director[i]) / (2*Dr)) / (np.pi*distanceCOM**2)
                    normal[d,j] += normalForce / (2*np.pi*distanceCOM) + (np.linalg.norm(vel[i])**2 + activeWork) / (np.pi*distanceCOM**2)
    virial = np.mean(virial, axis=0)*sigma**2
    thermal = np.mean(thermal, axis=0)*sigma**2
    active = np.mean(active, axis=0)*sigma**2
    normal = np.mean(normal, axis=0)*sigma**2
    newCenters = (centers[1:] + centers[:-1]) / 2
    tangential = (normal[1:] + normal[:-1])/2 + 0.5 * newCenters * (normal[1:] - normal[:-1]) / (centers[1:] - centers[:-1])
    normal = (normal[1:] + normal[:-1])/2
    virial = (virial[1:] + virial[:-1])/2
    thermal = (thermal[1:] + thermal[:-1])/2
    active = (active[1:] + active[:-1])/2
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((newCenters, normal, tangential, virial, thermal, active)))
    uplot.plotCorrelation(newCenters[1:], normal[1:], "$Pressure$ $profile$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(newCenters[1:], tangential[1:], "$Pressure$ $profile$", xlabel = "$Distance$", color='r', lw=1.5)
    uplot.plotCorrelation(newCenters[1:], normal[1:] - tangential[1:], "$Pressure$ $profile$", xlabel = "$Distance$", color='b', lw=1.5)
    plt.show()

####################### Average linear pressure profile ########################
def averageLinearPressureProfile(dirName, shiftx=0, dirSpacing=1, nDim=2):
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
    bins = np.arange(0, boxSize[0], 3*np.max(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((dirList.shape[0],bins.shape[0]-1,2))
    virial = np.zeros((dirList.shape[0],bins.shape[0]-1,2))
    active = np.zeros((dirList.shape[0],bins.shape[0]-1,2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        #pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for j in range(bins.shape[0]-1):
            for i in range(numParticles):
                binIndex = -1
                for j in range(bins.shape[0]-1):
                        if(binIndex==-1):
                            if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                                for dim in range(nDim):
                                    thermal[d,j,dim] += vel[i,dim]**2
                                    active[d,j,dim] += driving * vel[i,dim] * director[i,dim] / (2*Dr)
                                binIndex = j
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    radSum = rad[i] + rad[c]
                    delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                    distance = np.linalg.norm(delta)
                    overlap = 1 - distance / radSum
                    if(overlap > 0):
                        gradMultiple = ec * overlap / radSum
                        force = 0.5 * gradMultiple * delta / distance
                        if(pos[c,0] > bins[binIndex] and pos[c,0] <= bins[binIndex+1]):
                            for dim in range(nDim):
                                virial[d,binIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
                        elif(pos[c,0] <= bins[binIndex]): # i is in binIndex and c is in binIndex-1
                            #preBinIndex = binIndex-1
                            for dim in range(nDim):
                                virial[d,binIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
                                #virial[d,preBinIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
                        elif(pos[c,0] > bins[binIndex+1]):
                            #postBinIndex = (binIndex+1)%(bins.shape[0]-1)
                            for dim in range(nDim):
                                virial[d,binIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
                                #virial[d,postBinIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
    thermal = np.mean(thermal,axis=0)*sigma**2 / binArea
    virial = np.mean(virial,axis=0)*sigma**2 / binArea
    active = np.mean(active,axis=0)*sigma**2 / binArea
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((centers, thermal, virial, active)))
    print("surface tension: ", np.sum((virial[:,0]+active[:,0]-virial[:,1]-active[:,1])*binWidth))
    uplot.plotCorrelation(centers, thermal[:,0] + virial[:,0] + active[:,0], "$Pressure$ $profile$", xlabel = "$Distance$", color='b', lw=1.5)
    uplot.plotCorrelation(centers, thermal[:,1] + virial[:,1] + active[:,1], "$Pressure$ $profile$", xlabel = "$Distance$", color='g', lw=1.5)
    #plt.yscale('log')
    plt.show()

######################### Average cluster border work ##########################
def averageClusterSurfaceTension(dirName, dirSpacing=1, nDim=2, plot=False):
    ec = 240
    driving = float(utils.readFromDynParams(dirName, "f0"))
    Dr = float(utils.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    eps = np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    surfaceTension = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        angle = np.mod(angle, 2*np.pi)
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        # load simplices
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample, threshold=0.76, save='save')
        simplices = np.loadtxt(dirSample + os.sep + 'simplices.dat').astype(np.int64)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        #borderSimplexList = simplexList[:,1]
        simplexArea = simplexList[:,2]
        particleList = np.loadtxt(dirSample + os.sep + "particleList.dat")
        borderList = particleList[:,1]
        numSharedSimplices = np.zeros(numParticles)
        for p in range(numParticles):
            numSharedSimplices[p] = np.argwhere(simplices==p)[:,0].shape[0]
        # first need to compute the pressure in the gas
        gasArea = np.sum(simplexArea[denseSimplexList==0])
        fluidArea = np.sum(simplexArea[denseSimplexList==1])
        fluidPerimeter = np.sum(2*rad[borderList==1])
        gasWork = 0
        fluidWork = 0
        for i in range(denseSimplexList.shape[0]):
            if(denseSimplexList[i]==0):
                for p in simplices[i]:
                    gasWork += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
                    gasWork += driving * np.sum(vel[p] * director[p]) / (2*Dr*numSharedSimplices[p])
            else:
                for p in simplices[i]:
                    fluidWork += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
                    fluidWork += driving * np.sum(vel[p] * director[p]) / (2*Dr*numSharedSimplices[p])
            work = 0
            simplexNeighbor = utils.findNeighborSimplices(simplices, i)
            for neighbor in simplexNeighbor:
                edge = np.intersect1d(simplices[i], simplices[neighbor])
                radSum = rad[edge[0]] + rad[edge[1]]
                delta = utils.pbcDistance(pos[edge[0]], pos[edge[1]], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = 0.5 * gradMultiple * delta / distance
                    work += np.sum(force * delta) / nDim
            if(denseSimplexList[i]==0):
                gasWork += work
            else:
                fluidWork += work
        gasPressure = gasWork / gasArea
        fluidPressure = fluidWork / fluidArea
        surfaceTension = np.append(surfaceTension, (fluidPressure - gasPressure) * fluidArea / fluidPerimeter)
    surfaceTension *= sigma
    np.savetxt(dirName + os.sep + "surfaceTension.dat", surfaceTension)
    print("surface tension: ", np.mean(surfaceTension), " +/- ", np.std(surfaceTension))
    if(plot=='plot'):
        numBins = 40
        pdf, edges = np.histogram(surfaceTension, bins=np.linspace(np.min(surfaceTension), np.max(surfaceTension), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        uplot.plotCorrelation(edges, pdf, "$PDF(\\gamma)$", xlabel = "$\\gamma$", color='k')
        #plt.plot()
        plt.pause(0.5)

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
####################### Lennard-Jones potential pressure #######################
################################################################################
def computeLJParticleStress(dirName, LJcutoff=5.5, nDim=2):
    ec = 1
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
    stress = np.zeros((numParticles,4))
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    vel = np.loadtxt(dirName + "/particleVel.dat")
    contacts = np.loadtxt(dirName + "/particleNeighbors.dat").astype(np.int64)
    for i in range(numParticles):
        virial = 0
        shear = 0
        energy = 0
        work = np.zeros(nDim)
        for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
            radSum = rad[i] + rad[c]
            delta = utils.pbcDistance(pos[i], pos[c], boxSize)
            distance = np.linalg.norm(delta)
            if(distance <= (LJcutoff * radSum)):
                forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - forceShift
                force = gradMultiple * delta / distance
                virial += 0.5 * np.sum(force * delta)
                shear -= 0.5 * (force[0] * delta[1] + force[1] * delta[0])
                LJecut = 4*ec * (1/LJcutoff**12 - 1/LJcutoff**6)
                energy += 0.5 * (4*ec * ((radSum / distance)**12 - (radSum / distance)**6) - LJecut - forceShift * (distance - LJcutoff * radSum))
        stress[i,0] = virial*sigma**2/ec
        stress[i,1] = np.linalg.norm(vel[i])**2 / (nDim*sigma**2/ec)
        stress[i,2] = energy + 0.5 * np.linalg.norm(vel[i])**2/ec
        stress[i,3] = shear*sigma**2/ec
    np.savetxt(dirName + os.sep + "particleStress.dat", stress)
    print('virial: ', np.mean(stress[:,0]), ' +- ', np.std(stress[:,0]))
    print('shear: ', np.mean(stress[:,3]), ' +- ', np.std(stress[:,3]))
    print('thermal: ', np.mean(stress[:,1]), ' +- ', np.std(stress[:,1]))
    print('energy: ', np.mean(stress[:,2]), ' +- ', np.std(stress[:,2]))
    return stress

############################ Total stress components ###########################
def computeLJStressVSTime(dirName, LJcutoff=5.5, strain=0, dirSpacing=1, nDim=2):
    ec = 1
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    boxArea = boxSize[0]*boxSize[1]
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    stress = np.zeros((dirList.shape[0],6))
    for d in range(dirList.shape[0]):
        #print(dirList[d])
        dirSample = dirName + os.sep + dirList[d]
        if(strain > 0):
            pos = utils.getLEPBCPositions(dirSample + "/particlePos.dat", boxSize, strain)
        else:
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        for i in range(numParticles):
            # particle pressure components
            stress[d,3] += vel[i,0]**2
            stress[d,4] += vel[i,1]**2
            stress[d,5] -= (vel[i,0] * vel[i,1])
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                if(distance <= (LJcutoff * radSum)):
                    forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - forceShift
                    force = 0.5 * gradMultiple * delta / distance # double counting
                    stress[d,0] += force[0] * delta[0]
                    stress[d,1] += force[1] * delta[1]
                    stress[d,2] -= (force[0] * delta[1] + force[1] * delta[0]) * 0.5
    stress /= (ec / (boxArea * sigma**2))
    np.savetxt(dirName + os.sep + "timeStress.dat", np.column_stack((timeList, stress)))
    print("bulk stress: ", np.mean(stress[:,0]+stress[:,1]+stress[:,3]+stress[:,4]), " +/- ", np.std(stress[:,0]+stress[:,1]+stress[:,3]+stress[:,4]))
    print("shear stress: ", np.mean(stress[:,2]+stress[:,5]), " +/- ", np.std(stress[:,2]+stress[:,5]))
    uplot.plotCorrelation(timeList, stress[:,0]+stress[:,1]+stress[:,3]+stress[:,4], color='k', ylabel='$\\sigma$', xlabel='$t$', logx=True)
    uplot.plotCorrelation(timeList, stress[:,2]+stress[:,5], color='k', ylabel='$\\sigma$', xlabel='$t$', logx=True)
    plt.pause(0.5)

############################ Total stress components ###########################
def computeLJStressVSStrain(dirName, LJcutoff=5.5, strian=0, dirSpacing=1, nDim=2):
    ec = 1
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    boxArea = boxSize[0]*boxSize[1]
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
    dirList, strainList = utils.getShearDirectories(dirName)
    stress = np.zeros((dirList.shape[0],6))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(strain > 0):
            pos = utils.getLEPBCPositions(dirSample + "/particlePos.dat", boxSize, strain)
        else:
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        for i in range(numParticles):
            # particle pressure components
            stress[d,3] += vel[i,0]**2
            stress[d,4] += vel[i,1]**2
            stress[d,5] -= (vel[i,0] * vel[i,1])
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                if(distance <= (LJcutoff * radSum)):
                    forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - forceShift
                    force = 0.5 * gradMultiple * delta / distance # double counting
                    stress[d,0] += force[0] * delta[0]
                    stress[d,1] += force[1] * delta[1]
                    stress[d,2] -= (force[0] * delta[1] + force[1] * delta[0]) * 0.5
    stress /= (ec / (boxArea * sigma**2))
    np.savetxt(dirName + os.sep + "strainStress.dat", np.column_stack((strainList, stress)))
    print("bulk pressure: ", np.mean(stress[:,0]+stress[:,2]), " +/- ", np.std(stress[:,0]+stress[:,2]))
    print("bulk shear stress: ", np.mean(stress[:,1]+stress[:,2]), " +/- ", np.std(stress[:,1]+stress[:,2]))
    uplot.plotCorrelation(strainList, stress[:,1]+stress[:,2], color='k', ylabel='$\\sigma$', xlabel='$\\gamma$')
    plt.pause(0.5)

def computeClusterLJPressureVSTime(dirName, LJcutoff=2.5, dirSpacing=1, nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    pressureIn = np.zeros((dirList.shape[0],2))
    pressureOut = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample, threshold=0.3, save='save')
        simplices = np.loadtxt(dirSample + os.sep + "simplices.dat").astype(np.int64)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        borderSimplexList = simplexList[:,1]
        simplexArea = simplexList[:,2]
        numSharedSimplices = np.zeros(numParticles)
        for p in range(numParticles):
            numSharedSimplices[p] = np.argwhere(simplices==p)[:,0].shape[0]
        pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        # particle pressure components
        areaIn = np.sum(simplexArea[denseSimplexList==1])
        areaOut = np.sum(simplexArea[denseSimplexList==0])
        for i in range(simplices.shape[0]):
            if(denseSimplexList[i]==1):
                for p in simplices[i]:
                    pressureIn[d,0] += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
            else:
                for p in simplices[i]:
                    pressureOut[d,0] += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
            simplexNeighbor = utils.findNeighborSimplices(simplices, i)
            for neighbor in simplexNeighbor:
                edge = np.intersect1d(simplices[i], simplices[neighbor])
                radSum = rad[edge[0]] + rad[edge[1]]
                delta = utils.pbcDistance(pos[edge[0]], pos[edge[1]], boxSize)
                distance = np.linalg.norm(delta)
                if(distance <= (LJcutoff * radSum)):
                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                    force = 0.5 * gradMultiple * delta / distance
            if(denseSimplexList[i] == 1):
                pressureIn[d,1] += np.sum(force * delta) / nDim
            else:
                pressureOut[d,1] += np.sum(force * delta) / nDim
        if(areaIn > 0):
            pressureIn *= sigma**2 / areaIn
        if(areaOut > 0):
            pressureOut *= sigma**2 / areaOut
    np.savetxt(dirName + os.sep + "clusterPressure.dat", np.column_stack((timeList, pressureIn, pressureOut)))
    # pressure components in the fluid
    print("dense pressure: ", np.mean(np.sum(pressureIn,axis=1)), " +/- ", np.std(np.sum(pressureIn,axis=1)))
    print("dense thermal pressure: ", np.mean(pressureIn[:,0]), " +/- ", np.std(pressureIn[:,0]))
    print("dense virial pressure: ", np.mean(pressureIn[:,1]), " +/- ", np.std(pressureIn[:,1]))
    # pressure components in the gas
    print("\ndilute pressure: ", np.mean(np.sum(pressureOut,axis=1)), " +/- ", np.std(np.sum(pressureOut,axis=1)))
    print("dilute thermal pressure: ", np.mean(pressureOut[:,0]), " +/- ", np.std(pressureOut[:,0]))
    print("dilute virial pressure: ", np.mean(pressureOut[:,1]), " +/- ", np.std(pressureOut[:,1]), "\n")

def computeSimplexLJPressureVSTime(dirName, LJcutoff=2.5, dirSpacing=1, nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    borderPressure = np.zeros((dirList.shape[0],2))
    fluidPressure = np.zeros((dirList.shape[0],2))
    gasPressure = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        # load simplices
        if not(os.path.exists(dirSample + os.sep + "simplices.dat")):
            computeDelaunayCluster(dirSample, threshold=0.3, save='save')
        simplices = np.loadtxt(dirSample + os.sep + 'simplices.dat').astype(np.int64)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        borderSimplexList = simplexList[:,1]
        simplexArea = simplexList[:,2]
        numSharedSimplices = np.zeros(numParticles)
        for p in range(numParticles):
            numSharedSimplices[p] = np.argwhere(simplices==p)[:,0].shape[0]
        # first fill out the area occupied by fluid and the gas
        borderArea = 0
        fluidArea = 0
        gasArea = 0
        for i in range(simplexArea.shape[0]):
            if(borderSimplexList[i] == 1):
                borderArea += simplexArea[i]
            else:
                if(denseSimplexList[i]==1):
                    fluidArea += simplexArea[i]
                else:
                    gasArea += simplexArea[i]
        # compute stress components
        for i in range(simplexArea.shape[0]):
            # simplex stress components
            if(borderSimplexList[i]==1):
                for p in simplices[i]:
                    borderPressure[d,0] += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
            else:
                if(denseSimplexList[i]==1):
                    for p in simplices[i]:
                        fluidPressure[d,0] += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
                else:
                    for p in simplices[i]:
                        gasPressure[d,0] += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
            simplexNeighbor = utils.findNeighborSimplices(simplices, i)
            for neighbor in simplexNeighbor:
                edge = np.intersect1d(simplices[i], simplices[neighbor])
                radSum = rad[edge[0]] + rad[edge[1]]
                delta = utils.pbcDistance(pos[edge[0]], pos[edge[1]], boxSize)
                distance = np.linalg.norm(delta)
                if(distance <= (LJcutoff * radSum)):
                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                    force = 0.5 * gradMultiple * delta / distance
                    if(borderSimplexList[i]==1):
                        borderPressure[d,0] += np.sum(force * delta)
                    else:
                        if(denseSimplexList[i]==1):
                            fluidPressure[d,0] += np.sum(force * delta)
                        else:
                            gasPressure[d,0] += np.sum(force * delta)
        if(borderArea > 0):
            borderPressure[d,0] /= borderArea # dim k_B T / dim, dim cancels out
            borderPressure[d,1] /= (dim * borderArea) # double counting
        if(fluidArea > 0):
            fluidPressure[d,0] /= fluidArea # dim k_B T / dim, dim cancels out
            fluidPressure[d,1] /= (dim * fluidArea) # double counting
        if(gasArea > 0):
            gasPressure[d,0] /= gasArea # dim k_B T / dim, dim cancels out
            gasPressure[d,1] /= (dim * gasArea) # double counting
        #print(d, np.sum(fluidPressure[d])*sigma**2, np.sum(gasPressure[d])*sigma**2, np.sum(borderPressure[d])*sigma**2)
    borderPressure *= sigma**2
    fluidPressure *= sigma**2
    gasPressure *= sigma**2
    np.savetxt(dirName + os.sep + "simplexPressure.dat", np.column_stack((timeList, fluidPressure, gasPressure, borderPressure)))
    # pressure components in the fluid
    print("dense pressure: ", np.mean(np.sum(fluidPressure,axis=1)), " +/- ", np.std(np.sum(fluidPressure,axis=1)))
    print("dense thermal pressure: ", np.mean(fluidPressure[:,0]), " +/- ", np.std(fluidPressure[:,0]))
    print("dense virial pressure: ", np.mean(fluidPressure[:,1]), " +/- ", np.std(fluidPressure[:,1]))
    # pressure components in the gas
    print("\ndilute pressure: ", np.mean(np.sum(gasPressure,axis=1)), " +/- ", np.std(np.sum(gasPressure,axis=1)))
    print("dilute thermal pressure: ", np.mean(gasPressure[:,0]), " +/- ", np.std(gasPressure[:,0]))
    print("dilute virial pressure: ", np.mean(gasPressure[:,1]), " +/- ", np.std(gasPressure[:,1]), "\n")

###################### Average radial pressure profile #########################
def averageRadialLJPressureProfile(dirName, LJcutoff=2.5, dirSpacing=1, nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, np.sqrt(boxSize[0]**2 + boxSize[1]**2)/2, 2*np.max(rad))
    binArea = np.pi * (bins[1:]**2 - bins[:-1]**2)
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((dirList.shape[0],bins.shape[0]-1))
    virial = np.zeros((dirList.shape[0],bins.shape[0]-1))
    normal = np.zeros((dirList.shape[0],bins.shape[0]-1))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold=0.3)
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        centerOfMass = np.mean(pos[labels==maxLabel], axis=0)
        #print(dirList[d], "maxLabel", maxLabel, "clusterPos", centerOfMass)
        pos = utils.shiftPositions(pos, boxSize, 0.5-centerOfMass[0], 0.5-centerOfMass[1])
        centerOfMass = np.mean(pos[labels==maxLabel], axis=0)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        for i in range(numParticles):
            if(labels[i]==maxLabel or (labels[i]==-1 or labels[i]==0)):
                deltaCOM = utils.pbcDistance(pos[i], centerOfMass, boxSize)
                distanceCOM = np.linalg.norm(deltaCOM)
                deltaCOM /= distanceCOM
                #normalWork = 0
                normalForce = 0
                tangentialForce = 0
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    radSum = rad[i] + rad[c]
                    delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                    distance = np.linalg.norm(delta)
                    if(distance <= (LJcutoff * radSum)):
                        gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                        #normalWork += np.abs(np.sum(deltaCOM*delta)) * gradMultiple / distance
                        normalForce += np.abs(np.sum(deltaCOM*delta)) * gradMultiple / distance
                for j in range(bins.shape[0]-1):
                    if(distanceCOM > bins[j] and distanceCOM < bins[j+1]):
                        virial[d,j] += normalForce / (2*np.pi*distanceCOM)
                        thermal[d,j] += np.linalg.norm(vel[i])**2 / (np.pi*distanceCOM**2)
                        normal[d,j] += normalForce / (2*np.pi*distanceCOM) + np.linalg.norm(vel[i])**2 / (np.pi*distanceCOM**2)
    virial = np.mean(virial, axis=0)*sigma**2
    thermal = np.mean(thermal, axis=0)*sigma**2
    normal = np.mean(normal, axis=0)*sigma**2
    newCenters = (centers[1:] + centers[:-1]) / 2
    tangential = (normal[1:] + normal[:-1])/2 + 0.5 * newCenters * (normal[1:] - normal[:-1]) / (centers[1:] - centers[:-1])
    normal = (normal[1:] + normal[:-1])/2
    virial = (virial[1:] + virial[:-1])/2
    thermal = (thermal[1:] + thermal[:-1])/2
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((newCenters, normal, tangential, virial, thermal)))
    uplot.plotCorrelation(newCenters, normal, "$Pressure$ $profile$", xlabel = "$Distance$", color='k', lw=1.5)
    uplot.plotCorrelation(newCenters, tangential, "$Pressure$ $profile$", xlabel = "$Distance$", color='r', lw=1.5)
    uplot.plotCorrelation(newCenters, normal - tangential, "$Pressure$ $profile$", xlabel = "$Distance$", color='b', lw=1.5)
    plt.show()

##################### Average LJ linear pressure profile #######################
def averageLinearLJPressureProfile(dirName, LJcutoff=2.5, dirSpacing=1, nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], LJcutoff*np.mean(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((dirList.shape[0],bins.shape[0]-1))
    virial = np.zeros((dirList.shape[0],bins.shape[0]-1,2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        for i in range(numParticles):
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                    thermal[d,j] += np.linalg.norm(vel[i])**2
                    binIndex = j
                    j = bins.shape[0]
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
            #    if((rightOfCenter == True and pos[c,0] < centers[binIndex]) or (leftOfCenter==True and pos[c,0] > centers[binIndex])):
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                if(distance <= (LJcutoff * radSum)):
                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                    force = 0.5 * gradMultiple * delta / distance
                    # find bin where this force belongs to following Irving-Kirkwood
                    if(pos[c,0] > bins[binIndex] and pos[c,0] <= bins[binIndex+1]): # both are in the slab binIndex
                        for dim in range(nDim):
                            virial[d,binIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
                    elif(pos[c,0] <= bins[binIndex]): # i is in binIndex and c is in binIndex-1
                        #preBinIndex = binIndex-1
                        for dim in range(nDim):
                            virial[d,binIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
                            #virial[d,preBinIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
                    elif(pos[c,0] > bins[binIndex+1]):
                        #postBinIndex = (binIndex+1)%(bins.shape[0]-1)
                        for dim in range(nDim):
                            virial[d,binIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
                            #virial[d,postBinIndex,dim] += force[dim] * delta[dim] / np.abs(delta[0])
    thermal = np.mean(thermal,axis=0)*sigma**2 / binArea
    virial = np.mean(virial,axis=0)*sigma**2
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((centers, thermal, virial)))
    print("surface tension: ", np.sum((virial[:,0]-virial[:,1])*binWidth))
    uplot.plotCorrelation(centers, virial[:,0], "$Pressure$ $profile$", xlabel = "$Distance$", color='b', lw=1.5)
    uplot.plotCorrelation(centers, virial[:,1], "$Pressure$ $profile$", xlabel = "$Distance$", color='g', lw=1.5)
    #plt.pause(0.5)
    plt.show()

######################### Average cluster border work ##########################
def averageClusterLJSurfaceTension(dirName, LJcutoff=2.5, dirSpacing=1, nDim=2, plot=False):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    surfaceTension = np.empty(0)
    for d in range(dirList.shape[0]):
        print(dirList[d])
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        # load simplices
        if not(os.path.exists(dirSample + os.sep + "simplexList.dat")):
            computeDelaunayCluster(dirSample, threshold=0.3, save='save')
        simplices = np.loadtxt(dirSample + os.sep + 'simplices.dat').astype(np.int64)
        simplexList = np.loadtxt(dirSample + os.sep + "simplexList.dat")
        denseSimplexList = simplexList[:,0]
        #borderSimplexList = simplexList[:,1]
        simplexArea = simplexList[:,2]
        numSharedSimplices = np.zeros(numParticles)
        for p in range(numParticles):
            numSharedSimplices[p] = np.argwhere(simplices==p)[:,0].shape[0]
        # first need to compute the pressure in the gas
        gasArea = np.sum(simplexArea[denseSimplexList==0])
        gasWork = 0
        for i in range(denseSimplexList.shape[0]):
            if(denseSimplexList[i]==0):
                for p in simplices[i]:
                    gasWork += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
                simplexNeighbor = utils.findNeighborSimplices(simplices, i)
                for neighbor in simplexNeighbor:
                    edge = np.intersect1d(simplices[i], simplices[neighbor])
                    radSum = rad[edge[0]] + rad[edge[1]]
                    delta = utils.pbcDistance(pos[edge[0]], pos[edge[1]], boxSize)
                    distance = np.linalg.norm(delta)
                    if(distance <= (LJcutoff * radSum)):
                        gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                        force = 0.5 * gradMultiple * delta / distance
                        gasWork += np.sum(force * delta) / nDim
        gasPressure = gasWork / gasArea
        # now compute pressure inside each droplet
        labels,_ = getSimplexClusterLabels(dirSample, boxSize, eps, threshold=0.3)
        uniqueLabels = np.unique(labels).astype(np.int64)
        for label in uniqueLabels:
            if(label != -1 or labels[labels==label].shape[0] > 3):
                dropletArea = np.sum(simplexArea[labels==label])
                dropletPerimeter = 0
                for sIndex in np.argwhere(labels==label)[:,0]:
                    indices = utils.findNeighborSimplices(simplices, sIndex)
                    for idx in indices:
                        if(labels[idx] != label): # find the common edge with a simplex not in the droplet
                            edgeIndices = np.intersect1d(simplices[sIndex], simplices[idx])
                            dropletPerimeter += utils.pbcDistance(pos[edgeIndices[0]], pos[edgeIndices[1]], boxSize)
                dropletWork = 0
                for i in range(simplices.shape[0]):
                    if(labels[i]==label):
                        for p in simplices[i]:
                            gasWork += np.linalg.norm(vel[p])**2 / numSharedSimplices[p]
                        simplexNeighbor = utils.findNeighborSimplices(simplices, i)
                        for neighbor in simplexNeighbor:
                            edge = np.intersect1d(simplices[i], simplices[neighbor])
                            radSum = rad[edge[0]] + rad[edge[1]]
                            delta = utils.pbcDistance(pos[edge[0]], pos[edge[1]], boxSize)
                            distance = np.linalg.norm(delta)
                            if(distance <= (LJcutoff * radSum)):
                                gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                                force = 0.5 * gradMultiple * delta / distance
                                dropletWork += np.sum(force * delta) / nDim
                dropletPressure = dropletWork / dropletArea
                surfaceTension = np.append(surfaceTension, (dropletPressure - gasPressure) * dropletArea / dropletPerimeter)
    surfaceTension *= sigma
    np.savetxt(dirName + os.sep + "surfaceTension.dat", surfaceTension)
    print("surface tension: ", np.mean(surfaceTension), " +/- ", np.std(surfaceTension))
    if(plot=='plot'):
        numBins = 40
        pdf, edges = np.histogram(surfaceTension, bins=np.linspace(np.min(surfaceTension), np.max(surfaceTension), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        uplot.plotCorrelation(edges, pdf, "$PDF(\\gamma)$", xlabel = "$\\gamma$", color='k')
        plt.pause(0.5)
        #plt.plot()

####################### Average cluster energy interface #######################
def averageClusterLength(dirName, threshold=0.78, active=False, strain=0, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    spacing = 2*sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    if(active=='active'):
        energyLength = np.zeros((dirList.shape[0],4))
        Dr = float(utils.readFromDynParams(dirName, "Dr"))
        driving = float(utils.readFromDynParams(dirName, "f0"))
    else:
        length = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load positions and simplices
        if(strain!=0):
            pos = utils.getLEPBCPositions(dirSample + "/particlePos.dat", boxSize, strain)
            labels = getLEParticleClusterLabels(dirSample, boxSize, eps, threshold)
        else:
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        height = np.zeros(bins.shape[0]-1)
        posInterface = np.zeros((bins.shape[0]-1,2))
        clusterPos = pos[labels==maxLabel]
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            rightMask = np.argwhere(clusterPos[:,1] > bins[j])[:,0]
            binPos = clusterPos[rightMask]
            leftMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
            binPos = binPos[leftMask]
            if(binPos.shape[0] > 0):
                center = np.mean(binPos, axis=0)[0] # center of dense cluster
                binDistance = binPos[binPos[:,0]>center,0] - center
                borderMask = np.argsort(binDistance)[-5:]
                height[j] = np.mean(binDistance[borderMask])
                posInterface[j,0] = height[-1]
                posInterface[j,1] = np.mean(binPos[borderMask,1])
        if(height[height!=0].shape[0] == height.shape[0]):
            prevPos = posInterface[0]
            length = 0
            for j in range(1,bins.shape[0]-1):
                length += np.linalg.norm(posInterface[j] - prevPos)
                prevPos = posInterface[j]
            length[d,0] = length
            length[d,0] = np.mean(height)
    np.savetxt(dirName + os.sep + "timeLength.dat", np.column_stack((timeList, length)))
    print("average length:", np.mean(length[:,0]), "+-", np.std(length[:,0]))
    print("average height:", np.mean(length[:,1]), "+-", np.std(length[:,1]))
    uplot.plotCorrelation(timeList, length[:,0], "$Length$", xlabel = "$Simulation$ $time,$ $t$", color='k', logx=True)
    plt.pause(0.5)
    #plt.show()

############################ Total stress components ###########################
def computeClusterEnergy(dirName, threshold=0.78, active=False, strain=0, dirSpacing=1, nDim=2):
    if(active=='active'):
        ec = 240
        driving = float(utils.readFromDynParams(dirName, "f0"))
        Dr = float(utils.readFromDynParams(dirName, "Dr"))
    else:
        ec = 1
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    eps = 1.8*sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    if(active=='active'):
        denseEnergy = np.zeros((dirList.shape[0],3))
        diluteEnergy = np.zeros((dirList.shape[0],3))
    else:
        denseEnergy = np.zeros((dirList.shape[0],2))
        diluteEnergy = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(strain != 0):
            #labels = getLEParticleClusterLabels(dirSample, boxSize, eps, threshold)
            _,denseList = getLEParticleDenseLabel(dirSample, boxSize, eps, threshold)
        else:
            #labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
            _,denseList = getParticleDenseLabel(dirSample, boxSize, eps, threshold)
        #maxLabel = utils.findLargestParticleCluster(rad, labels)
        #particleList = np.loadtxt(dirSample + "/particleList.dat")
        #denseList = particleList[:,0]
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        epot = np.loadtxt(dirSample + "/particleEnergies.dat")
        if(active=='active'):
            angle = utils.getMOD2PIAngles(dirSample + "/particleAngles.dat")
            director = np.array([np.cos(angle), np.sin(angle)]).T
        for i in range(numParticles):
            if(denseList[i]==1):
                denseEnergy[d,0] += epot[i]
                denseEnergy[d,1] += np.linalg.norm(vel[i])**2
                if(active=='active'):
                    denseEnergy[d,2] += driving * np.sum(vel[i] * director[i]) / (2*Dr)
            #elif(denseList[i]==0):
            else:
                diluteEnergy[d,0] += epot[i]
                diluteEnergy[d,1] += np.linalg.norm(vel[i])**2
                if(active=='active'):
                    diluteEnergy[d,2] += driving * np.sum(vel[i] * director[i]) / (2*Dr)
    denseEnergy /= ec
    diluteEnergy /= ec
    np.savetxt(dirName + os.sep + "timeEnergy.dat", np.column_stack((timeList, denseEnergy, diluteEnergy)))
    print("dense energy: ", np.mean(np.sum(denseEnergy, axis=1)), " +/- ", np.std(np.sum(denseEnergy, axis=1)))
    print("dilute energy: ", np.mean(np.sum(diluteEnergy, axis=1)), " +/- ", np.std(np.sum(diluteEnergy, axis=1)))
    uplot.plotCorrelation(timeList, np.sum(denseEnergy, axis=1), color='k', ylabel='$Energy$', xlabel='$Simulation$ $time,$ $t$', logx=True)
    uplot.plotCorrelation(timeList, np.sum(diluteEnergy, axis=1), color='r', ylabel='$Energy$', xlabel='$Simulation$ $time,$ $t$', logx=True)
    plt.pause(0.5)

####################### Average cluster height interface #######################
def averageClusterHeightVSTime(dirName, threshold=0.78, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    spacing = 3*sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    #freq = rfftfreq(bins.shape[0]-1, spacing)*sigma
    interface = np.zeros((dirList.shape[0], bins.shape[0]-1))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        # load simplices
        labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        height = np.zeros(bins.shape[0]-1)
        energy = np.zeros((bins.shape[0]-1,2))
        posInterface = np.zeros((bins.shape[0]-1,2))
        clusterPos = pos[labels==maxLabel]
        #clusterPos = clusterPos[np.argwhere(clusterPos[:,0]>center)[:,0]]
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            rightMask = np.argwhere(clusterPos[:,1] > bins[j])[:,0]
            binPos = clusterPos[rightMask]
            leftMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
            binPos = binPos[leftMask]
            if(binPos.shape[0] > 0):
                center = np.mean(binPos, axis=0)[0] # center of dense cluster
                binDistance = binPos[:,0] - center
                borderMask = np.argsort(binDistance)[-3:]
                height[j] = np.mean(binDistance[borderMask])
                posInterface[j,0] = height[-1]
                posInterface[j,1] = np.mean(binPos[borderMask,1])
        if(height[height!=0].shape[0] == height.shape[0]):
            prevPos = posInterface[0]
            length = 0
            for j in range(1,bins.shape[0]-1):
                length += np.linalg.norm(posInterface[j] - prevPos)
                prevPos = posInterface[j]
            interface[d] = height
    np.savetxt(dirName + os.sep + "heightVStime.dat", interface)

####################### Average cluster height interface #######################
def averageClusterHeightFluctuations(dirName, threshold=0.78, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    spacing = 2*sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    #freq = rfftfreq(bins.shape[0]-1, spacing)*sigma
    freq = np.arange(1,centers.shape[0]+1,1)/(centers.shape[0] / spacing)
    deltaHeight = np.zeros((dirList.shape[0], bins.shape[0]-1))
    fourierDeltaHeight = np.zeros((dirList.shape[0], freq.shape[0]))
    energyLength = np.zeros((dirList.shape[0], 3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        epot = np.loadtxt(dirSample + "/particleEnergies.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        # load simplices
        labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        height = np.zeros(bins.shape[0]-1)
        energy = np.zeros((bins.shape[0]-1,2))
        posInterface = np.zeros((bins.shape[0]-1,2))
        clusterPos = pos[labels==maxLabel]
        clusterVel = vel[labels==maxLabel]
        clusterEpot = epot[labels==maxLabel]
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            rightMask = np.argwhere(clusterPos[:,1] > bins[j])[:,0]
            binVel = clusterVel[rightMask]
            binEpot = clusterEpot[rightMask]
            binPos = clusterPos[rightMask]
            leftMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
            binVel = binVel[leftMask]
            binEpot = binEpot[leftMask]
            binPos = binPos[leftMask]
            if(binPos.shape[0] > 0):
                center = np.mean(binPos, axis=0)[0] # center of dense cluster
                binDistance = binPos[:,0] - center
                borderMask = np.argsort(binDistance)[-3:]
                height[j] = np.mean(binDistance[borderMask])
                posInterface[j,0] = height[-1]
                posInterface[j,1] = np.mean(binPos[borderMask,1])
                energy[j,0] = np.mean(epot[borderMask])
                energy[j,1] = np.mean(0.5*np.linalg.norm(vel[borderMask]**2, axis=1))
        if(height[height!=0].shape[0] == height.shape[0]):
            prevPos = posInterface[0]
            length = 0
            for j in range(1,bins.shape[0]-1):
                length += np.linalg.norm(posInterface[j] - prevPos)
                prevPos = posInterface[j]
            energyLength[d,0] = length
            energyLength[d,1] = np.sum(energy[:,0])
            energyLength[d,2] = np.sum(energy[:,1])
            height -= np.mean(height)
            deltaHeight[d] = np.abs(height)**2
            fourierDeltaHeight[d] = np.abs(rfft(height))**2
    print(deltaHeight.shape, deltaHeight[energyLength[:,0]!=0].shape)
    deltaHeight = np.column_stack((np.mean(deltaHeight[energyLength[:,0]!=0], axis=0), np.std(deltaHeight[energyLength[:,0]!=0], axis=0)))
    fourierDeltaHeight = np.column_stack((np.mean(fourierDeltaHeight[energyLength[:,0]!=0], axis=0), np.std(fourierDeltaHeight[energyLength[:,0]!=0], axis=0)))
    #fourierDeltaHeight = rfft(deltaHeight[:,1])
    centers *= sigma
    deltaHeight *= sigma**2
    freq /= sigma
    fourierDeltaHeight /= sigma**2
    np.savetxt(dirName + os.sep + "heightFluctuations.dat", np.column_stack((centers, deltaHeight, freq, fourierDeltaHeight)))
    np.savetxt(dirName + os.sep + "energyLength.dat", np.column_stack((timeList, energyLength)))
    print("average interface length:", np.mean(energyLength[:,0]), np.std(energyLength[:,0]))
    if(plot=='plot'):
        uplot.plotCorrelation(freq[1:], fourierDeltaHeight[1:,0], "$Height$ $fluctuation$", xlabel = "$q$", color='k', logx=True, logy=True)
        #uplot.plotCorrWithError(centers, deltaHeight[:,0], deltaHeight[:,1], "$Height$ $fluctuation$", xlabel = "$y$", color='k')
        #uplot.plotCorrelation(timeList, np.sum(energyLength[:,1:],axis=1), "$Energy$", xlabel = "$Simulation$ $time$", color='k')
        plt.pause(0.5)
        #plt.show()

####################### Average cluster height interface #######################
def computeClusterHeightCorrelation(dirName, threshold=0.78, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    spacing = 3*sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    maxCorrIndex = int((bins.shape[0]-1) / 2)
    distances = centers[:maxCorrIndex]
    heightCorr = np.zeros((dirList.shape[0], maxCorrIndex))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        # load simplices
        labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        height = np.zeros(bins.shape[0]-1)
        clusterPos = pos[labels==maxLabel]
        #clusterPos = clusterPos[np.argwhere(clusterPos[:,0]>center)[:,0]]
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            binPos = clusterPos[np.argwhere(clusterPos[:,1] > bins[j])[:,0]]
            binPos = binPos[np.argwhere(binPos[:,1] <= bins[j+1])[:,0]]
            center = np.mean(binPos, axis=0)[0] # center of dense cluster
            binDistance = binPos[:,0] - center
            height[j] = np.mean(np.sort(binDistance)[-3:])
        corr = np.zeros(maxCorrIndex)
        counts = np.zeros(maxCorrIndex)
        meanHeight = np.mean(height)
        meanHeightSq = np.mean(height**2)
        for j in range(maxCorrIndex):
            for i in range(height.shape[0]):
                index = i+j
                if(index > height.shape[0]-1 and j < int(height.shape[0]/2)):
                    index -= height.shape[0]
                if(index < height.shape[0]):
                    corr[j] += height[i]*height[index]# - meanHeight**2) / (meanHeightSq - meanHeight**2)
                    counts[j] += 1
        heightCorr[d] = np.divide(corr, counts)
    heightCorr = np.column_stack((np.mean(heightCorr, axis=0), np.std(heightCorr, axis=0)))
    np.savetxt(dirName + os.sep + "clusterHeightCorr.dat", np.column_stack((distances, heightCorr)))
    if(plot=='plot'):
        uplot.plotCorrelation(distances[1:], heightCorr[1:,0], "$Height$ $correlation$", xlabel = "$Distance$", color='k')
        plt.pause(0.5)
        #plt.show()

####################### Average linear density profile ########################
def averageLinearDensityProfile(dirName, threshold=0.78, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 4*np.max(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # density lists
    particleDensity = np.zeros((dirList.shape[0], bins.shape[0]-1))
    interfaceWidth = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # first compute particle density
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        for i in range(numParticles):
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                    particleDensity[d,j] += np.pi*rad[i]**2
        particleDensity[d] /= binArea
        # compute width of interface
        center = np.mean(centers)
        x = np.abs(centers-center)
        y = particleDensity[d]
        y = y[np.argsort(x)]
        x = np.sort(x)
        interfaceWidth[d] = utils.computeInterfaceWidth(x, y)
    particleDensity = np.column_stack((np.mean(particleDensity, axis=0), np.std(particleDensity, axis=0))) / binArea
    #simplexDensity = np.column_stack((np.mean(simplexDensity, axis=0), np.std(simplexDensity, axis=0))) / binArea
    np.savetxt(dirName + os.sep + "densityProfile.dat", np.column_stack((centers, particleDensity)))#, simplexDensity)))
    np.savetxt(dirName + os.sep + "interfaceWidth.dat", np.column_stack((timeList, interfaceWidth)))
    print("Interface width:", np.mean(interfaceWidth), "+-", np.std(interfaceWidth))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, interfaceWidth, "$Interface$ $width$")
        #uplot.plotCorrWithError(centers, particleDensity[:,0], particleDensity[:,1], "$Density$ $profile$", xlabel = "$x$", color='k')
        #plt.show()
        plt.pause(0.5)

################################################################################
################### Repulsive-Attractive potential pressure ####################
################################################################################
def computeRAParticleStress(dirName, l1 = 0.04, nDim=2):
    ec = 240
    l2 = 0.5
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
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
        stress[i,0] = virial*sigma**2/ec
        stress[i,1] = np.linalg.norm(vel[i])**2 / (nDim*sigma**2/ec)
        stress[i,2] = energy/ec
    np.savetxt(dirName + os.sep + "particleStress.dat", stress)
    print('virial: ', np.mean(stress[:,0]), ' +- ', np.std(stress[:,0]))
    print('thermal: ', np.mean(stress[:,1]), ' +- ', np.std(stress[:,1]))
    print('energy: ', np.mean(stress[:,2]), ' +- ', np.std(stress[:,2]))
    return stress

def computeRAPressureVSTime(dirName, l1=0.1, dirSpacing=1):
    dim = 2
    ec = 240
    l2 = 0.5
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
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

    elif(whichCorr == "rate"):
        numBlocks = int(sys.argv[3])
        blockPower = int(sys.argv[4])
        blockFreq = float(sys.argv[5])
        spacing = sys.argv[6]
        threshold = float(sys.argv[7])
        plot = sys.argv[8]
        computeClusterRate(dirName, numBlocks, blockPower, blockFreq, spacing, threshold, plot)

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

    elif(whichCorr == "mixingbylabel"):
        numBlocks = int(sys.argv[3])
        blockPower = int(sys.argv[4])
        spacing = sys.argv[5]
        plot = sys.argv[6]
        computeClusterMixingByLabel(dirName, numBlocks, blockPower, spacing, plot)

    elif(whichCorr == "velpdf"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageClusterVelPDF(dirName, threshold, plot)

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

    elif(whichCorr == "droplettime"):
        plot = sys.argv[3]
        computeDropletSizeVSTime(dirName, plot)

############################### Cluster pressure ###############################
    elif(whichCorr == "stress"):
        computeParticleStress(dirName)

    elif(whichCorr == "ptime"):
        computePressureVSTime(dirName)

    elif(whichCorr == "stresstime"):
        computeStressVSTime(dirName, strain=float(sys.argv[3]))

    elif(whichCorr == "stressstrain"):
        computeStressVSStrain(dirName, strain=float(sys.argv[3]))

    elif(whichCorr == "clusterpressure"):
        computeClusterPressureVSTime(dirName)

    elif(whichCorr == "delpressure"):
        computeSimplexPressureVSTime(dirName)

    elif(whichCorr == "radialprofile"):
        averageRadialPressureProfile(dirName)

    elif(whichCorr == "linearprofile"):
        shiftx = float(sys.argv[3])
        averageLinearPressureProfile(dirName, shiftx=shiftx)

    elif(whichCorr == "tension"):
        plot = sys.argv[3]
        averageClusterSurfaceTension(dirName, plot=plot)

    elif(whichCorr == "veltime"):
        plot = sys.argv[3]
        computeVelMagnitudeVSTime(dirName, plot)

    elif(whichCorr == "clusterveltime"):
        plot = sys.argv[3]
        computeClusterVelMagnitudeVSTime(dirName, plot)

######################## Lennard-Jones cluster pressure ########################
    elif(whichCorr == "ljstress"):
        computeLJParticleStress(dirName, LJcutoff=float(sys.argv[3]))

    elif(whichCorr == "ljstresstime"):
        computeLJStressVSTime(dirName, LJcutoff=float(sys.argv[3]), strain=float(sys.argv[4]))

    elif(whichCorr == "ljstressstrain"):
        computeLJStressVSStrain(dirName, LJcutoff=float(sys.argv[3]), strain=float(sys.argv[4]))

    elif(whichCorr == "clusterljpressure"):
        computeClusterLJPressureVSTime(dirName, LJcutoff=float(sys.argv[3]))

    elif(whichCorr == "delljpressure"):
        computeSimplexLJPressureVSTime(dirName, LJcutoff=float(sys.argv[3]))

    elif(whichCorr == "radialljprofile"):
        averageRadialLJPressureProfile(dirName, LJcutoff=float(sys.argv[3]))

    elif(whichCorr == "linearljprofile"):
        averageLinearLJPressureProfile(dirName, LJcutoff=float(sys.argv[3]))

    elif(whichCorr == "ljtension"):
        plot = sys.argv[4]
        averageClusterLJSurfaceTension(dirName, LJcutoff=float(sys.argv[3]), plot=plot)

    elif(whichCorr == "height"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageClusterHeightVSTime(dirName, threshold=threshold, plot=plot)

    elif(whichCorr == "timelength"):
        threshold = float(sys.argv[3])
        active = sys.argv[4]
        strain = float(sys.argv[5])
        averageClusterLength(dirName, threshold=threshold, active=active, strain=strain)

    elif(whichCorr == "timeenergy"):
        threshold = float(sys.argv[3])
        active = sys.argv[4]
        strain = float(sys.argv[5])
        computeClusterEnergy(dirName, threshold=threshold, active=active, strain=strain)

    elif(whichCorr == "heightflu"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageClusterHeightFluctuations(dirName, threshold=threshold, plot=plot)

    elif(whichCorr == "heightcorr"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageClusterHeightCorrelation(dirName, threshold=threshold, plot=plot)

    elif(whichCorr == "profile"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageLinearDensityProfile(dirName, threshold=threshold, plot=plot)

    elif(whichCorr == "temp"):
        threshold = float(sys.argv[3])
        computeClusterTemperatureVSTime(dirName, threshold=threshold)

    else:
        print("Please specify the correlation you want to compute")
