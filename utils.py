'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from scipy.fft import fft, fftfreq, fft2
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from sklearn.cluster import DBSCAN
import spCluster as cluster
import random
import shutil
import os
import time

def lineFit(x, a, b):
    return a + b*x

def quadraticFit(x, a, b, c):
    return a + b*x + c*x**2

def polyFit(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

############################## general utilities ###############################
def pbcDistance(r1, r2, boxSize):
    delta = r1 - r2
    delta += boxSize / 2
    delta %= boxSize
    delta -= boxSize / 2
    return delta

def computeDistances(pos, boxSize):
    numParticles = pos.shape[0]
    distances = (np.repeat(pos[:, np.newaxis, :], numParticles, axis=1) - np.repeat(pos[np.newaxis, :, :], numParticles, axis=0))
    distances += boxSize / 2
    distances %= boxSize
    distances -= boxSize / 2
    distances = np.sqrt(np.sum(distances**2, axis=2))
    return distances

def computeDistancesFromPoint(pos, point, boxSize):
    distances = np.zeros(pos.shape[0])
    for i in range(pos.shape[0]):
        distances[i] = np.linalg.norm(pbcDistance(pos[i], point, boxSize))
    return distances

def computeNeighbors(pos, boxSize, cutoff, maxNeighbors=20):
    numParticles = pos.shape[0]
    neighbors = np.ones((numParticles, maxNeighbors))*-1
    neighborCount = np.zeros(numParticles, dtype=int)
    distance = computeDistances(pos, boxSize)
    for i in range(1,numParticles):
        for j in range(i):
            if(distance[i,j] < cutoff):
                neighbors[i,neighborCount[i]] = j
                neighbors[j,neighborCount[j]] = i
                neighborCount[i] += 1
                neighborCount[j] += 1
                if(neighborCount[i] > maxNeighbors-1 or neighborCount[j] > maxNeighbors-1):
                    print("maxNeighbors update")
                    newMaxNeighbors = np.max([neighborCount[i], neighborCount[j]])
                    neighbors = np.pad(neighbors, (0, newMaxNeighbors-maxNeighbors), 'constant', constant_values=-1)[:numParticles]
                    maxNeighbors = newMaxNeighbors
    return neighbors

def computeDeltas(pos, boxSize):
    numParticles = pos.shape[0]
    deltas = (np.repeat(pos[:, np.newaxis, :], numParticles, axis=1) - np.repeat(pos[np.newaxis, :, :], numParticles, axis=0))
    deltas += boxSize / 2
    deltas %= boxSize
    deltas -= boxSize / 2
    return deltas

def computeTimeDistances(pos1, pos2, boxSize):
    distances = np.zeros((pos1.shape[0], pos1.shape[0]))
    for i in range(pos1.shape[0]):
        for j in range(i):
            delta = pbcDistance(pos1[i], pos2[j], boxSize)
            distances[i,j] = np.linalg.norm(delta)
    return distances

def getPairCorr(pos, boxSize, bins, minRad):
    #distance = np.triu(computeDistances(pos, boxSize),1)
    distance = computeDistances(pos, boxSize)
    distance = distance.flatten()
    distance = distance[distance>0]
    pairCorr, edges = np.histogram(distance, bins=bins)
    binCenter = 0.5 * (edges[:-1] + edges[1:])
    return pairCorr / (2 * np.pi * binCenter)

def getPairCorrelationPeak(pcorr):
    peakIndex = np.argmax(pcorr[:,1])
    x = pcorr[peakIndex-5:peakIndex+5,0]
    y = pcorr[peakIndex-5:peakIndex+5,1]
    failed = False
    try:
        popt, pcov = curve_fit(polyFit, x, y)
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        x = np.linspace(x[0], x[-1], 100)
        y = polyFit(x, *popt)
        return x[np.argmax(y)]
    else:
        return pcorr[np.argmax(pcorr[:,1]),0]
    
def getPairCorrelationPeakLocation(dirName):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = 2 * np.mean(rad)
    bins = np.linspace(0.05*sigma, 4*sigma, 100)
    pos = getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    distance = computeDistances(pos, boxSize)
    pairCorr, edges = np.histogram(distance, bins=bins, density=True)
    binCenter = 0.5 * (edges[:-1] + edges[1:])
    pairCorr /= (2 * np.pi * binCenter)
    pcorr = np.column_stack((binCenter, pairCorr))
    firstPeakLocation = getPairCorrelationPeak(pcorr)
    return firstPeakLocation

def projectToNormalTangentialComp(vectorXY, unitVector): # only for d = 2
    vectorNT = np.zeros((2,2))
    vectorNT[0] = np.dot(vectorXY, unitVector) * unitVector
    vectorNT[1] = vectorXY - vectorNT[0]
    return vectorNT

def polarPos(r, alpha):
    return r * np.array([np.cos(alpha), np.sin(alpha)])

def checkAngle(alpha):
    if(alpha < 0):
        alpha += 2*np.pi
    elif(alpha > 2*np.pi):
        alpha -= 2*np.pi
    return alpha

def computeAdjacencyMatrix(dirName, numParticles=None):
    if(numParticles==None):
        numParticles = int(readFromParams(dirName, "numParticles"))
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat"), dtype=int)
    adjacency = np.zeros((numParticles, numParticles), dtype=int)
    for i in range(numParticles):
        adjacency[i,contacts[i,np.argwhere(contacts[i]!=-1)[:,0]].astype(int)] = 1
    return adjacency

def isNearWall(pos, rad, boxSize):
    isWall = False
    wallPos = np.zeros(pos.shape[0])
    if(pos[0] < rad):
        isWall = True
        wallPos = np.array([0, pos[1]])
    elif(pos[0] > (boxSize[0]-rad)):
        isWall = True
        wallPos = np.array([boxSize[0], pos[1]])
    if(pos[1] < rad):
        isWall = True
        wallPos = np.array([pos[0], 0])
    elif(pos[1] > (boxSize[1]-rad)):
        isWall = True
        wallPos = np.array([pos[0], boxSize[1]])
    return isWall, wallPos

def computePressure(dirName, dim=2, dynamical=False):
    sep = getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    stress = 0
    volume = 1
    for i in range(dim):
        volume *= boxSize[i]
    pos = getPBCPositions(dirName + "/particlePos.dat", boxSize)
    contacts = np.loadtxt(dirName + "/particleContacts.dat").astype(np.int64)
    if(dynamical == "dynamical"):
        vel = np.loadtxt(dirName + "/particleVel.dat")
    for i in range(numParticles):
        virial = 0
        for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
            radSum = rad[i] + rad[c]
            delta = pbcDistance(pos[i], pos[c], boxSize)
            distance = np.linalg.norm(delta)
            overlap = 1 - distance / radSum
            if(overlap > 0):
                gradMultiple = ec * overlap / radSum
                force = gradMultiple * delta / distance
                virial += 0.5 * np.sum(force * delta) # double counting
        stress += virial
        if(dynamical == "dyn"):
            stress += np.linalg.norm(vel[i])**2
    pressure = stress / volume
    # save pressure to params.dat
    fileParams = dirName + "/params.dat"
    if(os.path.exists(fileParams)):
        with open(fileParams, "a") as fparams:
            fparams.write("pressure" + "\t" + str(pressure) + "\n")
    return stress / volume

def computeNumberOfContacts(dirName):
    nContacts = 0
    contacts = np.loadtxt(dirName + "/particleContacts.dat").astype(np.int64)
    numParticles = contacts.shape[0]
    if(numParticles != 0):
        for c in range(contacts.shape[0]):
            nContacts += np.sum(contacts[c]>-1)
        nContacts /= numParticles
    # save pressure to params.dat
    fileParams = dirName + "/params.dat"
    if(os.path.exists(fileParams)):
        with open(fileParams, "a") as fparams:
            fparams.write("numContacts" + "\t" + str(nContacts) + "\n")
    return nContacts

def calcLJgradMultiple(ec, distance, radSum):
    return (24 * ec / distance) * (2 * (radSum / distance)**12 - (radSum / distance)**6)

def calcLJgradMultipleRep(ec, distance, radSum):
    return 2 * (24 * ec / distance) * (radSum / distance)**12

def calcLJgradMultipleAtt(ec, distance, radSum):
    return -(24 * ec / distance) * (radSum / distance)**6

def getEnergyScale(thisId, otherId, num1=0):
    ea = 2
    eb = 2
    eab = 0.5
    if(thisId < num1):
        if(otherId < num1):
            return ea
        else:
            return eab
    else:
        if(otherId >= num1):
            return eb
        else:
            return eab
        
def checkWallInteractionX(thisPos, thisRad, boxSize, dim, LJcutoff=4, ec=2):
    wallPos = np.empty(0)
    if(thisPos[0] < thisRad):
        wallPos = np.array([0, thisPos[1]])
    elif((boxSize[0] - thisPos[0]) < thisRad):
        wallPos = np.array([boxSize[0], thisPos[1]])
    if(wallPos.shape[0] != 0):
        radSum = thisRad
        delta = pbcDistance(thisPos, wallPos, boxSize)
        distance = np.linalg.norm(delta)
        if(distance <= (LJcutoff * radSum)):
            forceShift = calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
            gradMultiple = calcLJgradMultiple(ec, distance, radSum) - forceShift
            # force is only on x since the wall y coordinate is equal to thisPos[1]
            return gradMultiple * delta[0] / distance
    return 0

def checkWallInteractionY(thisPos, thisRad, boxSize, dim, LJcutoff=4, ec=2):
    wallPos = np.empty(0)
    if(thisPos[1] < thisRad):
        wallPos = np.array([thisPos[0], 0])
    elif((boxSize[1] - thisPos[1]) < thisRad):
        wallPos = np.array([thisPos[0], boxSize[1]])
    if(wallPos.shape[0] != 0):
        radSum = thisRad
        delta = pbcDistance(thisPos, wallPos, boxSize)
        distance = np.linalg.norm(delta)
        if(distance <= (LJcutoff * radSum)):
            forceShift = calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
            gradMultiple = calcLJgradMultiple(ec, distance, radSum) - forceShift
            # force is only on y since the wall x coordinate is equal to thisPos[0]
            return gradMultiple * delta[1] / distance
    return 0

############################ correlation functions #############################
def computeIsoCorrFunctions(pos1, pos2, boxSize, waveVector, scale, oneDim = False):
    #delta = pbcDistance(pos1, pos2, boxSize)
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    delta = np.linalg.norm(delta, axis=1)
    if(oneDim == True):
        delta = pos1[:,0] - pos2[:,0]
        delta -= np.mean(delta)
    msd = np.mean(delta**2)/scale
    isf = np.mean(np.sin(waveVector * delta) / (waveVector * delta))
    chi4 = np.mean((np.sin(waveVector * delta) / (waveVector * delta))**2) - isf*isf
    return msd, isf, chi4

def computeLongShortWaveCorr(pos1, pos2, boxSize, longWave, shortWave, scale):
    #delta = pbcDistance(pos1, pos2, boxSize)
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    SqLong = []
    SqShort = []
    angleList = np.arange(0, 2*np.pi, np.pi/8)
    for angle in angleList:
        q = np.array([np.cos(angle), np.sin(angle)])
        SqLong.append(np.mean(np.exp(1j*longWave*np.sum(np.multiply(q, delta), axis=1))))
        SqShort.append(np.mean(np.exp(1j*shortWave*np.sum(np.multiply(q, delta), axis=1))))
    SqLong = np.array(SqLong)
    SqShort = np.array(SqShort)
    ISFLong = np.real(np.mean(SqLong))
    ISFShort = np.real(np.mean(SqShort))
    Chi4Long = np.real(np.mean(SqLong**2) - np.mean(SqLong)**2)
    Chi4Short = np.real(np.mean(SqShort**2) - np.mean(SqShort)**2)
    delta = np.linalg.norm(delta, axis=1)
    MSD = np.mean(delta**2)/scale**2
    return MSD, ISFLong, ISFShort, Chi4Long, Chi4Short

def computeCorrFunctions(pos1, pos2, boxSize, waveVector, scale):
    #delta = pbcDistance(pos1, pos2, boxSize)
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    Sq = []
    angleList = np.arange(0, 2*np.pi, np.pi/8)
    for angle in angleList:
        q = np.array([np.cos(angle), np.sin(angle)])
        Sq.append(np.mean(np.exp(1j*waveVector*np.sum(np.multiply(q, delta), axis=1))))
    Sq = np.array(Sq)
    ISF = np.real(np.mean(Sq))
    Chi4 = np.real(np.mean(Sq**2) - np.mean(Sq)**2)
    delta = np.linalg.norm(delta, axis=1)
    MSD = np.mean(delta**2)/scale
    isoISF = np.mean(np.sin(waveVector * delta) / (waveVector * delta))
    isoChi4 = np.mean((np.sin(waveVector * delta) / (waveVector * delta))**2) - isoISF*isoISF
    alpha2 = np.mean(delta**4)/(2*np.mean(delta**2)**2) - 1
    alpha2new = np.mean(delta**2)/(2*np.mean(1/delta**2)) - 1
    return MSD, ISF, Chi4, isoISF, isoChi4, alpha2, alpha2new

def computeSingleParticleISF(pos1, pos2, boxSize, waveVector, scale):
    #delta = pbcDistance(pos1, pos2, boxSize)
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    delta = np.linalg.norm(delta, axis=1)
    #msd = delta**2
    isf = np.sin(waveVector * delta) / (waveVector * delta)
    return isf

def computeVelCorrFunction(vel1, vel2):
    return np.mean(np.sum(vel1 * vel2, axis=1))

def computeVelCorrFunctions(pos1, pos2, vel1, vel2, dir1, dir2, waveVector, numParticles):
    speed1 = np.linalg.norm(vel1, axis=1)
    velNorm1 = np.mean(speed1)
    speed2 = np.linalg.norm(vel2, axis=1)
    velNorm2 = np.mean(speed2)
    speedCorr = np.mean(speed1 * speed2)
    velCorr = np.mean(np.sum(np.multiply(vel1, vel2)))
    dirCorr = np.mean(np.sum(np.multiply(dir1, dir2)))
    # compute velocity weighted ISF
    delta = pos1 - pos2
    drift = np.mean(pos1 - pos2, axis=0)
    delta[:,0] -= drift[0]
    delta[:,1] -= drift[1]
    velSq = []
    angleList = np.arange(0, 2*np.pi, np.pi/8)
    for angle in angleList:
        unitk = np.array([np.cos(angle), np.sin(angle)])
        k = unitk*waveVector
        weight = np.exp(1j*np.sum(np.multiply(k,delta), axis=1))
        s1 = np.sum(vel1[:,0]*vel2[:,0]*weight)
        s2 = np.sum(vel1[:,0]*vel2[:,1]*weight)
        s3 = np.sum(vel1[:,1]*vel2[:,1]*weight)
        vsf = np.array([[s1, s2], [s2, s3]])
        velSq.append(np.dot(np.dot(unitk, vsf), unitk))
    velISF = np.real(np.mean(velSq))/numParticles
    return speedCorr, velCorr, dirCorr, velISF

def computeSusceptibility(pos1, pos2, field, waveVector, scale):
    delta = pos1[:,0] - pos2[:,0]
    delta -= np.mean(delta)
    chi = np.mean(delta / field)
    isf = np.exp(1.j*waveVector*delta/field)
    chiq = np.mean(isf**2) - np.mean(isf)**2
    return chi / scale, np.real(chiq)

def getRelaxationTime(data, index=2, threshold=np.exp(-1), normalized=False):
    if(normalized == True):
        data[:,index] /= data[0,index]
    relStep = np.argwhere(data[:,index]>threshold)[-1,0]
    if(relStep + 1 < data.shape[0]):
        t1 = data[relStep,0]
        t2 = data[relStep+1,0]
        ISF1 = data[relStep,index]
        ISF2 = data[relStep+1,index]
        slope = (ISF2 - ISF1)/(t2 - t1)
        intercept = ISF2 - slope * t2
        return (np.exp(-1) - intercept)/slope
    else:
        return data[relStep,0]

def computeDecay(x, y, threshold=np.exp(-1), normalize=False):
    if(normalize == True):
        y /= y[0]
    decayStep = np.argwhere(y>threshold)
    if(decayStep.shape[0] > 0):
        decayStep = decayStep[-1,0]
        if(decayStep + 1 < y.shape[0]):
            x1 = x[decayStep]
            x2 = x[decayStep+1]
            y1 = y[decayStep]
            y2 = y[decayStep+1]
            slope = (y2 - y1)/(x2 - x1)
            intercept = y2 - slope * x2
            return (np.exp(-1) - intercept)/slope
    else:
        print("not enough time to compute decay")
        return 0

def computeDeltaChi(data):
    maxStep = np.argmax(data[:,5])
    maxChi = np.max(data[:,5])
    if(maxStep + 1 < data.shape[0]):
        # find values of chi above the max/2
        domeSteps = np.argwhere(data[:,5]>maxChi*0.5)
        t1 = domeSteps[0]
        t2 = domeSteps[-1]
        return t2 - t1
    else:
        return 0
    
def getHeightCorr(height, maxCorrIndex):
    corr = np.zeros(maxCorrIndex)
    counts = np.zeros(maxCorrIndex)
    meanHeight = np.mean(height)
    meanHeightSq = np.mean(height**2)
    height -= meanHeight
    for j in range(maxCorrIndex):
        for i in range(height.shape[0]):
            index = i+j
            if(index > height.shape[0]-1 and j < int(height.shape[0]/2)):
                index -= height.shape[0]
            if(index < height.shape[0]):
                #corr[j] += (height[i] - height[index])**2
                corr[j] += (height[i]*height[index] - meanHeight**2) / (meanHeightSq - meanHeight**2)
                counts[j] += 1
    return np.divide(corr, counts)

def getClusterLength(numCluster, sigma, area):
    if(numCluster < 2):
        return 2 * np.sqrt(numCluster * np.pi * area)
    else:
        return numCluster * np.pi * sigma

############################## Fourier Analysis ################################
def getStructureFactor(pos, q, numParticles):
    sfList = np.zeros(q.shape[0])
    theta = np.arange(0, 2*np.pi, np.pi/8)
    for j in range(q.shape[0]):
        sf = []
        for i in range(theta.shape[0]):
            k = q[j]*np.array([np.cos(theta[i]), np.sin(theta[i])])
            posDotK = np.dot(pos,k)
            sf.append(np.sum(np.exp(-1j*posDotK))*np.sum(np.exp(1j*posDotK)))
        sfList[j] = np.real(np.mean(sf))/numParticles
    return sfList

def getVelocityStructureFactor(pos, vel, q, numParticles):
    velsfList = np.zeros(q.shape[0])
    theta = np.arange(0, 2*np.pi, np.pi/8)
    for j in range(q.shape[0]):
        velsf = []
        for i in range(theta.shape[0]):
            unitk = np.array([np.cos(theta[i]), np.sin(theta[i])])
            k = unitk*q[j]
            posDotK = np.dot(pos,k)
            s1 = np.sum(vel[:,0]*vel[:,0]*np.exp(-1j*posDotK))*np.sum(vel[:,0]*vel[:,0]*np.exp(1j*posDotK))
            s2 = np.sum(vel[:,0]*vel[:,1]*np.exp(-1j*posDotK))*np.sum(vel[:,0]*vel[:,1]*np.exp(1j*posDotK))
            s3 = np.sum(vel[:,1]*vel[:,1]*np.exp(-1j*posDotK))*np.sum(vel[:,1]*vel[:,1]*np.exp(1j*posDotK))
            vsf = np.array([[s1, s2], [s2, s3]])
            velsf.append(np.dot(np.dot(unitk, vsf), unitk))
        velsfList[j] = np.real(np.mean(velsf))/numParticles
    return velsfList

def getSpaceFourierEnergy(pos, vel, epot, q, numParticles):
    kq = np.zeros(q.shape[0])
    uq = np.zeros(q.shape[0])
    kcorr = np.zeros(q.shape[0])
    theta = np.arange(0, 2*np.pi, np.pi/8)
    for j in range(q.shape[0]):
        ktemp = []
        utemp = []
        kctemp = []
        for i in range(theta.shape[0]):
            unitk = np.array([np.cos(theta[i]), np.sin(theta[i])])
            k = unitk*q[j]
            posDotK = np.dot(pos,k)
            ekin = 0.5*np.linalg.norm(vel, axis=1)**2
            ktemp.append(np.sum(ekin.T*np.exp(-1j*posDotK)))
            utemp.append(np.sum(epot.T*np.exp(1j*posDotK))*np.sum(epot.T*np.exp(-1j*posDotK)))
            # correlations
            kctemp.append(np.sum(ekin.T*np.exp(1j*posDotK))*np.sum(ekin.T*np.exp(-1j*posDotK)))
        kq[j] = np.abs(np.mean(ktemp))/numParticles
        uq[j] = np.abs(np.mean(utemp))/numParticles
        kcorr[j] = np.abs(np.mean(kctemp))/numParticles
    return kq, uq, kcorr

def getTimeFourierEnergy(dirName, dirList, dirSpacing, numParticles):
    timeStep = readFromParams(dirName, "dt")
    numSteps = dirList.shape[0]
    freq = fftfreq(numSteps, dirSpacing*timeStep)
    energy = np.zeros((numSteps,numParticles,2))
    corre = np.zeros((numSteps,numParticles,2))
    initialEpot = np.array(np.loadtxt(dirName + os.sep + "t0/particleEnergy.dat"), dtype=np.float64)
    initialVel = np.array(np.loadtxt(dirName + os.sep + "t0/particleVel.dat"), dtype=np.float64)
    initialEkin = 0.5*np.linalg.norm(initialVel, axis=1)**2
    # collect instantaneous energy for 10 particles
    for i in range(numSteps):
        vel = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleVel.dat"), dtype=np.float64)
        epot = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleEnergy.dat"), dtype=np.float64)
        ekin = 0.5*np.linalg.norm(vel, axis=1)**2
        energy[i,:,0] = ekin
        energy[i,:,1] = epot
        corre[i,:,0] = ekin*initialEkin
        corre[i,:,1] = epot*initialEpot
    # compute fourier transform and average over particles
    energyf = np.zeros((numSteps, 3), dtype=complex)
    corref = np.zeros((numSteps, 3), dtype=complex)
    for pId in range(numParticles):
        energyf[:,0] += fft(energy[:,pId,0])
        energyf[:,1] += fft(energy[:,pId,1])
        energyf[:,2] += fft(energy[:,pId,0] + energy[:,pId,1])
        # correlations
        corref[:,0] += fft(corre[:,pId,0])
        corref[:,1] += fft(corre[:,pId,1])
        corref[:,2] += fft(corre[:,pId,0] + corre[:,pId,1])
    energyf /= numParticles
    energyf = energyf[np.argsort(freq)]
    corref /= numParticles
    corref = corref[np.argsort(freq)]
    freq = np.sort(freq)
    return np.column_stack((freq, np.abs(energyf)*2/numSteps, np.abs(corref)*2/numSteps))

def getSpaceFourierVelocity(pos, vel, q, numParticles):
    vq = np.zeros((q.shape[0],2))
    theta = np.arange(0, 2*np.pi, np.pi/8)
    for j in range(q.shape[0]):
        vqtemp = np.zeros(2)
        for i in range(theta.shape[0]):
            unitk = np.array([np.cos(theta[i]), np.sin(theta[i])])
            k = unitk*q[j]
            posDotK = np.dot(pos,k)
            vqx = np.sum(vel[:,0].T*np.exp(-1j*posDotK))
            vqy = np.sum(vel[:,1].T*np.exp(-1j*posDotK))
            vqxy = np.array([vqx, vqy])
            vqtemp[0] += np.mean(np.abs(np.dot(vqxy, unitk))**2)
            vqtemp[1] += np.mean(np.abs(np.cross(vqxy, unitk))**2)
        vq[j] = vqtemp/theta.shape[0]
    return vq

def getTimeFourierVel(dirName, dirList, dirSpacing, numParticles):
    timeStep = readFromParams(dirName, "dt")
    numSteps = dirList.shape[0]
    freq = fftfreq(numSteps, dirSpacing*timeStep)
    veltot = np.zeros((numSteps,numParticles,2))
    # collect instantaneous energy for 10 particles
    for i in range(numSteps):
        vel = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleVel.dat"), dtype=np.float64)
        veltot[i] = vel
    # compute fourier transform and average over particles
    velf = np.zeros((numSteps,2), dtype=complex)
    for pId in range(numParticles):
        velf[:,0] += fft(veltot[:,pId,0])
        velf[:,1] += fft(veltot[:,pId,1])
    velf /= numParticles
    velf = velf[np.argsort(freq)]
    velfSquared1 = np.mean(np.abs(velf)**2,axis=1)*2/numSteps
    # compute fourier transform and average over particles
    velf = np.zeros((numSteps,2))
    for pId in range(numParticles):
        velf[:,0] += np.abs(fft(veltot[:,pId,0]))**2
        velf[:,1] += np.abs(fft(veltot[:,pId,1]))**2
    velf /= numParticles
    velf = velf[np.argsort(freq)]
    velSquared2 = np.mean(velf,axis=1)*2/numSteps
    freq = np.sort(freq)
    return np.column_stack((freq, velfSquared1, velSquared2))

def wrapTripleClusterLabels(allLabels, maxLabel, numParticles):
    wrappedLabels = np.zeros(numParticles)
    leftLabels = allLabels[:numParticles]
    wrappedLabels = allLabels[numParticles:2*numParticles]
    rightLabels = allLabels[2*numParticles:]
    wrappedLabels[np.argwhere(leftLabels==maxLabel)] = maxLabel
    wrappedLabels[np.argwhere(rightLabels==maxLabel)] = maxLabel
    return wrappedLabels

def triplePositions(pos, boxSize):
    # copy positions to the right and store indices corresponding to positions in the main box
    triplePos = np.zeros((pos.shape[0]*3, pos.shape[1]))
    triplePos[:pos.shape[0],0] = pos[:,0] - boxSize[0]
    for dim in range(1,pos.shape[1]):
        triplePos[:pos.shape[0],dim] = pos[:,dim]
    triplePos[pos.shape[0]:2*pos.shape[0]] = pos
    triplePos[2*pos.shape[0]:,0] = pos[:,0] + boxSize[0]
    for dim in range(1,pos.shape[1]):
        triplePos[2*pos.shape[0]:,dim] = pos[:,dim]
    return triplePos

def wrapDoubleClusterLabels(allLabels, maxLabel, numParticles):
    wrappedLabels = np.zeros(numParticles)
    wrappedLabels = allLabels[:numParticles]
    rightLabels = allLabels[numParticles:]
    wrappedLabels[np.argwhere(rightLabels==maxLabel)] = maxLabel
    return wrappedLabels

def doublePositions(pos, boxSize):
    # copy positions to the right and store indices corresponding to positions in the main box
    doublePos = np.zeros((pos.shape[0]*2, pos.shape[1]))
    doublePos[:pos.shape[0]] = pos
    doublePos[pos.shape[0]:,0] = pos[:,0] + boxSize[0]
    for dim in range(1,pos.shape[1]):
        doublePos[pos.shape[0]:,dim] = pos[:,dim]
    return doublePos

############################### Delaunay analysis ##############################
def augmentPacking(pos, rad, fraction=0.15, lx=1, ly=1):
    # augment packing by copying a fraction of the particles around the walls
    Lx = np.array([lx,0])
    Ly = np.array([0,ly])
    leftPos = pos[pos[:,0]<fraction*lx]
    leftRad = rad[pos[:,0]<fraction*lx]
    leftIndices = np.argwhere(pos[:,0]<fraction*lx)[:,0]
    rightPos = pos[pos[:,0]>(1-fraction)*lx]
    rightRad = rad[pos[:,0]>(1-fraction)*lx]
    rightIndices = np.argwhere(pos[:,0]>(1-fraction)*lx)[:,0]
    bottomPos =  pos[pos[:,1]<fraction*ly]
    bottomRad =  rad[pos[:,1]<fraction*ly]
    bottomIndices = np.argwhere(pos[:,1]<fraction*ly)[:,0]
    topPos = pos[pos[:,1]>(1-fraction)*ly]
    topRad = rad[pos[:,1]>(1-fraction)*ly]
    topIndices = np.argwhere(pos[:,1]>(1-fraction)*ly)[:,0]
    bottomLeftPos = leftPos[leftPos[:,1]<fraction*ly]
    bottomLeftRad = leftRad[leftPos[:,1]<fraction*ly]
    bottomLeftIndices = leftIndices[np.argwhere(leftPos[:,1]<fraction*ly)[:,0]]
    bottomRightPos = rightPos[rightPos[:,1]<fraction*ly]
    bottomRightRad = rightRad[rightPos[:,1]<fraction*ly]
    bottomRightIndices = rightIndices[np.argwhere(rightPos[:,1]<fraction*ly)[:,0]]
    topLeftPos = leftPos[leftPos[:,1]>(1-fraction)*ly]
    topLeftRad = leftRad[leftPos[:,1]>(1-fraction)*ly]
    topLeftIndices = leftIndices[np.argwhere(leftPos[:,1]>(1-fraction)*ly)[:,0]]
    topRightPos = rightPos[rightPos[:,1]>(1-fraction)*ly]
    topRightRad = rightRad[rightPos[:,1]>(1-fraction)*ly]
    topRightIndices = rightIndices[np.argwhere(rightPos[:,1]>(1-fraction)*ly)[:,0]]
    newPos = np.vstack([pos, leftPos + Lx, rightPos - Lx, bottomPos + Ly, topPos - Ly, bottomLeftPos + Lx + Ly, bottomRightPos - Lx + Ly, topLeftPos + Lx - Ly, topRightPos - Lx - Ly])
    newRad = np.concatenate((rad, leftRad, rightRad, bottomRad, topRad, bottomLeftRad, bottomRightRad, topLeftRad, topRightRad))
    newIndices = np.concatenate((np.arange(0,rad.shape[0],1), leftIndices, rightIndices, bottomIndices, topIndices, bottomLeftIndices, bottomRightIndices, topLeftIndices, topRightIndices))
    return newPos, newRad, newIndices

def isSimplexNearWall(simplex, pos, rad, boxSize):
    isSimplexNearWall = False
    for p in simplex:
        isWall, _ = isNearWall(pos[p], 2*rad[p], boxSize)
        if(isWall == True):
            isSimplexNearWall = True
    return isSimplexNearWall

def isOutsideBox(pos, boxSize):
    isOutsideWall = 0
    if(pos[0] > boxSize[0] or pos[0] < 0):
        isOutsideWall += 1
    if(pos[1] > boxSize[1] or pos[1] < 0):
        isOutsideWall += 1
    if(isOutsideWall > 0):
        return 1
    else:
        return 0

def getInsideBoxDelaunaySimplices(simplices, pos, boxSize):
    insideIndex = np.ones(simplices.shape[0])
    for sIndex in range(simplices.shape[0]):
        isOutside = 0
        for i in range(simplices[sIndex].shape[0]):
            isOutside += isOutsideBox(pos[simplices[sIndex,i]], boxSize)
        if(isOutside == 3): # all the particles on the simplex are outside the box
            insideIndex[sIndex] = 0
    return insideIndex

def getOnWallDelaunaySimplices(simplices, pos, boxSize):
    onWallIndex = np.zeros(simplices.shape[0])
    for sIndex in range(simplices.shape[0]):
        isOutside = 0
        for i in range(simplices[sIndex].shape[0]):
            isOutside += isOutsideBox(pos[simplices[sIndex,i]], boxSize)
        if(isOutside == 1 or isOutside == 2): # one or two particles on the simplex are outside the box
            onWallIndex[sIndex] = 1
    return onWallIndex

def wrapSimplicesAroundBox(innerSimplices, augmentedIndices, numParticles):
    for sIndex in range(innerSimplices.shape[0]):
        for i in range(innerSimplices[sIndex].shape[0]):
            if(innerSimplices[sIndex,i] > (numParticles - 1)):
                innerSimplices[sIndex,i] = augmentedIndices[innerSimplices[sIndex,i]]
    return innerSimplices

def getPBCDelaunay(pos, rad, boxSize):
    newPos, _, newIndices = augmentPacking(pos, rad, 0.1, boxSize[0], boxSize[1])
    delaunay = Delaunay(newPos)
    insideIndex = getInsideBoxDelaunaySimplices(delaunay.simplices, newPos, boxSize)
    simplices = wrapSimplicesAroundBox(delaunay.simplices[insideIndex==1], newIndices, rad.shape[0])
    return np.unique(np.sort(simplices, axis=1), axis=0).astype(np.int64)
    #return np.unique(simplices), axis=0)

def findNeighborSimplices(simplices, sIndex):
    neighborList = []
    vertices = simplices[sIndex]
    # find simplices which have a pair of the three vertices
    index0List = np.argwhere(simplices==vertices[0])[:,0]
    index1List = np.argwhere(simplices==vertices[1])[:,0]
    index2List = np.argwhere(simplices==vertices[2])[:,0]
    intersect01 = np.intersect1d(index0List, index1List)
    intersect12 = np.intersect1d(index1List, index2List)
    intersect20 = np.intersect1d(index2List, index0List)
    # need to subtract shorter array from larger array
    neighbors = np.empty(0)
    if(intersect01.shape[0] >= intersect12.shape[0]):
        firstNeighborList = np.setdiff1d(intersect01, intersect12)
    else:
        firstNeighborList = np.setdiff1d(intersect12, intersect01)
    if(firstNeighborList.shape[0] != 0):
        neighbors = np.append(neighbors,firstNeighborList[0])
    if(intersect12.shape[0] >= intersect20.shape[0]):
        secondNeighborList = np.setdiff1d(intersect12, intersect20)
    else:
        secondNeighborList = np.setdiff1d(intersect20, intersect12)
    if(secondNeighborList.shape[0] != 0):
        neighbors = np.append(neighbors,secondNeighborList[0])
    if(intersect20.shape[0] >= intersect01.shape[0]):
        thirdNeighborList = np.setdiff1d(intersect20, intersect01)
    else:
        thirdNeighborList = np.setdiff1d(intersect01, intersect20)
    if(thirdNeighborList.shape[0] != 0):
        neighbors = np.append(neighbors,thirdNeighborList[0])
    return neighbors.astype(np.int64)

def findAllNeighborSimplices(simplices, sIndex):
    neighborList = []
    vertices = simplices[sIndex]
    # find simplices which at least one vertex in common
    index0List = np.argwhere(simplices==vertices[0])[:,0]
    index1List = np.argwhere(simplices==vertices[1])[:,0]
    index2List = np.argwhere(simplices==vertices[2])[:,0]
    vertex0Neighbors = np.setdiff1d(index0List, sIndex)
    vertex1Neighbors = np.setdiff1d(index1List, sIndex)
    vertex2Neighbors = np.setdiff1d(index2List, sIndex)
    return np.unique(np.concatenate((vertex0Neighbors, vertex1Neighbors, vertex2Neighbors)))

def getDelaunaySimplexPos(pos, rad, boxSize):
    simplices = getPBCDelaunay(pos, rad, boxSize)
    simplexPos = np.zeros((simplices.shape[0], 2))
    for i in range(simplices.shape[0]):
        # average positions of particles / vertices of simplex i
        simplexPos[i] = np.mean(pos[simplices[i]], axis=0)
    #simplexPos[:,0] -= np.floor(simplexPos[:,0]/boxSize[0]) * boxSize[0]
    #simplexPos[:,1] -= np.floor(simplexPos[:,1]/boxSize[1]) * boxSize[1]
    return simplexPos

# this functions checks for particles that intersect the edge in front of their center of mass in a Delaunay simplex
def checkDelaunayInclusivity(simplices, pos, rad, boxSize):
    intersectParticle = 0
    wallParticle = 0
    for sIndex in range(simplices.shape[0]):
        pos0 = pos[simplices[sIndex,0]]
        pos1 = pos[simplices[sIndex,1]]
        pos2 = pos[simplices[sIndex,2]]
        pos2 = pbcDistance(pos2, pos1, boxSize)
        pos0 = pbcDistance(pos0, pos1, boxSize)
        pos1 = np.zeros(pos1.shape[0])
        slope = pbcDistance(pos1[1],pos0[1],boxSize[1]) / pbcDistance(pos1[0],pos0[0],boxSize[0])
        intercept = pos0[1] - pos0[0] * slope
        projLength = np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 - 2*slope*pos2[0]*pos2[1]) / (1 + slope**2))
        if(rad[simplices[sIndex,2]] > projLength):
            intersectParticle += 1
            print("Particle", simplices[sIndex,2], "has radius", rad[simplices[sIndex,2]], "and distance from opposite edge", projLength)
            wallCheck, _ = isNearWall(pos[simplices[sIndex,2]], 2*rad[simplices[sIndex,2]], boxSize)
            if(wallCheck):
                print("AND THIS PARTICLE IS NEAR THE WALL")
                wallParticle += 1
    print("This packing has", intersectParticle, "particles that intersect the opposite Delaunay edge")
    print("AND", wallParticle, "OF THESE ARE NEAR A WALL")

def computeIntersectionArea(pos0, pos1, pos2, sigma, boxSize, sIndex):
    pos2 = pbcDistance(pos2, pos1, boxSize)
    pos0 = pbcDistance(pos0, pos1, boxSize)
    pos1 = np.zeros(pos1.shape[0])
    # full formula is: np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 + intercept**2 - 2*intercept*pos2[1] - 2*slope*pos2[0]*pos2[1] + 2*slope*intercept*pos2[1]) / (1 + slope**2))
    slope = pbcDistance(pos1[1],pos0[1],boxSize[1]) / pbcDistance(pos1[0],pos0[0],boxSize[0])
    intercept = pos0[1] - pos0[0] * slope
    # length of segment from point to projection
    projLength = np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 - 2*slope*pos2[0]*pos2[1]) / (1 + slope**2))
    #projLength = np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 + intercept**2 - 2*intercept*pos2[1] - 2*slope*pos2[0]*pos2[1] + 2*slope*intercept*pos2[1]) / (1 + slope**2))
    lengthRatio = projLength / np.linalg.norm(pos2)
    if(lengthRatio > 1 and (1-lengthRatio)<1e-12): #it's above 1 by a very small amount
        theta = np.arcsin(1)
    else:
        theta = np.arcsin(lengthRatio)
    intersectArea = 0.5 * sigma**2 * theta
    # check if distances from two opposite vertices to the projection point are both less than the distance between the two opposite vertices
    delta10 = np.linalg.norm(pos0)
    delta12 = np.linalg.norm(pos2)
    delta20 = np.linalg.norm(pbcDistance(pos2, pos0, boxSize))
    delta1Proj = delta12 * np.cos(theta) # this angle is the same as theta
    #delta0Proj = delta20 * np.cos(np.arcsin(projLength / delta20))
    internalProj = True
    #if(delta1Proj > delta10 or delta0Proj > delta10):
    #    internalProj = False
    if(delta1Proj > delta10 or projLength > delta20):
        internalProj = False
    return projLength, intersectArea, internalProj

def checkSegmentArea(projLength, sigma, internalProj):
    if(projLength < sigma and internalProj == True):
        smallTheta = np.arccos(projLength/sigma)
        smallTheta /= 2
        return smallTheta * sigma**2 - 0.5 * sigma**2 * np.sin(2*smallTheta)
    else:
        return 0

def findOppositeSimplexIndex(simplices, sIndex, indexA, indexB):
    # find simplex where the intersection is and add segmentArea to it
    indexList = np.intersect1d(np.argwhere(simplices==indexA)[:,0], np.argwhere(simplices==indexB)[:,0])
    # remove sIndex from indexList
    oppositeIndex = np.setdiff1d(indexList, sIndex)
    return oppositeIndex

def computeTriangleArea(pos0, pos1, pos2, boxSize):
    delta01 = np.linalg.norm(pbcDistance(pos0, pos1, boxSize))
    delta12 = np.linalg.norm(pbcDistance(pos1, pos2, boxSize))
    delta20 = np.linalg.norm(pbcDistance(pos2, pos0, boxSize))
    semiPerimeter = 0.5 * (delta01 + delta12 + delta20)
    return np.sqrt(semiPerimeter * (semiPerimeter - delta01) * (semiPerimeter - delta12) * (semiPerimeter - delta20))

def computeDelaunayDensity(simplices, pos, rad, boxSize):
    simplexDensity = np.zeros(simplices.shape[0])
    simplexArea = np.zeros(simplices.shape[0])
    occupiedArea = np.zeros(simplices.shape[0])
    for sIndex in range(simplices.shape[0]):
        if(np.unique(simplices[sIndex]).shape[0] != 3):#problem with the wrapping, some simplices have the same particle twice
            simplexArea[sIndex] = 0
            occupiedArea[sIndex] = 0
        else:
            pos0 = pos[simplices[sIndex,0]]
            pos1 = pos[simplices[sIndex,1]]
            pos2 = pos[simplices[sIndex,2]]
            # compute area of the triangle
            simplexArea[sIndex] = computeTriangleArea(pos0, pos1, pos2, boxSize)
            # compute the three areas of the intersecating circles
            # first compute projection distance for each vertex in the simplex and then check if the intersection is all inside the simplex
            # if not, remove the external segment from the intersection area and add it to the simplex where the segment is contained
            # first vertex
            projLength, intersectArea1, internalProj = computeIntersectionArea(pos0, pos1, pos2, rad[simplices[sIndex,1]], boxSize, sIndex)
            segmentArea2 = checkSegmentArea(projLength, rad[simplices[sIndex,2]], internalProj)
            # second vertex
            projLength, intersectArea2, internalProj = computeIntersectionArea(pos1, pos2, pos0, rad[simplices[sIndex,2]], boxSize, sIndex)
            segmentArea0 = checkSegmentArea(projLength, rad[simplices[sIndex,0]], internalProj)
            # third vertex
            projLength, intersectArea0, internalProj = computeIntersectionArea(pos2, pos0, pos1, rad[simplices[sIndex,0]], boxSize, sIndex)
            segmentArea1 = checkSegmentArea(projLength, rad[simplices[sIndex,1]], internalProj)
            # first correction
            if(segmentArea2 > 0):
                oppositeIndex = findOppositeSimplexIndex(simplices, sIndex, simplices[sIndex,0], simplices[sIndex,1])
                if(oppositeIndex.shape[0] == 1):
                    occupiedArea[oppositeIndex[0]] += segmentArea2
            # second correction
            if(segmentArea0 > 0):
                oppositeIndex = findOppositeSimplexIndex(simplices, sIndex, simplices[sIndex,1], simplices[sIndex,2])
                if(oppositeIndex.shape[0] == 1):
                    occupiedArea[oppositeIndex[0]] += segmentArea0
            # third correction
            if(segmentArea1 > 0):
                oppositeIndex = findOppositeSimplexIndex(simplices, sIndex, simplices[sIndex,2], simplices[sIndex,0])
                if(oppositeIndex.shape[0] == 1):
                    occupiedArea[oppositeIndex[0]] += segmentArea1
            occupiedArea[sIndex] += (intersectArea1 + intersectArea2 + intersectArea0 - segmentArea2 - segmentArea0 - segmentArea1)
            # subtract overlapping area, there are two halves for each simplex
            occupiedArea[sIndex] -= 0.5*computeOverlapArea(pos1, pos2, rad[simplices[sIndex,1]], rad[simplices[sIndex,2]], boxSize) + 0.5*computeOverlapArea(pos2, pos1, rad[simplices[sIndex,2]], rad[simplices[sIndex,1]], boxSize)
            occupiedArea[sIndex] -= 0.5*computeOverlapArea(pos1, pos0, rad[simplices[sIndex,1]], rad[simplices[sIndex,0]], boxSize) + 0.5*computeOverlapArea(pos0, pos1, rad[simplices[sIndex,0]], rad[simplices[sIndex,1]], boxSize)
            occupiedArea[sIndex] -= 0.5*computeOverlapArea(pos2, pos0, rad[simplices[sIndex,2]], rad[simplices[sIndex,0]], boxSize) + 0.5*computeOverlapArea(pos0, pos2, rad[simplices[sIndex,0]], rad[simplices[sIndex,2]], boxSize)
    for i in range(simplexArea.shape[0]):
        if(simplexArea[i] > 0):
            simplexDensity[i] = occupiedArea[i] / simplexArea[i]
    return simplexDensity, simplexArea

def computeIntersectionArea2(pos0, pos1, pos2, sigma, boxSize):
    # define reference frame to simplify projection formula
    # full formula is:
    #projLength = np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 + intercept**2 - 2*intercept*pos2[1] - 2*slope*pos2[0]*pos2[1] + 2*slope*intercept*pos2[1]) / (1 + slope**2))
    pos2 = pbcDistance(pos2, pos1, boxSize)
    pos0 = pbcDistance(pos0, pos1, boxSize)
    pos1 = np.zeros(pos1.shape[0])
    slope = (pos1[1] - pos0[1]) / (pos1[0] - pos0[0])
    intercept = pos0[1] - pos0[0] * slope
    # length of segment from point to projection
    projLength = np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 - 2*slope*pos2[0]*pos2[1]) / (1 + slope**2))
    theta = np.arcsin(projLength / np.linalg.norm(pos2))
    return 0.5*sigma**2*theta

def computeDelaunayDensity2(simplices, pos, rad, boxSize):
    simplexDensity = np.zeros(simplices.shape[0])
    simplexArea = np.zeros(simplices.shape[0])
    for sIndex in range(simplices.shape[0]):
        pos0 = pos[simplices[sIndex,0]]
        pos1 = pos[simplices[sIndex,1]]
        pos2 = pos[simplices[sIndex,2]]
        # compute area of the triangle
        triangleArea = 0.5 * np.abs(pos0[0]*pbcDistance(pos1[1],pos2[1],boxSize[1]) + pos1[0]*pbcDistance(pos2[1],pos0[1],boxSize[1]) + pos2[0]*pbcDistance(pos0[1],pos1[1],boxSize[1]))
        # compute the three areas of the intersecating circles
        intersectArea = computeIntersectionArea2(pos0, pos1, pos2, rad[simplices[sIndex,1]], boxSize)# - 0.5 * computeOverlapArea(pos1, pos2, rad[simplices[sIndex,1]], rad[simplices[sIndex,2]], boxSize)
        intersectArea += computeIntersectionArea2(pos1, pos2, pos0, rad[simplices[sIndex,2]], boxSize)# - 0.5 * computeOverlapArea(pos2, pos0, rad[simplices[sIndex,2]], rad[simplices[sIndex,0]], boxSize)
        intersectArea += computeIntersectionArea2(pos2, pos0, pos1, rad[simplices[sIndex,0]], boxSize)# - 0.5 * computeOverlapArea(pos0, pos1, rad[simplices[sIndex,0]], rad[simplices[sIndex,1]], boxSize)
        intersectArea -= computeOverlapArea(pos1, pos2, rad[simplices[sIndex,1]], rad[simplices[sIndex,2]], boxSize)
        intersectArea -= computeOverlapArea(pos1, pos0, rad[simplices[sIndex,1]], rad[simplices[sIndex,0]], boxSize)
        intersectArea -= computeOverlapArea(pos2, pos0, rad[simplices[sIndex,2]], rad[simplices[sIndex,0]], boxSize)
        simplexDensity[sIndex] = intersectArea / triangleArea
        simplexArea[sIndex] = triangleArea
    # translate simplex density into local density for particles
    return simplexDensity, simplexArea

def computeOverlapArea(pos1, pos2, rad1, rad2, boxSize):
    distance = np.linalg.norm(pbcDistance(pos1, pos2, boxSize))
    overlap = 1 - distance / (rad1 + rad2)
    if(overlap > 0):
        angle = np.arccos((rad2**2 + distance**2 - rad1**2) / (2*rad2*distance))
        return angle * rad2**2 - 0.5 * rad2**2 * np.sin(2*angle)
    else:
        return 0

def hyperbolicTan(x, a, b, x0, w):
    return 0.5*(a+b) - 0.5*(a-b)*np.tanh(2*(x-x0)/w)

def computeInterfaceWidth(x, y):
    failed = False
    try:
        popt, pcov = curve_fit(hyperbolicTan, x, y, bounds=([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
        return np.zeros(3)
    if(failed == False):
        return popt[3]

def labelDelaunaySimplices(dirLabel, simplices, denseSimplexList):
    # save dense simplices with 3, 2 and 1 dilute neighboring simplices
    dense3dilute = np.zeros(denseSimplexList.shape[0])
    dense2dilute = np.zeros(denseSimplexList.shape[0])
    dense1dilute = np.zeros(denseSimplexList.shape[0])
    dense3diluteAllNeighbors = np.zeros(denseSimplexList.shape[0])
    dense2diluteAllNeighbors = np.zeros(denseSimplexList.shape[0])
    dense1diluteAllNeighbors = np.zeros(denseSimplexList.shape[0])
    dense3diluteNeighbors = np.zeros(denseSimplexList.shape[0])
    dense2diluteNeighbors = np.zeros(denseSimplexList.shape[0])
    dense1diluteNeighbors = np.zeros(denseSimplexList.shape[0])
    for i in range(denseSimplexList.shape[0]):
        if(denseSimplexList[i] == 1):
            indices = findNeighborSimplices(simplices, i)
            allIndices = findAllNeighborSimplices(simplices, i)
            if(np.sum(denseSimplexList[indices]) == 0): # all are dilute
                dense3dilute[i] = 1
                dense3diluteAllNeighbors[allIndices] = 1
                dense3diluteNeighbors[indices[denseSimplexList[indices]==0]] = 1
            elif(np.sum(denseSimplexList[indices]) == 1): # two are dilute
                dense2dilute[i] = 1
                dense2diluteAllNeighbors[allIndices] = 1
                dense2diluteNeighbors[indices[denseSimplexList[indices]==0]] = 1
            elif(np.sum(denseSimplexList[indices]) == 2): # one is dilute
                dense1dilute[i] = 1
                dense1diluteAllNeighbors[allIndices] = 1
                dense1diluteNeighbors[indices[denseSimplexList[indices]==0]] = 1
    np.savetxt(dirLabel + "/dense3dilute.dat", dense3dilute)
    np.savetxt(dirLabel + "/dense2dilute.dat", dense2dilute)
    np.savetxt(dirLabel + "/dense1dilute.dat", dense1dilute)
    np.savetxt(dirLabel + "/dense3diluteAllNeighbors.dat", dense3diluteAllNeighbors)
    np.savetxt(dirLabel + "/dense2diluteAllNeighbors.dat", dense2diluteAllNeighbors)
    np.savetxt(dirLabel + "/dense1diluteAllNeighbors.dat", dense1diluteAllNeighbors)
    np.savetxt(dirLabel + "/dense3diluteNeighbors.dat", dense3diluteNeighbors)
    np.savetxt(dirLabel + "/dense2diluteNeighbors.dat", dense2diluteNeighbors)
    np.savetxt(dirLabel + "/dense1diluteNeighbors.dat", dense1diluteNeighbors)
    # save dilute simplices with 3, 2 and 1 dense neighboring simplices
    dilute3dense = np.zeros(denseSimplexList.shape[0])
    dilute2dense = np.zeros(denseSimplexList.shape[0])
    dilute1dense = np.zeros(denseSimplexList.shape[0])
    dilute3denseAllNeighbors = np.zeros(denseSimplexList.shape[0])
    dilute2denseAllNeighbors = np.zeros(denseSimplexList.shape[0])
    dilute1denseAllNeighbors = np.zeros(denseSimplexList.shape[0])
    dilute3denseNeighbors = np.zeros(denseSimplexList.shape[0])
    dilute2denseNeighbors = np.zeros(denseSimplexList.shape[0])
    dilute1denseNeighbors = np.zeros(denseSimplexList.shape[0])
    for i in range(denseSimplexList.shape[0]):
        if(denseSimplexList[i] == 0):
            indices = findNeighborSimplices(simplices, i)
            allIndices = findAllNeighborSimplices(simplices, i)
            if(np.sum(denseSimplexList[indices]) == 3): # all are dense
                dilute3dense[i] = 1
                dilute3denseAllNeighbors[allIndices] = 1
                dilute3denseNeighbors[indices[denseSimplexList[indices]==1]] = 1
            elif(np.sum(denseSimplexList[indices]) == 2): # two are dense
                dilute2dense[i] = 1
                dilute2denseAllNeighbors[allIndices] = 1
                dilute2denseNeighbors[indices[denseSimplexList[indices]==1]] = 1
            elif(np.sum(denseSimplexList[indices]) == 1): # one is dense
                dilute1dense[i] = 1
                dilute1denseAllNeighbors[allIndices] = 1
                dilute1denseNeighbors[indices[denseSimplexList[indices]==1]] = 1
    np.savetxt(dirLabel + "/dilute3dense.dat", dilute3dense)
    np.savetxt(dirLabel + "/dilute2dense.dat", dilute2dense)
    np.savetxt(dirLabel + "/dilute1dense.dat", dilute1dense)
    np.savetxt(dirLabel + "/dilute3denseAllNeighbors.dat", dilute3denseAllNeighbors)
    np.savetxt(dirLabel + "/dilute2denseAllNeighbors.dat", dilute2denseAllNeighbors)
    np.savetxt(dirLabel + "/dilute1denseAllNeighbors.dat", dilute1denseAllNeighbors)
    np.savetxt(dirLabel + "/dilute3denseNeighbors.dat", dilute3denseNeighbors)
    np.savetxt(dirLabel + "/dilute2denseNeighbors.dat", dilute2denseNeighbors)
    np.savetxt(dirLabel + "/dilute1denseNeighbors.dat", dilute1denseNeighbors)

def applyParticleFilters(contacts, denseList, simplices, denseSimplexList):
    numParticles = denseList.shape[0]
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
    return denseList, denseSimplexList

############################# Local density analysis ###########################
def computeLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea):
    density = 0
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        # remove the overlaps from the particle area
                        overlapArea = 0
                        for c in contacts[pId, np.argwhere(contacts[pId]!=-1)[:,0]]:
                            overlapArea += computeOverlapArea(pos[pId], pos[c], rad[pId], rad[c], boxSize)
                        localArea[x, y] += (np.pi*rad[pId]**2 - overlapArea)
                        density += (np.pi*rad[pId]**2 - overlapArea)
    return density

def computeLocalAreaBins(pos, rad, contacts, boxSize, bins, localArea):
    density = 0
    for pId in range(pos.shape[0]):
        for x in range(bins.shape[0]-1):
            if(pos[pId,0] > bins[x] and pos[pId,0] <= bins[x+1]):
                # remove the overlaps from the particle area
                overlapArea = 0
                for c in contacts[pId, np.argwhere(contacts[pId]!=-1)[:,0]]:
                    overlapArea += computeOverlapArea(pos[pId], pos[c], rad[pId], rad[c], boxSize)
                localArea[x] += (np.pi*rad[pId]**2 - overlapArea)
                density += (np.pi*rad[pId]**2 - overlapArea)
    return density

def computeWeightedLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea, cutoff):
    density = 0
    localWeight = np.zeros((xbin.shape[0]-1, ybin.shape[0]-1))
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        node = np.array([(xbin[x+1]+xbin[x])/2, (ybin[y+1]+ybin[y])/2])
                        distance = np.linalg.norm(pbcDistance(pos[pId], node, boxSize))
                        weight = np.exp(-cutoff**2 / (cutoff**2 - distance**2))
                        # remove the overlaps from the particle area
                        overlapArea = 0
                        for c in contacts[pId, np.argwhere(contacts[pId]!=-1)[:,0]]:
                            overlapArea += computeOverlapArea(pos[pId], pos[c], rad[pId], rad[c], boxSize)
                        localArea[x, y] += (np.pi*rad[pId]**2 - overlapArea) * weight
                        localWeight[x, y] += weight
                        density += (np.pi*rad[pId]**2 - overlapArea)
    localArea /= localWeight
    return density

def computeLocalVoronoiDensityGrid(pos, rad, contacts, boxSize, voroArea, xbin, ybin):
    density = 0
    localArea = np.zeros((xbin.shape[0]-1, ybin.shape[0]-1, 2))
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        # remove the overlaps from the particle area
                        overlapArea = 0
                        for c in contacts[pId, np.argwhere(contacts[pId]!=-1)[:,0]]:
                            overlapArea += computeOverlapArea(pos[pId], pos[c], rad[pId], rad[c], boxSize)
                        localArea[x, y, 0] += (np.pi * rad[pId]**2 - overlapArea)
                        localArea[x, y, 1] += voroArea[pId]
                        density += (np.pi * rad[pId]**2 - overlapArea)
    for x in range(xbin.shape[0]-1):
        for y in range(ybin.shape[0]-1):
            if(localArea[x,y,1] != 0):
                localArea[x,y,0] /= localArea[x,y,1]
    return localArea[:,:,0], density

def computeLocalDelaunayDensityGrid(simplexPos, simplexDensity, xbin, ybin):
    localDensity = np.zeros((xbin.shape[0]-1, ybin.shape[0]-1, 2))
    for sId in range(simplexDensity.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(simplexPos[sId,0] > xbin[x] and simplexPos[sId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(simplexPos[sId,1] > ybin[y] and simplexPos[sId,1] <= ybin[y+1]):
                        localDensity[x, y, 0] += simplexDensity[sId]
                        localDensity[x, y, 1] += 1
    for x in range(xbin.shape[0]-1):
        for y in range(ybin.shape[0]-1):
            if(localDensity[x,y,1] != 0):
                localDensity[x,y,0] /= localDensity[x,y,1]
    return localDensity[:,:,0]

def computeLocalAreaAndNumberGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea, localNumber):
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        # remove the overlaps from the particle area
                        overlapArea = 0
                        for c in contacts[pId, np.argwhere(contacts[pId]!=-1)[:,0]]:
                            overlapArea += computeOverlapArea(pos[pId], pos[c], rad[pId], rad[c], boxSize)
                        localArea[x, y] += (np.pi*rad[pId]**2 - overlapArea)
                        localNumber[x, y] += 1

def computeLocalTempGrid(pos, vel, xbin, ybin, localTemp): #this works only for 2d
    counts = np.zeros((localTemp.shape[0], localTemp.shape[1]))
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        localTemp[x, y] += np.linalg.norm(vel[pId])**2
                        counts[x, y] += 1
    localTemp[localTemp>0] /= counts[localTemp>0]*2

################################ DB clustering #################################
def getDBClusterLabels(pos, boxSize, eps, min_samples = 2, denseList = np.empty(0)):
    if(denseList.shape[0] > 0):
        start = time.time()
        #distance = computeDistances(pos[denseList==1], boxSize)
        #db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pos[denseList==1])
        end = time.time()
        #print("elapsed time:", end - start)
        labels = db.labels_
        return labels
    else:
        return np.zeros(pos.shape[0])

def computeSimplexPos(simplices, pos):
    simplexPos = np.zeros((simplices.shape[0],2))
    for sIndex in range(simplices.shape[0]):
        simplexPos[sIndex] = np.mean(pos[simplices[sIndex]], axis=0)
    return simplexPos

############################### read from files ################################
def getStepList(numFrames, firstStep, stepFreq):
    maxStep = int(firstStep + stepFreq * numFrames)
    stepList = np.arange(firstStep, maxStep, stepFreq, dtype=int)
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    return stepList

def getLogSpacedStepList(minDecade=5, maxDecade=9):
    stepList = np.empty(0, dtype=np.int64)
    stepList = np.append(stepList, 0)
    for i in range(minDecade-1, maxDecade-2):
        stepList = np.append(stepList, np.linspace(10**i, 10**(i+1),10)[:-1])
    stepList = np.append(stepList, np.linspace(10**(maxDecade-2), 10**(maxDecade-1),31))
    return stepList.astype(np.int64)

def getDirectories(dirName):
    listDir = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "dynamics" and dir != "affine" and dir != "extend1e-02" and dir != "augmented" and dir!="delaunayLabels" and dir!="denseFilterDelaunayLabels" and dir!="dense2FilterDelaunayLabels" and dir!="dense3FilterDelaunayLabels")):
            listDir.append(dir)
    return listDir

def getOrderedDirectories(dirName):
    listDir = []
    listScalar = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "dynamics" and dir != "affine" and dir != "extend1e-02" and dir != "augmented" and dir!="delaunayLabels" and dir!="denseFilterDelaunayLabels" and dir!="dense2FilterDelaunayLabels" and dir!="dense3FilterDelaunayLabels")):
            listDir.append(dir)
            listScalar.append(dir.strip('t'))
    listScalar = np.array(listScalar, dtype=np.int64)
    listDir = np.array(listDir)
    listDir = listDir[np.argsort(listScalar)]
    listScalar = np.sort(listScalar)
    return listDir, listScalar

def getOrderedStrainDirectories(dirName):
    listDir = []
    listScalar = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "dynamics" and dir != "affine" and dir != "extend1e-02" and dir != "augmented" and dir!="delaunayLabels" and dir!="denseFilterDelaunayLabels" and dir!="dense2FilterDelaunayLabels" and dir!="dense3FilterDelaunayLabels")):
            listDir.append(dir)
            listScalar.append(dir.strip('strain'))
    listScalar = np.array(listScalar, dtype=np.float64)
    listDir = np.array(listDir)
    listDir = listDir[np.argsort(listScalar)]
    listScalar = np.sort(listScalar)
    return listDir, listScalar

def getOrderedPhiDirectories(dirName):
    listDir = []
    listScalar = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "initial")):
            listDir.append(dir)
            listScalar.append(float(dir))
    listScalar = np.array(listScalar, dtype=np.float64)
    listDir = np.array(listDir)
    listDir = listDir[np.argsort(listScalar)]
    listScalar = np.sort(listScalar)
    return listDir, listScalar

def getFrontBackStrainDirectories(dirName):
    listDir = []
    listScalar = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir[:4] != 'back')):
            listDir.append(dir)
            listScalar.append(dir.strip('front'))
    listScalar = np.array(listScalar, dtype=np.float64)
    listDir = np.array(listDir)
    frontDir = listDir[np.argsort(listScalar)]
    frontScalar = np.sort(listScalar)
    listDir = []
    listScalar = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir[:5] != 'front')):
            listDir.append(dir)
            listScalar.append(dir.strip('back'))
    listScalar = np.array(listScalar, dtype=np.float64)
    listDir = np.array(listDir)
    backDir = listDir[np.flip(np.argsort(listScalar))]
    backScalar = np.flip(np.sort(listScalar))
    listDir = np.concatenate((frontDir, backDir))
    listScalar = np.concatenate((frontScalar, backScalar))
    return listDir, listScalar

def getShearDirectories(dirName):
    listDir = []
    listScalar = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            listDir.append(dir)
            listScalar.append(dir.strip('strain'))
    listScalar = np.array(listScalar, dtype=np.float64)
    listDir = np.array(listDir)
    listDir = listDir[np.argsort(listScalar)]
    listScalar = np.sort(listScalar)
    return listDir, listScalar

def getDirSep(dirName, fileName):
    if(os.path.exists(dirName + os.sep + fileName + ".dat")):
        return "/"
    else:
        return "/../"

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

def saveInParams(dirName, paramName, paramString):
    saved = False
    newFileContent = ""
    with open(dirName + os.sep + "params.dat") as file:
        for line in file:
            name, scalarString = line.strip().split("\t")
            if(name == paramName):
                newLine = paramName + '\t' + paramString + '\n'
                saved = True
                newFileContent += newLine
            else:
                newFileContent += line
    if(saved==False):
        newFileContent += paramName + '\t' + paramString + '\n'
    file = open(dirName + os.sep + "params.dat", "w")
    file.write(newFileContent)
    file.close()

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

def checkPair(dirName, index1, index2):
    if(os.path.exists(dirName + os.sep + "t" + str(index1))):
        if(os.path.exists(dirName + os.sep + "t" + str(index2))):
            return True
    return False

def readParticlePair(dirName, index1, index2):
    pPos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particlePos.dat"))
    pPos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particlePos.dat"))
    return pPos1, pPos2

def readVelPair(dirName, index1, index2):
    pVel1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particleVel.dat"))
    pVel2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particleVel.dat"))
    return pVel1, pVel2

def readPair(dirName, index1, index2):
    pPos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particlePos.dat"))
    pos1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "positions.dat"))
    pPos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particlePos.dat"))
    pos2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "positions.dat"))
    return pPos1, pos1, pPos2, pos2

def readDirectorPair(dirName, index1, index2):
    pAngle1 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "particleAngles.dat"))
    pAngle2 = np.array(np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "particleAngles.dat"))
    pDir1 = np.array([np.cos(pAngle1), np.sin(pAngle1)]).T
    pDir2 = np.array([np.cos(pAngle2), np.sin(pAngle2)]).T
    return pDir1, pDir2

def readDenseListPair(dirName, index1, index2):
    if(os.path.exists(dirName + os.sep + "t" + str(index1) + os.sep + "delaunayList.dat")):
        denseList1 = np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "delaunayList.dat")
    else:
        denseList1,_ = cluster.computeDelaunayCluster(dirName + os.sep + "t" + str(index1))
    if(os.path.exists(dirName + os.sep + "t" + str(index2) + os.sep + "delaunayList.dat")):
        denseList2 = np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "delaunayList.dat")
    else:
        denseList2,_ = cluster.computeDelaunayCluster(dirName + os.sep + "t" + str(index2))
    return denseList1, denseList2

############################### Packing tools ##################################
def getPBCPositions(fileName, boxSize):
    pos = np.array(np.loadtxt(fileName), dtype=np.float64)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    #pos -= np.floor(pos/boxSize)*boxSize
    return pos

def getLEPBCPositions(fileName, boxSize, strain):
    pos = np.array(np.loadtxt(fileName), dtype=np.float64)
    shifty = np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    pos[:,0] -= shifty*strain
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= shifty
    return pos

def getVelocityAngles(vel):
    return np.arctan2(vel[:,1], vel[:,0])

def isOutsideWhichWall(pos, boxSize):
    isOutsideWhichWall = np.zeros((2,2))
    if(pos[0] < 0):
        isOutsideWhichWall[0,0] = 1
    elif(pos[0] > boxSize[0]):
        isOutsideWhichWall[0,1] = 1
    if(pos[1] < 0):
        isOutsideWhichWall[1,0] = 1
    elif(pos[1] > boxSize[1]):
        isOutsideWhichWall[1,1] = 1
    return isOutsideWhichWall

def getDropletPosRad(pos, rad, boxSize, labelList):
    # this positions are augmented to avoid wrong computation of droplet center
    # of mass for droplets sitting on a boundary
    dropletPos = np.zeros((np.unique(labelList).shape[0]-1,2)) # discard label = -1
    dropletRad = np.zeros(np.unique(labelList).shape[0]-1)
    labelIndex = 0
    for label in np.unique(labelList):
        if(label!=-1):
            # check that all positions of particles in the droplet are inside the box
            # if not, get the indices and compute the droplet center of mass position
            # from the augmented positions
            dropletIndices = np.argwhere(labelList==label)[:,0]
            dropletRad[labelIndex] = np.sqrt(np.sum(rad[dropletIndices]**2)) # pi cancels out
            dropletPos[labelIndex] = getSingleDropletPosFromReference(pos, dropletRad[labelIndex], boxSize, dropletIndices)
            labelIndex += 1
    return dropletPos, dropletRad

def getSingleDropletPosFromReference(pos, dropletRad, boxSize, dropletIndices):
    thisDropletPos = np.mean(pos[dropletIndices], axis=0)
    dropletPosList = pos[dropletIndices]
    shift = False
    for d in range(2): # check if the droplet is sitting on the boundary
        shiftPlusL = False
        shiftMinusL = False
        nearLNum = 0
        near0Num = 0
        for index in dropletIndices:
            if(pos[index,d] > (boxSize[d] - dropletRad)):
                nearLNum += 1
            elif(pos[index,d] < dropletRad):
                near0Num += 1
        if(nearLNum != 0 and near0Num != 0):
            if(nearLNum >= near0Num):
                shiftPlusL = True
                shift = True
            else:
                shiftMinusL = True
                shift = True
        if(shiftPlusL == True):
            #print(label, "shiftPlusL", d)
            particleId = 0
            for index in dropletIndices:
                if(pos[index,d] < 2*dropletRad):
                    dropletPosList[particleId,d] = pos[index,d] + boxSize[d]
                else:
                    dropletPosList[particleId,d] = pos[index,d]
                particleId += 1
        elif(shiftMinusL == True):
            #print(label, "shiftMinusL", d)
            particleId = 0
            for index in dropletIndices:
                if(pos[index,d] > (boxSize[d] - 2*dropletRad)):
                    dropletPosList[particleId,d] = pos[index,d] - boxSize[d]
                else:
                    dropletPosList[particleId,d] = pos[index,d]
                particleId += 1
        else:
            #print(label, "no shift", d)
            dropletPosList[:,d] = pos[dropletIndices,d]
    if(shift == True):
        thisDropletPos = np.mean(dropletPosList, axis=0)
    return thisDropletPos

def centerPositions(pos, rad, boxSize, denseList=np.array([])):
    # first check if it needs to be shifted
    if(denseList.shape[0] != 0):
        centerOfMass = np.mean(pos[denseList==1], axis=0)
    else:
        centerOfMass = np.mean(pos, axis=0)
    if(centerOfMass[0] < 0.5):
        pos[:,0] += (0.5 - centerOfMass[0])
    else:
        pos[:,0] -= (centerOfMass[0] - 0.5)
    if(centerOfMass[1] < 0.5):
        pos[:,1] += (0.5 - centerOfMass[1])
    else:
        pos[:,1] -= (centerOfMass[1] - 0.5)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    # check if the dense region is in the center otherwise shift
    centerRad = rad[(pos[:,0]-centerOfMass[0])<0.2]
    centerPos = pos[(pos[:,0]-centerOfMass[0])<0.2]
    centerRad = centerRad[(centerPos[:,0]-centerOfMass[0])>-0.2]
    centerPos = centerPos[(centerPos[:,0]-centerOfMass[0])>-0.2]
    centerRad = centerRad[(centerPos[:,1]-centerOfMass[1])<0.2]
    centerPos = centerPos[(centerPos[:,1]-centerOfMass[1])<0.2]
    centerRad = centerRad[(centerPos[:,1]-centerOfMass[1])>-0.2]
    centerPos = centerPos[(centerPos[:,1]-centerOfMass[1])>-0.2]
    centerPhi = np.sum(np.pi*centerRad**2)/(0.4**2)
    phi = np.sum(np.pi*rad**2)
    # compute density in the four sides and shift to the densest region
    if(centerPhi < phi):
        # left
        leftRad = rad[pos[:,0]<0.1]
        leftPos = pos[pos[:,0]<0.1]
        leftPhi = np.sum(np.pi*leftRad**2)/(0.1)
        # rigth
        rightRad = rad[pos[:,0]>0.9]
        rightPos = pos[pos[:,0]>0.9]
        rightPhi = np.sum(np.pi*rightRad**2)/(0.1)
        # top
        topRad = rad[pos[:,1]>0.9]
        topPos = pos[pos[:,1]>0.9]
        topPhi = np.sum(np.pi*topRad**2)/(0.1)
        # bottom
        bottomRad = rad[pos[:,1]<0.1]
        bottomPos = pos[pos[:,1]<0.1]
        bottomPhi = np.sum(np.pi*bottomRad**2)/(0.1)
        # now check where the densest part is
        if((leftPhi + rightPhi) > (topPhi + bottomPhi)):
            if(leftPhi > rightPhi):
                if((leftPhi - rightPhi) > 0.1):
                    pos[:,0] += boxSize[1]/4
                else:
                    pos[:,0] += boxSize[1]/2
                #print("shifting to the right")
            else:
                if((rightPhi - leftPhi) > 0.1):
                    pos[:,0] -= boxSize[1]/4
                else:
                    pos[:,0] -= boxSize[1]/2
                #print("shifting to the left")
            pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
        else:
            if(bottomPhi > topPhi):
                if((bottomPhi - topPhi) > 0.1):
                    pos[:,1] += boxSize[1]/4
                else:
                    pos[:,1] += boxSize[1]/2
                #print("shifting to the top")
            else:
                if((topPhi - bottomPhi) > 0.1):
                    pos[:,1] -= boxSize[1]/4
                else:
                    pos[:,1] -= boxSize[1]/2
                #print("shifting to the bottom")
            pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    return pos

def centerDroplet(pos, rad, boxSize, labels, maxLabel, nDim=2):
    centerOfMass = np.mean(pos[labels==maxLabel], axis=0)
    #print("loaded center of mass:", centerOfMass[0], centerOfMass[1])
    dropletPos = pos[labels==maxLabel]
    dropletRad = np.sqrt(np.sum(rad[labels==maxLabel]**2))
    shift = np.zeros(nDim)
    isOutside = np.zeros(nDim)
    for i in range(dropletPos.shape[0]):
        delta = pbcDistance(centerOfMass, dropletPos[i], boxSize)
        for d in range(nDim):
            if(np.abs(delta[d]) > dropletRad):
                isOutside[d] += 1
    for d in range(nDim):
        if(isOutside[d] > int(dropletPos.shape[0]*0.5)):
            shift[d] = boxSize[d]*0.25
            #print("correction shift: ", shift)
            pos = shiftPositions(pos, boxSize, shift[0], shift[1])
    centerOfMass = np.mean(pos[labels==maxLabel], axis=0)
    pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-centerOfMass[0], boxSize[1]*0.5-centerOfMass[1])
    centerOfMass = np.mean(pos[labels==maxLabel], axis=0)
    pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-centerOfMass[0], boxSize[1]*0.5-centerOfMass[1])
    return pos

def centerCOM(pos, rad, boxSize):
    center = np.mean(pos, axis=0)[0]
    height = 0.5*np.pi*np.sum(rad**2) / boxSize[1]
    isOutside = 0
    for i in range(pos.shape[0]):
        delta = np.abs(pbcDistance(center, pos[i,0], boxSize[0]))
        if(delta > height):
            isOutside += 1
    corrected = False
    if(isOutside > int(pos.shape[0]*0.5)):
        corrected = True
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5, 0)
    center = np.mean(pos, axis=0)[0]
    if(corrected==True):
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-center, 0)
    else:
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-center, 0)
        center = np.mean(pos, axis=0)[0]
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-center, 0)
    return pos

def centerCOM1(pos, rad, boxSize, num1):
    center = np.mean(pos[:num1], axis=0)[0]
    height = 0.5*np.pi*np.sum(rad[:num1]**2) / boxSize[1]
    isOutside = 0
    for i in range(pos[:num1].shape[0]):
        delta = np.abs(pbcDistance(center, pos[i,0], boxSize[0]))
        if(delta > height):
            isOutside += 1
    corrected = False
    if(isOutside > int(pos[:num1].shape[0]*0.5)):
        corrected = True
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5, 0)
    center = np.mean(pos[:num1], axis=0)[0]
    if(corrected==True):
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-center, 0)
    else:
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-center, 0)
        center = np.mean(pos[:num1], axis=0)[0]
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-center, 0)
    return pos

def shiftCOM(pos, boxSize):
    center = np.mean(pos, axis=0)
    pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-center[0], boxSize[1]*0.5-center[1])
    return pos

def centerSlab(pos, rad, boxSize, labels, maxLabel):
    slabCenter = np.mean(pos[labels==maxLabel], axis=0)[0]
    #print("loaded center of mass:", centerOfMass[0], centerOfMass[1])
    slabPos = pos[labels==maxLabel]
    slabHeight = 0.5*np.pi*np.sum(rad[labels==maxLabel]**2) / boxSize[1]
    isOutside = 0
    for i in range(slabPos.shape[0]):
        delta = np.abs(pbcDistance(slabCenter, slabPos[i,0], boxSize[0]))
        if(delta > slabHeight):
            isOutside += 1
    corrected = False
    if(isOutside > int(slabPos.shape[0]*0.5)):
        corrected = True
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5, 0)
    slabCenter = np.mean(pos[labels==maxLabel], axis=0)[0]
    if(corrected==True):
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-slabCenter, 0)
    else:
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-slabCenter, 0)
        slabCenter = np.mean(pos[labels==maxLabel], axis=0)[0]
        pos = shiftPositions(pos, boxSize, boxSize[0]*0.5-slabCenter, 0)
    #print("final center of mass:", np.mean(pos[labels==maxLabel], axis=0))
    return pos

def shiftPositions(pos, boxSize, xshift, yshift):
    pos[:,0] += xshift
    pos[:,1] += yshift
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    return pos

def getMOD2PIAngles(fileName):
    angle = np.array(np.loadtxt(fileName), dtype=np.float64)
    return np.mod(angle, 2*np.pi)

def sortBorderPos(borderPos, borderList, boxSize, checkNumber=5):
    borderAngle = np.zeros(borderPos.shape[0])
    centerOfMass = np.mean(borderPos, axis=0)
    for i in range(borderPos.shape[0]):
        delta = pbcDistance(borderPos[i], centerOfMass, boxSize)
        borderAngle[i] = np.arctan2(delta[1], delta[0])
    borderPos = borderPos[np.argsort(borderAngle)]
    # swap nearest neighbor if necessary
    for i in range(borderPos.shape[0]-1):
        # check distances with the next three border particles
        distances = []
        for j in range(checkNumber):
            nextIndex = i+j+1
            if(nextIndex > borderPos.shape[0]-1):
                nextIndex -= borderPos.shape[0]
            distances.append(np.linalg.norm(pbcDistance(borderPos[i], borderPos[nextIndex], boxSize)))
        minIndex = np.argmin(distances)
        swapIndex = i+minIndex+1
        if(swapIndex > borderPos.shape[0]-1):
            swapIndex -= borderPos.shape[0]
        if(minIndex != 0):
            # pair swap
            tempPos = borderPos[i+1]
            borderPos[i+1] = borderPos[swapIndex]
            borderPos[swapIndex] = tempPos
    return borderPos

def shiftToOrigin(dirName, dirSave):
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    print("Center of mass:", np.mean(pos, axis=0))
    center = np.mean(pos, axis=0)
    pos[:,0] -= center[0]
    pos[:,1] -= center[1]
    print("Center of mass:", np.mean(pos, axis=0))
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', pos)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    np.savetxt(dirSave + '/particleRad.dat', rad)

def relabelPacking(dirName, dirSave, nDim=2):
    # label particles to the left of the center from 0 to N/2
    # and the particles to right of the center from N/2 + 1 to N
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    for i in range(nDim):
        pos[:,i] -= np.floor(pos[:,i]/boxSize[i]) * boxSize[i]
    center = boxSize[0] * 0.5
    rightIndices = np.argwhere(pos[:,0]>center)[:,0]
    leftIndices = np.argwhere(pos[:,0]<=center)[:,0]
    leftNum = leftIndices.shape[0]
    rightNum = rightIndices.shape[0]
    print("Number of left particles:", leftNum)
    print("Number of right particles:", rightNum)
    print("Total number of particles:", leftNum + rightNum)
    newPos = np.zeros(pos.shape)
    newRad = np.zeros(rad.shape)
    newVel = np.zeros(vel.shape)
    newPos[:leftNum] = pos[leftIndices]
    newPos[leftNum:] = pos[rightIndices]
    newRad[:leftNum] = rad[leftIndices]
    newRad[leftNum:] = rad[rightIndices]
    newVel[:leftNum] = vel[leftIndices]
    newVel[leftNum:] = vel[rightIndices]
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', newPos)
    np.savetxt(dirSave + '/particleVel.dat', newVel)
    np.savetxt(dirSave + '/particleRad.dat', newRad)

def resizeHalfPacking(dirName, dirSave, leftNum, scale):
    # label particles to the left of the center from 0 to N/2
    # and the particles to right of the center from N/2 + 1 to N
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    print("Number of left particles:", leftNum)
    newRad = np.zeros(rad.shape)
    newRad[:leftNum] = rad[:leftNum]/scale #22=0.66/0.03 approximate liquid gas densities
    newRad[leftNum:] = rad[leftNum:]
    boxArea = boxSize[0]*boxSize[1]
    print("Loaded packing fraction:", np.pi*np.sum(rad**2)/boxArea)
    print("Saved packing fraction:", np.pi*np.sum(newRad**2)/boxArea)
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', pos)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    np.savetxt(dirSave + '/particleRad.dat', newRad)

def increaseDensity(dirName, dirSave, targetDensity):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    #angle = np.loadtxt(dirName + '/particleAngles.dat')
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', pos)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    #np.savetxt(dirSave + '/particleAngles.dat', angle)
    # adjust particle radii to target density
    rad = np.loadtxt(dirName + '/particleRad.dat')
    currentDensity = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    print("Current density: ", currentDensity)
    multiple = np.sqrt(targetDensity / currentDensity)
    rad *= multiple
    np.savetxt(dirSave + '/particleRad.dat', rad)
    currentDensity = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    print("Current density: ", currentDensity)

def initializeRectangle(dirName, dirSave, xratio, yratio):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    #angle = np.loadtxt(dirName + '/particleAngles.dat')
    numParticles = rad.shape[0]
    density = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    #np.savetxt(dirSave + '/particleAngles.dat', angle)
    # increase boxsize along the x direction and pbc particles in new box
    print(boxSize)
    boxSize[0] *= xratio
    boxSize[1] *= yratio
    print(boxSize)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', pos)
    # increase the particle sizes such that the density stays the same
    currentDensity = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1]) #boxSize[0] has changed
    print("Current density: ", currentDensity)
    multiple = np.sqrt(density / currentDensity)
    rad *= multiple
    currentDensity = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    print("Current density: ", currentDensity)
    np.savetxt(dirSave + '/particleRad.dat', rad)

def adjustBoxSize(dirName, dirSave, xratio, yratio, zratio):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    #angle = np.loadtxt(dirName + '/particleAngles.dat')
    numParticles = rad.shape[0]
    if(zratio != 0):
        density = np.sum(np.pi*4*rad**3/3) / (boxSize[0] * boxSize[1] * boxSize[2])
    else:
        density = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    #np.savetxt(dirSave + '/particleAngles.dat', angle)
    np.savetxt(dirSave + '/particleRad.dat', rad)
    # increase boxsize along the x direction and pbc particles in new box
    print(boxSize)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    boxSize[0] *= xratio
    boxSize[1] *= yratio
    if(zratio != 0):
        pos[:,2] -= np.floor(pos[:,2]/boxSize[2]) * boxSize[2]
        boxSize[2] *= zratio
    center = np.mean(pos[:,0])
    pos[:,0] += (0.5 * boxSize[0] - center)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    #pos = centerCOM(pos, rad, boxSize)
    print(boxSize)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', pos)
    # increase the particle sizes such that the density stays the same
    if(zratio != 0):
        density = np.sum(np.pi*4*rad**3/3) / (boxSize[0] * boxSize[1] * boxSize[2])
    else:
        density = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    print("Current density: ", density)

def scalePacking(dirName, dirSave, scale):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    #angle = np.loadtxt(dirName + '/particleAngles.dat')
    density = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    print(boxSize)
    print("Current density: ", density)
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    # scale boxsize and radii
    boxSize /= scale
    rad /= scale
    print(boxSize)
    density = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    print("Current density: ", density)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', pos)
    np.savetxt(dirSave + '/particleRad.dat', rad)
    
def setSigmaToOne(dirName, dirSave):
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    #angle = np.loadtxt(dirName + '/particleAngles.dat')
    numParticles = rad.shape[0]
    density = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    print("Loaded density", density)
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    #np.savetxt(dirSave + '/particleAngles.dat', angle)
    sigma = np.mean(rad)
    boxSize /= sigma 
    pos /= sigma
    rad /= sigma
    density = np.sum(np.pi*rad**2) / (boxSize[0] * boxSize[1])
    print("Current density", density)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', pos)
    np.savetxt(dirSave + '/particleRad.dat', rad)

def findLargestParticleCluster(rad, labels):
    uniqueLabels = np.unique(labels).astype(np.int64)
    size = np.zeros(uniqueLabels.shape[0])
    for i in range(uniqueLabels.shape[0]):
        if(uniqueLabels[i]!=-1):
            size[i] = np.sum(rad[labels==uniqueLabels[i]]**2)
    return uniqueLabels[np.argmax(size)]

def removeSmallClusters(dirName, dirSave, threshold=0.76):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    eps = 1.4 * np.mean(rad)
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    copyAngle = False
    if(os.path.exists(dirName + '/particleAngles.dat')):
        angle = np.loadtxt(dirName + '/particleAngles.dat')
        copyAngle = True
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    labels = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold=threshold)
    #print(np.unique(labels))
    # make a list of indices to keep
    if(np.unique(labels).shape[0]<2):
        print("There is only one cluster here - keep all particles")
        keepList = np.arange(0,labels.shape[0],1).astype(np.int64)
    else:
        keepList = np.zeros(labels.shape[0])
        keepList[labels==-1] = 1
        maxLabel = findLargestParticleCluster(rad, labels)
        pos = centerSlab(pos, rad, boxSize, labels, maxLabel)
        keepList[labels==maxLabel] = 1
        threshold = int(labels.shape[0] / 1000)
        if(threshold < 3):
            threshold = 3
        for label in np.unique(labels):
            if(label !=-1 and label != maxLabel):
                if(np.argwhere(labels==label).shape[0] < threshold):
                    keepList[labels==label] = 1
                    #print(label)
    np.savetxt(dirSave + '/particleRad.dat', rad[keepList==1])
    np.savetxt(dirSave + '/particlePos.dat', pos[keepList==1])
    np.savetxt(dirSave + '/particleVel.dat', vel[keepList==1])
    if(copyAngle == True):
        np.savetxt(dirSave + '/particleAngles.dat', angle[keepList==1])
    shutil.copy(dirName + 'params.dat', dirSave + '/params.dat')
    numParticles = int(keepList[keepList==1].shape[0])
    saveInParams(dirSave, "numParticles", str(numParticles))
    print("Number of particles left in the packing:", keepList[keepList==1].shape[0], "saved:", int(readFromParams(dirSave, "numParticles")))
    phi = np.sum(rad[keepList==1]**2)*np.pi/(boxSize[0]*boxSize[1])
    saveInParams(dirSave, "phi", str(phi))
    print("Packing fraction:", np.sum(rad[keepList==1]**2)*np.pi/(boxSize[0]*boxSize[1]))

def perturbInterface(dirName, dirSave, threshold=0.3, fraction=0.1):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    eps = 1.4 * np.mean(rad)
    spacing = 50*np.mean(rad)
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    angle = np.loadtxt(dirName + '/particleAngles.dat')
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particleRad.dat', rad)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    np.savetxt(dirSave + '/particleAngles.dat', angle)
    labels = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold=threshold)
    maxLabel = findLargestParticleCluster(rad, labels)
    pos = centerSlab(pos, rad, boxSize, labels, maxLabel)
    clusterPos = pos[labels==maxLabel]
    binPos = clusterPos[np.argwhere(clusterPos[:,1] > (boxSize[1]*0.7 - spacing))[:,0]]
    binPos = binPos[np.argwhere(binPos[:,1] <= (boxSize[1]*0.7 + spacing))[:,0]]
    center = np.mean(binPos, axis=0)[0] # center of dense cluster
    binDistance = (binPos[:,0] - center)
    borderMask = np.argsort(binDistance)[-3:]
    height = np.mean(binDistance[borderMask])
    print("center:", center, "height:", height)
    # find particles near the border
    movedList = np.empty(0)
    for i in range(rad.shape[0]):
        if(labels[i]==maxLabel):
            if(pos[i,1] > (boxSize[1]*0.7 - spacing) and pos[i,1] <= (boxSize[1]*0.7 + spacing)):
                if(pos[i,0] > (center + 0.5*height) and pos[i,0] <= (center + 0.6*height)):
                    pos[i,0] += fraction*height
                    movedList = np.append(movedList, i)
                elif(pos[i,0] > (center + 0.6*height) and pos[i,0] <= (center + 0.7*height)):
                    pos[i,0] += 0.8*fraction*height
                    movedList = np.append(movedList, i)
                elif(pos[i,0] > (center + 0.7*height) and pos[i,0] <= (center + 0.8*height)):
                    pos[i,0] += 0.6*fraction*height
                    movedList = np.append(movedList, i)
    movedList = movedList.astype(np.int64)
    # apply perturbation
    #pos[movedList,0] += fraction*height
    print("Number of moved particles:", movedList.shape[0])
    print("Location:", np.mean(pos[movedList,0]), np.mean(pos[movedList,1]))
    np.savetxt(dirSave + '/particlePos.dat', pos)
    movedLabel = np.zeros(rad.shape[0])
    movedLabel[movedList] = 1
    np.savetxt(dirSave + '/movedList.dat', movedLabel)

def reshufflePacking(dirName, threshold=0.3, multiple=50, width=0.3, offset=0.2):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    eps = 1.4 * np.mean(rad)
    spacing = multiple*np.mean(rad)
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    angle = np.loadtxt(dirName + '/particleAngles.dat')
    labels = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold=threshold)
    maxLabel = findLargestParticleCluster(rad, labels)
    pos = centerSlab(pos, rad, boxSize, labels, maxLabel)
    clusterPos = pos[labels==maxLabel]
    binPos = clusterPos[np.argwhere(clusterPos[:,1] > (boxSize[1]*0.7 - spacing))[:,0]]
    binPos = binPos[np.argwhere(binPos[:,1] <= (boxSize[1]*0.7 + spacing))[:,0]]
    center = np.mean(binPos, axis=0)[0] # center of dense cluster
    binDistance = (binPos[:,0] - center)
    borderMask = np.argsort(binDistance)[-3:]
    height = np.mean(binDistance[borderMask])
    print("center:", center, "height:", height)
    # find particles near the border
    movedList = np.empty(0)
    for i in range(rad.shape[0]):
        if(labels[i]==maxLabel):
            if(pos[i,1] > (boxSize[1]*0.7 - spacing) and pos[i,1] <= (boxSize[1]*0.7 + spacing)):
                if(pos[i,0] > (center + (1-offset-width)*height) and pos[i,0] <= (center + (1-offset)*height)):
                    movedList = np.append(movedList, i)
    movedList = movedList.astype(np.int64)
    numMoved = movedList.shape[0]
    movedLabel = np.zeros(rad.shape[0])
    movedLabel[movedList] = 1
    newRad = np.zeros(rad.shape)
    newPos = np.zeros(pos.shape)
    newVel = np.zeros(vel.shape)
    newAngle = np.zeros(angle.shape)
    # reshuffle lists
    newRad[:numMoved] = rad[movedLabel==1]
    newRad[numMoved:] = rad[movedLabel==0]
    newPos[:numMoved] = pos[movedLabel==1]
    newPos[numMoved:] = pos[movedLabel==0]
    newVel[:numMoved] = vel[movedLabel==1]
    newVel[numMoved:] = vel[movedLabel==0]
    newAngle[:numMoved] = angle[movedLabel==1]
    newAngle[numMoved:] = angle[movedLabel==0]
    # save files to new directory
    dirSave = dirName + "-EF" + str(numMoved)
    print(dirSave)
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particleRad.dat', newRad)
    np.savetxt(dirSave + '/particlePos.dat', newPos)
    np.savetxt(dirSave + '/particleVel.dat', newVel)
    np.savetxt(dirSave + '/particleAngle.dat', newAngle)
    np.savetxt(dirSave + '/movedLabel.dat', movedLabel)
    print("Number of particles moved to the top of the lists:", numMoved)
    print(rad[movedList][0], rad[movedList][-1])
    print(newRad[:numMoved][0], newRad[:numMoved][-1])
    print(pos[movedList][0], pos[movedList][-1])
    print(newPos[:numMoved][0], newPos[:numMoved][-1])
    print(vel[movedList][0], vel[movedList][-1])
    print(newVel[:numMoved][0], newVel[:numMoved][-1])
    print(angle[movedList][0], angle[movedList][-1])
    print(newAngle[:numMoved][0], newAngle[:numMoved][-1])

def initializeDroplet(dirName, dirSave):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    angle = np.loadtxt(dirName + '/particleAngles.dat')
    numParticles = rad.shape[0]
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particleRad.dat', rad)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    np.savetxt(dirSave + '/particleAngles.dat', angle)
    # initialize particles with random positions on the center of the box
    r1 = np.random.rand(numParticles)
    r2 = np.random.rand(numParticles)
    x = np.sqrt(-2*np.log(r1)) * np.cos(2*np.pi*r2)
    y = np.sqrt(-2*np.log(r1)) * np.sin(2*np.pi*r2)
    x *= 0.05
    y *= 0.05
    x += 0.5
    y += 0.5
    pos = np.column_stack((x, y))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    np.savetxt(dirSave + '/particlePos.dat', pos)

def getUniqueRandomList(low, high, num):
    numRandom = 0
    randomList = [0]
    while(numRandom < num-1):
        rand = np.random.randint(low, high)
        if rand not in randomList:
            randomList.append(rand)
            numRandom += 1
    randomList = np.array(randomList)
    #print(randomList.shape[0], np.unique(randomList).shape[0])
    return randomList

def removeGasParticles(dirName, numRemove):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    angle = np.loadtxt(dirName + '/particleAngles.dat')
    density = np.sum(np.pi*rad**2)
    print("Current density: ", density)
    # remove particles in the gas at random
    if(os.path.exists(dirName + os.sep + "delaunayList.dat")):
        denseList = np.loadtxt(dirName + os.sep + "delaunayList.dat")
    else:
        denseList,_ = cluster.computeDelaunayCluster(dirName, 0.7, filter=False)
    gasIndices = np.argwhere(denseList==0)[:,0]
    maxRemove = gasIndices.shape[0]
    print("Maximum number of removable particles:", maxRemove)
    if(numRemove < maxRemove):
        randomList = getUniqueRandomList(0, maxRemove, numRemove)
        removeIndices = gasIndices[randomList]
        rad = np.delete(rad, removeIndices)
        pos = np.delete(pos, removeIndices, axis=0)
        vel = np.delete(vel, removeIndices)
        angle = np.delete(angle, removeIndices)
        # save to new directory
        dirSave = dirName + os.sep + str(numRemove) + "removed"
        if(os.path.isdir(dirSave)==False):
            os.mkdir(dirSave)
        np.savetxt(dirSave + '/boxSize.dat', boxSize)
        np.savetxt(dirSave + '/particleRad.dat', rad)
        np.savetxt(dirSave + '/particlePos.dat', pos)
        np.savetxt(dirSave + '/particleVel.dat', vel)
        np.savetxt(dirSave + '/particleAngles.dat', angle)
        density = np.sum(np.pi*rad**2)
        print("Density after removing " + str(numRemove) + " particles: ", density)
    else:
        print("Please remove a number of particles smaller than", maxRemove)

def removeParticles(dirName, numRemove):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    angle = np.loadtxt(dirName + '/particleAngles.dat')
    density = np.sum(np.pi*rad**2)
    print("Current density: ", density)
    maxRemove = rad.shape[0]-1
    if(numRemove < maxRemove):
        removeIndices = getUniqueRandomList(0, maxRemove, numRemove)
        rad = np.delete(rad, removeIndices)
        pos = np.delete(pos, removeIndices, axis=0)
        vel = np.delete(vel, removeIndices)
        angle = np.delete(angle, removeIndices)
        # save to new directory
        dirSave = dirName + os.sep + str(numRemove) + "removed"
        if(os.path.isdir(dirSave)==False):
            os.mkdir(dirSave)
        np.savetxt(dirSave + '/boxSize.dat', boxSize)
        np.savetxt(dirSave + '/particleRad.dat', rad)
        np.savetxt(dirSave + '/particlePos.dat', pos)
        np.savetxt(dirSave + '/particleVel.dat', vel)
        np.savetxt(dirSave + '/particleAngles.dat', angle)
        density = np.sum(np.pi*rad**2)
        print("Density after removing " + str(numRemove) + " particles: ", density)
    else:
        print("Please remove a number of particles smaller than", maxRemove)

def convertPosToXYZ(dirName, nDim=3):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    pos[:,2] -= np.floor(pos[:,2]/boxSize[2]) * boxSize[2]
    numParticles = str(pos.shape[0])
    np.savetxt(dirName + "/particlePos.xyz", pos, header=numParticles + "\n")

def computeNoEdgeMovingAverage(data, window=10):
    halfWindow = window // 2
    movingAvg = np.copy(data)
    for i in range(halfWindow, len(data) - halfWindow):
        movingAvg[i] = np.mean(data[i - halfWindow : i + halfWindow + 1])
    return movingAvg

import numpy as np

def computeMovingAverage(data, window=10):
    """
    Compute the moving average of a time series with a specified window size,
    replacing each element with its moving average.

    Parameters:
    time_series (array-like): The input time series data.
    window_size (int): The size of the moving window.

    Returns:
    numpy.ndarray: The time series with each element replaced by its moving average.
    """
    # Ensure the time series is a NumPy array
    data = np.asarray(data)
    
    # Calculate the number of elements to consider on each side of the current element
    halfWindow = window // 2
    
    # Pad the array to handle the edges
    paddedData = np.pad(data, (halfWindow, halfWindow - 1), mode='edge')
    
    # Initialize the array for the moving average result
    movingAvg = np.zeros_like(data)
    
    # Compute the moving average for each element
    for i in range(len(data)):
        movingAvg[i] = np.mean(paddedData[i:i + window])
    
    return movingAvg


def getTensionFromEnergyTime(dirName, which='total', strainStep=5e-06, compext='ext', reverse=False, window=10):
    # read energy at initial unstrained configuration
    numParticles = readFromParams(dirName, 'numParticles')
    epsilon = readFromParams(dirName, "epsilon")
    data = np.loadtxt(dirName + "../energy.dat")
    etot0 = np.mean(data[:,-1]/(2*epsilon))*numParticles # two interfaces in periodic boundaries
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    tension = np.zeros(2)
    # read energy during deformation
    data = np.loadtxt(dirName + "/energy.dat")
    data[:,2:] /= epsilon
    data[:,-1] /= 2 # two interfaces in periodic boundaries
    if(np.sum(np.isnan(data))!=0):
        print("There are NaNs in the file")
    if(reverse == 'reverse'):
        maxStrain = 0.1
        dataLength = int((data.shape[0] - 20) / 20)
        maxLength = int(maxStrain / strainStep)
        if(maxLength < dataLength):
            diffLength = dataLength - maxLength
            #print("maxLength:", maxLength, "dataLength:", dataLength, "diffLenth:", diffLength)
            strain = np.arange(strainStep, maxStrain + strainStep, strainStep)
            strain = np.concatenate((strain, np.flip(strain)[1:diffLength]))
            #maxStrain = (int(data.shape[0] / 2) // 20) * strainStep
            #strain = np.arange(strainStep, maxStrain + strainStep, strainStep)
            #strain = np.concatenate((strain, np.flip(strain)[1:]))
        else:
            maxStrain = (data.shape[0] // 20) * strainStep
            strain = np.arange(strainStep, maxStrain + strainStep, strainStep)
    else:
        maxStrain = (data.shape[0] // 20) * strainStep
        strain = np.arange(strainStep, maxStrain + strainStep, strainStep)
    etot = data[:,-1]
    freq = 20
    every = freq
    etot = np.zeros((strain.shape[0],2))
    for i in range(0,strain.shape[0]-every,every):
        #print(strain[i])
        stepEtot = []
        for j in range(every):
            #print(i+j, strain[i+j])
            stepEtot.append(np.mean(data[(i+j)*freq:(i+j+1)*freq,-1]))
        # two interfaces
        etot[i,0] = np.mean(stepEtot)
        etot[i,1] = np.std(stepEtot)
    strain = strain[etot[:,0]!=0]
    etot = etot[etot[:,0]!=0]
    etot = np.column_stack((computeMovingAverage(etot[:,0], window), computeMovingAverage(etot[:,1], window)))
    strain = computeMovingAverage(strain, window)
    otherStrain = -strain/(1 + strain)
    if(compext == 'ext' or compext == 'ext-rev'):
        height = (1 + strain)*boxSize[1]
        width = (1 + otherStrain)*boxSize[0]
    elif(compext == 'comp' or compext == 'comp-rev'):
        height = (1 + otherStrain)*boxSize[1]
        width = (1 + strain)*boxSize[0]
    height -= boxSize[1]
    etot *= numParticles
    if(reverse == 'reverse' and maxLength < dataLength):
        halfIndex = int(etot.shape[0] / 2)
        efront = etot[:halfIndex,0]
        hfront = height[:halfIndex]
        failed = False
        try:
            popt, pcov = curve_fit(lineFit, hfront, efront)
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if not failed:
            noise = np.sqrt(np.mean((lineFit(hfront, *popt) - efront)**2))/2
            tension[1] = noise**2
            tfront = popt[1]
            erfront = np.sqrt(np.diag(pcov)[1])
        eback = etot[halfIndex:,0]
        hback = height[halfIndex:]
        failed = False
        try:
            popt, pcov = curve_fit(lineFit, hback, eback)
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if not failed:
            noise = np.sqrt(np.mean((lineFit(hback, *popt) - eback)**2))/2
            tension[1] += noise**2
            tension[1] = np.sqrt(tension[1])
            tback = popt[1]
            erback = np.sqrt(np.diag(pcov)[1])
        tension[0] = np.mean([tfront, tback])
        #tension[1] = np.sqrt(erfront**2 + erback**2)
    else:
        failed = False
        try:
            popt, pcov = curve_fit(lineFit, height[strain<0.2], etot[strain<0.2,0])
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if not failed:
            noise = np.sqrt(np.mean((lineFit(height[strain<0.2], *popt) - etot[strain<0.2,0])**2))/2
            tension[0] = popt[1]
            tension[1] = noise
    return tension

########################### Circular packing tools #############################
def cartesianToPolar(pos):
    polarPos = np.zeros((pos.shape[0],2))
    polarPos[:,0] = np.sqrt(pos[:,0] * pos[:,0] + pos[:,1] * pos[:,1])
    polarPos[:,1] = np.arctan2(pos[:,1], pos[:,0])
    return polarPos

def calcTangentialVelocity(pos, vel):
    # compute tangential components of velocity for positions in a circle with center in the origin
    angle = np.arctan2(pos[:,1], pos[:,0])
    vDotn = vel[:,0] * np.cos(angle) + vel[:,1] * np.sin(angle)
    velnorm = np.column_stack([vDotn * np.cos(angle), vDotn * np.sin(angle)])
    veltang = vel - velnorm
    return veltang, angle

def checkParticlesInCircle(pos, boxSize):
    # check that particles are inside the circle
    numOut = 0
    polarPos = cartesianToPolar(pos)
    outSide = polarPos[polarPos[:,0]>boxSize].shape[0]
    if(outSide > 0):
        for i in range(polarPos.shape[0]):
            if(polarPos[i,0] > boxSize):
                numOut += 1
                #print("Particle", i, "is outside of the circle with radius", boxSize[0], "radial distance", polarPos[i,0], polarPos[i,0] - boxSize[0])
    if(numOut > 0):
        print(numOut, "particles are outside of boundary")
    return np.argwhere(polarPos[:,0]>boxSize)[:,0]
    

if __name__ == '__main__':
    print("library for correlation function utilities")


# the formula to compute the drift-subtracted msd is
#delta = np.linalg.norm(pos1 - pos2, axis=1)
#drift = np.linalg.norm(np.mean(pos1 - pos2, axis=0)**2)
#msd = np.mean(delta**2) - drift
# equivalent to
#drift = np.mean(delta)**2
# in one dimension
#gamma2 = (1/3) * np.mean(delta**2) * np.mean(1/delta**2) - 1


#distances = np.zeros((pos.shape[0], pos.shape[0]))
#for i in range(pos.shape[0]):
#    for j in range(i):
#        delta = pbcDistance(pos[i], pos[j], boxSize)
#        distances[i,j] = np.linalg.norm(delta)
