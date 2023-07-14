'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from scipy.fft import fft, fftfreq, fft2
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
import spCorrelation as spCorr
import os

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
    for i in range(pos.shape[0]):
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
        numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
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

def getWallForces(pos, rad, boxSize):
    wallForce = np.zeros((pos.shape[0], pos.shape[1]))
    for i in range(pos.shape[0]):
        delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
        distance = np.linalg.norm(delta)
        overlap = (1 - distance / radSum)
        gradMultiple = kc * (1 - distance / radSum) / radSum
    return wallForce

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

def computeTau(data, index=2, threshold=np.exp(-1), normalized=False):
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


############################### read from files ################################
def getStepList(numFrames, firstStep, stepFreq):
    maxStep = int(firstStep + stepFreq * numFrames)
    stepList = np.arange(firstStep, maxStep, stepFreq, dtype=int)
    if(stepList.shape[0] < numFrames):
        numFrames = stepList.shape[0]
    else:
        stepList = stepList[-numFrames:]
    return stepList

def getDirectories(dirName):
    listDir = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "short" and dir != "dynamics")):
            listDir.append(dir)
    return listDir

def getOrderedDirectories(dirName):
    listDir = []
    listScalar = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir) and (dir != "short" and dir != "dynamics")):
            listDir.append(dir)
            listScalar.append(dir.strip('t'))
    listScalar = np.array(listScalar, dtype=np.int64)
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

def readFromDynParams(dirName, paramName):
    with open(dirName + os.sep + "dynParams.dat") as file:
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
    if(os.path.exists(dirName + os.sep + "t" + str(index1) + os.sep + "delaunayList!.dat")):
        denseList1 = np.loadtxt(dirName + os.sep + "t" + str(index1) + os.sep + "delaunayList.dat")
    else:
        denseList1,_ = spCorr.computeDelaunayCluster(dirName + os.sep + "t" + str(index1))
    if(os.path.exists(dirName + os.sep + "t" + str(index2) + os.sep + "delaunayList!.dat")):
        denseList2 = np.loadtxt(dirName + os.sep + "t" + str(index2) + os.sep + "delaunayList.dat")
    else:
        denseList2,_ = spCorr.computeDelaunayCluster(dirName + os.sep + "t" + str(index2))
    return denseList1, denseList2

############################### Packing tools ##################################
def getPBCPositions(fileName, boxSize):
    pos = np.array(np.loadtxt(fileName), dtype=np.float64)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    return pos

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

def shiftPositions(pos, boxSize, xshift, yshift):
    pos[:,0] += xshift
    pos[:,1] += yshift
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    return pos

def getMOD2PIAngles(fileName):
    angle = np.array(np.loadtxt(fileName), dtype=np.float64)
    return np.mod(angle, 2*np.pi)

def increaseDensity(dirName, dirSave, targetDensity):
    # load all the packing files
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    angle = np.loadtxt(dirName + '/particleAngles.dat')
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/boxSize.dat', boxSize)
    np.savetxt(dirSave + '/particlePos.dat', pos)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    np.savetxt(dirSave + '/particleAngles.dat', angle)
    # adjust particle radii to target density
    rad = np.loadtxt(dirName + '/particleRad.dat')
    currentDensity = np.sum(np.pi*rad**2)
    print("Current density: ", currentDensity)
    multiple = np.sqrt(targetDensity / currentDensity)
    rad *= multiple
    np.savetxt(dirSave + '/particleRad.dat', rad)
    currentDensity = np.sum(np.pi*rad**2)
    print("Current density: ", currentDensity)

def initializeRectangle(dirName, dirSave):
    # load all the packing files
    numParticles = int(readFromParams(dirName, "numParticles"))
    boxSize = np.loadtxt(dirName + '/boxSize.dat')
    pos = np.loadtxt(dirName + '/particlePos.dat')
    rad = np.loadtxt(dirName + '/particleRad.dat')
    vel = np.loadtxt(dirName + '/particleVel.dat')
    angle = np.loadtxt(dirName + '/particleAngles.dat')
    density = np.sum(np.pi*rad**2)
    # save unchanged files to new directory
    if(os.path.isdir(dirSave)==False):
        os.mkdir(dirSave)
    np.savetxt(dirSave + '/particleVel.dat', vel)
    np.savetxt(dirSave + '/particleAngles.dat', angle)
    # increase boxsize along the x direction and pbc particles in new box
    boxSize[0] *= 2
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

############################ Clustering analysis ###############################
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

############################### Delaunay analysis ##############################
def augmentPacking(pos, rad, fraction=0.05):
    # augment packing by copying a fraction of the particles around the walls
    Lx = np.array([1,0])
    Ly = np.array([0,1])
    leftPos = pos[pos[:,0]<fraction]
    leftRad = rad[pos[:,0]<fraction]
    leftIndices = np.argwhere(pos[:,0]<fraction)[:,0]
    rightPos = pos[pos[:,0]>(1-fraction)]
    rightRad = rad[pos[:,0]>(1-fraction)]
    rightIndices = np.argwhere(pos[:,0]>(1-fraction))[:,0]
    bottomPos =  pos[pos[:,1]<fraction]
    bottomRad =  rad[pos[:,1]<fraction]
    bottomIndices = np.argwhere(pos[:,1]<fraction)[:,0]
    topPos = pos[pos[:,1]>(1-fraction)]
    topRad = rad[pos[:,1]>(1-fraction)]
    topIndices = np.argwhere(pos[:,1]>(1-fraction))[:,0]
    bottomLeftPos = leftPos[leftPos[:,1]<fraction]
    bottomLeftRad = leftRad[leftPos[:,1]<fraction]
    bottomLeftIndices = leftIndices[np.argwhere(leftPos[:,1]<fraction)[:,0]]
    bottomRightPos = rightPos[rightPos[:,1]<fraction]
    bottomRightRad = rightRad[rightPos[:,1]<fraction]
    bottomRightIndices = rightIndices[np.argwhere(rightPos[:,1]<fraction)[:,0]]
    topLeftPos = leftPos[leftPos[:,1]>(1-fraction)]
    topLeftRad = leftRad[leftPos[:,1]>(1-fraction)]
    topLeftIndices = leftIndices[np.argwhere(leftPos[:,1]>(1-fraction))[:,0]]
    topRightPos = rightPos[rightPos[:,1]>(1-fraction)]
    topRightRad = rightRad[rightPos[:,1]>(1-fraction)]
    topRightIndices = rightIndices[np.argwhere(rightPos[:,1]>(1-fraction))[:,0]]
    newPos = np.vstack([pos, leftPos + Lx, rightPos - Lx, bottomPos + Ly, topPos - Ly, bottomLeftPos + Lx + Ly, bottomRightPos - Lx + Ly, topLeftPos + Lx - Ly, topRightPos - Lx - Ly])
    newRad = np.concatenate((rad, leftRad, rightRad, bottomRad, topRad, bottomLeftRad, bottomRightRad, topLeftRad, topRightRad))
    newIndices = np.concatenate((np.arange(0,rad.shape[0],1), leftIndices, rightIndices, bottomIndices, topIndices, bottomLeftIndices, bottomRightIndices, topLeftIndices, topRightIndices))
    return newPos, newRad, newIndices

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
    newPos, newRad, newIndices = augmentPacking(pos, rad, 0.5)
    delaunay = Delaunay(newPos)
    insideIndex = getInsideBoxDelaunaySimplices(delaunay.simplices, newPos, boxSize)
    return wrapSimplicesAroundBox(delaunay.simplices[insideIndex==1], newIndices, rad.shape[0])

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
        slope = (pos1[1] - pos0[1]) / (pos1[0] - pos0[0])
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

def computeIntersectionArea(pos0, pos1, pos2, sigma, boxSize):
    # define reference frame to simplify projection formula
    pos2 = pbcDistance(pos2, pos1, boxSize)
    pos0 = pbcDistance(pos0, pos1, boxSize)
    pos1 = np.zeros(pos1.shape[0])
    # full formula is: np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 + intercept**2 - 2*intercept*pos2[1] - 2*slope*pos2[0]*pos2[1] + 2*slope*intercept*pos2[1]) / (1 + slope**2))
    slope = pbcDistance(pos1[1],pos0[1],boxSize[1]) / pbcDistance(pos1[0],pos0[0],boxSize[0])
    intercept = pos0[1] - pos0[0] * slope
    # length of segment from point to projection
    projLength = np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 - 2*slope*pos2[0]*pos2[1]) / (1 + slope**2))
    #projLength = np.sqrt((slope**2 * pos2[0]**2 + pos2[1]**2 + intercept**2 - 2*intercept*pos2[1] - 2*slope*pos2[0]*pos2[1] + 2*slope*intercept*pos2[1]) / (1 + slope**2))
    theta = np.arcsin(projLength / np.linalg.norm(pos2))
    intersectArea = 0.5 * sigma**2 * theta
    return projLength, intersectArea

def checkSegmentArea(projLength, sigma):
    if(projLength < sigma):
        smallTheta = np.arccos(projLength/sigma)
        smallTheta /= 2
        return smallTheta * sigma**2 - 0.5 * sigma**2 * np.sin(2*smallTheta)
    else:
        return 0

def findOppositeSimplexIndex(simplices, sIndex, indexA, indexB):
    # find simplex where the intersection is and add segmentArea to it
    indexList = np.intersect1d(np.argwhere(simplices==indexA)[:,0], np.argwhere(simplices==indexB)[:,0])
    # remove sIndex from indexList
    oppositeIndex = np.setdiff1d(indexList, np.array([sIndex]))
    return oppositeIndex

def isSimplexNearWall(pIndexList, pos, rad, boxSize):
    isSimplexNearWall = False
    for pIndex in pIndexList:
        isWall, _ = isNearWall(pos[pIndex], 2*rad[pIndex], boxSize)
        if(isWall == True):
            isSimplexNearWall = True
    return isSimplexNearWall

def computeTriangleArea(pos0, pos1, pos2, boxSize):
    #pos2 = pbcDistance(pos2, pos1, boxSize)
    #pos0 = pbcDistance(pos0, pos1, boxSize)
    #pos1 = np.zeros(pos1.shape[0])
    #return 0.5 * np.abs(pos0[0]*(pos1[1] - pos2[1]) + pos1[0]*(pos2[1] - pos0[1]) + pos2[0]*(pos0[1] - pos1[1]))
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
        pos0 = pos[simplices[sIndex,0]]
        pos1 = pos[simplices[sIndex,1]]
        pos2 = pos[simplices[sIndex,2]]
        # compute area of the triangle
        simplexArea[sIndex] = computeTriangleArea(pos0, pos1, pos2, boxSize)
        # compute the three areas of the intersecating circles
        # first compute projection distance for each vertex in the simplex and then check if the intersection is all inside the simplex
        # if not, remove the external segment from the intersection area and add it to the simplex where the segment is contained
        # first vertex
        projLength, intersectArea1 = computeIntersectionArea(pos0, pos1, pos2, rad[simplices[sIndex,1]], boxSize)
        segmentArea2 = checkSegmentArea(projLength, rad[simplices[sIndex,2]])
        # second vertex
        projLength, intersectArea2 = computeIntersectionArea(pos1, pos2, pos0, rad[simplices[sIndex,2]], boxSize)
        segmentArea0 = checkSegmentArea(projLength, rad[simplices[sIndex,0]])
        # third vertex
        projLength, intersectArea0 = computeIntersectionArea(pos2, pos0, pos1, rad[simplices[sIndex,0]], boxSize)
        segmentArea1 = checkSegmentArea(projLength, rad[simplices[sIndex,1]])
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
    simplexDensity = occupiedArea / simplexArea
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

############################# Local density analysis ###########################
def computeLocalDensityGrid(pos, rad, contacts, boxSize, localSquare, xbin, ybin):
    localArea = np.zeros((xbin.shape[0]-1, ybin.shape[0]-1))
    for pId in range(pos.shape[0]):
        for x in range(xbin.shape[0]-1):
            if(pos[pId,0] > xbin[x] and pos[pId,0] <= xbin[x+1]):
                for y in range(ybin.shape[0]-1):
                    if(pos[pId,1] > ybin[y] and pos[pId,1] <= ybin[y+1]):
                        localArea[x, y] += np.pi*rad[pId]**2
                        # remove the overlaps from the particle area
                        overlapArea = 0
                        for c in contacts[pId, np.argwhere(contacts[pId]!=-1)[:,0]]:
                            overlapArea += computeOverlapArea(pos[pId], pos[c], rad[pId], rad[c], boxSize)
                        localArea[x, y] += (np.pi * rad[pId]**2 - overlapArea)
    return localArea / localSquare

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