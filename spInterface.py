'''
Created by Francesco
14 July 2023
'''
#functions and script to compute cluster correlations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
import sys
import os
import time
import utils
import utilsPlot as uplot
import spCluster as cluster
from scipy.stats import norm
from scipy.spatial import cKDTree
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from numba import njit

def computeClusterTemperatureVSTime(dirName, threshold=0.3, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    temp = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + "/particleList.dat")):
            cluster.computeDelaunayCluster(dirSample, threshold=threshold)
        particleList = np.loadtxt(dirSample + "/particleList.dat")
        denseList = particleList[:,0]
        vel = np.loadtxt(dirSample + os.sep + "particleVel.dat")
        velNorm = np.linalg.norm(vel, axis=1)
        temp[d,0] = np.mean(velNorm**2)/2
        temp[d,1] = np.mean(velNorm[denseList==1]**2)/2
        temp[d,2] = np.mean(velNorm[denseList==0]**2)/2
    print("temperature:", np.mean(temp[:,0]))
    print("liquid:", np.mean(temp[:,1]))
    print("vapor:", np.mean(temp[:,2]))
    np.savetxt(dirName + os.sep + "clusterTemperature.dat", temp)
    return temp

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
def computePressureVSTime(dirName, dirSpacing=1, nDim=2, bound=False):
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
    uplot.plotCorrelation(strainList, bulk, color='k', ylabel='$\\sigma$', xlabel='$\\gamma$')
    uplot.plotCorrelation(strainList, shear, color='k', ylabel='$\\sigma$', xlabel='$\\gamma$')
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
            cluster.computeDelaunayCluster(dirSample, threshold=0.76, save='save')
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
            cluster.computeDelaunayCluster(dirSample, threshold=0.76, save='save')
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
            fluidPressure[d,0] /= (nDim * fluidArea)
            fluidPressure[d,1] /= fluidArea # dim k_B T / dim, dim cancels out
            fluidPressure[d,2] /= (nDim * fluidArea)
        if(gasArea > 0):
            gasPressure[d,0] /= (nDim * gasArea)
            gasPressure[d,1] /= gasArea # dim k_B T / dim, dim cancels out
            gasPressure[d,2] /= (nDim * gasArea)
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
    eps = 1.4 * sigma
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
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps)
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
    eps = 1.4 * sigma
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
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
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
def computeLJStressVSTime(dirName, LJcutoff=4, strain=0, dirSpacing=1, nDim=2):
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
def computeLJStressVSStrain(dirName, LJcutoff=5.5, strain=0, dirSpacing=1, nDim=2):
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
            cluster.computeDelaunayCluster(dirSample, threshold=0.3, save='save')
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
            cluster.computeDelaunayCluster(dirSample, threshold=0.3, save='save')
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
            borderPressure[d,1] /= (nDim * borderArea) # double counting
        if(fluidArea > 0):
            fluidPressure[d,0] /= fluidArea # dim k_B T / dim, dim cancels out
            fluidPressure[d,1] /= (nDim * fluidArea) # double counting
        if(gasArea > 0):
            gasPressure[d,0] /= gasArea # dim k_B T / dim, dim cancels out
            gasPressure[d,1] /= (nDim * gasArea) # double counting
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
    eps = 1.4 * sigma
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
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
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
def averageKBLJTension(dirName, LJcutoff=5.5, dirSpacing=1, plot=False, nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    #eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    tension = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        stress = np.zeros((numParticles, nDim))
        for i in range(numParticles):
            for dim in range(nDim):
                stress[i,dim] = vel[i,dim]**2
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                if(distance <= (LJcutoff * radSum)):
                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                    for dim in range(nDim):
                        stress[i,dim] += 0.5 * gradMultiple * delta[dim]**2 / distance
        stress /= boxSize[0] * boxSize[1]
        tension[d] = 0.5 * boxSize[0] * (np.sum(stress[:,0]) - np.sum(stress[:,1]))
    np.savetxt(dirName + os.sep + "KBTension.dat", np.column_stack((timeList, tension)))
    print("surface tension: ", np.mean(tension), "+-", np.std(tension))
    if(plot=="plot"):
        uplot.plotCorrelation(timeList, tension, "$\\gamma$", xlabel = "$Simulation$ $time,$ $t$", color='k', lw=1)
        #plt.pause(0.5)
        plt.show()

##################### Average LJ linear pressure profile #######################
def averageHLJPressureProfile(dirName, LJcutoff=5.5, dirSpacing=1, nDim=2, plot=False):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    #eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 2*LJcutoff*np.mean(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((dirList.shape[0],bins.shape[0]-1))
    virial = np.zeros((dirList.shape[0], bins.shape[0]-1, nDim))
    virialRep = np.zeros((dirList.shape[0], bins.shape[0]-1, nDim))
    virialAtt = np.zeros((dirList.shape[0], bins.shape[0]-1, nDim))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        print(dirSample)
        if(os.path.exists(dirSample + "/HProfile!.dat")):
            data = np.loadtxt(dirSample + "/HProfile.dat")
            thermal[d] = data[:,1]
            virial[d,:,0] = data[:,2]
            virial[d,:,1] = data[:,3]
            virialRep[d,:,0] = data[:,4]
            virialRep[d,:,1] = data[:,5]
            virialAtt[d,:,0] = data[:,6]
            virialAtt[d,:,1] = data[:,7]
        else:
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            #labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
            #pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
            pos = utils.centerCOM(pos, rad, boxSize)
            vel = np.loadtxt(dirSample + "/particleVel.dat")
            contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
            binIndex = np.zeros(numParticles, dtype=int)
            for i in range(numParticles):
                for j in range(bins.shape[0]-1):
                    if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                        binIndex[i] = j
                        thermal[d,j] += 0.5 * np.linalg.norm(vel[i])**2
            for i in range(numParticles):
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    radSum = rad[i] + rad[c]
                    delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                    distance = np.linalg.norm(delta)
                    if(distance <= (LJcutoff * radSum)):
                        gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                        gradMultipleRep = utils.calcLJgradMultipleRep(ec, distance, radSum) - utils.calcLJgradMultipleRep(ec, LJcutoff * radSum, radSum)
                        gradMultipleAtt = utils.calcLJgradMultipleAtt(ec, distance, radSum) - utils.calcLJgradMultipleAtt(ec, LJcutoff * radSum, radSum)
                        for dim in range(nDim):
                            virial[d,binIndex[i],dim] -= 0.5 * gradMultiple * delta[dim]**2 / distance
                            virialRep[d,binIndex[i],dim] -= 0.5 * gradMultipleRep * delta[dim]**2 / distance
                            virialAtt[d,binIndex[i],dim] -= 0.5 * gradMultipleAtt * delta[dim]**2 / distance
            np.savetxt(dirSample + os.sep + "HProfile.dat", np.column_stack((centers, thermal[d]/binArea, virial[d]/binArea, virialRep[d]/binArea, virialAtt[d]/binArea)))
    thermal = np.mean(thermal,axis=0) / binArea
    virial = np.mean(virial,axis=0) / binArea
    virialRep = np.mean(virialRep,axis=0) / binArea
    virialAtt = np.mean(virialAtt,axis=0) / binArea
    np.savetxt(dirName + os.sep + "HProfile.dat", np.column_stack((centers, thermal, virial, virialRep, virialAtt)))
    print("surface tension: ", np.sum((virial[:,0] - virial[:,1])*binWidth))
    if(plot=="plot"):
        uplot.plotCorrelation(centers, virial[:,0], "$\\sigma_{xx}$", xlabel = "$Distance$", color='g', lw=1)
        uplot.plotCorrelation(centers, virial[:,1], "$\\sigma_{yy}$", xlabel = "$Distance$", color='b', lw=1)
        #plt.pause(0.5)
        plt.show()

##################### Sample average LJ linear pressure profile #######################
def sampleHLJPressureProfile(dirPath, numSamples=30, LJcutoff=5.5, dirSpacing=1, nDim=2, plot=False, temp="0.30"):
    dirName = dirPath + "0/langevin-lj/" + temp + "/dynamics/"
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 2*LJcutoff*np.mean(rad))
    binWidth = bins[1] - bins[0]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((numSamples, dirList.shape[0], bins.shape[0]-1))
    virial = np.zeros((numSamples, dirList.shape[0],bins.shape[0]-1, nDim))
    virialRep = np.zeros((numSamples, dirList.shape[0],bins.shape[0]-1, nDim))
    virialAtt = np.zeros((numSamples, dirList.shape[0],bins.shape[0]-1, nDim))
    for s in range(numSamples):
        print(s)
        dirName = dirPath + str(s) + "/langevin-lj/" + temp + "/dynamics/"
        thermal[s], virial[s] = averageHLJPressureProfile(dirName, LJcutoff)
    thermal = np.mean(thermal,axis=0)
    virial = np.mean(virial,axis=0)
    virialRep = np.mean(virialRep,axis=0)
    virialAtt = np.mean(virialAtt,axis=0)
    np.savetxt(dirPath + "0/../HProfile-" + temp + ".dat", np.column_stack((centers, thermal, virial, virialRep, virialAtt)))
    print("surface tension: ", np.sum((virial[:,0] - virial[:,1])*binWidth))
    if(plot=="plot"):
        uplot.plotCorrelation(centers, thermal + virial[:,0], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='g', lw=1)
        uplot.plotCorrelation(centers, thermal + virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='b', lw=1)
        uplot.plotCorrelation(centers, virial[:,0] - virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='r', lw=1)
        #plt.pause(0.5)
        plt.show()

##################### Average IK LJ linear pressure profile #######################
def averageIKLJPressureProfile(dirName, LJcutoff=4, nDim=2, plot=False, dirSpacing=1):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    #eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], np.mean(rad))
    binWidth = bins[1] - bins[0]
    if(nDim == 2):
        binArea = binWidth*boxSize[1]
    elif(nDim == 3):
        binArea = binWidth*boxSize[1]*boxSize[2]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((dirList.shape[0],bins.shape[0]-1,2))
    virial = np.zeros((dirList.shape[0],bins.shape[0]-1,2))
    pressure = np.zeros((dirList.shape[0],nDim))
    for d in range(dirList.shape[0]):
        init = time.time()
        dirSample = dirName + os.sep + dirList[d]
        print(dirSample)
        if(os.path.exists(dirSample + "/IKProfile!.dat")):
            data = np.loadtxt(dirSample + "/IKProfile.dat")
            thermal[d] = data[:,1]
            virial[d,:,0] = data[:,2]
            virial[d,:,1] = data[:,3]
        else:
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            #pos = utils.centerCOM(pos, rad, boxSize)
            vel = np.loadtxt(dirSample + "/particleVel.dat")
            contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
            for i in range(numParticles):
                for dim in range(nDim):
                    pressure[d,dim] += vel[i,dim]**2
                for j in range(bins.shape[0]-1):
                    if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                        thermal[d,j,0] += vel[i,0]**2
                        if(nDim == 2):
                            thermal[d,j,1] += vel[i,1]**2
                        elif(nDim == 3):
                            thermal[d,j,1] += (vel[i,1]**2 + vel[i,2]**2)
                        binIndex = j
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    radSum = rad[i] + rad[c]
                    delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                    distance = np.linalg.norm(delta)
                    if(distance <= (LJcutoff * radSum)):
                        gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                        force = gradMultiple * delta / distance
                        for dim in range(nDim):
                            pressure[d,dim] += 0.5 * force[dim] * delta[dim]
                        # find bin where this force belongs to following Irving-Kirkwood
                        if(pos[c,0] > bins[binIndex] and pos[c,0] <= bins[binIndex+1]): # count bond entirely in binIndex
                            virial[d,binIndex,0] += 0.5 * force[0] * delta[0] / np.abs(delta[0])
                            if(nDim == 2):
                                virial[d,binIndex,1] += 0.5 * force[1] * delta[1] / np.abs(delta[0])
                            elif(nDim == 3):
                                virial[d,binIndex,1] += 0.5 * (force[1] * delta[1] + force[2] * delta[2]) / np.abs(delta[0])
                        elif(pos[c,0] > bins[binIndex+1]): # count bond from i to right edge of binIndex
                            virial[d,binIndex,0] += 0.5 * force[0] * delta[0] / (bins[binIndex+1] - pos[i,0])
                            if(nDim == 2):
                                virial[d,binIndex,1] += 0.5 * force[1] * delta[1] / (bins[binIndex+1] - pos[i,0])
                            elif(nDim == 3):
                                virial[d,binIndex,1] += 0.5 * (force[1] * delta[1] + force[2] * delta[2]) / (bins[binIndex+1] - pos[i,0])
                        elif(pos[c,0] < bins[binIndex]): # count bond from i to left edge of binIndex
                            virial[d,binIndex,0] += 0.5 * force[0] * delta[0] / (pos[i,0] - bins[binIndex])
                            if(nDim == 2):
                                virial[d,binIndex,1] += 0.5 * force[1] * delta[1] / (pos[i,0] - bins[binIndex])
                            elif(nDim == 3):
                                virial[d,binIndex,1] += 0.5 * (force[1] * delta[1] + force[2] * delta[2]) / (pos[i,0] - bins[binIndex])
                        for j in range(bins.shape[0]-1):
                            if(j!=binIndex):
                                if((pos[i,0] < bins[j] and pos[c,0] >= bins[j+1]) or (pos[c,0] < bins[j] and pos[i,0] >= bins[j+1])):
                                    virial[d,j,0] += 0.5 * force[0] * delta[0] / binWidth
                                    if(nDim == 2):
                                        virial[d,binIndex,1] += 0.5 * force[1] * delta[1] / binWidth
                                    elif(nDim == 3):
                                        virial[d,binIndex,1] += 0.5 * (force[1] * delta[1] + force[2] * delta[2]) / binWidth
            end = time.time()
            print("elapsed time in seconds:", end - init)
            np.savetxt(dirSample + os.sep + "IKProfile.dat", np.column_stack((centers, thermal[d]/binArea, virial[d]/binArea)))
    thermal = np.mean(thermal,axis=0) / binArea
    virial = np.mean(virial,axis=0) / binArea
    pressure /= (boxSize[0] * boxSize[1])
    print("surface tension from pressure:", boxSize[0] * np.mean(pressure[:,0] - pressure[:,1]))
    np.savetxt(dirName + os.sep + "IKProfile.dat", np.column_stack((centers, thermal, virial)))
    np.savetxt(dirName + os.sep + "pressure.dat", np.column_stack((timeList, pressure)))
    print("surface tension: ", np.sum((thermal[:,0] + virial[:,0] - (thermal[:,1] + virial[:,1]))*binWidth))
    if(plot=="plot"):
        uplot.plotCorrelation(centers, thermal[:,0] + virial[:,0], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$x$", color='g', lw=1)
        uplot.plotCorrelation(centers, thermal[:,1] + virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$x$", color='b', lw=1)
        uplot.plotCorrelation(centers, thermal[:,0] + virial[:,0] - (thermal[:,1] + virial[:,1]), "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$x$", color='k', lw=1)
        plt.show()

##################### Sample average LJ linear pressure profile #######################
def sampleIKLJPressureProfile(dirPath, numSamples=30, LJcutoff=5.5, dirSpacing=1, nDim=2, plot=False, temp="0.30"):
    dirName = dirPath + "0/langevin-lj/" + temp + "/dynamics/"
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 2*LJcutoff*np.mean(rad))
    binWidth = bins[1] - bins[0]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((numSamples,bins.shape[0]-1))
    virial = np.zeros((numSamples,bins.shape[0]-1,2))
    for s in range(numSamples):
        print(s)
        dirName = dirPath + str(s) + "/langevin-lj/" + temp + "/dynamics/"
        thermal[s], virial[s] = averageCGIKLJPressureProfile(dirName, LJcutoff)
    thermal = np.mean(thermal,axis=0)
    virial = np.mean(virial,axis=0)
    np.savetxt(dirPath + "0/../IKProfile-" + temp + ".dat", np.column_stack((centers, thermal, virial)))
    print("surface tension: ", np.sum((virial[:,0] - virial[:,1])*binWidth))
    if(plot=="plot"):
        uplot.plotCorrelation(centers, thermal + virial[:,0], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='g', lw=1)
        uplot.plotCorrelation(centers, thermal + virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='b', lw=1)
        uplot.plotCorrelation(centers, virial[:,0] - virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='r', lw=1)
        #plt.pause(0.5)
        plt.show()

##################### Average IK LJ linear pressure profile #######################
def averageCGIKLJPressureProfile(dirName, LJcutoff=4, plot=False, dirSpacing=1, nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    #eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 2*LJcutoff*np.mean(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((dirList.shape[0],bins.shape[0]-1,nDim))
    virial = np.zeros((dirList.shape[0],bins.shape[0]-1,nDim))
    for d in range(dirList.shape[0]):
        init = time.time()
        dirSample = dirName + os.sep + dirList[d]
        print(dirSample)
        if(os.path.exists(dirSample + "/CGIKProfile!.dat")):
            data = np.loadtxt(dirSample + "/CGIKProfile.dat")
            thermal[d] = data[:,1]
            virial[d,:,0] = data[:,2]
            virial[d,:,1] = data[:,3]
        else:
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            pos = utils.centerCOM(pos, rad, boxSize)
            vel = np.loadtxt(dirSample + "/particleVel.dat")
            contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
            for i in range(numParticles):
                for j in range(bins.shape[0]-1):
                    if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                        thermal[d,j,0] += vel[i,0]*vel[i,0]
                        thermal[d,j,1] += vel[i,1]*vel[i,1]
                        binIndex = j
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    radSum = rad[i] + rad[c]
                    delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                    distance = np.linalg.norm(delta)
                    if(distance <= (LJcutoff * radSum)):
                        gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff*radSum, radSum)
                        force = gradMultiple * delta / distance
                        # find bin where this force belongs to following Irving-Kirkwood
                        count = 0
                        if(pos[c,0] > bins[binIndex] and pos[c,0] <= bins[binIndex+1]):
                            for dim in range(nDim):
                                virial[d,binIndex,dim] += 0.5 * force[dim] * delta[dim] / np.abs(delta[0])
                        else:
                            if(pos[i,0] < pos[c,0]):
                                for j in np.arange(binIndex, bins.shape[0]-1):
                                    if(j == binIndex):
                                        xa = pos[i,0]
                                    else:
                                        xa = bins[j]
                                    if(pos[c,0] > bins[j+1]):
                                        xb = bins[j+1]
                                    else:
                                        xb = pos[c,0]
                                    for dim in range(nDim):
                                        virial[d,j,dim] += 0.5 * force[dim] * delta[dim] * (xb - xa) / np.abs(delta[0])
                            else:
                                for j in np.arange(binIndex):
                                    if(pos[c,0] > bins[j]):
                                        xb = pos[c,0]
                                    else:
                                        xb = bins[j]
                                    if(pos[i,0] > bins[j+1]):
                                        xa = bins[j+1]
                                    else:
                                        xa = pos[i,0]
                                    for dim in range(nDim):
                                        virial[d,j,dim] += 0.5 * force[dim] * delta[dim] * (xb - xa) / np.abs(delta[0])
            end = time.time()
            print("elapsed time:", end - init)
            np.savetxt(dirSample + os.sep + "CGIKProfile.dat", np.column_stack((centers, thermal[d]/binArea, virial[d]/binArea)))
    thermal = np.mean(thermal,axis=0) / binArea
    virial = np.mean(virial,axis=0) / binArea
    np.savetxt(dirName + os.sep + "CGIKProfile.dat", np.column_stack((centers, thermal, virial)))
    print("surface tension: ", np.sum((virial[:,0]-virial[:,1])*binWidth))
    if(plot=="plot"):
        uplot.plotCorrelation(centers, thermal[:,0]+virial[:,0], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='g', lw=1)
        uplot.plotCorrelation(centers, thermal[:,1]+virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='b', lw=1)
        uplot.plotCorrelation(centers, thermal[:,0]+virial[:,0] - thermal[:,1]-virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='k', lw=1)
        #plt.pause(0.5)
        plt.show()

##################### Sample average LJ linear pressure profile #######################
def sampleCGIKLJPressureProfile(dirPath, numSamples=30, LJcutoff=5.5, dirSpacing=1, nDim=2, plot=False, temp="T0.30"):
    dirName = dirPath + "0/langevin-lj/" + temp + "/dynamics/"
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], 2*LJcutoff*np.mean(rad))
    binWidth = bins[1] - bins[0]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    thermal = np.zeros((numSamples,bins.shape[0]-1))
    virial = np.zeros((numSamples,bins.shape[0]-1,nDim))
    virialRep = np.zeros((numSamples,bins.shape[0]-1,nDim))
    virialAtt = np.zeros((numSamples,bins.shape[0]-1,nDim))
    for s in range(numSamples):
        print(s)
        dirName = dirPath + str(s) + "/langevin-lj/" + temp + "/dynamics/"
        thermal[s], virial[s], virialRep[s], virialAtt[s] = averageCGIKLJPressureProfile(dirName, LJcutoff)
    thermal = np.mean(thermal,axis=0)
    virial = np.mean(virial,axis=0)
    virialRep = np.mean(virialRep,axis=0)
    virialAtt = np.mean(virialAtt,axis=0)
    np.savetxt(dirPath + "0/../CGIKProfile-" + temp + ".dat", np.column_stack((centers, thermal, virial, virialRep, virialAtt)))
    print("surface tension: ", np.sum((virial[:,0] - virial[:,1])*binWidth))
    if(plot=="plot"):
        uplot.plotCorrelation(centers, virial[:,0], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='g', lw=1)
        uplot.plotCorrelation(centers, virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='b', lw=1)
        uplot.plotCorrelation(centers, virial[:,0] - virial[:,1], "$\\sigma_{xx}(green),$ $\\sigma_{yy}(blue)$", xlabel = "$Distance$", color='r', lw=1)
        #plt.pause(0.5)
        plt.show()

############################ Total stress components ###########################
def compute2LJPressureVSTime(dirName, num1=0, active=False, dirSpacing=1, LJcutoff=4, nDim=2):
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    boxArea = boxSize[0]*boxSize[1]
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    sigma = utils.readFromParams(dirName + sep, "sigma")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    if active:
        driving = utils.readFromDynParams(dirName + sep, "f0")
        taup = utils.readFromDynParams(dirName + sep, "taup")
        stress = np.zeros((dirList.shape[0],3))
    else:
        stress = np.zeros((dirList.shape[0],2))
    wall = np.zeros(dirList.shape[0])
    temp = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        print(dirList[d])
        dirSample = dirName + os.sep + dirList[d]
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        temp[d] = 0.5 * np.sum(vel**2) / numParticles
        # particle pressure components
        stress[d,1] = np.sum(vel**2)
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        if active:
            angle = utils.getMOD2PIAngles(dirName + "/particleAngles.dat")
            stress[d,2] = 0.5 * (driving * taup) * np.sum((np.cos(angle) * vel[:,0] + np.sin(angle) * vel[:,1]))
        for i in range(numParticles):
            # wall pressure
            wall[d] += utils.checkWallInteractionX(pos[i], rad[i], boxSize, LJcutoff, ec=2)
            wall[d] += utils.checkWallInteractionY(pos[i], rad[i], boxSize, LJcutoff, ec=2)
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                if(distance <= (LJcutoff * radSum)):
                    ec = utils.getEnergyScale(i, c, num1)
                    forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - forceShift
                    force = 0.5 * gradMultiple * delta / distance # double counting
                    stress[d,0] += force[0] * delta[0] + force[1] * delta[1]
    stress /= (nDim * boxArea)
    wall /= (2*boxSize[0] + 2*boxSize[1])
    np.savetxt(dirName + os.sep + "timePressure.dat", np.column_stack((timeList, stress, wall)))
    print("temperature:", np.mean(temp), "+/-", np.std(temp))
    print("internal pressure \nsteric:", np.mean(stress[:,0]), " +/- ", np.std(stress[:,0]))
    print("thermal:", np.mean(stress[:,1]), " +/- ", np.std(stress[:,1]))
    if active: print("active:", np.mean(stress[:,2]), " +/- ", np.std(stress[:,2]))
    print("\nwall pressure: ", np.mean(wall), " +/- ", np.std(wall))
    uplot.plotCorrelation(timeList, np.sum(stress, axis=1), color='b', ylabel='$Pressure$', xlabel='$Simulation$ $time,$ $t$')
    #uplot.plotCorrelation(timeList, wall, color='k', ylabel='$Pressure$', xlabel='$Simulation$ $time,$ $t$')
    #plt.pause(0.5)
    plt.show()

def computeLinearPressureProfile(dirName, size=2, num1=0, plot=False):
    esame = 1
    ediff = 0.1
    LJcutoff = 4;
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    # distance bins
    bins = np.arange(0, boxSize[0], size*np.mean(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    profile = np.zeros(centers.shape[0])
    # first compute particle density
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    pos = utils.centerCOM(pos, rad, boxSize)
    contacts = np.loadtxt(dirName + "/particleNeighbors.dat").astype(np.int64)
    for i in range(numParticles):
        pressure = 0
        for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
            if((i < num1 and c >= num1) or (i >= num1 and c < num1)):
                ec = ediff
            else:
                ec = esame
            radSum = rad[i] + rad[c]
            delta = utils.pbcDistance(pos[i], pos[c], boxSize)
            distance = np.linalg.norm(delta)
            if(distance <= (LJcutoff * radSum)):
                forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - forceShift
                pressure += np.sum(gradMultiple * delta**2 / distance)
        for j in range(bins.shape[0]-1):
            if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                profile[j] += pressure
    profile /= binArea
    center = np.mean(centers)
    x = centers-center
    np.savetxt(dirName + os.sep + "singleProfile.dat", np.column_stack((x, profile)))
    if(plot=='plot'):
        uplot.plotCorrelation(x, profile, "$Density$ $profile$", xlabel = "$x$", color='k')
        #plt.pause(0.5)
        plt.show()

####################### Average linear density profile ########################
def computeLinearDensityProfile(dirName, threshold=0.3, size=2, correction=True, lj=True, cluster=False, plot=False):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    # distance bins
    bins = np.arange(0, boxSize[0], size*2*np.mean(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    localDensity = np.zeros(centers.shape[0])
    localArea = np.zeros(centers.shape[0])
    # first compute particle density
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    pos = utils.centerCOM(pos, rad, boxSize)
    if cluster:
        if lj:
            rad *= 2**(1/6)
        sigma = np.mean(rad)
        eps = 1.4 * sigma
        labels, maxLabel = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold)
        #print(maxLabel, labels[labels==maxLabel].shape[0])
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
    if correction:
        if lj:
            contacts = np.loadtxt(dirName + "/particleNeighbors.dat").astype(np.int64)
        else:
            contacts = np.loadtxt(dirName + "/particleContacts.dat").astype(np.int64)
        contacts = np.array(contacts)
        utils.computeLocalAreaBins(pos, rad, contacts, boxSize, bins, localArea)
    else:
        for i in range(numParticles):
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                   localArea[j] += np.pi*rad[i]**2
    localDensity = localArea / binArea
    # compute interface width
    center = np.mean(centers)
    x = centers-center
    y = localDensity
    np.savetxt(dirName + os.sep + "singleProfile.dat", np.column_stack((x, localDensity)))
    interfaceWidth = (utils.computeInterfaceWidth(x[x>0], y[x>0]) + utils.computeInterfaceWidth(-x[x<0], y[x<0]))/2
    print("Interface width:", interfaceWidth, "average density:", np.mean(localDensity))
    # compute fluid width
    width = 0
    liquidDensity = 0
    vaporDensity = 0
    if(x[np.argwhere(y>0.5)[:,0]].shape[0]>0):
        xFluid = x[np.argwhere(y>0.5)[:,0]]
        width = xFluid[-1] - xFluid[0]
        print("Fluid width:", width)
    if(y[y>0.5].shape[0]>0):
        liquidDensity = np.mean(y[y>0.5])
        print("Liquid density:", liquidDensity)
    if(y[y<0.5].shape[0]>0):
        vaporDensity = np.mean(y[y<0.5])
        print("Vapor density:", vaporDensity)
    if(plot=='plot'):
        uplot.plotCorrelation(x, localDensity, "$Density$ $profile$", xlabel = "$x$", color='k')
        #plt.pause(0.5)
        plt.show()
    return width, liquidDensity, vaporDensity

####################### Average linear density profile ########################
def averageLinearDensityProfile(dirName, threshold=0.3, size=2, correction=False, lj=True, cluster=False, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    if cluster:
        if lj:
            rad *= 2**(1/6)
        eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], size*2*np.mean(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # density lists
    localDensity = np.zeros((dirList.shape[0], bins.shape[0]-1))
    interfaceWidth = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        localArea = np.zeros(centers.shape[0])
        dirSample = dirName + os.sep + dirList[d]
        # first compute particle density
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = utils.centerCOM(pos, rad, boxSize)
        if cluster:
            labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
            pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        if correction:
            if lj:
                contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
            else:
                contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64) 
            contacts = np.array(contacts)
            utils.computeLocalAreaBins(pos, rad, contacts, boxSize, bins, localArea[d])
        else:
            for i in range(numParticles):
                for j in range(bins.shape[0]-1):
                    if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                        localArea[j] += np.pi*rad[i]**2
        localDensity[d] = localArea / binArea
        # compute width of interface
        center = np.mean(centers)
        x = centers-center
        y = localDensity[d]
        interfaceWidth[d] = (utils.computeInterfaceWidth(x[x>0], y[x>0]) + utils.computeInterfaceWidth(-x[x<0], y[x<0]))/2
    localDensity = np.column_stack((np.mean(localDensity, axis=0), np.std(localDensity, axis=0)))
    np.savetxt(dirName + os.sep + "densityProfile.dat", np.column_stack((centers, localDensity)))
    np.savetxt(dirName + os.sep + "interfaceWidth.dat", np.column_stack((timeList, interfaceWidth)))
    print("Interface width:", np.mean(interfaceWidth[interfaceWidth>0]), "+-", np.std(interfaceWidth[interfaceWidth>0]))
    # compute fluid width
    x = centers-center
    y = localDensity[:,0]
    xFluid = x[np.argwhere(y>0.5)[:,0]]
    if(xFluid.shape[0] != 0):
        width = xFluid[-1] - xFluid[0]
    else:
        width = 0
    liquidDensity = np.mean(y[y>0.5])
    vaporDensity = np.mean(y[y<0.5])
    print("Liquid width:", width, "liquid density:", liquidDensity, "vapor density:", vaporDensity)
    if(plot=='plot'):
        #uplot.plotCorrelation(timeList, interfaceWidth, "$Interface$ $width$")
        uplot.plotCorrWithError(centers, localDensity[:,0], localDensity[:,1], "$Density$ $profile$", xlabel = "$x$", color='k')
        plt.show()
        #plt.pause(0.5)
    return width, liquidDensity, vaporDensity

####################### Single linear density profile ########################
def compute2DensityProfile(dirName, num1=0):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    # distance bins
    bins = np.arange(0, boxSize[0], np.max(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # density lists
    particleDensity = np.zeros((bins.shape[0]-1,2))
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    pos = utils.centerCOM1(pos, rad, boxSize, num1)
    for i in range(numParticles):
        for j in range(bins.shape[0]-1):
            if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                if(i < num1):
                    particleDensity[j,0] += np.pi*rad[i]**2
                else:
                    particleDensity[j,1] += np.pi*rad[i]**2
    particleDensity[:,0] /= binArea
    particleDensity[:,1] /= binArea
    x = centers
    y = particleDensity[:,0]
    bulk = x[np.argwhere(y>0.5)[:,0]]
    if(bulk.shape[0] != 0):
        width = bulk[-1] - bulk[0]
    else:
        width = 0
    print("width of phase 1:", width)
    uplot.plotCorrelation(centers, particleDensity[:,0], "$Density$ $profile$", xlabel = "$x$", color='g')
    uplot.plotCorrelation(centers, particleDensity[:,1], "$Density$ $profile$", xlabel = "$x$", color='b')
    plt.show()

def hyperbolicTan(x, a, b, x0, w):
    return 0.5*(a+b) - 0.5*(a-b)*np.tanh(2*(x-x0)/w)

####################### Average linear density profile ########################
def average2DensityProfile(dirName, num1=0, plot=False, dirSpacing=500000):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "../particleRad.dat"))
    bins = np.arange(0, boxSize[0], np.max(rad))
    centers = (bins[1:] + bins[:-1])/2
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    print(numParticles, "particles in system")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    particleDensity = np.zeros((dirList.shape[0],bins.shape[0]-1,2))
    mask = np.zeros(dirList.shape[0], dtype=bool)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # first compute particle density
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if(pos.shape[0] == numParticles):
            if(rad.shape[0] == numParticles):
                mask[d] = True
                pos = utils.centerCOM1(pos, rad, boxSize, num1)
                for i in range(numParticles):
                    for j in range(bins.shape[0]-1):
                        if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                            if(i < num1):
                                particleDensity[d,j,0] += np.pi*rad[i]**2
                            else:
                                particleDensity[d,j,1] += np.pi*rad[i]**2
                particleDensity[d,:,0] /= binArea
                particleDensity[d,:,1] /= binArea
        else:
            print("WARNING: directory", dirList[d], "has", pos.shape[0], "particles. Skipping this directory.")
    particleDensity1 = np.column_stack((np.mean(particleDensity[mask,:,0], axis=0), np.std(particleDensity[mask,:,0], axis=0)))
    particleDensity2 = np.column_stack((np.mean(particleDensity[mask,:,1], axis=0), np.std(particleDensity[mask,:,1], axis=0)))
    np.savetxt(dirName + os.sep + "densityProfile.dat", np.column_stack((centers, particleDensity1, particleDensity2)))
    center = np.mean(centers)
    x = centers[centers>center]
    y = particleDensity1[centers>center,0]
    # Initial guess for parameters: [low_density, high_density, center_x, width]
    p0 = [np.max(y), np.min(y), np.mean(x), 10]
    # Fit the data
    popt, pcov = curve_fit(hyperbolicTan, x, y, p0=p0)
    # Extract fitted parameters
    a_fit, b_fit, x0_fit, w_fit = popt
    print(f"Fitted parameters: high_phi = {a_fit:.4f}, low_phi = {b_fit:.4f}, center = {x0_fit:.2f}, width = {w_fit:.5f}, error = {np.sqrt(pcov[3,3]):.5f}")
    if(plot=='plot' or plot=='show'):
        #uplot.plotCorrelation(timeList, width, "$Interface$ $width$")
        plt.errorbar(centers, particleDensity1[:,0], particleDensity1[:,1], color='g', marker='o', fillstyle='none', markersize=6, capsize=3)
        plt.errorbar(centers, particleDensity2[:,0], particleDensity2[:,1], color='b', marker='o', fillstyle='none', markersize=6, capsize=3)
        x_fit = np.linspace(np.min(x), np.max(x), 300)
        y_fit = hyperbolicTan(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='k', lw=1, ls='--')
        plt.xlim(np.min(x_fit), np.max(x_fit))
        plt.tick_params(axis='both', labelsize=14)
        plt.xlabel("$x$", fontsize=20)
        plt.ylabel("$Density$", fontsize=20)
        plt.tight_layout()
        if(plot=='show'):
            plt.show()
        else:
            plt.pause(0.5)

def plotWidth(dirName, num1, which, strain="0.0100", damping='1e01', pause='show'):
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    if which == 'nvt':
        dirList = np.array(['1e-09', '1e-08', '1e-07', '1e-06', '1e-05', '1e-04', '1e-03', '1e-02', '1e-01', '1', '1e01'])
    elif which == 'active-tp3e-01':
        dirList = np.array(['1', '2', '3', '4', '5', '6'])#, '7', '8', '9', '10'])
    elif which == 'active-tp3e-04':
        dirList = np.array(['1e02', '2e02', '4e02', '6e02', '8e02', '1e03'])#, '1.2e03', '1.4e03', '1.6e03', '1.8e03'])
    elif which == 'nvtvst':
        dirList = np.array(['1.10', '1.20', '1.30', '1.40', '1.50', '1.60', '1.70', '1.80', '1.90', '2.00', '2.10', '2.20'])
    else:
        which = 'nve'
        dirList = np.array(['0.90', '1.00', '1.10', '1.20', '1.30', '1.40', '1.50', '1.60', '1.70', '1.80', '1.90', '2.00', '2.10', '2.20'])
    if which == 'nvt':
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        beta = np.zeros(dirList.shape[0])
    else:
        colorList = cm.get_cmap('plasma')
        temp = np.zeros((dirList.shape[0],2))
    width = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if which == 'nvt':
            dirSample = dirName + "/strain" + strain + os.sep + "damping" + dirList[d] + os.sep + "lang2con/" 
        elif which == 'active-tp3e-01':
            dirSample = dirName + "/strain" + strain + os.sep + "damping1e01/tp3e-01-Ta" + dirList[d] + os.sep + "lang2con/" 
        elif which == 'active-tp3e-04':
            dirSample = dirName + "/strain" + strain + os.sep + "damping1e01/tp3e-04-Ta" + dirList[d] + os.sep + "lang2con/"
        elif which == 'nvtvst':
            dirSample = dirName + os.sep + "T" + dirList[d] + os.sep + "nve/nve-biaxial-ext5e-06-tmax2e03/strain" + strain + "/damping" + damping + "/lang2con/"
        else:
            dirSample = dirName + os.sep + "T" + dirList[d] + os.sep + "nve/nve-biaxial-ext5e-06-tmax2e03/strain" + strain + "/dynamics/"
        #print(dirSample)
        if which == 'nvt':
            beta[d] = utils.readFromDynParams(dirSample, "damping")
            colorId = d / dirList.shape[0]
        else:
            ekin = np.loadtxt(dirSample + os.sep + "energy.dat", usecols=(3,))
            epsilon = utils.readFromParams(dirSample, "epsilon")
            temp[d,0] = np.mean(ekin)/epsilon
            temp[d,1] = np.std(ekin)/epsilon
            colorId = (temp[d,0] - 0.45) / (1.05 - 0.45)
        if (os.path.exists(dirSample + os.sep + "t0")):
            if not(os.path.exists(dirSample + os.sep + "densityProfile.dat")):
                print("Computing average density profile in", dirSample)
                average2DensityProfile(dirSample, num1)
            data = np.loadtxt(dirSample + os.sep + "densityProfile.dat")
            centers = data[:,0]
            particleDensity1 = data[:,1:3]#particleDensity2 = data[:,3:5]
            center = np.mean(centers)
            boxSize = np.loadtxt(dirSample + os.sep + "boxSize.dat")
            x = centers[centers>center] / boxSize[0]
            y = particleDensity1[centers>center,0]
            p0 = [np.min(y), np.max(y), np.mean(x), 100]
            popt, pcov = curve_fit(hyperbolicTan, x, y, p0=p0)
            ax.plot(x, y, color=colorList(colorId), lw=1)
            x_fit = np.linspace(np.min(x), np.max(x), 300)
            y_fit = hyperbolicTan(x_fit, *popt)
            plt.plot(x_fit, y_fit, color='k', lw=1, ls='--')
            # Extract fitted parameters
            width[d,0] = np.abs(popt[3]) / boxSize[0]
            width[d,1] = np.sqrt(pcov[3,3]) / boxSize[0]
    # save width data
    if which == 'nvt':
        np.savetxt(dirName + os.sep + "width-nvt-strain" + strain + ".dat", np.column_stack((beta, width)))
    elif which == 'active-tp3e-01':
        np.savetxt(dirName + os.sep + "width-active-tp3e-01-strain" + strain + ".dat", np.column_stack((temp, width)))
    elif which == 'active-tp3e-04':
        np.savetxt(dirName + os.sep + "width-active-tp3e-04-strain" + strain + ".dat", np.column_stack((temp, width)))
    elif which == 'nvtvst':
        np.savetxt(dirName + os.sep + "width-nvtvst-damping" + damping + "-strain" + strain + ".dat", np.column_stack((temp, width)))
    else:
        np.savetxt(dirName + os.sep + "width-nve-strain" + strain + ".dat", np.column_stack((temp, width)))
    # plot density profile
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$x / L_x$", fontsize=14)
    ax.set_ylabel("$Density$", fontsize=14)
    plt.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/mips/profile-" + which + ".png", dpi=120)
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("$Interface$ $width,$ $w/L_x$", fontsize=14)
    if which == 'nvt':
        ax.set_xscale('log')
        ax.set_xlabel("$Damping,$ $\\beta$", fontsize=14)
        ax.errorbar(beta, width[:,0], width[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
    else:
        #ax.set_xlim(0.525, 1.015)
        ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
        ax.errorbar(temp[:,0], width[:,0], width[:,1], temp[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
    plt.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/mips/width-" + which + ".png", dpi=120)
    if pause == 'show':
        plt.show()
    else:
        plt.pause(0.5)

def plotAverageWidth(dirName, which, pause='show'):
    strainList = np.array(['0.0100', '0.0200', '0.0300', '0.0400', '0.0500', '0.0600', '0.0700', '0.0800',
                           '0.0900', '0.1000', '0.1100', '0.1200', '0.1300', '0.1400', '0.1500', '0.1600'])
    width = np.empty(0)
    temp = np.empty(0)
    beta = np.zeros(strainList.shape[0])
    for d in range(strainList.shape[0]):
        if which == 'nvt':
            data = np.loadtxt(dirName + os.sep + "width-nvt-strain" + strainList[d] + ".dat")
            beta = data[:,0]
        elif which == 'active-tp3e-01':
            data = np.loadtxt(dirName + os.sep + "width-active-tp3e-01-strain" + strainList[d] + ".dat")
        elif which == 'active-tp3e-04':
            data = np.loadtxt(dirName + os.sep + "width-active-tp3e-04-strain" + strainList[d] + ".dat")
        elif which == 'nvtvst-damping1e01':
            data = np.loadtxt(dirName + os.sep + "width-nvtvst-damping1e01-strain" + strainList[d] + ".dat")
        elif which == 'nvtvst-damping1e-03':
            data = np.loadtxt(dirName + os.sep + "width-nvtvst-damping1e-03-strain" + strainList[d] + ".dat")
        elif which == 'nvtvst-damping1e-07':
            data = np.loadtxt(dirName + os.sep + "width-nvtvst-damping1e-07-strain" + strainList[d] + ".dat")
        else:
            data = np.loadtxt(dirName + os.sep + "width-nve-strain" + strainList[d] + ".dat")
        dirLength = data.shape[0]
        if which != 'nvt':
            temp = np.append(temp, data[:,0])
        width = np.append(width, data[:,-2])
    if which != 'nvt':
        print(temp.shape, strainList.shape, dirLength)
        temp = temp.reshape((strainList.shape[0], dirLength))
        temp = np.column_stack((np.mean(temp, axis=0), np.std(temp, axis=0)))
    width = width.reshape((strainList.shape[0], dirLength))
    width = np.column_stack((np.mean(width, axis=0), np.std(width, axis=0)))
    # save average width data
    print(which)
    if which == 'nvt':
        np.savetxt(dirName + os.sep + "averageWidth-nvt.dat", np.column_stack((beta, width)))
        print("Saved in", dirName + os.sep + "averageWidth-nvt.dat")
    elif which == 'active-tp3e-01':
        np.savetxt(dirName + os.sep + "averageWidth-active-tp3e-01.dat", np.column_stack((temp, width)))
        print("Saved in", dirName + os.sep + "averageWidth-active-tp3e-01.dat")
    elif which == 'active-tp3e-04':
        np.savetxt(dirName + os.sep + "averageWidth-active-tp3e-04.dat", np.column_stack((temp, width)))
        print("Saved in", dirName + os.sep + "averageWidth-active-tp3e-04.dat")
    elif which == 'nvtvst-damping1e01':
        np.savetxt(dirName + os.sep + "averageWidth-nvtvst-damping1e01.dat", np.column_stack((temp, width)))
        print("Saved in", dirName + os.sep + "averageWidth-nvtvst-damping1e01.dat")
    elif which == 'nvtvst-damping1e-03':
        np.savetxt(dirName + os.sep + "averageWidth-nvtvst-damping1e-03.dat", np.column_stack((temp, width)))
        print("Saved in", dirName + os.sep + "averageWidth-nvtvst-damping1e-03.dat")
    elif which == 'nvtvst-damping1e-07':
        np.savetxt(dirName + os.sep + "averageWidth-nvtvst-damping1e-07.dat", np.column_stack((temp, width)))
        print("Saved in", dirName + os.sep + "averageWidth-nvtvst-damping1e-07.dat")
    else:
        np.savetxt(dirName + os.sep + "averageWidth-nve.dat", np.column_stack((temp, width)))
        print("Saved in", dirName + os.sep + "averageWidth-nve.dat")
    # plot density profile
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    # Force scientific notation on x and y axes
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  # always use scientific notation outside [-10^3, 10^3]

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    if which == 'nvt':
        ax.set_xscale('log')
        ax.set_xlabel("$Damping,$ $\\beta$", fontsize=14)
        ax.errorbar(beta, width[:,0], width[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
    else:
        ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
        #data = data[:-2,:]
        ax.errorbar(temp[:,0], width[:,0], width[:,1], temp[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
    ax.set_ylabel("$Interface$ $width,$ $w / L_x$", fontsize=14)
    plt.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/mips/averageWidth-" + which + ".png", dpi=120)
    if pause == 'show':
        plt.show()
    else:
        plt.pause(1)

def compareAverageWidth(dirName, figureName):
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    dirList = np.array(['/', '/', '/', '/', 'T1.00/nve/nve-biaxial-ext5e-06-tmax2e03/', 'T1.00/nve/nve-biaxial-ext5e-06-tmax2e03/'])
    fileList = np.array(['nve', 'nvtvst-damping1e-07', 'nvtvst-damping1e-03', 'nvtvst'])#, 'active-tp3e-01', 'active-tp3e-04'])
    labelList = np.array(['$NVE$', '$NVT,$ $\\tau_d = 10^{-4}$', '$NVT,$ $\\tau_d = 10^{-2}$', '$NVT,$ $\\tau_d = 1$', 
                          '$Active,$ $\\tau_p / \\tau_d = 10^{-2}$', '$Active,$ $\\tau_p / \\tau_d = 10^{-5}$'])
    colorList = ['k', 'b', 'g', 'c', 'r', [1,0.5,0]]
    markerList = ['o', 'd', 'D', 's', '^', '^']
    for d in range(fileList.shape[0]):
        print("Loading data from", dirName + dirList[d] + "averageWidth-" + fileList[d] + ".dat")
        if os.path.exists(dirName + dirList[d] + "averageWidth-" + fileList[d] + ".dat"):
            data = np.loadtxt(dirName + dirList[d] + "averageWidth-" + fileList[d] + ".dat")
            if(d == 0 or d == 1 or d == 2 or d == 3):
                data = data[:-2,:]
            ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color=colorList[d], marker=markerList[d], markersize=8, 
                        fillstyle='none', lw=1, capsize=3, label=labelList[d])
    ax.legend(loc='best', fontsize=10)
    ax.tick_params(axis='both', labelsize=12)
    # Force scientific notation on x and y axes
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  # always use scientific notation outside [-10^3, 10^3]

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
    ax.set_ylabel("$Interface$ $width,$ $w / L_x$", fontsize=14)
    plt.tight_layout()
    fig.savefig("/home/francesco/Pictures/soft/mips/compareWidth-vsTemp" + figureName + ".png", dpi=120)
    plt.show()

def computeWidth(centers, particleDensity1):
    center = np.mean(centers)
    x = centers[centers>center]
    y = particleDensity1[centers>center,0]
    p0 = [np.min(y), np.max(y), np.mean(x), 10]
    popt, pcov = curve_fit(hyperbolicTan, x, y, p0=p0)
    # Extract fitted parameters
    mean = np.abs(popt[3])
    std = np.sqrt(pcov[3,3])
    return np.array([mean, std])

def computeInterfaceLength(typePos, bins, thickness):
    leftPos = np.zeros((bins.shape[0]-1,2))
    rightPos = np.zeros((bins.shape[0]-1,2))
    rightInterface = np.zeros(bins.shape[0]-1)
    leftInterface = np.zeros(bins.shape[0]-1)
    # find particle positions in a vertical bin
    for j in range(bins.shape[0]-1): 
        topMask = np.argwhere(typePos[:,1] > bins[j])[:,0]
        binPos = typePos[topMask]
        bottomMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
        binPos = binPos[bottomMask]
        if(binPos.shape[0] > 0):
            binDistance = binPos[:,0]
            #left interface
            leftMask = np.argsort(binDistance)[:thickness]
            leftInterface[j] = np.mean(binDistance[leftMask])
            leftPos[j,0] = leftInterface[j]
            leftPos[j,1] = np.mean(binPos[leftMask,1])
            # right interface
            rightMask = np.argsort(binDistance)[-thickness:]
            rightInterface[j] = np.mean(binDistance[rightMask])
            rightPos[j,0] = rightInterface[j]
            rightPos[j,1] = np.mean(binPos[rightMask,1])
    # sum segments between interface coordinates
    length = 0
    if(rightInterface[rightInterface!=0].shape[0] == rightInterface.shape[0]):
        prevPos = rightPos[0]
        for j in range(1,bins.shape[0]-1):
            length += np.linalg.norm(rightPos[j] - prevPos)
            prevPos = rightPos[j]
    if(leftInterface[leftInterface!=0].shape[0] == leftInterface.shape[0]):
        prevPos = leftPos[0]
        for j in range(1,bins.shape[0]-1):
            length += np.linalg.norm(leftPos[j] - prevPos)
            prevPos = leftPos[j]
    return length

def average2InterfaceLength(dirName, num1=0, spacing=2, lj=True):
    boxSize = np.array(np.loadtxt(dirName + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + "particleRad.dat"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    if lj:
        rad *= 2**(1/6)
    sigma = 2 * np.mean(rad)
    eps = 1.4 * np.max(sigma)
    spacing *= sigma # vertical
    thickness = 3 # horizontal
    bins = np.arange(0, boxSize[1], spacing)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    length = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if not(os.path.exists(dirSample + os.sep + "interfaceLength.dat")):
            # load particle variables
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            labels = np.zeros(numParticles)
            labels[:num1] = 1
            clusterLabels, maxLabel = cluster.getTripleWrappedClusterLabels(pos, rad, boxSize, labels, eps)
            pos = utils.centerSlab(pos, rad, boxSize, clusterLabels, maxLabel)
            typePos = pos[clusterLabels==maxLabel]
            length[d] = computeInterfaceLength(typePos, bins, thickness)
            np.savetxt(dirName + os.sep + "interfaceLength.dat", np.column_stack((length[d], 0, 0)))
        else:
            data = np.loadtxt(dirSample + os.sep + "interfaceLength.dat")
            length[d] = data[0]
    np.savetxt(dirName + os.sep + "timeLength.dat", np.column_stack((timeList, length)))
    return length

def plotWidthVSLength(dirName, num1, dynamics, pause='show'):
    strainList = np.array(['0.0100', '0.0200', '0.0300', '0.0400', '0.0500', '0.0600', '0.0700', '0.0800',
                           '0.0900', '0.1000', '0.1100', '0.1200', '0.1300', '0.1400', '0.1500', '0.1600'])
    width1 = np.zeros((strainList.shape[0],2))
    width2 = np.zeros((strainList.shape[0],2))
    length = np.zeros((strainList.shape[0],2))
    height = np.zeros(strainList.shape[0])
    for d in range(strainList.shape[0]):
        dirSample = dirName + os.sep + "nve/nve-biaxial-ext5e-06-tmax2e03/strain" + strainList[d] + os.sep + dynamics
        boxSize = np.loadtxt(dirSample + os.sep + "boxSize.dat")
        if not(os.path.exists(dirSample + os.sep + "densityProfile.dat")):
            average2DensityProfile(dirSample, num1=0)
        data = np.loadtxt(dirSample + os.sep + "densityProfile.dat")
        centers = data[:,0]
        particleDensity1 = data[:,1:3]
        width1[d] = computeWidth(centers, particleDensity1) / boxSize[0]
        particleDensity2 = data[:,3:5]
        width2[d] = computeWidth(centers, particleDensity2) / boxSize[0]
        # Collect length data at current time
        if not(os.path.exists(dirSample + "timeLength.dat")):
            average2InterfaceLength(dirSample, num1)
        data = np.loadtxt(dirSample + "timeLength.dat")
        length[d,0] = np.mean(data[:,1])
        length[d,1] = np.std(data[:,1])
        height[d] = np.loadtxt(dirSample + os.sep + "boxSize.dat")[0]
    np.savetxt(dirName + os.sep + "widthLength.dat", np.column_stack((length, width1, width2)))
    # plot square width vs length
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Interface$ $length,$ $L$", fontsize=14)
    ax.set_ylabel("$(w / L_x)^2$", fontsize=14)
    #ax.errorbar(height, width1[:,0]**2, width1[:,1]*2*width1[:,0], color='g', marker='v', markersize=12, fillstyle='none', lw=1, capsize=3, label="$\\phi_1$")
    #ax.errorbar(height, width2[:,0]**2, width2[:,1]*2*width2[:,0], color='b', marker='v', markersize=12, fillstyle='none', lw=1, capsize=3, label="$\\phi_2$")
    width1 = width1[np.argsort(length[:,0])]
    width2 = width2[np.argsort(length[:,0])]
    length = np.sort(length, axis=0)
    ax.errorbar(length[:,0], width1[:,0]**2, width1[:,1]*2*width1[:,0], color='g', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3, label="$\\phi_1$")
    ax.errorbar(length[:,0], width2[:,0]**2, width2[:,1]*2*width2[:,0], color='b', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3, label="$\\phi_2$")
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    #fig.savefig("/home/francesco/Pictures/soft/mips/widthVSlength-" + dynamics + ".png", dpi=120)
    if pause == 'show':
        plt.show()
    else:
        plt.pause(0.5)
    

####################### Average cluster height interface #######################
def average2InterfaceCorrelation(dirName, num1=0, thickness=3, plot=False, dirSpacing=1000000):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    spacing = thickness * sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    maxCorrIndex = int((bins.shape[0]-1) / 2)
    centers = centers[:maxCorrIndex]
    leftHeightCorr = np.zeros((dirList.shape[0], maxCorrIndex))
    rightHeightCorr = np.zeros((dirList.shape[0], maxCorrIndex))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = utils.centerCOM1(pos, rad, boxSize, num1)
        pos = pos[:num1]
        leftHeight = np.zeros(bins.shape[0]-1)
        rightHeight = np.zeros(bins.shape[0]-1)
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            downMask = np.argwhere(pos[:,1] > bins[j])[:,0]
            binPos = pos[downMask]
            upMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
            binPos = binPos[upMask]
            if(binPos.shape[0] > 0):
                center = np.mean(binPos, axis=0)[0] # center of dense cluster
                binDistance = binPos[:,0] - center
                leftMask = np.argsort(binDistance)[:thickness]
                leftHeight[j] = np.mean(binDistance[leftMask])
                rightMask = np.argsort(binDistance)[-thickness:]
                rightHeight[j] = np.mean(binDistance[rightMask])
        leftHeightCorr[d] = utils.getHeightCorr(leftHeight, maxCorrIndex)
        rightHeightCorr[d] = utils.getHeightCorr(rightHeight, maxCorrIndex)
    leftHeightCorr = np.column_stack((np.mean(leftHeightCorr, axis=0), np.std(leftHeightCorr, axis=0)))
    rightHeightCorr = np.column_stack((np.mean(rightHeightCorr, axis=0), np.std(rightHeightCorr, axis=0)))
    heightCorr = np.zeros(leftHeightCorr.shape)
    heightCorr[:,0] = np.mean(np.column_stack((leftHeightCorr[:,0],rightHeightCorr[:,0])), axis=1)
    heightCorr[:,1] = np.sqrt(np.mean(np.column_stack((leftHeightCorr[:,1]**2,rightHeightCorr[:,1]**2)), axis=1))
    np.savetxt(dirName + os.sep + "heightCorr.dat", np.column_stack((centers, heightCorr)))
    if(plot=='plot' or plot=='show'):
        plt.errorbar(centers, heightCorr[:,0], color='k', marker='o', fillstyle='none', markersize=6, capsize=3)
        plt.tick_params(axis='both', labelsize=14)
        plt.xlabel("$\\Delta s$", fontsize=20)
        plt.ylabel("$C_{hh}$", fontsize=20)
        plt.tight_layout()
        if(plot=='show'):
            plt.show()
        else:
            plt.pause(0.5)

####################### Average linear density profile ########################
def compute2InterfaceEnergy(dirName, num1=0, which='strain', plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    if(which == 'strain'):
        dirList, timeList = utils.getOrderedStrainDirectories(dirName)
    elif(which == 'time'):
        dirList, timeList = utils.getOrderedDirectories(dirName)
        dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
        timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    else:
        print("Please choose the type of run between strain or time")
    interfaceEnergy = np.zeros((dirList.shape[0], 3))
    heteroEnergy = np.zeros((dirList.shape[0], 3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # first compute particle density
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        energy = np.loadtxt(dirSample + "/particleEnergies.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        for i in range(numParticles):
            heteroType = 0
            neighborIdx = neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]
            totNeighbors = neighborIdx.shape[0]
            if(i < num1):
                for n in neighborIdx:
                    if(n >= num1):
                        heteroType += 1
            else:
                for n in neighborIdx:
                    if(n < num1):
                        heteroType += 1
            if(heteroType != 0):
                heteroEnergy[d,0] += energy[i] + 0.5 * np.linalg.norm(vel[i])**2
                heteroEnergy[d,1] += energy[i]
                heteroEnergy[d,2] += 0.5 * np.linalg.norm(vel[i])**2
                if(heteroType / totNeighbors > 0.7):
                    interfaceEnergy[d,0] += energy[i] + 0.5 * np.linalg.norm(vel[i])**2
                    interfaceEnergy[d,1] += energy[i]
                    interfaceEnergy[d,2] += 0.5 * np.linalg.norm(vel[i])**2
    np.savetxt(dirName + os.sep + "interfaceEnergy.dat", np.column_stack((timeList, interfaceEnergy, heteroEnergy)))
    print("average interface energy: ", np.mean(interfaceEnergy[:,0]), "+-", np.std(interfaceEnergy[:,0]))
    print("average hetero energy: ", np.mean(heteroEnergy[:,0]), "+-", np.std(heteroEnergy[:,0]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, interfaceEnergy[:,0], "$Interface(black),$ $hetero(red)$", color='k')
        uplot.plotCorrelation(timeList, heteroEnergy[:,0], "$Interface(black),$ $hetero(red)$", color='r')
        #plt.pause(0.5)
        plt.show()

####################### Compute linear density profile ########################
def compute2PhaseWidthVSStrain(dirName, num1=0):
    dirList, strain = utils.getOrderedStrainDirectories(dirName)
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    # distance bins
    bins = np.arange(0, boxSize[0], np.max(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    width = np.zeros(dirList.shape[0])
    lengthBox = np.zeros(dirList.shape[0])
    widthBox = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        particleDensity = np.zeros((centers.shape[0],2))
        dirSample = dirName + dirList[d]
        # first compute particle density
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = utils.centerCOM1(pos, rad, boxSize, num1)
        lengthBox[d] = np.loadtxt(dirSample + "/boxSize.dat")[1]
        widthBox[d] = np.loadtxt(dirSample + "/boxSize.dat")[0]
        for i in range(numParticles):
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                    if(i < num1):
                        particleDensity[j,0] += np.pi*rad[i]**2
                    else:
                        particleDensity[j,1] += np.pi*rad[i]**2
        particleDensity[:,0] /= binArea
        particleDensity[:,1] /= binArea
        x = centers
        y = particleDensity[:,0]
        bulk = x[np.argwhere(y>0.5)[:,0]]
        if(bulk.shape[0] != 0):
            width[d] = bulk[-1] - bulk[0]
        else:
            width[d] = 0
    print("average phase 1 width during compression: ", np.mean(width), "+-", np.std(width))
    np.savetxt(dirName + "phaseWidth.dat", np.column_stack((strain, lengthBox, widthBox, width)))

####################### Compute ISF for the two fluids ########################
def average2FluidsCorr(dirName, startBlock, maxPower, freqPower, num1=0, plot=False):
    boxSize = np.loadtxt(dirName + "boxSize.dat")
    rad = np.loadtxt(dirName + "particleRad.dat").astype(np.float64)
    sigma = 2 * np.mean(rad)
    timeStep = utils.readFromParams(dirName, "dt")
    longWave = 2 * np.pi / sigma
    shortWave = 2 * np.pi / (20 * sigma)
    if(num1 != 0):
        particleCorr1 = []
        particleCorr2 = []
    else:
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
            if(num1 != 0):
                stepParticleCorr1 = []
                numPairs1 = 0
                stepParticleCorr2 = []
                numPairs2 = 0
            else:
                stepParticleCorr = []
                numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pos1, pos2 = utils.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        if(num1 != 0):
                            stepParticleCorr1.append(utils.computeLongShortWaveCorr(pos1[:num1,:], pos2[:num1,:], boxSize, longWave, shortWave, sigma))
                            numPairs1 += 1
                            stepParticleCorr2.append(utils.computeLongShortWaveCorr(pos1[num1:,:], pos2[num1:,:], boxSize, longWave, shortWave, sigma))
                            numPairs2 += 1
                        else:
                            stepParticleCorr.append(utils.computeLongShortWaveCorr(pos1, pos2, boxSize, longWave, shortWave, sigma))
                            numPairs += 1
            if(num1 != 0):
                if(numPairs1 > 0 or numPairs2 > 0):
                    stepList.append(spacing*spacingDecade)
                    particleCorr1.append(np.mean(stepParticleCorr1, axis=0))
                    particleCorr2.append(np.mean(stepParticleCorr2, axis=0))
            else:
                if(numPairs > 0):
                    stepList.append(spacing*spacingDecade)
                    particleCorr.append(np.mean(stepParticleCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList) * timeStep
    if(num1 != 0):
        particleCorr1 = np.array(particleCorr1).reshape((stepList.shape[0],6))
        particleCorr1 = particleCorr1[np.argsort(stepList)]
        particleCorr2 = np.array(particleCorr2).reshape((stepList.shape[0],6))
        particleCorr2 = particleCorr2[np.argsort(stepList)]
        np.savetxt(dirName + os.sep + "2logCorr.dat", np.column_stack((stepList, particleCorr1, particleCorr2)))
        data = np.column_stack((stepList, particleCorr1))
        tau = utils.getRelaxationTime(data)
        print("Fluid 1: relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
        data = np.column_stack((stepList, particleCorr2))
        tau = utils.getRelaxationTime(data)
        print("Fluid 2: relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
    else:
        particleCorr = np.array(particleCorr).reshape((stepList.shape[0],6))
        particleCorr = particleCorr[np.argsort(stepList)]
        np.savetxt(dirName + os.sep + "logCorr.dat", np.column_stack((stepList, particleCorr)))
        data = np.column_stack((stepList, particleCorr))
        tau = utils.getRelaxationTime(data)
        print("Relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
    if(plot=="plot"):
        stepList /= timeStep
        if(num1 != 0):
            uplot.plotCorrelation(stepList, particleCorr1[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k')
            uplot.plotCorrelation(stepList, particleCorr1[:,2], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'g')
            uplot.plotCorrelation(stepList, particleCorr2[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k', ls='dashed')
            uplot.plotCorrelation(stepList, particleCorr2[:,2], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'g', ls='dashed')
        else:
            uplot.plotCorrelation(stepList, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k')
            uplot.plotCorrelation(stepList, particleCorr[:,2], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'g')
        #plt.pause(0.5)
        plt.show()

############################ Velocity distribution #############################
def average2FluidsVelPDF(dirName, which='speed', num1=0, plot=False, figureName=None, dirSpacing=1):
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    #dirList = dirList[-50:]
    vel1 = np.empty(0)
    vel2 = np.empty(0)
    veltot = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        vel = np.loadtxt(dirSample + os.sep + "particleVel.dat")
        if(which == 'x'):
            velNorm = vel[:,0]
        elif(which == 'y'):
            velNorm = vel[:,1]
        else:
            velNorm = np.linalg.norm(vel, axis=1)
        vel1 = np.append(vel1, velNorm[:num1].flatten())
        vel2 = np.append(vel2, velNorm[num1:].flatten())
        veltot = np.append(veltot, velNorm.flatten())
    min = np.min(np.array([np.min(vel1), np.min(vel2), np.min(veltot)]))
    max = np.max(np.array([np.max(vel1), np.max(vel2), np.max(veltot)]))
    bins = np.linspace(min, max, 100)
    pdf1, edges = np.histogram(vel1, bins=bins, density=True)
    pdf2, edges = np.histogram(vel2, bins=bins, density=True)
    pdftot, edges = np.histogram(veltot, bins=bins, density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    np.savetxt(dirName + os.sep + "2velPDF.dat", np.column_stack((edges, pdf1, pdf2, pdftot)))
    mean = np.mean(vel1)
    temp = np.var(vel1)
    skewness = np.mean((vel1 - mean)**3)/temp**(3/2)
    kurtosis = np.mean((vel1 - mean)**4)/temp**2 - 3
    print("Variance of the velocity in fluid 1: ", temp, " kurtosis - 3: ", kurtosis, " skewness: ", skewness)
    #if(which == 'x' or which == 'y'): pdf1 /= norm.pdf(edges, loc=mean, scale=np.sqrt(temp))
    mean = np.mean(vel2)
    temp = np.var(vel2)
    skewness = np.mean((vel2 - mean)**3)/temp**(3/2)
    kurtosis = np.mean((vel2 - mean)**4)/temp**2 - 3
    print("Variance of the velocity in fluid 2: ", temp, " kurtosis - 3: ", kurtosis, " skewness: ", skewness)
    #if(which == 'x' or which == 'y'): pdf2 /= norm.pdf(edges, loc=mean, scale=np.sqrt(temp))
    mean = np.mean(veltot)
    temp = np.var(veltot)
    skewness = np.mean((veltot - mean)**3)/temp**(3/2)
    kurtosis = np.mean((veltot - mean)**4)/temp**2 - 3
    print("Variance of the velocity in total: ", temp, " kurtosis - 3: ", kurtosis, " skewness: ", skewness)
    #if(which == 'x' or which == 'y'): pdftot /= norm.pdf(edges, loc=mean, scale=np.sqrt(temp))
    if(plot == "plot"):
        if(which == 'x'):
            xlabel = "$v_x$"
            ylabel = "$PDF/G(\\langle v_x \\rangle, \\sigma_{v_x})$"
        elif(which == 'y'):
            xlabel = "$v_y$"
            ylabel = "$PDF/G(\\langle v_y \\rangle, \\sigma_{v_y})$"
        else:
            xlabel = "$Speed,$ $s = \\sqrt{v_x^2 + v_y^2}$"
            ylabel = "$PDF(s)$"
        #if(which == 'x' or which == 'y'):
        #    edges = edges[10:-10]
        #    pdf1 = pdf1[10:-10]
        #    pdf2 = pdf2[10:-10]
        #    pdftot = pdftot[10:-10]
        uplot.plotCorrelation(edges, pdf1, ylabel=ylabel, xlabel=xlabel, color='g', markersize=8, marker='v', fs='none')
        uplot.plotCorrelation(edges, pdf2, ylabel=ylabel, xlabel = xlabel, color='b', markersize=8, marker='D', fs='none')
        uplot.plotCorrelation(edges, pdftot, ylabel=ylabel, xlabel = xlabel, logy=True, color='k', markersize=8, fs='none')
        plt.legend(["$Phase$ $1$", "$Phase$ $2$", "$Total$"], fontsize=12, loc='best')
        plt.savefig('/home/francesco/Pictures/soft/mips/2velpdf-' + figureName + '-' + which + '.png', transparent=True, format = 'png')
        plt.pause(0.5)
        #plt.show()
    return veltot

####################### Compute Velocity Autocorrelation for the two fluids ########################
def average2FluidsVelCorr(dirName, startBlock, maxPower, freqPower, num1=0, plot=False):
    timeStep = utils.readFromParams(dirName, "dt")
    if(num1 != 0):
        particleVelCorr1 = []
        particleVelCorr2 = []
    else:
        particleVelCorr = []
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
            if(num1 != 0):
                stepParticleVelCorr1 = []
                numPairs1 = 0
                stepParticleVelCorr2 = []
                numPairs2 = 0
            else:
                stepParticleVelCorr = []
                numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        vel1, vel2 = utils.readVelPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        if(num1 != 0):
                            stepParticleVelCorr1.append(np.mean(np.sum(np.multiply(vel1[:num1],vel2[:num1]), axis=1))/np.mean(np.linalg.norm(vel1[:num1], axis=1)**2))
                            numPairs1 += 1
                            stepParticleVelCorr2.append(np.mean(np.sum(np.multiply(vel1[num1:],vel2[num1:]), axis=1))/np.mean(np.linalg.norm(vel1[num1:], axis=1)**2))
                            numPairs2 += 1
                        else:
                            stepParticleVelCorr.append(np.mean(np.sum(np.multiply(vel1,vel2), axis=1))/np.mean(np.linalg.norm(vel1, axis=1)**2))
                            numPairs += 1
            if(num1 != 0):
                if(numPairs1 > 0 or numPairs2 > 0):
                    stepList.append(spacing*spacingDecade)
                    particleVelCorr1.append([np.mean(stepParticleVelCorr1, axis=0), np.std(stepParticleVelCorr1, axis=0)])
                    particleVelCorr2.append([np.mean(stepParticleVelCorr2, axis=0), np.std(stepParticleVelCorr2, axis=0)])
            else:
                if(numPairs > 0):
                    stepList.append(spacing*spacingDecade)
                    particleVelCorr.append([np.mean(stepParticleVelCorr, axis=0), np.std(stepParticleVelCorr, axis=0)])
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList) * timeStep
    if(num1 != 0):
        particleVelCorr1 = np.array(particleVelCorr1).reshape((stepList.shape[0],2))
        particleVelCorr1 = particleVelCorr1[np.argsort(stepList)]
        particleVelCorr2 = np.array(particleVelCorr2).reshape((stepList.shape[0],2))
        particleVelCorr2 = particleVelCorr2[np.argsort(stepList)]
        np.savetxt(dirName + os.sep + "2velCorr.dat", np.column_stack((stepList, particleVelCorr1, particleVelCorr2)))
        data = np.column_stack((stepList, particleVelCorr1))
        tau = utils.getRelaxationTime(data, index=1)
        print("Fluid 1: velocity relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
        data = np.column_stack((stepList, particleVelCorr2))
        tau = utils.getRelaxationTime(data, index=1)
        print("Fluid 2: velocity relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
    else:
        particleVelCorr = np.array(particleVelCorr).reshape((stepList.shape[0],2))
        particleVelCorr = particleVelCorr[np.argsort(stepList)]
        np.savetxt(dirName + os.sep + "velCorr.dat", np.column_stack((stepList, particleVelCorr)))
        data = np.column_stack((stepList, particleVelCorr))
        tau = utils.getRelaxationTime(data, index=1)
        print("Velocity relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
    if(plot=="plot"):
        stepList /= timeStep
        if(num1 != 0):
            uplot.plotCorrWithError(stepList, particleVelCorr1[:,0], particleVelCorr1[:,1], "$C_{vv}(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'b')
            uplot.plotCorrWithError(stepList, particleVelCorr2[:,0], particleVelCorr2[:,1], "$C_{vv}(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'g')
        else:
            uplot.plotCorrWithError(stepList, particleVelCorr[:,0], particleVelCorr[:,1], "$C_{vv}(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k')
        plt.pause(0.5)
        #plt.show()

@njit
def pbc_delta(pos_i, pos_j, boxSize):
    delta = pos_i - pos_j
    delta += boxSize / 2
    delta %= boxSize
    delta -= boxSize / 2
    return delta

@njit
def compute_velocity_correlations_2D(pos, vel, bins, boxSize):
    num = pos.shape[0]
    vel_corr = np.zeros((len(bins) - 1, 4))
    count = np.zeros(len(bins) - 1, dtype=np.int64)

    for i in range(num):
        for j in range(i + 1, num):  # Only consider each unique pair once
            r_ij = pbc_delta(pos[i], pos[j], boxSize)
            dist = np.sqrt(r_ij[0]**2 + r_ij[1]**2)

            if dist >= bins[-1]:
                continue

            bin_idx = np.searchsorted(bins, dist) - 1
            if bin_idx < 0 or bin_idx >= len(bins) - 1:
                continue

            delta = r_ij / dist
            deltaPerp = np.array([-delta[1], delta[0]])

            parProj_i = np.dot(vel[i], delta)
            parProj_j = np.dot(vel[j], delta)
            perpProj_i = np.dot(vel[i], deltaPerp)
            perpProj_j = np.dot(vel[j], deltaPerp)

            vel_corr[bin_idx, 0] += parProj_i * parProj_j
            vel_corr[bin_idx, 1] += perpProj_i * perpProj_j
            vel_corr[bin_idx, 2] += 0.5 * (perpProj_i * parProj_j + parProj_i * perpProj_j)
            vel_corr[bin_idx, 3] += np.dot(vel[i], vel[j])
            count[bin_idx] += 1
    
    return vel_corr, count

############################# Velocity Correlation #############################
def average2FluidsSpaceVelCorr(dirName, num1=0, plot=False, dirSpacing=1000000):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    print("boxSize: ", boxSize)
    sigma = 2*np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    bins = np.arange(0, 4*sigma, sigma/10)
    print(bins[1]-bins[0], bins.shape[0])
    binCenter = (bins[1:] + bins[:-1])/2
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    print(dirList)
    # divide system in horizontal slices to manage RAM occupancy
    yList = np.linspace(0,boxSize[1],6)
    if(num1 != 0):
        velCorr1 = np.zeros((yList.shape[0], bins.shape[0]-1, 4))
        velCorr2 = np.zeros((yList.shape[0], bins.shape[0]-1, 4))
        counts1 = np.zeros((yList.shape[0], bins.shape[0]-1))
        counts2 = np.zeros((yList.shape[0], bins.shape[0]-1))
    else:
        velCorr = np.zeros((yList.shape[0], bins.shape[0]-1, 4))
        counts = np.zeros((yList.shape[0], bins.shape[0]-1))
    for d in range(dirList.shape[0]):
        print(dirList[d])
        pos = utils.getPBCPositions(dirName + os.sep + dirList[d] + os.sep + "particlePos.dat", boxSize)
        vel = np.array(np.loadtxt(dirName + os.sep + dirList[d] + os.sep + "particleVel.dat"))
        velNorm = np.linalg.norm(vel, axis=1)
        velNormSquared = np.mean(velNorm**2)
        for y in range(1, yList.shape[0]):
            slicedVel = vel[np.argwhere(pos[:,1] < yList[y])[:,0]]
            slicedPos = pos[np.argwhere(pos[:,1] < yList[y])[:,0]]
            slicedVel = slicedVel[np.argwhere(slicedPos[:,1] > yList[y-1])[:,0]]
            slicedPos = slicedPos[np.argwhere(slicedPos[:,1] > yList[y-1])[:,0]]
            print("Computing correlations in slice between: ", yList[y-1], " and ", yList[y], " with ", slicedPos.shape[0], " particles")
            if (num1 != 0):
                # type 1
                pos1 = slicedPos[:num1]
                vel1 = slicedVel[:num1]
                velCorr1[y], counts1[y] = compute_velocity_correlations_2D(pos1, vel1, bins, boxSize)
                # type 2
                pos2 = slicedPos[num1:]
                vel2 = slicedVel[num1:]
                velCorr2[y], counts2[y] = compute_velocity_correlations_2D(pos2, vel2, bins, boxSize)
            else:
                velCorr[y], counts[y] = compute_velocity_correlations_2D(slicedPos, slicedVel, bins, boxSize)
    if(num1 != 0):
        velCorr1 = np.sum(velCorr1, axis=0)
        velCorr2 = np.sum(velCorr2, axis=0)
        counts1 = np.sum(counts1, axis=0)
        counts2 = np.sum(counts2, axis=0)
        for i in range(velCorr1.shape[1]):
            velCorr1[counts1>0,i] /= counts1[counts1>0]
            velCorr2[counts2>0,i] /= counts2[counts2>0]
        velCorr1 /= velNormSquared
        velCorr2 /= velNormSquared
        np.savetxt(dirName + os.sep + "2spaceVelCorr.dat", np.column_stack((binCenter, velCorr1, velCorr2, counts1, counts2)))
    else:
        velCorr = np.sum(velCorr, axis=0)
        counts = np.sum(counts, axis=0)
        for i in range(velCorr.shape[1]):
            velCorr[counts>0,i] /= counts[counts>0]
        velCorr /= velNormSquared
        np.savetxt(dirName + os.sep + "spaceVelCorr.dat", np.column_stack((binCenter, velCorr, counts)))
    if plot == 'plot':
        if(num1 != 0):
            uplot.plotCorrelation(binCenter, velCorr1[:,0], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'r')
            uplot.plotCorrelation(binCenter, velCorr1[:,1], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'g')
            uplot.plotCorrelation(binCenter, velCorr1[:,2], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'k')
        else:
            uplot.plotCorrelation(binCenter, velCorr[:,0], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'r')
            uplot.plotCorrelation(binCenter, velCorr[:,1], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'g')
            uplot.plotCorrelation(binCenter, velCorr[:,2], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'k')
        #plt.pause(0.5)
        plt.show()

def compute_mobility(positions, delta_t):
    # positions: shape (T, N, dim)
    displacements = positions[delta_t:] - positions[:-delta_t]
    mobility = np.linalg.norm(displacements, axis=-1)  # shape: (T - delta_t, N)
    return mobility

def minimum_image_distance(ri, pos_t, boxSize):
    delta = pos_t - ri  # shape: (N, d)
    delta += boxSize / 2
    delta %= boxSize
    delta -= boxSize / 2
    return delta

def compute_spatial_mobility_correlation_PBC(wrapped_positions, unwrapped_positions, box_size, delta_t, r_max, dr):
    """
    wrapped_positions: shape (T, N, dim) — for distances
    unwrapped_positions: shape (T, N, dim) — for mobility
    box_size: array of length dim
    """
    T, N, dim = wrapped_positions.shape
    mobility = np.linalg.norm(unwrapped_positions[delta_t:] - unwrapped_positions[:-delta_t], axis=-1)  # (T - delta_t, N)
    
    r_bins = np.arange(0, r_max + dr, dr)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    g_r = np.zeros_like(r_centers)
    counts = np.zeros_like(r_centers)

    print("Number of time windows:", T - delta_t)
    for t in range(T - delta_t):
        print("Time window index:", t)
        pos_t = wrapped_positions[t]                # shape (N, dim)
        mu_t = mobility[t]                          # shape (N,)
        mu_fluct = mu_t - np.mean(mu_t)

        for i in range(N):
            #if i%1e04 == 0: print("Computing correlations of particle:", i)
            ri = pos_t[i]
            mui = mu_fluct[i]

            rij = minimum_image_distance(ri, pos_t, box_size)    # (N, dim)
            dist = np.linalg.norm(rij, axis=-1)                  # (N,)
            muj = mu_fluct
            correl = mui * muj

            dist[i] = np.inf   # exclude self
            correl[i] = 0.0

            bin_indices = np.floor(dist / dr).astype(int)
            for j in range(N):
                if j != i:
                    # Exclude self
                    b = bin_indices[j]
                    if b < len(g_r) and b >= 0:
                        g_r[b] += correl[j]
                        counts[b] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        g_r = np.where(counts > 0, g_r / counts, 0)

    return r_centers, g_r

def computeMobilityCorrelation(dirName, delta_t=10, plot=False, dirSpacing=100000):
    # Load box size
    box_size = np.array(np.loadtxt(os.path.join(dirName, "boxSize.dat")))
    dt = utils.readFromParams(dirName, "dt")
    sigma = 2*np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    r_max = 10*sigma
    dr = sigma/10
    # Get sorted directories and times
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    
    # Filter time points based on dirSpacing
    selected_indices = np.argwhere(timeList % dirSpacing == 0)[:, 0]
    dirList = dirList[selected_indices]
    print("Selected delta_t:", delta_t, "frame interval:", timeList[delta_t] - timeList[0], "time interval:", (timeList[delta_t] - timeList[0])*dt)

    # Load positions for each frame and stack
    wrapped_positions_list = []
    unwrapped_positions_list = []

    for d in dirList:
        dirSample = os.path.join(dirName, d)
        # Load wrapped and unwrapped positions
        wrapped = utils.getPBCPositions(os.path.join(dirSample, "particlePos.dat"), box_size)
        unwrapped = np.loadtxt(os.path.join(dirSample, "particlePos.dat"))  # assume shape (N, dim)
        wrapped_positions_list.append(wrapped)
        unwrapped_positions_list.append(unwrapped)

    # Convert to arrays
    wrapped_positions = np.stack(wrapped_positions_list, axis=0)      # shape: (T, N, dim)
    unwrapped_positions = np.stack(unwrapped_positions_list, axis=0)  # shape: (T, N, dim)

    # Compute spatial mobility correlation
    r_centers, g_r = compute_spatial_mobility_correlation_PBC(wrapped_positions,   # shape: (T, N, dim)
                                                              unwrapped_positions, # shape: (T, N, dim)
                                                              box_size=box_size,   # shape: (dim,)
                                                              delta_t=delta_t,     # time window for mobility
                                                              r_max=r_max,         # max distance for correlation
                                                              dr=dr                # bin size
                                                              )
    # Save results
    np.savetxt(os.path.join(dirName, "mobilityCorr-delta" + str(delta_t) + ".dat"), np.column_stack((r_centers, g_r)))
    # Plot if requested
    if plot:
        plt.plot(r_centers, g_r, color='k')
        plt.xlabel("Distance r")
        plt.ylabel("Mobility correlation G₄(r)")
        plt.title("Spatial correlation of mobility")
        plt.tight_layout()
        plt.pause(0.5)

    return r_centers, g_r

####################### Average interface fluctuations in binary mixture #######################
def average2InterfaceFluctuations(dirName, num1=0, thickness=3, plot=False, dirSpacing=100000):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    sigma = 2 * np.mean(rad)
    spacing = 3 * sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1]) * 0.5
    freq = np.fft.rfftfreq(bins.shape[0]-1, spacing)
    leftDeltaHeight = np.zeros((dirList.shape[0], bins.shape[0]-1))
    leftFourierDeltaHeight = np.zeros((dirList.shape[0], freq.shape[0]))
    rightDeltaHeight = np.zeros((dirList.shape[0], bins.shape[0]-1))
    rightFourierDeltaHeight = np.zeros((dirList.shape[0], freq.shape[0]))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = utils.centerCOM1(pos, rad, boxSize, num1)
        leftHeight = np.zeros(bins.shape[0]-1)
        rightHeight = np.zeros(bins.shape[0]-1)
        clusterPos = pos[:num1] #select particles of type A
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            downMask = np.argwhere(clusterPos[:,1] > bins[j])[:,0]
            binPos = clusterPos[downMask]
            upMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
            binPos = binPos[upMask]
            if(binPos.shape[0] > 0):
                center = np.mean(binPos, axis=0)[0] # center of dense cluster
                binDistance = binPos[:,0] - center
                leftMask = np.argsort(binDistance)
                leftMask = np.argsort(binDistance)[:thickness]
                leftHeight[j] = np.mean(binDistance[leftMask])
                rightMask = np.argsort(binDistance)[-thickness:]
                rightHeight[j] = np.mean(binDistance[rightMask])
        # use only continuous interfaces
        if(leftHeight[leftHeight!=0].shape[0] == leftHeight.shape[0]):
            leftHeight -= np.mean(leftHeight)
            leftDeltaHeight[d] = np.abs(leftHeight)**2
            leftFourierDeltaHeight[d] = np.abs(np.fft.rfft(leftHeight))**2
        if(rightHeight[rightHeight!=0].shape[0] == rightHeight.shape[0]):
            rightHeight -= np.mean(rightHeight)
            rightDeltaHeight[d] = np.abs(rightHeight)**2
            rightFourierDeltaHeight[d] = np.abs(np.fft.rfft(rightHeight))**2

    leftDeltaHeight = np.column_stack((np.mean(leftDeltaHeight, axis=0), np.std(leftDeltaHeight, axis=0)))
    rightDeltaHeight = np.column_stack((np.mean(rightDeltaHeight, axis=0), np.std(rightDeltaHeight, axis=0)))
    leftFourierDeltaHeight = np.column_stack((np.mean(leftFourierDeltaHeight, axis=0), np.std(leftFourierDeltaHeight, axis=0)))
    rightFourierDeltaHeight = np.column_stack((np.mean(rightFourierDeltaHeight, axis=0), np.std(rightFourierDeltaHeight, axis=0)))
    fourierDeltaHeight = np.zeros(leftFourierDeltaHeight.shape)
    fourierDeltaHeight[:,0] = np.mean(np.column_stack((leftFourierDeltaHeight[:,0],rightFourierDeltaHeight[:,0])), axis=1)
    fourierDeltaHeight[:,1] = np.sqrt(np.mean(np.column_stack((leftFourierDeltaHeight[:,1]**2,rightFourierDeltaHeight[:,1]**2)), axis=1))
    
    np.savetxt(dirName + os.sep + "fourierFluctuations.dat", np.column_stack((freq, fourierDeltaHeight)))
    np.savetxt(dirName + os.sep + "interfaceFluctuations.dat", np.column_stack((centers, leftDeltaHeight, rightDeltaHeight)))
    if(plot=='plot'):
        uplot.plotCorrWithError(freq[1:], fourierDeltaHeight[1:,0], fourierDeltaHeight[1:,1], "$Height$ $fluctuation$", xlabel = "$Wave$ $vector$ $magnitude,$ $q$", color='k', logx=True, logy=True)
        #uplot.plotCorrWithError(centers, leftDeltaHeight[:,0], leftDeltaHeight[:,1], "$Height$ $fluctuation$", xlabel = "$y$", color='k')
        #uplot.plotCorrWithError(centers, rightDeltaHeight[:,0], rightDeltaHeight[:,1], "$Height$ $fluctuation$", xlabel = "$y$", color='g')
        #plt.pause(0.5)
        plt.show()

####################### Average cluster height interface #######################
def get2InterfaceLength(dirName, num1=0, spacing='2', window=3, mixed=0, plot=False, lj=True):
    if mixed == 'mixed':
        lengthType = "MixedLength"
    else:
        lengthType = "Length"
        mixedLength = 0
    if spacing != 3 or window != 2:
        lengthType += f"-s{spacing}-w{window}"
    #print(lengthType)
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + "../particleRad.dat"))
    numParticles = int(utils.readFromParams(dirName + sep, "numParticles"))
    # load particle variables
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    if lj:
        rad *= 2**(1/6)
    sigma = 2*np.mean(rad)
    #meanArea = np.pi*(sigma/2)**2
    eps = 1.4 * np.max(rad)
    spacing = float(spacing) * sigma # vertical
    #spacing = utils.getPairCorrelationPeakLocation(dirName)
    thickness = 3 # horizontal
    bins = np.arange(0, boxSize[1], spacing)
    rightInterface = np.zeros(bins.shape[0]-1)
    leftInterface = np.zeros(bins.shape[0]-1)
    labels = np.zeros(numParticles)
    labels[:num1] = 1
    clusterLabels, maxLabel = cluster.getTripleWrappedClusterLabels(pos, rad, boxSize, labels, eps)
    pos = utils.centerSlab(pos, rad, boxSize, clusterLabels, maxLabel)
    #print(maxLabel, clusterLabels[clusterLabels==maxLabel].shape[0])
    typePos = pos[clusterLabels==maxLabel]
    leftPos = np.zeros((bins.shape[0]-1,2))
    rightPos = np.zeros((bins.shape[0]-1,2))
    for j in range(bins.shape[0]-1): # find particle positions in a vertical bin
        topMask = np.argwhere(typePos[:,1] > bins[j])[:,0]
        binPos = typePos[topMask]
        bottomMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
        binPos = binPos[bottomMask]
        if(binPos.shape[0] > 0):
            binDistance = binPos[:,0]
            #left interface
            leftMask = np.argsort(binDistance)[:thickness]
            leftInterface[j] = np.mean(binDistance[leftMask])
            leftPos[j,0] = leftInterface[j]
            leftPos[j,1] = np.mean(binPos[leftMask,1])
            # right interface
            rightMask = np.argsort(binDistance)[-thickness:]
            rightInterface[j] = np.mean(binDistance[rightMask])
            rightPos[j,0] = rightInterface[j]
            rightPos[j,1] = np.mean(binPos[rightMask,1])
    if(window > 1):
        leftPos = np.column_stack((utils.computeMovingAverage(leftPos[:,0], window), utils.computeMovingAverage(leftPos[:,1], window)))
        rightPos = np.column_stack((utils.computeMovingAverage(rightPos[:,0], window), utils.computeMovingAverage(rightPos[:,1], window)))
    length = 0
    if(rightInterface[rightInterface!=0].shape[0] == rightInterface.shape[0]):
        prevPos = rightPos[0]
        for j in range(1,bins.shape[0]-1):
            length += np.linalg.norm(rightPos[j] - prevPos)
            prevPos = rightPos[j]
    if(leftInterface[leftInterface!=0].shape[0] == leftInterface.shape[0]):
        prevPos = leftPos[0]
        for j in range(1,bins.shape[0]-1):
            length += np.linalg.norm(leftPos[j] - prevPos)
            prevPos = leftPos[j]
    # Add lengths of mixed regions
    if mixed == 'mixed':
        uniqueLabels = np.unique(clusterLabels)
        uniqueLabels = np.delete(uniqueLabels, np.argwhere(uniqueLabels==maxLabel)[0,0])
        mixedNum = 0
        mixedLength = 0
        for c in range(2,uniqueLabels.shape[0]):
            numCluster = clusterLabels[clusterLabels==uniqueLabels[c]].shape[0]
            radCluster = rad[clusterLabels==uniqueLabels[c]]
            mixedNum += numCluster
            mixedLength +=  np.sum(radCluster)
            #mixedLength += utils.getClusterLength(numCluster, sigma, meanArea)
            #mixedLength += mixedNum * np.pi * sigma
            #print("type1 ", c, clusterLabels[clusterLabels==uniqueLabels[c]].shape[0])
        # clusters of second particle type
        pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
        labels = np.zeros(numParticles)
        labels[num1:] = 1
        clusterLabels, maxLabel = cluster.getTripleWrappedClusterLabels(pos, rad, boxSize, labels, eps)
        uniqueLabels = np.unique(clusterLabels)
        uniqueLabels = np.delete(uniqueLabels, np.argwhere(uniqueLabels==maxLabel)[0,0])
        for c in range(2,uniqueLabels.shape[0]):
            numCluster = clusterLabels[clusterLabels==uniqueLabels[c]].shape[0]
            radCluster = rad[clusterLabels==uniqueLabels[c]]
            mixedNum += numCluster
            mixedLength +=  np.sum(radCluster)
            #mixedLength += utils.getClusterLength(numCluster, sigma, meanArea)
            #mixedLength += mixedNum * np.pi * sigma
            #print("type2 ", c, clusterLabels[clusterLabels==uniqueLabels[c]].shape[0])
        length += mixedLength
    np.savetxt(dirName + os.sep + "interface" + lengthType + ".dat", np.column_stack((length, length - mixedLength, mixedLength)))
    if(plot == "plot"):
        #print("Number of mixed particles:", mixedNum, "length:", mixedLength)
        print("Interface length:", length)
        #print("Without mixed length:", length - mixedLength)
        if(boxSize[0] > boxSize[1]):
            fig, ax = plt.subplots(figsize=(2.5*boxSize[0]/boxSize[1], 3), dpi = 120)
        else:
            fig, ax = plt.subplots(figsize=(3, 3*boxSize[1]/boxSize[0]), dpi = 120)
        ax.set_xlim(-0.02*boxSize[0],1.02*boxSize[0])
        ax.set_ylim(0,boxSize[1])
        ax.plot(leftPos[:,0], leftPos[:,1], color='g', marker='o', markersize=4, fillstyle='none', lw=1)
        ax.plot(rightPos[:,0], rightPos[:,1], color='g', marker='o', markersize=4, fillstyle='none', lw=1)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        fig.tight_layout()
        #plt.pause(0.5)
        plt.show()
    return length

def bin_and_average(length, etot, epot, temp, bin_size=1.0, bin_by='length'):
    '''
    Bin the data by either length or etot and return the average values per bin
    '''
    if bin_by == 'etot':
        bin_values = etot
        other_values = length
    elif bin_by == 'epot':
        bin_values = epot
        other_values = length
        energy_values = etot
    elif bin_by == 'temp':
        bin_values = temp
        other_values = length
        energy_values = etot
    else:
        bin_values = length
        other_values = etot

    # Create bin edges
    min_val = np.min(bin_values)
    max_val = np.max(bin_values)
    bins = np.arange(min_val, max_val + bin_size, bin_size)

    # Assign each value to a bin
    bin_indices = np.digitize(bin_values, bins) - 1  # make bins zero-indexed

    # Store average values per bin
    avg_bin_values = np.empty(0)
    err_bin_values = np.empty(0)
    avg_other_values = np.empty(0)
    err_other_values = np.empty(0)
    if bin_by == 'epot' or bin_by == 'temp':
        avg_energy_values = np.empty(0)
        err_energy_values = np.empty(0)

    for i in range(len(bins) - 1):
        in_bin = bin_indices == i
        if np.any(in_bin):  # skip empty bins
            avg_bin_values = np.append(avg_bin_values, np.mean(bin_values[in_bin]))
            err_bin_values = np.append(err_bin_values, np.std(bin_values[in_bin]))
            avg_other_values = np.append(avg_other_values, np.mean(other_values[in_bin]))
            err_other_values = np.append(err_other_values, np.std(other_values[in_bin]))
            if bin_by == 'epot' or bin_by == 'temp':
                avg_energy_values = np.append(avg_energy_values, np.mean(energy_values[in_bin]))
                err_energy_values = np.append(err_energy_values, np.std(energy_values[in_bin]))


    # Always return (length, etot)
    if bin_by == 'etot':
        return np.column_stack((avg_other_values, err_other_values)), np.column_stack((avg_bin_values, err_bin_values))
    elif bin_by == 'epot':
        return np.column_stack((avg_other_values, err_other_values)), np.column_stack((avg_energy_values, err_energy_values))
    elif bin_by == 'temp':
        return np.column_stack((avg_other_values, err_other_values)), np.column_stack((avg_energy_values, err_energy_values))
    else:
        return np.column_stack((avg_bin_values, err_bin_values)), np.column_stack((avg_other_values, err_other_values))

# Define linear fit function
def poly1(x, a, b):
    return a * x + b

def saveSignalOverFloor(signal, noise, signal_length, noise_length, temp, dirName, dynamics='dynamics', dynType='nve'):
    if dynType == 'nve':
        file_path = os.path.join(dirName, f"../../../signalFloor-nve.dat")
        data = np.array([np.mean(temp), signal, noise, signal_length, noise_length])
    elif dynType == 'nvt':
        file_path = os.path.join(dirName, f"../../../signalFloor-nvt.dat")
        damping = utils.readFromDynParams(dirName + os.sep + 'strain0.0100' + os.sep + dynamics, "damping")
        data = np.array([np.mean(temp), signal, noise, signal_length, noise_length, damping])
    else:
        file_path = os.path.join(dirName, f"../../../signalFloor-" + dynType + ".dat")
        data = np.array([np.mean(temp), signal, noise, signal_length, noise_length])
    # Check if file exists
    file_exists = os.path.isfile(file_path)
    # Open in append mode and write
    with open(file_path, 'a') as f:
        if not file_exists:
            # Write header only once
            if dynType == 'nve':
                f.write("# mean_temp  std_temp  slope  slope_err\n")
            elif dynType == 'nvt':
                f.write("# damping  mean_temp  std_temp  slope  slope_err\n")
            else:
                f.write("# mean_temp  std_temp  slope  slope_err\n")
        np.savetxt(f, data.reshape(1, -1), fmt="%.6e")

def saveLineTension(dirName, slope, slope_err, temp, dynamics='dynamics', dynType='nve', mixed='mixed', ab=False):
    # Save line tension data
    if ab == 'ab':
        fileName = "lineTension-ab-"
    else:
        fileName = "lineTension-"
    if mixed == 'mixed':
        file_path = os.path.join(dirName, f"../../../" + fileName + dynType + "-mixed.dat")
    else:
        file_path = os.path.join(dirName, f"../../../" + fileName + dynType + ".dat")
    if(dynType == 'nvt' or dynType == 'nvt-l1' or dynType == 'nvt-l2' or dynType == 'nvt-l1-long' or dynType == 'nvt-l2-long'):
        damping = utils.readFromDynParams(dirName + os.sep + 'strain0.0100' + os.sep + dynamics, "damping")
        data = np.array([np.mean(temp), np.std(temp), slope, slope_err, damping])
    else:
        data = np.array([np.mean(temp), np.std(temp), slope, slope_err])
    # Check if file exists
    file_exists = os.path.isfile(file_path)
    print(file_path)
    # Open in append mode and write
    with open(file_path, 'a') as f:
        if not file_exists:
            # Write header only once
            if dynType == 'nve':
                f.write("# mean_temp  std_temp  slope  slope_err\n")
            elif dynType == 'nvt':
                f.write("# damping  mean_temp  std_temp  slope  slope_err\n")
            else:
                f.write("# mean_temp  std_temp  slope  slope_err\n")
        np.savetxt(f, data.reshape(1, -1), fmt="%.6e")

def plotEnergyVSHeight(dirName, dynamics='dynamics', dynType='nve', which='etot', figureName='nve', show=False):
    '''
    plot energy and box height data for all the values of strain and time
    '''
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Box$ $height,$ $H$", fontsize=14)
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    strainList, strain = utils.getOrderedStrainDirectories(dirName)
    epot = np.empty(0)
    ekin = np.empty(0)
    etot = np.empty(0)
    heat = np.empty(0)
    height = np.empty(0)
    count = np.zeros(strainList.shape[0])
    for s in range(strainList.shape[0]):
        dirStrain = dirName + os.sep + strainList[s] + os.sep + dynamics + os.sep
        # Read energy and timestep data
        if(dynType == 'nve' or dynType == 'nh'):
            energy = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(2,3,4))) / utils.readFromParams(dirStrain, "epsilon")
        elif(dynType == 'nvt' or dynType == 'nvtvst' or dynType == 'nvt-l1' or dynType == 'nvt-l2' or dynType == 'nvt-l1-long' or dynType == 'nvt-l2-long'):
            energy = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(2,3,4,7))) / utils.readFromParams(dirStrain, "epsilon")
        else:
            energy = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(2,3,4,8))) / utils.readFromParams(dirStrain, "epsilon")
        epot = np.append(epot, energy[:,0])
        ekin = np.append(ekin, energy[:,1])
        etot = np.append(etot, energy[:,2])
        count[s] = energy.shape[0]
        if(dynType == 'nve' or dynType == 'nh'):
            heat = np.append(heat, 0)
        else:
            heat = np.append(heat, energy[:,3])
        boxSize = np.loadtxt(dirStrain + "/boxSize.dat")
        height = np.append(height, np.ones(energy.shape[0]) * boxSize[1])
        print(strainList[s])
    if which == 'epot':
        etot = epot
        ax.set_ylabel("$U$", fontsize=14, rotation='horizontal')
    elif which == 'ekin':
        ax.set_ylabel("$K$", fontsize=14, rotation='horizontal')
        etot = ekin
    elif which == 'heat':
        ax.set_ylabel("$Q$", fontsize=14, rotation='horizontal')
        etot = heat
    else:
        ax.set_ylabel("$E$", fontsize=14)
        etot = etot # + np.mean(ekin) + np.mean(heat)
    # Group length and energy data by strain
    meanEtot = np.zeros((strainList.shape[0], 2))
    meanHeight = np.zeros(strainList.shape[0])
    for c in range(count.shape[0]):
        if c == 0:
            meanEtot[c,0] = np.mean(etot[0:int(count[c])])
            meanEtot[c,1] = np.std(etot[0:int(count[c])])
            meanHeight[c] = np.mean(height[0:int(count[c])])
        else:
            meanEtot[c,0] = np.mean(etot[int(np.sum(count[0:c])):int(np.sum(count[0:c+1]))])
            meanEtot[c,1] = np.std(etot[int(np.sum(count[0:c])):int(np.sum(count[0:c+1]))])
            meanHeight[c] = np.mean(height[int(np.sum(count[0:c])):int(np.sum(count[0:c+1]))])
    ax.plot(height, etot, color='k', marker='o', markersize=4, fillstyle='none', lw=0, label='Raw Data')
    ax.errorbar(meanHeight, meanEtot[:,0], meanEtot[:,1], color='g', marker='s', markersize=8, fillstyle='none', lw=1, capsize=3, label='Average')
    # Fit curve to polynomial and get first-power coefficient
    x = meanHeight
    y = meanEtot[:,0] # Subtract first value to normalize
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    popt, pcov = curve_fit(poly1, x, y)
    slope, intercept = popt
    slope_err = np.sqrt(np.diag(pcov))[0]
    y_fit = poly1(x_fit, slope, intercept)
    print(f"Slope = {slope} ± {slope_err}")
    temp = ekin / numParticles
    print(f"Temperature = {np.mean(temp)} ± {np.std(temp)}, range = [{np.min(temp)} - {np.max(temp)}]")
    if(np.abs(slope) < 1e-03):
        ax.plot(x_fit, y_fit, color='r', lw=1, label=f'Linear Fit, slope={slope:.2e}±{slope_err:.2e}')
    else:
        ax.plot(x_fit, y_fit, color='r', lw=1, label=f'Linear Fit, slope={slope:.2f}±{slope_err:.2f}')
    ax.legend(loc='best', fontsize=12)
    fig.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/mips/" + which + "Height-" + figureName + ".png", dpi=120)
    if show == 'show':
        plt.show()
    else:
        plt.pause(0.5)

def plotLengthVSTime(dirName, dynamics='dynamics', spacing='3', window=2, mixed='mixed', show=False):
    '''
    plot interface length vs time for a set of strain values
    '''
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("$Length,$ $L$", fontsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=14)
    if mixed == 'mixed':
        lengthType = "MixedLength"
    else:
        lengthType = "Length"
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    num1 = int(utils.readFromParams(dirName, 'num1'))
    strainList = np.array(['strain0.0200', 'strain0.0800', 'strain0.1400'])
    colorList = ['b', 'g', 'c']
    for s in range(strainList.shape[0]):
        if(os.path.exists(dirName + os.sep + strainList[s] + os.sep + dynamics + os.sep)):
            dirStrain = dirName + os.sep + strainList[s] + os.sep + dynamics + os.sep
            dirList, timeList = utils.getOrderedDirectories(dirStrain)
            count = 0
            length = np.empty(0)
            time = np.empty(0)
            for t in range(dirList.shape[0]):
                dirSample = dirStrain + dirList[t] + os.sep
                if(os.path.exists(dirSample)):
                    # Collect length data at current time
                    collect = False
                    if (os.path.exists(dirSample + "particlePos.dat")):
                        pos = np.loadtxt(dirSample + "particlePos.dat")
                        rad = np.loadtxt(dirSample + "../particleRad.dat")
                        if (pos.shape[0] == numParticles and rad.shape[0] == numParticles):
                            collect = True
                            if not(os.path.exists(dirSample + "interface" + lengthType + ".dat")):
                                get2InterfaceLength(dirSample, num1, spacing, window, mixed=mixed)
                            currentLength = np.loadtxt(dirSample + "interface" + lengthType + ".dat")[0]
                        else:
                            print("Warning: number of particles changed in", dirSample, pos.shape[0], rad.shape[0])
                        if np.isnan(currentLength):
                            print("Warning: NaN length found in", dirSample + "interface" + lengthType + ".dat")
                            collect = False
                    if collect:
                        count += 1
                        length = np.append(length, currentLength)
                        time = np.append(time, timeList[t])
            print(strainList[s], "collected", count, "relative length fluctuation", np.std(length)/np.mean(length))
            ax.plot(time, length, color=colorList[s], lw=1, label=f'Strain {strainList[s][-6:]}')
    ax.legend(loc='best', fontsize=12)
    fig.tight_layout()
    if show == 'show':
        plt.show()
    else:
        plt.pause(0.5)

def groupEnergyLengthVSStrain(dirName, dynamics='dynamics', dynType='nve', fileName='nve', spacing='1', window=5, bin_by='length', mixed='mixed', degree=1, show=False):
    '''
    group energy and length data for all the values of strain and time
    '''
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Length,$ $L$", fontsize=14)
    ax.set_ylabel("$\\frac{E}{N}$", fontsize=20, rotation='horizontal', labelpad=15)
    if mixed == 'mixed':
        lengthType = "MixedLength"
    else:
        lengthType = "Length"
    if spacing != 3 or window != 2:
        lengthType += f"-s{spacing}-w{window}"
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    num1 = int(utils.readFromParams(dirName, 'num1'))
    if not(os.path.exists(dirName + os.sep + "energy" + lengthType + "-" + fileName + ".dat")):
        strainList, _ = utils.getOrderedStrainDirectories(dirName)
        strainList = strainList[:16]  # limit to first 16 strains to avoid large deformations
        epot = np.empty(0)
        ekin = np.empty(0)
        etot = np.empty(0)
        heat = np.empty(0)
        length = np.empty(0)
        for s in range(strainList.shape[0]):
            dirStrain = dirName + os.sep + strainList[s] + os.sep + dynamics + os.sep
            # Read energy and timestep data
            if(dynType == 'nve' or dynType == 'nh'):
                energy = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(2,3,4))) / utils.readFromParams(dirStrain, "epsilon")
            elif(dynType == 'nvt' or dynType == 'nvtvst' or dynType == 'nvt-l1' or dynType == 'nvt-l2' or dynType == 'nvt-l1-long' or dynType == 'nvt-l2-long'):
                energy = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(2,3,4,7))) / utils.readFromParams(dirStrain, "epsilon")
            else:
                energy = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(2,3,4,8))) / utils.readFromParams(dirStrain, "epsilon")
            dirList, _ = utils.getOrderedDirectories(dirStrain)
            count = 0
            for t in range(dirList.shape[0]):
                dirSample = dirStrain + dirList[t] + os.sep
                if(os.path.exists(dirSample)):
                    # Collect length data at current time
                    collect = False
                    if (os.path.exists(dirSample + "particlePos.dat")):
                        pos = np.loadtxt(dirSample + "particlePos.dat")
                        rad = np.loadtxt(dirSample + "../particleRad.dat")
                        if (pos.shape[0] == numParticles and rad.shape[0] == numParticles):
                            collect = True
                            if not(os.path.exists(dirSample + "interface" + lengthType + ".dat")):
                                get2InterfaceLength(dirSample, num1, spacing, window, mixed=mixed)
                            currentLength = np.loadtxt(dirSample + "interface" + lengthType + ".dat")[0]
                        else:
                            print("Warning: number of particles changed in", dirSample, pos.shape[0], rad.shape[0])
                        if np.isnan(currentLength):
                            print("Warning: NaN length found in", dirSample + "interface" + lengthType + ".dat")
                            collect = False
                    if collect:
                        count += 1
                        length = np.append(length, currentLength)
                        # Collect energy data at current time
                        epot = np.append(epot, energy[t,0])
                        ekin = np.append(ekin, energy[t,1])
                        etot = np.append(etot, energy[t,2])
                        if(dynType == 'nve' or dynType == 'nh'):
                            heat = np.append(heat, 0)
                        else:
                            heat = np.append(heat, energy[t,3])
            print(strainList[s], count)
        np.savetxt(dirName + os.sep + "energy" + lengthType + "-" + fileName + ".dat", np.column_stack((length, etot, epot, ekin, heat)))
    else:
        length, etot, epot, ekin, heat = np.loadtxt(dirName + os.sep + "energy" + lengthType + "-" + fileName + ".dat", unpack=True)
        #print("Loaded data from file:", dirName + os.sep + "energy" + lengthType + "-" + fileName + ".dat")
    #etot = epot + np.mean(ekin) + np.mean(heat)
    #etot += heat
    #print(np.isnan(length).any(), np.isnan(etot).any(), np.isnan(epot).any(), np.isnan(temp).any())
    # Group length and energy data by length
    if bin_by == 'etot':
        length = length[np.argsort(etot)]
        etot = np.sort(etot)
        bin_size = (np.max(etot) - np.min(etot)) / 16
    elif bin_by == 'epot':
        length = length[np.argsort(epot)]
        etot = etot[np.argsort(epot)]
        bin_size = (np.max(epot) - np.min(epot)) / 16
    elif bin_by == 'ekin':
        length = length[np.argsort(ekin)]
        etot = etot[np.argsort(ekin)]
        bin_size = (np.max(ekin) - np.min(ekin)) / 16
    else: # Bin by length by default 
        etot = etot[np.argsort(length)]
        length = np.sort(length)
        bin_size = (np.max(length) - np.min(length)) / 16
    # Compute binned data and averages
    bin_length, bin_etot = bin_and_average(length, etot, epot, ekin, bin_size, bin_by)
    ax.plot(np.sort(length), etot[np.argsort(length)]/numParticles, color='k', marker='o', markersize=4, fillstyle='none', lw=0, label='Raw Data')
    ax.errorbar(bin_length[:,0], bin_etot[:,0]/numParticles, bin_etot[:,1]/numParticles, bin_length[:,1], color='g', marker='s', markersize=8, fillstyle='none', lw=1, capsize=3, label='Binned Data')
    # Fit curve to polynomial and get first-power coefficient
    x = bin_length[:,0]
    y = bin_etot[:,0]  # Subtract first value to normalize
    y = y[np.argsort(x)]
    x = np.sort(x)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    if degree == 1:
        popt, pcov = curve_fit(poly1, x, y)
        slope, intercept = popt
        slope_err = np.sqrt(np.diag(pcov))[0]
        y_fit = poly1(x_fit, slope, intercept)
        print(f"Slope = {slope} ± {slope_err}")
        temp = ekin / numParticles
        print(f"Temperature = {np.mean(temp)} ± {np.std(temp)}, range = [{np.min(temp)} - {np.max(temp)}]")
        ax.plot(x_fit, y_fit/numParticles, color='r', lw=1, label=f'Linear Fit, slope $\\times N$={slope:.2f}±{slope_err:.2f}')
        #saveLineTension(dirName, slope, slope_err, temp, dynamics=dynamics, dynType=dynType, mixed=mixed)
        # Create interpolation function for the fitted curve
        fit_interp = interp1d(x_fit, y_fit, kind='linear', fill_value='extrapolate')
        # Evaluate the fit at the raw length values
        etot_fit = fit_interp(length)
        # Compute residuals
        residuals = etot - etot_fit
        # Compute standard deviation of residuals (this is your noise)
        noise = np.std(residuals, ddof=1)
        signal = np.max(bin_etot[:,0]) - np.min(bin_etot[:,0])
        print("Noise in energy (std of residuals):", noise, "signal:", signal, "signal/noise:", signal/noise)
        # Create inverse interpolation function: from energy to expected length
        inv_fit_interp = interp1d(y_fit, x_fit, kind='linear', fill_value='extrapolate')
        # Evaluate the expected length at each raw energy point
        length_fit = inv_fit_interp(etot)
        # Compute residuals in length
        residuals_length = length - length_fit
        # Compute standard deviation of residuals in length (noise)
        noise_length = np.std(residuals_length, ddof=1)
        signal_length = np.max(bin_length[:,0]) - np.min(bin_length[:,0])
        print("Noise in length (std of residuals):", noise_length, "signal:", signal_length, "signal/noise:", signal_length/noise_length)
        #saveSignalOverFloor(signal, noise, signal_length, noise_length, temp, dirName, dynamics=dynamics, dynType=dynType)
    else:
        coeffs, cov = np.polyfit(x, y, degree, cov=True)
        print(coeffs)
        slope = coeffs[-2]  # Coefficient of x (first power)
        slope_err = np.sqrt(np.diag(cov))[-2]
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color='r', lw=1, label=f'Polynomial Fit (deg={degree})')
    ax.legend(loc='best', fontsize=12)
    fig.tight_layout()
    if show == 'show':
        plt.show()
    else:
        plt.pause(0.5)

def groupEnergyABLengthVSStrain(dirName, dynamics='dynamics', dynType='nve', fileName='nve', spacing='3', window=2, bin_by='length', mixed='mixed', degree=1, show=False):
    '''
    group energy and length data for all the values of strain and time
    '''
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Length,$ $L$", fontsize=14)
    ax.set_ylabel("$\\frac{\\Delta U_{AB}}{N}$", fontsize=20, rotation='horizontal', labelpad=15)
    if mixed == 'mixed':
        lengthType = "MixedLength"
    else:
        lengthType = "Length"
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    num1 = int(utils.readFromParams(dirName, 'num1'))
    if not(os.path.exists(dirName + os.sep + "energyAB" + lengthType + "-" + fileName + ".dat")):
        strainList, strain = utils.getOrderedStrainDirectories(dirName)
        epot = np.empty(0)
        ekin = np.empty(0)
        etot = np.empty(0)
        length = np.empty(0)
        for s in range(strainList.shape[0]):
            dirStrain = dirName + os.sep + strainList[s] + os.sep + dynamics + os.sep
            energy = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(-4,-3,-2))) / utils.readFromParams(dirStrain, "epsilon")
            # divide energy components by number of particles with AB interactions
            numAB = np.empty(0)
            dirList, _ = utils.getOrderedDirectories(dirStrain)
            count = 0
            for t in range(dirList.shape[0]):
                dirSample = dirStrain + dirList[t] + os.sep
                if(os.path.exists(dirSample)):
                    # Collect length data at current time
                    collect = False
                    if (os.path.exists(dirSample + "particlePos.dat")):
                        pos = np.loadtxt(dirSample + "particlePos.dat")
                        rad = np.loadtxt(dirSample + "../particleRad.dat")
                        if (pos.shape[0] == numParticles and rad.shape[0] == numParticles):
                            collect = True
                            if not(os.path.exists(dirSample + "interface" + lengthType + ".dat")):
                                get2InterfaceLength(dirSample, num1, spacing, window, mixed=mixed)
                            currentLength = np.loadtxt(dirSample + "interface" + lengthType + ".dat")[0]
                        else:
                            print("Warning: number of particles changed in", dirSample, pos.shape[0], rad.shape[0])
                        if np.isnan(currentLength):
                            print("Warning: NaN length found in", dirSample + "interface" + lengthType + ".dat")
                            collect = False
                    if collect:
                        count += 1
                        length = np.append(length, currentLength)
                        # Collect energy data at current time
                        epot = np.append(epot, energy[t,0])
                        ekin = np.append(ekin, energy[t,1])
                        etot = np.append(etot, (energy[t,0] + energy[t,1]))
                        numAB = np.append(numAB, energy[t,2])
            print(strainList[s], count)
            #etot[-count:] = etot[-count:] / np.mean(numAB)
            #epot[-count:] = epot[-count:] / np.mean(numAB)
            #ekin[-count:] = ekin[-count:] / np.mean(numAB)
        np.savetxt(dirName + os.sep + "energyAB" + lengthType + "-" + fileName + ".dat", np.column_stack((length, etot, epot, ekin)))
    else:
        length, etot, epot, ekin = np.loadtxt(dirName + os.sep + "energyAB" + lengthType + "-" + fileName + ".dat", unpack=True)
        #print("Loaded data from file:", dirName + os.sep + "energy" + lengthType + "-" + fileName + ".dat")
    #print(np.isnan(length).any(), np.isnan(etot).any(), np.isnan(epot).any(), np.isnan(ekin).any())
    etot = epot / numParticles # + np.mean(ekin)
    # Group length and energy data by length
    if bin_by == 'etot':
        length = length[np.argsort(etot)]
        etot = np.sort(etot)
        bin_size = (np.max(etot) - np.min(etot)) / 16
    elif bin_by == 'epot':
        length = length[np.argsort(epot)]
        etot = etot[np.argsort(epot)]
        bin_size = (np.max(epot) - np.min(epot)) / 16
    elif bin_by == 'temp':
        length = length[np.argsort(ekin)]
        etot = etot[np.argsort(ekin)]
        bin_size = (np.max(ekin) - np.min(ekin)) / 16
    else: # Bin by length by default 
        etot = etot[np.argsort(length)]
        length = np.sort(length)
        bin_size = (np.max(length) - np.min(length)) / 16
    # Compute binned data and averages
    bin_length, bin_etot = bin_and_average(length, etot, epot, ekin, bin_size, bin_by)
    ax.plot(np.sort(length), etot[np.argsort(length)], color='k', marker='o', markersize=4, fillstyle='none', lw=0, label='Raw Data')
    ax.errorbar(bin_length[:,0], bin_etot[:,0], bin_etot[:,1], bin_length[:,1], color='g', marker='s', markersize=8, fillstyle='none', lw=1, capsize=3, label='Binned Data')
    # Fit curve to polynomial and get first-power coefficient
    x = bin_length[:,0]
    y = bin_etot[:,0] # Subtract first value to normalize
    y = y[np.argsort(x)]
    x = np.sort(x)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    if degree == 1:
        popt, pcov = curve_fit(poly1, x, y)
        slope, intercept = popt
        slope_err = np.sqrt(np.diag(pcov))[0]
        y_fit = poly1(x_fit, slope, intercept)
        print(f"Slope = {slope} ± {slope_err}")
        temp = ekin / numParticles
        print(f"Temperature = {np.mean(temp)} ± {np.std(temp)}, range = [{np.min(temp)} - {np.max(temp)}]")
        saveLineTension(dirName, slope, slope_err, temp, dynamics=dynamics, dynType=dynType, mixed=mixed, ab='ab')
        # Create interpolation function for the fitted curve
        fit_interp = interp1d(x_fit, y_fit, kind='linear', fill_value='extrapolate')
        # Evaluate the fit at the raw length values
        etot_fit = fit_interp(length)
        # Compute residuals
        residuals = etot - etot_fit
        # Compute standard deviation of residuals (this is your noise)
        noise = np.std(residuals, ddof=1)
        signal = np.max(bin_etot[:,0]) - np.min(bin_etot[:,0])
        print("Noise in energy (std of residuals):", noise, "signal:", signal, "signal/noise:", signal/noise)
        # Create inverse interpolation function: from energy to expected length
        inv_fit_interp = interp1d(y_fit, x_fit, kind='linear', fill_value='extrapolate')
        # Evaluate the expected length at each raw energy point
        length_fit = inv_fit_interp(etot)
        # Compute residuals in length
        residuals_length = length - length_fit
        # Compute standard deviation of residuals in length (noise)
        noise_length = np.std(residuals_length, ddof=1)
        signal_length = np.max(bin_length[:,0]) - np.min(bin_length[:,0])
        print("Noise in length (std of residuals):", noise_length, "signal:", signal_length, "signal/noise:", signal_length/noise_length)
        #saveSignalOverFloor(signal, noise, signal_length, noise_length, ekin, dirName, dynamics=dynamics, dynType=dynType)
    else:
        coeffs, cov = np.polyfit(x, y, degree, cov=True)
        print(coeffs)
        slope = coeffs[-2]  # Coefficient of x (first power)
        slope_err = np.sqrt(np.diag(cov))[-2]
        y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, color='r', lw=1, label=f'Linear Fit, slope $\\times N$={slope*numParticles:.2f}±{slope_err*numParticles:.2f}')
    ax.legend(loc='best', fontsize=12)
    fig.tight_layout()
    if show == 'show':
        plt.show()
    else:
        plt.pause(0.5)

def plotLineTension(dirName, which, mixed = 'mixed'):
    if mixed == 'mixed':
        data = np.loadtxt(dirName + os.sep + "lineTension-" + which + "-mixed.dat")
    else:
        data = np.loadtxt(dirName + os.sep + "lineTension-" + which + ".dat")
    fig, ax = plt.subplots(figsize=(6,4.5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("$\\frac{\\gamma \\sigma}{\\varepsilon}$", fontsize=20, rotation='horizontal', labelpad=20)
    if which == 'nve':
        #ax.set_xlim(0.525, 1.025)
        ax.set_xlabel("$Temperature,$ $k_B T / \\varepsilon$", fontsize=14)
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        x = np.linspace(0,2,100)
    elif which == 'nve-box13':
        #ax.set_xlim(0.525, 1.025)
        ax.set_xlabel("$Temperature,$ $k_B T / \\varepsilon$", fontsize=14)
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color='c', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        data_nve = np.loadtxt(dirName + os.sep + "../../../box31/2lj/nh2/lineTension-nve-mixed.dat")
        ax.errorbar(data_nve[1:,0], data_nve[1:,2], data_nve[1:,3], data_nve[1:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        x = np.linspace(0,2,100)
        ax.legend(['$NVE$ $box13$', '$NVE,$ $box31$'], loc='best', fontsize=12)
    elif which == 'nve-s1w5' or which == 'nve-s1w2':
        ax.set_xlim(0.525, 1.025)
        ax.set_xlabel("$Temperature,$ $k_B T / \\varepsilon$", fontsize=14)
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color='c', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        data_nve = np.loadtxt(dirName + os.sep + "lineTension-nve-mixed.dat")
        ax.errorbar(data_nve[1:,0], data_nve[1:,2], data_nve[1:,3], data_nve[1:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        x = np.linspace(0,2,100)
    elif which == 'nvt-l2':
        data0 = np.loadtxt(dirName + os.sep + "lineTension-nve-mixed.dat")
        ax.errorbar(1e-06, data0[1,2], data0[1,3], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        data_l2 = np.loadtxt(dirName + os.sep + "lineTension-nvt-l2-long-mixed.dat")
        ax.errorbar(data_l2[:,-1], data_l2[:,2], data_l2[:,3], color='b', marker='s', markersize=8, fillstyle='none', lw=1, capsize=3)
        ax.errorbar(data[:,-1], data[:,2], data[:,3], color='g', marker='v', markersize=8, fillstyle='none', lw=1, capsize=3)
        ax.set_xlabel("$Damping,$ $\\beta \\sigma / \\sqrt{m \\varepsilon}$", fontsize=14)
        ax.set_xscale('log')
        x = np.linspace(1e-06,np.max(data[:,-1]),100)
        ax.legend(['$NVE$', '$NVT,$ $T \\simeq 0.6$ $[0,t_0]$', '$NVT,$ $T \\simeq 0.6$ $[t0,2t_0]$'], loc='best', fontsize=12)
    elif which[:6] == 'nvtvst':
        ax.set_xlim(0.525, 1.025)
        ax.set_xlabel("$Temperature,$ $k_B T / \\varepsilon$", fontsize=14)
        data_nve = np.loadtxt(dirName + os.sep + "lineTension-nve-mixed.dat")
        ax.errorbar(data_nve[1:,0], data_nve[1:,2], data_nve[1:,3], data_nve[1:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        data_nvt = np.loadtxt(dirName + os.sep + "lineTension-nvtvst-damping1e-05-mixed.dat")
        ax.errorbar(data_nvt[:,0], data_nvt[:,2], data_nvt[:,3], data_nvt[:,1], color='b', marker='s', markersize=8, fillstyle='none', lw=1, capsize=3)
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color='g', marker='v', markersize=8, fillstyle='none', lw=1, capsize=3)
        x = np.linspace(0,2,100)
        ax.legend(['$NVE$ $[0,t_0]$', '$NVT,$ $\\beta = 10^{-5}$ $[0,t_0]$', '$NVT,$ $\\beta = 10^{-5}$ $[t_0,2t_0]$'], loc='best', fontsize=12)
    else:
        #ax.set_xlim(0.525, 1.015)
        ax.set_xlabel("$Temperature,$ $k_B T/\\varepsilon$", fontsize=14)
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        data0 = np.loadtxt(dirName + os.sep + "lineTension-nvt-mixed.dat")
        ax.errorbar(data0[-1,0], data0[-1,2], data0[-1,3], data0[-1,1], color='b', marker='D', markersize=10, markeredgewidth=1.5, fillstyle='none', lw=1, capsize=3)
        x = np.linspace(np.min(data[:,0]),np.max(data[:,0]),100)
    ax.plot(x, np.zeros(100), ls='dotted', color='k', lw=0.8)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/mips/lineTension-" + which + ".png", dpi=120)
    plt.show()

def compareLineTension(dirName, figureName, mixed = 'mixed'):
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("$Line$ $tension,$ $\\gamma$", fontsize=14)
    #ax.set_xlim(0.425, 1.025)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
    dirList = np.array(["nve", "nvtvst-damping1e-07", "nvtvst-damping1e-03", "nvtvst"])
    labelList = ["NVE", "NVT $\\beta=10^{-4}$", "NVT $\\beta=10^{-2}$", "NVT $\\beta=10^0$"]
    colorList = ['k', 'b', 'g', 'c']
    markerList = ['o', 's', '^', 'D']
    for d in range(dirList.shape[0]):
        if mixed == 'mixed':
            data = np.loadtxt(dirName + os.sep + "lineTension-" + dirList[d] + "-mixed.dat")
        else:
            data = np.loadtxt(dirName + os.sep + "lineTension-" + dirList[d] + ".dat")
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color=colorList[d], marker=markerList[d], markersize=8, fillstyle='none', lw=1, capsize=3, label=labelList[d])
    x = np.linspace(np.min(data[:,0]),np.max(data[:,0]),100)
    ax.plot(x, np.zeros(100), ls='dotted', color='k', lw=0.8)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/mips/compareTension-" + figureName + ".png", dpi=120)
    plt.show()

def compareLineTensionVSDamping(dirName, figureName, mixed = 'mixed'):
    fig, ax = plt.subplots(figsize=(6,4.5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("$Line$ $tension,$ $\\gamma$", fontsize=14)
    #ax.set_xlim(0.425, 1.025)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
    dirList = np.array(["nvt", "later-nvt", "ab-nvt", "later2-nvt"])
    labelList = ["$t_{eq} = 10^3$", "$t_{eq} = 2 \\times 10^3$", "AB $t_{eq} = 2 \\times 10^3$", "$t_{eq} = 3 \\times 10^3$"]
    colorList = ['k', 'b', 'g', 'k']
    markerList = ['o', 's', '^', 'X']
    alphaList = [0.5, 1, 1, 1]
    for d in range(dirList.shape[0]):
        if mixed == 'mixed':
            data = np.loadtxt(dirName + os.sep + "lineTension-" + dirList[d] + "-mixed.dat")
        else:
            data = np.loadtxt(dirName + os.sep + "lineTension-" + dirList[d] + ".dat")
        print(dirList[d], data.shape)
        if d == dirList.shape[0]-1:
            ax.errorbar(data[-1], data[2], data[3]/np.sqrt(20), color=colorList[d], marker=markerList[d], markersize=8, 
                    fillstyle='none', lw=1.2, capsize=3, label=labelList[d], alpha=alphaList[d])
        else:
            ax.errorbar(data[:,-1], data[:,2], data[:,3], color=colorList[d], marker=markerList[d], markersize=8, 
                    fillstyle='none', lw=1.2, capsize=3, label=labelList[d], alpha=alphaList[d])
    data0 = np.loadtxt(dirName + os.sep + "lineTension-nve-mixed.dat")
    dt = utils.readFromParams(dirName + os.sep + 'T1.00', 'dt')
    ax.errorbar(dt, data0[1,2], data0[1,3], color='g', marker='*', markersize=20, markeredgewidth=1.5, fillstyle='none', lw=1, capsize=3)
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/mips/compareTension-" + figureName + ".png", dpi=120)
    plt.show()

def plotSignalOverFloor(dirName, which):
    data = np.loadtxt(dirName + os.sep + "signalFloor-" + which + ".dat")
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("$Signal$ $over$ $floor,$ $\\langle \\Delta W \\rangle_{i,i+1} / \\langle \\sigma_w \\rangle_i$", fontsize=14)
    if which == 'nve':
        ax.set_xlim(0.525, 1.015)
        ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
    elif which == 'nvt':
        ax.set_xlabel("$Damping,$ $\\beta$", fontsize=14)
        ax.set_xscale('log')
    else:
        ax.set_xlim(0.525, 1.015)
        ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
    ax.plot(data[:,0], data[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
    x = np.linspace(np.min(data[:,0]),np.max(data[:,0]),100)
    ax.plot(x, np.zeros(100), ls='dotted', color='k', lw=0.8)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/mips/signalFloor-" + which + ".png", dpi=120)
    plt.show()

def bin_and_average_pressure(length, pressure, bin_size=1.0):
    '''
    Bin the data by length and return the average values per bin
    '''
    bin_values = length
    other_values = pressure
    # Create bin edges
    min_val = np.min(bin_values)
    max_val = np.max(bin_values)
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    # Assign each value to a bin
    bin_indices = np.digitize(bin_values, bins) - 1  # make bins zero-indexed
    # Store average values per bin
    avg_bin_values = np.empty(0)
    err_bin_values = np.empty(0)
    avg_other_values = np.empty(0)
    err_other_values = np.empty(0)

    for i in range(len(bins) - 1):
        in_bin = bin_indices == i
        if np.any(in_bin):  # skip empty bins
            avg_bin_values = np.append(avg_bin_values, np.mean(bin_values[in_bin]))
            err_bin_values = np.append(err_bin_values, np.std(bin_values[in_bin]))
            avg_other_values = np.append(avg_other_values, np.mean(other_values[in_bin]))
            err_other_values = np.append(err_other_values, np.std(other_values[in_bin]))

    # Always return (length, etot)
    return np.column_stack((avg_bin_values, err_bin_values)), np.column_stack((avg_other_values, err_other_values))

def savePressure(dirName, pressure, temp, dynamics='dynamics', dynType='nve'):
        # Save line tension data
        if dynType == 'nve':
            file_path = os.path.join(dirName, f"../../../pressure-nve.dat")
            data = np.array([np.mean(temp), np.std(temp), np.mean(pressure), np.std(pressure)])
        elif dynType == 'nvt':
            file_path = os.path.join(dirName, f"../../../pressure-nvt.dat")
            damping = utils.readFromDynParams(dirName + os.sep + 'strain0.0100' + os.sep + dynamics, "damping")
            data = np.array([np.mean(temp), np.std(temp), np.mean(pressure), np.std(pressure), damping])
        else:
            file_path = os.path.join(dirName, f"../../../pressure-" + dynType + ".dat")
            data = np.array([np.mean(temp), np.std(temp), np.mean(pressure), np.std(pressure)])
        # Check if file exists
        file_exists = os.path.isfile(file_path)
        # Open in append mode and write
        with open(file_path, 'a') as f:
            if not file_exists:
                # Write header only once
                if dynType == 'nve':
                    f.write("# mean_temp  std_temp  press  std_press\n")
                elif dynType == 'nvt':
                    f.write("# damping  mean_temp  std_temp  press  std_press\n")
                else:
                    f.write("# mean_temp  std_temp  press  std_press\n")
            np.savetxt(f, data.reshape(1, -1), fmt="%.6e")

def groupPressureLengthVSStrain(dirName, dynamics='dynamics', dynType='nve', fileName='nve', mixed='mixed', spacing='3', window=2, show=False):
    '''
    group pressure and length data for all the values of strain and time
    '''
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Length,$ $L$", fontsize=14)
    ax.set_ylabel("$Pressure,$ $P$", fontsize=14)
    if mixed == 'mixed':
        lengthType = "MixedLength"
    else:
        lengthType = "Length"
    if not(os.path.exists(dirName + os.sep + "!pressure" + lengthType + "-" + fileName + ".dat")):
        num1 = int(utils.readFromParams(dirName, 'num1'))
        numParticles = int(utils.readFromParams(dirName, "numParticles"))
        epsilon = utils.readFromParams(dirName, "epsilon")
        strainList, strain = utils.getOrderedStrainDirectories(dirName)
        temp = np.empty(0)
        pressure = np.empty(0)
        length = np.empty(0)
        for s in range(strainList.shape[0]):
            print("strain:", strain[s])
            dirStrain = dirName + os.sep + strainList[s] + os.sep + dynamics + os.sep
            # Read energy and timestep data
            data = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(3,-1)))
            dirList, _ = utils.getOrderedDirectories(dirStrain)
            for t in range(dirList.shape[0]):
                dirSample = dirStrain + dirList[t] + os.sep
                if(os.path.exists(dirSample)):
                    # Collect length data at current time
                    collect = False
                    if (os.path.exists(dirSample + "particlePos.dat")):
                        pos = np.loadtxt(dirSample + "particlePos.dat")
                        rad = np.loadtxt(dirSample + "../particleRad.dat")
                        if (pos.shape[0] == numParticles and rad.shape[0] == numParticles):
                            collect = True
                            if not(os.path.exists(dirSample + "interface" + lengthType + ".dat")):
                                get2InterfaceLength(dirSample, num1, spacing, window, mixed=mixed)
                            currentLength = np.loadtxt(dirSample + "interface" + lengthType + ".dat")[0]
                        else:
                            print("Warning: number of particles changed in", dirSample, pos.shape[0], rad.shape[0])
                        if np.isnan(currentLength):
                            print("Warning: NaN length found in", dirSample + "interface" + lengthType + ".dat")
                            collect = False
                    if collect:
                        length = np.append(length, currentLength)
                        # Collect pressure data at current time
                        temp = np.append(temp, data[t,0] / numParticles)
                        pressure = np.append(pressure, data[t,1] / epsilon)
        np.savetxt(dirName + os.sep + "pressure" + lengthType + "-" + fileName + ".dat", np.column_stack((temp, pressure)))
    else:
        temp, pressure = np.loadtxt(dirName + os.sep + "pressure" + lengthType + "-" + fileName + ".dat", unpack=True)
    ax.plot(np.sort(length), pressure[np.argsort(length)], color='k', marker='o', markersize=4, fillstyle='none', lw=0, label='Raw Data')
    # Group length and energy data by length
    pressure = pressure[np.argsort(length)]
    length = np.sort(length)
    bin_size = (np.max(length) - np.min(length)) / 16
    # Save average pressure and plot binned pressure data
    savePressure(dirName, pressure, temp, dynamics=dynamics, dynType=dynType)
    print("tempeturare:", np.mean(temp), np.std(temp))
    print("pressure:", np.mean(pressure), np.std(pressure))
    bin_length, bin_pressure = bin_and_average_pressure(length, pressure, bin_size)
    ax.errorbar(bin_length[:,0], bin_pressure[:,0], bin_pressure[:,1], bin_length[:,1], color='g', marker='s', markersize=8, fillstyle='none', lw=1, capsize=3, label='Binned Data')
    ax.legend(loc='best', fontsize=12)
    fig.tight_layout()
    if show == 'show':
        plt.show()
    else:
        plt.pause(0.5)

def groupPressureVSStrain(dirName, dynamics='dynamics', dynType='nve', fileName='nve', show=False):
    '''
    group pressure data for all the values of strain and time
    '''
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
    ax.set_ylabel("$Pressure,$ $P$", fontsize=14)
    if not(os.path.exists(dirName + os.sep + "pressure-" + fileName + ".dat")):
        strainList, strain = utils.getOrderedStrainDirectories(dirName)
        temp = np.empty(0)
        pressure = np.empty(0)
        for s in range(strainList.shape[0]):
            dirStrain = dirName + os.sep + strainList[s] + os.sep + dynamics + os.sep
            # Read energy and timestep data
            data = np.array(np.loadtxt(dirStrain + "/energy.dat", usecols=(3,-1))) / 2#utils.readFromParams(dirStrain, "epsilon")
            # Collect pressure data at current time
            temp = np.append(temp, data[:,0])
            pressure = np.append(pressure, data[:,1])
        np.savetxt(dirName + os.sep + "pressure-" + fileName + ".dat", np.column_stack((temp, pressure)))
    else:
        temp, pressure = np.loadtxt(dirName + os.sep + "pressure-" + fileName + ".dat", unpack=True)
    ax.plot(temp, pressure, color='k', marker='o', markersize=4, fillstyle='none', lw=0, label='Raw Data')
    # Save average pressure and plot binned pressure data
    savePressure(dirName, pressure, temp, dynamics=dynamics, dynType=dynType)
    print("tempeturare:", np.mean(temp), np.std(temp))
    print("pressure:", np.mean(pressure), np.std(pressure))
    fig.tight_layout()
    if show == 'show':
        plt.show()
    else:
        plt.pause(0.5)

def plotPressure(dirName, which, mixed = 'mixed'):
    if mixed == 'mixed':
        data = np.loadtxt(dirName + os.sep + "pressure-" + which + "-mixed.dat")
    else:
        data = np.loadtxt(dirName + os.sep + "pressure-" + which + ".dat")
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("$Pressure,$ $p$", fontsize=14)
    if which == 'nve':
        ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
    elif which == 'nvt':
        ax.set_xlabel("$Damping,$ $\\beta$", fontsize=14)
        ax.errorbar(data[:,0], data[:,-2], data[:,-1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3)
        ax.set_xscale('log')
    else:
        ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1, capsize=3, label='$Active,$ $\\tau/\\Delta t = 1.5$')
        data0 = np.loadtxt(dirName + os.sep + "pressure-nvt-vsT.dat")
        ax.errorbar(data0[:,0], data0[:,2], data0[:,3], data0[:,1], color='b', marker='D', markersize=8, fillstyle='none', lw=1, capsize=3, label='$NVT$')
        ax.legend(loc='best', fontsize=12)
    x = np.linspace(np.min(data[:,0]),np.max(data[:,0]),100)
    ax.plot(x, np.zeros(100), ls='dotted', color='k', lw=0.8)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/mips/pressure-" + which + ".png", dpi=120)
    plt.show()

def comparePressure(dirName, figureName, mixed = 'mixed'):
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel("$Pressure,$ $p$", fontsize=14)
    #ax.set_xlim(0.425, 1.025)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=14)
    dirList = np.array(["nvtvst-damping1e-07", "nvtvst-damping1e-03", "nvtvst-damping1e01"])
    labelList = ["NVT $\\beta=10^{-4}$", "NVT $\\beta=10^{-2}$", "NVT $\\beta=10^0$"]
    colorList = ['b', 'g', 'c']
    markerList = ['s', '^', 'D']
    for d in range(dirList.shape[0]):
        if mixed == 'mixed':
            data = np.loadtxt(dirName + os.sep + "pressure-" + dirList[d] + "-mixed.dat")
        else:
            data = np.loadtxt(dirName + os.sep + "pressure-" + dirList[d] + ".dat")
        ax.errorbar(data[:,0], data[:,2], data[:,3], data[:,1], color=colorList[d], marker=markerList[d], markersize=8, fillstyle='none', lw=1, capsize=3, label=labelList[d])
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig("/home/francesco/Pictures/soft/mips/comparePressure-" + figureName + ".png", dpi=120)
    plt.show()

####################### Average cluster height interface #######################
def getInterfaceLength(dirName, threshold=0.62, spacing=2, window=3, plot=False, lj=True):
    spacingName = str(spacing)
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    if lj:
        rad *= 2**(1/6)
    sigma = 2*np.mean(rad)
    eps = 1.4 * sigma
    spacing *= sigma
    thickness = 3
    bins = np.arange(0, boxSize[1], spacing)
    edges = (bins[1:] + bins[:-1]) * 0.5
    rightInterface = np.zeros(bins.shape[0]-1)
    leftInterface = np.zeros(bins.shape[0]-1)
    # load particle variables
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    # load simplices
    labels, maxLabel = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold)
    pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
    clusterPos = pos[labels==maxLabel]
    #pos = utils.centerCOM(pos, rad, boxSize)
    leftPos = np.zeros((bins.shape[0]-1,2))
    rightPos = np.zeros((bins.shape[0]-1,2))
    for j in range(bins.shape[0]-1): # find particle positions in a vertical bin
        topMask = np.argwhere(clusterPos[:,1] > bins[j])[:,0]
        binPos = clusterPos[topMask]
        bottomMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
        binPos = binPos[bottomMask]
        if(binPos.shape[0] > 0):
            binDistance = binPos[:,0]
            #left interface
            leftMask = np.argsort(binDistance)[:thickness]
            leftInterface[j] = np.mean(binDistance[leftMask])
            leftPos[j,0] = leftInterface[j]
            leftPos[j,1] = np.mean(binPos[leftMask,1])
            # right interface
            rightMask = np.argsort(binDistance)[-thickness:]
            rightInterface[j] = np.mean(binDistance[rightMask])
            rightPos[j,0] = rightInterface[j]
            rightPos[j,1] = np.mean(binPos[rightMask,1])
    leftPos = np.column_stack((utils.computeMovingAverage(leftPos[:,0], window), utils.computeMovingAverage(leftPos[:,1], window)))
    rightPos = np.column_stack((utils.computeMovingAverage(rightPos[:,0], window), utils.computeMovingAverage(rightPos[:,1], window)))
    length = 0
    if(rightInterface[rightInterface!=0].shape[0] == rightInterface.shape[0]):
        prevPos = rightPos[0]
        for j in range(1,bins.shape[0]-1):
            length += np.linalg.norm(rightPos[j] - prevPos)
            prevPos = rightPos[j]
    if(leftInterface[leftInterface!=0].shape[0] == leftInterface.shape[0]):
        prevPos = leftPos[0]
        for j in range(1,bins.shape[0]-1):
            length += np.linalg.norm(leftPos[j] - prevPos)
            prevPos = leftPos[j]
    print("Interface length:", length)
    if(length != 0):
        np.savetxt(dirName + os.sep + "interface" + spacingName + ".dat", np.column_stack((leftPos, rightPos)))
    if(plot == "plot"):
        uplot.plotCorrelation(leftPos[:,0], leftPos[:,1], "$y$", xlabel = "$x$", color='b')
        uplot.plotCorrelation(rightPos[:,0], rightPos[:,1], "$y$", xlabel = "$x$", color='g')
        plt.pause(0.5)

####################### Average cluster height interface #######################
def getInterfaceLengthFromBorder(dirName, threshold=0.62, spacing=2, window=3, plot=False, lj=True):
    spacingName = str(spacing)
    print(spacingName)
    sep = utils.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    if lj:
        rad *= 2**(1/6)
    sigma = 2*np.mean(rad)
    eps = 1.4 * sigma
    spacing *= sigma
    thickness = 3
    bins = np.arange(0, boxSize[1], spacing)
    rightInterface = np.zeros(bins.shape[0]-1)
    leftInterface = np.zeros(bins.shape[0]-1)
    rightError = np.zeros(bins.shape[0]-1)
    leftError = np.zeros(bins.shape[0]-1)
    # load particle variables
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    # load simplices
    labels, maxLabel = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold)
    #pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
    #pos = utils.centerCOM(pos, rad, boxSize)
    particleList = np.loadtxt(dirName + os.sep + "particleList.dat")
    borderList = particleList[:,1]
    borderList[labels != maxLabel] = 0
    borderPos = pos[borderList==1]
    center = np.mean(borderPos, axis=0)[0]
    leftPos = np.zeros((bins.shape[0]-1,2))
    rightPos = np.zeros((bins.shape[0]-1,2))
    for j in range(bins.shape[0]-1): # find particle positions in a vertical bin
        topMask = np.argwhere(borderPos[:,1] > bins[j])[:,0]
        binPos = borderPos[topMask]
        bottomMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
        binPos = binPos[bottomMask]
        if(binPos.shape[0] > 0):
            #left interface
            leftBorder = binPos[binPos[:,0] < center,0]
            leftMask = np.argsort(leftBorder)[:thickness]
            leftInterface[j] = np.mean(leftBorder[leftMask])
            leftError[j] = np.std(leftBorder[leftMask])
            leftPos[j,0] = leftInterface[j]
            leftPos[j,1] = np.mean(binPos[leftMask,1])
            # right interface
            rightBorder = binPos[binPos[:,0] > center,0]
            rightMask = np.argsort(rightBorder)[-thickness:]
            rightInterface[j] = np.mean(rightBorder[rightMask])
            rightError[j] = np.std(rightBorder[rightMask])
            rightPos[j,0] = rightInterface[j]
            rightPos[j,1] = np.mean(binPos[rightMask,1])
    # correct for missing bins
    corrected = 0
    for i in range(leftPos.shape[0]):
        nextId = i+1
        if(i+1 > leftPos.shape[0]):
            nextId = 0
        prevId = i-1
        if(i-1 < 0):
            prevId = leftPos.shape[0]-1
        if(np.isnan(leftPos[i,0])):
            leftPos[i] = np.mean(np.array([leftPos[nextId], leftPos[prevId]]), axis=0)
            corrected += 1
        if(np.isnan(rightPos[i,0])):
            rightPos[i] = np.mean(np.array([rightPos[nextId], rightPos[prevId]]), axis=0)
            corrected += 1
    if(corrected != 0):
        print("Number of corrections:", corrected)
    leftPos = np.column_stack((utils.computeMovingAverage(leftPos[:,0], window), utils.computeMovingAverage(leftPos[:,1], window)))
    rightPos = np.column_stack((utils.computeMovingAverage(rightPos[:,0], window), utils.computeMovingAverage(rightPos[:,1], window)))
    length = 0
    if(rightInterface[rightInterface!=0].shape[0] == rightInterface.shape[0]):
        prevPos = rightPos[0]
        for j in range(1,bins.shape[0]-1):
            length += np.linalg.norm(rightPos[j] - prevPos)
            prevPos = rightPos[j]
    if(leftInterface[leftInterface!=0].shape[0] == leftInterface.shape[0]):
        prevPos = leftPos[0]
        for j in range(1,bins.shape[0]-1):
            length += np.linalg.norm(leftPos[j] - prevPos)
            prevPos = leftPos[j]
    print("Interface length:", length, "error:", np.sum([np.mean(leftError), np.mean(rightError)]))
    if(length != 0):
        np.savetxt(dirName + os.sep + "border" + spacingName + ".dat", np.column_stack((leftPos, rightPos, leftError, rightError)))
    if(plot == "plot"):
        uplot.plotCorrelation(leftPos[:,0], leftPos[:,1], "$y$", xlabel = "$x$", color='b')
        uplot.plotCorrelation(rightPos[:,0], rightPos[:,1], "$y$", xlabel = "$x$", color='g')
        plt.gcf().set_size_inches(7.7, 3)
        plt.xlim(-10, boxSize[0]+10)
        plt.tight_layout()
        plt.pause(0.5)
        #plt.show()

####################### Average cluster height interface #######################
def averageInterfaceVSTime(dirName, threshold=0.78, plot=False, dirSpacing=1):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    spacing = 3*sigma
    bins = np.arange(0, boxSize[1], spacing)
    #freq = rfftfreq(bins.shape[0]-1, spacing)*sigma
    interface = np.zeros((dirList.shape[0], bins.shape[0]-1))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        # load simplices
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
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
                posInterface[j,0] = height[j]
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
def averageInterfaceFluctuations(dirName, threshold=0.3, thickness=3, plot=False, dirSpacing=50000):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    print(dirList.shape[0])
    spacing = sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    #freq = rfftfreq(bins.shape[0]-1, spacing)
    freq = np.arange(1,centers.shape[0]+1,1)/(centers.shape[0] / spacing)
    leftDeltaHeight = np.zeros((dirList.shape[0], bins.shape[0]-1))
    leftFourierDeltaHeight = np.zeros((dirList.shape[0], freq.shape[0]))
    rightDeltaHeight = np.zeros((dirList.shape[0], bins.shape[0]-1))
    rightFourierDeltaHeight = np.zeros((dirList.shape[0], freq.shape[0]))
    for d in range(dirList.shape[0]):
        #print(d)
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        # load simplices
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        #pos = utils.centerCOM(pos, rad, boxSize)
        leftHeight = np.zeros(bins.shape[0]-1)
        rightHeight = np.zeros(bins.shape[0]-1)
        clusterPos = pos[labels==maxLabel]
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            downMask = np.argwhere(clusterPos[:,1] > bins[j])[:,0]
            binPos = clusterPos[downMask]
            upMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
            binPos = binPos[upMask]
            if(binPos.shape[0] > 0):
                center = np.mean(binPos, axis=0)[0] # center of dense cluster
                binDistance = binPos[:,0] - center
                leftMask = np.argsort(binDistance)[:thickness]
                leftHeight[j] = np.mean(binDistance[leftMask])
                rightMask = np.argsort(binDistance)[-thickness:]
                rightHeight[j] = np.mean(binDistance[rightMask])
        if(leftHeight[leftHeight!=0].shape[0] == leftHeight.shape[0]):
            #leftHeight -= np.mean(leftHeight)
            leftDeltaHeight[d] = leftHeight
            leftFourierDeltaHeight[d] = np.abs(rfft(leftHeight))**2
        if(rightHeight[rightHeight!=0].shape[0] == rightHeight.shape[0]):
            #rightHeight -= np.mean(rightHeight)
            rightDeltaHeight[d] = rightHeight
            rightFourierDeltaHeight[d] = np.abs(rfft(rightHeight))**2
    print(leftDeltaHeight.shape, leftDeltaHeight[leftDeltaHeight[:,0]!=0].shape)
    print(rightDeltaHeight.shape, rightDeltaHeight[rightDeltaHeight[:,0]!=0].shape)
    leftDeltaHeight = np.column_stack((np.mean(leftDeltaHeight, axis=0), np.std(leftDeltaHeight, axis=0)))
    rightDeltaHeight = np.column_stack((np.mean(rightDeltaHeight, axis=0), np.std(rightDeltaHeight, axis=0)))
    leftFourierDeltaHeight = np.column_stack((np.mean(leftFourierDeltaHeight, axis=0), np.std(leftFourierDeltaHeight, axis=0)))
    rightFourierDeltaHeight = np.column_stack((np.mean(rightFourierDeltaHeight, axis=0), np.std(rightFourierDeltaHeight, axis=0)))
    fourierDeltaHeight = np.zeros(leftFourierDeltaHeight.shape)
    fourierDeltaHeight[:,0] = np.mean(np.column_stack((leftFourierDeltaHeight[:,0],rightFourierDeltaHeight[:,0])), axis=1)
    fourierDeltaHeight[:,1] = np.sqrt(np.mean(np.column_stack((leftFourierDeltaHeight[:,1]**2,rightFourierDeltaHeight[:,1]**2)), axis=1))
    np.savetxt(dirName + os.sep + "heightFluctuations.dat", np.column_stack((centers, leftDeltaHeight, rightDeltaHeight, freq, fourierDeltaHeight)))
    if(plot=='plot'):
        uplot.plotCorrWithError(freq[2:], fourierDeltaHeight[2:,0], fourierDeltaHeight[2:,1], "$Height$ $fluctuation$", xlabel = "$Wave$ $vector$ $magnitude,$ $q$", color='k', logx=True, logy=True)
        #uplot.plotCorrWithError(freq[2:], leftFourierDeltaHeight[2:,0], leftFourierDeltaHeight[2:,1], "$Height$ $fluctuation$", xlabel = "$Wave$ $vector$ $magnitude,$ $q$", color='b', logx=True, logy=True)
        #uplot.plotCorrWithError(freq[2:], rightFourierDeltaHeight[2:,0], rightFourierDeltaHeight[2:,1], "$Height$ $fluctuation$", xlabel = "$Wave$ $vector$ $magnitude,$ $q$", color='g', logx=True, logy=True)
        #uplot.plotCorrWithError(centers, leftDeltaHeight[:,0], leftDeltaHeight[:,1], "$Height$ $fluctuation$", xlabel = "$y$", color='k')
        #uplot.plotCorrWithError(centers, rightDeltaHeight[:,0], rightDeltaHeight[:,1], "$Height$ $fluctuation$", xlabel = "$y$", color='g')
        plt.pause(0.5)
        #plt.show()

####################### Average cluster height interface #######################
def sampleInterfaceFluctuations(dirPath, threshold=0.3, numSamples=30, temp="0.30", plot=False, dirSpacing=1):
    dirName = dirPath + "0/langevin-lj/T" + temp + "/dynamics/"
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    spacing = 2*sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    freq = rfftfreq(bins.shape[0]-1, spacing)*sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    #freq = np.arange(1,centers.shape[0]+1,1)/(centers.shape[0] / spacing)
    deltaHeight = np.zeros((dirList.shape[0]*numSamples, bins.shape[0]-1))
    fourierDeltaHeight = np.zeros((dirList.shape[0]*numSamples, freq.shape[0]))
    for s in range(numSamples):
        dirName = dirPath + str(s) + "/langevin-lj/T" + temp + "/dynamics/"
        for d in range(dirList.shape[0]):
            dirSample = dirName + os.sep + dirList[d]
            #print(s, d, s*dirList.shape[0] + d, dirSample)
            # load particle variables
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            # load simplices
            labels,_ = cluster.getParticleDenseLabel(dirSample, threshold)
            pos = utils.centerSlab(pos, rad, boxSize, labels, 1)
            height = np.zeros(bins.shape[0]-1)
            posInterface = np.zeros((bins.shape[0]-1,2))
            clusterPos = pos[labels==1]
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
                height -= np.mean(height)
                deltaHeight[s*dirList.shape[0] + d] = height
    meanHeight = np.mean(np.mean(deltaHeight, axis=1))
    #deltaHeight -= meanHeight
    fourierDeltaHeight = rfft(deltaHeight)
    deltaHeight = np.column_stack((np.mean(deltaHeight, axis=0), np.std(deltaHeight, axis=0)))
    fourierDeltaHeight = np.column_stack((np.mean(fourierDeltaHeight, axis=0), np.std(fourierDeltaHeight, axis=0)))
    np.savetxt(dirPath + "0/../heightFlu-T" + temp + ".dat", np.column_stack((centers, deltaHeight, freq, fourierDeltaHeight)))
    if(plot=='plot'):
        #uplot.plotCorrWithError(centers, deltaHeight[:,0], deltaHeight[:,1], "$Height$ $fluctuation$", xlabel = "$y$", color='k')
        uplot.plotCorrelation(freq, fourierDeltaHeight[:,0], "$Height$ $fluctuation$", xlabel = "$q$", color='k', logx=True, logy=True)
        #plt.pause(0.5)
        plt.show()

####################### Average cluster height interface #######################
def averageInterfaceCorrelation(dirName, threshold=0.3, thickness=3, plot=False, dirSpacing=1):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    spacing = sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    maxCorrIndex = int((bins.shape[0]-1) / 2)
    distances = centers[:maxCorrIndex]
    leftHeightCorr = np.zeros((dirList.shape[0], maxCorrIndex))
    rightHeightCorr = np.zeros((dirList.shape[0], maxCorrIndex))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        # load simplices
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        leftHeight = np.zeros(bins.shape[0]-1)
        rightHeight = np.zeros(bins.shape[0]-1)
        clusterPos = pos[labels==maxLabel]
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            downMask = np.argwhere(clusterPos[:,1] > bins[j])[:,0]
            binPos = clusterPos[downMask]
            upMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
            binPos = binPos[upMask]
            if(binPos.shape[0] > 0):
                center = np.mean(binPos, axis=0)[0] # center of dense cluster
                binDistance = binPos[:,0] - center
                leftMask = np.argsort(binDistance)[:thickness]
                leftHeight[j] = np.mean(binDistance[leftMask])
                rightMask = np.argsort(binDistance)[-thickness:]
                rightHeight[j] = np.mean(binDistance[rightMask])
        leftHeightCorr[d] = utils.getHeightCorr(leftHeight, maxCorrIndex)
        rightHeightCorr[d] = utils.getHeightCorr(rightHeight, maxCorrIndex)
    leftHeightCorr = np.column_stack((np.mean(leftHeightCorr, axis=0), np.std(leftHeightCorr, axis=0)))
    rightHeightCorr = np.column_stack((np.mean(rightHeightCorr, axis=0), np.std(rightHeightCorr, axis=0)))
    heightCorr = np.zeros(leftHeightCorr.shape)
    heightCorr[:,0] = np.mean(np.column_stack((leftHeightCorr[:,0],rightHeightCorr[:,0])), axis=1)
    heightCorr[:,1] = np.sqrt(np.mean(np.column_stack((leftHeightCorr[:,1]**2,rightHeightCorr[:,1]**2)), axis=1))
    np.savetxt(dirName + os.sep + "heightCorr.dat", np.column_stack((distances, heightCorr)))
    if(plot=='plot'):
        uplot.plotCorrelation(distances[1:], heightCorr[1:,0], "$Height$ $correlation$", xlabel = "$Distance$", color='k')
        plt.pause(0.5)
        #plt.show()

############################## Work done on a fictitious wall #############################
def computeWallForceVSStrain(dirName, which='lj', multiple=3):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = 2 * np.mean(rad)
    if(which=='harmonic'):
        ec = 240
        cutDistance = multiple*sigma
    elif(which=='lj'):
        LJcutoff = 4
        cutDistance = multiple*LJcutoff*sigma
    else:
        print("Please specify sample type")
    dirList, strainList = utils.getOrderedStrainDirectories(dirName)
    timeList = np.array(['0', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000'])
    wallForce = np.zeros((strainList.shape[0],10))
    for d in range(dirList.shape[0]):
        print(dirList[d], strainList[d])
        for t in range(timeList.shape[0]):
            dirSample = dirName + os.sep + dirList[d] + "/t" + timeList[t]
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            #pos = utils.centerCOM(pos, rad, boxSize)
            #pos = np.loadtxt(dirSample + "/particlePos.dat")
            if(which=='harmonic'):
                contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
            elif(which=='lj'):
                contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
            # first compute work across fictitious wall at half height
            midHeight = boxSize[1]/2
            for i in range(numParticles):
                if(pos[i,1] < midHeight and pos[i,1] > midHeight - cutDistance):
                    #for c in range(i):
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(pos[c,1] > midHeight and pos[c,1] < midHeight + cutDistance):
                            radSum = rad[i] + rad[c]
                            delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                            distance = np.linalg.norm(delta)
                            gradMultiple = 0
                            if(which=='harmonic'):
                                overlap = 1 - distance / radSum
                                if(overlap > 0):
                                    gradMultiple = ec * overlap / radSum
                            elif(which=='lj'):
                                if(distance <= (LJcutoff * radSum)):
                                    forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - forceShift
                            if(gradMultiple != 0):
                                wallForce[d,t] += gradMultiple * delta[1] / distance
            print(t, wallForce[d,t])
        print(np.mean(wallForce[d]), np.std(wallForce[d]))
    wallForce *= sigma/ec
    wallForce = np.mean(wallForce, axis=1)
    np.savetxt(dirName + os.sep + "strainForce.dat", np.column_stack((strainList, wallForce)))
    uplot.plotCorrelation(strainList, wallForce, color='k', ylabel='$Force$', xlabel='$Strain,$ $\\gamma$')
    #plt.pause(0.5)
    plt.show()

############################## Work done on a fictitious wall #############################
def computeWallForceVSTime(dirName, threshold=0.78, which='lj', nDim=2, dirSpacing=1):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    eps = 1.4 * sigma
    if(which=='harmonic'):
        ec = 240
        cutDistance = 2.5*sigma
    elif(which=='lj'):
        LJcutoff = 5.5
        cutDistance = 2.5*LJcutoff*sigma
    else:
        print("Please specify sample type")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    wallForce = np.zeros((timeList.shape[0],2))
    boxSize = np.loadtxt(dirName + "/boxSize.dat")
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        if(which=='harmonic'):
            contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        elif(which=='lj'):
            contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        # first compute work across fictitious wall at half height
        midHeight = boxSize[1]/2
        for i in range(numParticles):
            if(pos[i,1] < midHeight and pos[i,1] > midHeight - cutDistance):
                for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                    if(pos[c,1] > midHeight and pos[c,1] < midHeight + cutDistance):
                        radSum = rad[i] + rad[c]
                        delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                        distance = np.linalg.norm(delta)
                        gradMultiple = 0
                        if(which=='harmonic'):
                            overlap = 1 - distance / radSum
                            if(overlap > 0):
                                gradMultiple = ec * overlap / radSum
                        elif(which=='lj'):
                            if(distance <= (LJcutoff * radSum)):
                                forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                                gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - forceShift
                        if(gradMultiple != 0):
                            if(labels[i]==maxLabel):
                                wallForce[d,0] += gradMultiple * delta[1] / distance
                            elif(labels[i]==-1):
                                wallForce[d,1] += gradMultiple * delta[1] / distance
        print(d, wallForce[d,0], wallForce[d,1])
    wallForce *= sigma/ec
    np.savetxt(dirName + os.sep + "timeForce.dat", np.column_stack((timeList, wallForce)))
    print("Average force on wall - dense:", np.mean(wallForce[:,0]), "+-", np.std(wallForce[:,0]))
    print("dilute:", np.mean(wallForce[:,0]), "+-", np.std(wallForce[:,0]))
    uplot.plotCorrelation(timeList, wallForce[:,0], color='k', ylabel='$Force$', xlabel='$Strain,$ $\\gamma$')
    uplot.plotCorrelation(timeList, wallForce[:,1], color='r', ylabel='$Force$', xlabel='$Strain,$ $\\gamma$')
    plt.pause(0.5)

############################# Work done on a fictitious wall #############################
def computeLJWallForce(dirName, LJcutoff=4, rangeForce=2, size="size", frac=1, dirSpacing=1):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    sigma = 2 * np.mean(rad)
    eps = 1.8 * np.max(rad)
    rangeForce *= LJcutoff * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    wallForce = np.zeros((dirList.shape[0], 3))
    midHeight = boxSize[1]*(1/2)
    typicalDistance = []
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        #labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps)
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        # first compute work across fictitious wall at half height
        for i in range(numParticles):
            if(np.argwhere(contacts[i]!=-1)[:,0].shape[0]>0):# and labels[i]==maxLabel):
                if(size=="size"):
                    posCheck = np.array([pos[i,0], pos[i,1] + frac * rad[i]])
                else:
                    posCheck = pos[i]
                # compute bonds across wall
                if(posCheck[1] < midHeight and pos[i,1] > (midHeight - rangeForce)):
                    wallPos = np.array([pos[i,0], midHeight])
                    wallDelta = utils.pbcDistance(pos[i], wallPos, boxSize)
                    wallDistance = np.linalg.norm(wallDelta)
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(size=="size"):
                            posCheck = np.array([pos[c,0], pos[c,1] - frac * rad[c]])
                        else:
                            posCheck = pos[c]
                        if(posCheck[1] > midHeight and pos[c,1] < (midHeight + rangeForce)):
                            radSum = rad[i] + rad[c]
                            delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                            distance = np.linalg.norm(delta)
                            if(distance <= (LJcutoff * radSum)):
                                typicalDistance.append(distance)
                                forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                                gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - forceShift
                                wallForce[d,0] += gradMultiple * delta[0] / distance
                                wallForce[d,1] += gradMultiple * delta[1] / distance
                                #wallForce[d,0] += gradMultiple * wallDelta[0] / distance
                                #wallForce[d,1] += gradMultiple * wallDelta[1] / distance
                                if(wallDistance > distance):
                                    print(i, c, wallDistance, distance, wallDistance / distance)
                # compute interaction with the wall
                radSum = rad[i]
                wallPos = np.array([posCheck[0], midHeight])
                wallDelta = utils.pbcDistance(wallPos, posCheck, boxSize)
                wallDistance = np.linalg.norm(wallDelta)
                if(wallDistance < (LJcutoff * radSum)):
                    #forceShift = utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                    #gradMultiple = utils.calcLJgradMultiple(ec, wallDistance, radSum) - forceShift
                    overlap = 1 - wallDistance / radSum
                    gradMultiple = ec * overlap / radSum
                    wallForce[d,2] += gradMultiple * wallDelta[1] / wallDistance
        #if(d % 10 == 0):
        #    print(d, wallForce[d], np.mean(typicalDistance))
    wallForce *= sigma/ec
    if(size=="size"):
        np.savetxt(dirName + os.sep + "wallForce-size" + str(frac) + ".dat", np.column_stack((timeList, wallForce)))
    else:
        np.savetxt(dirName + os.sep + "wallForce.dat", np.column_stack((timeList, wallForce)))
    print("Average force across wall - x:", np.mean(wallForce[:,0]), "+-", np.std(wallForce[:,0]))
    print("Average force across wall - y:", np.mean(wallForce[:,1]), "+-", np.std(wallForce[:,1]))
    print("Average force with wall:", np.mean(wallForce[:,2]), "+-", np.std(wallForce[:,2]))
    uplot.plotCorrelation(timeList, wallForce[:,0], color='r', ylabel='$Force$', xlabel='$Simulation$ $time,$ $t$')
    uplot.plotCorrelation(timeList, wallForce[:,1], color='k', ylabel='$Force$', xlabel='$Simulation$ $time,$ $t$')
    plt.pause(0.5)
    #plt.show()

############################# Exchange timescales ##############################
def computeExchangeTimes(dirName, threshold=0.3, lj='lj', plot=False, dirSpacing=1):
    numBins = 20
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    if(lj=='lj'):
        rad *= 2**(1/6)
    eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]*timeStep
    condenseTime = np.empty(0)
    evaporateTime = np.empty(0)
    time0 = np.zeros(numParticles)
    dirSample = dirName + os.sep + dirList[0]
    labels0, _ = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
    denseList0 = np.loadtxt(dirSample + "/particleList.dat")[:,0]
    for d in range(1,dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        denseList = np.loadtxt(dirSample + "/particleList.dat")[:,0]
        for i in range(numParticles):
            if(denseList[i]==1 and labels[i]==maxLabel):
                if(denseList0[i]==0):
                    if(time0[i]==0):
                        time0[i] = timeList[d]
                    else:
                        condenseTime = np.append(condenseTime, timeList[d] - time0[i])
                        time0[i] = timeList[d]
                    denseList0[i] = 1
                    labels0[i] = maxLabel
            if(denseList[i]==0):
                if(denseList0[i]==1 and labels0[i]==maxLabel):
                    if(time0[i]==0):
                        time0[i] = timeList[d]
                    else:
                        evaporateTime = np.append(evaporateTime, timeList[d] - time0[i])
                        time0[i] = timeList[d]
                    denseList0[i] = 0
                    labels0[i] = -1
    np.savetxt(dirName + os.sep + "condensationTime.dat", condenseTime)
    np.savetxt(dirName + os.sep + "evaporationTime.dat", evaporateTime)
    print("Black, condensation time:", np.mean(condenseTime), "+-", np.std(condenseTime))
    print("Red, evaporation time:", np.mean(evaporateTime), "+-", np.std(evaporateTime))
    if(plot=='plot'):
        uplot.plotCorrelation(np.arange(1, condenseTime.shape[0]+1, 1), np.sort(condenseTime), "$Condensation$ $time$", xlabel = "$index$", color='k', logy=True)
        uplot.plotCorrelation(np.arange(1, evaporateTime.shape[0]+1, 1), np.sort(evaporateTime), "$Evaporation$ $time$", xlabel = "$index$", color='r', logy=True)
        plt.pause(0.5)
        #plt.show()

############################# Exchange rates ##############################
def computeExchangeRates(dirName, threshold=0.3, lj='lj', plot=False, dirSpacing=1):
    numBins = 20
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    if(lj=='lj'):
        rad *= 2**(1/6)
    eps = 1.4 * sigma
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]*timeStep
    timeSpacing = timeList[1] - timeList[0]
    evaporateNum = np.zeros(dirList.shape[0]-1)
    dirSample = dirName + os.sep + dirList[0]
    denseList0,_ = cluster.getParticleDenseLabel(dirSample, threshold, compute='True')
    #labels0, maxLabels0 = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
    #denseList0 = np.loadtxt(dirSample + "/particleList.dat")[:,0]
    for d in range(1,dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        denseList,_ = cluster.getParticleDenseLabel(dirSample, threshold)
        #labels, maxLabel = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        #denseList = np.loadtxt(dirSample + "/particleList.dat")[:,0]
        evaporateNum[d-1] = (denseList0[(np.not_equal(denseList,denseList0))]==1).shape[0]
        #print(timeList[d], evaporateNum[d-1])
        denseList0 = denseList
    evaporateRate = evaporateNum / timeSpacing
    np.savetxt(dirName + os.sep + "evaporationRate.dat", evaporateRate)
    print("Evaporation rate:", np.mean(evaporateRate), "+-", np.std(evaporateRate))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList[1:], evaporateRate, "$Evaporation$ $rate$", xlabel = "$Time,$ $t$", color='k')
        plt.pause(0.5)
        #plt.show()

############################# Mixing for 2 species ##############################
def computeMixingDistribution(dirName, num1=0, numBins=20, which='time', plot=False, dirSpacing=1):
    if(plot=='plot'):
        fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel("$Neighbor$ $fraction,$ $f$", fontsize=16)
        ax.set_ylabel("$P(f_{AA}),$ $P(f_{AB}),$ $P(f_{BB})$", fontsize=16)
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    if(which=='time'):
        dirList, timeList = utils.getOrderedDirectories(dirName)
        dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    elif(which=='strain'):
        dirSpacing = 0.001
        dirList, strainList = utils.getOrderedStrainDirectories(dirName)
        dirList = dirList[np.argwhere(strainList%dirSpacing==0)[:,0]]
    neighborFracAA = []
    neighborFracAB = []
    neighborFracBB = []
    for d in range(dirList.shape[0]):
        print(dirList[d])
        fracAA = []
        fracAB = []
        fracBB = []
        meanNum = 0
        dirSample = dirName + os.sep + dirList[d]
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        # count neighbors of different types for each particle
        for i in range(numParticles):
            numNeighbors = np.argwhere(neighbors[i]!=-1)[:,0].shape[0]
            meanNum += numNeighbors
            fAA = 0
            fAB = 0
            fBB = 0
            for c in range(numNeighbors):
                if(i < num1):
                    if(neighbors[i,c] >= num1):
                        fAB += 1
                    else:
                        fAA += 1
                else:
                    if(neighbors[i,c] >= num1):
                        fBB += 1
                    else:
                        fAB += 1
            if(fAA != 0):
                fracAA.append(fAA / numNeighbors)
                neighborFracAA.append(fAA / numNeighbors)
            if(fAB != 0):
                fracAB.append(fAB / numNeighbors)
                neighborFracAB.append(fAB / numNeighbors)
            if(fBB != 0):
                fracBB.append(fBB / numNeighbors)
                neighborFracBB.append(fBB / numNeighbors)
        if(plot=='plot'):
            binMin = np.min([np.min(fracAA), np.min(fracAB), np.min(fracBB)])
            binMax = np.max([np.max(fracAA), np.max(fracAB), np.max(fracBB)])
            bins = np.linspace(binMin, binMax, numBins)
            pdfAA, edges = np.histogram(fracAA, bins, density=True)
            pdfAB, edges = np.histogram(fracAB, bins, density=True)
            pdfBB, edges = np.histogram(fracBB, bins, density=True)
            edges = (edges[1:] + edges[:-1]) * 0.5
            ax.plot(edges, pdfAA, color='g')
            ax.plot(edges, pdfAB, color='r')
            ax.plot(edges, pdfBB, color='b')
            plt.tight_layout()
            plt.pause(0.5)
        print("average number of neighbors:", meanNum/numParticles)
    binMin = np.min([np.min(neighborFracAA), np.min(neighborFracAB), np.min(neighborFracBB)])
    binMax = np.max([np.max(neighborFracAA), np.max(neighborFracAB), np.max(neighborFracBB)])
    bins = np.linspace(binMin, binMax, numBins)
    pdfAA, edges = np.histogram(neighborFracAA, bins, density=True)
    pdfAB, edges = np.histogram(neighborFracAB, bins, density=True)
    pdfBB, edges = np.histogram(neighborFracBB, bins, density=True)
    edges = (edges[1:] + edges[:-1]) * 0.5
    np.savetxt(dirName + os.sep + "neighborFraction.dat", np.column_stack((edges, pdfAA, pdfAB, pdfBB)))
    if(plot=='plot'):
        ax.plot(edges, pdfAA, color='g', linestyle='--', lw=1.5)
        ax.plot(edges, pdfAB, color='r', linestyle='--', lw=1.5)
        ax.plot(edges, pdfBB, color='b', linestyle='--', lw=1.5)
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    if(whichCorr == "temp"):
        threshold = float(sys.argv[3])
        computeClusterTemperatureVSTime(dirName, threshold=threshold)

############################# Harmonic cluster pressure #########################
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

    elif(whichCorr == "pressureprofile"):
        shiftx = float(sys.argv[3])
        averageLinearPressureProfile(dirName, shiftx=shiftx)

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

    elif(whichCorr == "kbtension"):
        averageKBLJTension(dirName, LJcutoff=float(sys.argv[3]), plot=sys.argv[4])

    elif(whichCorr == "hprofile"):
        averageHLJPressureProfile(dirName, LJcutoff=float(sys.argv[3]), plot=sys.argv[4])

    elif(whichCorr == "samplehprofile"):
        sampleHLJPressureProfile(dirName, numSamples=int(sys.argv[3]), LJcutoff=float(sys.argv[4]), plot=sys.argv[5])

    elif(whichCorr == "ikprofile"):
        averageIKLJPressureProfile(dirName, LJcutoff=float(sys.argv[3]), nDim=int(sys.argv[4]), plot=sys.argv[5])

    elif(whichCorr == "sampleikprofile"):
        sampleIKLJPressureProfile(dirName, numSamples=int(sys.argv[3]), LJcutoff=float(sys.argv[4]), plot=sys.argv[5], temp=sys.argv[6])

    elif(whichCorr == "cgikprofile"):
        averageCGIKLJPressureProfile(dirName, LJcutoff=float(sys.argv[3]), plot=sys.argv[4])

    elif(whichCorr == "samplecgikprofile"):
        sampleCGIKLJPressureProfile(dirName, numSamples=int(sys.argv[3]), LJcutoff=float(sys.argv[4]), plot=sys.argv[5], temp=sys.argv[6])

    elif(whichCorr == "2pressure"):
        num1 = int(sys.argv[3])
        active = sys.argv[4]
        if(active == "active"):
            active = True
        dirSpacing = int(float(sys.argv[5]))
        compute2LJPressureVSTime(dirName, num1, active, dirSpacing)
    
############################## Interface structure ##############################
    elif(whichCorr == "pprofile"):
        size = float(sys.argv[3])
        num1 = int(sys.argv[4])
        plot = sys.argv[5]
        computeLinearPressureProfile(dirName, size, num1, plot)

    elif(whichCorr == "phiprofile"):
        threshold = float(sys.argv[3])
        size = float(sys.argv[4])
        plot = sys.argv[5]
        computeLinearDensityProfile(dirName, threshold, size, plot=plot)

    elif(whichCorr == "profile"):
        threshold = float(sys.argv[3])
        size = float(sys.argv[4])
        plot = sys.argv[5]
        averageLinearDensityProfile(dirName, threshold, size, plot=plot)

    elif(whichCorr == "2phiprofile"):
        num1 = int(sys.argv[3])
        compute2DensityProfile(dirName, num1)

    elif(whichCorr == "2profile"):
        num1 = int(sys.argv[3])
        plot = sys.argv[4]
        average2DensityProfile(dirName, num1, plot=plot)

    elif(whichCorr == "2width"):
        num1 = int(sys.argv[3])
        which = sys.argv[4]
        strain = sys.argv[5]
        damping = sys.argv[6]
        pause = sys.argv[7]
        plotWidth(dirName, num1, which, strain, damping, pause)

    elif(whichCorr == "2widthstrain"):
        which = sys.argv[3]
        pause = sys.argv[4]
        plotAverageWidth(dirName, which, pause)
    
    elif(whichCorr == "2widthcompare"):
        figureName = sys.argv[3]
        compareAverageWidth(dirName, figureName)

    elif(whichCorr == "2widthlength"):
        num1 = int(sys.argv[3])
        dynamics = sys.argv[4]
        pause = sys.argv[5]
        plotWidthVSLength(dirName, num1, dynamics, pause)

    elif(whichCorr == "2intercorr"):
        num1 = int(sys.argv[3])
        thickness = float(sys.argv[4])
        plot = sys.argv[5]
        average2InterfaceCorrelation(dirName, num1, thickness, plot)

    elif(whichCorr == "2energy"):
        num1 = int(sys.argv[3])
        which = sys.argv[4]
        plot = sys.argv[5]
        compute2InterfaceEnergy(dirName, num1, which, plot)

    elif(whichCorr == "2width"):
        num1 = int(sys.argv[3])
        compute2PhaseWidthVSStrain(dirName, num1)

    elif(whichCorr == "2corr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = float(sys.argv[5])
        num1 = int(sys.argv[6])
        plot = sys.argv[7]
        average2FluidsCorr(dirName, startBlock, maxPower, freqPower, num1, plot)

    elif(whichCorr == "2velpdf"):
        which = sys.argv[3]
        num1 = int(sys.argv[4])
        plot = sys.argv[5]
        figureName = sys.argv[6]
        average2FluidsVelPDF(dirName, which, num1, plot, figureName)

    elif(whichCorr == "2velcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = float(sys.argv[5])
        num1 = int(sys.argv[6])
        plot = sys.argv[7]
        average2FluidsVelCorr(dirName, startBlock, maxPower, freqPower, num1, plot)

    elif(whichCorr == "2spacevelcorr"):
        num1 = int(sys.argv[3])
        plot = sys.argv[4]
        average2FluidsSpaceVelCorr(dirName, num1, plot)

    elif(whichCorr == "mobility"):
        delta_t = int(sys.argv[3])
        plot = sys.argv[4]
        computeMobilityCorrelation(dirName, delta_t, plot)

    elif(whichCorr == "2interface"):
        num1 = int(sys.argv[3])
        thickness = int(sys.argv[4])
        plot = sys.argv[5]
        average2InterfaceFluctuations(dirName, num1, thickness, plot)

    elif(whichCorr == "wallstrain"):
        which = sys.argv[3]
        multiple = float(sys.argv[4])
        computeWallForceVSStrain(dirName, which, multiple)

    elif(whichCorr == "walltime"):
        threshold = float(sys.argv[3])
        which = sys.argv[4]
        computeWallForceVSTime(dirName, threshold, which)

    elif(whichCorr == "wallforce"):
        LJcutoff = float(sys.argv[3])
        rangeForce = float(sys.argv[4])
        size = sys.argv[5]
        frac = float(sys.argv[6])
        computeLJWallForce(dirName, LJcutoff, rangeForce, size, frac)

    elif(whichCorr == "2interlength"):
        num1 = int(sys.argv[3])
        spacing = sys.argv[4]
        window = int(sys.argv[5])
        plot = sys.argv[6]
        get2InterfaceLength(dirName, num1, spacing, window, plot=plot)

    elif(whichCorr == "lengthtime"):
        dynamics = sys.argv[3]
        spacing = float(sys.argv[4])
        window = int(sys.argv[5])
        mixed = sys.argv[6]
        show = sys.argv[7]
        plotLengthVSTime(dirName, dynamics, spacing, window, mixed, show)

    elif(whichCorr == "groupenergy"):
        dynamics = sys.argv[3]
        dynType = sys.argv[4]
        fileName = sys.argv[5]
        spacing = float(sys.argv[6])
        window = int(sys.argv[7])
        bin_by = sys.argv[8]
        mixed = sys.argv[9]
        degree = int(sys.argv[10])
        show = sys.argv[11]
        groupEnergyLengthVSStrain(dirName, dynamics, dynType, fileName, spacing, window, bin_by, mixed, degree, show)

    elif(whichCorr == "groupenergyab"):
        dynamics = sys.argv[3]
        dynType = sys.argv[4]
        fileName = sys.argv[5]
        spacing = float(sys.argv[6])
        window = int(sys.argv[7])
        bin_by = sys.argv[8]
        mixed = sys.argv[9]
        degree = int(sys.argv[10])
        show = sys.argv[11]
        groupEnergyABLengthVSStrain(dirName, dynamics, dynType, fileName, spacing, window, bin_by, mixed, degree, show)

    elif(whichCorr == "plotgamma"):
        which = sys.argv[3]
        mixed = sys.argv[4]
        plotLineTension(dirName, which, mixed)

    elif(whichCorr == "comparegamma"):
        figureName = sys.argv[3]
        mixed = sys.argv[4]
        compareLineTension(dirName, figureName, mixed)

    elif(whichCorr == "gammabeta"):
        figureName = sys.argv[3]
        mixed = sys.argv[4]
        compareLineTensionVSDamping(dirName, figureName, mixed)

    elif(whichCorr == "grouppress"):
        dynamics = sys.argv[3]
        dynType = sys.argv[4]
        fileName = sys.argv[5]
        mixed = sys.argv[6]
        spacing = float(sys.argv[7])
        window = int(sys.argv[8])
        show = sys.argv[9]
        groupPressureLengthVSStrain(dirName, dynamics, dynType, fileName, mixed, spacing, window, show)

    elif(whichCorr == "groupp"):
        dynamics = sys.argv[3]
        dynType = sys.argv[4]
        fileName = sys.argv[5]
        show = sys.argv[6]
        groupPressureVSStrain(dirName, dynamics, dynType, fileName, show)

    elif(whichCorr == "plotpress"):
        which = sys.argv[3]
        plotPressure(dirName, which)

    elif(whichCorr == "comparepress"):
        figureName = sys.argv[3]
        mixed = sys.argv[4]
        comparePressure(dirName, figureName, mixed)

    elif(whichCorr == "plotfloor"):
        which = sys.argv[3]
        plotSignalOverFloor(dirName, which)

    elif(whichCorr == "energyheight"):
        dynamics = sys.argv[3]
        dynType = sys.argv[4]
        which = sys.argv[5]
        figureName = sys.argv[6]
        show = sys.argv[7]
        plotEnergyVSHeight(dirName, dynamics, dynType, which, figureName, show)

    elif(whichCorr == "interlength"):
        threshold = float(sys.argv[3])
        spacing = float(sys.argv[4])
        window = int(sys.argv[5])
        plot = sys.argv[6]
        getInterfaceLength(dirName, threshold, spacing, window, plot)

    elif(whichCorr == "borderlength"):
        threshold = float(sys.argv[3])
        spacing = float(sys.argv[4])
        window = int(sys.argv[5])
        plot = sys.argv[6]
        getInterfaceLengthFromBorder(dirName, threshold, spacing, window, plot)

    elif(whichCorr == "interfacetime"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageInterfaceVSTime(dirName, threshold, plot)

    elif(whichCorr == "interface"):
        threshold = float(sys.argv[3])
        thickness = int(sys.argv[4])
        plot = sys.argv[5]
        averageInterfaceFluctuations(dirName, threshold, thickness, plot)

    elif(whichCorr == "sampleinterface"):
        threshold = float(sys.argv[3])
        numSamples = int(sys.argv[4])
        temp = sys.argv[5]
        plot = sys.argv[6]
        sampleInterfaceFluctuations(dirName, threshold, numSamples, temp, plot)

    elif(whichCorr == "interfacecorr"):
        threshold = float(sys.argv[3])
        thickness = int(sys.argv[4])
        plot = sys.argv[5]
        averageInterfaceCorrelation(dirName, threshold, thickness, plot)

    elif(whichCorr == "exchange"):
        threshold = float(sys.argv[3])
        lj = sys.argv[4]
        plot = sys.argv[5]
        computeExchangeTimes(dirName, threshold, lj, plot)

    elif(whichCorr == "rate"):
        threshold = float(sys.argv[3])
        lj = sys.argv[4]
        plot = sys.argv[5]
        computeExchangeRates(dirName, threshold, lj, plot)

    elif(whichCorr == "2mixing"):
        num1 = int(sys.argv[3])
        numBins = int(sys.argv[4])
        which = sys.argv[5]
        plot = sys.argv[6]
        computeMixingDistribution(dirName, num1, numBins, which, plot)

    else:
        print("Please specify the correlation you want to compute")
