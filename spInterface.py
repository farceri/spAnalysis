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
import spCluster as cluster

def computeClusterTemperatureVSTime(dirName, threshold=0.76, dirSpacing=1):
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
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps)
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
    eps = 1.8*np.max(rad)
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
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
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
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold=0.3)
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
def averageLinearLJPressureProfile(dirName, LJcutoff=5.5, dirSpacing=1, nDim=2, plot=False):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    #eps = 1.8*np.max(rad)
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
    tangential = np.zeros((dirList.shape[0],bins.shape[0]-1))
    normal = np.zeros((dirList.shape[0], bins.shape[0]-1))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        #labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        #maxLabel = utils.findLargestParticleCluster(rad, labels)
        #pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        pos = utils.centerCOM(pos, rad, boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        for bin in range(bins.shape[0]-1):
            for i in range(numParticles):
                if(pos[i,0] < centers[bin]):
                    for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                        if(pos[c,0] > pos[i,0] and pos[c,0] > centers[bin]):
                            radSum = rad[i] + rad[c]
                            delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                            distance = np.linalg.norm(delta)
                            add = False
                            if(distance <= (LJcutoff * radSum)):
                                gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                                force = 0.5 * gradMultiple * delta / distance
                                add = True
                            if(add == True):
                                tangential[d,bin] += force[1] * delta[1]**2 / (distance * np.abs(delta[0]))
                                normal[d,bin] += force[0] * delta[0] / distance
                    # find bin where this force belongs to following Irving-Kirkwood
    tangential = np.mean(tangential,axis=0) / binArea
    normal = np.mean(normal,axis=0) / binArea
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((centers, normal, tangential)))
    print("surface tension: ", np.sum((normal - tangential)*binWidth))
    if(plot=="plot"):
        uplot.plotCorrelation(centers, normal, "$\\sigma_{xx}$", xlabel = "$Distance$", color='g', lw=1)
        uplot.plotCorrelation(centers, tangential, "$\\sigma_{yy}$", xlabel = "$Distance$", color='b', lw=1)
        #plt.pause(0.5)
        plt.show()

##################### Average IK LJ linear pressure profile #######################
def averageIKLJPressureProfile(dirName, LJcutoff=5.5, plot=False, dirSpacing=1, nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    sigma = np.mean(rad)
    #eps = 1.8*np.max(rad)
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
    virial = np.zeros((dirList.shape[0],bins.shape[0]-1,2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        #labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        #maxLabel = utils.findLargestParticleCluster(rad, labels)
        #pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        pos = utils.centerCOM(pos, rad, boxSize)
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
    if(plot=="plot"):
        uplot.plotCorrelation(centers, virial[:,0], "$Pressure$ $profile$", xlabel = "$Distance$", color='b', lw=1.5)
        uplot.plotCorrelation(centers, virial[:,1], "$Pressure$ $profile$", xlabel = "$Distance$", color='g', lw=1.5)
        #plt.pause(0.5)
        plt.show()

##################### Sample average LJ linear pressure profile #######################
def sampleLinearLJPressureProfile(dirPath, numSamples=30, LJcutoff=5.5, dirSpacing=1, nDim=2, plot=False):
    ec = 1
    dirName = dirPath + "0/langevin-lj/T0.30/dynamics/"
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    #eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], LJcutoff*2*np.mean(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    # pressure bins
    tangential = np.zeros((dirList.shape[0]*numSamples,bins.shape[0]-1))
    normal = np.zeros((dirList.shape[0]*numSamples, bins.shape[0]-1))
    for s in range(numSamples):
        for d in range(dirList.shape[0]):
            dirSample = dirPath + str(s) + "/langevin-lj/T0.30/dynamics/" + dirList[d]
            #print(s, d, s*dirList.shape[0] + d, dirSample)
            pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
            #labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
            #maxLabel = utils.findLargestParticleCluster(rad, labels)
            #pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
            pos = utils.centerCOM(pos, rad, boxSize)
            vel = np.loadtxt(dirSample + "/particleVel.dat")
            contacts = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
            for bin in range(bins.shape[0]-1):
                for i in range(numParticles):
                    if(pos[i,0] < centers[bin]):
                        for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                            if(pos[c,0] > pos[i,0] and pos[c,0] > centers[bin]):
                                radSum = rad[i] + rad[c]
                                delta = utils.pbcDistance(pos[i], pos[c], boxSize)
                                distance = np.linalg.norm(delta)
                                add = False
                                if(distance <= (LJcutoff * radSum)):
                                    gradMultiple = utils.calcLJgradMultiple(ec, distance, radSum) - utils.calcLJgradMultiple(ec, LJcutoff * radSum, radSum)
                                    force = 0.5 * gradMultiple * delta / distance
                                    add = True
                                if(add == True):
                                    tangential[s*dirList.shape[0] + d,bin] += force[1] * delta[1]**2 / (distance * np.abs(delta[0]))
                                    normal[s*dirList.shape[0] + d,bin] += force[0] * delta[0] / distance
                        # find bin where this force belongs to following Irving-Kirkwood
    tangential = np.mean(tangential,axis=0) / binArea
    normal = np.mean(normal,axis=0) / binArea
    np.savetxt(dirName + os.sep + "pressureProfile.dat", np.column_stack((centers, normal, tangential)))
    print("surface tension: ", np.sum((normal - tangential)*binWidth))
    if(plot=="plot"):
        uplot.plotCorrelation(centers, normal, "$\\sigma_{xx}$", xlabel = "$Distance$", color='g', lw=1)
        uplot.plotCorrelation(centers, tangential, "$\\sigma_{yy}$", xlabel = "$Distance$", color='b', lw=1)
        #plt.pause(0.5)
        plt.show()

####################### Average linear density profile ########################
def computeLinearDensityProfile(dirName, threshold=0.3, lj='lj', plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    if(lj=='lj'):
        rad *= 2**(1/6)
    eps = 1.8*np.max(rad)
    # distance bins
    bins = np.arange(0, boxSize[0], np.max(rad))
    binWidth = bins[1] - bins[0]
    binArea = binWidth*boxSize[1]
    centers = (bins[1:] + bins[:-1])/2
    particleDensity = np.zeros(centers.shape[0])
    # first compute particle density
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    pos = utils.centerCOM(pos, rad, boxSize)
    #labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold, compute=True)
    #maxLabel = utils.findLargestParticleCluster(rad, labels)
    #print(maxLabel, labels[labels==maxLabel].shape[0])
    #pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
    for i in range(numParticles):
        for j in range(bins.shape[0]-1):
            if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                particleDensity[j] += np.pi*rad[i]**2
    particleDensity /= binArea
    # compute interface width
    center = np.mean(centers)
    x = centers-center
    y = particleDensity
    np.savetxt(dirName + os.sep + "singleProfile.dat", np.column_stack((x, particleDensity)))
    interfaceWidth = (utils.computeInterfaceWidth(x[x>0], y[x>0]) + utils.computeInterfaceWidth(-x[x<0], y[x<0]))/2
    print("Interface width:", interfaceWidth)
    # compute fluid width
    xFluid = x[np.argwhere(y>0.5)[:,0]]
    width = xFluid[-1] - xFluid[0]
    print("Fluid width:", width)
    if(plot=='plot'):
        uplot.plotCorrelation(x, particleDensity, "$Density$ $profile$", xlabel = "$x$", color='k')
        #plt.pause(0.5)
        plt.show()

####################### Average linear density profile ########################
def averageLinearDensityProfile(dirName, threshold=0.3, lj='lj', plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    if(lj=='lj'):
        rad *= 2**(1/6)
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    # distance bins
    bins = np.arange(0, boxSize[0], np.max(rad))
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
        pos = utils.centerCOM(pos, rad, boxSize)
        #labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold, compute=True)
        #maxLabel = utils.findLargestParticleCluster(rad, labels)
        #print(maxLabel, labels[labels==maxLabel].shape[0])
        #pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        for i in range(numParticles):
            for j in range(bins.shape[0]-1):
                if(pos[i,0] > bins[j] and pos[i,0] <= bins[j+1]):
                    particleDensity[d,j] += np.pi*rad[i]**2
        particleDensity[d] /= binArea
        # compute width of interface
        center = np.mean(centers)
        x = centers-center
        y = particleDensity[d]
        interfaceWidth[d] = (utils.computeInterfaceWidth(x[x>0], y[x>0]) + utils.computeInterfaceWidth(-x[x<0], y[x<0]))/2
    particleDensity = np.column_stack((np.mean(particleDensity, axis=0), np.std(particleDensity, axis=0)))
    np.savetxt(dirName + os.sep + "densityProfile.dat", np.column_stack((centers, particleDensity)))
    np.savetxt(dirName + os.sep + "interfaceWidth.dat", np.column_stack((timeList, interfaceWidth)))
    print("Interface width:", np.mean(interfaceWidth[interfaceWidth>0]), "+-", np.std(interfaceWidth[interfaceWidth>0]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, interfaceWidth, "$Interface$ $width$")
        #uplot.plotCorrWithError(centers, particleDensity[:,0], particleDensity[:,1], "$Density$ $profile$", xlabel = "$x$", color='k')
        #plt.show()
        plt.pause(0.5)

############################## Work done on a fictitious wall #############################
def computeWallForceVSStrain(dirName, threshold=0.78, which='lj', nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    eps = 1.8*np.max(rad)
    if(which=='harmonic'):
        ec = 240
        cutDistance = 2.5*sigma
    elif(which=='lj'):
        LJcutoff = 5.5
        cutDistance = 2.5*LJcutoff*sigma
    else:
        print("Please specify sample type")
    dirList, strainList = utils.getOrderedStrainDirectories(dirName)
    wallForce = np.zeros((strainList.shape[0],2))
    for d in range(dirList.shape[0]):
        print(dirList[d], strainList[d])
        dirSample = dirName + os.sep + dirList[d]
        boxSize = np.loadtxt(dirSample + "/boxSize.dat")
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        print(maxLabel, labels[labels==maxLabel].shape[0])
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
        print(wallForce[d,0], wallForce[d,1])
    wallForce *= sigma/ec
    np.savetxt(dirName + os.sep + "strainForce.dat", np.column_stack((strainList, wallForce)))
    uplot.plotCorrelation(strainList, wallForce[:,0], color='k', ylabel='$Force$', xlabel='$Strain,$ $\\gamma$')
    uplot.plotCorrelation(strainList, wallForce[:,1], color='r', ylabel='$Force$', xlabel='$Strain,$ $\\gamma$')
    plt.pause(0.5)

############################## Work done on a fictitious wall #############################
def computeWallForceVSTime(dirName, threshold=0.78, which='lj', nDim=2):
    ec = 1
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    eps = 1.8*np.max(rad)
    if(which=='harmonic'):
        ec = 240
        cutDistance = 2.5*sigma
    elif(which=='lj'):
        LJcutoff = 5.5
        cutDistance = 2.5*LJcutoff*sigma
    else:
        print("Please specify sample type")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    wallForce = np.zeros((timeList.shape[0],2))
    boxSize = np.loadtxt(dirName + "/boxSize.dat")
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
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
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
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
    energyLength = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # load particle variables
        epot = np.loadtxt(dirSample + "/particleEnergies.dat")
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        # load simplices
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
        pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
        height = np.zeros(bins.shape[0]-1)
        energy = np.zeros(bins.shape[0]-1)
        posInterface = np.zeros((bins.shape[0]-1,2))
        clusterPos = pos[labels==maxLabel]
        clusterEpot = epot[labels==maxLabel]
        for j in range(bins.shape[0]-1): # find particle positions in a bin
            rightMask = np.argwhere(clusterPos[:,1] > bins[j])[:,0]
            binEpot = clusterEpot[rightMask]
            binPos = clusterPos[rightMask]
            leftMask = np.argwhere(binPos[:,1] <= bins[j+1])[:,0]
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
        if(height[height!=0].shape[0] == height.shape[0]):
            prevPos = posInterface[0]
            length = 0
            for j in range(1,bins.shape[0]-1):
                length += np.linalg.norm(posInterface[j] - prevPos)
                prevPos = posInterface[j]
            energyLength[d,0] = length
            energyLength[d,1] = np.sum(energy)
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
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
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

############################# Exchange timescales ##############################
def computeExchangeTimes(dirName, threshold=0.3, lj='lj', plot=False, dirSpacing=1):
    numBins = 20
    timeStep = float(utils.readFromParams(dirName, "dt"))
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    if(lj=='lj'):
        rad *= 2**(1/6)
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]*timeStep
    condenseTime = np.empty(0)
    evaporateTime = np.empty(0)
    time0 = np.zeros(numParticles)
    dirSample = dirName + os.sep + dirList[0]
    labels0 = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
    denseList0 = np.loadtxt(dirSample + "/particleList.dat")[:,0]
    for d in range(1,dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        maxLabel = utils.findLargestParticleCluster(rad, labels)
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
    if(lj=='lj'):
        rad *= 2**(1/6)
    eps = 1.8*np.max(rad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]*timeStep
    timeSpacing = timeList[1] - timeList[0]
    evaporateNum = np.zeros(dirList.shape[0]-1)
    dirSample = dirName + os.sep + dirList[0]
    denseList0,_ = cluster.getParticleDenseLabel(dirSample, threshold, compute='True')
    #labels0 = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
    #maxLabel0 = utils.findLargestParticleCluster(rad, labels0)
    #denseList0 = np.loadtxt(dirSample + "/particleList.dat")[:,0]
    for d in range(1,dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        denseList,_ = cluster.getParticleDenseLabel(dirSample, threshold)
        #labels = cluster.getParticleClusterLabels(dirSample, boxSize, eps, threshold)
        #maxLabel = utils.findLargestParticleCluster(rad, labels)
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

    elif(whichCorr == "profile"):
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

    elif(whichCorr == "ljprofile"):
        averageLinearLJPressureProfile(dirName, LJcutoff=float(sys.argv[3]), plot=sys.argv[4])

    elif(whichCorr == "ikprofile"):
        averageIKLJPressureProfile(dirName, LJcutoff=float(sys.argv[3]), plot=sys.argv[4])

    elif(whichCorr == "sampleprofile"):
        sampleLinearLJPressureProfile(dirName, numSamples=int(sys.argv[3]), LJcutoff=float(sys.argv[4]), plot=sys.argv[5])
    
############################## Interface structure ##############################
    elif(whichCorr == "singleprofile"):
        threshold = float(sys.argv[3])
        lj = sys.argv[4]
        plot = sys.argv[5]
        computeLinearDensityProfile(dirName, threshold, lj, plot)

    elif(whichCorr == "profile"):
        threshold = float(sys.argv[3])
        lj = sys.argv[4]
        plot = sys.argv[5]
        averageLinearDensityProfile(dirName, threshold, lj, plot)

    elif(whichCorr == "wallstrain"):
        threshold = float(sys.argv[3])
        which = sys.argv[4]
        computeWallForceVSStrain(dirName, threshold, which)

    elif(whichCorr == "walltime"):
        threshold = float(sys.argv[3])
        which = sys.argv[4]
        computeWallForceVSTime(dirName, threshold, which)

    elif(whichCorr == "height"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageClusterHeightVSTime(dirName, threshold, plot)

    elif(whichCorr == "heightflu"):
        threshold = float(sys.argv[3])
        plot = sys.argv[4]
        averageClusterHeightFluctuations(dirName, threshold, plot)

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

    else:
        print("Please specify the correlation you want to compute")
