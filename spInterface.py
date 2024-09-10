'''
Created by Francesco
14 July 2023
'''
#functions and script to compute cluster correlations
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import irfft, rfft, rfftfreq, fft, fftfreq
import sys
import os
import time
import utils
import utilsPlot as uplot
import spCluster as cluster

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
    #eps = 1.8*np.max(rad)
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
    sigma = np.mean(rad)
    #eps = 1.8*np.max(rad)
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
        eps = 1.8*np.max(rad)
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
    if cluster:
        if lj:
            rad *= 2**(1/6)
        eps = 1.8*np.max(rad)
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

####################### Average linear density profile ########################
def average2DensityProfile(dirName, num1=0, plot=False, dirSpacing=1):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
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
    particleDensity = np.zeros((dirList.shape[0], bins.shape[0]-1,2))
    width = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        # first compute particle density
        pos = utils.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
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
        x = centers
        y = particleDensity[d,:,0]
        bulk = x[np.argwhere(y>0.5)[:,0]]
        if(bulk.shape[0] != 0):
            width[d] = bulk[-1] - bulk[0]
        else:
            width[d] = 0
    particleDensity1 = np.column_stack((np.mean(particleDensity[:,:,0], axis=0), np.std(particleDensity[:,:,0], axis=0)))
    particleDensity2 = np.column_stack((np.mean(particleDensity[:,:,1], axis=0), np.std(particleDensity[:,:,1], axis=0)))
    np.savetxt(dirName + os.sep + "densityProfile.dat", np.column_stack((centers, particleDensity1, particleDensity2)))
    np.savetxt(dirName + os.sep + "phaseWidth.dat", np.column_stack((timeList, width)))
    print("average phase 1 width during compression: ", np.mean(width), "+-", np.std(width))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, width, "$Interface$ $width$")
        #uplot.plotCorrWithError(centers, particleDensity1[:,0], particleDensity1[:,1], "$Density$ $profile$", xlabel = "$x$", color='g')
        #uplot.plotCorrWithError(centers, particleDensity2[:,0], particleDensity2[:,1], "$Density$ $profile$", xlabel = "$x$", color='b')
        #plt.show()
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
def compute2FluidsCorr(dirName, startBlock, maxPower, freqPower, num1=0, plot=False):
    boxSize = np.loadtxt(dirName + "boxSize.dat")
    rad = np.loadtxt(dirName + "particleRad.dat").astype(np.float64)
    sigma = 2 * np.mean(rad)
    timeStep = utils.readFromParams(dirName, "dt")
    longWave = 2 * np.pi / sigma
    shortWave = 2 * np.pi / boxSize[1]
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
        particleCorr1 = np.array(particleCorr1).reshape((stepList.shape[0],5))
        particleCorr1 = particleCorr1[np.argsort(stepList)]
        particleCorr2 = np.array(particleCorr2).reshape((stepList.shape[0],5))
        particleCorr2 = particleCorr2[np.argsort(stepList)]
        np.savetxt(dirName + os.sep + "2logCorr.dat", np.column_stack((stepList, particleCorr1, particleCorr2)))
        data = np.column_stack((stepList, particleCorr1))
        tau = utils.getRelaxationTime(data)
        print("Fluid 1: relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
        data = np.column_stack((stepList, particleCorr2))
        tau = utils.getRelaxationTime(data)
        print("Fluid 2: relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
    else:
        particleCorr = np.array(particleCorr).reshape((stepList.shape[0],5))
        particleCorr = particleCorr[np.argsort(stepList)]
        np.savetxt(dirName + os.sep + "logCorr.dat", np.column_stack((stepList, particleCorr)))
        data = np.column_stack((stepList, particleCorr))
        tau = utils.getRelaxationTime(data)
        print("Relaxation time:", tau, "time step:", timeStep, " relaxation step:", tau / timeStep)
    if(plot=="plot"):
        stepList /= timeStep
        if(num1 != 0):
            uplot.plotCorrelation(stepList, particleCorr1[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k')
            uplot.plotCorrelation(stepList, particleCorr1[:,2], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
            uplot.plotCorrelation(stepList, particleCorr2[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k', ls='dashed')
            uplot.plotCorrelation(stepList, particleCorr2[:,2], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r', ls='dashed')
        else:
            uplot.plotCorrelation(stepList, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k')
            uplot.plotCorrelation(stepList, particleCorr[:,2], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
        plt.pause(0.5)
        #plt.show()

############################ Velocity distribution #############################
def average2FluidsVelPDF(dirName, num1=0, plot=False, dirSpacing=1000000):
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
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
        velNorm = np.linalg.norm(vel, axis=1)
        vel1 = np.append(vel1, velNorm[:num1].flatten())
        vel2 = np.append(vel2, velNorm[num1:].flatten())
        veltot = np.append(veltot, velNorm.flatten())
    vel1 = vel1[vel1>0]
    vel2 = vel2[vel2>0]
    veltot = veltot[veltot>0]
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
    kurtosis = np.mean((vel1 - mean)**4)/temp**2
    print("Variance of the velocity in fluid 1: ", temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    print("4th moment:", np.sqrt(0.5*np.mean((vel1 - mean)**4)))
    mean = np.mean(vel2)
    temp = np.var(vel2)
    skewness = np.mean((vel2 - mean)**3)/temp**(3/2)
    kurtosis = np.mean((vel2 - mean)**4)/temp**2
    print("Variance of the velocity in fluid 2: ", temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    print("4th moment:", np.sqrt(0.5*np.mean((vel1 - mean)**4)))
    mean = np.mean(veltot)
    temp = np.var(veltot)
    skewness = np.mean((veltot - mean)**3)/temp**(3/2)
    kurtosis = np.mean((veltot - mean)**4)/temp**2
    print("Variance of the velocity in total: ", temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    print("4th moment:", np.sqrt(0.5*np.mean((veltot - mean)**4)))
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf1, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='g')
        uplot.plotCorrelation(edges, pdf2, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='b')
        uplot.plotCorrelation(edges, pdftot, "$Speed$ $distribution,$ $P(s)$", xlabel = "$Speed,$ $s$", color='k')
        #plt.pause(0.5)
        plt.show()

####################### Average cluster height interface #######################
def average2InterfaceFluctuations(dirName, num1=0, thickness=3, plot=False, dirSpacing=1):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    sigma = 2 * np.mean(rad)
    spacing = 2 * sigma
    bins = np.arange(0, boxSize[1], spacing)
    centers = (bins[1:] + bins[:-1])/2
    #freq = rfftfreq(bins.shape[0]-1, spacing)
    freq = np.arange(1,centers.shape[0]+1,1)/(centers.shape[0] / spacing)
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
        clusterPos = pos[:num1]
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
            leftFourierDeltaHeight[d] = np.abs(rfft(leftHeight))**2
        if(rightHeight[rightHeight!=0].shape[0] == rightHeight.shape[0]):
            rightHeight -= np.mean(rightHeight)
            rightDeltaHeight[d] = np.abs(rightHeight)**2
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
        #plt.pause(0.5)
        plt.show()

####################### Average cluster height interface #######################
def getInterfaceLength(dirName, threshold=0.62, spacing=2, window=3, plot=False, lj=True):
    spacingName = str(spacing)
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    if lj:
        rad *= 2**(1/6)
    sigma = 2*np.mean(rad)
    eps = 1.8*np.max(rad)
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
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    if lj:
        rad *= 2**(1/6)
    sigma = 2*np.mean(rad)
    eps = 1.8*np.max(rad)
    spacing *= sigma
    thickness = 3
    bins = np.arange(0, boxSize[1], spacing)
    edges = (bins[1:] + bins[:-1]) * 0.5
    rightInterface = np.zeros(bins.shape[0]-1)
    leftInterface = np.zeros(bins.shape[0]-1)
    rightError = np.zeros(bins.shape[0]-1)
    leftError = np.zeros(bins.shape[0]-1)
    # load particle variables
    pos = utils.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    # load simplices
    labels, maxLabel = cluster.getParticleClusterLabels(dirName, boxSize, eps, threshold)
    pos = utils.centerSlab(pos, rad, boxSize, labels, maxLabel)
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
        plt.pause(0.5)
        #plt.show()

####################### Average cluster height interface #######################
def averageInterfaceVSTime(dirName, threshold=0.78, plot=False, dirSpacing=1):
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
    sigma = np.max(rad)
    eps = 1.8*np.max(rad)
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
    sigma = np.max(rad)
    eps = 1.8*np.max(rad)
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
    np.savetxt(dirName + os.sep + "heightCorrelation.dat", np.column_stack((distances, heightCorr)))
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
        average2DensityProfile(dirName, num1, plot)

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
        compute2FluidsCorr(dirName, startBlock, maxPower, freqPower, num1, plot)

    elif(whichCorr == "2velpdf"):
        num1 = int(sys.argv[3])
        plot = sys.argv[4]
        average2FluidsVelPDF(dirName, num1, plot)

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

    elif(whichCorr == "interfacelength"):
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
