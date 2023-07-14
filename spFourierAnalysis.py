'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import utilsCorr as ucorr
import utilsPlot as uplot
import sys
import os

############################ Static Structure Factor ###########################
def computeStructureFactor(dirName, plot="plot"):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = ucorr.readFromParams(dirName, "phi")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    meanRad = np.mean(rad)
    qList = np.arange(np.pi/2, np.pi/(15*np.min(rad)), 2*np.min(rad))
    pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
    sfList = ucorr.getStructureFactor(pos, qList, numParticles)
    if(plot == "plot"):
        uplot.plotCorrelation(qList, sfList, "$Structure$ $factor,$ $S(q)$", "$Wave$ $vector$ $magnitude,$ $q$")
    else:
        return sfList

def averageStructureFactor(dirName, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    scale = 2*np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"),dtype=np.float64)
    qList = np.arange(1.2, np.pi/(5*scale), 2*scale)
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    sf = np.zeros(qList.shape[0])
    for dir in dirList:
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)/scale
        sf += ucorr.getStructureFactor(pos, qList, numParticles)
    sf[sf>0] /= dirList.shape[0]
    np.savetxt(dirName + os.sep + "structureFactor.dat", np.column_stack((qList, sf)))
    uplot.plotCorrelation(qList, sf, "$Structure$ $factor,$ $S(q)$", "$Wave$ $vector$ $magnitude,$ $q$", markersize=0, linewidth=1)#, logx=True, logy=True)

########################## Average Velocity Correlator #########################
def averageVelocityStructureFactor(dirName, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    scale = 2*np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    qList = np.concatenate((np.geomspace(1e-03, 1e-01, 30), np.geomspace(1e-01, 1, 50), np.geomspace(1, 3, 50)))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    velsf = np.zeros(qList.shape[0])
    for dir in dirList:
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)/scale
        vel = np.array(np.loadtxt(dirName + os.sep + dir + "/particleVel.dat"), dtype=np.float64)
        velsf += ucorr.getVelocityStructureFactor(pos, vel, qList, numParticles)
    velsf[velsf>0] /= dirList.shape[0]
    np.savetxt(dirName + os.sep + "velocitySF.dat", np.column_stack((qList, velsf)))
    uplot.plotCorrelation(qList, velsf/velsf[-1], "$Velocity$ $structure$ $factor,$ $S_{v}(q)$", "$Wave$ $vector$ $magnitude,$ $q$", markersize=0, linewidth=1)

########################### Energy Transform in Space ##########################
def averageSpaceFourierEnergy(dirName, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    scale = 2*np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    #qList = np.linspace(1e-03, np.pi/scale, 50)
    qList = np.concatenate((np.geomspace(1e-03, 1e-01, 30), np.geomspace(1e-01, 1, 50), np.geomspace(1, 3, 50)))
    #qList = np.concatenate((-qList, qList))
    qList = np.sort(qList)
    energyq = np.zeros((qList.shape[0],3))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    for dir in dirList:
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)/scale
        vel = np.array(np.loadtxt(dirName + os.sep + dir + "/particleVel.dat"), dtype=np.float64)
        epot = np.array(np.loadtxt(dirName + os.sep + dir + "/particleEnergy.dat"), dtype=np.float64)
        kqtemp, uqtemp, kctemp = ucorr.getSpaceFourierEnergy(pos, vel, epot, qList, numParticles)
        energyq[:,0] += kqtemp
        energyq[:,1] += uqtemp
        energyq[:,2] += kctemp
    energyq[energyq>0] /= dirList.shape[0]
    np.savetxt(dirName + os.sep + "spaceFourierEnergy.dat", np.column_stack((qList, energyq)))
    uplot.plotCorrelation(qList, energyq[:,0], "$\\tilde{K}(q)$", "$Wave$ $vector$ $magnitude,$ $q$", markersize=0, linewidth=1, logy=True)
    #plt.show()

########################### Energy Transform in Time ###########################
def averageTimeFourierEnergy(dirName, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    energyTransform = ucorr.getTimeFourierEnergy(dirName, dirList, dirSpacing, numParticles)
    np.savetxt(dirName + os.sep + "timeFourierEnergy.dat", energyTransform)
    uplot.plotCorrelation(energyTransform[:,0], energyTransform[:,1], "$\\langle \\tilde{K}(\\omega) \\rangle$", "$Frequency,$ $\\omega$", markersize=0, linewidth=1, logy=True)
    #plt.show()

########################### Average Velocity Transform #########################
def averageSpaceFourierVelocity(dirName, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    scale = 2*np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    qList = np.geomspace(1e-03, 10, 150)#, np.geomspace(1, 3, 50)))
    #qList = np.linspace(2*np.pi/scale, np.sqrt(boxSize[0]*boxSize[0] + boxSize[1]*boxSize[1]), 100)
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    vf = np.zeros((qList.shape[0],2))
    for dir in dirList:
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)/scale
        vel = np.array(np.loadtxt(dirName + os.sep + dir + "/particleVel.dat"), dtype=np.float64)
        vf += ucorr.getSpaceFourierVelocity(pos, vel, qList, numParticles)
    vf[vf>0] /= dirList.shape[0]
    np.savetxt(dirName + os.sep + "spaceFourierVel.dat", np.column_stack((qList, vf)))
    uplot.plotCorrelation(qList, vf[:,0], "$\\tilde{v}_\\parallel(q)$", "$Wave$ $vector$ $magnitude,$ $q$", markersize=0, linewidth=1, logx=True)
    #plt.show()

######################### Velocity Correlation in Time #########################
def averageTimeFourierVel(dirName, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    timeStep = ucorr.readFromParams(dirName, "dt")
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    velTransform = ucorr.getTimeFourierVel(dirName, dirList, dirSpacing, numParticles)
    spacing = (timeList[1]-timeList[0])
    velTransform[:,0] /= spacing
    np.savetxt(dirName + os.sep + "fourierVelCorr.dat", velTransform)
    uplot.plotCorrelation(velTransform[:,0], velTransform[:,2], "$\\langle | \\vec{v}(\\omega) |^2 \\rangle$", "$Frequency,$ $\\omega$", markersize=0, linewidth=1, logy=True, logx=True)
    #plt.show()

if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    if(whichCorr == "psf"):
        plot = sys.argv[3]
        computeStructureFactor(dirName)

    elif(whichCorr == "averagesf"):
        dirSpacing = int(sys.argv[3])
        averageStructureFactor(dirName, dirSpacing)

    elif(whichCorr == "averagevsf"):
        dirSpacing = int(sys.argv[3])
        averageVelocityStructureFactor(dirName, dirSpacing)

    elif(whichCorr == "spaceeq"):
        dirSpacing = int(sys.argv[3])
        averageSpaceFourierEnergy(dirName, dirSpacing)

    elif(whichCorr == "timeeq"):
        dirSpacing = int(sys.argv[3])
        averageTimeFourierEnergy(dirName, dirSpacing)

    elif(whichCorr == "spacevf"):
        dirSpacing = int(sys.argv[3])
        averageSpaceFourierVelocity(dirName, dirSpacing)

    elif(whichCorr == "velf"):
        dirSpacing = int(sys.argv[3])
        averageTimeFourierVel(dirName, dirSpacing)

    else:
        print("Please specify the correlation you want to compute")
