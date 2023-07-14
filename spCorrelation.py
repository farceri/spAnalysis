'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
import utilsCorr as ucorr
import utilsPlot as uplot
import pyvoro
import sys
import os

########################### Pair Correlation Function ##########################
def computePairCorr(dirName, plot="plot"):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = ucorr.readFromParams(dirName, "phi")
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    meanRad = np.mean(rad)
    bins = np.linspace(0.1*meanRad, 10*meanRad, 50)
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    distance = ucorr.computeDistances(pos, boxSize)
    pairCorr, edges = np.histogram(distance, bins=bins, density=True)
    binCenter = 0.5 * (edges[:-1] + edges[1:])
    pairCorr /= (2 * np.pi * binCenter)
    firstPeak = binCenter[np.argmax(pairCorr)]
    print("First peak of pair corr is at distance:", firstPeak, "equal to", firstPeak/meanRad, "times the mean radius:", meanRad)
    if(plot == "plot"):
        uplot.plotCorrelation(binCenter, pairCorr, "$Pair$ $correlation$ $function,$ $g(r)$")
        plt.show()
    else:
        return firstPeak

############################ Particle Susceptibility ###########################
def computeParticleSusceptibility(dirName, sampleName, maxPower):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    timeStep = ucorr.readFromParams(dirName, "dt")
    particleChi = []
    particleCorr = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pos0 = np.array(np.loadtxt(dirName + os.sep + "../" + sampleName + "/t" + str(stepRange[0]) + "/particlePos.dat"))
    pField = np.array(np.loadtxt(dirName + os.sep + "externalField.dat"))
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleRad.dat")))
    pWaveVector = np.pi / pRad
    damping = 1e03
    scale = pRad**2
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        pos = np.array(np.loadtxt(dirName + os.sep + "../" + sampleName + "/t" + str(stepRange[i]) + "/particlePos.dat"))
        particleChi.append(ucorr.computeSusceptibility(pPos, pPos0, pField, pWaveVector, scale))
        #particleChi.append(ucorr.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, scale, oneDim = True))
        particleCorr.append(ucorr.computeCorrFunctions(pos, pos0, boxSize, pWaveVector, scale, oneDim = True))
    particleChi = np.array(particleChi)
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,7))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "sus-lin-xdim.dat", np.column_stack((stepRange*timeStep, particleChi)))
    np.savetxt(dirName + os.sep + "../dynamics-test/corr-lin-xdim.dat", np.column_stack((stepRange*timeStep, particleCorr)))
    print("susceptibility: ", np.mean(particleChi[-20:,0]/(stepRange[-20:]*timeStep)), " ", np.std(particleChi[-20:,0]/(stepRange[-20:]*timeStep)))
    #uplot.plotCorrelation(stepRange*timeStep, particleChi[:,0]/(stepRange*timeStep), "$\\chi_0/t$", "$Simulation$ $step$", logx = True, color='k')
    #uplot.plotCorrelation(stepRange*timeStep, particleCorr[:,0]/(2*particleChi[:,0]), "$T_{FDT}$", "$Simulation$ $step$", logx = True, color='k')
    uplot.plotCorrelation(particleCorr[:,1], particleChi[:,1], "$\\chi$", "$ISF$", color='k')

###################### One Dim Particle Self Correlations ######################
def computeParticleSelfCorrOneDim(dirName, maxPower):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    #pWaveVector = np.pi / (2 * np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pWaveVector = np.pi / computePairCorr(dirName, plot=False)
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    pWaveVector = np.pi / pRad
    print("wave vector: ", pWaveVector)
    particleCorr = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleRad.dat")))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        particleCorr.append(ucorr.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2, oneDim = True))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,7))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "corr-lin-xdim.dat", np.column_stack((stepRange * timeStep, particleCorr)))
    print("diffusivity: ", np.mean(particleCorr[-20:,0]/(2*stepRange[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(2*stepRange[-20:]*timeStep)))
    uplot.plotCorrelation(stepRange * timeStep, particleCorr[:,0]/(stepRange*timeStep), "$MSD(t)/t$", "$Simulation$ $time,$ $t$", logx = True, color='k')
    #uplot.plotCorrelation(stepRange * timeStep, particleCorr[:,0], "$MSD(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color='k')

########### One Dim Time-averaged Self Corr in log-spaced time window ##########
def computeParticleLogSelfCorrOneDim(dirName, startBlock, maxPower, freqPower):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
    pWaveVector = np.pi / pRad
    print("wave vector: ", pWaveVector)
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
                    if(ucorr.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = ucorr.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(ucorr.computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, pRad**2, oneDim = True))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],7))
    particleCorr = particleCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "corr-log-xdim.dat", np.column_stack((stepList * timeStep, particleCorr)))
    print("diffusivity on x: ", np.mean(particleCorr[-20:,0]/(2*stepList[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(2*stepList[-20:]*timeStep)))
    uplot.plotCorrelation(stepList * timeStep, particleCorr[:,0]/(stepList*timeStep), "$MSD(\\Delta t)/\\Delta t$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
    #uplot.plotCorrelation(stepList * timeStep, particleCorr[:,0], "$MSD(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color = 'r')
    #uplot.plotCorrelation(stepList * timeStep, particleCorr[:,1], "$ISF(t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

########################## Particle Self Correlations ##########################
def computeParticleSelfCorr(dirName, maxPower):
    computeFrom = 20
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    #pWaveVector = np.pi / (2 * np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pWaveVector = np.pi / computePairCorr(dirName, plot=False)
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    pWaveVector = np.pi / pRad
    particleCorr = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleRad.dat")))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        particleCorr.append(ucorr.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
        #particleCorr.append(ucorr.computeScatteringFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,7))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "linCorr.dat", np.column_stack((stepRange*timeStep, particleCorr)))
    #print("diffusivity: ", np.mean(particleCorr[-20:,0]/(4*stepRange[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(4*stepRange[-20:]*timeStep)))
    #uplot.plotCorrelation(stepRange * timeStep, particleCorr[:,0], "$MSD(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logy = True, logx = True, color = 'k')
    uplot.plotCorrelation(stepRange * timeStep, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

########## Check Self Correlations by logarithmically spaced blocks ############
def checkParticleSelfCorr(dirName, initialBlock, numBlocks, maxPower, plot="plot", computeTau="tau"):
    colorList = cm.get_cmap('viridis', 10)
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    energy = np.loadtxt(dirName + "/energy.dat")
    if(energy[-1,3] < energy[-1,4]):
        T = np.mean(energy[:,3])
    else:
        T = np.mean(energy[:,4])
    print(T)
    #pWaveVector = np.pi / (np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pWaveVector = np.pi /computePairCorr(dirName, plot=False)
    #pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    firstPeak = np.loadtxt(dirName + os.sep + "pcorrFirstPeak.dat")[0]
    pWaveVector = 2*np.pi / firstPeak
    print("wave vector: ", pWaveVector)
    tau = []
    diff = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    decade = int(10**(maxPower-1))
    start = np.argwhere(stepRange==(initialBlock-1)*decade)[0,0]
    for block in np.arange(initialBlock, numBlocks+1, 1, dtype=int):
        particleCorr = []
        pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str((block-1)*decade) + "/particlePos.dat"))
        pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str((block-1)*decade) + "/particleRad.dat")))
        end = np.argwhere(stepRange==(block*decade-int(decade/10)))[0,0]
        #print((block-1)*decade, start, block*decade, end)
        stepBlock = stepRange[start:end+1]
        print(stepBlock[0], stepBlock[-1])
        start = end+1
        for i in range(1,stepBlock.shape[0]):
            pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepBlock[i]) + "/particlePos.dat"))
            particleCorr.append(ucorr.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
        particleCorr = np.array(particleCorr).reshape((stepBlock.shape[0]-1,7))
        stepBlock = stepBlock[1:]-(block-1)*decade#discard initial time
        if(plot=="plot"):
            #uplot.plotCorrelation(stepBlock*timeStep, particleCorr[:,0], "$MSD(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color=colorList(block/10), show=False)
            uplot.plotCorrelation(stepBlock*timeStep, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color=colorList(block/10), show=False)
            plt.pause(0.2)
        if(computeTau=="tau"):
            diff.append(np.mean(particleCorr[-20:,0]/(2*stepBlock[-20:]*timeStep)))
            ISF = particleCorr[:,1]
            step = stepBlock
            relStep = np.argwhere(ISF>np.exp(-1))[-1,0]
            if(relStep + 1 < step.shape[0]):
                t1 = step[relStep]
                t2 = step[relStep+1]
                ISF1 = ISF[relStep]
                ISF2 = ISF[relStep+1]
                slope = (ISF2 - ISF1)/(t2 - t1)
                intercept = ISF2 - slope * t2
                tau.append(timeStep*(np.exp(-1) - intercept)/slope)
            else:
                tau.append(timeStep*step[relStep])
    if(computeTau=="tau"):
        print("relaxation time: ", np.mean(tau), " +- ", np.std(tau))
        print("diffusivity: ", np.mean(diff), " +- ", np.std(diff))
        np.savetxt(dirName + "relaxationData.dat", np.array([[timeStep, phi, T, np.mean(tau), np.std(tau), np.mean(diff), np.std(diff)]]))

########### Time-averaged Self Correlations in log-spaced time window ##########
def computeParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac = 1, computeTau = "tau"):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
    if(qFrac == "read"):
        if(os.path.exists(dirName + os.sep + "pairCorr.dat")):
            pcorr = np.loadtxt(dirName + os.sep + "pairCorr.dat")
            firstPeak = pcorr[np.argmax(pcorr[:,1]),0]
        else:
            firstPeak = computePairCorr(dirName, plot=False)
        pWaveVector = 2 * np.pi / firstPeak
    else:
        pWaveVector = 2 * np.pi / (float(qFrac) * 2 * pRad)
    print("wave vector: ", pWaveVector, " meanRad: ", pRad)
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
                    if(ucorr.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = ucorr.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(ucorr.computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, pRad**2))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(np.mean(stepParticleCorr, axis=0))
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],7))
    particleCorr = particleCorr[np.argsort(stepList)]
    if(qFrac == "read"):
        np.savetxt(dirName + os.sep + "logCorr.dat", np.column_stack((stepList, particleCorr)))
    else:
        np.savetxt(dirName + os.sep + "logCorr-q" + qFrac + ".dat", np.column_stack((stepList, particleCorr)))
    print("diffusivity: ", np.mean(particleCorr[-20:,0]/(4*stepList[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(4*stepList[-20:]*timeStep)))
    #uplot.plotCorrelation(stepList * timeStep, particleCorr[:,0]/(stepList*timeStep), "$MSD(\\Delta t)/\\Delta t$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
    uplot.plotCorrelation(stepList * timeStep, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k')
    uplot.plotCorrelation(stepList * timeStep, particleCorr[:,3], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
    if(computeTau=="tau"):
        diff = np.mean(particleCorr[-1:,0]/(2 * stepRange[-1:] * timeStep))
        ISF = particleCorr[:,1]
        step = stepList
        relStep = np.argwhere(ISF>np.exp(-1))[-1,0]
        if(relStep + 1 < step.shape[0]):
            t1 = step[relStep]
            t2 = step[relStep+1]
            ISF1 = ISF[relStep]
            ISF2 = ISF[relStep+1]
            slope = (ISF2 - ISF1)/(t2 - t1)
            intercept = ISF2 - slope * t2
            tau = timeStep*(np.exp(-1) - intercept)/slope
            print("relaxation time: ", tau)
        else:
            tau = 0
            print("not enough data to compute relaxation time")
        with open(dirName + "../../../tauDiff.dat", "ab") as f:
            np.savetxt(f, np.array([[timeStep, pWaveVector, phi, T, tau, diff]]))

########## Time-averaged Single Correlations in log-spaced time window #########
def computeSingleParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac = 1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = ucorr.readFromParams(dirName, "phi")
    timeStep = ucorr.readFromParams(dirName, "dt")
    T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
    pWaveVector = 2 * np.pi / (float(qFrac) * 2 * pRad)
    print("wave vector: ", pWaveVector)
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
            stepParticleCorr = np.zeros(numParticles)
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(ucorr.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = ucorr.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr += ucorr.computeSingleParticleISF(pPos1, pPos2, boxSize, pWaveVector, pRad**2)
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleCorr.append(stepParticleCorr/numPairs)
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleCorr = np.array(particleCorr).reshape((stepList.shape[0],numParticles))
    particleCorr = particleCorr[np.argsort(stepList)]
    tau = []
    step = stepList
    for i in range(0,numParticles,20):
        #uplot.plotCorrelation(stepList * timeStep, particleCorr[:,0]/(stepList*timeStep), "$MSD(\\Delta t)/\\Delta t$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
        #uplot.plotCorrelation(stepList * timeStep, particleCorr[:,i], "$ISF(t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
        ISF = particleCorr[:,i]
        relStep = np.argwhere(ISF>np.exp(-1))[-1,0]
        #tau.append(step[relStep] * timeStep)
        if(relStep + 1 < step.shape[0]):
            t1 = step[relStep]
            t2 = step[relStep+1]
            ISF1 = ISF[relStep]
            ISF2 = ISF[relStep+1]
            slope = (ISF2 - ISF1)/(t2 - t1)
            intercept = ISF2 - slope * t2
            tau.append(timeStep*(np.exp(-1) - intercept)/slope)
    print("mean relaxation time: ", np.mean(tau), ", std: ", np.std(tau))
    np.savetxt(dirName + "tauSingles.dat", np.array([[timeStep, pWaveVector, phi, T, np.mean(tau), np.var(tau), np.std(tau)]]))

############################## Local Temperature ###############################
def computeLocalTemperaturePDF(dirName, numBins, plot = False):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    tempData = []
    numSamples = 0
    for dir in os.listdir(dirName):
        if(os.path.exists(dirName + os.sep + dir + os.sep + "particleRad.dat")):
            localEkin = np.zeros((numBins, numBins))
            pVel = np.array(np.loadtxt(dirName + os.sep + dir + os.sep + "particleVel.dat"))
            pPos = np.array(np.loadtxt(dirName + os.sep + dir + os.sep + "particlePos.dat"))
            Temp = np.mean(np.linalg.norm(pVel,axis=1)**2)
            pPos[:,0] -= np.floor(pPos[:,0]/boxSize[0]) * boxSize[0]
            pPos[:,1] -= np.floor(pPos[:,1]/boxSize[1]) * boxSize[1]
            ucorr.computeLocalTempGrid(pPos, pVel, xbin, ybin, localTemp)
            tempData.append(localTemp.flatten()/Temp)
    tempData = np.sort(tempData)
    cdf = np.arange(len(tempData))/len(tempData)
    pdf, edges = np.histogram(tempData, bins=np.linspace(np.min(tempData), np.max(tempData), 50), density=True)
    edges = (edges[:-1] + edges[1:])/2
    np.savetxt(dirName + os.sep + "localTemperature-N" + str(numBins) + ".dat", np.column_stack((edges, pdf)))
    if(plot=="plot"):
        fig = plt.figure(dpi=120)
        ax = plt.gca()
        ax.semilogy(edges[1:], pdf[1:], linewidth=1.2, color='k')
        #ax.plot(sampleDensity, cdf, linewidth=1.2, color='k')
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylabel('$P(T_{local})$', fontsize=18)
        ax.set_xlabel('$T_{local}$', fontsize=18)
        plt.tight_layout()
        plt.pause(1)
    mean = np.mean(tempData)
    var = np.var(tempData)
    print("data stats: ", np.min(tempData), np.max(tempData), mean, var)
    return mean, var, np.mean((tempData - np.mean(tempData))**4)/(3*var**2) - 1

def collectLocalTemperaturePDF(dirName, numBins, plot):
    dataSetList = np.array(["0.06", "0.07", "0.08", "0.09", "0.1", "0.11", "0.12", "0.13", "0.14", "0.15", "0.16", "0.17", "0.18", "0.19",
                            "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    data = np.zeros((dataSetList.shape[0], 3))
    for i in range(dataSetList.shape[0]):
        dirSample = dirName + "/T" + dataSetList[i] + "/dynamics/"
        if(os.path.exists(dirSample + os.sep + "t0/params.dat")):
            data[i] = computeLocalTemperaturePDF(dirSample, numBins, plot)
    np.savetxt(dirName + "temperatureData-N" + numBins + ".dat", data)

########################### Hexitic Order Parameter ############################
def computeHexaticOrder(dirName, boxSize):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    contacts = np.array(np.loadtxt(dirName + os.sep + "contacts.dat"), dtype = int)
    psi6 = np.zeros(numParticles)
    for i in range(numParticles):
        numContacts = 0
        for c in range(contacts[i].shape[0]):
            if(contacts[i,c] != -1):
                numContacts += 1
                delta = ucorr.pbcDistance(pPos[i], pPos[contacts[i,c]], boxSize)
                theta = np.arctan2(delta[1], delta[0])
                psi6[i] += np.exp(6j*theta)
        if(numContacts > 0):
            psi6[i] /= numContacts
            psi6[i] = np.abs(psi6[i])
    return psi6

########################## Hexitic Order Correlation ###########################
def computeHexaticCorrelation(dirName, boxSize):
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    psi6 = computeHexaticOrder(dirName)
    distance = ucorr.computeDistances(pPos, boxSize) / boxSize[0]
    bins = np.linspace(np.min(distance[distance>0]), np.max(distance), 50)
    binCenter = 0.5 * (bins[:-1] + bins[1:])
    hexCorr = np.zeros(binCenter.shape[0])
    counts = np.zeros(binCenter.shape[0])
    for i in range(1,pPos.shape[0]):
        for j in range(i):
            for k in range(bins.shape[0]-1):
                if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                    hexCorr[k] += psi6[i] * np.conj(psi6[j])
                    counts[k] += 1
    hexCorr /= counts
    return binCenter, hexCorr

############################ Velocity correlations #############################
def computeParticleVelPDF(dirName, plot=True):
    vel = []
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            vel.append(np.loadtxt(dirName + os.sep + dir + os.sep + "particleVel.dat"))
    vel = np.array(vel).flatten()
    mean = np.mean(vel)
    Temp = np.var(vel)
    skewness = np.mean((vel - mean)**3)/Temp**(3/2)
    kurtosis = np.mean((vel - mean)**4)/Temp**2
    vel /= np.sqrt(2*Temp)
    velPDF, edges = np.histogram(vel, bins=np.linspace(np.min(vel), np.max(vel), 100), density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    print("Variance of the velocity pdf: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    uplot.plotCorrelation(edges, velPDF, "$Velocity$ $distribution,$ $P(c)$", logy = True)

def computeParticleVelPDFSubSet(dirName, firstIndex=10, mass=1e06, plot="plot"):
    vel = []
    velSubSet = []
    temp = []
    tempSubSet = []
    var = []
    varSubSet = []
    step = []
    numParticles = ucorr.readFromParams(dirName + os.sep + "t0", "numParticles")
    nDim = 2
    for dir in os.listdir(dirName):
        if(os.path.isdir(dirName + os.sep + dir)):
            pVel = np.loadtxt(dirName + os.sep + dir + os.sep + "particleVel.dat")
            vel.append(pVel[firstIndex:,:])
            subset = pVel[:firstIndex,:] * np.sqrt(mass)
            velSubSet.append(subset)
            temp.append(np.sum(pVel[firstIndex:,:]**2)/((numParticles - firstIndex)*nDim))
            tempSubSet.append(np.sum(subset**2)/(firstIndex*nDim))
            var.append(np.var(pVel[firstIndex:,:]))
            varSubSet.append(np.var(subset))
            step.append(float(dir[1:]))
    vel = np.array(vel).flatten()
    velSubSet = np.array(velSubSet).flatten()
    temp = np.array(temp)
    tempSubSet = np.array(tempSubSet)
    temp = temp[np.argsort(step)]
    tempSubSet = tempSubSet[np.argsort(step)]
    var = np.array(var)
    varSubSet = np.array(varSubSet)
    var = var[np.argsort(step)]
    varSubSet = varSubSet[np.argsort(step)]
    step = np.sort(step)
    #velSubSet /= np.sqrt(2*np.var(velSubSet))
    velBins = np.linspace(np.min(velSubSet), np.max(velSubSet), 30)
    velPDF, edges = np.histogram(vel, bins=velBins, density=True)
    velSubSetPDF, edges = np.histogram(velSubSet, bins=velBins, density=True)
    edges = 0.5 * (edges[:-1] + edges[1:])
    np.savetxt(dirName + os.sep + "velocityPDF.dat", np.column_stack((edges, velPDF, velSubSetPDF)))
    #print("Variance of the velocity pdf:", np.var(vel), " variance of the subset velocity pdf: ", np.var(velSubSet))
    if(plot=="plot"):
        uplot.plotCorrelation(edges, velPDF / np.sqrt(mass), "$Velocity$ $distribution,$ $P(v)$", xlabel = "$Velocity,$ $v$", logy = True)
    return np.var(vel), np.var(velSubSet)

################## Single Particle Self Velocity Correlations ##################
def computeSingleParticleVelTimeCorr(dirName, particleId = 100):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    timeStep = ucorr.readFromParams(dirName, "dt")
    particleVelCorr = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    stepRange = stepRange[stepRange*timeStep<0.2]
    pVel0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleVel.dat"))[particleId]
    pVel0Squared = np.linalg.norm(pVel0)**2
    for i in range(0,stepRange.shape[0]):
        pVel = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particleVel.dat"))[particleId]
        particleVelCorr.append(np.sum(np.multiply(pVel,pVel0)))
    particleVelCorr /= pVel0Squared
    np.savetxt(dirName + os.sep + "singleVelCorr.dat", np.column_stack(((stepRange+1)*timeStep, particleVelCorr)))
    uplot.plotCorrelation((stepRange + 1) * timeStep, particleVelCorr, "$C_{vv}(\\Delta t)$", "$Time$ $interval,$ $\\Delta t$", color='k')
    plt.show()

##################### Particle Self Velocity Correlations ######################
def computeParticleVelTimeCorr(dirName):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    timeStep = ucorr.readFromParams(dirName, "dt")
    particleVelCorr = []
    particleVelVar = []
    # get trajectory directories
    stepRange = ucorr.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pVel0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleVel.dat"))
    pVel0Squared = np.mean(np.linalg.norm(pVel0,axis=1)**2)
    for i in range(0,stepRange.shape[0]):
        pVel = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particleVel.dat"))
        particleVelCorr.append(np.mean(np.sum(np.multiply(pVel,pVel0), axis=1)))
        meanVel = np.mean(pVel, axis=0)
        particleVelVar.append(np.mean((pVel - meanVel)**2))
    particleVelCorr /= pVel0Squared
    np.savetxt(dirName + os.sep + "velCorr.dat", np.column_stack(((stepRange+1)*timeStep, particleVelCorr, particleVelVar)))
    uplot.plotCorrelation((stepRange + 1) * timeStep, particleVelCorr, "$C_{vv}(\\Delta t)$", "$Time$ $interval,$ $\\Delta t$", color='k')
    uplot.plotCorrelation((stepRange + 1) * timeStep, particleVelVar, "$\\langle \\vec{v}(t) - \\langle \\vec{v}(t) \\rangle \\rangle$", "$Simulation$ $time$", color='r')
    width = stepRange[np.argwhere(particleVelCorr/particleVelCorr[0] < np.exp(-1))[0,0]]*timeStep
    print("Measured damping coefficient: ", 1/width)
    #plt.show()

##################### Particle Self Velocity Correlations ######################
def computeParticleBlockVelTimeCorr(dirName, numBlocks):
    numParticles = ucorr.readFromParams(dirName, "numParticles")
    timeStep = ucorr.readFromParams(dirName, "dt")
    # get trajectory directories
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    blockFreq = dirList.shape[0]//numBlocks
    timeList = timeList[:blockFreq]
    blockVelCorr = np.zeros((blockFreq, numBlocks))
    blockVelVar = np.zeros((blockFreq, numBlocks))
    for block in range(numBlocks):
        pVel0 = np.array(np.loadtxt(dirName + os.sep + dirList[block*blockFreq] + "/particleVel.dat"))
        pVel0Squared = np.mean(np.linalg.norm(pVel0,axis=1)**2)
        for i in range(blockFreq):
            pVel = np.array(np.loadtxt(dirName + os.sep + dirList[block*blockFreq + i] + "/particleVel.dat"))
            blockVelCorr[i, block] = np.mean(np.sum(np.multiply(pVel,pVel0), axis=1))
            meanVel = np.mean(pVel, axis=0)
            blockVelVar[i, block] = np.mean((pVel - meanVel)**2)
        blockVelCorr[:, block] /= pVel0Squared
    particleVelCorr = np.column_stack((np.mean(blockVelCorr, axis=1), np.std(blockVelCorr, axis=1)))
    particleVelVar = np.mean(blockVelVar, axis=1)
    np.savetxt(dirName + os.sep + "blockVelCorr.dat", np.column_stack((timeList * timeStep, particleVelCorr, particleVelVar)))
    uplot.plotCorrelation(timeList * timeStep, particleVelCorr[:,0], "$C_{vv}(\\Delta t)$", "$Time$ $interval,$ $\\Delta t$", color='k')
    uplot.plotCorrelation(timeList * timeStep, particleVelVar, "$\\langle \\vec{v}(t) - \\langle \\vec{v}(t) \\rangle \\rangle$", "$Simulation$ $time$", color='r')
    plt.xscale('log')
    #plt.show()
    width = timeList[np.argwhere(particleVelCorr/particleVelCorr[0] < np.exp(-1))[0,0]]*timeStep
    print("Measured damping coefficient: ", 1/width)

############# Time-averaged Self Vel Corr in log-spaced time window ############
def computeParticleLogVelTimeCorr(dirName, startBlock, maxPower, freqPower):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    timeStep = ucorr.readFromParams(dirName, "dt")
    #pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "/particleRad.dat")))
    #phi = ucorr.readFromParams(dirName, "phi")
    #waveVector = np.pi / (2 * np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    #waveVector = np.pi / (float(radMultiple) * pRad)
    particleVelCorr = []
    #particleDirCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            stepParticleVelCorr = []
            #stepParticleDirCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(ucorr.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        pVel1, pVel2 = ucorr.readVelPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        #pPos1, pPos2 = ucorr.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        #pDir1, pDir2 = ucorr.readDirectorPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        #stepParticleVelCorr.append(ucorr.computeVelCorrFunctions(pPos1, pPos2, pVel1, pVel2, pDir1, pDir2, waveVector, numParticles))
                        stepParticleVelCorr.append(np.mean(np.sum(np.multiply(pVel1,pVel2), axis=1)))
                        #stepParticleDirCorr.append(np.mean(np.sum(np.multiply(pDir1,pDir2), axis=1)))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                particleVelCorr.append([np.mean(stepParticleVelCorr, axis=0), np.std(stepParticleVelCorr, axis=0)])
                #particleDirCorr.append([np.mean(stepParticleDirCorr, axis=0), np.std(stepParticleDirCorr, axis=0)])
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    particleVelCorr = np.array(particleVelCorr).reshape((stepList.shape[0],2))
    particleVelCorr = particleVelCorr[np.argsort(stepList)]
    #particleDirCorr = np.array(particleDirCorr).reshape((stepList.shape[0],2))
    #particleDirCorr = particleDirCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "logVelCorr.dat", np.column_stack((stepList*timeStep, particleVelCorr)))
    #np.savetxt(dirName + os.sep + "logDirCorr.dat", np.column_stack((stepList*timeStep, particleDirCorr)))
    uplot.plotCorrWithError(stepList*timeStep, particleVelCorr[:,0], particleVelCorr[:,1], ylabel="$C_{vv}(\\Delta t)$", logx = True, color = 'k')
    #uplot.plotCorrWithError(stepList*timeStep, particleDirCorr[:,0], particleDirCorr[:,1], ylabel="$C_{nn}(\\Delta t)$", logx = True, color = 'r')
    plt.show()

############################# Velocity Correlation #############################
def computeParticleVelSpaceCorr(dirName):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    minRad = np.min(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    distance = ucorr.computeDistances(pos, boxSize)
    bins = np.arange(2*minRad, np.sqrt(2)*boxSize[0]/2, 2*minRad)
    vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
    velNorm = np.linalg.norm(vel, axis=1)
    velNormSquared = np.mean(velNorm**2)
    velCorr = np.zeros((bins.shape[0]-1,4))
    counts = np.zeros(bins.shape[0]-1)
    for i in range(distance.shape[0]):
        for j in range(i):
            for k in range(bins.shape[0]-1):
                if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                    # parallel
                    delta = ucorr.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
                    parProj1 = np.dot(vel[i],delta)
                    parProj2 = np.dot(vel[j],delta)
                    # perpendicular
                    deltaPerp = np.array([-delta[1], delta[0]])
                    perpProj1 = np.dot(vel[i],deltaPerp)
                    perpProj2 = np.dot(vel[j],deltaPerp)
                    # correlations
                    velCorr[k,0] += parProj1 * parProj2
                    velCorr[k,1] += perpProj1 * perpProj2
                    velCorr[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                    velCorr[k,3] += np.dot(vel[i],vel[j])
                    counts[k] += 1
    binCenter = (bins[1:] + bins[:-1])/2
    for i in range(velCorr.shape[1]):
        velCorr[counts>0,i] /= counts[counts>0]
    velCorr /= velNormSquared
    np.savetxt(dirName + os.sep + "spaceVelCorr1.dat", np.column_stack((binCenter, velCorr, counts)))
    uplot.plotCorrelation(binCenter, velCorr[:,0], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'r')
    uplot.plotCorrelation(binCenter, velCorr[:,1], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'g')
    uplot.plotCorrelation(binCenter, velCorr[:,2], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'k')
    plt.show()

############################# Velocity Correlation #############################
def averageParticleVelSpaceCorr(dirName, dirSpacing=1000):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    minRad = np.min(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    bins = np.arange(2*minRad, np.sqrt(2)*boxSize[0]/2, 2*minRad)
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    dirList = dirList[-1:]
    velCorr = np.zeros((bins.shape[0]-1,4))
    counts = np.zeros(bins.shape[0]-1)
    for d in range(dirList.shape[0]):
        pos = np.array(np.loadtxt(dirName + os.sep + dirList[d] + os.sep + "particlePos.dat"))
        distance = ucorr.computeDistances(pos, boxSize)
        vel = np.array(np.loadtxt(dirName + os.sep + dirList[d] + os.sep + "particleVel.dat"))
        velNorm = np.linalg.norm(vel, axis=1)
        vel[:,0] /= velNorm
        vel[:,1] /= velNorm
        velNormSquared = np.mean(velNorm**2)
        for i in range(distance.shape[0]):
            for j in range(i):
                    for k in range(bins.shape[0]-1):
                        if(distance[i,j] > bins[k] and distance[i,j] <= bins[k+1]):
                            # parallel
                            delta = ucorr.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
                            parProj1 = np.dot(vel[i],delta)
                            parProj2 = np.dot(vel[j],delta)
                            velCorr[k,0] += parProj1 * parProj2
                            # perpendicular
                            deltaPerp = np.array([-delta[1], delta[0]])
                            perpProj1 = np.dot(vel[i],deltaPerp)
                            perpProj2 = np.dot(vel[j],deltaPerp)
                            velCorr[k,1] += perpProj1 * perpProj2
                            # off-diagonal
                            velCorr[k,2] += (perpProj1 * parProj2 + parProj1 * perpProj2)*0.5
                            # total
                            velCorr[k,3] += np.dot(vel[i],vel[j])
                            counts[k] += 1
    for i in range(velCorr.shape[1]):
        velCorr[counts>0,i] /= counts[counts>0]
    #velCorr /= velNormSquared
    binCenter = (bins[1:] + bins[:-1])/2
    np.savetxt(dirName + os.sep + "spaceVelCorr.dat", np.column_stack((binCenter, velCorr, counts)))
    uplot.plotCorrelation(binCenter, velCorr[:,0], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'r')
    uplot.plotCorrelation(binCenter, velCorr[:,1], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'g')
    uplot.plotCorrelation(binCenter, velCorr[:,2], "$C_{vv}(r)$", "$Distance,$ $r$", color = 'k')
    #plt.show()

########################### Average Space Correlator ###########################
def averagePairCorr(dirName, dirSpacing=1000000):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    phi = ucorr.readFromParams(dirName, "phi")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    minRad = np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    rbins = np.arange(0, np.sqrt(2)*boxSize[0]/2, 0.02*minRad)
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    pcorr = np.zeros(rbins.shape[0]-1)
    for dir in dirList:
        #pos = np.array(np.loadtxt(dirName + os.sep + dir + "/particlePos.dat"))
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
        pcorr += ucorr.getPairCorr(pos, boxSize, rbins, minRad)/(numParticles * phi)
    pcorr[pcorr>0] /= dirList.shape[0]
    binCenter = (rbins[:-1] + rbins[1:])*0.5
    firstPeak = binCenter[np.argmax(pcorr)]
    np.savetxt(dirName + os.sep + "pairCorr.dat", np.column_stack((binCenter, pcorr)))
    print("First peak of pair corr is at distance:", firstPeak, "equal to", firstPeak/minRad, "times the min radius:", minRad)
    np.savetxt(dirName + os.sep + "pcorrFirstPeak.dat", np.column_stack((firstPeak, np.max(pcorr))))
    uplot.plotCorrelation(binCenter/minRad, pcorr, "$g(r/\\sigma)$", "$r/\\sigma$")
    plt.pause(0.5)

############################ Collision distribution ############################
def getCollisionIntervalPDF(dirName, check=False, numBins=40):
    timeStep = ucorr.readFromParams(dirName, "dt")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    if(os.path.exists(dirName + "/collisionIntervals.dat") and check=="check"):
        interval = np.loadtxt(dirName + os.sep + "collisionIntervals.dat")
    else:
        interval = np.empty(0)
        previousTime = np.zeros(numParticles)
        previousVel = np.array(np.loadtxt(dirName + os.sep + "t0/particleVel.dat"))
        for i in range(1,dirList.shape[0]):
            currentTime = timeList[i]
            currentVel = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleVel.dat"), dtype=np.float64)
            colIndex = np.argwhere(currentVel[:,0]!=previousVel[:,0])[:,0]
            currentInterval = currentTime-previousTime[colIndex]
            interval = np.append(interval, currentInterval[currentInterval>1])
            previousTime[colIndex] = currentTime
            previousVel = currentVel
        interval = np.sort(interval)
        interval = interval[interval>10]
        interval *= timeStep
        #np.savetxt(dirName + os.sep + "collisionIntervals.dat", interval)
    bins = np.linspace(np.min(interval), np.max(interval), numBins)
    pdf, edges = np.histogram(interval, bins=bins, density=True)
    centers = (edges[1:] + edges[:-1])/2
    print("average collision time:", np.mean(interval), " standard deviation: ", np.std(interval))
    np.savetxt(dirName + os.sep + "collision.dat", np.column_stack((centers, pdf)))
    uplot.plotCorrelation(centers, pdf, "$PDF(\\Delta_c)$", "$Time$ $between$ $collisions,$ $\\Delta_c$", logy=True)
    print("max time: ", timeList[-1]*timeStep, " max interval: ", np.max(interval))
    #plt.show()

###################### Contact rearrangement distribution ######################
def getContactCollisionIntervalPDF(dirName, check=False, numBins=40):
    timeStep = ucorr.readFromParams(dirName, "dt")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    #dirSpacing = 1e04
    #timeList = timeList.astype(int)
    #dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    if(os.path.exists(dirName + "/contactCollisionIntervals.dat") and check=="check"):
        print("loading already existing file")
        interval = np.loadtxt(dirName + os.sep + "contactCollisionIntervals.dat")
    else:
        interval = np.empty(0)
        previousTime = np.zeros(numParticles)
        previousContacts = np.array(np.loadtxt(dirName + os.sep + "t0/particleContacts.dat"))
        for i in range(1,dirList.shape[0]):
            currentTime = timeList[i]
            currentContacts = np.array(np.loadtxt(dirName + os.sep + dirList[i] + "/particleContacts.dat"), dtype=np.int64)
            colIndex = np.unique(np.argwhere(currentContacts!=previousContacts)[:,0])
            currentInterval = currentTime-previousTime[colIndex]
            interval = np.append(interval, currentInterval[currentInterval>1])
            previousTime[colIndex] = currentTime
            previousContacts = currentContacts
        interval = np.sort(interval)
        interval = interval[interval>10]
        interval *= timeStep
        np.savetxt(dirName + os.sep + "contactCollisionIntervals.dat", interval)
    bins = np.arange(np.min(interval), np.max(interval), 10*np.min(interval))
    #bins = np.linspace(np.min(interval), np.max(interval), numBins)
    pdf, edges = np.histogram(interval, bins=bins, density=True)
    centers = (edges[1:] + edges[:-1])/2
    print("average collision time:", np.mean(interval), " standard deviation: ", np.std(interval))
    np.savetxt(dirName + os.sep + "contactCollision.dat", np.column_stack((centers, pdf)))
    centers = centers[np.argwhere(pdf>0)[:,0]]
    pdf = pdf[pdf>0]
    uplot.plotCorrelation(centers, pdf, "$PDF(\\Delta_c)$", "$Time$ $between$ $collisions,$ $\\Delta_c$", logy=True)
    print("max time: ", timeList[-1]*timeStep, " max interval: ", np.max(interval))
    #plt.xlim(0, timeList[-1]*timeStep)
    #plt.show()

############################ Local Packing Fraction ############################
def computeLocalDensity(dirName, numBins, plot = False):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    localArea = np.zeros((numBins, numBins))
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    ucorr.computeLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    timeStep = ucorr.readFromParams(dirName, "dt")
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    cutoff = 2*xbin[1]
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    localDensity = np.empty(0)
    globalDensity = np.empty(0)
    for dir in dirList:
        localArea = np.zeros((numBins, numBins))
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
        contacts = np.array(np.loadtxt(dirName + os.sep + dir + "/particleContacts.dat")).astype(int)
        if(weight == 'weight'):
            globalDensity = np.append(globalDensity, ucorr.computeWeightedLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea, cutoff))
        else:
            globalDensity = np.append(globalDensity, ucorr.computeLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea))
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    xbin = np.linspace(0, boxSize[0], numBins+1)
    ybin = np.linspace(0, boxSize[1], numBins+1)
    localSquare = (boxSize[0]/numBins)*(boxSize[1]/numBins)
    pRad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    localDensityVar = np.empty(0)
    localNumberVar = np.empty(0)
    for dir in dirList:
        localArea = np.zeros((numBins, numBins))
        localNumber = np.zeros((numBins, numBins))
        pos = ucorr.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
        contacts = np.array(np.loadtxt(dirName + os.sep + dir + "/particleContacts.dat"), dtype=int)
        ucorr.computeLocalAreaAndNumberGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea, localNumber)
        localDensity = localArea/localSquare
        localDensityVar = np.append(localDensityVar, np.var(localDensity))
        localNumberVar = np.append(localNumberVar, np.var(localNumber))
    if(plot=="plot"):
        np.savetxt(dirName + "localDensityAndNumberVarVSTime-N" + str(numBins) + ".dat", np.column_stack((timeList, localDensityVar, localNumberVar)))
        uplot.plotCorrelation(timeList, localDensityVar, "$Variance$ $of$ $local$ $density$", "$Time,$ $t$", color='k')
        uplot.plotCorrelation(timeList, localNumberVar, "$Variance$ $of$ $local$ $number$", "$Time,$ $t$", color='g')
        plt.show()

######################### Cluster Velocity Correlation #########################
def searchClusters(dirName, numParticles=None, plot=False, cluster="cluster"):
    if(numParticles==None):
        numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
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
    #    delta = ucorr.pbcDistance(pos[i], clusterPos, boxSize)
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
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat"), dtype=int)
    # use 0.03 as typical distance
    if(eps == 0):
        eps = 2 * np.max(np.loadtxt(dirName + sep + "particleRad.dat"))
    labels = ucorr.getDBClusterLabels(pos, boxSize, eps, min_samples, contacts, contactFilter)
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
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
                pos = ucorr.getPBCPositions(dirSample + os.sep + "particlePos.dat", boxSize)
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

########################### Average Space Correlator ###########################
def averagePairCorrCluster(dirName, dirSpacing=1000000):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    phi = ucorr.readFromParams(dirName, "phi")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    particleRad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    minRad = np.mean(particleRad)
    rbins = np.arange(0, np.sqrt(2)*boxSize[0]/2, 0.02*minRad)
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + os.sep + "particlePos.dat", boxSize)
        pcorrInCluster += ucorr.getPairCorr(pos[clusterLabels==1], boxSize, rbins, minRad)/(NpInCluster * phiInCluster)
        pcorrOutCluster += ucorr.getPairCorr(pos[clusterLabels==0], boxSize, rbins, minRad)/(NpOutCluster * phiOutCluster)
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

################# Cluster contact rearrangement distribution ###################
def getClusterContactCollisionIntervalPDF(dirName, check=False, numBins=40):
    timeStep = ucorr.readFromParams(dirName, "dt")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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

##################### Velocity Correlation in/out Cluster ######################
def computeParticleVelSpaceCorrCluster(dirName):
    sep = ucorr.getDirSep(dirName, 'boxSize')
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
    distance = ucorr.computeDistances(pos, boxSize)
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
                        delta = ucorr.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
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

##################### Velocity Correlation in/out Cluster ######################
def averageParticleVelSpaceCorrCluster(dirName, dirSpacing=1000000):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    bins = np.arange(np.min(rad)/2, np.sqrt(2)*boxSize[0]/2, 2*np.mean(rad))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        distance = ucorr.computeDistances(pos, boxSize)
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
                            delta = ucorr.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
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

########################## Cluster border calculation ##########################
def computeClusterBorder(dirName, plot='plot'):
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    sep = ucorr.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
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
    distances = ucorr.computeDistances(pos, boxSize)
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
    ucorr.computeLocalAreaGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea)
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
            distance = np.linalg.norm(ucorr.pbcDistance(pos[i], probe, boxSize))
            if(distance < (rad[i] + sigma)):
                contact = True
                firstId = i
                break
    # find the closest particle to the initial contact
    minDistance = 1
    for i in clusterParticles:
        distance = np.linalg.norm(ucorr.pbcDistance(pos[i], pos[firstId], boxSize))
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
        delta = ucorr.pbcDistance(probe, pos[contactId], boxSize)
        theta0 = ucorr.checkAngle(np.arctan2(delta[1], delta[0]))
        #print("contactId: ", contactId, " contact angle: ", theta0)
        director = np.array([np.cos(theta0), np.sin(theta0)])
        probe = pos[contactId] + ucorr.polarPos(rad[contactId], theta0)
        theta = ucorr.checkAngle(theta0 + step)
        currentNeighbors = neighbors[contactId]
        currentNeighbors = np.setdiff1d(currentNeighbors, previousNeighbors)
        while theta > theta0:
            newProbe = pos[contactId] + ucorr.polarPos(rad[contactId], theta)
            distance = np.linalg.norm(ucorr.pbcDistance(newProbe, probe, boxSize))
            borderLength += distance
            theta = ucorr.checkAngle(theta + step)
            # loop over neighbors of the current cluster particle and have not been traveled yet
            for i in currentNeighbors[currentNeighbors!=-1]:
                distance = np.linalg.norm(ucorr.pbcDistance(pos[i], newProbe, boxSize))
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
                    distance = np.linalg.norm(ucorr.pbcDistance(pos[i], pos[contactId], boxSize))
                    if(distance < minDistance):
                        minDistance = distance
                        nextId = i
            if(minDistance == 1):
                #print("couldn't find another close particle within the distance cutoff - check all the particles in the cluster")
                for i in currentParticles:
                    if(checkedParticles[i] == 0):
                        distance = np.linalg.norm(ucorr.pbcDistance(pos[i], pos[contactId], boxSize))
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
            distance = np.linalg.norm(ucorr.pbcDistance(pos[contactId], pos[firstId], boxSize))
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

######################### Cluster Velocity Correlation #########################
def computeVelocityField(dirName, numBins=100, plot=False, figureName=None, read=False):
    sep = ucorr.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    bins = np.linspace(-0.5*boxSize[0],0, numBins)
    bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
    bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
    if(read=='read' and os.path.exists(dirName + os.sep + "velocityField.dat")):
        grid = np.loadtxt(dirName + os.sep + "velocityGrid.dat")
        field = np.loadtxt(dirName + os.sep + "velocityField.dat")
    else:
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
        delta = ucorr.computeDeltas(pos, boxSize)
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
    sep = ucorr.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    bins = np.linspace(-0.5*boxSize[0],0, numBins)
    bins = np.concatenate((np.array([bins[0]-(bins[1]-bins[0])]), bins))
    bins = np.concatenate((bins, np.linspace(0,0.5*boxSize[0],numBins)[1:]))
    if(read=='read' and os.path.exists(dirName + os.sep + "dbClusterLabels.dat")):
        clusterLabels = np.loadtxt(dirName + os.sep + "dbClusterLabels.dat")[:,0]
    else:
        clusterLabels,_,_ = searchDBClusters(dirName, eps=0, min_samples=10)
        vel = np.array(np.loadtxt(dirName + os.sep + "particleVel.dat"))
        pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
        delta = ucorr.computeDeltas(pos, boxSize)
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
    sep = ucorr.getDirSep(dirName, 'boxSize')
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    phi = int(ucorr.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    phi = int(ucorr.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
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
        ucorr.computeLocalAreaAndNumberGrid(pos, rad, contacts, boxSize, xbin, ybin, localArea, localNumber)
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
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    phi = int(ucorr.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    phi = int(ucorr.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    timeStep = ucorr.readFromParams(dirName, "dt")
    phi = int(ucorr.readFromParams(dirName, "phi"))
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
                    if(ucorr.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        denseList1, denseList2 = ucorr.readDenseListPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    timeStep = float(ucorr.readFromParams(dirName, "dt"))
    phi = int(ucorr.readFromParams(dirName, "phi"))
    particleRad = np.array(np.loadtxt(dirName + "/particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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

############################## Voronoi clustering ##############################
def computeVoronoiCluster(dirName, threshold=0.65, filter='filter', plot=False):
    sep = ucorr.getDirSep(dirName, "boxSize")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    localDensity = np.zeros(numParticles)
    denseList = np.zeros(numParticles)
    pos = ucorr.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
    # need to center the cluster for voronoi border detection
    pos = ucorr.centerPositions(pos, rad, boxSize)
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
    sep = ucorr.getDirSep(dirName, "boxSize")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    phi = ucorr.readFromParams(dirName + sep, "phi")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    meanRad = np.mean(rad)
    localDensity = np.zeros(numParticles)
    denseList = np.zeros(numParticles)
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    # need to center the cluster for voronoi border detection
    pos = ucorr.centerPositions(pos, rad, boxSize)
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
    borderPos = ucorr.sortBorderPos(borderPos, borderList, boxSize)
    np.savetxt(dirName + os.sep + "borderPos.dat", borderPos)
    # compute border length by summing over segments on the border
    borderLength = 0
    for i in range(1,borderPos.shape[0]):
        borderLength += np.linalg.norm(ucorr.pbcDistance(borderPos[i], borderPos[i-1], boxSize))
    #print("Number of dense particles at the interface: ", borderList[borderList==1].shape[0])
    print("Border length from voronoi edges: ", borderLength)
    np.savetxt(dirName + os.sep + "borderList.dat", borderList)
    np.savetxt(dirName + os.sep + "borderLength.dat", np.array([borderLength]))
    return borderList, borderLength

######################## Average voronoi local density #########################
def averageLocalVoronoiDensity(dirName, numBins=16, plot=False, dirSpacing=1):
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + os.sep + "particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + os.sep + "particleContacts.dat").astype(np.int64)
        localVoroDensity, density = ucorr.computeLocalVoronoiDensityGrid(pos, rad, contacts, boxSize, voroArea, xbin, ybin)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    voronoiDensity = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
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
                overlapArea += ucorr.computeOverlapArea(pos[i], pos[c], rad[i], rad[c], boxSize)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    voronoiArea = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
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
                overlapArea += ucorr.computeOverlapArea(pos[i], pos[c], rad[i], rad[c], boxSize)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    shapeParam = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
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
                overlapArea += ucorr.computeOverlapArea(pos[i], pos[c], rad[i], rad[c], boxSize)
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

############################ Velocity distribution #############################
def averageParticleVelPDFCluster(dirName, plot=False, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    #dirList = dirList[-50:]
    velInCluster = np.empty(0)
    velOutCluster = np.empty(0)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        if(os.path.exists(dirSample + os.sep + "denseList!.dat")):
            denseList = np.loadtxt(dirSample + os.sep + "denseList.dat")
        else:
            denseList,_ = computeVoronoiCluster(dirSample)
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
    print("Variance of the velocity in cluster: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
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
    print("Variance of the velocity out cluster: ", Temp, " kurtosis: ", kurtosis, " skewness: ", skewness)
    if(plot == "plot"):
        uplot.plotCorrelation(edges, pdf, "$Velocity$ $distribution,$ $P(v)$", xlabel = "$Velocity,$ $v$", color='g')
    np.savetxt(dirName + os.sep + "velPDFOutCluster.dat", np.column_stack((edges, pdf)))
    if(plot == "plot"):
        plt.pause(0.5)

############################## Particle pressure ###############################
def computeParticleStress(dirName, prop='prop'):
    dim = 2
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    if(prop == 'prop'):
        Dr = float(ucorr.readFromDynParams(dirName + sep, "Dr"))
        gamma = float(ucorr.readFromDynParams(dirName + sep, "damping"))
        driving = float(ucorr.readFromDynParams(dirName + sep, "f0"))
    stress = np.zeros((numParticles,4))
    #pos = np.loadtxt(dirName + "/particlePos.dat")
    pos = ucorr.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    vel = np.loadtxt(dirName + "/particleVel.dat")
    if(prop == 'prop'):
        #angle = np.loadtxt(dirName + "/particleAngles.dat")
        angle = ucorr.getMOD2PIAngles(dirName + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
    contacts = np.loadtxt(dirName + "/particleContacts.dat").astype(np.int64)
    for i in range(numParticles):
        virial = 0
        #active = v0 - np.sum(vel[i]*director[i])
        energy = 0
        for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
            radSum = rad[i] + rad[c]
            delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        gamma = float(ucorr.readFromDynParams(dirName, "damping"))
        driving = float(ucorr.readFromDynParams(dirName, "f0"))
        Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    wallPressure = np.zeros(dirList.shape[0])
    pressure = np.zeros((dirList.shape[0],3))
    boxLength = 2 * (boxSize[0] + boxSize[1])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        if(prop == "prop"):
            angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
            director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        wall = 0
        virial = 0
        thermal = 0
        active = 0
        for i in range(numParticles):
            # wall pressure
            isWall, wallPos = ucorr.isNearWall(pos[i], rad[i], boxSize)
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
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
        gamma = float(ucorr.readFromDynParams(dirName, "damping"))
        driving = float(ucorr.readFromDynParams(dirName, "f0"))
        Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        #pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        if(prop == "prop"):
            angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
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
            isWall, wallPos = ucorr.isNearWall(pos[i], rad[i], boxSize)
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
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
    Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    driving = float(ucorr.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
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
                    delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
    Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    driving = float(ucorr.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if(shiftx != 0 or shiftx != 0):
            pos = ucorr.shiftPositions(pos, boxSize, -shiftx, -shifty)
        #pos = ucorr.centerPositions(pos, rad, boxSize, denseList)
        centerOfMass = np.mean(pos[denseList==1], axis=0)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            distanceToCOM = np.linalg.norm(ucorr.pbcDistance(pos[i], centerOfMass, boxSize))
            work = 0
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
    Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    driving = float(ucorr.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if(shiftx != 0 or shiftx != 0):
            pos = ucorr.shiftPositions(pos, boxSize, -shiftx, -shifty)
        #pos = ucorr.centerPositions(pos, rad, boxSize, denseList)
        centerOfMass = np.mean(pos[denseList==1], axis=0)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        workn = 0
        workt = 0
        for i in range(numParticles):
            # Convert pressure components into normal and tangential components
            relPos = ucorr.pbcDistance(pos[i], centerOfMass, boxSize)
            relAngle = np.arctan2(relPos[1], relPos[0])
            unitVector = np.array([np.cos(relAngle), np.sin(relAngle)])
            velNT = ucorr.projectToNormalTangentialComp(vel[i], unitVector)
            directorNT = ucorr.projectToNormalTangentialComp(director[i], unitVector)
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
                distance = np.linalg.norm(delta)
                overlap = 1 - distance / radSum
                if(overlap > 0):
                    gradMultiple = ec * overlap / radSum
                    force = gradMultiple * delta / distance
                    # normal minus tangential component
                    forceNT = ucorr.projectToNormalTangentialComp(force, unitVector)
                    deltaNT = ucorr.projectToNormalTangentialComp(delta, unitVector)
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
    Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    driving = float(ucorr.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = ucorr.shiftPositions(pos, boxSize, shiftx, 0)
        #pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            work = 0
            workx = 0
            worky = 0
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
    Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    driving = float(ucorr.readFromDynParams(dirName, "f0"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        pos = ucorr.shiftPositions(pos, boxSize, shiftx, 0)
        #pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
        director = np.array([np.cos(angle), np.sin(angle)]).T
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        for i in range(numParticles):
            workx = 0
            worky = 0
            for c in contacts[i, np.argwhere(contacts[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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

############################### Droplet pressure ###############################
def computeDropletParticleStress(dirName, l1 = 0.04):
    dim = 2
    ec = 240
    l2 = 0.2
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.loadtxt(dirName + sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    rad = np.loadtxt(dirName + sep + "particleRad.dat")
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    stress = np.zeros((numParticles,3))
    #pos = np.loadtxt(dirName + "/particlePos.dat")
    pos = ucorr.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    vel = np.loadtxt(dirName + "/particleVel.dat")
    neighbors = np.loadtxt(dirName + "/particleNeighbors.dat").astype(np.int64)
    for i in range(numParticles):
        virial = 0
        energy = 0
        for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
            radSum = rad[i] + rad[c]
            delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        computeDropletParticleStress(dirSample)

############################### Droplet pressure ###############################
def computeDropletPressureVSTime(dirName, l1=0.1, dirSpacing=1):
    dim = 2
    ec = 240
    l2 = 0.2
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
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
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        if(shiftx != 0 or shiftx != 0):
            pos = ucorr.shiftPositions(pos, boxSize, -shiftx, -shifty)
        #pos = ucorr.centerPositions(pos, rad, boxSize, denseList)
        centerOfMass = np.mean(pos[denseList==1], axis=0)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        workn = 0
        workt = 0
        for i in range(numParticles):
            # Convert pressure components into normal and tangential components
            relPos = ucorr.pbcDistance(pos[i], centerOfMass, boxSize)
            relAngle = np.arctan2(relPos[1], relPos[0])
            unitVector = np.array([np.cos(relAngle), np.sin(relAngle)])
            for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
                forceNT = ucorr.projectToNormalTangentialComp(force, unitVector)
                deltaNT = ucorr.projectToNormalTangentialComp(delta, unitVector)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        for i in range(numParticles):
            work = 0
            workx = 0
            worky = 0
            for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = float(ucorr.readFromDynParams(dirName, "sigma"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        #pos = np.loadtxt(dirSample + "/particlePos.dat")
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        neighbors = np.loadtxt(dirSample + "/particleNeighbors.dat").astype(np.int64)
        workx = 0
        worky = 0
        for i in range(numParticles):
            for c in neighbors[i, np.argwhere(neighbors[i]!=-1)[:,0]]:
                radSum = rad[i] + rad[c]
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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

########################## Total velocity components ###########################
def computeVelMagnitudeVSTime(dirName, plot=False, dirSpacing=1):
    dim = 2
    ec = 240
    gamma = float(ucorr.readFromDynParams(dirName, "damping"))
    driving = float(ucorr.readFromDynParams(dirName, "f0"))
    Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    velMagnitude = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
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
    gamma = float(ucorr.readFromDynParams(dirName, "damping"))
    driving = float(ucorr.readFromDynParams(dirName, "f0"))
    Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
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

############################## Delaunay inclusivity ############################
def checkDelaunay(dirName):
    sep = ucorr.getDirSep(dirName, "boxSize")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    pos = ucorr.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    delaunay = Delaunay(pos)
    checkDelaunayInclusivity(delaunay.simplices, pos, rad, boxSize)

############################## Delaunay clustering #############################
def computeDelaunayCluster(dirName, threshold=0.84, filter='filter', plot=False):
    sep = ucorr.getDirSep(dirName, "boxSize")
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    denseList = np.zeros(numParticles)
    pos = ucorr.getPBCPositions(dirName + "/particlePos.dat", boxSize)
    contacts = np.array(np.loadtxt(dirName + os.sep + "particleContacts.dat")).astype(int)
    pos[:,0] -= np.floor(pos[:,0]/boxSize[0]) * boxSize[0]
    pos[:,1] -= np.floor(pos[:,1]/boxSize[1]) * boxSize[1]
    #delaunay = Delaunay(pos)
    #simplices = delaunay.simplices
    simplices = ucorr.getPBCDelaunay(pos, rad, boxSize)
    # compute delaunay densities
    simplexDensity, simplexArea = ucorr.computeDelaunayDensity(simplices, pos, rad, boxSize)
    denseSimplexList = np.zeros(simplexDensity.shape[0])
    #print("average local density:", np.mean(simplexDensity))
    # first find dense simplices
    for i in range(simplexDensity.shape[0]):
        if(simplexDensity[i] > threshold):
            denseSimplexList[i] = 1
    # if all the simplices touching a particle are dense then the particle is dense
    for i in range(numParticles):
        count = 0
        indices = np.argwhere(simplices==i)[:,0]
        for sIndex in indices:
            if(denseSimplexList[sIndex] == 1):
                denseList[i] = 1
    #print("Number of dense particles: ", denseList[denseList==1].shape[0])
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        pos = ucorr.getPBCPositions(dirSample + os.sep + "particlePos.dat", boxSize)
        simplexPos = ucorr.getDelaunaySimplexPos(pos, rad, boxSize)
        delaunayDensity = ucorr.computeLocalDelaunayDensityGrid(simplexPos, simplexDensity, xbin, ybin)
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
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    delaunayDensity = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        if(os.path.exists(dirSample + os.sep + "denseSimplexList!.dat")):
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexDensity = np.loadtxt(dirSample + os.sep + "simplexDensity.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        else:
            _, simplexDensity = computeDelaunayCluster(dirSample)
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

########################## Compute delaunay density ############################
def computeConsecutiveDelaunayDensity(dirName, plot=False, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    delaunayDensity = np.zeros((dirList.shape[0],3))
    # first load previous density lists to compare consecutive data
    dirSample = dirName + os.sep + dirList[0]
    if(os.path.exists(dirSample + os.sep + "denseSimplexList.dat")):
        denseSimplexList0 = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
        simplexDensity0 = np.loadtxt(dirSample + os.sep + "simplexDensity.dat")
        simplexArea0 = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
    else:
        _, simplexDensity0 = computeDelaunayCluster(dirSample)
        denseSimplexList0 = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
        simplexArea0 = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
    for d in range(1, dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        contacts = np.loadtxt(dirSample + "/particleContacts.dat").astype(np.int64)
        if(os.path.exists(dirSample + os.sep + "denseSimplexList.dat")):
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexDensity = np.loadtxt(dirSample + os.sep + "simplexDensity.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        else:
            _, simplexDensity = computeDelaunayCluster(dirSample)
            denseSimplexList = np.loadtxt(dirSample + os.sep + "denseSimplexList.dat")
            simplexArea = np.loadtxt(dirSample + os.sep + "simplexArea.dat")
        if(denseSimplexList0.shape[0] == denseSimplexList.shape[0]):
            occupiedArea = simplexDensity * simplexArea
            fluidArea = 0
            occupiedFluidArea = 0
            gasArea = 0
            occupiedGasArea = 0
            for i in range(simplexDensity.shape[0]):
                if(denseSimplexList[i]==1 and denseSimplexList0[i]==1):
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
        denseSimplexList0 = denseSimplexList
        simplexDensity0 = simplexDensity
        simplexArea0 = simplexArea
    np.savetxt(dirName + os.sep + "delaunayDensity-con.dat", np.column_stack((timeList, delaunayDensity)))
    print("Density in the fluid: ", np.mean(delaunayDensity[:,0]), " +- ", np.std(delaunayDensity[:,0]))
    print("Density in the gas: ", np.mean(delaunayDensity[:,1]), " +- ", np.std(delaunayDensity[:,1]))
    print("Density in the whole system: ", np.mean(delaunayDensity[:,2]), " +- ", np.std(delaunayDensity[:,2]))
    if(plot=='plot'):
        uplot.plotCorrelation(timeList, delaunayDensity[:,0], "$\\varphi^{Delaunay}$", xlabel = "$Time,$ $t$", color='b')
        uplot.plotCorrelation(timeList, delaunayDensity[:,1], "$\\varphi^{Delaunay}$", xlabel = "$Time,$ $t$", color='g')
        uplot.plotCorrelation(timeList, delaunayDensity[:,2], "$\\varphi^{Delaunay}$", xlabel = "$Time,$ $t$", color='k')
        #plt.pause(0.5)
        plt.show()

######################## Compute delaunay cluster border #######################
def computeDelaunayBorder(dirName, threshold=0.85, filter='filter'):
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    numParticles = int(ucorr.readFromParams(dirName + sep, "numParticles"))
    phi = ucorr.readFromParams(dirName + sep, "phi")
    pos = ucorr.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    # need to center the cluster for voronoi border detection
    pos = ucorr.centerPositions(pos, rad, boxSize)
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
    borderPos = ucorr.sortBorderPos(borderPos, borderList, boxSize)
    np.savetxt(dirName + os.sep + "borderPos.dat", borderPos)
    # compute border length by summing over segments on the border
    borderLength = 0
    for i in range(1,borderPos.shape[0]):
        borderLength += np.linalg.norm(ucorr.pbcDistance(borderPos[i], borderPos[i-1], boxSize))
    #print("Number of dense particles at the interface: ", borderList[borderList==1].shape[0])
    #print("Border length from delaunay edges: ", borderLength)
    np.savetxt(dirName + os.sep + "delaunayBorderList.dat", borderList)
    np.savetxt(dirName + os.sep + "delaunayBorderLength.dat", np.array([borderLength]))
    return borderList, borderLength

######################## Compute cluster shape parameter #######################
def computeClusterDelaunayShape(dirName, plot=False, dirSpacing=1):
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    sep = ucorr.getDirSep(dirName, "boxSize")
    boxSize = np.array(np.loadtxt(dirName + sep + "boxSize.dat"))
    rad = np.array(np.loadtxt(dirName + sep + "particleRad.dat"))
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    timeList = timeList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    shapeParam = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d]
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
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
    gamma = float(ucorr.readFromDynParams(dirName, "damping"))
    driving = float(ucorr.readFromDynParams(dirName, "f0"))
    Dr = float(ucorr.readFromDynParams(dirName, "Dr"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    numParticles = int(ucorr.readFromParams(dirName, "numParticles"))
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    sigma = np.mean(rad)
    ec = 240
    dirList, timeList = ucorr.getOrderedDirectories(dirName)
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
        # first fill out the area occupied by fluid, gas and interface
        fluidArea = 0
        gasArea = 0
        for i in range(simplexArea.shape[0]):
            if(denseSimplexList[i]==1):
                fluidArea += simplexArea[i]
            else:
                gasArea += simplexArea[i]
        # compute stress components
        pos = ucorr.getPBCPositions(dirSample + "/particlePos.dat", boxSize)
        vel = np.loadtxt(dirSample + "/particleVel.dat")
        angle = ucorr.getMOD2PIAngles(dirSample + "/particleAngles.dat")
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
            isWall, wallPos = ucorr.isNearWall(pos[i], rad[i], boxSize)
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
                delta = ucorr.pbcDistance(pos[i], pos[c], boxSize)
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


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichCorr = sys.argv[2]

    if(whichCorr == "paircorr"):
        plot = sys.argv[3]
        computePairCorr(dirName, plot)

    elif(whichCorr == "sus"):
        sampleName = sys.argv[3]
        maxPower = int(sys.argv[4])
        computeParticleSusceptibility(dirName, sampleName, maxPower)

    elif(whichCorr == "lincorrx"):
        maxPower = int(sys.argv[3])
        computeParticleSelfCorrOneDim(dirName, maxPower)

    elif(whichCorr == "logcorrx"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeParticleLogSelfCorrOneDim(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "lincorr"):
        maxPower = int(sys.argv[3])
        computeParticleSelfCorr(dirName, maxPower)

    elif(whichCorr == "checkcorr"):
        initialBlock = int(sys.argv[3])
        numBlocks = int(sys.argv[4])
        maxPower = int(sys.argv[5])
        plot = sys.argv[6]
        computeTau = sys.argv[7]
        checkParticleSelfCorr(dirName, initialBlock, numBlocks, maxPower, plot=plot, computeTau=computeTau)

    elif(whichCorr == "logcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        qFrac = sys.argv[6]
        computeTau = sys.argv[7]
        computeParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac, computeTau=computeTau)

    elif(whichCorr == "corrsingle"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        qFrac = sys.argv[6]
        computeSingleParticleLogSelfCorr(dirName, startBlock, maxPower, freqPower, qFrac)

    elif(whichCorr == "temppdf"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        computeLocalTemperaturePDF(dirName, numBins, plot)

    elif(whichCorr == "collecttemppdf"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        collectLocalTemperaturePDF(dirName, numBins, plot)

    elif(whichCorr == "velpdf"):
        computeParticleVelPDF(dirName)

    elif(whichCorr == "velsubset"):
        firstIndex = int(sys.argv[3])
        mass = float(sys.argv[4])
        computeParticleVelPDFSubSet(dirName, firstIndex, mass)

    elif(whichCorr == "singlevelcorr"):
        particleId = int(sys.argv[3])
        computeSingleParticleVelTimeCorr(dirName, particleId)

    elif(whichCorr == "velcorr"):
        computeParticleVelTimeCorr(dirName)

    elif(whichCorr == "blockvelcorr"):
        numBlocks = int(sys.argv[3])
        computeParticleBlockVelTimeCorr(dirName, numBlocks)

    elif(whichCorr == "logvelcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        computeParticleLogVelTimeCorr(dirName, startBlock, maxPower, freqPower)

    elif(whichCorr == "vc"):
        computeParticleVelSpaceCorr(dirName)

    elif(whichCorr == "averagevc"):
        averageParticleVelSpaceCorr(dirName)

    elif(whichCorr == "averagepc"):
        dirSpacing = int(sys.argv[3])
        averagePairCorr(dirName, dirSpacing)

    elif(whichCorr == "collision"):
        check = sys.argv[3]
        numBins = int(sys.argv[4])
        getCollisionIntervalPDF(dirName, check, numBins)

    elif(whichCorr == "contactcol"):
        check = sys.argv[3]
        numBins = int(sys.argv[4])
        getContactCollisionIntervalPDF(dirName, check, numBins)

    elif(whichCorr == "riorient"):
        check = sys.argv[3]
        numBins = int(sys.argv[4])
        getRiorientationIntervalPDF(dirName, check, numBins)

    elif(whichCorr == "activep"):
        computeActivePressure(dirName)

    elif(whichCorr == "density"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        computeLocalDensity(dirName, numBins, plot)

    elif(whichCorr == "nphitime"):
        numBins = int(sys.argv[3])
        plot = sys.argv[4]
        computeLocalDensityAndNumberVSTime(dirName, numBins, plot)

    elif(whichCorr == "averageld"):
        numBins = int(sys.argv[3])
        weight = sys.argv[4]
        plot = sys.argv[5]
        averageLocalDensity(dirName, numBins, weight, plot)

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

    elif(whichCorr == "velpdfcluster"):
        plot = sys.argv[3]
        averageParticleVelPDFCluster(dirName, plot)

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

    elif(whichCorr == "border"):
        plot = sys.argv[3]
        computeClusterBorder(dirName, plot)

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

############################## pressure functions ##############################
    elif(whichCorr == "stress"):
        prop = sys.argv[3]
        computeParticleStress(dirName, prop)

    elif(whichCorr == "stresstime"):
        computeParticleStressVSTime(dirName)

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

    elif(whichCorr == "veltime"):
        plot = sys.argv[3]
        computeVelMagnitudeVSTime(dirName, plot)

    elif(whichCorr == "clusterveltime"):
        plot = sys.argv[3]
        computeClusterVelMagnitudeVSTime(dirName, plot)

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

    elif(whichCorr == "delcondensity"):
        plot = sys.argv[3]
        computeConsecutiveDelaunayDensity(dirName, plot)

    elif(whichCorr == "delshape"):
        plot = sys.argv[3]
        computeClusterDelaunayShape(dirName, plot)

    elif(whichCorr == "clusterdptime"):
        computeDelaunayClusterPressureVSTime(dirName)

    else:
        print("Please specify the correlation you want to compute")
