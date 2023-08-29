'''
Created by Francesco
29 November 2021
'''
#functions and script to compute correlations in space and time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
import pyvoro
import sys
import os
import utils
import utilsPlot as uplot

########################### Pair Correlation Function ##########################
def computePairCorr(dirName, plot="plot"):
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = utils.readFromParams(dirName, "phi")
    rad = np.loadtxt(dirName + os.sep + "particleRad.dat")
    meanRad = np.mean(rad)
    bins = np.linspace(0.1*meanRad, 10*meanRad, 50)
    pos = utils.getPBCPositions(dirName + os.sep + "particlePos.dat", boxSize)
    distance = utils.computeDistances(pos, boxSize)
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
    numParticles = utils.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    timeStep = utils.readFromParams(dirName, "dt")
    particleChi = []
    particleCorr = []
    # get trajectory directories
    stepRange = utils.getDirectories(dirName)
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
        particleChi.append(utils.computeSusceptibility(pPos, pPos0, pField, pWaveVector, scale))
        #particleChi.append(utils.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, scale, oneDim = True))
        particleCorr.append(utils.computeCorrFunctions(pos, pos0, boxSize, pWaveVector, scale, oneDim = True))
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
    numParticles = utils.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = utils.readFromParams(dirName, "phi")
    timeStep = utils.readFromParams(dirName, "dt")
    #pWaveVector = np.pi / (2 * np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pWaveVector = np.pi / computePairCorr(dirName, plot=False)
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    pWaveVector = np.pi / pRad
    print("wave vector: ", pWaveVector)
    particleCorr = []
    # get trajectory directories
    stepRange = utils.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleRad.dat")))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        particleCorr.append(utils.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2, oneDim = True))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,7))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "corr-lin-xdim.dat", np.column_stack((stepRange * timeStep, particleCorr)))
    print("diffusivity: ", np.mean(particleCorr[-20:,0]/(2*stepRange[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(2*stepRange[-20:]*timeStep)))
    uplot.plotCorrelation(stepRange * timeStep, particleCorr[:,0]/(stepRange*timeStep), "$MSD(t)/t$", "$Simulation$ $time,$ $t$", logx = True, color='k')
    #uplot.plotCorrelation(stepRange * timeStep, particleCorr[:,0], "$MSD(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color='k')

########### One Dim Time-averaged Self Corr in log-spaced time window ##########
def computeParticleLogSelfCorrOneDim(dirName, startBlock, maxPower, freqPower):
    numParticles = utils.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = utils.readFromParams(dirName, "phi")
    timeStep = utils.readFromParams(dirName, "dt")
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
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = utils.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(utils.computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, pRad**2, oneDim = True))
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
    numParticles = utils.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = utils.readFromParams(dirName, "phi")
    timeStep = utils.readFromParams(dirName, "dt")
    #pWaveVector = np.pi / (2 * np.sqrt(boxSize[0] * boxSize[1] * phi / (np.pi * numParticles)))
    #pWaveVector = np.pi / computePairCorr(dirName, plot=False)
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    pWaveVector = np.pi / pRad
    particleCorr = []
    # get trajectory directories
    stepRange = utils.getDirectories(dirName)
    stepRange = np.array(np.char.strip(stepRange, 't'), dtype=int)
    stepRange = np.sort(stepRange)
    pPos0 = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particlePos.dat"))
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[0]) + "/particleRad.dat")))
    stepRange = stepRange[stepRange<int(10**maxPower)]
    for i in range(1,stepRange.shape[0]):
        pPos = np.array(np.loadtxt(dirName + os.sep + "t" + str(stepRange[i]) + "/particlePos.dat"))
        particleCorr.append(utils.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
        #particleCorr.append(utils.computeScatteringFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
    particleCorr = np.array(particleCorr).reshape((stepRange.shape[0]-1,7))
    stepRange = stepRange[1:]#discard initial time
    np.savetxt(dirName + os.sep + "linCorr.dat", np.column_stack((stepRange*timeStep, particleCorr)))
    #print("diffusivity: ", np.mean(particleCorr[-20:,0]/(4*stepRange[-20:]*timeStep)), " ", np.std(particleCorr[-20:,0]/(4*stepRange[-20:]*timeStep)))
    #uplot.plotCorrelation(stepRange * timeStep, particleCorr[:,0], "$MSD(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logy = True, logx = True, color = 'k')
    uplot.plotCorrelation(stepRange * timeStep, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')

########## Check Self Correlations by logarithmically spaced blocks ############
def checkParticleSelfCorr(dirName, initialBlock, numBlocks, maxPower, plot="plot", computeTau="tau"):
    colorList = cm.get_cmap('viridis', 10)
    numParticles = utils.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    phi = utils.readFromParams(dirName, "phi")
    timeStep = utils.readFromParams(dirName, "dt")
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
    stepRange = utils.getDirectories(dirName)
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
            particleCorr.append(utils.computeCorrFunctions(pPos, pPos0, boxSize, pWaveVector, pRad**2))
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
    numParticles = utils.readFromParams(dirName, "numParticles")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = utils.readFromParams(dirName, "phi")
    timeStep = utils.readFromParams(dirName, "dt")
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
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = utils.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr.append(utils.computeCorrFunctions(pPos1, pPos2, boxSize, pWaveVector, pRad**2))
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
    uplot.plotCorrelation(stepList * timeStep, particleCorr[:,0]/(stepList*timeStep), "$MSD(\\Delta t)/\\Delta t$", "$time$ $interval,$ $\\Delta t$", logx = True, logy = True, color = 'k')
    #uplot.plotCorrelation(stepList * timeStep, particleCorr[:,1], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'k')
    #uplot.plotCorrelation(stepList * timeStep, particleCorr[:,3], "$ISF(\\Delta t)$", "$time$ $interval,$ $\\Delta t$", logx = True, color = 'r')
    plt.show()
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "particleRad.dat")))
    phi = utils.readFromParams(dirName, "phi")
    timeStep = utils.readFromParams(dirName, "dt")
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
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        #print(multiple, i, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        pPos1, pPos2 = utils.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepParticleCorr += utils.computeSingleParticleISF(pPos1, pPos2, boxSize, pWaveVector, pRad**2)
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
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
            utils.computeLocalTempGrid(pPos, pVel, xbin, ybin, localTemp)
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    pPos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    contacts = np.array(np.loadtxt(dirName + os.sep + "contacts.dat"), dtype = int)
    psi6 = np.zeros(numParticles)
    for i in range(numParticles):
        numContacts = 0
        for c in range(contacts[i].shape[0]):
            if(contacts[i,c] != -1):
                numContacts += 1
                delta = utils.pbcDistance(pPos[i], pPos[contacts[i,c]], boxSize)
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
    distance = utils.computeDistances(pPos, boxSize) / boxSize[0]
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
    numParticles = utils.readFromParams(dirName + os.sep + "t0", "numParticles")
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
    numParticles = utils.readFromParams(dirName, "numParticles")
    timeStep = utils.readFromParams(dirName, "dt")
    particleVelCorr = []
    # get trajectory directories
    stepRange = utils.getDirectories(dirName)
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
    numParticles = utils.readFromParams(dirName, "numParticles")
    timeStep = utils.readFromParams(dirName, "dt")
    particleVelCorr = []
    particleVelVar = []
    # get trajectory directories
    stepRange = utils.getDirectories(dirName)
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
    numParticles = utils.readFromParams(dirName, "numParticles")
    timeStep = utils.readFromParams(dirName, "dt")
    # get trajectory directories
    dirList, timeList = utils.getOrderedDirectories(dirName)
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    timeStep = utils.readFromParams(dirName, "dt")
    #pRad = np.mean(np.array(np.loadtxt(dirName + os.sep + "/particleRad.dat")))
    #phi = utils.readFromParams(dirName, "phi")
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
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        pVel1, pVel2 = utils.readVelPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        #pPos1, pPos2 = utils.readParticlePair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        #pDir1, pDir2 = utils.readDirectorPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        #stepParticleVelCorr.append(utils.computeVelCorrFunctions(pPos1, pPos2, pVel1, pVel2, pDir1, pDir2, waveVector, numParticles))
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    minRad = np.min(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    pos = np.array(np.loadtxt(dirName + os.sep + "particlePos.dat"))
    distance = utils.computeDistances(pos, boxSize)
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
                    delta = utils.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    boxSize = np.array(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    minRad = np.min(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    bins = np.arange(2*minRad, np.sqrt(2)*boxSize[0]/2, 2*minRad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    dirList = dirList[-1:]
    velCorr = np.zeros((bins.shape[0]-1,4))
    counts = np.zeros(bins.shape[0]-1)
    for d in range(dirList.shape[0]):
        pos = np.array(np.loadtxt(dirName + os.sep + dirList[d] + os.sep + "particlePos.dat"))
        distance = utils.computeDistances(pos, boxSize)
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
                            delta = utils.pbcDistance(pos[i], pos[j], boxSize)/distance[i,j]
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
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    phi = utils.readFromParams(dirName, "phi")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    minRad = np.mean(np.loadtxt(dirName + os.sep + "particleRad.dat"))
    rbins = np.arange(0, np.sqrt(2)*boxSize[0]/2, 0.02*minRad)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    timeList = timeList.astype(int)
    dirList = dirList[np.argwhere(timeList%dirSpacing==0)[:,0]]
    pcorr = np.zeros(rbins.shape[0]-1)
    for dir in dirList:
        #pos = np.array(np.loadtxt(dirName + os.sep + dir + "/particlePos.dat"))
        pos = utils.getPBCPositions(dirName + os.sep + dir + "/particlePos.dat", boxSize)
        pcorr += utils.getPairCorr(pos, boxSize, rbins, minRad)/(numParticles * phi)
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
    timeStep = utils.readFromParams(dirName, "dt")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
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
    timeStep = utils.readFromParams(dirName, "dt")
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    dirList, timeList = utils.getOrderedDirectories(dirName)
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

    else:
        print("Please specify the correlation you want to compute")
