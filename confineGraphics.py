'''
Created by Francesco
7 November 2024
'''
#functions for soft particle packing visualization
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import sys
import os
import utils
import utilsPlot as uplot

def getIndexYlabel(which):
    if(which == "corr"):
        index = 5
        ylabel = "$Velocity$ $correlation,$ $C_{vv}$"
    elif(which == "epot"):
        index = 2
        ylabel = "$Potential$ $energy,$ $U$"
    elif(which == "ekin"):
        index = 3
        ylabel = "$Kinetic$ $energy,$ $K$"
    elif(which == "prad"):
        index = -2
        ylabel = "$Radial$ $pressure,$ $P_r$"
    elif(which == "ptheta"):
        index = -1
        ylabel = "$Tangential$ $pressure,$ $P_\\phi$"
    else:
        index = 4
        ylabel = "$|\\langle L \\rangle |$"
    return index, ylabel

########################## plot alignment in active systems ##########################
def plotAlignment(dirName, figureName):
    if(os.path.exists(dirName + "/energy.dat")):
        energy = np.loadtxt(dirName + os.sep + "energy.dat")
        print("potential energy:", np.mean(energy[:,2]), "+-", np.std(energy[:,2]))
        print("temperature:", np.mean(energy[:,3]), "+-", np.std(energy[:,3]))
        print("velocity alignment:", np.mean(energy[:,-1]), "+-", np.std(energy[:,-1]), "relative error:", np.std(energy[:,-1])/np.mean(energy[:,-1]))
        fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
        ax.plot(energy[:,0], energy[:,-1], linewidth=1.5, color='k')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel("$Simulation$ $step$", fontsize=16)
        ax.set_ylabel("$Velocity$ $alignment$", fontsize=16)
        plt.tight_layout()
        figureName = "/home/francesco/Pictures/soft/align-" + figureName
        fig.savefig(figureName + ".png", transparent=True, format = "png")
        plt.show()
    else:
        print("no energy.dat file was found in", dirName)

def plotAlignmentVSInteraction(dirName, figureName, which, dynamics="/"):
    dirList = np.array(["1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02"])
    jvic = np.zeros(dirList.shape[0])
    align = np.zeros((dirList.shape[0], 2))
    index, ylabel = getIndexYlabel(which)
    for d in range(dirList.shape[0]):
        dirSample = dirName + "j" + dirList[d] + "-tp1" + dynamics
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            if(index == 4):
                align[d,0] = np.mean(np.abs(data[:,index]))
                align[d,1] = np.std(np.abs(data[:,index]))
            else:
                align[d,0] = np.mean(data[:,index])
                align[d,1] = np.std(data[:,index])
            jvic[d] = utils.readFromDynParams(dirSample, "Jvicsek") / utils.readFromDynParams(dirSample, "damping")
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    plt.errorbar(jvic[jvic!=0], align[jvic!=0,0], align[jvic!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Alignment$ $interaction,$ $J_{Vicsek}$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/alignVSinter-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotAlignmentVSDamping(dirName, figureName, which, dynamics="/"):
    typeList = np.array(["fixed/langevin", "fixed/driven", "reflect/langevin", "reflect/driven"])
    labelList = np.array(["$WCA$ $Langevin$", "$WCA$ $damped$", "$Elastic$ $Langevin$", "$Elastic$ $damped$"])
    dirList = np.array(["1e-06", "1e-04", "1e-02", "1e-01", "1", "1e01", "1e02"])
    markerList = ['v', 'v', 'o', 'o']
    colorList = ['b', 'b', 'g', 'g']
    fillList = ['full', 'none', 'full', 'none']
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    index, ylabel = getIndexYlabel(which)
    for t in range(typeList.shape[0]):
        beta = np.zeros(dirList.shape[0])
        align = np.zeros((dirList.shape[0], 2))
        for d in range(dirList.shape[0]):
            dirSample = dirName + typeList[t] + dirList[d] + os.sep + "j1e03" + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")  
                if(index == 4):
                    align[d,0] = np.mean(np.abs(data[:,index]))
                    align[d,1] = np.std(np.abs(data[:,index]))
                else:
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])
                beta[d] = utils.readFromDynParams(dirSample, "damping")
        ax.errorbar(beta[beta!=0], align[beta!=0,0], align[beta!=0,1], color=colorList[t], marker=markerList[t], markersize=8, capsize=3, fillstyle=fillList[t], lw=1, label=labelList[t])
    ax.legend(fontsize=12, loc='best')
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Damping$ $coefficient,$ $\\beta$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/alignVSBeta-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareAlignmentVSTemperature(dirName, figureName, which, dynamics="/"):
    typeList = np.array(["reflect/langevin1e-01/", "reflect/langevin1/", "reflect/langevin2/", "reflect/langevin1e01/"])
    beta = ['1e-01', '1', '2', '1e01']
    dirList = np.array(["4e-04", "1e-03", "2e-03", "4e-03", "1e-02", "2e-02", "4e-02", "1e-01", "2e-01", "4e-01", "1"])
    markerList = ['o', 's', 'v', '^']
    colorList = ['k', 'r', 'g', 'b']
    fillList = ['none', 'none', 'none', 'none']
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    index, ylabel = getIndexYlabel(which)
    for t in range(typeList.shape[0]):
        noise = np.zeros(dirList.shape[0])
        align = np.zeros((dirList.shape[0], 2))
        for d in range(dirList.shape[0]):
            dirSample = dirName + typeList[t] + "T" + dirList[d] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                if(index == 4):
                    align[d,0] = np.mean(np.abs(data[:,index]))
                    align[d,1] = np.std(np.abs(data[:,index]))
                elif(index == -1 or index == -2):
                    align[d,0] = np.mean(867*data[:,index])
                    align[d,1] = np.std(867*data[:,index])/np.sqrt(data.shape[0])
                else:
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])
                noise[d] = np.sqrt(2 * utils.readFromDynParams(dirSample, "damping") * float(dirList[d]))
        ax.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color=colorList[t], marker=markerList[t], markersize=8, capsize=3, fillstyle=fillList[t], lw=1, label="$\\beta =$" + beta[t])
    ax.legend(fontsize=12, loc='best')
    ax.set_xscale('log')
    #if(which == 'corr'):
        #ax.set_xlim(0.008,)
        #ax.set_ylim(-0.02,1.08)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2 m \\beta k_B T}$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/" + which + "VSTemp-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotAlignmentVSNoise(dirName, figureName, which, jvic="1e02", dynamics="/"):
    dirList = np.array(["1e-04", "2e-04", "3e-04", "5e-04", "1e-03", "2e-03", "3e-03", "5e-03", "1e-02", "1.5e-02", "2e-02", "2.5e-02", "3e-02", "4e-02",
                        "5e-02", "7e-02", "1e-01", "1.5e-01", "2e-01", "3e-01", "5e-01", "1", "2", "3", "5", "1e01", "2e01", "3e01", "5e01", "1e02",
                        "2e02", "3e02", "5e02", "1e03", "2e03", "3e03", "5e03", "1e04", "2e04", "3e04", "5e04", "1e05", "2e05", "3e05", "5e05", "1e06"])
    noise = np.zeros(dirList.shape[0])
    align = np.zeros((dirList.shape[0], 2))
    index, ylabel = getIndexYlabel(which)
    for d in range(dirList.shape[0]):
        if(jvic == "active"):
            dirSample = dirName + "tp" + dirList[d] + dynamics
        else:
            dirSample = dirName + "j" + jvic + "-tp" + dirList[d] + dynamics
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            if(index == 4):
                align[d,0] = np.mean(np.abs(data[:,index]))
                align[d,1] = np.std(np.abs(data[:,index]))
            else:
                align[d,0] = np.mean(data[:,index])
                align[d,1] = np.std(data[:,index])
            noise[d] = np.sqrt(2*utils.readFromParams(dirSample, "dt")/utils.readFromDynParams(dirSample, "taup"))
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    plt.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
    #plt.errorbar(noise[22], align[22,0], align[22,1], color='r', marker='*', markersize=8, capsize=3, fillstyle='none', lw=1)
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2\\Delta t/\\tau_p}$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    if(jvic == "active"):
        figureName = "/home/francesco/Pictures/soft/alignVSnoise-" + which + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/alignVSnoise-" + which + "j" + jvic + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareAlignmentVSNoise(dirName, figureName, which, dynamics="/"):
    typeList = np.array(["vicsek/reflect", "vicsek/fixed"])#, "active/fixed"])
    labelList = np.array(["$Elastic$ $wall$", "$WCA$ $wall$", "$Active$ $noise$"])
    markerList = ['o', 's', 'v']
    colorList = ['k', 'r', 'b']
    dirList = np.array(["1e-04", "2e-04", "3e-04", "5e-04", "1e-03", "2e-03", "3e-03", "5e-03", "1e-02", "1.5e-02", "2e-02", "2.5e-02", "3e-02", "4e-02",
                        "5e-02", "7e-02", "1e-01", "1.5e-01", "2e-01", "3e-01", "5e-01", "1", "2", "3", "5", "1e01", "2e01", "3e01", "5e01", "1e02",
                        "2e02", "3e02", "5e02", "1e03", "2e03", "3e03", "5e03", "1e04", "2e04", "3e04", "5e04", "1e05", "2e05", "3e05", "5e05", "1e06"])
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    index, ylabel = getIndexYlabel(which)
    print(which, index)
    for t in range(typeList.shape[0]):
        noise = np.zeros(dirList.shape[0])
        align = np.zeros((dirList.shape[0], 2))
        for d in range(dirList.shape[0]):
            if(typeList[t] == "active/fixed"):
                dirSample = dirName + typeList[t] + "/tp" + dirList[d] + dynamics
            else:
                dirSample = dirName + typeList[t] + "/j1e03-tp" + dirList[d] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                if(index == 4):
                    if(typeList[t] == "active/fixed"):
                        align[d,0] = np.mean(np.abs(data[:,-1]))
                        align[d,1] = np.std(np.abs(data[:,-1]))
                    else:
                        align[d,0] = np.mean(np.abs(data[:,index]))
                        align[d,1] = np.std(np.abs(data[:,index]))
                else:
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])
                noise[d] = np.sqrt(2*utils.readFromParams(dirSample, "dt")/utils.readFromDynParams(dirSample, "taup"))
        plt.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color=colorList[t], marker=markerList[t], markersize=8, label=labelList[t], capsize=3, fillstyle='none', lw=1)
    #plt.errorbar(noise[17], align[17,0], align[17,1], color='r', marker='*', markersize=8, capsize=3, fillstyle='none', lw=1)
    ax.legend(fontsize=12, loc='best')
    ax.set_xscale('log')
    if(index == 4):
        ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=16, rotation='horizontal')
    else:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2\\Delta t/\\tau_p}$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/compare-" + which + "VSnoise-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

############# Time-averaged Velocity Correlation in log-spaced time windows ############
def computeLogTimeVelocityAlignment(dirName, startBlock, maxPower, freqPower, plot=False):
    timeStep = utils.readFromParams(dirName, "dt")
    velCorr = []
    dirCorr = []
    stepList = []
    freqDecade = int(10**freqPower)
    decadeSpacing = 10
    spacingDecade = 1
    stepDecade = 10
    numBlocks = int(10**(maxPower-freqPower))
    for power in range(maxPower):
        for spacing in range(1,decadeSpacing):
            stepRange = np.arange(0,stepDecade,spacing*spacingDecade,dtype=int)
            stepVelCorr = []
            stepDirCorr = []
            numPairs = 0
            for multiple in range(startBlock, numBlocks):
                for i in range(stepRange.shape[0]-1):
                    if(utils.checkPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])):
                        vel1, vel2 = utils.readVelPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        dir1, dir2 = utils.readDirectorPair(dirName, multiple*freqDecade + stepRange[i], multiple*freqDecade + stepRange[i+1])
                        stepVelCorr.append(np.mean(np.sum(np.multiply(vel1,vel2), axis=1))/np.mean(np.linalg.norm(vel1, axis=1)**2))
                        stepDirCorr.append(np.mean(np.sum(np.multiply(dir1,dir2), axis=1)))
                        numPairs += 1
            if(numPairs > 0):
                stepList.append(spacing*spacingDecade)
                velCorr.append([np.mean(stepVelCorr, axis=0), np.std(stepVelCorr, axis=0)])
                dirCorr.append([np.mean(stepDirCorr, axis=0), np.std(stepDirCorr, axis=0)])
        stepDecade *= 10
        spacingDecade *= 10
    stepList = np.array(stepList)
    velCorr = np.array(velCorr).reshape((stepList.shape[0],2))
    velCorr = velCorr[np.argsort(stepList)]
    dirCorr = np.array(dirCorr).reshape((stepList.shape[0],2))
    dirCorr = dirCorr[np.argsort(stepList)]
    np.savetxt(dirName + os.sep + "alignment.dat", np.column_stack((stepList*timeStep, velCorr, dirCorr)))
    if(plot == 'plot'):
        uplot.plotCorrWithError(stepList*timeStep, velCorr[:,0], velCorr[:,1], ylabel="$C_{vv}(\\Delta t),$ $C_{nn}(\\Delta t)$", logx = True, color = 'k')
        uplot.plotCorrWithError(stepList*timeStep, dirCorr[:,0], dirCorr[:,1], ylabel="$C_{vv}(\\Delta t),$ $C_{nn}(\\Delta t)$", logx = True, color = 'r')
        #plt.show()
        plt.pause(0.5)

def plotVelTimeCorrVSNoise(dirName, figureName, which="vicsek"):
    fig, ax = plt.subplots(2, 1, figsize=(7.5,7), dpi = 120, sharex = True, constrained_layout=True)
    dirList = np.array(["1e-03", "1e-02", "5e-02", "1e-01", "5e-01", "1", "1e01", "1e02"])
    taup = dirList.astype(np.float64)
    colorList = cm.get_cmap('plasma', dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(which == "active"):
            dirSample = dirName + "tp" + dirList[d] + "/dynamics-log/"
        else:
            dirSample = dirName + "j1e02-tp" + dirList[d] + "/dynamics-log/"
        if not(os.path.exists(dirSample + "alignment.dat")):
            computeLogTimeVelocityAlignment(dirSample, 0, 7, 6)
        data = np.loadtxt(dirSample + "alignment.dat")
        ax[0].errorbar(data[:,0], data[:,1], data[:,2], color=colorList(d/dirList.shape[0]), marker='o', markersize=6, capsize=3, fillstyle='none', lw=1, label="$\\tau_p=$" + dirList[d])
        ax[1].errorbar(data[:,0], data[:,3], data[:,4], color=colorList(d/dirList.shape[0]), marker='o', markersize=6, capsize=3, fillstyle='none', lw=1)
    ax[1].set_xscale('log')
    ax[0].set_ylim(-0.12,1.08)
    #ax[0].legend(fontsize=10, loc='best', ncol=2)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar, ax=ax, pad=0, aspect=30)
    label = "$\\tau_p$"
    min = np.min(taup)
    max = np.max(taup)
    cb.set_ticks(np.linspace(0,1,4))
    cb.ax.tick_params(labelsize=12, length=0)
    ticks = np.geomspace(min, max, 4)
    ticklabels = []
    for i in range(ticks.shape[0]):
        ticklabels.append(np.format_float_scientific(ticks[i], 0))
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=16, labelpad=5, rotation='horizontal')
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].set_ylabel("$C_{vv}(t)$", fontsize=16)
    ax[1].set_ylabel("$C_{nn}(t)$", fontsize=16)
    ax[1].set_xlabel("$Elapsed$ $time,$ $t$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/alignCorrVSNoise-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotWallAngleDynamics(dirName, figureName, which='angle'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    # get wall dynamics from directories in dirName
    boxRadius = np.loadtxt(dirName + os.sep + "boxSize.dat")
    dirList, timeList = utils.getOrderedDirectories(dirName)
    angleDyn = np.zeros((dirList.shape[0], 4))
    for i in range(dirList.shape[0]):
        angleDyn[i,:3] = np.loadtxt(dirName + os.sep + dirList[i] + os.sep + "wallDynamics.dat").astype(np.float64)
        angleDyn[i,3] = 0.5 * angleDyn[i,1]**2 * boxRadius**2
    print("average omega: ", np.mean(angleDyn[:,1]), np.std(angleDyn[:,1]))
    print("average alpha: ", np.mean(angleDyn[:,2]), np.std(angleDyn[:,2]))
    if which == 'ekin':
        index = 3
        ylabel = '$Kinetic$ $energy,$ $\\omega^2 R^2 / 2$'
    elif which == 'alpha':
        index = 2
        ylabel = '$Angular$ $acceleration,$ $\\alpha$'
    elif which == 'omega':
        index = 1
        ylabel = '$Angular$ $velocity,$ $\\omega$'
    else:
        index = 0
        ylabel = '$Angle,$ $\\theta$'
    ax.plot(timeList, angleDyn[:,index], lw=1, color='k')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/wallDyn-" + which + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

    if(whichPlot == "align"):
        figureName = sys.argv[3]
        plotAlignment(dirName, figureName)

    elif(whichPlot == "aligninter"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        plotAlignmentVSInteraction(dirName, figureName, which, dynamics)

    elif(whichPlot == "alignbeta"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        plotAlignmentVSDamping(dirName, figureName, which, dynamics)

    elif(whichPlot == "comparetemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareAlignmentVSTemperature(dirName, figureName, which, dynamics)

    elif(whichPlot == "alignnoise"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        jvic = sys.argv[5]
        dynamics = sys.argv[6]
        plotAlignmentVSNoise(dirName, figureName, which, jvic, dynamics)

    elif(whichPlot == "comparealign"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareAlignmentVSNoise(dirName, figureName, which, dynamics)

    elif(whichPlot == "logvcorr"):
        startBlock = int(sys.argv[3])
        maxPower = int(sys.argv[4])
        freqPower = int(sys.argv[5])
        plot = sys.argv[6]
        computeLogTimeVelocityAlignment(dirName, startBlock, maxPower, freqPower, plot)

    elif(whichPlot == "vcorrnoise"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotVelTimeCorrVSNoise(dirName, figureName, which)

    elif(whichPlot == "walldyn"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotWallAngleDynamics(dirName, figureName, which)

    else:
        print("Please specify the type of plot you want")
