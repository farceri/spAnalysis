'''
Created by Francesco
7 November 2024
'''
#functions for soft particle packing visualization
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import sys
import os
import utils
import utilsPlot as uplot

def getIndexYlabel(which):
    if(which == "epot"):
        index = 2
        ylabel = "$Potential$ $energy,$ $U$"
    elif(which == "ekin"):
        index = 3
        ylabel = "$Kinetic$ $energy,$ $K$"
    elif(which == "prad"):
        index = 4
        ylabel = "$Radial$ $pressure,$ $P_r$"
    elif(which == "ptheta"):
        index = 5
        ylabel = "$Tangential$ $pressure,$ $P_\\phi$"
    elif(which == "pos"):
        index = 6
        ylabel = "$|\\underbar{\\phi_r}|$"
    elif(which == "vel"):
        index = 7
        ylabel = "$|\\underbar{\\phi_v}|$"
    elif(which == "velpos"):
        index = 8
        ylabel = "$|\\underbar{\\alpha}|$"
    elif(which == "corr"):
        index = -2
        ylabel = "$Velocity$ $correlation,$ $C_{vv}$"
    else:
        index = -1
        ylabel = "$|\\langle L \\rangle |$"
    return index, ylabel

########################## plot alignment in active systems ##########################
def plotAlignment(dirName, figureName, which='corr'):
    if(os.path.exists(dirName + "/energy.dat")):
        energy = np.loadtxt(dirName + os.sep + "energy.dat")
        print("potential energy:", np.mean(energy[:,2]), "+-", np.std(energy[:,2]))
        print("temperature:", np.mean(energy[:,3]), "+-", np.std(energy[:,3]))
        print("velocity alignment:", np.mean(energy[:,-2]), "+-", np.std(energy[:,-2]), "relative error:", np.std(energy[:,-2])/np.mean(energy[:,-2]))
        fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
        if(which == "pos"):
            ylabel = "$\\phi_r$"
            ax.plot(energy[:,0], energy[:,6], linewidth=1.5, color='k')
        elif(which == "vel"):
            ylabel = "$\\phi_v$"
            ax.plot(energy[:,0], energy[:,7], linewidth=1.5, color='k')
        elif(which == "velpos"):
            ylabel = "$\\phi_\\alpha$"
            ax.plot(energy[:,0], energy[:,8], linewidth=1.5, color='k')
        elif(which == "pos2"):
            ylabel = "$\\phi_{r,2}$"
            ax.plot(energy[:,0], energy[:,9], linewidth=1.5, color='k')
        else:
            ylabel = "$C_{vv}$"
            ax.plot(energy[:,0], energy[:,-2], linewidth=1.5, color='k')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylim(0.722, 1.022)
        ax.set_xlabel("$Simulation$ $step$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=16, rotation='horizontal', labelpad=20)
        plt.tight_layout()
        figureName = "/home/francesco/Pictures/soft/align-" + figureName
        fig.savefig(figureName + ".png", transparent=True, format = "png")
        plt.show()
    else:
        print("no energy.dat file was found in", dirName)

def compareAlignment(dirName, figureName, which='corr', dynamics='/', log=False):
    fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    dirList = np.array(["vicsek-force", "vicsek-vel"])
    colorList = ['b', 'g']
    labelList = ["$Force$", "$Velocity$"]
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d] + os.sep + "reflect/damping1e02/j1e03-tp1e01" + dynamics
        if(os.path.exists(dirSample + "/energy.dat")):
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            print("potential energy:", np.mean(energy[:,2]), "+-", np.std(energy[:,2]))
            print("temperature:", np.mean(energy[:,3]), "+-", np.std(energy[:,3]))
            print("velocity alignment:", np.mean(energy[:,-2]), "+-", np.std(energy[:,-2]), "relative error:", np.std(energy[:,-2])/np.mean(energy[:,-2]))
            if(which == "pos"):
                ylabel = "$\\phi_r$"
                ax.plot(energy[:,0], energy[:,6], linewidth=1.5, color=colorList[d], label=labelList[d])
            elif(which == "vel"):
                ylabel = "$\\phi_v$"
                ax.plot(energy[:,0], energy[:,7], linewidth=1.5, color=colorList[d], label=labelList[d])
            elif(which == "velpos"):
                ylabel = "$\\phi_\\alpha$"
                ax.plot(energy[:,0], energy[:,8], linewidth=1.5, color=colorList[d], label=labelList[d])
            elif(which == "pos2"):
                ylabel = "$\\phi_{r,2}$"
                ax.plot(energy[:,0], energy[:,9], linewidth=1.5, color=colorList[d], label=labelList[d])
            else:
                ylabel = "$C_{vv}$"
                ax.plot(energy[:,0], energy[:,-2], linewidth=1.5, color=colorList[d], label=labelList[d])
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_ylim(-0.022, 1.022)
    if log == 'log':
        ax.set_xscale('log')
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Simulation$ $step$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=16, rotation='horizontal', labelpad=10)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/compareAlign-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def computeAngleDistance(dirName, ratio=0., mask=True):
    dirList, timeList = utils.getOrderedDirectories(dirName)
    boxRadius = np.loadtxt(dirName + os.sep + "boxSize.dat")
    angleDistance = np.zeros((timeList.shape[0],2))
    for t in range(timeList.shape[0]):
        dirSample = dirName + os.sep + dirList[t]
        pos = np.loadtxt(dirSample + os.sep + "particlePos.dat")
        radial = np.linalg.norm(pos, axis=1)
        # Compute angles of each particle in polar coordinates
        angle = np.arctan2(pos[:,1], pos[:,0])
        # Filter angles based on proximity to boundary
        angle = angle[radial > (ratio*boxRadius)]
        # Weigh final distance by the fraction of particles near the boundary
        fraction = angle.shape[0] / pos.shape[0]
        # Compute all pairwise angle differences using broadcasting
        angle_matrix = angle[:, None] - angle[None, :]  # Shape (N, N)
        delta_matrix = np.abs(angle_matrix)
        delta_matrix = np.minimum(delta_matrix, 2 * np.pi - delta_matrix)

        # Mask out self-comparisons (delta = 0 on diagonal)
        np.fill_diagonal(delta_matrix, np.nan)

        # Mask elements >= pi/2
        if mask:
            delta_matrix[delta_matrix >= (np.pi / 2)] = np.nan
            norm = 2 * np.pi / 9
        else:
            norm = np.pi / 2

        # Compute mean for each row, ignoring NaNs
        valid_rows = ~np.isnan(delta_matrix).all(axis=1)  # Boolean mask for valid rows
        distList = np.nanmean(delta_matrix[valid_rows], axis=1) / norm
        distList *= (fraction / (1 - ratio**2))
        angleDistance[t,0] = np.mean(distList)
        angleDistance[t,1] = np.std(distList)/np.sqrt(distList.shape[0])
    np.savetxt(dirName + os.sep + "angleDistance.dat", np.column_stack((timeList, angleDistance)))

def plotAngleDistance(dirName, figureName, log=False):
    computeAngleDistance(dirName)
    distance = np.loadtxt(dirName + os.sep + "angleDistance.dat")
    fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    ax.errorbar(distance[:,0], distance[:,1], distance[:,2], linewidth=1, color='k', capsize=3, marker='o', fillstyle='none', markersize=8)
    if log == 'log': ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $step$", fontsize=14)
    ax.set_ylabel("$\\langle \\Delta \\phi_r \\rangle$", fontsize=16, rotation='horizontal', labelpad=10)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/deltaAngle-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareAngleDistance(dirName, figureName, dynamics='/', log=False):
    fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    dirList = np.array(["vicsek-force", "vicsek-vel"])
    colorList = ['b', 'g']
    labelList = ["$Force$", "$Velocity$"]
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d] + os.sep + "reflect/damping1e02/j1e03-tp1e01" + dynamics
        if not(os.path.exists(dirSample + "/angleDistance.dat")):
            computeAngleDistance(dirSample)
        distance = np.loadtxt(dirSample + os.sep + "angleDistance.dat")
        ax.errorbar(distance[:,0], distance[:,1], distance[:,2], linewidth=1, color=colorList[d], label=labelList[d], capsize=3, marker='o', fillstyle='none', markersize=8)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_ylim(-0.022, 1.022)
    if log == 'log':
        ax.set_xscale('log')
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Simulation$ $step$", fontsize=14)
    ax.set_ylabel("$\\langle \\Delta \\phi_r \\rangle$", fontsize=16, rotation='horizontal', labelpad=20)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/compareAngle-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

# Truncate colormap between 1/(N+2) and N/(N+2)
def truncated_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return cm.colors.LinearSegmentedColormap.from_list(f'trunc({cmap.name},{minval:.2f},{maxval:.2f})', cmap(np.linspace(minval, maxval, n)))

def darken_cmap(cmap, factor=0.8):
    """Darkens a colormap by multiplying RGB values by `factor` (0 < factor < 1)."""
    colors = cmap(np.linspace(0, 1, 256))
    darkened_colors = colors.copy()
    darkened_colors[:, :3] *= factor  # Only modify RGB, leave alpha unchanged
    return mcolors.LinearSegmentedColormap.from_list(f"{cmap.name}_dark", darkened_colors)

def compareAngleDistanceVSInteraction(dirName, figureName, ratio=0., dynamics='/', log=False, compute=False):
    fig, ax = plt.subplots(figsize=(7.5,5), dpi = 120)
    dirList = np.array(["1e-01", "3e-01", "4e-01", "1", "7", "1e02", "1e03", "1e04"])
    N = dirList.shape[0]
    full_cmap = cm.get_cmap('jet')
    trunc_cmap = truncated_colormap(full_cmap, minval=1/(N+2), maxval=(N+1)/(N+2), n=N)
    colorList = darken_cmap(trunc_cmap, factor=0.95)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + "j" + dirList[d] + "-tp1e03" + dynamics
        if compute == 'compute':
            computeAngleDistance(dirSample, ratio)
        else:
            if not(os.path.exists(dirSample + "/angleDistance.dat")):
                computeAngleDistance(dirSample)
        distance = np.loadtxt(dirSample + os.sep + "angleDistance.dat")
        ax.errorbar(distance[:,0], distance[:,1], distance[:,2], linewidth=1, color=colorList(d/(N-1)), label="$J=$" + dirList[d], capsize=3, marker='o', fillstyle='none', markersize=8)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_ylim(-0.022, 1.022)
    if log == 'log':
        ax.set_xscale('log')
    #ax.legend(fontsize=11, loc='best', ncols=2)
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.)
    cbar = fig.colorbar(colorBar, cax)
    cbar.set_label('$J$', rotation='horizontal', fontsize=16, labelpad=5)
    cbar.ax.tick_params(labelsize=14, length=0)
    cbar.set_ticks(np.linspace(0,1,6))
    cbar.set_ticklabels(['$10^{-1}$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])
    ax.set_xlabel("$Simulation$ $step$", fontsize=14)
    ax.set_ylabel("$\\frac{\\langle \\Delta \\phi_r \\rangle_{\\pi/2}}{\\langle \\Delta \\phi_r \\rangle_{\\pi/2}^U} \\frac{\\rho_C}{\\rho}$", fontsize=20, rotation='horizontal', labelpad=40)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/angleInter-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotAlignmentVSInteraction(dirName, figureName, which, taup="0", dynamics="/"):
    dirList = np.array(["1e-03", "1e-02", "3e-02", "1e-01", "3e-01", "4e-01", "5e-01", "7e-01", "1", "1.5", "2", "3", "5", "7",
                        "1e01", "3e01", "1e02", "3e02", "1e03", "1e04", "1e05", "1e06", "1e07"])
    jvic = np.zeros(dirList.shape[0])
    align = np.zeros((dirList.shape[0], 2))
    index, ylabel = getIndexYlabel(which)
    fig, ax = plt.subplots(figsize=(5,4.5), dpi = 120)
    for d in range(dirList.shape[0]):
        dirSample = dirName + "j" + dirList[d] + "-tp" + taup + dynamics
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            if(index == -1):
                align[d,0] = np.mean(np.abs(data[:,index]))
                align[d,1] = np.std(np.abs(data[:,index]))
            else:
                align[d,0] = np.mean(data[:,index])
                align[d,1] = np.std(data[:,index])
            jvic[d] = utils.readFromDynParams(dirSample, "Jvicsek")
            #print(dirList[d], 1/jvic[d])
            if(d == 0 and index == -2):
                noisetime = utils.readFromDynParams(dirSample, "taup")
                ax.plot(np.ones(100)*noisetime, np.linspace(-0.3,1.3,100), ls='dotted', color='k', lw=0.8)
    ax.errorbar(1/jvic[jvic!=0], align[jvic!=0,0], align[jvic!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
    ax.set_xscale('log')
    if(index == -1):
        ax.set_yscale('log')
    if(index == 5):
        ax.set_ylim(-0.057, 1.112)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/alignVSinter-" + which + "-tp" + taup + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareAlignmentVSInteraction(dirName, figureName, which, dynamics="/"):
    noiseList = np.array(["1e-02", "1e-01", "1", "1e01", "1e02", "0"])
    labelList = np.array(["$\\tau_p = 10^{-2}$", "$\\tau_p = 10^{-1}$", "$\\tau_p = 10^0$", "$\\tau_p = 10^1$", "$\\tau_p = 10^2$", "$\\tau_p \\rightarrow \\infty$"])
    dirList = np.array(["1e-03", "1e-02", "3e-02", "1e-01", "3e-01", "4e-01", "5e-01", "7e-01", "1", "1.5", "2", "3", "5", "7",
                        "1e01", "3e01", "1e02", "3e02", "1e03", "1e04", "1e05", "1e06", "1e07"])
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    index, ylabel = getIndexYlabel(which)
    colorList = [[1,0.5,0], 'r', 'g', 'c', 'b', 'k']
    markerList = ['v', 'd', 's', 'D', '^', 'o']
    for i in range(noiseList.shape[0]):
        jvic = np.zeros(dirList.shape[0])
        align = np.zeros((dirList.shape[0], 2))
        noisetime = 0
        for d in range(dirList.shape[0]):
            dirSample = dirName + "j" + dirList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                if(index == -1):
                    align[d,0] = np.mean(np.abs(data[:,index]))
                    align[d,1] = np.std(np.abs(data[:,index]))
                else:
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])
                jvic[d] = utils.readFromDynParams(dirSample, "Jvicsek")
                if(noisetime == 0 and index == -2):
                    noisetime = utils.readFromDynParams(dirSample, "taup")
                    plt.plot(np.ones(100)*noisetime, np.linspace(-0.3,1.3,100), ls='dotted', color=colorList[i], lw=0.8)
        plt.errorbar(1/jvic[jvic!=0], align[jvic!=0,0], align[jvic!=0,1], color=colorList[i], marker=markerList[i], markersize=8, capsize=3, fillstyle='none', lw=1, label=labelList[i])
    ax.legend(fontsize=11, loc='best')
    ax.set_xscale('log')
    if(index == -1):
        ax.set_yscale('log')
    if(index == 5):
        ax.set_ylim(-0.057, 1.112)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/compare-" + which + "VSinter-" + figureName
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
                if(index == -1):
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

def plotAlignmentVSNoise(dirName, figureName, which, jvic="1e02", dynamics="/"):
    dirList = np.array(["1e-04", "2e-04", "3e-04", "5e-04", "1e-03", "2e-03", "3e-03", "5e-03", "1e-02", "1.5e-02", "2e-02", "2.5e-02", "3e-02", "4e-02",
                        "5e-02", "7e-02", "1e-01", "1.5e-01", "2e-01", "3e-01", "5e-01", "1", "2", "3", "5", "1e01", "2e01", "3e01", "5e01", "1e02",
                        "2e02", "3e02", "5e02", "1e03", "2e03", "3e03", "5e03", "1e04", "2e04", "3e04", "5e04", "1e05", "2e05", "3e05", "5e05",
                        "1e06", "1e07", "1e08", "1e09"])
    noise = np.zeros(dirList.shape[0])
    align = np.zeros((dirList.shape[0], 2))
    index, ylabel = getIndexYlabel(which)
    fig, ax = plt.subplots(figsize=(5,4.5), dpi = 120)
    for d in range(dirList.shape[0]):
        if(jvic == "active"):
            dirSample = dirName + "tp" + dirList[d] + dynamics
        else:
            dirSample = dirName + "j" + jvic + "-tp" + dirList[d] + dynamics
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            if(index == -1):
                align[d,0] = np.mean(np.abs(data[:,index]))
                align[d,1] = np.std(np.abs(data[:,index]))
            else:
                align[d,0] = np.mean(data[:,index])
                align[d,1] = np.std(data[:,index])
            #noise[d] = np.sqrt(2*utils.readFromParams(dirSample, "dt")/utils.readFromDynParams(dirSample, "taup"))
            noise[d] = utils.readFromDynParams(dirSample, "taup")
            if(d == 0 and index == -2):
                aligntime = 1/utils.readFromDynParams(dirSample, "Jvicsek")
                plt.plot(np.ones(100)*aligntime, np.linspace(-0.3,1.3,100), ls='dotted', color='k', lw=0.8)
    plt.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
    ax.set_xscale('log')
    if(index == -1):
        ax.set_yscale('log')
    if(index == 5):
        ax.set_ylim(-0.057, 1.112)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2\\Delta t/\\tau_p}$", fontsize=16)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    if(jvic == "active"):
        figureName = "/home/francesco/Pictures/soft/alignVSnoise-" + which + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/alignVSnoise-" + which + "j" + jvic + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareAlignmentVSNoise(dirName, figureName, which, dynamics="/"):
    interList = np.array(["1e-01", "4e-01", "1", "7", "3e01", "1e03"])
    #labelList = np.array(["$J = 0.1$", "$J = 0.4$", "$J = 1$", "$J = 7$", "$J = 30$", "$J = 10^3$"])
    labelList = np.array(["$\\tau_K = 7.1 \\times 10^1$", "$\\tau_K = 1.8 \\times 10^1$", "$\\tau_K = 7.1 \\times 10^0$", "$\\tau_K = 10^0$", "$\\tau_K = 2.3 \\times 10^{-1}$", "$\\tau_K = 7.1 \\times 10^{-3}$"])
    colorList = [[1,0.5,0], 'r', 'g', 'c', 'b', 'k']
    markerList = ['v', 'd', 's', 'D', '^', 'o']
    dirList = np.array(["1e-04", "2e-04", "3e-04", "5e-04", "1e-03", "2e-03", "3e-03", "5e-03", "1e-02", "1.5e-02", "2e-02", "2.5e-02", "3e-02", "4e-02",
                        "5e-02", "7e-02", "1e-01", "1.5e-01", "2e-01", "3e-01", "5e-01", "1", "2", "3", "5", "1e01", "2e01", "3e01", "5e01", "1e02",
                        "2e02", "3e02", "5e02", "1e03", "2e03", "3e03", "5e03", "1e04", "2e04", "3e04", "5e04", "1e05", "2e05", "3e05", "5e05",
                        "1e06", "1e07", "1e08", "1e09"])
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    index, ylabel = getIndexYlabel(which)
    for t in range(interList.shape[0]):
        noise = np.zeros(dirList.shape[0])
        align = np.zeros((dirList.shape[0], 2))
        aligntime = 0
        for d in range(dirList.shape[0]):
            dirSample = dirName + "j" + interList[t] + "-tp" + dirList[d] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                if(index == -1):
                    align[d,0] = np.mean(np.abs(data[:,index]))
                    align[d,1] = np.std(np.abs(data[:,index]))
                else:
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])
                #noise[d] = np.sqrt(2*utils.readFromParams(dirSample, "dt")/utils.readFromDynParams(dirSample, "taup"))
                noise[d] = utils.readFromDynParams(dirSample, "taup")
                if(aligntime == 0 and index == -2):
                    aligntime = 1/utils.readFromDynParams(dirSample, "Jvicsek")
                    #print(interList[t], aligntime)
                    plt.plot(np.ones(100)*aligntime, np.linspace(-0.3,1.3,100), ls='dotted', color=colorList[t], lw=0.8)
                    #labelList[t] = "$\\tau_K =$" + str(np.format_float_scientific(aligntime,1))
        plt.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color=colorList[t], marker=markerList[t], markersize=8, label=labelList[t], capsize=3, fillstyle='none', lw=1)
    ax.legend(fontsize=12, loc='best')
    ax.set_xscale('log')
    if(index == -1):
        ax.set_yscale('log')
    if(index == 5):
        ax.set_ylim(-0.057, 1.112)
    if(index == 4):
        ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=16, rotation='horizontal')
    else:
        ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2\\Delta t/\\tau_p}$", fontsize=16)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/compare-" + which + "VSnoise-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def phaseDiagramNoiseAlignment(dirName, figureName, dynamics="/", which="vcorr", interpolate=False):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    aligntime = np.array([])
    noisetime = np.array([])
    corr = np.array([])
    if(which == "pos"):
        cbar_label = "$\\phi_r$"
        index = 6
    elif(which == "vel"):
        cbar_label = "$\\phi_v$"
        index = 7
    elif(which == "velpos"):
        cbar_label = "$\\phi_\\alpha$"
        index = 8
    else:
        cbar_label = "$C_{vv}$"
        index = -2
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08", "0"])
    noiseDirList = np.array(["1e-01", "4e-01", "1", "3", "7", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    for i in range(noiseList.shape[0]):
        for d in range(noiseDirList.shape[0]):
            dirSample = dirName + "j" + noiseDirList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                corr = np.append(corr, np.mean(data[:,index]))
                aligntime = np.append(aligntime, 1/utils.readFromDynParams(dirSample, "Jvicsek"))
                #print(noiseList[i], noiseDirList[d], aligntime[-1])
                tp = utils.readFromDynParams(dirSample, "taup")
                if(tp == 0):
                    tp = 1e09
                noisetime = np.append(noisetime, tp)
    if(interpolate == 'interpolate'):
        #grid_tp, grid_tk = np.meshgrid(noisetime, aligntime)
        #grid_corr = griddata((noisetime, aligntime), corr, (grid_tp, grid_tk), method='cubic')
        # Add a small random noise to avoid numerical issues
        noisetime += np.random.uniform(0, 0.01, size=noisetime.shape)
        aligntime += np.random.uniform(0, 0.01, size=aligntime.shape)
        # Convert to log-space for interpolation
        log_noisetime = np.log10(noisetime)
        log_aligntime = np.log10(aligntime)
        # Define a log-spaced grid
        log_tp_lin = np.linspace(log_noisetime.min(), log_noisetime.max(), 100)
        log_tk_lin = np.linspace(log_aligntime.min(), log_aligntime.max(), 100)
        grid_tp, grid_tk = np.meshgrid(log_tp_lin, log_tk_lin)
        grid_corr = griddata((log_noisetime, log_aligntime), corr, (grid_tp, grid_tk), method='linear')
        # Convert grid back to linear scale for plotting
        grid_tp_lin = 10**grid_tp
        grid_tk_lin = 10**grid_tk
        contour = plt.contourf(grid_tp_lin, grid_tk_lin, grid_corr, levels=20, cmap='plasma')
        cbar = plt.colorbar(contour, ax=ax, pad=0, aspect=20)
    else:
        vmin = np.min(corr)
        vmax = np.max(corr)
        sc = plt.scatter(noisetime, aligntime, c=corr, cmap='plasma', s=200, edgecolors='k', marker='s', linewidths=0.5, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax, pad=0, aspect=20)
    cbar.set_label(cbar_label, rotation='horizontal', fontsize=16, labelpad=20)
    cbar.ax.tick_params(labelsize=14, length=0)
    cbar.set_ticks(np.linspace(np.min(corr),np.max(corr),6))
    # Set log scales for proper visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    ax.set_ylabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/phaseDiagram-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def phaseDiagramAngleDistance(dirName, figureName, dynamics="/", interpolate=False):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    aligntime = np.array([])
    noisetime = np.array([])
    angle_delta = np.array([])
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08", "0"])
    noiseDirList = np.array(["1e-01", "4e-01", "1", "3", "7", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    for i in range(noiseList.shape[0]):
        for d in range(noiseDirList.shape[0]):
            dirSample = dirName + "j" + noiseDirList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                if not(os.path.exists(dirSample + "/angleDistance.dat")):
                    computeAngleDistance(dirSample)
                distance = np.loadtxt(dirSample + os.sep + "angleDistance.dat")
                angle_delta = np.append(angle_delta, np.mean(distance[:,1]))
                aligntime = np.append(aligntime, 1/utils.readFromDynParams(dirSample, "Jvicsek"))
                #print(noiseList[i], noiseDirList[d], angle_delta[-1])
                tp = utils.readFromDynParams(dirSample, "taup")
                if(tp == 0):
                    tp = 1e09
                noisetime = np.append(noisetime, tp)
    if(interpolate == 'interpolate'):
        #grid_tp, grid_tk = np.meshgrid(noisetime, aligntime)
        #grid_corr = griddata((noisetime, aligntime), corr, (grid_tp, grid_tk), method='cubic')
        # Add a small random noise to avoid numerical issues
        noisetime += np.random.uniform(0, 0.01, size=noisetime.shape)
        aligntime += np.random.uniform(0, 0.01, size=aligntime.shape)
        # Convert to log-space for interpolation
        log_noisetime = np.log10(noisetime)
        log_aligntime = np.log10(aligntime)
        # Define a log-spaced grid
        log_tp_lin = np.linspace(log_noisetime.min(), log_noisetime.max(), 100)
        log_tk_lin = np.linspace(log_aligntime.min(), log_aligntime.max(), 100)
        grid_tp, grid_tk = np.meshgrid(log_tp_lin, log_tk_lin)
        grid_corr = griddata((log_noisetime, log_aligntime), angle_delta, (grid_tp, grid_tk), method='linear')
        # Convert grid back to linear scale for plotting
        grid_tp_lin = 10**grid_tp
        grid_tk_lin = 10**grid_tk
        vmin = np.nanmin(grid_corr)
        vmax = np.nanmax(grid_corr)
        contour = plt.contourf(grid_tp_lin, grid_tk_lin, grid_corr, levels=20, cmap='rainbow', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contour, ax=ax, pad=0, aspect=20)
    else:
        vmin = np.min(angle_delta)
        vmax = np.max(angle_delta)
        print(vmax, vmin)
        full_cmap = cm.get_cmap('jet')
        N = angle_delta.shape[0]
        trunc_cmap = truncated_colormap(full_cmap, minval=vmin, maxval=vmax-0.25, n=N)
        trunc_cmap = darken_cmap(trunc_cmap, factor=0.95)
        sc = plt.scatter(noisetime, aligntime, c=angle_delta, cmap=trunc_cmap, s=200, edgecolors='k', marker='s', linewidths=0.5, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax, pad=0, aspect=20)
    # Set colorbar ticks and labels
    cbar.set_label("$\\langle \\tilde{\\Delta \\phi_r} \\rangle$", rotation='horizontal', fontsize=16, labelpad=30)
    min = np.min(angle_delta)
    max = np.max(angle_delta)
    cbar.set_ticks(np.linspace(min,max,5))
    cbar.set_ticklabels([np.format_float_positional(min,2), np.format_float_positional(max/4,2), 
                         np.format_float_positional(max/2,2), np.format_float_positional(3*max/4,2), np.format_float_positional(max,2)])
    cbar.ax.tick_params(labelsize=14, size=0)
    # Set log scales for proper visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    ax.set_ylabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/phaseDiagramAngle-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareAlignmentVSTemperature(dirName, figureName, which, dynamics="/", scale=0):
    typeList = np.array(["langevin1e-01/", "langevin3e-01/", "langevin1/", "langevin2/", "langevin1e01/", "langevin3e01/", "langevin1e02/"])
    beta = ['1e-01', '3e-01', '1', '2', '1e01', '3e01', '1e02']
    dirList = np.array(["4e-04", "1e-03", "2e-03", "4e-03", "1e-02", "2e-02", "4e-02", "1e-01", "2e-01", "4e-01", "1"])
    markerList = ['o', 's', 'v', '^', 'd', 'D', '*']
    colorList = ['k', 'r', 'g', 'b', 'c', [1,0.5,0], [0.5,0,1]]
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
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])/np.sqrt(data.shape[0])
                else:
                    align[d,0] = np.mean(data[:,index])
                    align[d,1] = np.std(data[:,index])
                if(scale == 'scale'):
                    noise[d] = np.sqrt(2 * float(dirList[d]) / utils.readFromDynParams(dirSample, "damping"))
                else:
                    if(which == 'ekin' or which == 'epot'):
                        noise[d] = float(dirList[d])
                    else:
                        noise[d] = np.sqrt(2 * float(dirList[d]) * utils.readFromDynParams(dirSample, "damping"))
        ax.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color=colorList[t], marker=markerList[t], markersize=8, capsize=3, fillstyle='none', lw=1, label="$\\gamma =$" + beta[t])
    ax.legend(fontsize=11, loc='best')
    ax.set_xscale('log')
    #if(which == 'corr'):
        #ax.set_xlim(0.008,)
        #ax.set_ylim(-0.02,1.08)
    ax.tick_params(axis='both', labelsize=14)
    if(scale == 'scale'):
        ax.set_xlabel("$Noise$ $over$ $friction,$ $\\sqrt{2 \\gamma k_B T} / \\gamma \\sigma$", fontsize=16)
    else:
        if(which == 'ekin' or which == 'epot'):
            ax.set_xlabel("$Bath$ $temperature,$ $T$", fontsize=16)
        else:
            ax.set_xlabel("$Noise,$ $\\sqrt{2 \\gamma k_B T}$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    if(scale == 'scale'):
        figureName = "/home/francesco/Pictures/soft/scaled-" + which + "VSTemp-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/" + which + "VSTemp-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareRingnessVSTemperature(dirName, figureName, which, dynamics="/dynamics-log/"):
    typeList = np.array(["langevin1e-01/", "langevin3e-01/", "langevin1/", "langevin2/", "langevin1e01/", "langevin3e01/", "langevin1e02/"])
    beta = ['1e-01', '3e-01', '1', '2', '1e01', '3e01', '1e02']
    dirList = np.array(["4e-04", "1e-03", "2e-03", "4e-03", "1e-02", "2e-02", "4e-02", "1e-01", "2e-01", "4e-01", "1"])
    markerList = ['o', 's', 'v', '^', 'd', 'D', '*']
    colorList = ['k', 'r', 'g', 'b', 'c', [1,0.5,0], [0.5,0,1]]
    fig, ax = plt.subplots(figsize=(7,6), dpi = 120)
    for t in range(typeList.shape[0]):
        noise = np.zeros(dirList.shape[0])
        ring = np.zeros((dirList.shape[0], 2))
        for d in range(dirList.shape[0]):
            dirSample = dirName + typeList[t] + "T" + dirList[d] + dynamics
            if(os.path.exists(dirSample)):
                if which == 'ring':
                    ring[d] = utils.computeRingness(dirSample)
                elif which == 'spread':
                    ring[d] = utils.computeSpreadness(dirSample, 50, 5)
                else:
                    ring[d] = utils.computeClusterRingness(dirSample, 10)
                noise[d] = np.sqrt(2 * utils.readFromDynParams(dirSample, "damping") * float(dirList[d]))
        ax.errorbar(noise[noise!=0], ring[noise!=0,0], ring[noise!=0,1], color=colorList[t], marker=markerList[t], markersize=8, capsize=3, fillstyle='none', lw=1, label="$\\gamma =$" + beta[t])
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2 \\gamma k_B T}$", fontsize=16)
    if which == 'ring':
        ax.plot(np.linspace(np.min(noise), np.max(noise), 100), np.ones(100), ls='--', lw=0.5, color='k')
        ax.set_ylabel("$Uniformity,$ $\\sigma_\\theta \\sqrt{3}/\\pi$", fontsize=16)
    elif which == 'spread':
        ax.set_ylabel("$Occupied$ $boundary$ $fraction$", fontsize=16)
    else:
        ax.set_ylabel("$\\frac{\\langle \\sigma_\\theta^C \\rangle}{\\pi / \\sqrt{3}}$", rotation='horizontal', fontsize=24, labelpad=15)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/ringVSTemp-" + which + "-" + figureName
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

def plotWallAngleDynamics(dirName, figureName, which='omega'):
    typeList = np.array(["langevin1e-02/", "langevin1e-01/", "langevin1/", "langevin1e01/", "langevin1e02/"])
    damping = np.array(['1e-02', '1e-01', '1', '1e01', '1e02'])
    colorList = cm.get_cmap('plasma', typeList.shape[0]+1)
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    omega = np.zeros((typeList.shape[0],2))
    alpha = np.zeros((typeList.shape[0],2))
    ekin = np.zeros((typeList.shape[0],2))
    noise = np.zeros(typeList.shape[0])
    for t in range(typeList.shape[0]):
        dirSample = dirName + os.sep + typeList[t] + os.sep + "tp1e02"
        # get wall dynamics from directories in dirName
        boxRadius = np.loadtxt(dirSample + os.sep + "boxSize.dat")
        dirList, timeList = utils.getOrderedDirectories(dirSample)
        angleDyn = np.zeros((dirList.shape[0], 3))
        for i in range(dirList.shape[0]):
            angleDyn[i] = np.loadtxt(dirSample + os.sep + dirList[i] + os.sep + "wallDynamics.dat").astype(np.float64)
        omega[t,0] = np.mean(np.abs(angleDyn[:,1]))
        omega[t,1] = np.std(np.abs(angleDyn[:,1]))
        alpha[t,0] = np.mean(np.abs(angleDyn[:,2]))
        alpha[t,1] = np.std(np.abs(angleDyn[:,2]))
        ekin[t,0] = np.mean(0.5*angleDyn[:,1]**2*boxRadius**2*utils.readFromWallParams(dirSample, "numWall"))
        ekin[t,1] = np.std(0.5*angleDyn[:,1]**2*boxRadius**2*utils.readFromWallParams(dirSample, "numWall"))
        noise[t] = utils.readFromDynParams(dirSample, "damping")
        ax.plot(timeList, angleDyn[:,0], lw=1, color=colorList((typeList.shape[0]-t-1)/typeList.shape[0]), label="$\\gamma=$" + damping[t])
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel('$Angle,$ $\\theta$', fontsize=16)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/wall-angle-" + figureName
    # second plot
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    if which == 'ekin':
        mean = ekin[:,0]
        error = ekin[:,1]
        ylabel = '$Kinetic$ $energy,$ $N_w \\omega^2 R^2 / 2$'
    elif which == 'alpha':
        mean = alpha[:,0]
        error = alpha[:,1]
        ylabel = '$|\\alpha|$'
    else:
        mean = omega[:,0]
        error = omega[:,1]
        ylabel = '$|\\omega|$'
    ax.errorbar(noise, mean, error, lw=1, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel("$Friction,$ $\\gamma$", fontsize=16)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/wall-" + which + "-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

    if(whichPlot == "align"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotAlignment(dirName, figureName, which)
    
    elif(whichPlot == "compare"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        log = sys.argv[6]
        compareAlignment(dirName, figureName, which, dynamics, log)

    elif(whichPlot == "angle"):
        figureName = sys.argv[3]
        log = sys.argv[4]
        plotAngleDistance(dirName, figureName, log)

    elif(whichPlot == "compareangle"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        log = sys.argv[5]
        compareAngleDistance(dirName, figureName, dynamics, log)

    elif(whichPlot == "angleinter"):
        figureName = sys.argv[3]
        ratio = float(sys.argv[4])
        dynamics = sys.argv[5]
        log = sys.argv[6]
        compute = sys.argv[7]
        compareAngleDistanceVSInteraction(dirName, figureName, ratio, dynamics, log, compute)

    elif(whichPlot == "aligninter"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        taup = sys.argv[5]
        dynamics = sys.argv[6]
        plotAlignmentVSInteraction(dirName, figureName, which, taup, dynamics)

    elif(whichPlot == "compareinter"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareAlignmentVSInteraction(dirName, figureName, which, dynamics)

    elif(whichPlot == "alignbeta"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        plotAlignmentVSDamping(dirName, figureName, which, dynamics)

    elif(whichPlot == "alignnoise"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        jvic = sys.argv[5]
        dynamics = sys.argv[6]
        plotAlignmentVSNoise(dirName, figureName, which, jvic, dynamics)

    elif(whichPlot == "comparenoise"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareAlignmentVSNoise(dirName, figureName, which, dynamics)

    elif(whichPlot == "phasediagram"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        which = sys.argv[5]
        interpolate = sys.argv[6]
        phaseDiagramNoiseAlignment(dirName, figureName, dynamics, which, interpolate)

    elif(whichPlot == "pdangle"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        interpolate = sys.argv[5]
        phaseDiagramAngleDistance(dirName, figureName, dynamics, interpolate)

    elif(whichPlot == "comparetemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        scale = sys.argv[6]
        compareAlignmentVSTemperature(dirName, figureName, which, dynamics, scale)

    elif(whichPlot == "comparering"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareRingnessVSTemperature(dirName, figureName, which, dynamics)

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
