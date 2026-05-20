'''
Created by Francesco
7 November 2024
'''
#functions for soft particle packing visualization
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import CloughTocher2DInterpolator
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import sys
import os
import utils
import utilsPlot as uplot

def getIndexYlabel(which):
    if(which == "epot"):
        index = 2
        ylabel = "$\\frac{U}{N}$"
    elif(which == "ekin"):
        index = 3
        ylabel = "$\\frac{K}{N}$"
    elif(which == "prad"):
        index = 4
        ylabel = "$P_r$"
    elif(which == "pphi"):
        index = 5
        ylabel = "$| P_\\phi |$"
    elif(which == "pos"):
        index = 6
        ylabel = "$\\phi_r$"
    elif(which == "vel"):
        index = 7
        ylabel = "$\\phi_v$"
    elif(which == "velpos"):
        index = 8
        ylabel = "$\\phi_\\alpha$"
    elif(which == "moi"):
        index = -2
        ylabel = "$\\tilde{I}$"
    elif(which == "angmom"):
        index = -1
        ylabel = "$\\tilde{L}$"
    else:
        index = -3
        ylabel = "$C_{vv}$"
    return index, ylabel

########################## plot alignment in active systems ##########################
def plotEnergyFile(dirName, figureName, which='corr'):
    if(os.path.exists(dirName + "/energy.dat")):
        energy = np.loadtxt(dirName + os.sep + "energy.dat")
        print("potential energy:", np.mean(energy[:,2]), "+-", np.std(energy[:,2]))
        print("temperature:", np.mean(energy[:,3]), "+-", np.std(energy[:,3]))
        fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
        index, ylabel = getIndexYlabel(which)
        ax.plot(energy[::2,0], energy[::2,index], linewidth=1.2, color='k')
        ax.tick_params(axis='both', labelsize=14)
        #ax.set_ylim(0.722, 1.022)
        ax.set_xlabel("$Simulation$ $step$", fontsize=16)
        if index == 2 or index == 3: ax.set_ylabel(ylabel, fontsize=24, rotation='horizontal', labelpad=15)
        else: ax.set_ylabel(ylabel, fontsize=16, rotation='horizontal', labelpad=15)
        plt.tight_layout()
        figureName = "/home/francesco/Pictures/soft/align-" + figureName
        fig.savefig(figureName + ".png", transparent=True, format = "png")
        plt.show()
    else:
        print("no energy.dat file was found in", dirName)

def compareAlignment(dirName, figureName, which='corr', boundary1='reflect', boundary2='fixed', dynamics='/', log=False, p1=0, p2=3):
    if p1 == 0 and p2 == 4: fig, ax = plt.subplots(1,2, sharey=True, figsize=(13,6), dpi = 120)
    else: fig, ax = plt.subplots(1,2, sharey=True, figsize=(10,4), dpi = 120)
    dirList = np.array(["kuramoto-vel", "kuramoto-force"])
    labelList = ["$Velocity$", "$Force$"]
    linestyleList = ['solid', 'dashed']
    colorList = ['k', 'limegreen', 'blueviolet', 'orange']
    index, ylabel = getIndexYlabel(which)
    sampleList = np.array(["j3-tp1", "j1e-04-tp1e03", "j1e01-tp1e03", "j1e04-tp1e03"])
    sampleLabelList = np.array(["$J_K=3,$ $\\tau_p=1$", "$J_K=10^{-4},$ $\\tau_p=10^3$", "$J_K=10,$ $\\tau_p=10^3$", "$J_K=10^4,$ $\\tau_p=10^3$"])
    boxRadius = np.atleast_1d(np.loadtxt(dirName + os.sep + "boxSize.dat"))
    maxI = boxRadius**2
    f0 = float(utils.readFromDynParams(dirName + "kuramoto-vel/reflect/damping1e02/j3-tp1e03/", "f0"))
    gamma = float(utils.readFromDynParams(dirName + "kuramoto-vel/reflect/damping1e02/j3-tp1e03/", "damping"))
    maxL = boxRadius * (f0/gamma)
    for d in range(dirList.shape[0]):
        for s in range(p1,p2):
        #for s in range(sampleList.shape[0]):
            if s == 2: dirSample = dirName + dirList[d] + os.sep + boundary1 + "/damping1e02/j3-tp1e03" + dynamics
            else:
                dirSample = dirName + dirList[d] + os.sep + boundary1 + "/damping1e02/" + sampleList[s] + dynamics
            label = labelList[d] + " " + sampleLabelList[s]
            if(os.path.exists(dirSample + "/energy.dat")):
                energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                if which == 'moi':
                    energy[:,index] /= maxI
                elif which == 'angmom':
                    energy[:,index] = np.abs(energy[:,index]) / maxL
                if which == 'corr': # correct for large fluctuations of velocity correlation
                    energy = energy[energy[:,index]>-0.1,:]
                    energy = energy[energy[:,index]<1.1,:]
                ax[0].plot(energy[energy[:,index]!=0,0], energy[energy[:,index]!=0,index], linewidth=1.2, linestyle=linestyleList[d], color=colorList[s], label=label)
    for d in range(dirList.shape[0]):
        for s in range(p1,p2):
        #for s in range(sampleList.shape[0]):
            dirSample = dirName + dirList[d] + os.sep + boundary2 + "/damping1e02/" + sampleList[s] + dynamics
            label = labelList[d] + " " + sampleLabelList[s]
            if(os.path.exists(dirSample + "/energy.dat")):
                energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                if which == 'moi':
                    energy[:,index] /= maxI
                elif which == 'angmom':
                    energy[:,index] = np.abs(energy[:,index]) / maxL
                if which == 'corr': # correct for large fluctuations of velocity correlation
                    energy = energy[energy[:,index]>-0.1,:]
                    energy = energy[energy[:,index]<1.1,:]
                ax[1].plot(energy[energy[:,index]!=0,0], energy[energy[:,index]!=0,index], linewidth=1.2, linestyle=linestyleList[d], color=colorList[s], label=label)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    if boundary1 == "fixed": ax[0].set_title("$WCA$ $boundary$", fontsize=16, loc='left')
    else: ax[0].set_title("$Smooth$ $boundary$", fontsize=16, loc='left')
    if boundary2 == "ipl": ax[1].set_title("$IPL$ $boundary$", fontsize=16, loc='right')
    else: ax[1].set_title("$WCA$ $boundary$", fontsize=16, loc='right')
    if which == 'moi': ax[0].set_ylim(0.328, 1.022)
    elif which == 'angmom': ax[0].set_ylim(-0.022, 1.022)
    if log == 'log': 
        for i in range(2): ax[i].set_yscale('log')
    if p1 == 0 and p2 == 4: plt.legend(fontsize=10, ncols=2, labelcolor='linecolor', fancybox=True, shadow=True, bbox_to_anchor=(0.38, 1.31))
    else: plt.legend(fontsize=10, ncols=2, labelcolor='linecolor', fancybox=True, shadow=True, bbox_to_anchor=(0.55, 1.1))
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=16)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=16)
    if index == 2 or index == 3: ax[0].set_ylabel(ylabel, fontsize=24, rotation='horizontal', labelpad=15)
    else: ax[0].set_ylabel(ylabel, fontsize=16, rotation='horizontal', labelpad=15)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    figureName = "/home/francesco/Pictures/soft/compare-" + which + "-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def compareBoundary(dirName, figureName, which='corr', dynamics='/'):
    fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    dirList = np.array(["reflect", "fixed", "rough", "rigid"])
    colorList = ['k', 'b', 'g', 'r']
    labelList = ["$Reflect$", "$Smooth$", "$Rough$", "$Pinned$"]
    index, ylabel = getIndexYlabel(which)
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d] + dynamics
        if(os.path.exists(dirSample + "/energy.dat")):
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            print("potential energy:", np.mean(energy[:,2]), "+-", np.std(energy[:,2]))
            print("temperature:", np.mean(energy[:,3]), "+-", np.std(energy[:,3]))
            print("velocity alignment:", np.mean(energy[:,-2]), "+-", np.std(energy[:,-2]), "relative error:", np.std(energy[:,-2])/np.mean(energy[:,-2]))
            ax.plot(energy[:,0], energy[:,index], linewidth=1.2, color=colorList[d], label=labelList[d])
    ax.tick_params(axis='both', labelsize=14)
    space = 0.04*np.max(energy[:,0])
    ax.set_xlim(np.min(energy[:,0])-space, np.max(energy[:,0])+space)
    ax.plot(np.linspace(np.min(energy[:,0])-space, np.max(energy[:,0])+space, 100), np.zeros(100), ls='dotted', color='k', lw=0.8)
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Simulation$ $step$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/bound-" + which + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareBoundaryAlign(dirName, figureName, which='corr', dynamics='/'):
    fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True, sharex=True, dpi = 120)
    boxRadius = np.loadtxt(dirName + "j1e03-tp1e03/dynamics-vel/boxSize.dat") 
    #fig = plt.figure(figsize=(10, 4), dpi=120)
    #gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    #ax0 = fig.add_subplot(gs[0])
    #ax1 = fig.add_subplot(gs[1], sharey=ax0)
    #ax = [ax0, ax1]
    dirList = np.array(["1e-01", "3e-01", "1", "1.5", "2.5", "3", "5", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    colorList = cm.get_cmap('plasma')
    index, ylabel = getIndexYlabel(which)
    aligntime = np.zeros(dirList.shape[0])
    mean1 = np.zeros(dirList.shape[0])
    error1 = np.zeros(dirList.shape[0])
    mean2 = np.zeros(dirList.shape[0])
    error2 = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        aligntime[d] = 1/utils.readFromDynParams(dirName + "j" + dirList[d] + "-tp1e03", "Jvicsek")
        dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/rough" + dynamics
        if(os.path.exists(dirSample + "/energy.dat")):
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            ax[0].plot(energy[::2,0], energy[::2,index], linewidth=1.2, color=colorList(d/dirList.shape[0]))
            mean1[d] = np.mean(energy[:,index])
            error1[d] = np.std(energy[:,index])
        dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/rigid" + dynamics
        if(os.path.exists(dirSample + "/energy.dat")):
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            ax[1].plot(energy[::2,0], energy[::2,index], linewidth=1.2, color=colorList(d/dirList.shape[0]))
            mean2[d] = np.mean(energy[:,index])
            error2[d] = np.std(energy[:,index])
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].legend(["$Rough$"], fontsize=12, loc='best')
    ax[1].legend(["$Pinned$"], fontsize=12, loc='best')
    space = 0.04*np.max(energy[:,0])
    #ax[0].set_xlim(np.min(energy[:,0])-space, np.max(energy[:,0])+space)
    ax[0].plot(np.linspace(np.min(energy[:,0])-space, np.max(energy[:,0])+space, 100), np.zeros(100), ls='dotted', color='k', lw=0.8)
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.)
    #cax = fig.add_subplot(gs[2])
    cbar = fig.colorbar(colorBar, cax)
    cbar.set_label('$J_K$', rotation='horizontal', fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=14, length=0)
    cbar.set_ticks(np.linspace(0,1,3))
    cbar.set_ticklabels(['$1$', '$10^2$', '$10^4$'])
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=14)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=14)
    ax[0].set_ylabel(ylabel, fontsize=14)
    ax[1].tick_params(labelleft=True)  # Enables y-tick labels on the right plot
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/boundAlign-" + which + "-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    if which == 'pphi':
        fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    else:
        fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    if which == 'pphi' or which == 'prad':
        #mean1 /= p0
        #mean2 /= p0
        f0 = 2
        gamma = 1e02
        p0 = 0.5 * 1024 * (f0/gamma)**2 / (np.pi * boxRadius**2)
        print(p0)
    ax.errorbar(aligntime, np.abs(mean1), error1, lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Rough$')
    ax.plot(np.linspace(np.min(aligntime), np.max(aligntime), 100), np.zeros(100), ls='--', color='r')
    if which == 'pphi':
        axr = ax.twinx()
        axr.errorbar(aligntime, np.abs(mean2), error2, lw=1.2, color='b', marker='s', markersize=8, fillstyle='none', capsize=3, label='$Pinned$')
        axr.tick_params(axis='y', colors='b', labelsize=12)
        axr.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axr.plot(np.linspace(np.min(aligntime), np.max(aligntime), 100), np.zeros(100), ls='--')
        ax.set_ylim(-1.405,)
        #axr.legend(fontsize=12, loc='best')
    else:
        ax.errorbar(aligntime, np.abs(mean2), error2, lw=1.2, color='b', marker='s', markersize=8, fillstyle='none', capsize=3, label='$Pinned$')
    #ax.legend(fontsize=12, loc='best')
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=12)
    if(which == 'pphi'):
        ax.set_ylabel("$|P_\\phi|$", rotation='horizontal', labelpad=15, fontsize=16)
        #ax.set_ylabel("$\\alpha_\phi$", rotation='horizontal', labelpad=15, fontsize=16)
    else:
        if(which == 'prad'): ax.set_yscale('log')
        ax.set_ylabel(ylabel, rotation='horizontal', labelpad=20, fontsize=16)
    ax.set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=14)
    plt.tight_layout()
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/boundAlign-" + which + "-vstp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def computeAngleDistance(dirName, ratio=0., mask=True):
    dirList, timeList = utils.getOrderedDirectories(dirName)
    if(dirList.shape[0] != 0):
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
        #np.savetxt(dirName + os.sep + "angleDistance-ratio" + str(ratio) + "-nonorm.dat", np.column_stack((timeList, angleDistance)))

def plotAngleDistance(dirName, figureName, log=False):
    if not(os.path.exists(dirName + os.sep + "angleDistance.dat")):
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

def compareAngleDistance(dirName, figureName, dynamics='/', which='ratio', log=False):
    fig, ax = plt.subplots(figsize=(7,2.5), dpi = 120)
    if which == 'ratio':
        dirList = np.array(["0.2", "0.4", "0.6", "0.8", "0.9"])
        labelList = ["$r_c = 0.2$", "$r_c = 0.4$", "$r_c = 0.6$", "$r_c = 0.8$", "$r_c = 0.9$"]
        colorList = ['b', 'c', 'g', 'm', 'r', [1,0.5,0]]
    else:
        dirList = np.array(["vicsek-force", "vicsek-vel"])
        labelList = ["$Force$", "$Velocity$"]
        colorList = ['b', 'g']
    for d in range(dirList.shape[0]):
        if which == 'ratio':
            distance = np.loadtxt(dirName + dynamics + "angleDistance-ratio" + dirList[d] + "-nonorm.dat")
        else:
            dirSample = dirName + dirList[d] + os.sep + "reflect/damping1e02/j1e03-tp1e01" + dynamics
            if not(os.path.exists(dirSample + "angleDistance.dat")):
                computeAngleDistance(dirSample)
            distance = np.loadtxt(dirSample + "angleDistance.dat")
        ax.errorbar(distance[:,0], distance[:,1], distance[:,2], linewidth=1, color=colorList[d], label=labelList[d], capsize=3, marker='o', fillstyle='none', markersize=8)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(-0.022, 6.222)
    if log == 'log':
        ax.set_xscale('log')
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Simulation$ $step$", fontsize=12)
    ax.set_ylabel("$\\frac{\\langle \\Delta \\phi_r \\rangle_{\\pi/2}}{\\langle \\Delta \\phi_r \\rangle_{\\pi/2}^U}$", fontsize=20, rotation='horizontal', labelpad=30)
    #ax.set_ylabel("$\\frac{\\langle \\Delta \\phi_r \\rangle_{\\pi/2}}{\\langle \\Delta \\phi_r \\rangle_{\\pi/2}^U} \\frac{\\rho_C}{\\rho}$", fontsize=18, rotation='horizontal', labelpad=40)
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
            if not(os.path.exists(dirSample + "angleDistance.dat")):
                computeAngleDistance(dirSample)
        distance = np.loadtxt(dirSample + "angleDistance.dat")
        ax.errorbar(distance[:,0], distance[:,1], distance[:,2], linewidth=1, color=colorList(d/(N-1)), label="$J=$" + dirList[d], capsize=3, marker='o', fillstyle='none', markersize=8)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_ylim(-0.022, 1.022)
    ax.set_xlim(-3.5e06, 8.35e07)
    ax.plot(np.linspace(-3.5e06, 8.35e07, 100), np.ones(100)/(1-ratio**2), ls='dotted', color='k', lw=0.8)
    print("ratio:", ratio, "fraction:", 1/(1-ratio**2))
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
    #ax.set_ylabel("$\\frac{\\langle \\Delta \\phi_r \\rangle_{\\pi/2}}{\\langle \\Delta \\phi_r \\rangle_{\\pi/2}^U}$", fontsize=20, rotation='horizontal', labelpad=40)
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
                aligntime = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
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
                    aligntime = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
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

def compareAlignmentVSBoundary(dirName, figureName, which, dynamics="/"):
    boundaryList = np.array(["reflect", "fixed"])
    labelList = np.array(["$Elastic$", "$WCA$"])
    colorList = ['k', 'r']
    markerList = ['s', 'o']
    dirList = np.array(["1e-04", "2e-04", "3e-04", "5e-04", "1e-03", "2e-03", "3e-03", "5e-03", "1e-02", "1.5e-02", "2e-02", "2.5e-02", "3e-02", "4e-02",
                        "5e-02", "7e-02", "1e-01", "1.5e-01", "2e-01", "3e-01", "5e-01", "1", "2", "3", "5", "1e01", "2e01", "3e01", "5e01", "1e02",
                        "2e02", "3e02", "5e02", "1e03", "2e03", "3e03", "5e03", "1e04", "2e04", "3e04", "5e04", "1e05", "2e05", "3e05", "5e05",
                        "1e06", "1e07", "1e08", "1e09"])
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    index, ylabel = getIndexYlabel(which)
    for t in range(boundaryList.shape[0]):
        noise = np.zeros(dirList.shape[0])
        align = np.zeros((dirList.shape[0], 2))
        aligntime = 0
        for d in range(dirList.shape[0]):
            dirSample = dirName + boundaryList[t] + "/damping1e02/j1e03-tp" + dirList[d] + dynamics
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
                    aligntime = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
                    #print(interList[t], aligntime)
                    plt.plot(np.ones(100)*aligntime, np.linspace(-0.3,1.3,100), ls='dotted', color=colorList[t], lw=0.8)
                    #labelList[t] = "$\\tau_K =$" + str(np.format_float_scientific(aligntime,1))
        plt.errorbar(noise[noise!=0], align[noise!=0,0], align[noise!=0,1], color=colorList[t], marker=markerList[t], markersize=8, label=labelList[t], capsize=3, fillstyle='none', lw=1)
    ax.legend(fontsize=16, loc='best')
    ax.set_xscale('log')
    fontsize = 26
    if(index == 3):
        fontsize = 32
    elif(index == 5):
        ax.set_ylim(-0.057, 1.112)
    elif(index == 4 or index == -1):
        ax.set_yscale('log')
    ax.set_ylabel(ylabel, fontsize=fontsize, rotation='horizontal', labelpad=25)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Noise$ $magnitude,$ $\\sqrt{2\\Delta t/\\tau_p}$", fontsize=16)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/boundary-" + which + "VSnoise-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def phaseDiagramNoiseAlignment(dirName, figureName, dynamics="/", which="corr", interpolate=False):
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    aligntime = np.array([])
    noisetime = np.array([])
    corr = np.array([])
    index, cbar_label = getIndexYlabel(which)
    boxSize = np.atleast_1d(np.loadtxt(dirName + "j1e03-tp1e03/boxSize.dat"))
    maxI = boxSize**2
    f0 = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "f0"))
    gamma = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "damping"))
    maxL = boxSize * (f0/gamma)
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08", "0"])
    alignList = np.array(["3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    for i in range(noiseList.shape[0]):
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                if which == 'moi':
                    data[:,index] = data[:,index] / maxI
                elif which == 'angmom':
                    data[:,index] = np.abs(data[:,index]) / maxL
                elif which == 'pphi':
                    data[:,index] = np.abs(data[:,index])
                corr = np.append(corr, np.mean(data[:,index]))
                aligntime = np.append(aligntime, 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"]))
                print(alignList[d], noiseList[i], corr[-1])
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
        if vmin < 0: vmin = 0
        vmax = np.max(corr)
        sc = plt.scatter(noisetime, aligntime, c=corr, cmap='plasma', s=200, edgecolors='k', marker='s', linewidths=0.5, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax, pad=0, aspect=20)
    cbar.set_label(cbar_label, rotation='horizontal', fontsize=22, labelpad=20)
    cbar.ax.tick_params(labelsize=14, length=0)
    min = np.min(corr)
    max = np.max(corr)
    cbar.set_ticks(np.linspace(min,max,5))
    #cbar.set_ticks(np.linspace(0,1,5))
    if which == "vcorr" or which == "velpos":
        cbar.set_ticklabels(["$0.00$", "$0.25$", "$0.50$", "$0.75$", "$1.00$"])
    # Set log scales for proper visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    ax.set_ylabel("$Alignment$ $time,$ $\\tau_K$", fontsize=22)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/phaseDiagram-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def computeMaxClusterKuramoto(dirSample, eps=3, maxCluster=10, minFraction=0.3):
    dirList, timeList = utils.getOrderedDirectories(dirSample)
    numParticles = int(utils.readFromParams(dirSample, "numParticles"))
    eps *= 2 * np.mean(np.loadtxt(dirSample + "particleRad.dat"))
    clusterPhi_r = np.zeros((dirList.shape[0], 2))
    numCluster = np.empty(0)
    for d in range(dirList.shape[0]):
        dirFrame = dirSample + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        angles = np.arctan2(pos[:,1], pos[:,0])
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        if uniqueLabels.shape[0] < maxCluster:
            # Do not check for largest clusters if there are many clusters
            numCluster = np.append(numCluster, uniqueLabels.shape[0])
            # if there are few clusters, restrict computation of Kuramoto order parameter for the largest cluster
            maxLabel = -1
            numMaxLabel = 0
            for label in uniqueLabels:
                numLabel = labels[labels==label].shape[0]
                #print("label", label, "num particles in cluster", numLabel)
                if numLabel > numMaxLabel:
                    maxLabel = label
                    numMaxLabel = numLabel
            fraction = labels[labels==maxLabel].shape[0] / numParticles
            if fraction > minFraction and maxLabel != -1:
                #print("largest cluster:", maxLabel, numMaxLabel, "particles", fraction*numParticles)
                clusterIndices = np.where(labels == maxLabel)[0]
                angles = angles[clusterIndices]
        numParticles = angles.shape[0]
        # compute Kuramoto order parameter for the cluster
        sumReal = 0
        sumImag = 0
        for i in range(numParticles):
            sumReal += np.cos(angles[i])
            sumImag += np.sin(angles[i])
        phi_r = np.sqrt(sumReal**2 + sumImag**2) / numParticles
        clusterPhi_r[d,0] = timeList[d]
        clusterPhi_r[d,1] = phi_r
    if numCluster.shape[0] > 0:
        print("Average number of clusters:", np.mean(numCluster), "number of used particles:", numParticles)
    np.savetxt(dirSample + "/clusterKuramoto.dat", clusterPhi_r)

def computeClusterKuramoto(dirSample, eps=1.5, maxCluster=10):
    dirList, timeList = utils.getOrderedDirectories(dirSample)
    numParticles = int(utils.readFromParams(dirSample, "numParticles"))
    eps *= 2 * np.mean(np.loadtxt(dirSample + "particleRad.dat"))
    clusterPhi_r = np.zeros((dirList.shape[0], 2))
    numLabels = np.empty(0)
    for d in range(dirList.shape[0]):
        dirFrame = dirSample + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        angles = np.arctan2(pos[:,1], pos[:,0])
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        numLabels = np.append(numLabels, uniqueLabels.shape[0])
        phi_r = np.empty(0)
        for label in uniqueLabels:
            if label != -1:
                fraction = labels[labels==label].shape[0] / numParticles
                #print(dirList[d], "label", label, "num particles in cluster", fraction*numParticles)
                clusterAngles = angles[labels==label]
                numCluster = clusterAngles.shape[0]
                # compute Kuramoto order parameter for the cluster
                sumReal = 0
                sumImag = 0
                for i in range(numCluster):
                    sumReal += np.cos(clusterAngles[i])
                    sumImag += np.sin(clusterAngles[i])
                phi_r = np.append(phi_r, np.sqrt(sumReal**2 + sumImag**2) / numCluster)
        clusterPhi_r[d,0] = timeList[d]
        clusterPhi_r[d,1] = np.mean(phi_r)
        #print("Time:", timeList[d], "Num clusters:", uniqueLabels.shape[0], "Average phi_r:", clusterPhi_r[d,1])
        #print(dirFrame)
    if(numLabels.shape[0] != 0): print("Average number of clusters:", np.mean(numLabels))
    np.savetxt(dirSample + "/clusterKuramoto.dat", clusterPhi_r)

def plotClustersCOM(dirName, figureName, eps=1.5, maxCluster=12):
    fig, ax = plt.subplots(2, 1, figsize=(7,6), sharex=True,dpi = 120)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    boxRadius = np.loadtxt(dirName + "boxSize.dat")
    eps *= 2.5 * np.mean(np.loadtxt(dirName + "particleRad.dat"))
    cluster_com1 = np.empty((0,2))
    cluster_com2 = np.empty((0,2))
    times = np.array([])
    dt = float(utils.readFromParams(dirName, "dt"))
    for d in range(dirList.shape[0]):
        dirFrame = dirName + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        # transform positions to polar coordinates
        angles = np.arctan2(pos[:,1], pos[:,0])
        radial = np.sqrt(pos[:,0]**2 + pos[:,1]**2) / boxRadius
        polarPos = np.column_stack((radial, angles))
        if d == 0:
            prevPos = polarPos
        else:
            # account for periodicity in theta
            deltaTheta = polarPos[:,1] - prevPos[:,1]
            deltaTheta = (deltaTheta + np.pi) % (2 * np.pi) - np.pi
            polarPos[:,1] = prevPos[:,1] + deltaTheta
            prevPos = polarPos
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        #print("Time:", timeList[d], "Num clusters:", uniqueLabels.shape[0])
        if uniqueLabels.shape[0] < maxCluster:
            # find largest cluster and plot its COM
            numCluster = np.empty(0)
            clusterLabel = np.empty(0)
            for label in uniqueLabels:
                numCluster = np.append(numCluster, labels[labels==label].shape[0])
                clusterLabel = np.append(clusterLabel, label)
            maxIndex = np.argmax(numCluster)
            maxLabel = clusterLabel[maxIndex]
            clusterIndices1 = np.where(labels == maxLabel)[0]
            clusterPos1 = polarPos[clusterIndices1]
            com1 = np.mean(clusterPos1, axis=0)
            cluster_com1 = np.vstack((cluster_com1, com1))
            #print("largest cluster:", maxLabel, numCluster[maxIndex], "particles", com1)
            if uniqueLabels.shape[0] > 2:
                # find second largest cluster and plot its COM
                numCluster[maxIndex] = -1
                secondMaxIndex = np.argmax(numCluster)
                secondMaxLabel = clusterLabel[secondMaxIndex]
                clusterIndices2 = np.where(labels == secondMaxLabel)[0]
                clusterPos2 = polarPos[clusterIndices2]
                com2 = np.mean(clusterPos2, axis=0)
                cluster_com2 = np.vstack((cluster_com2, com2))
                #print("second largest cluster:", secondMaxLabel, numCluster[secondMaxIndex], "particles", com2)
            times = np.append(times, timeList[d] * dt)
    ax[0].plot(times, cluster_com1[:,0], color='b', lw=1)
    ax[1].plot(times, cluster_com1[:,1], color='b', lw=1, label="$\\theta_1$")
    #ax[1].plot(times[1:], (cluster_com1[1:,1]-cluster_com1[:-1,1])/(times[1:]-times[:-1]), color='b', lw=1, label="$Cluster$ $1$")
    if cluster_com2.shape[0] == times.shape[0]: 
        ax[0].plot(times, cluster_com2[:,0], color='g', lw=1)
        ax[1].plot(times, cluster_com2[:,1], color='g', lw=1, label="$\\theta_2$")
        #ax[1].plot(times[1:], (cluster_com2[1:,1]-cluster_com2[:-1,1])/(times[1:]-times[:-1]), color='g', lw=1, label="$Cluster$ $2$")
        delta_theta = cluster_com1[:,1] - cluster_com2[:,1]
        ax[1].plot(times, delta_theta, color='k', lw=1.1, ls='dashed', label="$\\theta_1 - \\theta_2$")
    ax[1].legend(fontsize=12, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$Time,$ $t$", fontsize=16)
    ax[0].set_ylabel("$\\frac{r(t)}{R}$", fontsize=22, rotation='horizontal', labelpad=20)
    #ax[1].set_ylabel("$\\theta(t)$", fontsize=16, rotation='horizontal', labelpad=20)
    #ax[1].set_ylabel("$\\Delta \\theta(t)$", fontsize=16, rotation='horizontal', labelpad=10)
    ax[1].set_yticks([-np.pi, 0, np.pi])
    ax[1].set_yticklabels(['$-\\pi$', '$0$', '$\\pi$'])
    ax[1].set_yticks([-np.pi, np.pi, 5*np.pi, 10*np.pi, 15*np.pi, 20*np.pi])
    ax[1].set_yticklabels(['$-\\pi$', '$\\pi$', '$5\\pi$', '$10\\pi$', '$15\\pi$', '$20\\pi$'])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/soft/clustersCOM-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotClusterDensity(dirName, figureName, eps=1.5, maxCluster=5):
    fig, ax = plt.subplots(2, 1, figsize=(7,6), sharex=True,dpi = 120)
    dirList, timeList = utils.getOrderedDirectories(dirName)
    numParticles = int(utils.readFromParams(dirName, "numParticles"))
    rad = np.array(np.loadtxt(dirName + os.sep + 'particleRad.dat'))
    eps *= 2.5 * np.mean(rad)
    rgyr = np.empty(0)
    density = np.empty(0)
    times = np.empty(0)
    theta = np.empty(0)
    dt = float(utils.readFromParams(dirName, "dt"))
    for d in range(dirList.shape[0]):
        dirFrame = dirName + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        #print("Time:", timeList[d], "Num clusters:", uniqueLabels.shape[0])
        if uniqueLabels.shape[0] < maxCluster:
            # find largest cluster and plot its COM
            numCluster = np.empty(0)
            clusterLabel = np.empty(0)
            for label in uniqueLabels:
                numCluster = np.append(numCluster, labels[labels==label].shape[0])
                clusterLabel = np.append(clusterLabel, label)
            maxIndex = np.argmax(numCluster)
            maxLabel = clusterLabel[maxIndex]
            clusterIndices = np.where(labels == maxLabel)[0]
            clusterPos = pos[clusterIndices]
            clusterRad = rad[clusterIndices]
            com = np.mean(clusterPos, axis=0)
            clusterPos -= com
            rgyr = np.append(rgyr, np.mean(np.linalg.norm(clusterPos,axis=1)))
            density = np.append(density, np.sum(clusterRad**2) / (2 * rgyr[-1]**2)) # cluster radius is sqrt(2) * rgyr
            times = np.append(times, timeList[d] * dt)
            #print("time:", times[-1], "radius of gyration:", rgyr[-1], "density:", density[-1])
    ax[0].plot(times, rgyr, color='k', lw=1)
    ax[1].plot(times, density, color='k', lw=1, label="$Cluster$ $1$")
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$Time,$ $t$", fontsize=16)
    ax[0].set_ylabel("$Radius$ $of$ $gyration,$ $R_g$", fontsize=16)
    ax[1].set_ylabel("$Cluster$ $density,$ $\\varphi_c$", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/soft/clustePhi-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def phaseDiagrams3(dirName, figureName, dynamics="/", cluster=False, maxCluster=6, interpolate=False):
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(11,3), dpi = 150)
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])#, "0"])
    alignList = np.array(["3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    aligntime = np.array([])
    noisetime = np.array([])
    corr1 = np.array([])
    corr2 = np.array([])
    corr3 = np.array([])
    interCut = 0
    noiseCut = 1e03
    boxSize = np.atleast_1d(np.loadtxt(dirName + "j1e03-tp1e03/boxSize.dat"))
    maxI = boxSize**2
    f0 = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "f0"))
    gamma = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "damping"))
    maxL = boxSize * (f0/gamma)
    colorMap1 = 'plasma'
    colorMap2 = 'plasma'
    colorMap3 = 'plasma'
    if figureName == 'smooth':
        points = np.array(['1e-01', '3', '1e04'])
        color1 = 'darkgrey'#[0.6,0.6,0.6]
        color2 = 'darkviolet'
        color3 = [1,0.7,0]
    elif figureName == 'rough':
        points = np.array(['1e-01', '3', '1e02'])
        color1 = 'forestgreen'#[0,0.6,0.4]
        color2 = 'dodgerblue'#[0.5,0.5,1]
        color3 = [0.6,0,0.8]
    for i in range(noiseList.shape[0]):
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                aligntime = np.append(aligntime, 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"]))
                tp = utils.readFromDynParams(dirSample, "taup")
                tauk = aligntime[-1]
                if i == 0 and alignList[d] == "1e-01":
                    interCut = tauk
                if noiseList[i] == '1e03':
                    if alignList[d] == points[0]: point1 = np.array([tp, tauk])
                    elif alignList[d] == points[1]: point2 = np.array([tp, tauk])
                    if alignList[d] == points[2]: point3 = np.array([tp, tauk])
                data = np.loadtxt(dirSample + "energy.dat")
                if cluster:
                    if(os.path.exists(dirSample + "/t0/")):
                        if not(os.path.exists(dirSample + "/clusterKuramoto!.dat")):
                            print("tp =", noiseList[i], "j =", alignList[d])
                            computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster, minFraction=0.3)
                        corr1 = np.append(corr1, np.mean(np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]))
                    else:
                        corr1 = np.append(corr1, np.mean(data[:,6]))
                else:
                    corr1 = np.append(corr1, np.mean(data[:,6]))
                corr2 = np.append(corr2, np.mean(data[:,-2] / maxI))
                corr3 = np.append(corr3, np.mean(np.abs(data[:,-1]) / maxL))
                if(tp == 0):
                    tp = 1e09
                noisetime = np.append(noisetime, tp)
    vmin = np.min(corr3)
    if vmin < 0: vmin = 0
    vmax = np.max(corr3)
    if vmax > 1: vmax = 1
    if(interpolate == 'interpolate'):
        # Convert to log-space for interpolation
        log_noisetime = np.log10(noisetime)
        log_aligntime = np.log10(aligntime)
        # Remove duplicate points
        points = np.column_stack((log_noisetime, log_aligntime))
        points_unique, idx = np.unique(points, axis=0, return_index=True)
        corr1_unique = corr1[idx]
        corr2_unique = corr2[idx]
        corr3_unique = corr3[idx]
        # Define log grid
        log_tp_lin = np.linspace(points_unique[:,0].min(), points_unique[:,0].max(), 200)
        log_tk_lin = np.linspace(points_unique[:,1].min(), points_unique[:,1].max(), 200)
        grid_tp, grid_tk = np.meshgrid(log_tp_lin, log_tk_lin)
        inter1 = CloughTocher2DInterpolator(points_unique, corr1_unique)
        grid_corr1 = inter1(grid_tp, grid_tk)
        grid_corr1 = gaussian_filter(grid_corr1, sigma=2)
        grid_corr1 = np.clip(grid_corr1, 0, 1)
        inter2 = CloughTocher2DInterpolator(points_unique, corr2_unique)
        grid_corr2 = inter2(grid_tp, grid_tk)
        grid_corr2 = gaussian_filter(grid_corr2, sigma=2)
        grid_corr2 = np.clip(grid_corr2, 0, 1)
        inter3 = CloughTocher2DInterpolator(points_unique, corr3_unique)
        grid_corr3 = inter3(grid_tp, grid_tk)
        grid_corr3 = gaussian_filter(grid_corr3, sigma=2)
        grid_corr3 = np.clip(grid_corr3, 0, 1)
        # Convert grid back to linear scale for plotting
        grid_tp_lin = 10**grid_tp
        grid_tk_lin = 10**grid_tk
        ax[0].contourf(grid_tp_lin, grid_tk_lin, grid_corr1, levels=150, cmap=colorMap1, vmin=vmin, vmax=vmax)
        ax[1].contourf(grid_tp_lin, grid_tk_lin, grid_corr2, levels=150, cmap=colorMap2, vmin=vmin, vmax=vmax)
        contour = ax[2].contourf(grid_tp_lin, grid_tk_lin, grid_corr3, levels=150, cmap=colorMap3, vmin=vmin, vmax=vmax)
        # create a floating inset for the colorbar, relative to the figure
        cax = inset_axes(ax[2], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0.0, 1, 1),  # position outside right edge
                        bbox_transform=ax[2].transAxes, borderpad=0.0)
        cbar = plt.colorbar(contour, cax=cax)
    else:
        ax[0].scatter(noisetime, aligntime, c=corr1, cmap=colorMap1, s=200, edgecolors='k', marker='s', linewidths=0.4, vmin=vmin, vmax=vmax)
        ax[1].scatter(noisetime, aligntime, c=corr2, cmap=colorMap2, s=200, edgecolors='k', marker='s', linewidths=0.4, vmin=vmin, vmax=vmax)
        sc3 = ax[2].scatter(noisetime, aligntime, c=corr3, cmap=colorMap3, s=200, edgecolors='k', marker='s', linewidths=0.4, vmin=vmin, vmax=vmax)
        # create a floating inset for the colorbar, relative to the figure
        cax = inset_axes(ax[2], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0.0, 1, 1),  # position outside right edge
                        bbox_transform=ax[2].transAxes, borderpad=0.0)
        cbar = plt.colorbar(sc3, cax=cax)
    cbar.ax.tick_params(labelsize=12, length=0)
    cbar.set_ticks(np.linspace(vmin,vmax,3))
    cbar.set_ticklabels(["$0.0$", "$0.5$", "$1.0$"])
    # Set log scales for proper visualization
    for i in range(3):
        ax[i].axvline(x=noiseCut, color='aqua', linestyle='solid', lw=3)
        ax[i].axhline(y=interCut, color='chartreuse', linestyle='solid', lw=3)
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xticks([1e-02, 1e01, 1e4, 1e7])
        ax[i].tick_params(axis='both', labelsize=12)
        ax[i].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=14)
    for i in range(3):
        ax[i].plot(point1[0], point1[1], marker='s', color=color1, markersize=12, markeredgecolor='k')
        ax[i].plot(point2[0], point2[1], marker='s', color=color2, markersize=12, markeredgecolor='k')
        if figureName == 'smooth': ax[i].plot(point3[0], point3[1], marker='s', color=color3, markersize=12, markeredgecolor='k', clip_on=False, zorder=10)
        else: ax[i].plot(point3[0], point3[1], marker='s', color=color3, markersize=12, markeredgecolor='k')
    ax[0].set_ylabel("$Alignment$ $time,$ $\\tau_K$", fontsize=14)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.2)
    plt.subplots_adjust(wspace=0.05)
    figureName = "/home/francesco/Pictures/soft/3diagrams-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def phaseDiagrams2(dirName, figureName, dynamics="/", obs='params', cluster=False, maxCluster=20, interpolate=False):
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(9,4), dpi = 180)
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])#,"0"
    alignList = np.array(["3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    aligntime = np.array([])
    noisetime = np.array([])
    corrp = np.array([])
    corrvp = np.array([])
    interCut = 0
    noiseCut = 1e03
    if obs == 'moms':
        colorMap1 = 'plasma'
        colorMap2 = 'plasma'
        #colors = [
        #    (0.0, "white"),
        #    (0.5, 'purple'),
        #    (0.75, 'orange'),
        #    (1.0, 'gold')]
        #colorMap1 = LinearSegmentedColormap.from_list("my_cmap", colors)
        #colorMap2 = LinearSegmentedColormap.from_list("my_cmap", colors)
        cluster = False
        boxSize = np.atleast_1d(np.loadtxt(dirName + "j1e03-tp1e03/boxSize.dat"))
        maxI = boxSize**2
        f0 = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "f0"))
        gamma = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "damping"))
        maxL = boxSize * (f0/gamma)
    else:
        colorMap1 = 'plasma'
        colorMap2 = 'plasma'
    for i in range(noiseList.shape[0]):
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-tp" + noiseList[i] + dynamics
            #if noiseList[i] == '1e03' or alignList[d] == '1e-01':
            #    dirSample = dirName + "j" + alignList[d] + "-tp" + noiseList[i] + dynamics + "dynamics/"
            if(os.path.exists(dirSample)):
                aligntime = np.append(aligntime, 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"]))
                tp = utils.readFromDynParams(dirSample, "taup")
                if i == 0 and alignList[d] == "1e-01":
                    interCut = aligntime[-1]
                data = np.loadtxt(dirSample + "energy.dat")
                if obs == 'params':
                    if cluster == 'cluster':
                        if(os.path.exists(dirSample + "/t0/")):
                            if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                                print("tp =", noiseList[i], "j =", alignList[d])
                                computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster)
                            corrp = np.append(corrp, np.mean(np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]))
                        else:
                            corrp = np.append(corrp, np.mean(data[:,6]))
                    else:
                        corrp = np.append(corrp, np.mean(data[:,6]))
                    corrvp = np.append(corrvp, np.mean(data[:,8]))
                else:
                    corrp = np.append(corrp, np.mean(data[:,-2] / maxI))
                    corrvp = np.append(corrvp, np.mean(np.abs(data[:,-1]) / maxL))
                if(tp == 0):
                    tp = 1e09
                noisetime = np.append(noisetime, tp)
    vmin = np.min(corrvp)
    if vmin < 0: vmin = 0
    vmax = np.max(corrvp)
    if vmax > 1: vmax = 1
    if(interpolate == 'interpolate'):
        # Add a small random noise to avoid numerical issues
        #noisetime += np.random.uniform(0, 0.01, size=noisetime.shape)
        #aligntime += np.random.uniform(0, 0.01, size=aligntime.shape)
        # Convert to log-space for interpolation
        log_noisetime = np.log10(noisetime)
        log_aligntime = np.log10(aligntime)
        # Remove duplicate points
        points = np.column_stack((log_noisetime, log_aligntime))
        points_unique, idx = np.unique(points, axis=0, return_index=True)
        corrp_unique = corrp[idx]
        corrvp_unique = corrvp[idx]
        # Define log grid
        log_tp_lin = np.linspace(points_unique[:,0].min(), points_unique[:,0].max(), 200)
        log_tk_lin = np.linspace(points_unique[:,1].min(), points_unique[:,1].max(), 200)
        grid_tp, grid_tk = np.meshgrid(log_tp_lin, log_tk_lin)
        interp = CloughTocher2DInterpolator(points_unique, corrp_unique)
        grid_corrp = interp(grid_tp, grid_tk)
        grid_corrp = gaussian_filter(grid_corrp, sigma=2)
        grid_corrp = np.clip(grid_corrp, 0, 1)
        intervp = CloughTocher2DInterpolator(points_unique, corrvp_unique)
        grid_corrvp = intervp(grid_tp, grid_tk)
        grid_corrvp = gaussian_filter(grid_corrvp, sigma=2)
        grid_corrvp = np.clip(grid_corrvp, 0, 1)
        # Convert grid back to linear scale for plotting
        grid_tp_lin = 10**grid_tp
        grid_tk_lin = 10**grid_tk
        ax[0].contourf(grid_tp_lin, grid_tk_lin, grid_corrp, levels=150, cmap=colorMap1, vmin=vmin, vmax=vmax)
        contour = ax[1].contourf(grid_tp_lin, grid_tk_lin, grid_corrvp, levels=150, cmap=colorMap2, vmin=vmin, vmax=vmax)
        # create a floating inset for the colorbar, relative to the figure
        cax = inset_axes(ax[1], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0.0, 1, 1),  # position outside right edge
                        bbox_transform=ax[1].transAxes, borderpad=0.0)
        cbar = plt.colorbar(contour, cax=cax)
    else:
        ax[0].scatter(noisetime, aligntime, c=corrp, cmap=colorMap1, s=200, edgecolors='k', marker='s', linewidths=0.4, vmin=vmin, vmax=vmax)
        sc_vp = ax[1].scatter(noisetime, aligntime, c=corrvp, cmap=colorMap2, s=200, edgecolors='k', marker='s', linewidths=0.4, vmin=vmin, vmax=vmax)
        # create a floating inset for the colorbar, relative to the figure
        cax = inset_axes(ax[1], width="5%", height="100%", loc='lower left', bbox_to_anchor=(1, 0.0, 1, 1),  # position outside right edge
                        bbox_transform=ax[1].transAxes, borderpad=0.0)
        cbar = plt.colorbar(sc_vp, cax=cax)
    cbar.ax.tick_params(labelsize=18, length=0)
    cbar.set_ticks(np.linspace(vmin,vmax,3))
    cbar.set_ticklabels(["$0.0$", "$0.5$", "$1.0$"])
    # Set log scales for proper visualization
    print("interCut:", interCut, "noiseCut:", noiseCut)
    for i in range(2):
        ax[i].axvline(x=noiseCut, color='aqua', linestyle='solid', lw=3)
        ax[i].axhline(y=interCut, color='chartreuse', linestyle='solid', lw=3)
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xticks([1e-02, 1e01, 1e4, 1e7])
        ax[i].tick_params(axis='both', labelsize=18)
        ax[i].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    ax[0].set_ylabel("$Alignment$ $time,$ $\\tau_K$", fontsize=22)
    plt.subplots_adjust(left=0.12, right=0.9, top=0.95, bottom=0.2)
    plt.subplots_adjust(wspace=0.05)
    figureName = "/home/francesco/Pictures/soft/2diagrams-" + obs + "-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def phaseDiagramAngleDistance(dirName, figureName, dynamics="/", interpolate=False):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    aligntime = np.array([])
    noisetime = np.array([])
    angle_delta = np.array([])
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08", "0"])
    alignList = np.array(["3e-02", "1e-01", "3e-01", "4e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    for i in range(noiseList.shape[0]):
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-tp" + noiseList[i] + dynamics
            if(os.path.exists(dirSample)):
                dirList, _ = utils.getOrderedDirectories(dirSample)
                if(dirList.shape[0] != 0):
                    flag = False
                    if not(os.path.exists(dirSample + "/angleDistance.dat")):
                        flag = True
                        computeAngleDistance(dirSample)
                    distance = np.loadtxt(dirSample + os.sep + "angleDistance.dat")
                    angle_delta = np.append(angle_delta, np.mean(distance[:,1]))
                    aligntime = np.append(aligntime, 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"]))
                    if(flag == True): print(noiseList[i], alignList[d], angle_delta[-1])
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
        contour = plt.contourf(grid_tp_lin, grid_tk_lin, grid_corr, levels=20, cmap='jet_r', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contour, ax=ax, pad=0, aspect=20)
    else:
        vmin = np.min(angle_delta)
        vmax = np.max(angle_delta)
        print("max:", vmax, "min:", vmin)
        full_cmap = cm.get_cmap('jet_r')
        N = angle_delta.shape[0]
        trunc_cmap = truncated_colormap(full_cmap, minval=vmin, maxval=vmax-0.32, n=N)
        trunc_cmap = darken_cmap(trunc_cmap, factor=0.99)
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

def getWallIndexYlabel(which):
    if which == 'omega':
        index = 3
        ylabel = '$|\\omega|$'
    elif which == 'krot':
        index = 3
        ylabel = '$K_{rot}$'
    elif which == 'alpha':
        index = 4
        ylabel = '$|\\alpha|$'
    else:
        index = 2
        ylabel = '$|\\Delta \\theta|$'
    return index, ylabel

def plotWallDynamics(dirName, figureName, which='omega'):
    fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    index, ylabel = getWallIndexYlabel(which)
    # get wall dynamics from directories in dirName
    angleDyn = np.loadtxt(dirName + "wallDynamics.dat")
    if which == 'krot':
        boxRadius = np.loadtxt(dirName + "boxSize.dat")
        angleDyn[:,index] = 0.5 * angleDyn[:,index]**2 * boxRadius **2
    ax.plot(angleDyn[:,0], angleDyn[:,index], lw=1.2, color='k')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylabel(ylabel, rotation='horizontal', labelpad=20, fontsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=14)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/wall-" + which + "-" + figureName
    # second plot
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareWallDynamicsAlign(dirName, figureName, which='omega', dynamics='/'):
    alignList = np.array(["1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    alignList = np.array(["1e-01", "3e-01", "1", "1.5", "2.5", "3", "5", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    index, ylabel = getWallIndexYlabel(which)
    colorList = cm.get_cmap('plasma')
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    if which != 'angle':
        aligntime = np.zeros(alignList.shape[0])
        mean = np.zeros(alignList.shape[0])
        error = np.zeros(alignList.shape[0])
    for d in range(alignList.shape[0]):
        dirSample = dirName + "j" + alignList[d] + "-tp1e03/dynamics-vel/rigid" + dynamics
        if os.path.exists(dirSample + "wallDynamics.dat"):
            angleDyn = np.loadtxt(dirSample + "wallDynamics.dat")
            angleDyn = angleDyn[10:,:]  # remove initial transient
            angleDyn[:,2] -= angleDyn[0,2]  # subtract initial angle
            #print(angleDyn.shape)
            if which == 'krot':
                boxRadius = np.loadtxt(dirSample + "boxSize.dat")
                angleDyn[:,index] = 0.5 * angleDyn[:,index]**2 * boxRadius **2
            else:
                angleDyn[:,index] = np.abs(angleDyn[:,index])
            ax.plot(angleDyn[:,0], angleDyn[:,index], lw=1.2, color=colorList(d/alignList.shape[0]))
            if which != 'angle':
                aligntime[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
                mean[d] = np.mean(angleDyn[:-100,index])
                error[d] = np.std(angleDyn[:-100,index])
    if which != 'angle':
        ax.plot(np.linspace(0, angleDyn[-1,0], 100), np.zeros(100), ls='--', lw=0.8, color='k')
    ax.tick_params(axis='both', labelsize=12)
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.)
    cbar = fig.colorbar(colorBar, cax)
    cbar.set_label('$J_K$', rotation='horizontal', fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=14, length=0)
    cbar.set_ticks(np.linspace(0,1,3))
    cbar.set_ticklabels(['$1$', '$10^2$', '$10^4$'])
    ax.set_ylabel(ylabel, rotation='horizontal', labelpad=20, fontsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=14)
    plt.tight_layout()
    # second plot
    if which != 'angle':
        fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
        ax.errorbar(aligntime, mean, error/np.sqrt(mean.shape[0]), lw=1.2, color='b', marker='s', markersize=8, fillstyle='none', capsize=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylabel(ylabel, rotation='horizontal', labelpad=0, fontsize=16)
        ax.set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=15)
        plt.tight_layout()
        figureName = "/home/francesco/Pictures/soft/wallAlign-" + which + "-time-" + figureName
        fig.savefig(figureName + ".png", transparent=True, format = "png")
    else:
        figureName = "/home/francesco/Pictures/soft/wallAlign-" + which + "-" + figureName
        fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareWallDynamicsDamping(dirName, figureName, which='omega'):
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
        angleDyn = np.laodtxt(dirSample + "wallDynamics.dat")
        omega[t,0] = np.mean(np.abs(angleDyn[:,3]))
        omega[t,1] = np.std(np.abs(angleDyn[:,3]))
        alpha[t,0] = np.mean(np.abs(angleDyn[:,4]))
        alpha[t,1] = np.std(np.abs(angleDyn[:,4]))
        ekin[t,0] = np.mean(0.5*angleDyn[:,3]**2*boxRadius**2*utils.readFromWallParams(dirSample, "numWall"))
        ekin[t,1] = np.std(0.5*angleDyn[:,3]**2*boxRadius**2*utils.readFromWallParams(dirSample, "numWall"))
        noise[t] = utils.readFromDynParams(dirSample, "damping")
        ax.plot(angleDyn[:,0], angleDyn[:,1], lw=1, color=colorList((typeList.shape[0]-t-1)/typeList.shape[0]), label="$\\gamma=$" + damping[t])
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

def plotAlignParamsVSInteraction(dirName, figureName, cluster=False, maxCluster=32): # for maximum number of clusters to consider
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,7), dpi = 120)
    alignList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "2e-01", "3e-01", "5e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    dirList = np.array(["/reflect/", "/rough/dynamics/"])
    for a in range(dirList.shape[0]):
        ax_twin = ax[a].twinx()
        tauk = np.zeros(alignList.shape[0])
        corr1 = np.zeros((alignList.shape[0], 2))
        corr2 = np.zeros((alignList.shape[0], 2))
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-" + figureName + "/dynamics-vel" + dirList[a]
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                tauk[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
                if alignList[d] == "1e02" and dirList[a] == "/rough/dynamics/":
                    print(tauk[d])
                if cluster == 'cluster':
                    if(os.path.exists(dirSample + "/t0/")):
                        computed = False
                        if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                            computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster)
                            computed = True
                        corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                        if computed:
                            print(dirList[a], alignList[d], tauk[d], "computed cluster kuramoto", np.mean(corrCluster))
                        corr1[d,0] = np.mean(corrCluster)
                        corr1[d,1] = np.std(corrCluster)
                    else:
                        corr1[d,0] = np.mean(data[:,6])
                        corr1[d,1] = np.std(data[:,6])
                else:
                    corr1[d,0] = np.mean(data[:,6])
                    corr1[d,1] = np.std(data[:,6])
                corr2[d,0] = np.mean(data[:,8])
                corr2[d,1] = np.std(data[:,8])
        #corr1 = np.column_stack((utils.computeMovingAverage(corr1[:,0], 2), utils.computeMovingAverage(corr1[:,1], 2)))
        #corr2 = np.column_stack((utils.computeMovingAverage(corr2[:,0], 2), utils.computeMovingAverage(corr2[:,1], 2)))
        ax[a].errorbar(tauk[tauk!=0], corr1[tauk!=0,0], corr1[tauk!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax_twin.errorbar(tauk[tauk!=0], corr2[tauk!=0,0], corr2[tauk!=0,1], color='b', marker='s', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax[a].set_xscale('log')
        ax[a].set_ylim(-0.08, 1.08)
        ax_twin.set_ylim(ax[a].get_ylim())
        ax[a].tick_params(axis='both', labelsize=14)
        ax_twin.tick_params(axis='y', colors='b', labelsize=14)
        ax[a].set_ylabel("$\\phi_r^C$", fontsize=18, rotation='horizontal', labelpad=5)
        ax_twin.set_ylabel("$\\phi_\\alpha$", fontsize=18, color='b', rotation='horizontal', labelpad=5)
        # Align labels vertically at center (x is position from the axis)
        ax[a].yaxis.set_label_coords(-0.15, 0.46)
        ax_twin.yaxis.set_label_coords(1.15, 0.57)
    ax[1].tick_params(axis='x', which='both', labeltop=False, top=True)
    ax[1].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    if cluster == 'cluster':
        figureName = "/home/francesco/Pictures/soft/interParamsCluster-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/interParams-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotAlignParamsVSNoise(dirName, figureName, cluster=False, maxCluster=32): # for maximum number of clusters to consider
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,7), dpi = 120)
    noiseList = np.array(["1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
    dirList = np.array(["/reflect/", "/rough/dynamics/"])
    for a in range(dirList.shape[0]):
        ax_twin = ax[a].twinx()
        taup = np.zeros(noiseList.shape[0])
        corr1 = np.zeros((noiseList.shape[0], 2))
        corr2 = np.zeros((noiseList.shape[0], 2))
        for d in range(noiseList.shape[0]):
            dirSample = dirName + figureName + "-tp" + noiseList[d] + "/dynamics-vel" + dirList[a]
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                taup[d] = utils.readFromDynParams(dirSample, "taup")
                if cluster == 'cluster':
                    if(os.path.exists(dirSample + "/t0/")):
                        computed = False
                        if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                            computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster)
                            computed = True
                        corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                        if computed:
                            print(dirList[a], noiseList[d], taup[d], "computed cluster kuramoto", np.mean(corrCluster))
                        corr1[d,0] = np.mean(corrCluster)
                        corr1[d,1] = np.std(corrCluster)
                    else:
                        corr1[d,0] = np.mean(data[:,6])
                        corr1[d,1] = np.std(data[:,6])
                else:
                    corr1[d,0] = np.mean(data[:,6])
                    corr1[d,1] = np.std(data[:,6])
                corr2[d,0] = np.mean(data[:,8])
                corr2[d,1] = np.std(data[:,8])
        ax[a].errorbar(taup[taup!=0], corr1[taup!=0,0], corr1[taup!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax_twin.errorbar(taup[taup!=0], corr2[taup!=0,0], corr2[taup!=0,1], color='b', marker='s', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax[a].set_xscale('log')
        ax[a].set_ylim(-0.08, 1.08)
        ax_twin.set_ylim(ax[a].get_ylim())
        ax[a].tick_params(axis='both', labelsize=14)
        ax_twin.tick_params(axis='y', colors='b', labelsize=14)
        ax[a].set_ylabel("$\\phi_r^C$", fontsize=18, rotation='horizontal', labelpad=5)
        ax_twin.set_ylabel("$\\phi_\\alpha$", fontsize=18, color='b', rotation='horizontal', labelpad=5)
        # Align labels vertically at center (x is position from the axis)
        ax[a].yaxis.set_label_coords(-0.15, 0.46)
        ax_twin.yaxis.set_label_coords(1.15, 0.57)
    ax[1].tick_params(axis='x', which='both', labeltop=False, top=True)
    ax[1].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    if cluster == 'cluster':
        figureName = "/home/francesco/Pictures/soft/noiseParamsCluster-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/noiseParams-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotAlphaParamsVSInteraction(dirName, figureName, cluster=False, maxCluster=32): # for maximum number of clusters to consider
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,7), dpi = 120)
    alignList = np.array(["1e-04", "3e-04", "1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    dirList = np.array(["/reflect/", "/rough/dynamics/"])
    for a in range(dirList.shape[0]):
        ax_twin = ax[a].twinx()
        tauk = np.zeros(alignList.shape[0])
        corr1 = np.zeros((alignList.shape[0], 2))
        corr2 = np.zeros((alignList.shape[0], 2))
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-tp1e03/dynamics-vel" + dirList[a]
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "energy.dat")
                tauk[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
                if cluster == 'cluster':
                    if(os.path.exists(dirSample + "/t0/")):
                        computed = False
                        if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                            computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster)
                            computed = True
                        corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                        if computed:
                            print(dirList[a], alignList[d], tauk[d], "computed cluster kuramoto", np.mean(corrCluster))
                        corr1[d,0] = np.mean(corrCluster)
                        corr1[d,1] = np.std(corrCluster)
                    else:
                        corr1[d,0] = np.mean(data[:,9])
                        corr1[d,1] = np.std(data[:,9])
                else:
                    corr1[d,0] = np.mean(data[:,9])
                    corr1[d,1] = np.std(data[:,9])
                corr2[d,0] = np.mean(data[:,10])
                corr2[d,1] = np.std(data[:,10])
                #print(dirList[a], alignList[d], tauk[d], "mean alpha_r", np.mean(corr1[d,0]), "mean alpha_phi", np.mean(corr2[d,0]))
        #corr1 = np.column_stack((utils.computeMovingAverage(corr1[:,0], 2), utils.computeMovingAverage(corr1[:,1], 2)))
        #corr2 = np.column_stack((utils.computeMovingAverage(corr2[:,0], 2), utils.computeMovingAverage(corr2[:,1], 2)))
        ax[a].errorbar(tauk[tauk!=0], corr1[tauk!=0,0], corr1[tauk!=0,1], color='k', marker='o', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax_twin.errorbar(tauk[tauk!=0], corr2[tauk!=0,0], corr2[tauk!=0,1], color='b', marker='s', markersize=8, capsize=3, fillstyle='none', lw=1)
        ax[a].set_xscale('log')
        ax[a].set_ylim(-0.08,1.22)
        ax_twin.set_ylim(ax[a].get_ylim())
        ax[a].tick_params(axis='both', labelsize=14)
        ax_twin.tick_params(axis='y', colors='b', labelsize=14)
        ax[a].set_ylabel("$\\alpha_r$", fontsize=18, rotation='horizontal', labelpad=-5)
        ax_twin.set_ylabel("$\\alpha_\\phi$", fontsize=18, color='b', rotation='horizontal', labelpad=5)
        # Align labels vertically at center (x is position from the axis)
        ax[a].yaxis.set_label_coords(-0.15, 0.5)
        ax_twin.yaxis.set_label_coords(1.15, 0.57)
    ax[1].tick_params(axis='x', which='both', labeltop=False, top=True)
    ax[1].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    if cluster == 'cluster':
        figureName = "/home/francesco/Pictures/soft/alphaParamsCluster-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/alphaParams-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotBoundaryVSTime(dirName, figureName, which='corr', dynamics='/'):
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    dirList = np.array(["1e-04", "3e-04", "1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    colorList = cm.get_cmap('plasma')
    index, ylabel = getIndexYlabel(which)
    for d in range(dirList.shape[0]):
        dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/" + dynamics
        if(os.path.exists(dirSample + "/energy.dat")):
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            ax.plot(energy[::2,0], energy[::2,index], linewidth=1.2, color=colorList(d/dirList.shape[0]), label="$J_K=$" + dirList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Simulation$ $step$", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/boundTime-" + which + "-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotBoundaryType(dirName, figureName, versus='inter', which='pres', dynamics='/', cluster=True, maxCluster=12):
    fig, ax = plt.subplots(2, 1, figsize=(5,6), dpi = 120)
    if versus == 'inter':
        dirList = np.array(["1e-04", "3e-04", "1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01",
                            "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
        xlabel = "$Alignment$ $time,$ $\\tau_K$"
    else:
        dirList = np.array(["1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    aligntime = np.zeros(dirList.shape[0])
    noisetime = np.zeros(dirList.shape[0])
    corr1 = np.zeros((dirList.shape[0], 2))
    corr2 = np.zeros((dirList.shape[0], 2))
    if which == 'pres':
        index1 = 4
        index2 = 5
        ylabel1 = "$P_r$"
        ylabel2 = "$|P_\\phi|$"
    elif which == 'velpos':
        index1 = 6
        index2 = 8
        ylabel1 = "$\\phi_r$"
        ylabel2 = "$\\phi_\\alpha$"
    elif which == 'moms':
        index1 = -2
        index2 = -1
        ylabel1 = "$\\tilde{I}$"
        ylabel2 = "$\\tilde{L}$"
        boxSize = np.atleast_1d(np.loadtxt(dirName + "j1e03-tp1e03/boxSize.dat"))
        maxI = boxSize**2
        f0 = utils.readFromDynParams(dirName + "j1e03-tp1e03", "f0")
        gamma = utils.readFromDynParams(dirName + "j1e03-tp1e03", "damping")
        maxL = boxSize * (f0/gamma)
    else:
        index1 = 2
        index2 = 3
        ylabel1 = "$U/N$"
        ylabel2 = "$K/N$"
    for d in range(dirList.shape[0]):
        if versus == 'inter':
            dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/" + dynamics
        else:
            dirSample = dirName + "j1e-01-tp" + dirList[d] + "/dynamics-vel/" + dynamics
        if(os.path.exists(dirSample + "/energy.dat")):
            aligntime[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
            noisetime[d] = utils.readFromDynParams(dirSample, "taup")
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            if which == 'velpos':
                if cluster == 'cluster':
                    if(os.path.exists(dirSample + "/t0/")):
                        if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                            computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster)
                        corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                        corr1[d,0] = np.mean(corrCluster)
                        corr1[d,1] = np.std(corrCluster)
                    else:
                        corr1[d,0] = np.mean(energy[:,index1])
                        corr1[d,1] = np.std(energy[:,index1])
                else:
                    corr1[d,0] = np.mean(energy[:,index1])
                    corr1[d,1] = np.std(energy[:,index1])
                    corr2[d,0] = np.mean(energy[:,index2])
                    corr2[d,1] = np.std(energy[:,index2])
            else:
                if which == 'moms':
                    energy[:,index1] /= maxI
                    energy[:,index2] = np.abs(energy[:,index2]) / maxL
                elif which == 'pres':
                    energy[:,index2] = np.abs(energy[:,index2])
                corr1[d,0] = np.mean(energy[:,index1])
                corr1[d,1] = np.std(energy[:,index1])
                corr2[d,0] = np.mean(energy[:,index2])
                corr2[d,1] = np.std(energy[:,index2])
    if versus == 'inter':
        x = aligntime
    else:
        x = noisetime
    ax[0].errorbar(x[corr1[:,0]!=0], corr1[corr1[:,0]!=0,0], corr1[corr1[:,0]!=0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax[1].errorbar(x[corr2[:,0]!=0], corr2[corr2[:,0]!=0,0], corr2[corr2[:,0]!=0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].tick_params(axis='x', labelbottom=False)
    ax[1].tick_params(axis='both', labelsize=14)
    #ax[0].set_xlabel(xlabel, fontsize=16)
    ax[1].set_xlabel(xlabel, fontsize=16)
    if which == 'moms':
        ax[0].set_ylabel(ylabel1, fontsize=18, rotation='horizontal', labelpad=15)
        ax[1].set_ylabel(ylabel2, fontsize=18, rotation='horizontal', labelpad=15)
    else:
        ax[0].set_ylabel(ylabel1, fontsize=16, rotation='horizontal', labelpad=15)
        ax[1].set_ylabel(ylabel2, fontsize=16, rotation='horizontal', labelpad=25)
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    ax[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    figureName = "/home/francesco/Pictures/soft/" + which + "-" + figureName + "-" + versus
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def computeNumClusterVSTime(dirSample, minNum=2, eps=1.5):
    dirList, timeList = utils.getOrderedDirectories(dirSample)
    rad = np.loadtxt(dirSample + "particleRad.dat")
    numParticles = rad.shape[0]
    eps *= 2 * np.mean(rad)
    numLabels = np.zeros(dirList.shape[0])
    numInCluster = np.zeros(dirList.shape[0])
    fraction = np.zeros(dirList.shape[0])
    free = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirFrame = dirSample + os.sep + dirList[d] + os.sep
        pos = np.array(np.loadtxt(dirFrame + os.sep + 'particlePos.dat'))
        labels = utils.getDBClusterLabels(pos, eps, min_samples=2, denseList=np.ones(pos.shape[0]))
        uniqueLabels = np.unique(labels)
        numClusters = 0 # number of clusters with more than minNum particles
        fracCluster = 0
        for label in uniqueLabels:
            if label != -1 and labels[labels==label].shape[0] > minNum:
                numClusters += 1
                fracCluster += labels[labels==label].shape[0]
        numLabels[d] = numClusters
        if numClusters > 0:
            numInCluster[d] = fracCluster / numClusters
        fraction[d] = fracCluster / numParticles
        free[d] = labels[labels==-1].shape[0] / numParticles
        #print("Time:", timeList[d], "Num clusters:", numClusters, "fraction in clusters:", fraction[d], "fraction free:", free[d])
    np.savetxt(dirSample + "/numCluster.dat", np.column_stack((timeList, numLabels, numInCluster, fraction, free)))

def compareNumClusterVSTime(dirName, figureName, versus="inter", which="num", dynamics="/", minNum=2):
    fig, ax = plt.subplots(1, 2, figsize=(10,4), dpi = 120)
    dt = float(utils.readFromParams(dirName + "j1e03-tp1e03", "dt"))
    if versus == "inter":
        dirList = np.array(["1e-04", "3e-04", "1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "2e-01", "3e-01", "5e-01", 
                            "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
        cblabel = "$J_K$"
        xlabel = "$Alignment$ $time,$ $\\tau_K$"
    else:
        dirList = np.array(["1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
        cblabel = "$\\tau_p$"
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    colorList = cm.get_cmap('plasma')
    aligntime = np.zeros(dirList.shape[0])
    noisetime = np.zeros(dirList.shape[0])
    numCluster1 = np.zeros((dirList.shape[0],2))
    numCluster2 = np.zeros((dirList.shape[0],2))
    numCluster3 = np.zeros((dirList.shape[0],2))
    if which == "num":
        index = 1
        ylabel1 = "$N_C$"
        ylabel2 = "$\\langle N_C \\rangle$"
    elif which == "numin":
        index = 2
        ylabel1 = "$\\bar{N}_p$"
        ylabel2 = "$\\langle \\bar{N}_p \\rangle$"
    elif which == "frac":
        index = 3
        ylabel1 = "$f_C$"
        ylabel2 = "$\\langle f_C \\rangle$"
    else:
        index = -1
        ylabel1 = "$f_0$"
        ylabel2 = "$\\langle f_0 \\rangle$"
    for d in range(dirList.shape[0]):
        if versus == "inter":
            dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/" + dynamics
        else:
            dirSample = dirName + "j1e-01-tp" + dirList[d] + "/dynamics-vel/" + dynamics
        aligntime[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
        noisetime[d] = utils.readFromDynParams(dirSample, "taup")
        if(os.path.exists(dirSample + "/t0/")):
            if not(os.path.exists(dirSample + "/numCluster.dat")):
                computeNumClusterVSTime(dirSample, minNum)
            clusterData = np.loadtxt(dirSample + "/numCluster.dat")
            ax[1].plot(clusterData[:,0]*dt, clusterData[:,index], linewidth=1, color=colorList(d/dirList.shape[0]), label ="$J_K=$" + dirList[d])
            numCluster1[d,0] = np.mean(clusterData[-20:,index])
            numCluster1[d,1] = np.std(clusterData[-20:,index])
            numCluster2[d,0] = np.mean(clusterData[-50:,index])
            numCluster2[d,1] = np.std(clusterData[-50:,index])
            numCluster3[d,0] = np.mean(clusterData[:,index])
            numCluster3[d,1] = np.std(clusterData[:,index])
    if which == 'num':
        upper_lim = 92
        ax[1].set_ylim(-3,upper_lim)
        ax[0].set_ylim(-3,upper_lim)
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.)
    #cax = fig.add_subplot(gs[2])
    cbar = fig.colorbar(colorBar, cax)
    cbar.set_label(cblabel, rotation='horizontal', fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=14, length=0)
    if versus == 'inter':
        cbar.set_ticks(np.linspace(0,1,4))
        cbar.set_ticklabels(['$10^{-4}$', '$10^{-2}$', '$10^2$', '$10^4$'])
        x = aligntime
    else:
        cbar.set_ticks(np.linspace(0,1,3))
        cbar.set_ticklabels(['$1$', '$10^4$', '$10^8$'])
        x = noisetime
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$Time,$ $t$", fontsize=14)
    ax[1].set_ylabel(ylabel1, fontsize=14, rotation='horizontal', labelpad=10)
    ax[0].errorbar(x[numCluster1[:,0]!=0], numCluster1[numCluster1[:,0]!=0,0], numCluster1[numCluster1[:,0]!=0,1], 
                   lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label="$t > 0.8 t_{max}$")
    ax[0].errorbar(x[numCluster2[:,0]!=0], numCluster2[numCluster2[:,0]!=0,0], numCluster2[numCluster2[:,0]!=0,1], 
                   lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label="$t > 0.5 t_{max}$", alpha=0.4)
    ax[0].errorbar(x[numCluster3[:,0]!=0], numCluster3[numCluster3[:,0]!=0,0], numCluster3[numCluster3[:,0]!=0,1], 
                   lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label="$t > 0$", alpha=0.2)
    ax[0].legend(fontsize=11, loc='best')
    ax[0].set_xscale('log')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].set_xlabel(xlabel, fontsize=14)
    ax[0].set_ylabel(ylabel2, fontsize=14, rotation='horizontal', labelpad=15)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/numCluster-" + figureName + "-" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotClustersVSBoundary(dirName, figureName, minNum=2):
    fig, ax = plt.subplots(2, 2, figsize=(9,7), dpi = 200)
    ax[0,1].sharey(ax[0,0])   # top row
    ax[1,1].sharey(ax[1,0])   # bottom row
    numParticles = float(utils.readFromParams(dirName + "j1e03-tp1e03", "numParticles"))
    interList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1",
                          "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    noiseList = np.array(["1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
    aligntime = np.zeros(interList.shape[0])
    noisetime = np.zeros(noiseList.shape[0])
    boundary = np.array(["reflect/dynamics/", "rough/dynamics/", "fixed/dynamics/"])
    labelList = np.array(["$Reflective$", "$Rough$", "$Fixed$"])
    colorList = [[0.6,0.6,0.6], 'k', [0.6,0.6,1]]
    markerList = ['o', 's', 'D']
    index1 = 1
    ylabel1 = "$\\langle N_c \\rangle$"
    index2 = 2
    ylabel2 = "$\\frac{\\langle N_p \\rangle}{N}$"
    ax[0,0].tick_params(axis='both', labelsize=17)
    ax[0,1].tick_params(axis='both', labelsize=17)
    ax[1,0].tick_params(axis='both', labelsize=17)
    ax[1,1].tick_params(axis='both', labelsize=17)
    ax[0,0].set_ylabel(ylabel1, fontsize=24, rotation='horizontal', labelpad=35)
    ax[1,0].set_ylabel(ylabel2, fontsize=34, rotation='horizontal', labelpad=30)
    ax[1,0].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=22)
    ax[1,1].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    # collect data vs interaction strength
    for b in range(boundary.shape[0]):
        numCluster = np.zeros((interList.shape[0],2))
        clusterSize = np.zeros((interList.shape[0],2))
        for d in range(interList.shape[0]):
            dirSample = dirName + "j" + interList[d] + "-tp1e03/dynamics-vel/" + boundary[b]
            aligntime[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
            if interList[d] == "1e-01":
                interCut = aligntime[d]
            if(os.path.exists(dirSample + "/t0/")):
                if not(os.path.exists(dirSample + "/numCluster.dat")):
                    computeNumClusterVSTime(dirSample, minNum)
                #if boundary[b] == "rough/dynamics/" and interList[d] == "1e04":
                #    print("interaction:", interList[d], "aligntime:", aligntime[d])
                #    computeNumClusterVSTime(dirSample, minNum)
                clusterData = np.loadtxt(dirSample + "/numCluster.dat")
                numCluster[d,0] = np.mean(clusterData[:,index1])
                numCluster[d,1] = np.std(clusterData[:,index1])
                clusterSize[d,0] = np.mean(clusterData[:,index2])/numParticles
                clusterSize[d,1] = np.std(clusterData[:,index2])/numParticles
                if interList[d] == "1e-01":
                    numCluster_same = numCluster[d]
                    clusterSize_same = clusterSize[d]
                    print("clusterSize at inter cut:", clusterSize[d])
                if interList[d] == "3e03":
                    print("tauk:", aligntime[d])
        ax[0,0].errorbar(aligntime[numCluster[:,0]!=0], numCluster[numCluster[:,0]!=0,0], numCluster[numCluster[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        ax[1,0].errorbar(aligntime[clusterSize[:,0]!=0], clusterSize[clusterSize[:,0]!=0,0], clusterSize[clusterSize[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        #ax[0,0].errorbar(interCut, numCluster_same[0], numCluster_same[1], 
        #                 lw=0, color=starList[b], marker=markStarList[b], markersize=18, fillstyle='full', capsize=3, markeredgecolor='k', markeredgewidth=0.7)
        #ax[1,0].errorbar(interCut, clusterSize_same[0], clusterSize_same[1], 
        #                 lw=0, color=starList[b], marker=markStarList[b], markersize=18, fillstyle='full', capsize=3, markeredgecolor='k', markeredgewidth=0.7)
    #ax[0,0].axvline(x=interCut, color='k', linestyle='dotted', lw=0.7)
    #ax[1,0].axvline(x=interCut, color='k', linestyle='dotted', lw=0.7)
    # collect data vs noise strength
    for b in range(boundary.shape[0]):
        numCluster = np.zeros((noiseList.shape[0],2))
        clusterSize = np.zeros((noiseList.shape[0],2))
        for d in range(noiseList.shape[0]):
            dirSample = dirName + "j1e-01-tp" + noiseList[d] + "/dynamics-vel/" + boundary[b]
            noisetime[d] = utils.readFromDynParams(dirSample, "taup")
            if noiseList[d] == "1e03":
                noiseCut = noisetime[d]
            if(os.path.exists(dirSample + "/t0/")):
                if not(os.path.exists(dirSample + "/numCluster.dat")):
                    computeNumClusterVSTime(dirSample, minNum)
                #if boundary[b] == "rough/dynamics/" and noiseList[d] == "1e-02":
                #    print("noise:", noiseList[d], "noisetime:", noisetime[d])
                #    computeNumClusterVSTime(dirSample, minNum)
                clusterData = np.loadtxt(dirSample + "/numCluster.dat")
                numCluster[d,0] = np.mean(clusterData[:,index1])
                numCluster[d,1] = np.std(clusterData[:,index1])
                clusterSize[d,0] = np.mean(clusterData[:,index2])/numParticles
                clusterSize[d,1] = np.std(clusterData[:,index2])/numParticles
                if noiseList[d] == "1e03":
                    numCluster_same = numCluster[d]
                    clusterSize_same = clusterSize[d]
                    print("clusterSize at noise cut:", clusterSize[d])
        ax[0,1].errorbar(noisetime[numCluster[:,0]!=0], numCluster[numCluster[:,0]!=0,0], numCluster[numCluster[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        ax[1,1].errorbar(noisetime[clusterSize[:,0]!=0], clusterSize[clusterSize[:,0]!=0,0], clusterSize[clusterSize[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        #ax[0,1].errorbar(noiseCut, numCluster_same[0], numCluster_same[1], 
        #                 lw=0, color=starList[b], marker=markStarList[b], markersize=18, fillstyle='full', capsize=3, markeredgecolor='k', markeredgewidth=0.7)
        #ax[1,1].errorbar(noiseCut, clusterSize_same[0], clusterSize_same[1], 
        #                 lw=0, color=starList[b], marker=markStarList[b], markersize=18, fillstyle='full', capsize=3, markeredgecolor='k', markeredgewidth=0.7)
    #ax[0,1].axvline(x=noiseCut, color='k', linestyle='dotted', lw=0.7)
    #ax[1,1].axvline(x=noiseCut, color='k', linestyle='dotted', lw=0.7)
    print("Interaction cut tau_K =", interCut, ", J_K =", 1/interCut, "K =", np.pi * 1.5**2/interCut)
    print("Noise cut tau_p =", noiseCut)
    ax[1,0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[1,1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[1,0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[0,0].set_xscale('log')
    ax[1,0].set_xscale('log')
    ax[0,1].set_xscale('log')
    ax[1,1].set_xscale('log')
    ax[0,0].set_yscale('log')
    ax[1,0].set_yscale('log')
    ax[0,1].set_yscale('log')
    ax[1,1].set_yscale('log')
    ax[1,0].xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax[1,0].xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax[1,1].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax[1,1].xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    #ax[0,1].set_ylim(0.016,)
    #ax[1,1].set_ylim(0.000064,)
    ax[0,1].set_ylim(0.12,)
    ax[1,1].set_ylim(0.00034,)
    ax[0,1].tick_params(labelleft=False)
    ax[1,1].tick_params(labelleft=False)
    ax[1,0].tick_params(top=True)
    ax[1,1].tick_params(top=True)
    ax[0,0].tick_params(labelbottom=False, bottom=True)
    ax[0,1].tick_params(labelbottom=False, bottom=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    figureName = "/home/francesco/Pictures/soft/cluster-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotMomentsVSBoundary(dirName, figureName):
    fig, ax = plt.subplots(2, 2, figsize=(9,7), dpi = 200)
    ax[0,1].sharey(ax[0,0])   # top row
    ax[1,1].sharey(ax[1,0])   # bottom row
    boxSize = np.atleast_1d(np.loadtxt(dirName + "j1e03-tp1e03/boxSize.dat"))
    maxI = boxSize[0]**2
    f0 = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "f0"))
    gamma = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "damping"))
    maxL = boxSize[0] * (f0/gamma)
    interList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1",
                          "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    noiseList = np.array(["1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
    aligntime = np.zeros(interList.shape[0])
    noisetime = np.zeros(noiseList.shape[0])
    boundary = np.array(["reflect/dynamics/", "rough/dynamics/", "fixed/dynamics/"])
    labelList = np.array(["$Reflective$", "$Rough$", "$Fixed$"])
    colorList = [[0.6,0.6,0.6], 'k', [0.6,0.6,1]]
    markerList = ['o', 's', 'D']
    index1 = -2
    ylabel1 = "$\\langle \\tilde{I} \\rangle$" #"$\\frac{\\langle I \\rangle}{M R^2}$"
    index2 = -1
    ylabel2 = "$\\langle \\tilde{L} \\rangle$" #"$\\frac{\\langle |L| \\rangle}{M v_0 R}$"
    ax[0,0].tick_params(axis='both', labelsize=17)
    ax[0,1].tick_params(axis='both', labelsize=17)
    ax[1,0].tick_params(axis='both', labelsize=17)
    ax[1,1].tick_params(axis='both', labelsize=17)
    ax[0,0].set_ylabel(ylabel1, fontsize=22, rotation='horizontal', labelpad=25)
    ax[1,0].set_ylabel(ylabel2, fontsize=22, rotation='horizontal', labelpad=25)
    ax[1,0].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=22)
    ax[1,1].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    # collect data vs interaction strength
    for b in range(boundary.shape[0]):
        plotline = True
        MOI = np.zeros((interList.shape[0],2))
        angMom = np.zeros((interList.shape[0],2))
        for d in range(interList.shape[0]):
            dirSample = dirName + "j" + interList[d] + "-tp1e03/dynamics-vel/" + boundary[b]
            if os.path.exists(dirSample + "/energy.dat"):
                aligntime[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
                energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                MOI[d,0] = np.abs(np.mean(energy[:,index1])) / maxI
                MOI[d,1] = np.std(energy[:,index1]) / maxI
                angMom[d,0] = np.abs(np.mean(energy[:,index2])) / maxL
                angMom[d,1] = np.std(energy[:,index2]) / maxL
                if interList[d] == "1e-01":
                    interCut = aligntime[d]
                    MOI_same = MOI[d]
                    angmom_same = angMom[d]
                    print("angular momentum at inter cut:", angMom[d])
                    if plotline:
                        #ax[0,0].axvline(x=interCut, color='chartreuse', linestyle='solid', lw=3)
                        #ax[1,0].axvline(x=interCut, color='chartreuse', linestyle='solid', lw=3)
                        ax[0,0].axhline(y=0.93, color='k', linestyle='dashed', lw=1)
                        ax[0,1].axhline(y=0.93, color='k', linestyle='dashed', lw=1)
                        plotline = False
                if interList[d] == "1e04":
                    print("boundary:", boundary[b], "ball MOI:", MOI[d])
        ax[0,0].errorbar(aligntime[MOI[:,0]!=0], MOI[MOI[:,0]!=0,0], MOI[MOI[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        ax[1,0].errorbar(aligntime[angMom[:,0]!=0], angMom[angMom[:,0]!=0,0], angMom[angMom[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        #ax[0,0].errorbar(interCut, MOI_same[0], MOI_same[1], 
        #                 lw=0, color=starList[b], marker=markStarList[b], markersize=18, fillstyle='full', capsize=3, markeredgecolor='k', markeredgewidth=0.7)
        #ax[1,0].errorbar(interCut, angmom_same[0], angmom_same[1], 
        #                 lw=0, color=starList[b], marker=markStarList[b], markersize=18, fillstyle='full', capsize=3, markeredgecolor='k', markeredgewidth=0.7)
    # collect data vs noise strength
    for b in range(boundary.shape[0]):
        plotline = True
        MOI = np.zeros((noiseList.shape[0],2))
        angMom = np.zeros((noiseList.shape[0],2))
        for d in range(noiseList.shape[0]):
            dirSample = dirName + "j1e-01-tp" + noiseList[d] + "/dynamics-vel/" + boundary[b]
            if os.path.exists(dirSample + "/energy.dat"):
                noisetime[d] = utils.readFromDynParams(dirSample, "taup")
                energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                MOI[d,0] = np.abs(np.mean(energy[:,index1])) / maxI
                MOI[d,1] = np.std(energy[:,index1]) / maxI
                angMom[d,0] = np.abs(np.mean(energy[:,index2])) / maxL
                angMom[d,1] = np.std(energy[:,index2]) / maxL
                if noiseList[d] == "1e03":
                    noiseCut = noisetime[d]
                    MOI_same = MOI[d]
                    angmom_same = angMom[d]
                    print("angular momentum at noise cut:", angMom[d])
                    if plotline:
                        #ax[0,1].axvline(x=noiseCut, color='aqua', linestyle='solid', lw=3)
                        #ax[1,1].axvline(x=noiseCut, color='aqua', linestyle='solid', lw=3)
                        plotline = False
        ax[0,1].errorbar(noisetime[MOI[:,0]!=0], MOI[MOI[:,0]!=0,0], MOI[MOI[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        ax[1,1].errorbar(noisetime[angMom[:,0]!=0], angMom[angMom[:,0]!=0,0], angMom[angMom[:,0]!=0,1], 
                         lw=1.1, color=colorList[b], marker=markerList[b], markersize=9, fillstyle='none', capsize=3, label=labelList[b])
        #ax[0,1].errorbar(noiseCut, MOI_same[0], MOI_same[1], 
        #                 lw=0, color=starList[b], marker=markStarList[b], markersize=18, fillstyle='full', capsize=3, markeredgecolor='k', markeredgewidth=0.7)
        #ax[1,1].errorbar(noiseCut, angmom_same[0], angmom_same[1], 
        #                 lw=0, color=starList[b], marker=markStarList[b], markersize=18, fillstyle='full', capsize=3, markeredgecolor='k', markeredgewidth=0.7)
    #print("Interaction cut tau_K =", interCut, ", J_K =", 1/interCut, "K =", np.pi * 1.5**2/interCut)
    print("Noise cut tau_p =", noiseCut)
    #ax[1,0].yaxis.set_major_locator(MaxNLocator(nbins=5))
    #ax[1,1].xaxis.set_major_locator(MaxNLocator(nbins=5))
    #ax[1,0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax[0,0].set_xscale('log')
    ax[1,0].set_xscale('log')
    ax[0,1].set_xscale('log')
    ax[1,1].set_xscale('log')
    ax[1,0].xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax[1,0].xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax[1,1].xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax[1,1].xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    #ax[1,0].set_yscale('log')
    #ax[1,1].set_yscale('log')
    ax[0,1].tick_params(labelleft=False)
    ax[1,1].tick_params(labelleft=False)
    ax[1,0].tick_params(top=True)
    ax[1,1].tick_params(top=True)
    ax[0,0].tick_params(labelbottom=False, bottom=True)
    ax[0,1].tick_params(labelbottom=False, bottom=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    figureName = "/home/francesco/Pictures/soft/moments-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def compareBoundaryRoughness(dirName, figureName, type='rough', which='angmom', dynamics='/'):
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    boxRadius = np.loadtxt(dirName + "/boxSize.dat")
    f0 = float(utils.readFromDynParams(dirName, "f0"))
    gamma = float(utils.readFromDynParams(dirName, "damping"))
    maxL = boxRadius * (f0/gamma)
    maxI = boxRadius**2
    dirList = np.array(["0.1", "0.2", "0.4", "0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8"])#, "2", "2.4", "2.8", "3.2", "3.6", "4"])
    colorList = cm.get_cmap('viridis')
    index, ylabel = getIndexYlabel(which)
    roughness = np.zeros(dirList.shape[0])
    mean = np.zeros(dirList.shape[0])
    error = np.zeros(dirList.shape[0])
    krot = np.zeros(dirList.shape[0])
    krot_err = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        roughness[d] = 2 * utils.readFromWallParams(dirName + type + dirList[d], "wallRad")
        dirSample = dirName + type + dirList[d] + dynamics
        if(os.path.exists(dirSample + "/energy.dat")):
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            ax.plot(energy[::2,0], energy[::2,index], linewidth=1.2, color=colorList(d/dirList.shape[0]))
            if which == "angmom":
                energy[:,index] /= maxL
            elif which == "moi":
                energy[:,index] /= maxI
            mean[d] = np.mean(np.abs(energy[:,index]))
            error[d] = np.std(np.abs(energy[:,index]))
        if type == 'rigid':
            if(os.path.exists(dirSample + "/wallDynamics.dat")):
                angleDyn = np.loadtxt(dirSample + "wallDynamics.dat")
                angleDyn[:,3] = 0.5 * angleDyn[:,3]**2 * boxRadius **2
                krot[d] = np.mean(angleDyn[:,3])
                krot_err[d] = np.std(angleDyn[:,3])
                #ax.plot(angleDyn[:,0], angleDyn[:,3], linewidth=1.2, color=colorList(d/dirList.shape[0]))
    ax.tick_params(axis='both', labelsize=14)
    #space = 0.04*np.max(energy[:,0])
    #ax.plot(np.linspace(np.min(energy[:,0])-space, np.max(energy[:,0])+space, 100), np.zeros(100), ls='dotted', color='k', lw=0.8)
    colorBar = cm.ScalarMappable(cmap=colorList)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.)
    #cax = fig.add_subplot(gs[2])
    cbar = fig.colorbar(colorBar, cax)
    cbar.set_label('$r$', rotation='horizontal', fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=14, length=0)
    cbar.set_ticks(np.linspace(0,1,3))
    cbar.set_ticklabels(['$0.4$', '$1$', '$1.4$'])
    ax.set_xlabel("$Simulation$ $step$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/boundRough-" + which + "-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    # plot krot versus roughness
    if type == 'rigid':
        fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
        ax.errorbar(roughness, krot, krot_err, lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=14)
        ax.set_ylabel("$K_{rot}$", rotation='horizontal', labelpad=20, fontsize=18)
        ax.set_xlabel("$Roughness,$ $\\sigma_m / \\sigma$", fontsize=18)
        plt.tight_layout()
        figure2Name = "/home/francesco/Pictures/soft/krotRough-" + figureName
        fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    # plot angular momentum or pphi versus roughness
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    ax.errorbar(roughness, mean, error, lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    if which == "angmom":
        ax.set_ylabel("$\\frac{\\langle |L| \\rangle}{v_0 R}$", fontsize=24, rotation='horizontal', labelpad=30)
    elif which == "moi":
        ax.set_ylabel("$\\frac{\\langle I \\rangle}{M R^2}$", fontsize=24, rotation='horizontal', labelpad=30)
    elif which == "pphi":
        ax.set_ylabel("$\\langle |P_\\phi| \\rangle$", fontsize=18, rotation='horizontal', labelpad=30)
        ax.set_yscale('log')
    else:
        ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel("$Roughness,$ $\\sigma_m / \\sigma$", fontsize=18)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/" + which + "Rough-" + figureName
    fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def comparePhaseRoughness(dirName, figureName, versus='size', which1='moi', which2='angmom', dynamics='/'):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6.5,5.5), dpi = 120)
    alignList = np.array(["1e-01", "3", "1e02", "1e04"])
    labelList = np.array(["$7.1 \\times 10^1$", "$2.4$", "$7.1 \\times 10^{-2}$", "$7.1 \\times 10^{-4}$"])
    labelList = np.array(["$TG$", "$PDC+G$", "$PDC$", "$LC$"])
    colorList = ['forestgreen', 'dodgerblue', [0.6,0,0.8], [1,0.7,0]]
    markerList = ['v', 'o', 's', '^']
    if versus == 'size': dirList = np.array(["0", "0.1", "0.2", "0.4", "0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8"])
    else: 
        dirList = np.array(["1e-01", "1", "1e01", "1e02"])
        ew = dirList.astype(float)
    index1, ylabel1 = getIndexYlabel(which1)
    index2, ylabel2 = getIndexYlabel(which2)
    start = 1
    for a in range(alignList.shape[0]):
        obs1 = np.zeros((dirList.shape[0],2))
        obs2 = np.zeros((dirList.shape[0],2))
        dirPath = dirName + "j" + alignList[a] + "-tp1e03/dynamics-vel/"
        boxRadius = np.loadtxt(dirPath + "/boxSize.dat")
        f0 = float(utils.readFromDynParams(dirPath, "f0"))
        gamma = float(utils.readFromDynParams(dirPath, "damping"))
        maxI = boxRadius**2
        maxL = boxRadius * (f0/gamma)
        if versus == 'size':
            roughness = np.zeros(dirList.shape[0])
            for d in range(start, dirList.shape[0]):
                if dirList[d] == '0': dirSample = dirPath + "fixed" + dynamics
                else: dirSample = dirPath + "rough" + dirList[d] + dynamics
                if os.path.exists(dirSample):
                    if dirList[d] != '0': roughness[d] = 2 * utils.readFromWallParams(dirPath + "rough" + dirList[d], "wallRad")
                    if(os.path.exists(dirSample + "/energy.dat")):
                        energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                        if which1 == "moi" and which2 == "angmom":
                            energy[:,index1] /= maxI
                            energy[:,index2] /= maxL
                        obs1[d,0] = np.mean(np.abs(energy[:,index1]))
                        obs1[d,1] = np.std(np.abs(energy[:,index1]))
                        obs2[d,0] = np.mean(np.abs(energy[:,index2]))
                        obs2[d,1] = np.std(np.abs(energy[:,index2]))
            # get color from value of velpos parameter
            #obs1[1:] = np.column_stack((utils.computeMovingAverage(obs1[1:,0], 2), utils.computeMovingAverage(obs1[1:,1], 2)))
            #obs2[1:] = np.column_stack((utils.computeMovingAverage(obs2[1:,0], 2), utils.computeMovingAverage(obs2[1:,1], 2)))
            if which1 == "moi": colorId = 1 - (np.mean(energy[:,-2]) - 0.7) / 0.3
            else: colorId = 1 - (np.mean(energy[:,-2]) / maxI - 0.7) / 0.3
            print("color index:", colorId)
            ax[0].errorbar(roughness[1:], obs1[1:,0], obs1[1:,1], lw=1.2, color=colorList[a], marker=markerList[a], 
                        markersize=8, fillstyle='none', capsize=3, label="$\\tau_K \\approx$" + labelList[a])
            ax[1].errorbar(roughness[1:], obs2[1:,0], obs2[1:,1], lw=1.2, color=colorList[a], marker=markerList[a], 
                        markersize=8, fillstyle='none', capsize=3, label="$\\tau_K \\approx$" + labelList[a])
            if start == 0:
                ax[0].errorbar(roughness[0], obs1[0,0], obs1[0,1], lw=1.2, color=colorList[a], marker=markerList[a], markersize=8, fillstyle='none', capsize=3)
                ax[1].errorbar(roughness[0], obs2[0,0], obs2[0,1], lw=1.2, color=colorList[a], marker=markerList[a], markersize=8, fillstyle='none', capsize=3)
        else:
            for d in range(dirList.shape[0]):
                if dirList[d] == "1e01": dirSample = dirPath + "fixed" + dynamics
                else: dirSample = dirPath + "fixed-ew" + dirList[d] + dynamics
                if os.path.exists(dirSample):
                    if(os.path.exists(dirSample + "/energy.dat")):
                        energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                        if which1 == "moi" and which2 == "angmom":
                            energy[:,index2] /= maxI
                            energy[:,index1] /= maxL
                        obs1[d,0] = np.mean(np.abs(energy[:,index1]))
                        obs1[d,1] = np.std(np.abs(energy[:,index1]))
                        obs2[d,0] = np.mean(np.abs(energy[:,index2]))
                        obs2[d,1] = np.std(np.abs(energy[:,index2]))
            # get color from value of velpos parameter
            if which1 == "moi": colorId = 1 - (np.mean(energy[:,-2]) - 0.7) / 0.3
            else: colorId = 1 - (np.mean(energy[:,-2]) / maxI - 0.7) / 0.3
            print("color index:", colorId)
            ax[0].errorbar(ew, obs1[:,0], obs1[:,1], lw=1.2, color=colorList(colorId), marker=markerList[a], 
                        markersize=8, fillstyle='none', capsize=3, label="$\\tau_K \\approx$" + labelList[a])
            ax[1].errorbar(ew, obs2[:,0], obs2[:,1], lw=1.2, color=colorList(colorId), marker=markerList[a], 
                        markersize=8, fillstyle='none', capsize=3, label="$\\tau_K \\approx$" + labelList[a])
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    if which1 == "moi" and which2 == "angmom":
        ax[0].set_ylabel("$\\frac{\\langle I \\rangle}{M R^2}$", fontsize=24, rotation='horizontal', labelpad=25)
        ax[1].set_ylabel("$\\frac{\\langle |L| \\rangle}{M v_0 R}$", fontsize=24, rotation='horizontal', labelpad=30)
        ax[0].set_ylabel("$\\tilde{ I }$", fontsize=18, rotation='horizontal', labelpad=25)
        ax[1].set_ylabel("$\\tilde{ L }$", fontsize=18, rotation='horizontal', labelpad=30)
        ax[0].set_ylim(0.47,1.06)
        ax[1].set_ylim(-0.12,1.12)
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax[0].legend(loc='lower right', fontsize=14, ncols=2)
    elif which1 == "pphi":
        ax[0].set_ylabel("$|P_\\phi|$", fontsize=18, rotation='horizontal', labelpad=30)
        ax[1].set_ylabel(ylabel2, fontsize=18, rotation='horizontal', labelpad=30)
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].legend(loc='lower right', fontsize=12, ncols=2)
    else:
        ax[0].set_ylabel(ylabel1, fontsize=18, rotation='horizontal', labelpad=30)
        ax[1].set_ylabel(ylabel2, fontsize=18, rotation='horizontal', labelpad=30)
        ax[1].legend(loc='lower right', fontsize=12, ncols=2)
    if versus == 'size': ax[1].set_xlabel("$Roughness,$ $\\sigma_m / \\sigma$", fontsize=16)
    else: 
        ax[1].set_xlabel("$Relative$ $strength,$ $\\epsilon_w / \\epsilon$", fontsize=16)
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
    ax[1].tick_params(top=True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figure1Name = "/home/francesco/Pictures/soft/phaseRough-" + which1 + "-" + which2 + "-" + figureName
    fig.savefig(figure1Name + ".png", transparent=False, format = "png")
    plt.show()

def compareAngMomRoughness(dirName, figureName, dynamics='/'):
    fig, ax = plt.subplots(figsize=(7.2,2.8), dpi = 200)
    alignList = np.array(["1e-01", "3", "1e02", "1e04"])
    labelList = np.array(["$7.1 \\times 10^1$", "$2.4$", "$7.1 \\times 10^{-2}$", "$7.1 \\times 10^{-4}$"])
    labelList = np.array(["$TG$", "$PDC+G$", "$PDC$", "$LC$"])
    colorList = ['forestgreen', 'dodgerblue', [0.6,0,0.8], [1,0.7,0]]
    markerList = ['v', 'o', 's', '^']
    dirList = np.array(["0.1", "0.2", "0.4", "0.6", "0.8", "1", "1.2", "1.4", "1.6", "1.8"])
    index, ylabel = getIndexYlabel('angmom')
    for a in range(alignList.shape[0]):
        obs = np.zeros((dirList.shape[0],2))
        dirPath = dirName + "j" + alignList[a] + "-tp1e03/dynamics-vel/"
        boxRadius = np.loadtxt(dirPath + "/boxSize.dat")
        f0 = float(utils.readFromDynParams(dirPath, "f0"))
        gamma = float(utils.readFromDynParams(dirPath, "damping"))
        maxL = boxRadius * (f0/gamma)
        roughness = np.zeros(dirList.shape[0])
        for d in range(dirList.shape[0]):
            dirSample = dirPath + "rough" + dirList[d] + dynamics
            if os.path.exists(dirSample):
                roughness[d] = 2 * utils.readFromWallParams(dirPath + "rough" + dirList[d], "wallRad")
                if(os.path.exists(dirSample + "/energy.dat")):
                    energy = np.loadtxt(dirSample + os.sep + "energy.dat")
                    energy[:,index] /= maxL
                    obs[d,0] = np.mean(np.abs(energy[:,index]))
                    obs[d,1] = np.std(np.abs(energy[:,index]))
        obs = np.column_stack((utils.computeMovingAverage(obs[:,0], 2), utils.computeMovingAverage(obs[:,1], 2)))
        ax.errorbar(roughness, obs[:,0], obs[:,1], lw=1.2, color=colorList[a], marker=markerList[a], 
                    markersize=10, fillstyle='none', capsize=3, label=labelList[a])#label="$\\tau_K \\approx$" + labelList[a])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$\\langle \\tilde{L} \\rangle$", fontsize=18, rotation='horizontal', labelpad=20)
    ax.set_xlabel("$Roughness,$ $\\sigma_m / \\sigma$", fontsize=18)
    ax.set_ylim(-0.12,1.42)
    ax.legend(loc='upper right', fontsize=14, ncols=4, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figure1Name = "/home/francesco/Pictures/soft/angmomRough-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    plt.show()

def plotOrderParams(dirName, figureName, dynamics='reflect', cluster=False, maxCluster=10): # compute cluster is there are less then maxCluster
    fig, ax = plt.subplots(1, 2, figsize=(11.7,4), dpi = 120)
    alignList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01",
                        "1", "2", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    noiseList = np.array(["1e-03", "3e-02", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
    tauk = np.zeros(alignList.shape[0])
    taup = np.zeros(noiseList.shape[0])
    # First make plot vs alignment time
    corr1 = np.zeros((alignList.shape[0], 2))
    corr2 = np.zeros((alignList.shape[0], 2))
    corr3 = np.zeros((alignList.shape[0], 2))
    boxRadius = np.loadtxt(dirName + "j1e-03-tp1e03/dynamics-vel/boxSize.dat")
    maxI = boxRadius**2
    f0 = float(utils.readFromDynParams(dirName + "j1e-03-tp1e03/dynamics-vel/", "f0"))
    gamma = float(utils.readFromDynParams(dirName + "j1e-03-tp1e03/dynamics-vel/", "damping"))
    maxL = boxRadius * (f0/gamma)
    if figureName[:5] == 'rough':
        color1 = 'forestgreen'#[0,0.6,0.4]
        color2 = 'dodgerblue'#[0.5,0.5,1]
        color3 = [0.6,0,0.8]
    else:
        color1 = 'darkgrey'#[0.6,0.6,0.6]
        color2 = [1,0,1]
        color3 = [1,0.7,0]
    for d in range(alignList.shape[0]):
        dirSample = dirName + "j" + alignList[d] + "-tp1e03/dynamics-vel/" + dynamics + "/"
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            tauk[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
            if cluster == 'cluster':
                if(os.path.exists(dirSample + "/t0/")):
                    #computed = False
                    if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                        computeMaxClusterKuramoto(dirSample, eps=3, maxCluster=maxCluster)
                    #    computed = True
                    corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                    #if computed:
                    #    print(dirList[d], tauk[d], taup[d], "computed cluster kuramoto", np.mean(corrCluster))
                    corr1[d,0] = np.mean(corrCluster)
                    corr1[d,1] = np.std(corrCluster)
                else:
                    corr1[d,0] = np.mean(data[:,6])
                    corr1[d,1] = np.std(data[:,6])
            else:
                corr1[d,0] = np.mean(data[:,6])
                corr1[d,1] = np.std(data[:,6])
            #corr2[d,0] = np.mean(data[:,8])
            #corr2[d,1] = np.std(data[:,8])
            corr2[d,0] = np.mean(np.abs(data[:,-1]) / maxL)
            corr2[d,1] = np.std(np.abs(data[:,-1]) / maxL)
            corr3[d,0] = np.mean(data[:,-2] / maxI)
            corr3[d,1] = np.std(data[:,-2] / maxI)
            if figureName[:5] == 'rough':
                if alignList[d] == "1e-01":
                    point1 = np.array([tauk[d], corr1[d,0], corr2[d,0], corr3[d,0]])
                elif alignList[d] == "3":
                    point2 = np.array([tauk[d], corr1[d,0], corr2[d,0], corr3[d,0]])
                elif alignList[d] == "1e02":
                    point3 = np.array([tauk[d], corr1[d,0], corr2[d,0], corr3[d,0]])
            else:
                if alignList[d] == "1e-01":
                    point1 = np.array([tauk[d], corr1[d,0], corr2[d,0], corr3[d,0]])
                elif alignList[d] == "3":
                    point2 = np.array([tauk[d], corr1[d,0], corr2[d,0], corr3[d,0]])
                elif alignList[d] == "1e04":
                    point3 = np.array([tauk[d], corr1[d,0], corr2[d,0], corr3[d,0]])
            #print(alignList[d], tauk[d], "phi_r:", corr1[d,0], "phi_alpha:", corr2[d,0])
    ax[0].errorbar(tauk[tauk!=0], corr1[tauk!=0,0], corr1[tauk!=0,1], color='k', marker='o', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\phi_r^C$")
    ax[0].errorbar(tauk[tauk!=0], corr2[tauk!=0,0], corr2[tauk!=0,1], color='b', marker='s', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\tilde{L}$")
    ax[0].errorbar(tauk[tauk!=0], corr3[tauk!=0,0], corr3[tauk!=0,1], color='g', marker='^', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\tilde{I}$")
    ax[0].plot(point1[0], point1[1], color=color1, marker='o', markersize=14, fillstyle='full')
    ax[0].plot(point1[0], point1[2], color=color1, marker='s', markersize=14, fillstyle='full')
    ax[0].plot(point2[0], point2[1], color=color2, marker='o', markersize=14, fillstyle='full')
    ax[0].plot(point2[0], point2[2], color=color2, marker='s', markersize=14, fillstyle='full')
    ax[0].plot(point3[0], point3[1], color=color3, marker='o', markersize=14, fillstyle='full')
    ax[0].plot(point3[0], point3[2], color=color3, marker='s', markersize=14, fillstyle='full')
    ax[0].plot(point1[0], point1[3], color=color1, marker='^', markersize=14, fillstyle='full')
    ax[0].plot(point2[0], point2[3], color=color2, marker='^', markersize=14, fillstyle='full')
    ax[0].plot(point3[0], point3[3], color=color3, marker='^', markersize=14, fillstyle='full')
    ax[0].legend(loc='best', fontsize=16, frameon=False)
    # Second make plot vs noise time
    corr1 = np.zeros((noiseList.shape[0], 2))
    corr2 = np.zeros((noiseList.shape[0], 2))
    corr3 = np.zeros((noiseList.shape[0], 2))
    for d in range(noiseList.shape[0]):
        dirSample = dirName + "j1e-01-tp" + noiseList[d] + "/dynamics-vel/" + dynamics + "/"
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            taup[d] = utils.readFromDynParams(dirSample, "taup")
            if cluster == 'cluster':
                if(os.path.exists(dirSample + "/t0/")):
                    #computed = False
                    if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                        computeMaxClusterKuramoto(dirSample, eps=3, maxCluster=maxCluster)
                    #    computed = True
                    corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                    #if computed:
                    #    print(dirList[d], tauk[d], taup[d], "computed cluster kuramoto", np.mean(corrCluster))
                    corr1[d,0] = np.mean(corrCluster)
                    corr1[d,1] = np.std(corrCluster)
                else:
                    corr1[d,0] = np.mean(data[:,6])
                    corr1[d,1] = np.std(data[:,6])
            else:
                corr1[d,0] = np.mean(data[:,6])
                corr1[d,1] = np.std(data[:,6])
            #corr2[d,0] = np.mean(data[:,8])
            #corr2[d,1] = np.std(data[:,8])
            corr2[d,0] = np.mean(np.abs(data[:,-1]) / maxL)
            corr2[d,1] = np.std(np.abs(data[:,-1]) / maxL)
            corr3[d,0] = np.mean(data[:,-2] / maxI)
            corr3[d,1] = np.std(data[:,-2] / maxI)
            if noiseList[d] == "1e03": point1 = np.array([taup[d], corr1[d,0], corr2[d,0], corr3[d,0]])
            #print(noiseList[d], taup[d], "phi_r:", corr1[d,0], "phi_alpha:", corr2[d,0])
    ax[1].errorbar(taup[taup!=0], corr1[taup!=0,0], corr1[taup!=0,1], color='k', marker='o', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\phi_r^C$")
    ax[1].errorbar(taup[taup!=0], corr2[taup!=0,0], corr2[taup!=0,1], color='b', marker='s', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\tilde{L}$")
    ax[1].errorbar(taup[taup!=0], corr3[taup!=0,0], corr3[taup!=0,1], color='green', marker='^', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\tilde{I}$")
    ax[1].plot(point1[0], point1[1], color=color1, marker='o', markersize=14, fillstyle='full')
    ax[1].plot(point1[0], point1[2], color=color1, marker='s', markersize=14, fillstyle='full')
    ax[1].plot(point1[0], point1[3], color=color1, marker='^', markersize=14, fillstyle='full')
    ax[1].legend(loc='best', fontsize=16, frameon=False)
    for i in range(2):
        ax[i].set_xscale('log')
        ax[i].set_ylim(-0.08, 1.08)
        ax[i].tick_params(axis='both', labelsize=16)
    ax[0].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=20)
    ax[1].set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    if cluster == 'cluster':
        figureName = "/home/francesco/Pictures/soft/orderParamsCluster-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/orderParams-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotMoments(dirName, figureName, which='inter', dynamics='reflect'): # for maximum number of clusters to consider
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    if which == 'inter':
        dirList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01",
                            "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
        xlabel = "$Alignment$ $time,$ $\\tau_K$"
    else:
        dirList = np.array(["1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    tauk = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    corr1 = np.zeros((dirList.shape[0], 2))
    corr2 = np.zeros((dirList.shape[0], 2))
    dirPath = dirName + "j1e-01-tp1e03/dynamics-vel/"
    boxSize = np.atleast_1d(np.loadtxt(dirName + "j1e03-tp1e03/boxSize.dat"))
    maxI = boxSize**2
    f0 = float(utils.readFromDynParams(dirPath, "f0"))
    gamma = float(utils.readFromDynParams(dirPath, "damping"))
    maxL = boxSize * (f0/gamma)
    for d in range(dirList.shape[0]):
        if which == 'inter': dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/" + dynamics + "/"
        else: dirSample = dirName + "j1e-01-tp" + dirList[d] + "/dynamics-vel/" + dynamics + "/"
        #print(dirSample)
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            tauk[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
            taup[d] = utils.readFromDynParams(dirSample, "taup")
            corr1[d,0] = np.mean(data[:,-2]/maxI)
            corr1[d,1] = np.std(data[:,-2]/maxI)
            corr2[d,0] = np.mean(np.abs(data[:,-1])/maxL)
            corr2[d,1] = np.std(np.abs(data[:,-1])/maxL)
    if which == 'inter':
        x = tauk
    else:
        x = taup
    ax.errorbar(x[x!=0], corr1[x!=0,0], corr1[x!=0,1], color='k', marker='o', markersize=10, capsize=3, fillstyle='none', lw=1, 
                label="$\\langle I \\rangle / M R^2$")
    ax.errorbar(x[x!=0], corr2[x!=0,0], corr2[x!=0,1], color='b', marker='s', markersize=10, capsize=3, fillstyle='none', lw=1, 
                label="$\\langle |L| \\rangle / M v_0 R$")
    ax.legend(loc='best', fontsize=16)
    ax.set_xscale('log')
    ax.set_ylim(-0.08, 1.08)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(xlabel, fontsize=20)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/moments-" + figureName + "-vs" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotEnergies(dirName, figureName, which='inter', dynamics='reflect'): # for maximum number of clusters to consider
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    if which == 'inter':
        dirList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01",
                            "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
        xlabel = "$Alignment$ $time,$ $\\tau_K$"
    else:
        dirList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    tauk = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    corr1 = np.zeros((dirList.shape[0], 2))
    corr2 = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        if which == 'inter': dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/" + dynamics + "/"
        else: dirSample = dirName + "j1e-01-tp" + dirList[d] + "/dynamics-vel/" + dynamics + "/"
        #print(dirSample)
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            tauk[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
            taup[d] = utils.readFromDynParams(dirSample, "taup")
            corr1[d,0] = np.mean(data[:,2])
            corr1[d,1] = np.std(data[:,2])
            corr2[d,0] = np.mean(data[:,3])
            corr2[d,1] = np.std(data[:,3])
    if which == 'inter':
        x = tauk
    else:
        x = taup
    ax.errorbar(x[x!=0], corr1[x!=0,0], corr1[x!=0,1], color='k', marker='o', markersize=10, capsize=3, fillstyle='none', lw=1, 
                label="$U / N$")
    ax.errorbar(x[x!=0], corr2[x!=0,0], corr2[x!=0,1], color='b', marker='s', markersize=10, capsize=3, fillstyle='none', lw=1, 
                label="$K / N$")
    ax.legend(loc='best', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(xlabel, fontsize=20)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/energies-" + figureName + "-vs" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotMomentAndOrderParams(dirName, figureName, which='inter', dynamics='reflect', cluster=False, maxCluster=32): # for maximum number of clusters to consider
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    if which == 'inter':
        dirList = np.array(["1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01",
                            "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
        xlabel = "$Alignment$ $time,$ $\\tau_K$"
    else:
        dirList = np.array(["1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08"])
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    tauk = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    corr1 = np.zeros((dirList.shape[0], 2))
    corr2 = np.zeros((dirList.shape[0], 2))
    corr3 = np.zeros((dirList.shape[0], 2))
    dirPath = dirName + "j1e-01-tp1e03/dynamics-vel/"
    boxRadius = np.loadtxt(dirPath + "/boxSize.dat")
    maxI = boxRadius**2
    for d in range(dirList.shape[0]):
        if which == 'inter': dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/" + dynamics + "/"
        else: dirSample = dirName + "j1e-01-tp" + dirList[d] + "/dynamics-vel/" + dynamics + "/"
        #print(dirSample)
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "energy.dat")
            tauk[d] = 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"])
            taup[d] = utils.readFromDynParams(dirSample, "taup")
            if cluster == 'cluster':
                if(os.path.exists(dirSample + "/t0/")):
                    computed = False
                    if not(os.path.exists(dirSample + "/clusterKuramoto.dat")):
                        computeMaxClusterKuramoto(dirSample, eps=1.5, maxCluster=maxCluster)
                        computed = True
                    corrCluster = np.loadtxt(dirSample + "/clusterKuramoto.dat")[:,1]
                    if computed:
                        print(dirList[d], tauk[d], taup[d], "computed cluster kuramoto", np.mean(corrCluster))
                    corr1[d,0] = np.mean(corrCluster)
                    corr1[d,1] = np.std(corrCluster)
                else:
                    corr1[d,0] = np.mean(data[:,6])
                    corr1[d,1] = np.std(data[:,6])
            else:
                corr1[d,0] = np.mean(data[:,6])
                corr1[d,1] = np.std(data[:,6])
            corr2[d,0] = np.mean(data[:,8])
            corr2[d,1] = np.std(data[:,8])
            corr3[d,0] = np.mean(data[:,-2]/maxI)
            corr3[d,1] = np.std(data[:,-2]/maxI)
    if which == 'inter':
        x = tauk
    else:
        x = taup
    ax.errorbar(x[x!=0], corr1[x!=0,0], corr1[x!=0,1], color='k', marker='o', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\phi_r^C$")
    ax.errorbar(x[x!=0], corr2[x!=0,0], corr2[x!=0,1], color='b', marker='s', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\phi_\\alpha$")
    ax.errorbar(x[x!=0], corr3[x!=0,0], corr3[x!=0,1], color='g', marker='v', markersize=10, capsize=3, fillstyle='none', lw=1, label="$\\tilde{I}$")
    ax.legend(loc='best', fontsize=16)
    ax.set_xscale('log')
    ax.set_ylim(-0.08, 1.08)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(xlabel, fontsize=20)
    plt.tight_layout()
    if cluster == 'cluster':
        figureName = "/home/francesco/Pictures/soft/paramsMomCluster-" + figureName + "-vs" + which
    else:
        figureName = "/home/francesco/Pictures/soft/paramsMom-" + figureName + "-vs" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotPhaseVSReflection(dirName, figureName, which='moms', dynamics='/'):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(5.5,6.5), dpi = 120)
    dirList = np.array(["reflect-rp1", "reflect-rp0.999", "reflect-rp0.99", "reflect-rp0.95", "reflect-rp0.9", "reflect-rp0.8", "reflect-rp0.7", "reflect-rp0.6", "fixed"])
    labelList = np.array(["$r=1$", "$r=0.999$", "$r=0.99$", "$r=0.95$", "$r=0.9$", "$r=0.8$", "$r=0.7$", "$r=0.6$", "$WCA$"])
    restparam = np.array([0.999, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6])
    colorList = cm.get_cmap('plasma')
    corr1 = np.zeros((dirList.shape[0],2))
    corr2 = np.zeros((dirList.shape[0],2))
    boxRadius = np.atleast_1d(np.loadtxt(dirName + "reflect/damping1e02/j1e-04-tp1e03/boxSize.dat"))
    f0 = float(utils.readFromDynParams(dirName + "reflect/damping1e02/j1e-04-tp1e03/", "f0"))
    gamma = float(utils.readFromDynParams(dirName + "reflect/damping1e02/j1e-04-tp1e03/", "damping"))
    if which == 'moms':
        maxI = boxRadius**2
        maxL = boxRadius * (f0/gamma)
        ylabel1 = "$\\frac{I}{MR^2}$"
        ylabel2 = "$\\frac{|L|}{Mv_0R}$"
        label1 = "$I / MR^2$"
        label2 = "$|L| / Mv_0R$"
        index1 = -2
        index2 = -1
    elif which == 'energy':
        ylabel1 = "$\\frac{U}{N}$"
        ylabel2 = "$\\frac{K}{N}$"
        label1 = "$U/N$"
        label2 = "$K/N$"
        index1 = 2
        index2 = 3
    else:
        numParticles = float(utils.readFromParams(dirName + "reflect/damping1e02/j1e-04-tp1e03/", "numParticles"))
        pIdeal = 0.5 * (numParticles / (np.pi * boxRadius**2)) * (f0/gamma)**2
        ylabel1 = "$\\frac{P_r}{P_0}$"
        ylabel2 = "$\\frac{|P_\\phi|}{P_0}$"
        label1 = "$P_r / P_0$"
        label2 = "$P_\\phi / P_0$"
        index1 = 4
        index2 = 5
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d] + "/damping1e02/j1e-04-tp1e03" + dynamics
        if d == dirList.shape[0] - 1: color = 'k'
        else: color = colorList((dirList.shape[0] - 1 - d)/dirList.shape[0])
        if(os.path.exists(dirSample + "/energy.dat")):
            data = np.loadtxt(dirSample + os.sep + "energy.dat")
            if which == 'moms':
                corr1[d,0] = np.mean(data[:,-2]/maxI)
                corr1[d,1] = np.std(data[:,-2]/maxI)
                corr2[d,0] = np.mean(np.abs(data[:,-1])/maxL)
                corr2[d,1] = np.std(np.abs(data[:,-1])/maxL)
                data[:,-2] = data[:,-2]/maxI
                data[:,-1] = np.abs(data[:,-1])/maxL
            elif which == 'energy':
                corr1[d,0] = np.mean(data[:,2])
                corr1[d,1] = np.std(data[:,2])
                corr2[d,0] = np.mean(data[:,3])
                corr2[d,1] = np.std(data[:,3])
            else:
                corr1[d,0] = np.mean(data[:,4]/pIdeal)
                corr1[d,1] = np.std(data[:,4]/pIdeal)
                corr2[d,0] = np.mean(np.abs(data[:,5])/pIdeal)
                corr2[d,1] = np.std(np.abs(data[:,5])/pIdeal)
                data[:,4] = utils.computeMovingAverage(data[:,4], window=10)
                data[:,5] = utils.computeMovingAverage(data[:,5], window=10)
            ax[0].plot(data[:,0], data[:,index1], linewidth=1, color=color, label=labelList[d])
            ax[1].plot(data[:,0], data[:,index2], linewidth=1, color=color, label=labelList[d])
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax[1].legend(fontsize=12, ncols=2, labelcolor='linecolor')
    ax[1].set_xlabel("$Time,$ $t$", fontsize=16)
    if which == 'moms':
        ax[0].set_ylim(0.388, 1.022)
        ax[1].set_ylim(-0.022, 1.022)
        ax[0].set_ylabel(ylabel1, fontsize=22, rotation='horizontal', labelpad=25)
        ax[1].set_ylabel(ylabel2, fontsize=22, rotation='horizontal', labelpad=25)
    else:
        ax[0].set_ylabel(ylabel1, fontsize=22, rotation='horizontal', labelpad=20)
        ax[1].set_ylabel(ylabel2, fontsize=22, rotation='horizontal', labelpad=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    figure1Name = "/home/francesco/Pictures/soft/momsipl-" + figureName
    fig.savefig(figure1Name + ".png", transparent=False, format = "png")
    # plot average moments vs softness
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 4]}, dpi=120, figsize=(5.2,3.8))
    # left: linear near zero
    ax1.errorbar(0, corr1[0,0], corr1[0,1], color='b', marker='o', markersize=8, lw=1, capsize=3, fillstyle='none')
    ax1.errorbar(0, corr2[0,0], corr2[0,1], color='g', marker='s', markersize=8, lw=1, capsize=3, fillstyle='none')
    # right: log part
    corr1 = corr1[1:-1]
    corr2 = corr2[1:-1]
    ax2.errorbar(1-restparam[corr1[:,0]!=0], corr1[corr1[:,0]!=0,0], corr1[corr1[:,0]!=0,1], 
                color='b', marker='o', markersize=8, lw=1, label=label1, capsize=3, fillstyle='none')
    ax2.errorbar(1-restparam[corr2[:,0]!=0], corr2[corr2[:,0]!=0,0], corr2[corr2[:,0]!=0,1], 
                color='g', marker='s', markersize=8, lw=1, label=label2, capsize=3, fillstyle='none')
    ax1.set_xticks([0])
    if which == 'moms':
        ax1.set_yticks([0, 0.5, 1])
        ax2.legend(fontsize=12, loc='upper left')
    else:
        ax1.set_yscale('log')
        if which == 'pres': ax2.legend(fontsize=12, loc='best')
        else: ax2.legend(fontsize=12, loc='upper left')
    ax2.set_xscale('log')
    # hide spines between axes
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # add diagonal break marks
    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel("$1-r$", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    figure2Name = "/home/francesco/Pictures/soft/momsrest-" + figureName
    fig.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def plotPinnedBoundary(dirName, figureName, which='friction', dynamics='/'):
    if which == 'friction' or which == 'omega':
        fig, ax = plt.subplots(figsize=(5.5,4), dpi = 120)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10,4), dpi = 120)
    boxRadius = np.loadtxt(dirName + "j1e03-tp1e03/dynamics-vel/boxSize.dat")
    dirList = np.array(["1e-04", "3e-04", "1e-03", "3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    aligntime = np.zeros(dirList.shape[0])
    if which == 'friction':
        eta = np.zeros(dirList.shape[0])
        eta_err = np.zeros(dirList.shape[0])
    elif which == 'omega':
        omega = np.zeros(dirList.shape[0])
        omega_err = np.zeros(dirList.shape[0])
        pphi = np.zeros(dirList.shape[0])
        pphi_err = np.zeros(dirList.shape[0])
    else:
        corr = np.zeros(dirList.shape[0])
        corr_err = np.zeros(dirList.shape[0])
        krot = np.zeros(dirList.shape[0])
        krot_err = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        aligntime[d] = 1/utils.readFromDynParams(dirName + "j" + dirList[d] + "-tp1e03", "Jvicsek")
        dirSample = dirName + "j" + dirList[d] + "-tp1e03/dynamics-vel/rigid" + dynamics
        if(os.path.exists(dirSample + "/energy.dat") and os.path.exists(dirSample + "/wallDynamics.dat")):
            energy = np.loadtxt(dirSample + os.sep + "energy.dat")
            angleDyn = np.loadtxt(dirSample + "wallDynamics.dat")
            if which == 'friction':
                eta[d] = np.mean(energy[:,5] * boxRadius / angleDyn[:,3])
                eta_err[d] = np.std(energy[:,5] * boxRadius / angleDyn[:,3])
            elif which == 'omega':
                omega[d] = np.mean(angleDyn[:,3])
                omega_err[d] = np.std(angleDyn[:,3])
                pphi[d] = np.mean(energy[:,5] * boxRadius)
                pphi_err[d] = np.std(energy[:,5] * boxRadius)
            else:
                corr[d] = np.mean(energy[:,5])
                corr_err[d] = np.std(energy[:,5])
                angleDyn[:,3] = 0.5 * angleDyn[:,3]**2 * boxRadius**2
                krot[d] = np.mean(angleDyn[:,3])
                krot_err[d] = np.std(angleDyn[:,3])
    if which == 'friction':
        ax.errorbar(aligntime, eta, eta_err, lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
        ax.set_ylabel("$Rotational$ $friction,$ $\\zeta_{eff}$", fontsize=16)
        ax.set_xscale('log')
    elif which == 'omega':
        ax.errorbar(np.abs(pphi), np.abs(omega), omega_err, lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax.tick_params(axis='both', labelsize=14)
        #ax.set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
        ax.set_xlabel("$Wall$ $torque,$ $|T_\\phi|$", fontsize=16)
        ax.set_ylabel("$Angular$ $velocity,$ $|\\omega|$", fontsize=16)
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax[0].errorbar(aligntime, np.abs(corr), corr_err, lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax[1].errorbar(aligntime, krot, krot_err, lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax[0].tick_params(axis='both', labelsize=14)
        ax[1].tick_params(axis='both', labelsize=14)
        ax[1].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
        ax[0].set_xlabel("$Alignment$ $time,$ $\\tau_K$", fontsize=16)
        ax[1].set_ylabel("$K_{rot}$", fontsize=16, rotation='horizontal', labelpad=15)
        ax[0].set_ylabel("$|P_\\phi|$", fontsize=16, rotation='horizontal', labelpad=15)
        ax[0].set_xscale('log')
        ax[1].set_xscale('log')
        ax[0].yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[1].yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/soft/pinned-" + figureName + "-" + which
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def phaseDiagramWallDynamics(dirName, which="omega", dynamics="/", interpolate=False):
    fig, ax = plt.subplots(figsize=(6.5,5), dpi = 120)
    aligntime = np.array([])
    noisetime = np.array([])
    corr = np.array([])
    index, cbar_label = getIndexYlabel(which)
    if which == 'krot':
        index = 3
        cbar_label = "$\\tilde{K}_{rot}$"
    elif which == 'omega':
        index = 3
        cbar_label = "$|\\omega|$"
    elif which == 'tau':
        index = 5
        cbar_label = "$|\\tau_{wall}|$"
    elif which == 'tauactive':
        index = 6
        cbar_label = "$|\\tau_{active}|$"
    f0 = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "f0"))
    gamma = float(utils.readFromDynParams(dirName + "j1e03-tp1e03", "damping"))
    maxK = (f0/gamma)**2 / 2
    # get color map for each cut of the phase diagram
    noiseList = np.array(["1e-04", "1e-03", "1e-02", "1e-01", "1", "1e01", "1e02", "1e03", "1e04", "1e05", "1e06", "1e07", "1e08", "0"])
    alignList = np.array(["3e-03", "1e-02", "3e-02", "1e-01", "3e-01", "1", "3", "1e01", "3e01", "1e02", "3e02", "1e03", "3e03", "1e04"])
    for i in range(noiseList.shape[0]):
        for d in range(alignList.shape[0]):
            dirSample = dirName + "j" + alignList[d] + "-tp" + noiseList[i] + "/dynamics-vel/rigid" + dynamics
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "wallDynamics.dat")
                if which == 'krot':
                    data[:,index] = 0.5 * data[:,index]**2
                elif which == 'omega':
                    data[:,index] = np.abs(data[:,index])
                elif which == 'tau' or which == 'tauactive':
                    data[:,index] = np.abs(data[:,index]) / maxK
                corr = np.append(corr, np.mean(data[:,index]))
                aligntime = np.append(aligntime, 1 / utils.readFromDynParams(dirSample, ["Jk", "Jvicsek"]))
                print(alignList[d], noiseList[i], corr[-1])
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
        if vmin < 0: vmin = 0
        vmax = np.max(corr)
        sc = plt.scatter(noisetime, aligntime, c=corr, cmap='plasma', s=200, edgecolors='k', marker='s', linewidths=0.5, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(sc, ax=ax, pad=0, aspect=20)
    cbar.set_label(cbar_label, rotation='horizontal', fontsize=22, labelpad=20)
    cbar.ax.tick_params(labelsize=14, length=0)
    min = np.min(corr)
    max = np.max(corr)
    cbar.set_ticks(np.linspace(min,max,5))
    #cbar.set_ticks(np.linspace(0,1,5))
    if which == "vcorr" or which == "velpos":
        cbar.set_ticklabels(["$0.00$", "$0.25$", "$0.50$", "$0.75$", "$1.00$"])
    # Set log scales for proper visualization
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=22)
    ax.set_ylabel("$Alignment$ $time,$ $\\tau_K$", fontsize=22)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/wallDiagram-" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()



if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

    if(whichPlot == "energy"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotEnergyFile(dirName, figureName, which)
    
    elif(whichPlot == "compare"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        boundary1 = sys.argv[5]
        boundary2 = sys.argv[6]
        dynamics = sys.argv[7]
        log = sys.argv[8]
        p1 = int(sys.argv[9])
        p2 = int(sys.argv[10])
        compareAlignment(dirName, figureName, which, boundary1, boundary2, dynamics, log, p1, p2)

    elif(whichPlot == "boundary"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareBoundary(dirName, figureName, which, dynamics)

    elif(whichPlot == "boundalign"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareBoundaryAlign(dirName, figureName, which, dynamics)

    elif(whichPlot == "angle"):
        figureName = sys.argv[3]
        log = sys.argv[4]
        plotAngleDistance(dirName, figureName, log)

    elif(whichPlot == "compareangle"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        which = sys.argv[5]
        log = sys.argv[6]
        compareAngleDistance(dirName, figureName, dynamics, which, log)

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

    elif(whichPlot == "noiseboundary"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareAlignmentVSBoundary(dirName, figureName, which, dynamics)

    elif(whichPlot == "3diagrams"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        cluster = sys.argv[5]
        if cluster == 'cluster':
            cluster = True
        else:
            cluster = False
        maxCluster = int(sys.argv[6])
        interpolate = sys.argv[7]
        phaseDiagrams3(dirName, figureName, dynamics, cluster, maxCluster, interpolate)

    elif(whichPlot == "2diagrams"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        obs = sys.argv[5]
        cluster = sys.argv[6]
        maxCluster = int(sys.argv[7])
        interpolate = sys.argv[8]
        phaseDiagrams2(dirName, figureName, dynamics, obs, cluster, maxCluster, interpolate)

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
        plotWallDynamics(dirName, figureName, which)

    elif(whichPlot == "wallalign"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        compareWallDynamicsAlign(dirName, figureName, which, dynamics)

    elif(whichPlot == "wallbeta"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        compareWallDynamicsDamping(dirName, figureName, which)

    elif(whichPlot == "interparams"):
        figureName = sys.argv[3]
        cluster = sys.argv[4]
        maxCluster = int(sys.argv[5])
        plotAlignParamsVSInteraction(dirName, figureName, cluster, maxCluster)

    elif(whichPlot == "noiseparams"):
        figureName = sys.argv[3]
        cluster = sys.argv[4]
        maxCluster = int(sys.argv[5])
        plotAlignParamsVSNoise(dirName, figureName, cluster, maxCluster)

    elif(whichPlot == "alphaparams"):
        figureName = sys.argv[3]
        cluster = sys.argv[4]
        maxCluster = int(sys.argv[5])
        plotAlphaParamsVSInteraction(dirName, figureName, cluster, maxCluster)

    elif(whichPlot == "boundtime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        plotBoundaryVSTime(dirName, figureName, which, dynamics)

    elif(whichPlot == "clustercom"):
        figureName = sys.argv[3]
        plotClustersCOM(dirName, figureName)

    elif(whichPlot == "clusterphi"):
        figureName = sys.argv[3]
        plotClusterDensity(dirName, figureName)

    elif(whichPlot == "numcluster"):
        figureName = sys.argv[3]
        versus = sys.argv[4]
        which = sys.argv[5]
        dynamics = sys.argv[6]
        minNum = float(sys.argv[7])
        compareNumClusterVSTime(dirName, figureName, versus, which, dynamics, minNum)

    elif(whichPlot == "clusterbound"):
        figureName = sys.argv[3]
        minNum = float(sys.argv[4])
        plotClustersVSBoundary(dirName, figureName, minNum)

    elif(whichPlot == "momsbound"):
        figureName = sys.argv[3]
        plotMomentsVSBoundary(dirName, figureName)

    elif(whichPlot == "boundtype"):
        figureName = sys.argv[3]
        versus = sys.argv[4]
        which = sys.argv[5]
        dynamics = sys.argv[6]
        plotBoundaryType(dirName, figureName, versus, which, dynamics)

    elif(whichPlot == "boundrough"):
        figureName = sys.argv[3]
        type = sys.argv[4]
        which = sys.argv[5]
        dynamics = sys.argv[6]
        compareBoundaryRoughness(dirName, figureName, type, which, dynamics)

    elif(whichPlot == "phaserough"):
        figureName = sys.argv[3]
        versus = sys.argv[4]
        which1 = sys.argv[5]
        which2 = sys.argv[6]
        dynamics = sys.argv[7]
        comparePhaseRoughness(dirName, figureName, versus, which1, which2, dynamics)

    elif(whichPlot == "angmomrough"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        compareAngMomRoughness(dirName, figureName, dynamics)

    elif(whichPlot == "orderparams"):
        figureName = sys.argv[3]
        dynamics = sys.argv[4]
        cluster = sys.argv[5]
        maxCluster = int(sys.argv[6])
        plotOrderParams(dirName, figureName, dynamics, cluster, maxCluster)

    elif(whichPlot == "moments"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        plotMoments(dirName, figureName, which, dynamics)

    elif(whichPlot == "energies"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        plotEnergies(dirName, figureName, which, dynamics)

    elif(whichPlot == "momparams"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        cluster = sys.argv[6]
        maxCluster = int(sys.argv[7])
        plotMomentAndOrderParams(dirName, figureName, which, dynamics, cluster, maxCluster)

    elif(whichPlot == "reflect"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        plotPhaseVSReflection(dirName, figureName, which, dynamics)

    elif(whichPlot == "pinned"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        dynamics = sys.argv[5]
        plotPinnedBoundary(dirName, figureName, which, dynamics)

    elif(whichPlot == "walldiagram"):
        which = sys.argv[3]
        dynamics = sys.argv[4]
        interpolate = sys.argv[5]
        phaseDiagramWallDynamics(dirName, which, dynamics, interpolate)

    else:
        print("Please specify the type of plot you want")
