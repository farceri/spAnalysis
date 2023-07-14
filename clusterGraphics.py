'''
Created by Francesco
14 July 2023
'''
#functions for clustering visualization
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import splev, splrep
import scipy.stats as st
import itertools
import sys
import os
import utils
import spCorrelation as spCorr

############################ local density analysis ############################
def plotSimplexDensity(dirName, figureName, pad = 1, logy=False):
    if(os.path.exists(dirName + os.sep + 'simplexDensity.dat')):
        simplexDensity = np.loadtxt(dirName + os.sep + 'simplexDensity.dat')
    else:
        _, simplexDensity = spCorr.computeDelaunayCluster(dirName, threshold=0.84, filter=True)
    denseSimplexList = np.loadtxt(dirName + os.sep + 'denseSimplexList.dat')
    #simplexDensity = simplexDensity[denseSimplexList==1]
    fig, ax = plt.subplots(1, 2, figsize=(9,4), dpi=150)
    ax[0].plot(np.arange(1, simplexDensity.shape[0]+1, 1), np.sort(simplexDensity), color='k', marker='.', markersize=8, lw=0.8, fillstyle='none')
    ax[0].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel('$Simplex$ $index$', fontsize=16)
    ax[0].set_ylabel('$\\varphi^{Simplex}$', fontsize=16)
    numBins = 100
    pdf, edges = np.histogram(simplexDensity, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax[1].plot(edges, pdf, color='k', marker='.', markersize=8, lw=0.8, fillstyle='none')
    y = np.linspace(np.min(pdf)-pad, np.max(pdf)+pad, 100)
    if(logy == 'logy'):
        ax[1].set_yscale('log')
        #ax[1].set_ylim(np.min(pdf), np.max(pdf)+pad/2)
        ax[1].set_ylim(6.4e-03, 50.6)
        y = np.linspace(1e-04, 100, 100)
    else:
        #ax[1].set_ylim(np.min(pdf)-pad/2, np.max(pdf)+pad/2)
        ax[1].set_ylim(-2.8, 43.2)
        y = np.linspace(-5, 50, 100)
    ax[1].plot(np.ones(100)*0.906899682, y, ls='dotted', color='k', lw=1, label='$Triangular$ $lattice$')
    ax[1].plot(np.ones(100)*0.785398163, y, ls='dashdot', color='k', lw=1, label='$Square$ $lattice$')
    ax[1].legend(fontsize=10, loc='best')
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].set_ylabel('$PDF(\\varphi^{Simplex})$', fontsize=16)
    ax[1].set_xlabel('$\\varphi^{Simplex}$', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    plt.savefig("/home/francesco/Pictures/soft/DelaunayDensityPDF" + figureName + ".png", transparent=True, format="png")
    plt.show()

def fitPhiPDF(dirName, figureName, numBins):
    fig1, ax1 = plt.subplots(figsize=(5,4), dpi = 120)
    fig2, ax2 = plt.subplots(figsize=(6,3.5), dpi = 120)
    if(os.path.exists(dirName + "localDensity-N" + numBins + ".dat")):
        data = np.loadtxt(dirName + "localDensity-N" + numBins + ".dat")
        #data = data[1:-1]
        data = data[2:-4]
        ax2.semilogy(data[:,0], data[:,1], color='k', lw=1, marker='o', fillstyle='none')
        T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
        x = data[data[:,1]>0,0]
        y = -np.log(data[data[:,1]>0,1])
        ax1.plot(x, T*y, color='k', lw=1, marker='o', fillstyle='none', label="$data$")
        # interpolate
        spl = splrep(x, y, s=0.5)
        interX = np.linspace(np.min(x), np.max(x), 1000)
        interY = splev(interX, spl)
        #ax1.plot(interX, interY, lw=1, color='b', label="$smooth$ $interpolation$")
        # fit interpolation
        failed = False
        try:
            popt, pcov = curve_fit(curve4Poly, interX, interY)
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            ax1.plot(interX, T*curve4Poly(interX, *popt), color=[0,0.4,1], lw=2, linestyle='--', label="$4Poly$ $fit$")
        print("Fitting parameters: c4 ", popt[0], " c3 ", popt[1], " c2 ", popt[2], " c1 ", popt[3], " c0 ", popt[4])
        print("Mass: ", np.sqrt(popt[2]/(4*popt[0])))
        #y = np.linspace(-2, 8, 100)
        #ax1.plot(np.ones(100)*(-popt[1]/(4*popt[0])), y, linestyle='--', color='r', lw=1)
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    #ax1.legend(fontsize=12, loc='upper left')
    ax1.set_xlim(-0.007, 1.37)
    ax1.set_ylim(-0.0082, 0.028)
    ax2.set_xlim(0.27, 1.57)
    ax2.set_ylim(1.4e-03, 7.58)
    ax1.set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=16)
    ax1.set_ylabel("$Free$ $energy,$ $F(\\varphi_l)$", fontsize=16)
    ax2.set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=16)
    ax2.set_ylabel("$Distribution,$ $P(\\varphi_l)$", fontsize=16)
    fig1.tight_layout()
    fig2.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pfitPhiFreeEnergy-" + figureName
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pfitPhiPDF-" + figureName
    fig2.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def fitPhiPDF2(dirName, figureName, numBins):
    fig, ax = plt.subplots(2, 1, figsize = (6,7.5), sharex=True, dpi = 120)
    if(os.path.exists(dirName + "localDensity-N" + numBins + ".dat")):
        data = np.loadtxt(dirName + "localDensity-N" + numBins + ".dat")
        #data = data[1:-1]
        data = data[2:-4]
        ax[0].semilogy(data[:,0], data[:,1], color='k', lw=1, marker='o', fillstyle='none')
        T = np.mean(np.loadtxt(dirName + "energy.dat")[:,4])
        x = data[data[:,1]>0,0]
        y = -np.log(data[data[:,1]>0,1])
        ax[1].plot(x, T*y, color='k', lw=1, marker='o', fillstyle='none', label="$data$")
        # interpolate
        spl = splrep(x, y, s=0.5)
        interX = np.linspace(np.min(x), np.max(x), 1000)
        interY = splev(interX, spl)
        #ax1.plot(interX, interY, lw=1, color='b', label="$smooth$ $interpolation$")
        # fit interpolation
        failed = False
        try:
            popt, pcov = curve_fit(curve4Poly, interX, interY)
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            ax[1].plot(interX, T*curve4Poly(interX, *popt), color=[0,0.4,1], lw=2, linestyle='--', label="$4Poly$ $fit$")
        print("Fitting parameters: c4 ", popt[0], " c3 ", popt[1], " c2 ", popt[2], " c1 ", popt[3], " c0 ", popt[4])
        print("Mass: ", np.sqrt(popt[2]/(4*popt[0])))
        #y = np.linspace(-2, 8, 100)
        #ax1.plot(np.ones(100)*(-popt[1]/(4*popt[0])), y, linestyle='--', color='r', lw=1)
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    #ax1.legend(fontsize=12, loc='upper left')
    ax[0].set_xlim(-0.007, 1.57)
    ax[0].set_ylim(1.4e-03, 7.58)
    ax[1].set_xlim(-0.007, 1.57)
    ax[1].set_ylim(-0.0082, 0.028)
    ax[0].set_ylabel("$Distribution,$ $P(\\varphi_l)$", fontsize=16)
    ax[1].set_ylabel("$Free$ $energy,$ $F(\\varphi_l)$", fontsize=16, labelpad=-5)
    ax[1].set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=16)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0)
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pfitPhiPDF-F-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDensityVSWindowSize(dirName, figureName, weight=False):
    fig, ax = plt.subplots(figsize=(7.5,5), dpi = 120)
    fileList = np.arange(8,34,2)
    colorList = cm.get_cmap('coolwarm', fileList.shape[0])
    for d in range(fileList.shape[0]):
        if(weight == 'weight'):
            fileName = dirName + os.sep + "localVoroDensity-N" + str(fileList[d]) + "-weight.dat"
        else:
            fileName = dirName + os.sep + "localVoroDensity-N" + str(fileList[d]) + ".dat"
        if(os.path.exists(fileName)):
            data = np.loadtxt(fileName)
            plt.plot(data[data[:,1]>0,0], data[data[:,1]>0,1], lw=1.2, color=colorList(d/fileList.shape[0]))
            if(fileList[d]==16):
                plt.semilogy(data[data[:,1]>0,0], data[data[:,1]>0,1], color='k', marker='*', markersize=10, lw=0, fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=18)
    ax.set_ylabel("$PDF(\\varphi_l)$", fontsize=18)
    ax.set_xlim(-0.05, 1.05)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    label = "$N_{bins}$"
    cb.set_ticks([1/26, 7/26, 13/26, 19/26, 25/26])
    cb.ax.tick_params(labelsize=14)
    ticklabels = ['$8$', '$14$', '$20$', '$26$', '$32$']
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=18, labelpad=25, rotation='horizontal')
    fig.tight_layout()
    if(weight == 'weight'):
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pWeightedWindow-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pWindow-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPLocalDensity(dirName, figureName, fixed='phi', which='30'):
    fig, ax = plt.subplots(figsize=(8,4), dpi = 120)
    if(fixed=="phi"):
        dirList = np.array(['10', '1', '1e-01', '1e-02', '1e-03', '1e-04', '1e-05'])
        labelList = np.array(['$10$', '$1$', '$0.1$', '$0.01$', '$0.001$', '$10^{-4}$', '$10^{-5}$'])
        taup = np.zeros(dirList.shape[0])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['thermal25', 'thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85', 'thermal85', 'thermal88', 'thermal94', 'thermal1'])
        labelList = np.array(['$\\varphi = 0.25$', '$\\varphi = 0.30$', '$\\varphi = 0.35$', '$\\varphi = 0.40$', '$\\varphi = 0.45$', '$\\varphi = 0.52$', '$\\varphi = 0.58$', '$\\varphi = 0.62$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.83$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr1e-03-f0200/dynamics/"
        if(os.path.exists(dirSample + "/localVoroDensity-N" + which + ".dat")):
            data = np.loadtxt(dirSample + os.sep + "localVoroDensity-N" + which + ".dat")
            plt.semilogy(data[:,0], data[:,1], lw=1.2, color=colorList(d/dirList.shape[0]), label=labelList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=18)
    ax.set_ylabel("$PDF(\\varphi_l)$", fontsize=18)
    ax.set_xlim(-0.05, 1.05)
    if(fixed=="phi"):
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pLocalPhi-vsDr-" + figureName
        colorBar = cm.ScalarMappable(cmap=colorList)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = plt.colorbar(colorBar, cax=cax)
        cb.ax.tick_params(labelsize=14)
        cb.set_ticks([0.5/labelList.shape[0],1-0.5/labelList.shape[0]])
        ticklabels = [np.format_float_scientific(taup[0], precision=2), np.format_float_scientific(taup[-1], precision=2)]
        ticklabels = ['$10^{-3}$', '$10^3$']
        cb.set_ticklabels(ticklabels)
        cb.set_label(label="$\\tau_p$", fontsize=16, labelpad=-20, rotation='horizontal')
    else:
        ax.legend(loc='upper right', fontsize=12)
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pLocalPhi-vsPhi-" + figureName
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPCompareLocalDensity(dirName, figureName, which='30'):
    fig, ax = plt.subplots(3,1, sharex=True, figsize=(8,10), dpi = 120)
    phiList = np.array(['thermal94', 'thermal45', 'thermal25'])
    dirList = np.array(['10', '1e-01', '1e-03', '1e-05'])
    labelList = np.array(['$10$', '$0.1$', '$0.001$', '$10^{-5}$'])
    taup = np.zeros(dirList.shape[0])
    colorList = cm.get_cmap('plasma', dirList.shape[0])
    colorBar = cm.ScalarMappable(cmap=colorList)
    for i in range(phiList.shape[0]):
        ax[i].tick_params(axis='both', labelsize=14)
        ax[i].set_ylabel("$PDF(\\varphi_l)$", fontsize=18)
        ax[i].set_ylim(0.0055, 64)
        divider = make_axes_locatable(ax[i])
        for d in range(dirList.shape[0]):
            dirSample = dirName + os.sep + phiList[i] + "/langevin/T0.001/iod10/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            if(os.path.exists(dirSample + "/localVoroDensity-N" + which + ".dat")):
                if(d<1):
                    data = np.loadtxt(dirSample + os.sep + "localVoroDensity-N" + which + ".dat")
                    ax[i].semilogy(data[:,0], data[:,1], lw=1.2, color=colorList(d/dirList.shape[0]), label=labelList[d])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = plt.colorbar(colorBar, cax=cax)
        cb.ax.tick_params(labelsize=14)
        cb.set_ticks([0.5/labelList.shape[0],1-2.5/labelList.shape[0], 1-1.5/labelList.shape[0], 1-0.5/labelList.shape[0]])
        #ticklabels = [np.format_float_scientific(taup[0], precision=2), np.format_float_scientific(taup[-1], precision=2)]
        ticklabels = ['$10^{-3}$', '$10^{-1}$', '$10^{1}$', '$10^3$']
        cb.set_ticklabels(ticklabels)
        cb.set_label(label="$\\tau_p$", fontsize=16, labelpad=20, rotation='horizontal')
    ax[2].set_xlabel("$Local$ $density,$ $\\varphi_l$", fontsize=18)
    ax[2].set_xlim(-0.05, 1.05)
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pCompareLocalPhi-vsDr-" + figureName
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################## cluster static properties ###########################
def plotSPClusterLengthscale(dirName, figureName, fixed=False, which='10'):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-06', '1e-05', '5e-05', '1e-04', '3e-04', '5e-04', '1e-03', '3e-03', '5e-03', '6e-03', '7e-03', '8e-03', '1e-02', '1.2e-02', '1.5e-02', '2e-02', '3e-02', '5e-02', '1e-01', '2e-01', '5e-01', '1', '5'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    clusterLength = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    lp = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = utils.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "clusterRad.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            lp[d] = utils.readFromDynParams(dirSample, "f0") * taup[d] / utils.readFromDynParams(dirSample, "damping")
            data = 2*np.loadtxt(dirSample + "clusterRad.dat")
            clusterLength[d,0] = np.mean(data)
            clusterLength[d,1] = np.std(data)
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterLength-vsPhi-" + figureName + "-iod" + which
        ax.plot(np.linspace(phi[0], phi[-1], 50), np.ones(50), color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterLength-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterLength-vsDamping-" + figureName + "-Dr" + which
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Cluster$ $lengthscale,$ $L_{cluster} / L_{box}$", fontsize=18)
    ax.errorbar(x, clusterLength[:,0], clusterLength[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
    if(fixed!="iod"):
        ax.set_xscale('log')
        #ax.set_yscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterPairCorr(dirName, figureName, fixed=False, which='10'):
    fig1, ax1 = plt.subplots(figsize=(6,5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-06', '1e-05', '5e-05', '1e-04', '3e-04', '5e-04', '1e-03', '3e-03', '5e-03', '6e-03', '7e-03', '8e-03', '1e-02', '1.2e-02', '1.5e-02', '2e-02', '3e-02', '5e-02'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    interDistance = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    lp = np.zeros(dirList.shape[0])
    meanRad = np.mean(np.loadtxt(dirName + "/particleRad.dat"))
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = utils.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "pairCorrCluster.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            lp[d] = utils.readFromDynParams(dirSample, "f0") * taup[d] / utils.readFromDynParams(dirSample, "damping")
            data = np.loadtxt(dirSample + "pairCorrCluster.dat")[20:]
            interDistance[d,0] = data[np.argmax(data[:,1]),0]
            interDistance[d,1] = data[np.argmax(data[:,2]),0]
            ax1.plot(data[data[:,0]<0.5,0], data[data[:,0]<0.5,1], color=colorList(d/dirList.shape[0]), lw=1)
            ax2.plot(data[data[:,0]<0.5,0], data[data[:,0]<0.5,2], color=colorList(d/dirList.shape[0]), lw=1)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel("$r$", fontsize=18)
    ax1.set_ylabel("$g_{Fluid}(r)$", labelpad=3, fontsize=18)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.set_xlabel("$r$", fontsize=18)
    ax2.set_ylabel("$g_{Gas}(r)$", labelpad=3, fontsize=18)
    #ax.set_ylim(-0.004, 0.036)
    fig1.tight_layout()
    fig2.tight_layout()
    fig, ax = plt.subplots(figsize = (6.5,4.5), dpi = 120)
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterLength-vsPhi-" + figureName + "-iod" + which
        ax.plot(np.linspace(phi[0], phi[-1], 50), np.ones(50), color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterLength-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterLength-vsDamping-" + figureName + "-Dr" + which
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Inter-particle$ $distance,$ $d / \\sigma$", fontsize=18)
    ax.semilogx(x[interDistance[:,0]>0], interDistance[interDistance[:,0]>0,0]/meanRad, color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none')
    ax.semilogx(x[interDistance[:,1]>0], interDistance[interDistance[:,1]>0,1]/meanRad, color='b', lw=1.2, marker='o', markersize = 7, fillstyle='none')
    ax.legend(("$Fluid$", "$Gas$"), fontsize=14, loc='best')
    if(fixed!="iod"):
        ax.set_xscale('log')
        #ax.set_yscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterFluctuations(dirName, figureName, fixed=False, which='10'):
    fig1, ax1 = plt.subplots(figsize=(6,5), dpi = 120)
    fig2, ax2 = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78',  'thermal80',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])
        labelList = np.array(['$\\varphi = 0.30$', '$\\varphi = 0.35$', '$\\varphi = 0.40$', '$\\varphi = 0.45$', '$\\varphi = 0.52$', '$\\varphi = 0.58$', '$\\varphi = 0.62$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.80$', '$\\varphi = 0.83$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-05', '5e-05', '1e-04', '5e-04', '1e-03', '5e-03', '7e-03', '1e-02', '1.2e-02', '1.5e-02', '2e-02', '3e-02', '5e-02', '1e-01', '2e-01', '5e-01', '1', '5', '10', '50', '100', '1000'])
        labelList = np.array(['$D_r = 10^{-5}$', '$D_r = 5 \\times 10^{-5}$', '$D_r = 10^{-4}$', '$D_r = 5 \\times 10^{-4}$', '$D_r = 0.001$', '$D_r = 0.005$', '$D_r = 0.007$', '$D_r = 0.01$', '$D_r = 0.012$', '$D_r = 0.015$', '$D_r = 0.02$', '$D_r = 0.03$', '$D_r = 0.05$', '$D_r = 0.1$', '$D_r = 0.2$', '$D_r = 0.5$', '$D_r = 1$', '$D_r = 5$', '$D_r = 10$', '$D_r = 50$', '$D_r = 100$', '$D_r = 1000$'])
        #dirList = np.array(['1e-05', '1e-04', '1e-03', '1e-02', '1e-01', '1'])
        #labelList = np.array(['$D_r = 0.00001$', '$D_r = 0.0001$', '$D_r = 0.001$', '$D_r = 0.01$', '$D_r = 0.1$', '$D_r = 1$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    clusterNum = np.zeros((dirList.shape[0],2))
    clusterSize = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = utils.readFromDynParams(dirSample, "Dr")
            phi[d] = utils.readFromParams(dirSample, "phi")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = utils.readFromDynParams(dirSample, "damping")
        #if(os.path.exists(dirSample + "dbAllClustersSize.dat")):
        #    data = np.loadtxt(dirSample + "dbAllClustersSize.dat")
        #    clusterSize[d,0] = np.mean(data)
        #    clusterSize[d,1] = np.std(data)
        if(os.path.exists(dirSample + "clusterFluctuations.dat")):
            data = np.loadtxt(dirSample + "clusterFluctuations.dat")
            clusterNum[d,0] = np.mean(data[:,1])
            clusterNum[d,1] = np.std(data[:,1])
            clusterSize[d,0] = np.mean(data[:,2])
            clusterSize[d,1] = np.std(data[:,2])
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterNum-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterSize-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = 1/Dr
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterNum-vsDr-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterSize-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterNum-vsDamping-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterSize-vsDamping-" + figureName + "-Dr" + which
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel("$Cluster$ $number,$ $\\langle N_c \\rangle$", fontsize=18)
    ax2.set_ylabel("$Cluster$ $area,$ $\\langle A_c \\rangle$", fontsize=18)
    ax1.errorbar(x, clusterNum[:,0], clusterNum[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
    ax2.errorbar(x, clusterSize[:,0], clusterSize[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
    if(fixed!="iod"):
        ax1.set_xscale('log')
        ax2.set_xscale('log')
    ax2.set_xlim(5.8e-06, 2.8e03)
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig2.savefig(figure2Name + ".png", transparent=False, format = "png")
    plt.show()

def plotSPNumberDensityFluctuations(dirName, figureName, fixed='Dr', which='1e-03'):
    #fig, ax = plt.subplots(2, 1, figsize = (7,8), sharex=True, dpi = 120)
    fig, ax = plt.subplots(figsize=(8,4.5), dpi = 120)
    if(fixed=="Dr"):
        dirList = np.array(['thermal94', 'thermal88', 'thermal85', 'thermal78', 'thermal67', 'thermal58', 'thermal45', 'thermal40', 'thermal35', 'thermal30', 'thermal27', 'thermal25'])
        labelList = np.array(['$\\varphi = 0.94$', '$\\varphi = 0.88$', '$\\varphi = 0.85$', '$\\varphi = 0.78$', '$\\varphi = 0.67$', '$\\varphi = 0.58$', '$\\varphi = 0.45$', '$\\varphi = 0.40$', '$\\varphi = 0.35$', '$\\varphi = 0.30$', '$\\varphi = 0.27$', '$\\varphi = 0.25$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['10', '1', '1e-01', '1.5e-02', '1e-02', '1e-03', '1e-04', '1e-05'])
        labelList = np.array(['$10$', '$1$', '$0.1$', '$0.015$', '$0.01$', '$0.001$', '$10^{-4}$', '$10^{-5}$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        taup = np.zeros(dirList.shape[0])
    else:
        print("please specify fixed parameter")
    for d in range(dirList.shape[0]):
        if(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "-f0200/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        if(os.path.exists(dirSample + "averageLocalNumberDensity.dat")):
            if(d<8):
                data = np.loadtxt(dirSample + "averageLocalNumberDensity.dat")
                ax.errorbar(data[:,1], data[:,7], data[:,8], lw=1.2, color=colorList(d/dirList.shape[0]), marker='o', fillstyle='none', capsize=3, label=labelList[d])
                #ax.plot(data[:,1], data[:,4], lw=1.2, color=colorList((dirList.shape[0]-d-1)/dirList.shape[0]), marker='o', fillstyle='none', label=labelList[d])
    ax.set_ylim(0.57, 2886)
    ax.set_ylim(4.6e-06, 0.34)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Density$ $variance,$ $\\Delta \\varphi^2$", fontsize=18)
    ax.set_xlabel("$Number$ $of$ $particles,$ $\\langle N_{sub} \\rangle$", fontsize=18)
    if(fixed=="Dr"):
        ax.legend(fontsize=10, loc='lower left', ncol=3)
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pNumberPhiVar-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        colorBar = cm.ScalarMappable(cmap=colorList)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = plt.colorbar(colorBar, cax=cax)
        cb.ax.tick_params(labelsize=14)
        cb.set_ticks([0.5/labelList.shape[0], 1-2.5/labelList.shape[0], 1-0.5/labelList.shape[0]])
        #ticklabels = [np.format_float_scientific(taup[0], precision=2), np.format_float_scientific(taup[-1], precision=2)]
        ticklabels = ['$10^{-3}$', '$10^{1}$', '$10^3$']
        cb.set_ticklabels(ticklabels)
        cb.set_label(label="$\\tau_p$", fontsize=16, labelpad=-5, rotation='horizontal')
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pNumberPhiVar-vsDr-" + figureName + "-iod" + which
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterDistribution(dirName, figureName, fixed=False, which='10', numBins=40):
    fig, ax = plt.subplots(figsize=(6.5,4), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72',  'thermal78',  'thermal80',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$0.45$', '$0.58$', '$0.67$', '$0.72$', '$0.78$', '$0.80$', '$0.83$', '$0.85$', '$0.88$', '$0.94$', '$1.00$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1.5e-02', '5e-02', '1e-01', '2e-01', '5e-01', '1'])
        labelList = np.array(['$0.015$', '$0.05$', '$0.1$', '$0.2$', '$0.5$', '$1$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+3)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = utils.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = utils.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "clusterNumbers.dat")):
            clusterNumber = np.loadtxt(dirSample + "clusterNumbers.dat")
            clusterNumber = clusterNumber[:-100]
            pdf, edges = np.histogram(clusterNumber, bins=np.geomspace(np.min(clusterNumber), np.max(clusterNumber), numBins), density=True)
            edges = (edges[1:] + edges[:-1])/2
            data = np.column_stack((edges, pdf))
            #data = np.loadtxt(dirSample + "clusterNumberPDF.dat")[:-1,:]
            data = data[data[:,1]>0]
            if(d == 1):
                data = data[:-2]
            else:
                data = data[:-1]
            ax.loglog(data[1:,0], data[1:,1]/data[1,1], lw=1.2, color=colorList((dirList.shape[0]-d-3)/dirList.shape[0]), marker='o', fillstyle='none')
    # make color bar for legend
    if(fixed=="iod"):
        label = "$\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pNumberPhiVar-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        label = "$D_r$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pNumberPhiVar-vsDr-" + figureName + "-iod" + which
    else:
        label = "$m/\\gamma$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pNumberPhiVar-vsDamping-" + figureName + "-Dr" + which
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Number$ $of$ $particles$ $in$ $cluster,$ $N_c$", fontsize=18)
    ax.set_ylabel("$Distribution,$ $P(N_c)$", fontsize=18)
    fig.tight_layout()
    #plt.subplots_adjust(hspace=0)
    #colorBar = cm.ScalarMappable(cmap=colorList)
    #cb = fig.colorbar(colorBar, ax=ax)#, shrink=0.7)
    #cb.set_ticks(np.arange(0.5/labelList.shape[0],1,1/labelList.shape[0]))
    #cb.ax.tick_params(labelsize=12)
    #ticklabels = labelList
    #cb.set_ticklabels(ticklabels)
    #cb.set_label(label=label, fontsize=16, labelpad=10, rotation='horizontal')
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSizeVSTime(dirName, figureName, fixed=False, which='10'):
    fig, ax = plt.subplots(figsize=(7.5,5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72',  'thermal78',  'thermal80',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$0.45$', '$0.58$', '$0.67$', '$0.72$', '$0.78$', '$0.80$', '$0.83$', '$0.85$', '$0.88$', '$0.94$', '$1.00$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-05', '5e-05', '1e-04', '5e-04', '1e-03', '5e-03', '7e-03', '1e-02', '1.5e-02', '3e-02', '1e-01', '2e-01', '5e-01', '1', '10'])
        labelList = np.array(['$10^{-5}$', '$5 \\times 10^{-5}$', '$10^{-4}$', '$5 \\times 10^{-4}$', '$0.001$', '$0.005$', '$0.007$', '$0.01$', '$0.015$', '$0.03$', '$0.1$', '$0.2$', '$0.5$', '$1$', '$10$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    for d in range(4,dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics/"
            Dr[d] = utils.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics/"
            damping[d] = utils.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "clusterFluctuations.dat")):
            data = np.loadtxt(dirSample + "clusterFluctuations.dat")
            data = data[np.argwhere(data[:,0]%100000==0)[:,0]]
            ax.plot(data[:,0], data[:,2], lw=1.2, color=colorList(d/dirList.shape[0]))
    ax.set_ylim(-0.018, 0.406)
    # make color bar for legend
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    if(fixed=="iod"):
        label = "$\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterTime-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        label = "$D_r$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterTime-vsDr-" + figureName + "-iod" + which
    else:
        label = "$m/\\gamma$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterTime-vsDamping-" + figureName + "-Dr" + which
    cb.set_ticks(np.arange(0.5/labelList.shape[0],1,1/labelList.shape[0])+4/labelList.shape[0])
    cb.ax.tick_params(labelsize=12)
    ticklabels = labelList[4:]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=16, labelpad=10, rotation='horizontal')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $time,$ $t$", fontsize=18)
    ax.set_ylabel("$Cluster$ $area,$ $A_c(t)$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterShape(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['thermal25', 'thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        dirList = np.array(['0.25', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    meanShape = np.zeros((dirList.shape[0],3))
    errorShape = np.zeros((dirList.shape[0],3))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            #phi[d] = utils.readFromParams(dirSample, "phi")
            phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
            #print(phi[d], utils.readFromParams(dirSample, "phi"))
            if(d==0):
                numParticles = utils.readFromParams(dirSample, "numParticles")
        if(os.path.exists(dirSample + "shapeParameter.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "shapeParameter.dat")
            meanShape[d,0] = np.mean(data[:,1])
            errorShape[d,0] = np.std(data[:,1])
            meanShape[d,1] = np.mean(data[:,2])
            errorShape[d,1] = np.std(data[:,2])
            meanShape[d,2] = np.mean(data[:,3])
            errorShape[d,2] = np.std(data[:,3])
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterShape-vsPhi-" + figureName
        ax.plot(x, phi, color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterShape-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Shape$ $parameter$", fontsize=18)
    ax.errorbar(x[meanShape[:,2]>0], meanShape[meanShape[:,2]>0,2], errorShape[meanShape[:,2]>0,2], color='k', lw=1.2, marker='s', markersize = 8, fillstyle='none', elinewidth=1, capsize=4)
    if(fixed!="Dr"):
        ax.set_xscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotClusterShapeVSTime(dirName, figureName):
    #numParticles = int(utils.readFromParams(dirName, "numParticles"))
    shape = np.loadtxt(dirName + os.sep + "shapeParam.dat")
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    ax.plot(shape[:,0], shape[:,3], linewidth=1.2, color='k', ls='solid')
    # plotting settings
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax.set_ylabel("$Shape$ $parameter$", fontsize=15)
    ax.legend(fontsize=12, loc='best')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterShapeVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterMixingTime(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        dirList = np.array(['0.26', '0.28', '0.30', '0.32', '0.35', '0.40', '0.45', '0.50'])#, '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    mixingTime = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/short/"
            #phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]
        if(os.path.exists(dirSample + "logMixingTime.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "logMixingTime.dat")
            ax.errorbar(data[:,0], data[:,1], data[:,2], color=colorList(d/dirList.shape[0]), lw=1, marker='o', capsize=3, label="$\\varphi=$" + dirList[d])
    ax.set_xscale('log')
    if(fixed=="Dr"):
        #x = phi
        #xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pMxingTime-vsPhi-" + figureName
    elif(fixed=="phi"):
        #x = taup
        #xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pMixingTime-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.set_xlabel("$Elapsed$ $time,$ $\\Delta t$", fontsize=18)
    ax.set_ylabel("$Mixing$ $ratio$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

######################### cluster pressure properties ##########################
def plotClusterPressureVSTime(dirName, figureName, bound=False, prop=False):
    #numParticles = int(utils.readFromParams(dirName, "numParticles"))
    pressure = np.loadtxt(dirName + os.sep + "clusterPressure.dat")
    fig, ax = plt.subplots(1, 2, sharey=True, figsize = (11, 5), dpi = 120)
    if(bound == "bound"):
        ax[0].plot(pressure[:,0], pressure[:,1], linewidth=1.5, color='g', label="$wall$")
        ax[1].plot(pressure[:,0], pressure[:,1], linewidth=1.5, color='g', label="$wall$")
    # dense particles
    ax[0].plot(pressure[:,0], pressure[:,2], linewidth=1.5, color='k', label="$virial$")
    ax[0].plot(pressure[:,0], pressure[:,3], linewidth=1.5, color='r', label="$thermal$")
    ptot = pressure[:,2] + pressure[:,3]
    if(prop == "prop"):
        ax[0].plot(pressure[:,0], pressure[:,4], linewidth=1.5, color=[1,0.5,0], label="$active$")
        ptot += pressure[:,4]
    ax[0].plot(pressure[:,0], ptot, linewidth=1.5, color='b', ls='dotted', label="$total$")
    # dilute particles
    ax[1].plot(pressure[:,0], pressure[:,5], linewidth=1.5, color='k', ls='--', label="$virial$")
    ax[1].plot(pressure[:,0], pressure[:,6], linewidth=1.5, color='r', ls='--', label="$thermal$")
    ptot = pressure[:,5] + pressure[:,6]
    if(prop == "prop"):
        ax[1].plot(pressure[:,0], pressure[:,7], linewidth=1.5, color=[1,0.5,0], ls='--', label="$active$")
        ptot += pressure[:,7]
    ax[1].plot(pressure[:,0], ptot, linewidth=1.5, color='b', ls='dotted', label="$total$")
    # plotting settings
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[0].set_ylabel("$Pressure$", fontsize=15)
    ax[0].legend(fontsize=12, loc='best')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    #figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterPressure-" + figureName
    #fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPressureProfile(dirName, figureName, shift=0, which='pressure'):
    fig, ax = plt.subplots(figsize = (8,4), dpi = 120)
    if(os.path.exists(dirName + "/pressureProfile.dat")):
        data = np.loadtxt(dirName + "/pressureProfile.dat")
        sigma = float(utils.readFromDynParams(dirName, 'sigma'))
        data[:,1] = np.roll(data[:,1], shift)
        data[:,2] = np.roll(data[:,2], shift)
        data[:,3] = np.roll(data[:,3], shift)
        data[:,4] = np.roll(data[:,4], shift)
        data[:,5] = np.roll(data[:,5], shift)
        data[:,6] = np.roll(data[:,6], shift)
        data[:,7] = np.roll(data[:,7], shift)
        data[:,8] = np.roll(data[:,8], shift)
        half = int(data.shape[0]/2)
        print("surface tension: ", np.sum(data[:half,0] * (data[:half,2] + data[:half,6] - data[:half,3] - data[:half,7]))/sigma)
        if(which=='pressure'):
            ax.plot(data[:,0], data[:,1], lw=1.5, color='k', ls='--', label='$Steric$')
            ax.plot(data[:,0], data[:,4], lw=1.5, color='r', ls='dotted', label='$Thermal$')
            ax.plot(data[:,0], data[:,5], lw=1.5, color=[1,0.5,0], label='$Active$')
            ax.plot(data[:,0], data[:,8], lw=1.5, color='b', ls='dashdot', label='$Total$')
        elif(which=='delta'):
            ax.plot(data[:,0], data[:,2] - data[:,3], lw=1.5, color='k', ls='solid', label='$Steric$')
            ax.plot(data[:,0], data[:,6] - data[:,7], lw=1.5, color=[1,0.5,0], ls='solid', label='$Active$')
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Radial$ $distance,$ $r$", fontsize=18)
    ax.set_xlabel("$Position,$ $x$", fontsize=18)
    if(which=="pressure"):
        ax.set_ylabel("$Pressure,$ $p \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,4])-0.2, np.max(data[:,8])+0.6)
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pProfile-" + figureName + ".png"
    elif(which=="delta"):
        ax.set_ylabel("$\\Delta p^\ast = p_{xx} - p_{yy} \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,6] - data[:,7])-0.2, np.max(data[:,6] - data[:,7])+0.6)
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/deltaProfile-" + figureName + ".png"
    ax.legend(fontsize=13, loc='upper right', ncol=4)
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPClusterDensity(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['thermal25', 'thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        dirList = np.array(['0.25', '0.26', '0.27', '0.28', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    fluidDensity = np.zeros((dirList.shape[0],2))
    gasDensity = np.zeros((dirList.shape[0],2))
    density = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            #phi[d] = utils.readFromParams(dirSample, "phi")
            #phi[d] = np.loadtxt(dirSample + 'localDelaunayDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
            if(d==0):
                numParticles = utils.readFromParams(dirSample, "numParticles")
        if(os.path.exists(dirSample + "delaunayDensity.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "delaunayDensity.dat")
            fluidDensity[d,0] = np.mean(data[:,1])
            fluidDensity[d,1] = np.std(data[:,1])
            gasDensity[d,0] = np.mean(data[:,2])
            gasDensity[d,1] = np.std(data[:,2])
            phi[d] = np.mean(data[:,3])
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterVoro-vsPhi-" + figureName
        ax.plot(x, x, color='k', lw=1.2, ls='--')
        # interpolate to find upper bound
        fluid = fluidDensity[fluidDensity[:,0]>0,0][10:]
        error = fluidDensity[fluidDensity[:,0]>0,1][10:]
        phiup = phi[fluidDensity[:,0]>0][10:]
        index = np.argwhere((fluid - error*0.5) <= phiup)[0,0]
        error = error[index]*0.5
        p = np.linspace(phiup[index-1], phiup[index],100)
        fluidSlope = (fluid[index] - fluid[index-1]) / (phiup[index] - phiup[index-1])
        phiSlope = (phiup[index] - phiup[index-1]) / (phiup[index] - phiup[index-1])
        fluidInter = fluid[index-1] + fluidSlope*(p - phiup[index-1])
        phiInter = phiup[index-1] + phiSlope*(p - phiup[index-1])
        phiupper = p[np.argwhere((fluidInter - error) <= phiInter)[0,0]]
        ax.plot(phiInter, phiInter, color='r')
        ax.plot(phiInter, fluidInter, color='r', ls='--', lw=4)
        # interpolate to find lower bound
        gas = gasDensity[gasDensity[:,0]>0,0][:10]
        philow = phi[fluidDensity[:,0]>0][:10]
        index = np.argwhere((gas + 1e-02) < philow)[0,-1]
        p = np.linspace(philow[index-1], philow[index],100)
        gasSlope = (gas[index] - gas[index-1]) / (philow[index] - philow[index-1])
        phiSlope = (philow[index] - philow[index-1]) / (philow[index] - philow[index-1])
        gasInter = gas[index-1] + gasSlope*(p - philow[index-1])
        phiInter = philow[index-1] + phiSlope*(p - philow[index-1])
        philower = p[np.argwhere((gasInter + 1e-02) < phiInter)[0,-1]]
        ax.plot(phiInter, phiInter, color='r')
        ax.plot(phiInter, gasInter, color='r', ls='--', lw=4)
        print("phiup:", phiupper, "phidown:", philower)
        np.savetxt(dirName + "MIPSBounds.dat", np.column_stack((numParticles, philower, phiupper)))
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterVoro-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Area$ $fraction$", fontsize=18)
    ax.errorbar(x[fluidDensity[:,0]>0], fluidDensity[fluidDensity[:,0]>0,0], fluidDensity[fluidDensity[:,0]>0,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Fluid$')
    ax.errorbar(x[gasDensity[:,0]>0], gasDensity[gasDensity[:,0]>0,0], gasDensity[gasDensity[:,0]>0,1], color='g', lw=1.2, marker='o', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Gas$')
    ax.legend(fontsize=14, loc='best')
    ax.set_ylim(-0.18, 1.02)
    y = np.linspace(-0.2, 1.05, 100)
    dirSample = dirName + os.sep + "0.30/active-langevin/Dr" + which + "/dynamics/"
    phi1 = np.mean(np.loadtxt(dirSample + "delaunayDensity.dat")[:,3])
    dirSample = dirName + os.sep + "0.31/active-langevin/Dr" + which + "/dynamics/"
    phi2 = np.mean(np.loadtxt(dirSample + "delaunayDensity.dat")[:,3])
    ax.plot(np.ones(100)*(phi1+phi2)/2, y, ls='dotted', color='k', lw=1)
    if(fixed!="Dr"):
        ax.set_xscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSurfaceTension(dirName, figureName, fixed='Dr', which='2e-04'):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        taup = np.zeros(dirList.shape[0])
    elif(fixed=="Dr"):
        #dirList = np.array(['30', '35', '40', '45', '52', '58', '62', '67', '72',  '78', '82', '85', '88', '91', '94'])
        dirList = np.array(['25', '28', '29', '30', '31', '32', '35', '40', '45', '50', '55', '60', '65', '70',  '75', '80', '82', '84', '86', '88', '90', '92', '94', '96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    tension = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(utils.readFromParams(dirSample, "Dr") * utils.readFromParams(dirSample, "sigma"))
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + "0." + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
        if(os.path.exists(dirSample + "/borderEnergy.dat")):
            data = np.loadtxt(dirSample + "/surfaceWork.dat")
            tension[d,0] = np.mean((data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]))
            tension[d,1] = np.std((data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]))
    ax.tick_params(axis='both', labelsize=14)
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pSurfaceTension-vsDr-" + figureName + "-phi" + which
        ax.set_xscale('log')
        #ax.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pSurfaceTension-vsPhi-" + figureName + "-Dr" + which
    ax.errorbar(x[tension[:,0]>0], tension[tension[:,0]>0,0], tension[tension[:,0]>0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Surface$ $tension,$ $\\gamma$", fontsize=18, labelpad=15)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSurfaceTensionVSTime(dirName, figureName):
    #numParticles = int(utils.readFromParams(dirName, "numParticles"))
    data = np.loadtxt(dirName + os.sep + "surfaceWork.dat")
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    ax.plot(data[1:,0], (data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]), linewidth=1.2, color='k')
    # plotting settings
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax.set_ylabel("$Surface$ $tension,$ $\\Delta W / \\Delta l_c$", fontsize=15)
    ax.legend(fontsize=12, loc='best')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pSurfaceTensionVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPForceVelMagnitude(dirName, figureName, fixed=False, which='1.5e-04'):
    fig, ax = plt.subplots(figsize=(7.5,5), dpi = 120)
    if(fixed=="phi"):
        if(which == '0.45'):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        dirList = np.array(['thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    vmIn = np.zeros((dirList.shape[0], 3, 2))
    vmOut = np.zeros((dirList.shape[0], 3, 2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        if(os.path.exists(dirSample + "clusterVelMagnitude.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "clusterVelMagnitude.dat")
            # dense steric
            vmIn[d,0,0] = np.mean(data[:,1])
            vmIn[d,0,1] = np.std(data[:,1])
            # dense thermal
            vmIn[d,1,0] = np.mean(data[:,2])
            vmIn[d,1,1] = np.std(data[:,2])
            # dense active
            vmIn[d,2,0] = np.mean(data[:,3])
            vmIn[d,2,1] = np.std(data[:,3])
            # dilute steric
            vmOut[d,0,0] = np.mean(data[:,4])
            vmOut[d,0,1] = np.std(data[:,4])
            # dilute thermal
            vmOut[d,1,0] = np.mean(data[:,5])
            vmOut[d,1,1] = np.std(data[:,5])
            # dilute active
            vmOut[d,2,0] = np.mean(data[:,6])
            vmOut[d,2,1] = np.std(data[:,6])
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pVelMagnitude-vsDr-" + figureName
        ax.set_xscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pVelMagnitude-vsPhi-" + figureName
    ax.errorbar(x[vmIn[:,0,0]>0], vmIn[vmIn[:,0,0]>0,0,0], vmIn[vmIn[:,0,0]>0,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
    ax.errorbar(x[vmIn[:,1,0]>0], vmIn[vmIn[:,1,0]>0,1,0], vmIn[vmIn[:,1,0]>0,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
    ax.errorbar(x[vmIn[:,2,0]>0], vmIn[vmIn[:,2,0]>0,2,0], vmIn[vmIn[:,2,0]>0,2,1], lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Active$')
    ax.errorbar(x[vmOut[:,0,0]>0], vmOut[vmOut[:,0,0]>0,0,0], vmOut[vmOut[:,0,0]>0,0,1], ls='--', lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax.errorbar(x[vmOut[:,1,0]>0], vmOut[vmOut[:,1,0]>0,1,0], vmOut[vmOut[:,1,0]>0,1,1], ls='--', lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3)
    ax.errorbar(x[vmOut[:,2,0]>0], vmOut[vmOut[:,2,0]>0,2,0], vmOut[vmOut[:,2,0]>0,2,1], ls='--', lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Velocity$ $magnitude$", fontsize=18)
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPTotalPressure(dirName, figureName, fixed='Dr', which='2e-04'):
    fig, ax = plt.subplots(figsize = (7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['30', '35', '40', '45', '52', '58', '62', '67', '72',  '78', '82', '85', '88', '91', '94'])
        dirList = np.array(['25', '28', '29', '30', '31', '32', '35', '40', '45', '50', '55', '60', '65', '70',  '75', '80', '82', '84', '86', '88', '90', '92', '94', '96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    p = np.zeros((dirList.shape[0], 4, 2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + "thermal" + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            dirSample = dirName + os.sep + "0." + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            #phi[d] = utils.readFromParams(dirSample, "phi")
            phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
            #print(phi[d], utils.readFromParams(dirSample, "phi"))
        if(os.path.exists(dirSample + "/pressure.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "/pressure.dat")
            sigma = np.mean(np.loadtxt(dirSample + "/particleRad.dat"))
            # steric
            p[d,0,0] = np.mean(data[:,2])
            p[d,0,1] = np.std(data[:,2])
            # thermal
            p[d,1,0] = np.mean(data[:,3])
            p[d,1,1] = np.std(data[:,3])
            # active
            p[d,2,0] = np.mean(data[:,4])
            p[d,2,1] = np.std(data[:,4])
            # dense total
            p[d,3,0] = np.mean(data[:,2] + data[:,3] + data[:,4])
            p[d,3,1] = np.std(data[:,2] + data[:,3] + data[:,4])
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pTotPressure-vsDr-" + figureName + "-phi" + which
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pTotPressure-vsPhi-" + figureName + "-Dr" + which
    ax.tick_params(axis='both', labelsize=14)
    # pressure components
    ax.errorbar(x[p[:,0,0]>0], p[p[:,0,0]>0,0,0], p[p[:,0,0]>0,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
    ax.errorbar(x[p[:,1,0]>0], p[p[:,1,0]>0,1,0], p[p[:,1,0]>0,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
    ax.errorbar(x[p[:,2,0]>0], p[p[:,2,0]>0,2,0], p[p[:,2,0]>0,2,1], lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Actve$')
    #ax.errorbar(x[p[:,3,0]>0], p[p[:,3,0]>0,3,0], p[p[:,3,0]>0,3,1], lw=1.2, color='b', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Total$')
    #ax.set_xlim(5.8e-06, 2.8e03)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Pressure,$ $P \\sigma^2$", fontsize=18)
    ax.legend(fontsize=14, loc='best')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterPressure(dirName, figureName, fixed='Dr', inter=False, which='gasFluid'):
    fig1, ax1 = plt.subplots(figsize = (7,4), dpi = 120)
    fig2, ax2 = plt.subplots(figsize = (7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['thermal30', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85', 'thermal85', 'thermal88', 'thermal94'])
        dirList = np.array(['0.25', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    pIn = np.zeros((dirList.shape[0], 3, 2))
    pOut = np.zeros((dirList.shape[0], 3, 2))
    ptotIn = np.zeros((dirList.shape[0], 2))
    ptotOut = np.zeros((dirList.shape[0], 2))
    deltap = np.zeros((dirList.shape[0], 2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr1.5e-04/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr2e-04/dynamics/"
            if(os.path.exists(dirSample + "localVoroDensity-N16-stats.dat")):
                phi[d] = np.loadtxt(dirSample + "localVoroDensity-N16-stats.dat")[0]
            else:
                phi[d] = spCorr.averageLocalVoronoiDensity(dirSample)
        if(os.path.exists(dirSample + "/delaunayPressure.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "/delaunayPressure.dat")
            # dense steric
            pIn[d,0,0] = np.mean(data[:,2])
            pIn[d,0,1] = np.std(data[:,2])
            # dense thermal
            pIn[d,1,0] = np.mean(data[:,3])
            pIn[d,1,1] = np.std(data[:,3])
            # dense active
            pIn[d,2,0] = np.mean(data[:,4])
            pIn[d,2,1] = np.std(data[:,4])
            # dense total
            ptotIn[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4])
            ptotIn[d,1] = np.std(data[:,2] + data[:,3] + data[:,4])
            # dilute steric
            pOut[d,0,0] = np.mean(data[:,5])
            pOut[d,0,1] = np.std(data[:,5])
            # thermal
            pOut[d,1,0] = np.mean(data[:,6])
            pOut[d,1,1] = np.std(data[:,6])
            # active
            pOut[d,2,0] = np.mean(data[:,7])
            pOut[d,2,1] = np.std(data[:,7])
            # dilute total
            ptotOut[d,0] = np.mean(data[:,5] + data[:,6] + data[:,7])
            ptotOut[d,1] = np.std(data[:,5] + data[:,6] + data[:,7])
            # delta pressure
            deltap[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
            deltap[d,1] = np.std(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
    # interpolate to find tradeoff point
    if(inter == 'inter'):
        check = False
        if(which == "gasFluid"):
            bottom = ptotIn[ptotIn[:,0]>0,0]
            top = ptotOut[ptotIn[:,0]>0,0]
            tau = taup[ptotIn[:,0]>0]
        elif(which == "stericActive"):
            bottom = pIn[pIn[:,2,0]>0,0,0]
            top = pOut[pOut[:,2,0]>0,2,0]
            tau = taup[pIn[:,2,0]>0]
        for i in range(top.shape[0]):
            if(top[i] > bottom[i]):
                check = True
        if(check):
            index = np.argwhere(bottom < top)[0,0]
            t = np.linspace(tau[index-1], tau[index],100)
            bottomSlope = (bottom[index] - bottom[index-1]) / (tau[index] - tau[index-1])
            topSlope = (top[index] - top[index-1]) / (tau[index] - tau[index-1])
            bottomInter = bottom[index-1] + bottomSlope*(t - tau[index-1])
            topInter = top[index-1] + topSlope*(t - tau[index-1])
            taupc = t[np.argwhere(bottomInter < topInter)[0,0]]
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[index] + "/dynamics/"
            if(os.path.exists(dirSample + "localVoroDensity-N16-stats.dat")):
                phic = np.loadtxt(dirSample + "localVoroDensity-N16-stats.dat")[0]
            else:
                phic = spCorr.averageLocalVoronoiDensity(dirSample)
            print("interpolation indices:", index-1, index, "tauc:", taupc, "density:", phic)
            np.savetxt(dirName + "iod10/active-langevin/" + which + "Tradeoff.dat", np.array([phic, taupc]))
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pGasFluid-vsDr-" + figureName
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pPressures-vsDr-" + figureName
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pGasFLuid-vsPhi-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pPressures-vsPhi-" + figureName + "-Dr" + which
        ax1.set_yscale('log')
        ax2.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    # dense and dilute pressure
    ax1.errorbar(x[ptotIn[:,0]>0], ptotIn[ptotIn[:,0]>0,0], ptotIn[ptotIn[:,0]>0,1], color='b', marker='s', markersize=8, fillstyle='none', lw=1.2, capsize=3, label='$Fluid$')
    ax1.errorbar(x[ptotOut[:,0]>0], ptotOut[ptotOut[:,0]>0,0], ptotOut[ptotOut[:,0]>0,1], color='g', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3, label='$Gas$')
    #ax1.errorbar(x[deltap[:,0]!=0], np.abs(deltap[deltap[:,0]!=0,0]), deltap[deltap[:,0]!=0,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3, label='$\\Delta p$')
    # pressure components
    if(fixed=='phi'):
        ax2.errorbar(x[pIn[:,0,0]>0], pIn[pIn[:,0,0]>0,0,0], pIn[pIn[:,0,0]>0,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
        ax2.errorbar(x[pIn[:,1,0]>0], pIn[pIn[:,1,0]>0,1,0], pIn[pIn[:,1,0]>0,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
        ax2.errorbar(x[pIn[:,2,0]>0], pIn[pIn[:,2,0]>0,2,0], pIn[pIn[:,2,0]>0,2,1], lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Actve$')
        ax2.errorbar(x[pOut[:,0,0]>0], pOut[pOut[:,0,0]>0,0,0], pOut[pOut[:,0,0]>0,0,1], ls='--', lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax2.errorbar(x[pOut[:,1,0]>0], pOut[pOut[:,1,0]>0,1,0], pOut[pOut[:,1,0]>0,1,1], ls='--', lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3)
        ax2.errorbar(x[pOut[:,2,0]>0], pOut[pOut[:,2,0]>0,2,0], pOut[pOut[:,2,0]>0,2,1], ls='--', lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3)
    else:
        ax2.errorbar(x, pIn[:,0,0], pIn[:,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
        ax2.errorbar(x, pIn[:,1,0], pIn[:,1,1], lw=1.2, color='r', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
        ax2.errorbar(x, pIn[:,2,0], pIn[:,2,1], lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Active$')
        ax2.errorbar(x, pOut[:,0,0], pOut[:,0,1], ls='--', lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax2.errorbar(x, pOut[:,1,0], pOut[:,1,1], ls='--', lw=1.2, color='r', marker='o', markersize=8, fillstyle='none', capsize=3)
        ax2.errorbar(x, pOut[:,2,0], pOut[:,2,1], ls='--', lw=1.2, color=[1,0.5,0], marker='s', markersize=8, fillstyle='none', capsize=3)
    if(inter == 'inter' and check):
        if(which=='gasFluid'):
            ax1.plot(np.ones(100)*taupc, np.linspace(np.min(ptotOut[:,0]), np.max(ptotIn[:,0]), 100), lw=0.7, ls='--', color='k')
            ax1.plot(t, topInter, lw=2, color='r')
            ax1.plot(t, bottomInter, lw=2, color='k')
        else:
            ax2.plot(np.ones(100)*taupc, np.linspace(np.min(pOut[:,0]), np.max(pIn[:,0]), 100), lw=0.7, ls='--', color='k')
            ax2.plot(t, topInter, lw=2, color='r')
            ax2.plot(t, bottomInter, lw=2, color='k')
    #ax.set_xlim(5.8e-06, 2.8e03)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel("$Total$ $pressure$", fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("$Pressure$ $components$", fontsize=18)
    ax1.legend(fontsize=14, loc='best')
    ax2.legend(fontsize=14, loc='best')
    fig1.tight_layout()
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig2.tight_layout()
    fig2.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterTradeoff(dirName, figureName, which='gasFluid'):
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    dirList = np.array(['thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal83', 'thermal88'])
    taupc = np.zeros(dirList.shape[0])
    phi = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/"
        if(os.path.exists(dirSample + os.sep + which + "Tradeoff.dat")):
            data = np.loadtxt(dirSample + os.sep + which + "Tradeoff.dat")
            phi[d] = data[0]
            taupc[d] = data[1]
    ax.plot(taupc[taupc>0], phi[taupc>0], color='k', lw=1, marker='o', fillstyle='none')
    np.savetxt(dirName + os.sep + which + "Tradeoff.dat", np.column_stack((phi[taupc>0], taupc[taupc>0])))
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterTradeoff-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Density$", fontsize=18)
    ax.set_xlabel("$Tradeoff$ $time,$ $\\tau_p^*$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPhaseDiagram(dirName, numBins, figureName, which='16', log=False):
    fig, ax = plt.subplots(figsize=(7.5,6), dpi = 120)
    phi = np.array(['25', '27', '30', '35', '40', '45', '52', '58', '62', '67', '72', '78', '82', '85', '88', '91', '94'])
    Dr = np.array(['100', '10', '5', '1', '5e-01', '1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    taup = np.zeros((phi.shape[0], Dr.shape[0]))
    # load the data
    deltaPhi = np.zeros((phi.shape[0], Dr.shape[0]))
    meanPhi = np.zeros((phi.shape[0], Dr.shape[0]))
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            dirSample = dirName + 'thermal' + phi[i] + '/langevin/T0.001/iod10/active-langevin/Dr' + Dr[j] + '/dynamics/'
            deltaFile = dirSample + 'localVoroDensity-N' + which + '-stats.dat'#'localDensity-N16-stats.dat'
            #phiFile = dirSample + 'voronoiDensity.dat'
            phiFile = dirSample + 'delaunayDensity.dat'
            if(os.path.exists(deltaFile)):
                #data = np.loadtxt(phiFile)
                #meanPhi[i,j] = np.mean(data[:,3])
                data = np.loadtxt(deltaFile)
                meanPhi[i,j] = data[0]
                deltaPhi[i,j] = data[1]
                taup[i,j] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
    # assign color based on deltaPhi
    colorId = np.zeros((phi.shape[0], Dr.shape[0]))
    min = np.min(deltaPhi)
    max = np.max(deltaPhi)
    if(log=='log'):
        bins = np.geomspace(min+1e-04, max+0.001, numBins)
    else:
        bins = np.linspace(min, max+0.001, numBins)
    print('minimum intensity: ', min, ' maximum intensity: ', max)
    colorMap = cm.get_cmap('inferno', numBins)
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            for k in range(numBins-1):
                if(deltaPhi[i,j] > bins[k] and deltaPhi[i,j] < bins[k+1]):
                    ax.semilogx(taup[i,j], meanPhi[i,j], color=colorMap((numBins-k)/numBins), marker='s', markersize=15, lw=0)
    #data = np.loadtxt(dirName + "/gasFluidTradeoff.dat")
    #ax.plot(data[:,1], data[:,0], color='k', marker='o', markersize=6, markeredgewidth=1.2, fillstyle='none', lw=1)
    #data = np.loadtxt(dirName + "/stericActiveTradeoff.dat")#[0.3,0.7,0]
    #ax.plot(data[:,1], data[:,0], color='k', marker='v', markersize=8, markeredgewidth=1.2, fillstyle='none', lw=1)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Average$ $local$ $density,$ $\\langle \\varphi_l \\rangle$", fontsize=18)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=18)
    ax.set_ylim(0.22, 0.95)
    ax.plot(np.ones(50)*1e06, np.linspace(0,1.2,50), ls='dotted', color='k', lw=0.7)
    colorBar = cm.ScalarMappable(cmap=colorMap)
    cb = plt.colorbar(colorBar)
    label = "$\\Delta \\varphi_l$"#"$\\Delta \\varphi^2_{16}}$"
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=14, length=0)
    cb.ax.invert_yaxis()
    ticklabels = [np.format_float_positional(max, 2), np.format_float_positional(min, 2)]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=18, labelpad=10, rotation='horizontal')
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pPhaseDiagram-" + figureName
    if(log=='log'):
        figureName += "-" + log
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPhaseDiagramDeltaPressure(dirName, numBins, figureName, which='16'):
    figp, axp = plt.subplots(figsize=(7.5,5), dpi = 120)
    labelList = np.array(['$\\varphi=0.25$', '$\\varphi=0.27$', '$\\varphi=0.30$', '$\\varphi=0.35$', '$\\varphi=0.40$', '$\\varphi=0.45$', '$\\varphi=0.52$', '$\\varphi=0.58$', '$\\varphi=0.62$', '$\\varphi=0.67$', '$\\varphi=0.72$', '$\\varphi=0.78$', '$\\varphi=0.82$', '$\\varphi=0.85$', '$\\varphi=0.88$', '$\\varphi=0.91$', '$\\varphi=0.94$'])
    phi = np.array(['25', '27', '30', '35', '40', '45', '52', '58', '62', '67', '72', '78', '82', '85', '88', '91', '94'])
    Dr = np.array(['5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    colorList = cm.get_cmap('viridis', phi.shape[0])
    taup = np.zeros((phi.shape[0], Dr.shape[0]))
    # load the data
    voroPhi = np.zeros((phi.shape[0], Dr.shape[0]))
    deltaPressure = np.zeros((phi.shape[0], Dr.shape[0],2))
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            dirSample = dirName + 'thermal' + phi[i] + '/langevin/T0.001/iod10/active-langevin/Dr' + Dr[j] + '/dynamics/'
            fileName1 = dirSample + 'localVoroDensity-N' + which + '-stats.dat'
            fileName2 = dirSample + 'clusterPressure.dat'
            if(os.path.exists(fileName1) and os.path.exists(fileName2)):
                data = np.loadtxt(fileName1)
                voroPhi[i,j] = data[0]
                data = np.loadtxt(fileName2)
                deltaPressure[i,j,0] = np.mean(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
                deltaPressure[i,j,1] = np.std(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
                taup[i,j] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        if(float(phi[i]) < 89 and float(phi[i]) > 28):
            if(deltaPressure[i,j,0] < 0):
                axp.errorbar(taup[i,deltaPressure[i,:,0]!=0], -deltaPressure[i,deltaPressure[i,:,0]!=0,0], deltaPressure[i,deltaPressure[i,:,0]!=0,1], marker='o', capsize=3, color=colorList(i/phi.shape[0]), lw=1.2, label=labelList[i])
            else:
                axp.errorbar(taup[i,deltaPressure[i,:,0]!=0], deltaPressure[i,deltaPressure[i,:,0]!=0,0], deltaPressure[i,deltaPressure[i,:,0]!=0,1], marker='o', capsize=3, color=colorList(i/phi.shape[0]), lw=1.2, label=labelList[i])
    axp.set_xscale('log')
    axp.set_yscale('log')
    axp.tick_params(axis='both', labelsize=14)
    axp.set_ylabel("$Pressure$ $difference,$ $|\\Delta p| \\sigma^2$", fontsize=18)
    axp.set_xlabel("$Persistence$ $time,$ $\\tau_p \\sigma$", fontsize=18)
    axp.legend(fontsize=10, loc='best', ncol=2)
    figp.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pDeltaPressureVSPhi-" + figureName
    figp.savefig(figure1Name + ".png", transparent=True, format = "png")
    # assign color based on deltaPhi
    fig, ax = plt.subplots(figsize=(7.5,6), dpi = 120)
    ax.plot(np.ones(50)*1e06, np.linspace(0,1.2,50), ls='dotted', color='k', lw=0.7)
    colorId = np.zeros((phi.shape[0], Dr.shape[0]))
    colorMap = cm.get_cmap('inferno', numBins)
    pmagnitude = np.abs(deltaPressure[:,:,0])
    min = np.min(pmagnitude)
    max = np.max(pmagnitude)
    print('minimum intensity: ', min, ' maximum intensity: ', max)
    bins = np.geomspace(min, max, numBins)
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            for k in range(numBins-1):
                if(pmagnitude[i,j] > bins[k] and pmagnitude[i,j] < bins[k+1]):
                    if(deltaPressure[i,j,0] < 0):
                        ax.semilogx(taup[i,j], voroPhi[i,j], color=colorMap((numBins-k)/numBins), marker='s', markersize=15, lw=0, markeredgewidth=2.5, fillstyle='none')
                    else:
                        ax.semilogx(taup[i,j], voroPhi[i,j], color=colorMap((numBins-k)/numBins), marker='s', markersize=15, lw=0)
    #data = np.loadtxt(dirName + "/gasFluidTradeoff.dat")
    #ax.plot(data[:,1], data[:,0], color='k', marker='o', markersize=6, markeredgewidth=1.2, fillstyle='none', lw=1)
    #data = np.loadtxt(dirName + "/stericActiveTradeoff.dat")#[0.3,0.7,0]
    #ax.plot(data[:,1], data[:,0], color='k', marker='v', markersize=8, markeredgewidth=1.2, fillstyle='none', lw=1)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Average$ $local$ $density,$ $\\langle \\varphi_l \\rangle$", fontsize=18)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p \\sigma$", fontsize=18)
    ax.set_ylim(0.22, 0.95)
    colorBar = cm.ScalarMappable(cmap=colorMap)
    cb = plt.colorbar(colorBar)
    label = "$|\\Delta p|$"
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cb.ax.tick_params(labelsize=12, length=0)
    cb.ax.invert_yaxis()
    spacing = (max - min)/5
    ticklabels = [np.format_float_scientific(max, 2), np.format_float_scientific(min + 4*spacing, 2), np.format_float_scientific(min + 3*spacing, 2), np.format_float_scientific(min + 2*spacing, 2), np.format_float_scientific(min + spacing, 2), np.format_float_scientific(min, 2)]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=18, labelpad=10, rotation='horizontal')
    fig.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pDeltaPressurePhaseDiagram-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSurfaceTension(dirName, figureName, fixed='Dr', which='2e-04'):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        taup = np.zeros(dirList.shape[0])
    elif(fixed=="Dr"):
        #dirList = np.array(['30', '35', '40', '45', '52', '58', '62', '67', '72',  '78', '82', '85', '88', '91', '94'])
        dirList = np.array(['25', '28', '29', '30', '31', '32', '35', '40', '45', '50', '55', '60', '65', '70',  '75', '80', '82', '84', '86', '88', '90', '92', '94', '96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    tension = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(utils.readFromParams(dirSample, "Dr") * utils.readFromParams(dirSample, "sigma"))
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + "0." + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
        if(os.path.exists(dirSample + "/borderEnergy.dat")):
            data = np.loadtxt(dirSample + "/surfaceWork.dat")
            tension[d,0] = np.mean((data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]))
            tension[d,1] = np.std((data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]))
    ax.tick_params(axis='both', labelsize=14)
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pSurfaceTension-vsDr-" + figureName + "-phi" + which
        ax.set_xscale('log')
        #ax.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pSurfaceTension-vsPhi-" + figureName + "-Dr" + which
    ax.errorbar(x[tension[:,0]>0], tension[tension[:,0]>0,0], tension[tension[:,0]>0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Surface$ $tension,$ $\\gamma$", fontsize=18, labelpad=15)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSurfaceTensionVSTime(dirName, figureName):
    #numParticles = int(utils.readFromParams(dirName, "numParticles"))
    data = np.loadtxt(dirName + os.sep + "surfaceWork.dat")
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    ax.plot(data[1:,0], (data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]), linewidth=1.2, color='k')
    # plotting settings
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax.set_ylabel("$Surface$ $tension,$ $\\Delta W / \\Delta l_c$", fontsize=15)
    ax.legend(fontsize=12, loc='best')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pSurfaceTensionVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDeltaPVSPhi(dirName, figureName, which='2e-04'):
    fig, ax = plt.subplots(figsize = (7,5), dpi = 120)
    #dirList = np.array(['thermal25', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85', 'thermal85', 'thermal88'])
    dirList = np.array(['0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50'])
    deltap = np.zeros((dirList.shape[0], 2))
    clusterRad = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
        dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
        if(os.path.exists(dirSample + "/clusterPressure.dat")):
            data = np.loadtxt(dirSample + "/clusterPressure.dat")
            # delta pressure
            deltap[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
            deltap[d,1] = np.std(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
        if(os.path.exists(dirSample + "/voronoiArea.dat")):
            data = np.loadtxt(dirSample + "/voronoiArea.dat")
            # fluid radius
            clusterRad[d,0] = np.mean(np.sqrt(data[:,1]/np.pi))
            clusterRad[d,1] = np.std(np.sqrt(data[:,1]/np.pi))
    ax.tick_params(axis='both', labelsize=14)
    # plot delta versus fluid radius
    ax.errorbar(clusterRad[deltap[:,0]!=0,0], deltap[deltap[:,0]!=0,0]/clusterRad[deltap[:,0]!=0,0], deltap[deltap[:,0]!=0,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3)
    ax.set_xlabel('$Fluid$ $radius,$ $R_c$', fontsize=18)
    ax.set_ylabel('$Pressure$ $difference,$ $\\Delta p \\sigma^2$', fontsize=18)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pDeltaP-vsPhi-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

############################# system size analysis #############################
def plotSPDeltaPVSSystemSize(dirName, figureName, which='1.5e-04'):
    fig, ax = plt.subplots(figsize = (7,5), dpi = 120)
    dirList = np.array(['1024', '2048', '4096', '8192'])
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$'])
    numParticles = np.array([1024, 2048, 4096, 8192])
    deltap = np.zeros((dirList.shape[0], 2))
    clusterRad = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/thermal45/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
        if(os.path.exists(dirSample + "/clusterPressure.dat")):
            data = np.loadtxt(dirSample + "/clusterPressure.dat")
            sigma = utils.readFromDynParams(dirSample, "sigma")
            # delta pressure
            deltap[d,0] = np.mean(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
            deltap[d,1] = np.std(data[:,2] + data[:,3] + data[:,4] - data[:,5] - data[:,6] - data[:,7])
        if(os.path.exists(dirSample + "/voronoiArea.dat")):
            data = np.loadtxt(dirSample + "/voronoiArea.dat")
            # fluid radius
            clusterRad[d,0] = np.mean(np.sqrt(data[:,1]/np.pi))
            clusterRad[d,1] = np.std(np.sqrt(data[:,1]/np.pi))
    ax.tick_params(axis='both', labelsize=14)
    # plot delta versus fluid radius
    ax.errorbar(clusterRad[deltap[:,0]!=0,0], deltap[deltap[:,0]!=0,0]*clusterRad[deltap[:,0]!=0,0], deltap[deltap[:,0]!=0,1], color='k', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3)
    ax.set_xlabel('$Fluid$ $radius,$ $R_c$', fontsize=18)
    ax.set_ylabel('$Surface$ $tension,$ $\\gamma \\sigma$', fontsize=18)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pDeltaP-vsSystemSize-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPMIPSBoundsVSSystemSize(dirName, figureName, which='up'):
    fig, ax = plt.subplots(figsize = (7,3.5), dpi = 120)
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    dirList = np.array(['1024', '2048', '4096', '8192', '16384'])#, '32768'])
    numParticles = np.array([1024, 2048, 4096, 8192, 16384])#, 32768])
    bounds = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/densitySweep/"
        if(os.path.exists(dirSample + "/MIPSBounds.dat")):
            data = np.loadtxt(dirSample + "/MIPSBounds.dat")
            bounds[d,0] = data[1]
            bounds[d,1] = data[2]
    if(which=='down'):
        label = '$Lower$ $bound$'
        ax.semilogx(numParticles, bounds[:,0], color='k', marker='v', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, ls='dashdot', label='$Lower$ $bound$')
        figureName += '-lower'
    elif(which=='up'):
        label = '$Upper$ $bound$'
        ax.semilogx(numParticles, bounds[:,1], color='k', marker='^', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, label='$Upper$ $bound$')
        figureName += '-upper'
    else:
        label = '$MIPS$ $bounds$'
        ax.semilogx(numParticles, bounds[:,1], color='k', marker='^', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, label='$Upper$ $bound$')
        ax.semilogx(numParticles, bounds[:,0], color='k', marker='v', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, ls='dashdot', label='$Lower$ $bound$')
        figureName += '-both'
        ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xscale('log', basex=2)
    ax.set_xlabel('$System$ $size,$ $N$', fontsize=18)
    ax.set_ylabel(label, fontsize=18)
    ax.set_xticks(numParticles)
    ax.set_xticklabels(dirList)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pBounds-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterTensionVSSystemSize(dirName, figureName):
    fig, ax = plt.subplots(figsize = (7,5), dpi = 120)
    labelList = np.array(['$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    dirList = np.array(['4096', '8192', '16384'])#, '32768'])
    numParticles = np.array([4096, 8192, 16384])#, 32768])
    tension = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        dirSample = dirName  + os.sep + dirList[d] + "-2d/densitySweep/0.31/active-langevin/Dr2e-04/dynamics/"
        if(os.path.exists(dirSample + "/clusterPressure.dat") and os.path.exists(dirSample + "/voronoiArea.dat")):
            pressure = np.loadtxt(dirSample + "/clusterPressure.dat")
            area = np.loadtxt(dirSample + "/voronoiArea.dat")
            tension[d,0] = np.mean((pressure[:,2] + pressure[:,3] + pressure[:,4] - pressure[:,5] - pressure[:,6] - pressure[:,7]) * np.sqrt(area[:,0]/np.pi))
            tension[d,1] = np.std((pressure[:,2] + pressure[:,3] + pressure[:,4] - pressure[:,5] - pressure[:,6] - pressure[:,7]) * np.sqrt(area[:,0]/np.pi))
    ax.errorbar(numParticles, tension[:,0], tension[:,1], color='k', marker='o', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(-282, 2)
    ax.set_xscale('log', basex=2)
    ax.set_xlabel('$System$ $size,$ $N$', fontsize=18)
    ax.set_ylabel('$Surface$ $tension,$ $\\gamma = \\Delta p R_c$', fontsize=18)
    ax.set_xticks(numParticles)
    ax.set_xticklabels(dirList)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pClusterTension-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSystemSize(dirName, figureName, which = "tension"):
    fig, ax = plt.subplots(figsize=(7.5,4.5), dpi = 120)
    dirList = np.array(['1024', '2048', '4096', '8192', '16384', '32768'])
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    numParticles = np.array([1024, 2048, 4096, 8192, 16384, 32678])
    clusterDensity = np.zeros((dirList.shape[0],2))
    border = np.zeros((dirList.shape[0],2))
    energy = np.zeros((dirList.shape[0],2))
    tension = np.zeros((dirList.shape[0],2))
    sigma = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/thermal45/langevin/T0.001/iod10/active-langevin/Dr1e-03-f0200/dynamics/"
        if(os.path.exists(dirSample + "borderEnergy.dat")):
            sigma[d] = np.mean(np.loadtxt(dirSample + 'particleRad.dat'))
            data = np.loadtxt(dirSample + "borderEnergy.dat")
            #clusterDensity[d,0] = np.mean(data)
            #clusterDensity[d,1] = np.std(data)
            border[d,0] = np.mean(data[:,1])
            border[d,1] = np.std(data[:,1])#/np.sqrt(data[:,0].shape[0])
            energy[d,0] = np.mean(data[:,2])
            energy[d,1] = np.std(data[:,2])#/np.sqrt(data[:,0].shape[0])
            tension[d,0] = np.mean(data[:,2]/data[:,1])
            tension[d,1] = np.std(data[:,2]/data[:,1])#/np.sqrt(data[:,0].shape[0])
    if(which=='tension'):
        ax.errorbar(numParticles, tension[:,0]*sigma, tension[:,1]*sigma, color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
        ax.set_ylabel("$Surface$ $tension,$ $\\gamma_c \\sigma$", fontsize=18)
    elif(which=='border'):
        ax.errorbar(numParticles, border[:,0], border[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
        ax.set_ylabel("$Border$ $length,$ $L_c$", fontsize=18)
    elif(which=='energy'):
        ax.errorbar(numParticles, energy[:,0], energy[:,1], color='k', lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4)
        ax.set_ylabel("$Border$ $energy,$ $E_c$", fontsize=18)
    #x = np.linspace(numParticles[0], numParticles[-1], 100)
    #ax.plot(x, 0.05*np.log(x), linestyle='--', color='b')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$System$ $size,$ $N$", fontsize=18)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_ylim(0.00082,)
    #ax.set_ylabel("$Cluster$ $density,$ $\\langle \\varphi_c \\rangle$", fontsize=18)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pCluster-vsSystemSize-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDynamicsVSSystemSize(dirName, figureName):
    fig, ax = plt.subplots(figsize=(7.5,4.5), dpi = 120)
    dirList = np.array(['1024', '2048', '4096', '8192'])#, '16384', '32768'])
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    numParticles = np.array([1024, 2048, 4096, 8192, 16384, 32678])
    colorList = cm.get_cmap('inferno', dirList.shape[0]+1)
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/thermal45/langevin/T0.001/iod10/dynamics/"
        if(os.path.exists(dirSample + "logVelCorr.dat")):
            sigma = np.mean(np.loadtxt(dirSample + 'particleRad.dat'))
            data = np.loadtxt(dirSample + "logVelCorr.dat")
            ax.errorbar(data[:,0]/sigma, data[:,1], data[:,2], color=colorList(d/dirList.shape[0]), lw=1.2, marker='o', markersize = 7, fillstyle='none', elinewidth=1, capsize=4, label=labelList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Elapsed$ $time,$ $\\Delta t / \\sigma$", fontsize=18)
    ax.set_ylabel("$C_{vv}(\\Delta t)$", fontsize=18)
    ax.set_xscale('log')
    ax.legend(fontsize=12, loc='upper right')
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pDynamics-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

######################### attractive droplet analysis ##########################
def plotDropletPressureVSTime(dirName, figureName):
    pressure = np.loadtxt(dirName + os.sep + "pressure.dat")
    fig, ax = plt.subplots(1, 2, sharey=True, figsize = (11, 5), dpi = 120)
    # droplet particles
    ax[0].plot(pressure[:,0], pressure[:,1], linewidth=1.5, color='k', label="$virial$")
    ax[0].plot(pressure[:,0], pressure[:,2], linewidth=1.5, color='r', label="$thermal$")
    ptot = (pressure[:,1] + pressure[:,2])
    ax[0].plot(pressure[:,0], ptot, linewidth=1.5, color='b', ls='dotted', label="$total$")
    # dilute particles
    ax[1].plot(pressure[:,0], pressure[:,4], linewidth=1.5, color='k', ls='--', label="$virial$")
    ax[1].plot(pressure[:,0], pressure[:,5], linewidth=1.5, color='r', ls='--', label="$thermal$")
    ptot = (pressure[:,4] + pressure[:,5])
    ax[1].plot(pressure[:,0], ptot, linewidth=1.5, color='b', ls='dotted', label="$total$")
    # plotting settings
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[0].set_ylabel("$Pressure$", fontsize=15)
    ax[0].legend(fontsize=12, loc='best')
    ax[0].set_yscale('log')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    #figureName = "/home/francesco/Pictures/soft/pClusterPressure-" + figureName
    #fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDropletProfile(dirName, figureName, shift=0, which='pressure'):
    fig, ax = plt.subplots(figsize = (8,4), dpi = 120)
    if(os.path.exists(dirName + "/pressureProfile.dat")):
        data = np.loadtxt(dirName + "/pressureProfile.dat")
        data[:,1] = np.roll(data[:,1], shift)
        data[:,2] = np.roll(data[:,2], shift)
        data[:,3] = np.roll(data[:,3], shift)
        data[:,4] = np.roll(data[:,4], shift)
        data[:,5] = np.roll(data[:,5], shift)
        data[:,6] = np.roll(data[:,6], shift)
        data[:,7] = np.roll(data[:,7], shift)
        half = int(data.shape[0]/2)
        print("surface tension: ", np.sum(data[:half,0] * (data[:half,2] + data[:half,5] - data[:half,3] - data[:half,6])))
        if(which=='pressure'):
            ax.plot(data[:,0], data[:,1], lw=1.5, color='k', ls='solid', label='$Steric$')
            ax.plot(data[:,0], data[:,4], lw=1.5, color='r', ls='dashed', label='$Thermal$')
        elif(which=='delta'):
            ax.plot(data[:,0], data[:,2] - data[:,3], lw=1.5, color='k', ls='solid', label='$Steric$')
            ax.plot(data[:,0], data[:,5] - data[:,6], lw=1.5, color='r', ls='dashed', label='$Thermal$')
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Radial$ $distance,$ $r$", fontsize=18)
    ax.set_xlabel("$Position,$ $x$", fontsize=18)
    if(which=="pressure"):
        ax.set_ylabel("$Pressure,$ $p \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,4])-0.2, np.max(data[:,1])+0.4)
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pProfile-" + figureName + ".png"
    elif(which=="delta"):
        ax.set_ylabel("$\\Delta p^\ast = p_{xx} - p_{yy} \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,2] - data[:,3])-0.2, np.max(data[:,2] - data[:,3])+0.6)
        figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/deltaProfile-" + figureName + ".png"
    ax.legend(fontsize=13, loc='upper right', ncol=4)
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPDropletPressure(dirName, figureName, fixed='temp', which='0.0023'):
    fig1, ax1 = plt.subplots(figsize = (7,4), dpi = 120)
    fig2, ax2 = plt.subplots(figsize = (7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        dirList = np.array(['0.0023', '0.02', '0.2'])
        temp = np.zeros(dirList.shape[0])
    elif(fixed=="temp"):
        dirList = np.array(['thermal25', 'thermal27', 'thermal30'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    border = np.zeros((dirList.shape[0], 3, 2))
    droplet = np.zeros((dirList.shape[0], 3, 2))
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "/langevin-u/T" + dirList[d] + "-u0.035/dynamics/"
            temp[d] = np.mean(np.loadtxt(dirSample + "energy.dat")[:,4])
        elif(fixed=="temp"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin-u/T" + which + "-u0.035/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        if(os.path.exists(dirSample + "/pressure.dat")):
            data = np.loadtxt(dirSample + "/pressure.dat")
            # border
            border[d,0,0] = np.mean(data[:,1])
            border[d,0,1] = np.std(data[:,1])
            border[d,1,0] = np.mean(data[:,2])
            border[d,1,1] = np.std(data[:,2])
            border[d,2,0] = np.mean(data[:,1] + data[:,2])
            border[d,2,1] = np.std(data[:,1] + data[:,2])
            # droplet
            droplet[d,0,0] = np.mean(data[:,4])
            droplet[d,0,1] = np.std(data[:,4])
            droplet[d,1,0] = np.mean(data[:,5])
            droplet[d,1,1] = np.std(data[:,5])
            droplet[d,2,0] = np.mean(data[:,4] + data[:,5])
            droplet[d,2,1] = np.std(data[:,4] + data[:,5])
    if(fixed=="phi"):
        x = temp
        xlabel = "$Temperature,$ $T$"
        loc1 = 'lower right'
        loc2 = 'upper left'
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pDropletPressure-vsT-" + figureName + "-phi" + which
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pTotalDropletPressure-vsT-" + figureName + "-phi" + which
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        loc1 = 'lower right'
        loc2 = 'upper left'
        figure1Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pDropletPressure-vsPhi-" + figureName + "-T" + which
        figure2Name = "/home/francesco/Pictures/soft/nve-nvt-nva/pTotalDropletPressure-vsPhi-" + figureName + "-T" + which
        #ax1.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    # droplet
    ax1.errorbar(x, droplet[:,0,0], droplet[:,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
    ax1.errorbar(x, droplet[:,1,0], droplet[:,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
    # border
    ax1.errorbar(x, border[:,0,0], border[:,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, ls = 'dotted')
    ax1.errorbar(x, border[:,1,0], border[:,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, ls = 'dotted')
    # total
    ax2.errorbar(x, droplet[:,2,0], droplet[:,2,1], lw=1.2, color='b', marker='s', markersize=8, fillstyle='none', capsize=3, label='$Droplet$')
    #ax2.errorbar(x, border[:,2,0], border[:,2,1], lw=1.2, color='g', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Border$')
    #ax.set_xlim(5.8e-06, 2.8e03)
    ax1.set_xlabel(xlabel, fontsize=18)
    ax1.set_ylabel("$Pressure,$ $P \\sigma^2$", fontsize=18)
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("$Pressure,$ $P \\sigma^2$", fontsize=18)
    ax1.legend(fontsize=12, loc='best')
    ax2.legend(fontsize=12, loc='best')
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig2.savefig(figure1Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPDropletTensionVSSystemSize(dirName, figureName):
    fig, ax = plt.subplots(figsize = (7,5), dpi = 120)
    labelList = np.array(['$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    dirList = np.array(['4096', '8192', '16384'])#, '32768'])
    numParticles = np.array([4096, 8192, 16384])#, 32768])
    tension = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        dirSample = dirName  + os.sep + dirList[d] + "-2d/densitySweep/0.31/active-langevin/Dr2e-04/langevin-u/T0.004-u0.1/dynamics/"
        if(os.path.exists(dirSample + "/dropletPressure.dat") and os.path.exists(dirSample + "/voronoiArea.dat")):
            pressure = np.loadtxt(dirSample + "/dropletPressure.dat")
            area = np.loadtxt(dirSample + "/voronoiArea.dat")
            tension[d,0] = np.mean((pressure[:,1] + pressure[:,2] - pressure[:,3] - pressure[:,4]) * np.sqrt(area[:,0]/np.pi))
            tension[d,1] = np.std((pressure[:,1] + pressure[:,2] - pressure[:,3] - pressure[:,4]) * np.sqrt(area[:,0]/np.pi))
    ax.errorbar(numParticles, tension[:,0], tension[:,1], color='k', marker='o', markersize=10, markeredgewidth=1.2, fillstyle='none', lw=1.2, capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(-0.1, 82)
    ax.set_xscale('log', basex=2)
    ax.set_xlabel('$System$ $size,$ $N$', fontsize=18)
    ax.set_ylabel('$Surface$ $tension,$ $\\gamma = \\Delta p R_c$', fontsize=18)
    ax.set_xticks(numParticles)
    ax.set_xticklabels(dirList)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/nve-nvt-nva/pDropletTension-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()


if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

############################ local density analysis ############################
    if(whichPlot == "simplex"):
        figureName = sys.argv[3]
        pad = float(sys.argv[4])
        logy = sys.argv[5]
        plotSimplexDensity(dirName, figureName, pad, logy)

    elif(whichPlot == "fitphi"):
        figureName = sys.argv[3]
        numBins = sys.argv[4]
        fitPhiPDF(dirName, figureName, numBins)

    elif(whichPlot == "fitphi2"):
        figureName = sys.argv[3]
        numBins = sys.argv[4]
        fitPhiPDF2(dirName, figureName, numBins)

    elif(whichPlot == "phiwindow"):
        figureName = sys.argv[3]
        weight = sys.argv[4]
        plotSPDensityVSWindowSize(dirName, figureName, weight)

    elif(whichPlot == "localphi"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPLocalDensity(dirName, figureName, fixed, which)

    elif(whichPlot == "comparelocalphi"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPCompareLocalDensity(dirName, figureName, which)

########################## cluster static properties ###########################
    elif(whichPlot == "clusterlength"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterLengthscale(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterpaircorr"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterPairCorr(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterflu"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterFluctuations(dirName, figureName, fixed, which)

    elif(whichPlot == "numphiflu"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPNumberDensityFluctuations(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterpdf"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        numBins = int(sys.argv[6])
        plotSPClusterDistribution(dirName, figureName, fixed, which, numBins)

    elif(whichPlot == "clustertime"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterSizeVSTime(dirName, figureName, fixed, which)

    elif(whichPlot == "shapetime"):
        figureName = sys.argv[3]
        plotClusterShapeVSTime(dirName, figureName)

    elif(whichPlot == "clustershape"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterShape(dirName, figureName, fixed, which)

    elif(whichPlot == "clustermixing"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterMixingTime(dirName, figureName, fixed, which)

######################### cluster pressure properties ##########################
    elif(whichPlot == "clusterptime"):
        figureName = sys.argv[3]
        bound = sys.argv[4]
        prop = sys.argv[5]
        plotClusterPressureVSTime(dirName, figureName, bound, prop)

    elif(whichPlot == "pprofile"):
        figureName = sys.argv[3]
        shift = int(sys.argv[4])
        which = sys.argv[5]
        plotSPPressureProfile(dirName, figureName, shift, which)

    elif(whichPlot == "clusterphi"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterDensity(dirName, figureName, fixed, which)

    elif(whichPlot == "gammatime"):
        figureName = sys.argv[3]
        plotSPClusterSurfaceTensionVSTime(dirName, figureName)

    elif(whichPlot == "clustergamma"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterSurfaceTension(dirName, figureName, fixed, which)

    elif(whichPlot == "forcevel"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        plotSPForceVelMagnitude(dirName, figureName, fixed)

    elif(whichPlot == "tradeoff"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterTradeoff(dirName, figureName, which)

    elif(whichPlot == "totalp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPTotalPressure(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        inter = sys.argv[5]
        which = sys.argv[6]
        plotSPClusterPressure(dirName, figureName, fixed, inter, which)

    elif(whichPlot == "phased"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        which = sys.argv[5]
        log = sys.argv[6]
        plotSPPhaseDiagram(dirName, numBins, figureName, which, log)

    elif(whichPlot == "phasedeltap"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        which = sys.argv[5]
        plotSPPhaseDiagramDeltaPressure(dirName, numBins, figureName, which)

    elif(whichPlot == "deltapphi"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPDeltaPVSPhi(dirName, figureName, which)

############################# system size analysis #############################
    elif(whichPlot == "deltapnum"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPDeltaPVSSystemSize(dirName, figureName, which)

    elif(whichPlot == "clustersize"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterSystemSize(dirName, figureName, which)

    elif(whichPlot == "dynsize"):
        figureName = sys.argv[3]
        plotSPDynamicsVSSystemSize(dirName, figureName)

    elif(whichPlot == "bounds"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPMIPSBoundsVSSystemSize(dirName, figureName, which)

    elif(whichPlot == "activetension"):
        figureName = sys.argv[3]
        plotSPClusterTensionVSSystemSize(dirName, figureName)

######################### attractive droplet analysis ##########################
    elif(whichPlot == "dropletptime"):
        figureName = sys.argv[3]
        plotDropletPressureVSTime(dirName, figureName)

    elif(whichPlot == "dprofile"):
        figureName = sys.argv[3]
        shift = int(sys.argv[4])
        which = sys.argv[5]
        plotSPDropletProfile(dirName, figureName, shift, which)

    elif(whichPlot == "dropletp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPDropletPressure(dirName, figureName, fixed, which)

    elif(whichPlot == "droptension"):
        figureName = sys.argv[3]
        plotSPDropletTensionVSSystemSize(dirName, figureName)