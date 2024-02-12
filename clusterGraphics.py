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
from scipy.special import kn
import itertools
import sys
import os
import utils
import spCluster as cluster
import spInterface as interface

def lineFit(x, a, b):
    return a*x + b

def powerLaw(x, a, b):
    return a*x**b

def powerLawMass(x, a, b, c):
    return a*(x**b + c)

def hyperbolicTan(x, a, b, x0, w):
    return 0.5*(a+b) - 0.5*(a-b)*np.tanh(2*(x-x0)/w)

def hyperbolicSecSq(x, a, x0, w):
    return a/np.cosh(2*(x-x0)/w)*2/w

def curveCumSum(x, a, b, c):
    return 1 - c*np.exp(-(x*a)**b)

def besselFunc(x, a, b):
    return b*kn(0,(x/a))

def curveCvv(x, a, b):
    return a * np.exp(-b*x)

def curve4Poly(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def curve2Poly(x, a, b, c):
    return a*x**2 + b*x + c

################################################################################
############################ Local density analysis ############################
################################################################################
def plotSimplexDensity(dirName, figureName, threshold=0.76, which=False, logy=False, pad=1):
    fig1, ax1 = plt.subplots(figsize=(4.5,4), dpi=150)
    fig2, ax2 = plt.subplots(figsize=(4.5,4), dpi=150)
    if not(os.path.exists(dirName + os.sep + 'simplexList.dat')):
        cluster.computeDelaunayCluster(dirName, threshold=threshold, filter='filter')
    simplexList = np.loadtxt(dirName + os.sep + 'simplexList.dat')
    denseSimplexList = simplexList[:,0]
    borderSimplexList = simplexList[:,1]
    simplexDensity = simplexList[:,3]
    if(which == 'fluid'):
        simplexDensity = simplexDensity[denseSimplexList==1]
        figureName = figureName + '-' + which
    elif(which == 'gas'):
        simplexDensity[borderSimplexList==1] = 0
        simplexDensity = simplexDensity[denseSimplexList==0]
        figureName = figureName + '-' + which
    elif(which == 'border'):
        simplexDensity = simplexDensity[borderSimplexList==1]
    simplexDensity = simplexDensity[simplexDensity>0]
    simplexDensity = np.sort(simplexDensity)[:-1]
    ax1.plot(np.arange(1, simplexDensity.shape[0]+1, 1), np.sort(simplexDensity), color='k', marker='.', markersize=8, lw=0.8, fillstyle='none')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.set_xlabel('$Simplex$ $index$', fontsize=16)
    ax1.set_ylabel('$\\varphi^{Simplex}$', fontsize=16)
    numBins = 100
    pdf, edges = np.histogram(simplexDensity, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax2.plot(edges, pdf, color='k', marker='.', markersize=8, lw=0.8, fillstyle='none')
    y = np.linspace(np.min(pdf)-pad, np.max(pdf)+pad, 100)
    if(logy == 'logy'):
        ax2.set_yscale('log')
        ax2.set_ylim(6.4e-03, 50.6)
        y = np.linspace(1e-04, 100, 100)
    else:
        ax2.set_ylim(-2.8, 43.2)
        y = np.linspace(-5, 50, 100)
    #ax2.plot(np.ones(100)*0.906899682, y, ls='dotted', color='k', lw=1, label='$Triangular$ $lattice$')
    #ax2.plot(np.ones(100)*0.785398163, y, ls='dashdot', color='k', lw=1, label='$Square$ $lattice$')
    #ax2.legend(fontsize=10, loc='best')
    ax2.tick_params(axis='both', labelsize=12)
    ax2.set_ylabel('$PDF(\\varphi^{Simplex})$', fontsize=16)
    ax2.set_xlabel('$\\varphi^{Simplex}$', fontsize=16)
    fig1.tight_layout()
    fig2.tight_layout()
    fig2.savefig("/home/francesco/Pictures/soft/mips/sPDF-" + figureName + ".png", transparent=True, format="png")
    plt.show()

def plotSimplexLabels(dirName, figureName, threshold=0.76, filter=False, logy=False, pad = 1):
    numBins = 100
    fig, ax = plt.subplots(2, 2, sharex = 'col', sharey = 'col', figsize=(10,6), dpi=150)
    if not(os.path.exists(dirName + os.sep + 'simplexList.dat')):
        cluster.computeDelaunayCluster(dirName, threshold=threshold, filter=filter)
    simplexList = np.loadtxt(dirName + os.sep + 'simplexList.dat')
    simplexDensity = simplexList[:,3]
    # dilute surrounded by dense
    dilute3dense =  np.loadtxt(dirName + os.sep + 'delaunayLabels/dilute3dense.dat')
    dilute2dense =  np.loadtxt(dirName + os.sep + 'delaunayLabels/dilute2dense.dat')
    dilute1dense =  np.loadtxt(dirName + os.sep + 'delaunayLabels/dilute1dense.dat')
    dilute3dense = simplexDensity[dilute3dense==1]
    dilute2dense = simplexDensity[dilute2dense==1]
    dilute1dense = simplexDensity[dilute1dense==1]
    ax[0,0].plot(np.arange(1, dilute3dense.shape[0]+1, 1), np.sort(dilute3dense), color='r', marker='o', markersize=4, lw=0.8, fillstyle='none', label="$Dilute,$ $3$ $dense$")
    ax[0,0].plot(np.arange(1, dilute2dense.shape[0]+1, 1), np.sort(dilute2dense), color='g', marker='v', markersize=4, lw=0.8, fillstyle='none', label="$Dilute,$ $2$ $dense$")
    ax[0,0].plot(np.arange(1, dilute1dense.shape[0]+1, 1), np.sort(dilute1dense), color='b', marker='s', markersize=4, lw=0.8, fillstyle='none', label="$Dilute,$ $1$ $dense$")
    pdf, edges = np.histogram(dilute3dense, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax[0,1].plot(edges[pdf>0], pdf[pdf>0], color='r', marker='o', markersize=6, lw=0.8, fillstyle='none')
    pdf, edges = np.histogram(dilute2dense, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax[0,1].plot(edges[pdf>0], pdf[pdf>0], color='g', marker='v', markersize=6, lw=0.8, fillstyle='none')
    pdf, edges = np.histogram(dilute1dense, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax[0,1].plot(edges[pdf>0], pdf[pdf>0], color='b', marker='s', markersize=6, lw=0.8, fillstyle='none')
    # dense surrounded by dilute
    dense3dilute =  np.loadtxt(dirName + os.sep + 'delaunayLabels/dense3dilute.dat')
    dense2dilute =  np.loadtxt(dirName + os.sep + 'delaunayLabels/dense2dilute.dat')
    dense1dilute =  np.loadtxt(dirName + os.sep + 'delaunayLabels/dense1dilute.dat')
    dense3dilute = simplexDensity[dense3dilute==1]
    dense2dilute = simplexDensity[dense2dilute==1]
    dense1dilute = simplexDensity[dense1dilute==1]
    ax[1,0].plot(np.arange(1, dense3dilute.shape[0]+1, 1), np.sort(dense3dilute), color='r', marker='o', markersize=4, lw=0.8, fillstyle='none', label="$Dense,$ $3$ $dilute$")
    ax[1,0].plot(np.arange(1, dense2dilute.shape[0]+1, 1), np.sort(dense2dilute), color='g', marker='v', markersize=4, lw=0.8, fillstyle='none', label="$Dense,$ $2$ $dilute$")
    ax[1,0].plot(np.arange(1, dense1dilute.shape[0]+1, 1), np.sort(dense1dilute), color='b', marker='s', markersize=4, lw=0.8, fillstyle='none', label="$Dense,$ $1$ $dilute$")
    pdf, edges = np.histogram(dense3dilute, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax[1,1].plot(edges[pdf>0], pdf[pdf>0], color='r', marker='o', markersize=6, lw=0.8, fillstyle='none')
    pdf, edges = np.histogram(dense2dilute, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax[1,1].plot(edges[pdf>0], pdf[pdf>0], color='g', marker='v', markersize=6, lw=0.8, fillstyle='none')
    pdf, edges = np.histogram(dense1dilute, bins=np.linspace(0, 1, numBins), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax[1,1].plot(edges[pdf>0], pdf[pdf>0], color='b', marker='s', markersize=6, lw=0.8, fillstyle='none')
    # figure settings
    for i in range(2):
        ax[i,0].legend(fontsize=10, loc='best')
        ax[i,0].tick_params(axis='both', labelsize=12)
        ax[i,0].set_ylabel('$\\varphi^{Simplex}$', fontsize=16)
        ax[i,0].set_ylim(-0.06, 1.06)
    ax[1,0].set_xlabel('$Simplex$ $index$', fontsize=16)
    # plot triangular and square densities
    y = np.linspace(np.min(pdf)-pad, np.max(pdf)+pad, 100)
    if(logy == 'logy'):
        ax[0,1].set_yscale('log')
        ax[0,1].set_ylim(6.4e-03, 50.6)
        y = np.linspace(1e-04, 100, 100)
    else:
        ax[0,1].set_ylim(-2.8, 43.2)
        y = np.linspace(-5, 50, 100)
    for i in range(2):
        ax[i,1].plot(np.ones(100)*0.906899682, y, ls='dotted', color='k', lw=1, label='$Triangular$ $lattice$')
        ax[i,1].plot(np.ones(100)*0.785398163, y, ls='dashdot', color='k', lw=1, label='$Square$ $lattice$')
        ax[i,1].tick_params(axis='both', labelsize=12)
        ax[i,1].set_ylabel('$PDF(\\varphi^{Simplex})$', fontsize=16)
        ax[i,1].set_xlim(-0.02, 1.02)
    ax[1,1].legend(fontsize=10, loc='best')
    ax[1,1].set_xlabel('$\\varphi^{Simplex}$', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0)
    plt.savefig("/home/francesco/Pictures/soft/mips/simplexLabels-" + figureName + ".png", transparent=True, format="png")
    plt.show()

def fitPhiPDF(dirName, figureName, numBins):
    fig1, ax1 = plt.subplots(figsize=(5,4), dpi = 120)
    fig2, ax2 = plt.subplots(figsize=(6,3.5), dpi = 120)
    if not(os.path.exists(dirName + "localDensity-N" + str(numBins) + ".dat")):
        cluster.averageLocalDensity(dirName, numBins)
    data = np.loadtxt(dirName + "localDensity-N" + str(numBins) + ".dat")
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
    figure1Name = "/home/francesco/Pictures/soft/mips/pfitPhiFreeEnergy-" + figureName
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    figure2Name = "/home/francesco/Pictures/soft/mips/pfitPhiPDF-" + figureName
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
    figureName = "/home/francesco/Pictures/soft/mips/pfitPhiPDF-F-" + figureName
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
        figureName = "/home/francesco/Pictures/soft/mips/pWeightedWindow-" + figureName
    else:
        figureName = "/home/francesco/Pictures/soft/mips/pWindow-" + figureName
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
        figureName = "/home/francesco/Pictures/soft/mips/pLocalPhi-vsDr-" + figureName
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
        figureName = "/home/francesco/Pictures/soft/mips/pLocalPhi-vsPhi-" + figureName
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
    figureName = "/home/francesco/Pictures/soft/mips/pCompareLocalPhi-vsDr-" + figureName
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

################################################################################
########################## Cluster static properties ###########################
################################################################################
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
        figureName = "/home/francesco/Pictures/soft/mips/pClusterLength-vsPhi-" + figureName + "-iod" + which
        ax.plot(np.linspace(phi[0], phi[-1], 50), np.ones(50), color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterLength-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterLength-vsDamping-" + figureName + "-Dr" + which
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
        figureName = "/home/francesco/Pictures/soft/mips/pClusterLength-vsPhi-" + figureName + "-iod" + which
        ax.plot(np.linspace(phi[0], phi[-1], 50), np.ones(50), color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterLength-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterLength-vsDamping-" + figureName + "-Dr" + which
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
        figure1Name = "/home/francesco/Pictures/soft/mips/pClusterNum-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/mips/pClusterSize-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = 1/Dr
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/soft/mips/pClusterNum-vsDr-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/mips/pClusterSize-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/soft/mips/pClusterNum-vsDamping-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/soft/mips/pClusterSize-vsDamping-" + figureName + "-Dr" + which
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
        dirList = np.array(['1', '1e-01', '1e-02', '1e-03', '1e-04', '1e-05', '1e-06', '1e-07'])
        labelList = np.array(['$1$', '$0.1$', '$0.01$', '$0.001$', '$10^{-4}$', '$10^{-5}$', '$10^{-6}$', '$10^{-7}$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        taup = np.zeros(dirList.shape[0])
    else:
        print("please specify fixed parameter")
    for d in range(dirList.shape[0]):
        if(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        if(os.path.exists(dirSample + "averageLocalNumberDensity.dat")):
            #if(d<8):
            data = np.loadtxt(dirSample + "averageLocalNumberDensity.dat")
            #ax.errorbar(data[:,1], data[:,7], data[:,8], lw=1.2, color=colorList(d/dirList.shape[0]), marker='o', fillstyle='none', capsize=3, label=labelList[d])
            ax.errorbar(data[:,1], data[:,3]/data[:,1], data[:,4]/data[:,1], lw=1.2, color=colorList(d/dirList.shape[0]), marker='o', fillstyle='none', capsize=3, label=labelList[d])
            #ax.plot(data[:,1], data[:,4], lw=1.2, color=colorList((dirList.shape[0]-d-1)/dirList.shape[0]), marker='o', fillstyle='none', label=labelList[d])
    ax.set_ylim(0.57, 2886)
    #ax.set_ylim(4.6e-06, 0.34)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_ylabel("$Density$ $variance,$ $\\Delta \\varphi^2$", fontsize=18)
    ax.set_ylabel("$\\frac{\\Delta N_c}{\\langle N_c \\ rangle}$", fontsize=18)
    ax.set_xlabel("$Cluster$ $size,$ $\\langle N_c \\rangle$", fontsize=18)
    if(fixed=="Dr"):
        ax.legend(fontsize=10, loc='lower left', ncol=3)
        figureName = "/home/francesco/Pictures/soft/mips/pNumberPhiVar-vsPhi-" + figureName
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
        figureName = "/home/francesco/Pictures/soft/mips/pNumberPhiVar-vsDr-" + figureName
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterPDF(dirName, figureName, numBins=40, which='simplex', cdf='cdf'):
    #print("taup:", 1/(utils.readFromDynParams(dirName, "Dr")*utils.readFromDynParams(dirName, "sigma")))
    fig, ax = plt.subplots(figsize=(5,3.5), dpi = 120)
    if(which=='simplex'):
        if not(os.path.exists(dirName + "simplexDistribution.dat")):
            cluster.computeSimplexClusterDistribution(dirName)
        data = np.loadtxt(dirName + "simplexDistribution.dat")
        data[:,0] /= 2*utils.readFromParams(dirName, "numParticles")
    elif(which=='particle'):
        if not(os.path.exists(dirName + "clusterDistribution.dat")):
            cluster.computeDelaunayClusterDistribution(dirName)
        data = np.loadtxt(dirName + "clusterDistribution.dat")
        data[:,0] /= utils.readFromParams(dirName, "numParticles")
    if(cdf=='cdf'):
        x = np.arange(1, data[:,1].shape[0]+1, 1)
        ax.semilogy(x, np.sort(data[:,1]), color='k', marker='.', markersize=8, lw=0.8, fillstyle='none')
        ax.set_xlabel("$Cluster$ $index$", fontsize=16)
        ax.set_ylabel("$Cluster$ $size,$ $A_c / A_{box}$", fontsize=16)
        ax.set_xticks(np.linspace(np.min(x), np.max(x), 4))
        ax.set_ylim(8e-07, 0.9)
    else:
        pdf, edges = np.histogram(data[:,1], bins=np.geomspace(np.min(data[:,1]), np.max(data[:,1]), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        ax.semilogx(edges, pdf, lw=1.2, color='k', marker='o', fillstyle='none')
        ax.set_xlabel("$Cluster$ $size,$ $A_c / A_{box}$", fontsize=16)
        ax.set_ylabel("$Distribution,$ $P(A_c / A_{box})$", fontsize=16)
        ax.set_xlim(2.2e-06, 0.55)
        ax.set_ylim(-1.2e03, 2.9e04)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pClusterPDF-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterShapePDF(dirName, figureName, numBins=40, cdf='cdf'):
    #print("taup:", 1/(utils.readFromDynParams(dirName, "Dr")*utils.readFromDynParams(dirName, "sigma")))
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    if not(os.path.exists(dirName + "clusterShape.dat")):
        cluster.computeClusterDelaunayShape(dirName)
    data = np.loadtxt(dirName + "clusterShape.dat")
    if(cdf=='cdf'):
        x = np.arange(1, data.shape[0]+1, 1)
        ax.plot(x, np.sort(data), color='k', marker='.', markersize=8, lw=0.8, fillstyle='none')
        ax.set_xlabel("$Cluster$ $index$", fontsize=16)
        ax.set_ylabel("$Cluster$ $shape,$ $P^2/4 \\pi A$", fontsize=16)
        ax.set_xticks(np.linspace(np.min(x), np.max(x), 4))
        #ax.set_ylim(8e-07, 0.9)
    else:
        pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        ax.plot(edges, pdf, lw=1.2, color='k', marker='o', fillstyle='none')
        ax.set_xlabel("$Cluster$ $shape,$ $P^2/4 \\pi A$", fontsize=16)
        ax.set_ylabel("$Distribution,$ $P(P^2/4 \\pi A)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pClusterShapePDF-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterDistribution(dirName, figureName, fixed=False, numBins=20, which='simplex'):
    fig, ax = plt.subplots(figsize=(8,5), dpi = 120)
    if(fixed=="Dr"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72',  'thermal78',  'thermal80',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$0.45$', '$0.58$', '$0.67$', '$0.72$', '$0.78$', '$0.80$', '$0.83$', '$0.85$', '$0.88$', '$0.94$', '$1.00$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        #dirList = np.array(['100', '10', '1', '1e-01', '1e-02', '1e-03', '3e-04', '1e-04', '2e-05', '1e-05', '2e-06', '1e-06', '2e-07', '1e-07'])
        dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr2e-04/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, "Dr")*utils.readFromDynParams(dirSample, "sigma"))
        if(which=='simplex'):
            if not(os.path.exists(dirSample + "simplexDistribution.dat")):
                cluster.computeSimplexClusterDistribution(dirSample)
            data = np.loadtxt(dirSample + "simplexDistribution.dat")
        elif(which=='particle'):
            if not(os.path.exists(dirSample + "clusterDistribution.dat")):
                cluster.computeDelaunayClusterDistribution(dirSample)
            data = np.loadtxt(dirSample + "clusterDistribution.dat")
        pdf, edges = np.histogram(data[:,1], bins=np.geomspace(np.min(data[:,1]), np.max(data[:,1]), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        ax.plot(edges, data[:,1], lw=1.2, color=colorList(d/dirList.shape[0]), marker='o', fillstyle='none')#, label='$D_r=$' + labelList[d])
    # make color bar for legend
    if(fixed=="Dr"):
        x = phi
        label = "$\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterDistro-vsPhi-" + figureName
    elif(fixed=="phi"):
        x = taup
        label = "$\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterDistro-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    if(which=='simplex'):
        ax.set_xlabel("$Number$ $of$ $simplices$ $in$ $cluster,$ $N_c$", fontsize=18)
    elif(which=='particle'):
        ax.set_xlabel("$Number$ $of$ $particles$ $in$ $cluster,$ $N_c$", fontsize=18)
    ax.set_ylabel("$Distribution,$ $P(N_c)$", fontsize=18)
    #ax.legend(fontsize=10, loc='best', ncol=2)
    fig.tight_layout()
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar, ax=ax)
    cb.set_ticks(np.linspace(0,1,5))
    ticks = np.geomspace(np.min(x), np.max(x), 5)
    cb.ax.tick_params(labelsize=14, length=0)
    ticklabels = []
    for i in range(5):
        ticklabels.append(np.format_float_scientific(ticks[i],2))
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=16, labelpad=20, rotation='horizontal')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSize(dirName, figureName, fixed=False, which='fraction', simplex=False):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="Dr"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72',  'thermal78',  'thermal80',  'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$0.45$', '$0.58$', '$0.67$', '$0.72$', '$0.78$', '$0.80$', '$0.83$', '$0.85$', '$0.88$', '$0.94$', '$1.00$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        #dirList = np.array(['1e-01', '1e-02', '1e-03', '1e-04', '1e-05', '1e-06', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
        taup = np.zeros(dirList.shape[0])
    clusterSize = np.zeros((dirList.shape[0], 2))
    clusterArea = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        if(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr2e-04/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-long/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        if(simplex=='simplex'):
            if not(os.path.exists(dirSample + "simplexDistribution.dat")):
                print("computing: ", dirList[d])
                cluster.computeSimplexClusterDistribution(dirSample)
            data = np.loadtxt(dirSample + "simplexDistribution.dat")
        else:
            if not(os.path.exists(dirSample + "clusterDistribution.dat")):
                cluster.computeDelaunayClusterDistribution(dirSample)
            data = np.loadtxt(dirSample + "clusterDistribution.dat")
        #weight = data[:,1] / np.sum(data[:,1])
        clusterSize[d,0] = np.sum(data[:,0])# * weight)
        clusterSize[d,1] = np.std(data[:,0])# * np.sqrt(np.sum(weight))
        #weight = data[:,0] / np.sum(data[:,0])
        clusterArea[d,0] = np.sum(data[:,1])# * weight)
        clusterArea[d,1] = np.std(data[:,1])# * np.sqrt(np.sum(weight))
    if(fixed=="Dr"):
        x = phi
        label = "$\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pCluster" + which + "Var-vsPhi-" + figureName
    elif(fixed=="phi"):
        x = taup
        label = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pCluster" + which + "Var-vsDr-" + figureName
    ax.set_xscale('log')
    if(which=='fraction'):
        ax.errorbar(x[clusterSize[:,0]>0], clusterSize[clusterSize[:,0]>0,0], clusterSize[clusterSize[:,0]>0,1], lw=1.2, marker='o', fillstyle='none', capsize=3, color='b')
        ax.set_ylabel("$Cluster$ $fraction,$ $\\langle f_c \\rangle$", fontsize=18)
    elif(which=='area'):
        ax.errorbar(x[clusterArea[:,0]>0], clusterArea[clusterArea[:,0]>0,0], clusterArea[clusterArea[:,0]>0,1], lw=1.2, marker='o', fillstyle='none', capsize=3, color='b')
        ax.set_ylabel("$Cluster$ $area,$ $\\langle A_c \\rangle$", fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(label, fontsize=18)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

########################### Interface fluctuations #############################
def plotSPClusterHeightFluctuations(dirName, figureName, which='active', qmax=1):
    sigma = utils.readFromDynParams(dirName, 'sigma')
    fig, ax = plt.subplots(2, 1, figsize=(7,7), dpi = 120)
    if not(os.path.exists(dirName + os.sep + "heightFluctuations.dat")):
        if(which=='lj'):
            cluster.averageClusterHeightFluctuations(dirName, 0.3)
        elif(which=='active'):
            cluster.averageClusterHeightFluctuations(dirName, 0.78)
    data = np.loadtxt(dirName + os.sep + "heightFluctuations.dat")
    boxSize = np.loadtxt(dirName + os.sep + "boxSize.dat")
    ly = boxSize[1]*sigma
    data[:,4:] *= ly
    ax[0].errorbar(data[:,0]/sigma, data[:,1]/sigma**2, data[:,2]/sigma**2, lw=1, marker='o', markersize=6, color='k', fillstyle='none', capsize=3, elinewidth=0.5)
    ax[1].errorbar(data[1:,3], data[1:,4], data[1:,5], lw=1, marker='o', markersize=6, color='k', fillstyle='none', capsize=3, elinewidth=0.5)
    data = data[data[:,3]<qmax]
    x = data[1:,3]
    y = data[1:,4]
    failed = False
    try:
        popt, pcov = curve_fit(powerLawMass, x, y, bounds=([0, -2.01, -np.inf], [np.inf, -1.99, np.inf]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        ax[1].plot(x, powerLawMass(x, *popt), color='b', lw=1.2, ls='solid', label="$Fit,$ $a(q^b+c):$ $a=$" + str(np.around(popt[0],3)) + ", $b=$" + str(np.around(popt[1],3)) + ", $c=$" + str(np.around(popt[2],3)))
        print("fitting parameters - a:", popt[0], "b:", popt[1])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].legend(fontsize=12, loc='best')
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].set_xlabel("$q$", fontsize=16)
    ax[1].set_ylabel("$\\langle |\delta h(q)|^2 \\rangle L$", fontsize=16)
    ax[0].set_xlabel("$y$", fontsize=16)
    ax[0].set_ylabel("$\\langle |\delta h(y)|^2 \\rangle$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pHeightFlu-" + figureName + ".png"
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPClusterHeightVSTemp(dirName, figureName, which='active', qmax=0.4):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        dirList = np.array(['0.35', '0.36', '0.37', '0.38'])
        labelName = "$T=$"
    elif(which=='active'):
        dirList = np.array(['4e-04', '3e-04', '2e-04', '1.5e-04', '1.2e-04', '1e-04', '9e-05', '7e-05', '5e-05'])
        labelName = "$D_r=$"
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('plasma', dirList.shape[0]+2)
    taup = np.zeros(dirList.shape[0])
    temp = np.zeros((dirList.shape[0],2))
    slope = np.zeros(dirList.shape[0])
    power = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(which=='lj'):
            dirSample = dirName + "/box6-2/0.238/langevin-lj/T" + dirList[d] + "/dynamics/"
            labelName = '$T$ = '
        elif(which=='active'):
            dirSample = dirName + "/box6-2/0.30/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            labelName = '$D_r$ = '
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "heightFluctuations.dat")):
            if(which=='lj'):
                cluster.averageClusterHeightFluctuations(dirSample, 0.3)
            elif(which=='active'):
                cluster.averageClusterHeightFluctuations(dirSample, 0.78)
        boxSize = np.loadtxt(dirSample + os.sep + "boxSize.dat")
        ly = boxSize[1]*sigma
        data = np.loadtxt(dirSample + "heightFluctuations.dat")
        data[:,4:] *= ly
        ax.plot(data[1:,3], data[1:,4], lw=1, markersize=6, color=colorList(d/dirList.shape[0]), fillstyle='none', label=labelName + dirList[d])
        data = data[1:,:]
        data = data[data[:,3]<qmax]
        x = data[2:,3]
        y = data[2:,4]
        failed = False
        try:
            #popt, pcov = curve_fit(powerLaw, x, y)
            popt, pcov = curve_fit(powerLawMass, x, y, bounds=([0, -2.01, -np.inf], [np.inf, -1.99, np.inf]))
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            #ax.plot(x, powerLaw(x, *popt), color='k', lw=1.2, ls='dashed')#, label="$Fit,$ $aq^b:$ $a=$" + str(np.around(popt[0],3)) + ", $b=$" + str(np.around(popt[1],3)))
            ax.plot(x, powerLawMass(x, *popt), color='k', lw=1.2, ls='dashed')#, label="$Fit,$ $aq^b:$ $a=$" + str(np.around(popt[0],3)) + ", $b=$" + str(np.around(popt[1],3)))
            #ax.set_yscale('log')
            #ax.set_xscale('log')
            #plt.pause(0.5)
            print("Dr", dirList[d], "slope (stiffness):", popt[0], "power(2):", popt[1])
            slope[d] = popt[0]
            power[d] = popt[1]
        # compute length of interface
        if not(os.path.exists(dirSample + os.sep + "clusterTemperature.dat")):
            cluster.computeClusterTemperatureVSTime(dirSample)
        data = np.loadtxt(dirSample + os.sep + "clusterTemperature.dat")
        temp[d,0] = np.mean(data[:,0])
        temp[d,1] = np.std(data[:,0])
    ax.legend(fontsize=11, ncol=2, loc='best')
    ax.set_xlabel("$q$", fontsize=16)
    ax.set_ylabel("$\\langle |\delta h(q)|^2 \\rangle L$", fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/pHeightFluVSLength-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    # second plot
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        x = temp[:,0]
        xlabel = "$Temperature,$ $T$"
    elif(which=='active'):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Fitting$ $parameters$", fontsize=16)
    ax.plot(x, slope, lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', label="$Slope,$ $\\frac{U}{\\gamma}$")
    ax.plot(x, power, lw=1.2, marker='v', markersize=8, color='b', fillstyle='none', label="$Power$")
    ax.legend(fontsize=14, loc='best')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/stiffVSTemp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterHeightCorrelation(dirName, figureName, which='Dr'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='Dr'):
        dirList = np.array(['3e-04', '2e-04', '1.5e-04', '1.2e-04', '9e-05', '7e-05', '5e-05'])
    elif(which=='T'):
        dirList = np.array(['0.41', '0.35'])
    markerList = ['o', 'v', '^', 's', '*', 'd', 'D', '+', 'x']
    gamma = np.zeros(dirList.shape[0])
    temp = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(which=='Dr'):
            dirSample = dirName + os.sep + which + dirList[d] + "/dynamics/"
        elif(which=='T'):
            dirSample = dirName + os.sep + which + dirList[d] + "-removed/dynamics/"
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + os.sep + "clusterHeightCorr.dat")):
            cluster.computeClusterInterfaceCorrelation(dirSample)
        data = np.loadtxt(dirSample + os.sep + "clusterHeightCorr.dat")
        #ax.errorbar(data[:,0], data[:,1], data[:,2], lw=1, marker=markerList[d], markersize=6, color='k', fillstyle='none', capsize=3, elinewidth=0.5, label=which + '$=$' + dirList[d])
        ax.plot(data[1:,0], data[1:,1]/data[0,1], lw=1, marker=markerList[d], markersize=6, color='k', fillstyle='none', label=which + '$=$' + dirList[d])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.set_xlabel("$Distance,$ $\Delta x$", fontsize=16)
    ax.set_ylabel("$g_h(\Delta x)$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pHeightCorr-" + figureName + "-vs" + which + ".png"
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPDensityProfile(dirName, figureName):
    #print("taup:", 1/(utils.readFromDynParams(dirName, "Dr")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if not(os.path.exists(dirName + "densityProfile.dat")):
        cluster.averageLinearDensityProfile(dirName)
    data = np.loadtxt(dirName + "densityProfile.dat")
    center = np.mean(data[:,0])
    x = np.abs(data[:,0]-center)
    y = data[np.argsort(x),1]
    yerr = data[np.argsort(x),2]
    x = np.sort(x)
    ax.errorbar(x, y, yerr, lw=1, marker='o', markersize=6, color='k', capsize=3, fillstyle='none', label='$Profile$')
    failed = False
    try:
        popt, pcov = curve_fit(hyperbolicTan, x, y, bounds=([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf]))
    except RuntimeError:
        print("Error - curve_fit failed")
        failed = True
    if(failed == False):
        ax.plot(x, hyperbolicTan(x, *popt), color='g', lw=2, linestyle='dashed', label='$Fit$')
        print("center - x0:", popt[2], "width:", popt[3], 'phi-:', popt[0] - popt[1], 'phi+:', popt[0] + popt[1])
        width = popt[3]
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$\\varphi(x)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/phiProfile-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterWidth(dirName, figureName, which='active', param='1e-04'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    dirList = np.array(['0.6', '0.8', '1', '1.2', '1.4', '1.6', '1.8', '2', '2.2'])
    markerList = ['o', 's', 'd', 'v', '^', 'D', 'x', '*', 'h']
    colorList = cm.get_cmap('plasma', dirList.shape[0]+2)
    ly = dirList.astype(np.float64)
    length = np.zeros((dirList.shape[0],2))
    width1 = np.zeros((dirList.shape[0],2))
    width2 = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(which=='lj'):
            dirSample = dirName + "/box6-" + dirList[d] + "/0.238/langevin-lj/T" + param + "/dynamics/"
        elif(which=='active'):
            dirSample = dirName + "/box6-" + dirList[d] + "/0.30/active-langevin/Dr" + param + "/dynamics/"
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "energyLength.dat")):
            if(which=='lj'):
                cluster.averageClusterHeightFluctuations(dirSample, 0.3)
            elif(which=='active'):
                cluster.averageClusterHeightFluctuations(dirSample, 0.78)
        energyLength = np.loadtxt(dirSample + "energyLength.dat")
        energyLength[:,1] /= sigma
        length[d,0] = np.mean(energyLength[:,1])
        length[d,1] = np.std(energyLength[:,1])
        if not(os.path.exists(dirSample + "densityProfile.dat")):
            if(which=='lj'):
                cluster.averageLinearDensityProfile(dirSample, 0.3)
            elif(which=='active'):
                cluster.averageLinearDensityProfile(dirSample, 0.78)
        data = np.loadtxt(dirSample + "densityProfile.dat")
        center = np.mean(data[:,0])
        x = np.abs(data[:,0]-center)
        y = data[np.argsort(x),1]
        yerr = data[np.argsort(x),2]
        x = np.sort(x)
        ax.errorbar(x, y, yerr, lw=1, color=colorList(d/dirList.shape[0]), marker=markerList[d], markersize=6, capsize=3, fillstyle='none', label="$L_y=$" + dirList[d])
        #ax.plot(data[:,0], data[:,1], lw=1, marker=markerList[d], markersize=6, color='k', fillstyle='none', label=labelName + dirList[d])
        width2[d] = utils.computeInterfaceWidth(x, y)/sigma
        data = np.loadtxt(dirSample + "interfaceWidth.dat")/sigma
        width1[d,0] = np.mean(data[:,1]**2)
        width1[d,1] = np.std(data[:,1]**2)/np.sqrt(data.shape[0])
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$\\varphi(x)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figure1Name = "/home/francesco/Pictures/soft/mips/pProfileVSLength-" + figureName
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    # second plot
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax.set_ylabel("$Squared$ $width,$ $w^2$", fontsize=16)
    ax.errorbar(ly, width1[:,0], width1[:,1], lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', capsize=3, label='$Average$ $from$ $profiles$')
    ax.plot(ly, width2**2, lw=1.2, marker='v', markersize=8, color='b', fillstyle='none', label='$From$ $averaged$ $profile$')
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/widthVSLength-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterWidthVSTemp(dirName, figureName, which='lj'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        dirList = np.array(['0.30', '0.31', '0.32', '0.33', '0.34', '0.35', '0.36', '0.37', '0.38',
                            '0.39', '0.40', '0.41', '0.42', '0.43', '0.44', '0.45'])
    elif(which=='active'):
        dirList = np.array(['9e-03', '8e-03', '7e-03', '6e-03', '5.5e-03', '5e-03', '4.5e-03', '4e-03', '3.7e-03', '3.5e-03', '3.3e-03',
                            '2.8e-03', '2.6e-03', '2.4e-03', '2.2e-03', '2e-03'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    temp = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    width = np.zeros((dirList.shape[0],3))
    for d in range(dirList.shape[0]):
        if(which=='lj'):
            dirSample = dirName + "box/0.30compress/langevin-lj/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "box/0.30/active-lj/T0.35-Dr" + dirList[d] + "-f08/dynamics/"
            labelName = "$D_r=$"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "densityProfile.dat")):
            interface.averageLinearDensityProfile(dirSample)
        data = np.loadtxt(dirSample + "densityProfile.dat")
        #ax.errorbar(data[:,0], data[:,1], data[:,2], lw=1, marker='o', markersize=6, color=colorList(d/dirList.shape[0]), capsize=3, fillstyle='none', label=labelName + dirList[d])
        center = np.mean(data[:,0])
        x = data[:,0]-center
        y = data[:,1]
        yerr = data[:,2]
        y = y[x>0]
        yerr = yerr[x>0]
        x = x[x>0]
        #x = (x[1:] + x[:-1])/2
        #y = (y[1:] - y[:-1])/2
        #yerr = (yerr[1:] + yerr[:-1])/2
        ax.errorbar(x, y, yerr, lw=1, marker='o', markersize=6, color=colorList(d/dirList.shape[0]), capsize=3, fillstyle='none', label=labelName + dirList[d])
        failed = False
        try:
            popt, pcov = curve_fit(hyperbolicTan, x, y, bounds=([0, 0, -np.inf, 0], [1, 1, np.inf, np.inf]))
            #popt, pcov = curve_fit(hyperbolicSecSq, x, y)
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            ax.plot(x, hyperbolicTan(x, *popt), color='k')
            #ax.plot(x, hyperbolicSecSq(x, *popt), color='k')
            width[d,2] = popt[3]
            print("fitting parameters - a, b, x0, w:", popt)
        #width[d,2] = utils.computeInterfaceWidth(x, y)
        data = np.loadtxt(dirSample + "interfaceWidth.dat")
        width[d,0] = np.mean(data[:,1])
        width[d,1] = np.std(data[:,1])
        if not(os.path.exists(dirSample + "clusterTemperature.dat")):
            interface.computeClusterTemperatureVSTime(dirSample)
        data = np.loadtxt(dirSample + "clusterTemperature.dat")
        temp[d] = np.mean(data[:,1])
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.set_xlabel("$x/L_x$", fontsize=16)
    ax.set_ylabel("$Profile,$ $\\varphi(x)$", fontsize=16)
    figure1Name = "/home/francesco/Pictures/soft/mips/profileVSTemp-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        x = temp
        xlabel = "$Temperature,$ $T/ \\varepsilon$"
    elif(which=='active'):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p / \\tau_i$"
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Interface$ $width,$ $w/L_x$", fontsize=16)
    ax.errorbar(x, width[:,0], width[:,1], lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', capsize=3, label='$Average$ $of$ $fits$')
    #ax.plot(x, width[:,2], lw=1.2, marker='v', markersize=8, color='g', markeredgecolor='k', fillstyle='full', label='$Fit$ $after$ $average$')
    ax.legend(loc='best', fontsize=12)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/widthVSTemp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPActiveEnergyVSTemp(dirName, figureName):
    fig, ax = plt.subplots(figsize=(7.5,5.5), dpi = 120)
    # plot passive sample
    tempList = np.array(['0.20', '0.21', '0.22', '0.23', '0.24', '0.25', '0.26', '0.27', '0.28', '0.29', 
                         '0.30', '0.31', '0.32', '0.33', '0.34', '0.35', '0.36', '0.37', '0.38', '0.39', 
                         '0.40', '0.41', '0.42', '0.43', '0.44', '0.45'])
    temp = np.zeros((tempList.shape[0],2))
    epot = np.zeros((tempList.shape[0],2))
    for d in range(tempList.shape[0]):
        dirSample = dirName + "box/0.30compress/langevin-lj/T" + tempList[d] + "/dynamics/"
        data = np.loadtxt(dirSample + "energy.dat")
        temp[d,0] = np.mean(data[:,3])
        temp[d,1] = np.std(data[:,3])
        epot[d,0] = np.mean(data[:,2])
        epot[d,1] = np.std(data[:,2])
    ax.errorbar(temp[:,0], epot[:,0], epot[:,1], temp[:,1], lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', capsize=3, label='$Passive$')
    # plot active samples
    tempList = np.array(['0.30'])
    markerList = ['s', 'v', 'd', 'D', '^']
    f0List = np.array(['1', '2', '4'])
    colorList = ['c', 'g', 'b', 'y', 'r']
    ratio = np.zeros((tempList.shape[0], f0List.shape[0]))
    ftList = np.zeros(tempList.shape[0])
    for i in range(tempList.shape[0]):
        ftList[i] = np.sqrt(2*np.sqrt(10)*float(tempList[i]))
        for j in range(f0List.shape[0]):
            ratio[i,j] = ftList[i] / float(f0List[j])
            #print(ftList[i], f0List[j], ratio[i,j])
    tpList = np.array(['1e-03', '1e-02', '2e-02', '4e-02', '6e-02', '8e-02', '1e-01', '2e-01', '4e-01', '6e-01', '8e-01', #11
                        '1', '2', '4', '6', '8', '10', '20', '40', '60', '80', '100', '200', '400', '600', '800', '1000', #27
                        '2000', '4000', '6000', '8000', '10000'])
    temp = np.zeros((tpList.shape[0],2))
    epot = np.zeros((tpList.shape[0],2))
    taup = np.zeros(tpList.shape[0])
    for t in range(tempList.shape[0]):
        for f in range(f0List.shape[0]):
            scale = float(f0List[f])
            labelName = "$\\frac{\\sqrt{2\\gamma k_B T}}{f_0} =$" + str(np.format_float_positional(ratio[t,f],2))
            labelName = "$f_T=$" + str(np.format_float_positional(ftList[t],2)) + ", $f_0=$" + f0List[f]
            for d in range(tpList.shape[0]):
                #print("T:", tempList[t], "f0:", f0List[f], "tp:", tpList[d], "Peclet:", float(f0List[f])*float(tpList[d]) / np.sqrt(10))
                dirSample = dirName + "box/0.30compress/langevin-lj/T" + tempList[t] + "/active-lj/T" + tempList[t] + "-tp" + tpList[d] + "-f0" + f0List[f] + "/"
                taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
                data = np.loadtxt(dirSample + "energy.dat")
                temp[d,0] = np.mean(data[:,3])
                temp[d,1] = np.std(data[:,3])
                epot[d,0] = np.mean(data[:,2])
                epot[d,1] = np.std(data[:,2])
                fill = 'none'
                edge = colorList[f]
                if((t==2 and f==3) and (d>3 and d<20)):
                    fill = 'full'
                    edge = 'k'
                if((t==2 and f==2) and (d>6 and d<21)):
                    fill = 'full'
                    edge = 'k'
                if((t==2 and f==1) and (d>11 and d<22)):
                    fill = 'full'
                    edge = 'k'
                if((t==1 and f==3) and (d>4 and d<20)):
                    fill = 'full'
                    edge = 'k'
                if((t==1 and f==2) and (d>7 and d<21)):
                    fill = 'full'
                    edge = 'k'
                if((t==1 and f==1) and (d>12 and d<22)):
                    fill = 'full'
                    edge = 'k'
                if((t==0 and f==3) and (d>6 and d<20)):
                    fill = 'full'
                    edge = 'k'
                if((t==0 and f==2) and (d>9 and d<21)):
                    fill = 'full'
                    edge = 'k'
                if((t==0 and f==1) and (d>14 and d<22)):
                    fill = 'full'
                    edge = 'k'
                if(d==0):
                    ax.errorbar(temp[d,0], epot[d,0], epot[d,1], temp[d,1], marker=markerList[t], markersize=10, color=colorList[f], markeredgecolor=edge, fillstyle=fill, capsize=3, label=labelName)
                else:
                    ax.errorbar(temp[d,0], epot[d,0], epot[d,1], temp[d,1], marker=markerList[t], markersize=10, color=colorList[f], markeredgecolor=edge, fillstyle=fill, capsize=3)
            ax.errorbar(temp[:,0], epot[:,0], epot[:,1], temp[:,1], lw=1.2, color=colorList[f], capsize=3)
    ax.legend(fontsize=11, loc='lower right', ncol=2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Temperature,$ $T$", fontsize=16)
    ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/energyVSTemp-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterISF(dirName, figureName, compare="compare"):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    data = np.loadtxt(dirName + "/dynamics-log/clusterLogCorr.dat")
    ax.semilogx(data[:,0], data[:,1], linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', label="$Passive$")
    if(compare=="compare"):
        tpList = np.array(['1e-03'])
        colorList = cm.get_cmap('viridis', tpList.shape[0])
        for i in range(tpList.shape[0]):
            dirSample = dirName + "/active-lj/T0.30-tp" + tpList[i] + "-f02/"
            data = np.loadtxt(dirSample + "/dynamics-log/clusterLogCorr.dat")
            ax.semilogx(data[:,0], data[:,2], linewidth=1.2, marker='o', markersize=3, fillstyle='none', color=colorList(i/tpList.shape[0]), label="$\\tau_p=$"+tpList[i])
        ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_ylabel("$ISF$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/clusterISF-" + figureName
    plt.tight_layout()
    if(compare=='compare'):
        figureName += "-compare"
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPWallForceVSTime(dirName, figureName, which='average', index=4, numSamples=20):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(which=="single"):
        data = np.loadtxt(dirName + "/energy.dat")
        ax.plot(data[:,1], data[:,index], linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k')
    elif(which=="average"):
        data = np.loadtxt(dirName + "-1/energy.dat")
        time = data[:,1]
        wall = np.zeros((numSamples, data.shape[0]))
        wall[0] = data[:,index]
        for i in range(2,numSamples+1):
            data = np.loadtxt(dirName + "-" + str(i) + "/energy.dat")
            wall[i-1] = data[:time.shape[0],index]
        mean = np.mean(wall, axis=0)
        error = np.std(wall, axis=0)
        ax.errorbar(time, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_xscale('log')
    if(index==2):
        ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        figureName = "/home/francesco/Pictures/soft/mips/wallEnergyVSTime-" + figureName
    elif(index==4):
        ax.set_ylabel("$Force$ $on$ $wall,$ $F_{wall}$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallForceVSTime-" + figureName
    elif(index==5):
        ax.set_ylabel("$Pressure,$ $p$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallPressureVSTime-" + figureName
    plt.tight_layout()
    if(which=='average'):
        figureName += "-average"
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPWallForceVSStrain(dirName, figureName, which='average', index=4):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    numParticles = 8192
    dirList, strain = utils.getOrderedStrainDirectories(dirName)
    mean = np.zeros(dirList.shape[0])
    error = np.zeros(dirList.shape[0])
    Ly = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d]
        if(which=='average'):
            dirSample += "/dynamics/"
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "/energy.dat")
            #data[:,4] /= np.sqrt(numParticles)
            mean[d] = np.mean(data[:,index])
            error[d] = np.std(data[:,index])
            if(index==4 or index==5):
                error[d] /= np.sqrt(data[:,index].shape[0])
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
            Ly[d] = boxSize[1]
    if(which=='average'):
        mean = mean[mean!=0]
        error = error[error!=0]
        strain = strain[Ly!=0]
        Ly = Ly[Ly!=0]
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
    ax.set_xlim(-0.04, 0.84)
    ax.errorbar(strain, mean, error, lw=1, marker='s', markersize=5, color='k', capsize=3, markeredgecolor='k', fillstyle='left', label='$From$ $simulation$')
    if(index==2):
        ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        print("change in energy:", mean[-1] - mean[0], "change in length:", Ly[-1] - Ly[0], "slope:", (mean[-1] - mean[0])/(Ly[-1] - Ly[0]))
        figureName = "/home/francesco/Pictures/soft/mips/wallEnergyVSStrain-" + figureName
    elif(index==4):
        ax.set_ylabel("$\\frac{F_{wall} \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=28)
        midForce = (mean[1:] + mean[:-1])*0.5
        width = strain[1:] - strain[:-1]
        work = np.sum(midForce*width)
        print("work during extension:", work, "line tension:", work/strain[-1])
        figureName = "/home/francesco/Pictures/soft/mips/wallForceVSStrain-" + figureName
    elif(index==5):
        ax.set_ylabel("$Pressure,$ $p$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallPressureVSStrain-" + figureName
    plt.tight_layout()
    if(which=='average'):
        figureName += "-average"
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSampleWallForceVSTime(dirName, figureName, type='active', which='average', index=4, numSamples=30):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    numParticles = 8192
    if(type=='active'):
        dirType = "langevin-lj/T0.30/active-lj/T0.30-tp1e-03-f02/biaxial4e-02/"
    elif(type=='passive'):
        dirType = "langevin-lj/T0.30/biaxial4e-02/"
    else:
        print("please specify sample type, active or passive")
    if(which=="single"):
        data = np.loadtxt(dirName + "/energy.dat")
        ax.plot(data[:,1], data[:,index], linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k')
    elif(which=="average"):
        data = np.loadtxt(dirName + "-0/" + dirType + "energy.dat")
        time = data[:,1]
        wall = np.zeros((numSamples, data.shape[0]))
        wall[0] = data[:,index]
        for i in range(1,numSamples):
            data = np.loadtxt(dirName + "-" + str(i) + "/" + dirType + "energy.dat")
            wall[i] = data[:,index]
        mean = np.mean(wall, axis=0)
        error = np.std(wall, axis=0)
        ax.errorbar(time, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Time,$ $t$", fontsize=16)
    ax.set_xscale('log')
    if(index==2):
        ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        figureName = "/home/francesco/Pictures/soft/mips/wallEnergyVSTime-" + figureName
    elif(index==4):
        ax.set_ylabel("$Force$ $on$ $wall,$ $F_{wall}$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallForceVSTime-" + figureName
    elif(index==5):
        ax.set_ylabel("$Pressure,$ $p$", fontsize=16)
        figureName = "/home/francesco/Pictures/soft/mips/wallPressureVSTime-" + figureName
    plt.tight_layout()
    if(which=='average'):
        figureName += "-average"
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSampleWallDeltaForceVSStrain(dirPath, figureName, type='active', numSamples=30, maxStrain=0.06):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    # first average force at rest over samples
    force0 = np.empty(0)
    for i in range(numSamples):
        dirName = dirPath + "-" + str(i) + "/langevin-lj/T0.30/dynamics-F/"
        data = np.loadtxt(dirName + "/energy.dat")
        force0 = np.append(force0, data[:,4])
    pdf, edges = np.histogram(force0, bins=np.linspace(np.min(force0), np.max(force0), 30), density=True)
    edges = (edges[1:] + edges[:-1])/2
    ax.plot(edges, pdf, lw=1, color='b')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$F(L_y^0)$", fontsize=16)
    ax.set_ylabel("$P(F(L_y^0))$", fontsize=16)
    err0 = np.std(force0)
    force0 = np.mean(force0)
    figure1Name = "/home/francesco/Pictures/soft/mips/sampleForce0PDF-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    print("Average force at rest: ", force0, "+-", err0)
    # now plot force as a function of strain
    if(type=='active'):
        dirType = "langevin-lj/T0.30/active-lj/T0.30-tp1e-03-f01/biaxial2e-03-tmax2e05/"
    elif(type=='passive'):
        dirType = "langevin-lj/T0.32/biaxial2e-03-tmax2e05/"
    else:
        print("please specify sample type, active or passive")
    # initialize force and energy with first sample
    numSamples = 30
    which = 0
    index = 4
    boxSize = np.loadtxt(dirPath + "-0/boxSize.dat")
    dirName = dirPath + "-0/" + dirType
    dirList, strain = utils.getOrderedStrainDirectories(dirName)
    dirList = dirList[strain<maxStrain]
    strain = strain[strain<maxStrain]
    force = np.zeros(dirList.shape[0])
    Ly = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d]
        if(which=='average'):
            dirSample += "/dynamics/"
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "/energy.dat")
            force[d] = np.mean(data[:,index])
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
            Ly[d] = boxSize[1]
    if(which=='average'):
        force = force[force!=0]
        strain = strain[Ly!=0]
        Ly = Ly[Ly!=0]
    wall = np.zeros((numSamples, force.shape[0]))
    wall[0] = force
    # read other samples and average
    for i in range(1,numSamples):
        dirName = dirPath + "-" + str(i) + "/" + dirType
        dirList, strainList = utils.getOrderedStrainDirectories(dirName)
        dirList = dirList[strainList<maxStrain]
        strainList = strainList[strainList<maxStrain]
        force = np.zeros(dirList.shape[0])
        for d in range(dirList.shape[0]):
            dirSample = dirName + dirList[d]
            if(which=='average'):
                dirSample += "/dynamics/"
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "/energy.dat")
                force[d] = np.mean(data[0,index] - data[1:,index])
        if(which=='average'):
            force = force[force!=0]
        wall[i-1] = force
    mean = np.mean(wall, axis=0)
    error = np.std(wall, axis=0)
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax.errorbar(Ly, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    ax.set_ylabel("$\\frac{\\Delta F_{wall} \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=28)
    print("Average line tension in linear response: ", np.mean(wall), "+-", np.std(wall))
    print("Line tension:", np.mean(wall))
    figure2Name = "/home/francesco/Pictures/soft/mips/sampleDeltaForceVSStrain-" + figureName
    plt.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSampleWallForceVSStrain(dirPath, figureName, type='active', which='average', index=4, numSamples=30, limit=30, maxStrain=0.6, temp="0.30", taup="1e-03"):
    fileName = dirPath + "-0/../lineTension-Active.dat"
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(type=='active'):
        dirType = "langevin-lj/T0.30/active-lj/T0.30-tp" + taup + "-f01/biaxial2e-03-tmax2e05/"
    elif(type=='passive'):
        dirType = "langevin-lj/T" + temp + "/biaxial2e-03-tmax2e05/"
    else:
        print("please specify sample type, active or passive")
    # initialize force and energy with first sample
    dirName = dirPath + "-0/" + dirType
    dirList, strain = utils.getOrderedStrainDirectories(dirName)
    dirList = dirList[strain<maxStrain]
    strain = strain[strain<maxStrain]
    force = np.zeros(dirList.shape[0])
    Ly = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        dirSample = dirName + dirList[d]
        if(which=='average'):
            dirSample += "/dynamics/"
        if(os.path.exists(dirSample)):
            data = np.loadtxt(dirSample + "/energy.dat")
            force[d] = np.mean(data[:,index])
            boxSize = np.loadtxt(dirSample + "/boxSize.dat")
            Ly[d] = boxSize[1]
    if(which=='average'):
        force = force[force!=0]
        strain = strain[Ly!=0]
        Ly = Ly[Ly!=0]
    wall = np.zeros((numSamples, force.shape[0]))
    #print(force.shape[0])
    wall[0] = force
    # read other samples and average
    temp = np.empty(0)
    for i in range(1,numSamples):
        #print(i)
        dirName = dirPath + "-" + str(i) + "/" + dirType
        en = np.loadtxt(dirName + "../energy.dat")
        temp = np.append(temp, en[:,3])
        dirList, strainList = utils.getOrderedStrainDirectories(dirName)
        dirList = dirList[strainList<maxStrain]
        strainList = strainList[strainList<maxStrain]
        force = np.zeros(dirList.shape[0])
        for d in range(dirList.shape[0]):
            dirSample = dirName + dirList[d]
            #print(dirSample)
            if(which=='average'):
                dirSample += "/dynamics/"
            if(os.path.exists(dirSample)):
                #print(dirSample)
                data = np.loadtxt(dirSample + "/energy.dat")
                force[d] = np.mean(data[:,index])
        if(which=='average'):
            force = force[force!=0]
        #print(force.shape[0])
        wall[i-1] = force
    mean = np.mean(wall, axis=0)
    error = np.std(wall, axis=0)
    ax.tick_params(axis='both', labelsize=14)
    #ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
    #ax.errorbar(strain, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax.errorbar(Ly, mean, error, linewidth=1.2, marker='o', markersize=3, fillstyle='none', color='k', capsize=3)
    #ax.set_xlim(-0.04, 0.84)
    if(index==2):
        failed = False
        try:
            popt, pcov = curve_fit(lineFit, strain[:-2], mean[:-2])
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            ax.plot(strain, lineFit(strain, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax + b,$ $a=$" + str(popt[0]))
            ax.legend(fontsize=12, loc='best')
        ax.set_ylabel("$\\frac{E_{pot}}{N}$", fontsize=24, rotation='horizontal', labelpad=20)
        print("change in energy:", mean[-1] - mean[0], "change in strain:", strain[-1] - strain[0], "slope:", (mean[-1] - mean[0])/(strain[-1] - strain[0]))
        figure1Name = "/home/francesco/Pictures/soft/mips/sampleWallEnergyVSStrain-" + figureName
    elif(index==4):
        ax.set_ylabel("$\\frac{F_{wall} \\sigma}{\\varepsilon}$", fontsize=24, rotation='horizontal', labelpad=28)
        midForce = (mean[1:] + mean[:-1])*0.5
        midStrain = (strain[1:] + strain[:-1])*0.5
        midLength = (Ly[1:] + Ly[:-1])*0.5
        width = strain[1] - strain[0]
        widthl = Ly[1] - Ly[0]
        work = np.sum((midForce[0]-midForce[1:limit])*width)*Ly[0]
        workl = np.sum((midForce[0]-midForce[1:limit])*widthl)
        print("work during extension:", work, "line tension:", work/(midStrain[limit]*Ly[0]))
        print("work during extension:", workl, "line tension:", work/(Ly[limit]-Ly[0]))
        figure1Name = "/home/francesco/Pictures/soft/mips/sampleWallPressureVSStrain-" + figureName
    elif(index==5):
        ax.set_ylabel("$Pressure,$ $p$", fontsize=16)
        figure1Name = "/home/francesco/Pictures/soft/mips/sampleWallPressureVSStrain-" + figureName
    plt.tight_layout()
    if(which=='average'):
        figure1Name += "-average"
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    if(index==4):
        work = np.zeros(midForce.shape[0]-1)
        for i in range(midForce.shape[0]-1):
            #work[i] = np.sum((midForce[0]-midForce[:i+1])*width)
            work[i] = np.sum((midForce[0]-midForce[:i+1])*widthl)
        midStrain = midStrain[:-1]
        midLength = midLength[:-1]
        fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
        failed = False
        try:
            #popt, pcov = curve_fit(lineFit, midStrain[:limit], work[:limit])
            popt, pcov = curve_fit(lineFit, midLength[:limit], work[:limit])
        except RuntimeError:
            print("Error - curve_fit failed")
            failed = True
        if(failed == False):
            #ax.plot(midStrain, lineFit(midStrain, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax^2 + bx + c$")
            ax.plot(midLength, lineFit(midLength, *popt), color='g', lw=1.2, linestyle='dashdot', label="$ax^2 + bx + c$")
            #print((lineFit(midStrain, *popt))/midStrain)
            print("a, b:", popt, "line tension from fit:", popt[0], "average temperature:", np.mean(temp))
        #ax.plot(midStrain, work, marker='o', fillstyle='none', color='k', lw=1.2)
        ax.plot(midLength, work, marker='o', fillstyle='none', color='k', lw=1.2)
        ax.tick_params(axis='both', labelsize=14)
        #ax.set_xlabel("$Strain,$ $\\gamma$", fontsize=16)
        ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
        ax.set_ylabel("$\\int_{L_y^0}^{L_y} \\left[ F(L_y^0) - F(L_y') \\right] dL_y'$", fontsize=16)
        #ax.set_ylim(-0.6,11.6)
        figure2Name = "/home/francesco/Pictures/soft/mips/sampleTensionVSStrain-" + figureName
        plt.tight_layout()
        fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotGammaVSActivity(fileName, figureName):
    fig, ax = plt.subplots(figsize=(5,4), dpi = 120)
    data = np.loadtxt(fileName)
    ax.semilogx(data[:,0], data[:,1]/0.3, color='k', marker='o', markersize=8, lw=1, fillstyle='none')
    ax.set_ylim(-0.02,4.92)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    ax.set_ylabel("$\\frac{\\gamma \\sigma}{k_B T}$", rotation='horizontal', fontsize=24, labelpad=24)
    figureName = "/home/francesco/Pictures/soft/mips/gammaVSactivity-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPWallForceVSTemp(dirName, figureName, sample='passive', which='average'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(sample=='passive'):
        tempList = np.array(['0.30', '0.35', '0.40', '0.45'])
    elif(sample=='active'):
        tempList = np.array(['1e-03', '1e-02', '1e-01'])
        taup = tempList.astype(np.float64)
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', tempList.shape[0]+2)
    markerList = ['s', 'v', 'd', '*']
    temp = np.zeros((tempList.shape[0],2))
    gamma = np.zeros((tempList.shape[0],2))
    for t in range(tempList.shape[0]):
        if(sample=='passive'):
            dirPath = dirName + "/T" + tempList[t] + "/biaxial2e-03-tmax2e05/"
            labelName = "$T=$"
        elif(sample=='active'):
            dirPath = dirName + "T0.30-tp" + tempList[t] + "-f02/biaxial2e-03-tmax2e05/"
            labelName = "$\\tau_p=$"
        boxSize = np.loadtxt(dirPath + "/boxSize.dat")
        Ly0 = boxSize[1]
        dirList, strainList = utils.getOrderedStrainDirectories(dirPath)
        force = np.zeros(dirList.shape[0])
        error = np.zeros(dirList.shape[0])
        Ly = np.zeros(dirList.shape[0])
        for d in range(dirList.shape[0]):
            dirSample = dirPath + dirList[d]
            if(which=='average'):
                dirSample += "/dynamics/"
            if(os.path.exists(dirSample)):
                data = np.loadtxt(dirSample + "/energy.dat")
                force[d] = np.mean(data[:,4])
                error[d] = np.std(data[:,4])/np.sqrt(data[:,4].shape[0])
                boxSize = np.loadtxt(dirSample + "/boxSize.dat")
                #Ly[d] = (boxSize[1] - Ly0)/Ly0
                Ly[d] = boxSize[1]
        if(which=='average'):
            force = force[Ly>0]
            error = error[Ly>0]
            Ly = Ly[Ly>0]
        ax.errorbar(Ly, force, error, lw=1.2, marker=markerList[t], markersize=8, color=colorList(t/tempList.shape[0]), capsize=3, fillstyle='none', label=labelName + tempList[t])
        midForce = (force[1:] + force[:-1])*0.5
        height = (Ly[1:] + Ly[:-1])*0.5
        gamma[t,0] = np.sum(midForce*height)/(height[-1]-height[0])
        gamma[t,1] = np.mean(error*Ly)/(height[-1]-height[0])
        data = np.loadtxt(dirPath + "../energy.dat")
        temp[t,0] = np.mean(data[:,3])
        temp[t,1] = np.std(data[:,3])/np.sqrt(data[:,3].shape[0])
        print("Temperature:", temp[t,0], "line tension:", gamma[t,0])
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlabel("$Box$ $height,$ $L_y$", fontsize=16)
    ax.set_xlabel("$Strain,$ $\\Delta L_y / {L_y}_0$", fontsize=16)
    ax.set_ylabel("$Force$ $on$ $wall,$ $F_{wall}$", fontsize=16)
    figure1Name = "/home/francesco/Pictures/soft/mips/wallForceVSTemp-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    xlabel = "$Temperature,$ $T/ \\varepsilon$"
    #if(which=='passive'):
    #    x = temp[:,0]
    #    xlabel = "$Temperature,$ $T/ \\varepsilon$"
    #elif(which=='active'):
    #    x = taup
    #    xlabel = "$Persistence$ $time,$ $\\tau_p / \\tau_i$"
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Line$ $tension,$ $\\gamma = W / \\Delta L_y$", fontsize=16)
    ax.errorbar(temp[:,0], gamma[:,0], gamma[:,1], temp[:,1], lw=1.2, marker='o', markersize=8, color='k', fillstyle='none', capsize=3)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/wallWorkVSTemp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPProfileVSSystemSize(dirName, figureName):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    dirList = np.array(['8192', '9192', '10192'])#, '11192', '12192'])#, '13192'])
    phiList = np.array(['0.30', '0.34-strain0.136', '0.36-strain0.246'])#, '0.38', '0.40'])#, '0.42'])
    colorList = cm.get_cmap('viridis', dirList.shape[0]+4)
    for i in range(dirList.shape[0]):
        dirSample = dirName + dirList[i] + "-2d/box/" + phiList[i]
        labelName = "$N=$"
        if not(os.path.exists(dirSample + "/singleProfile.dat")):
            interface.computeLinearDensityProfile(dirSample)
            print('computing')
        data = np.loadtxt(dirSample + "/singleProfile.dat")
        ax.plot(data[:,0], data[:,1], lw=1, marker='o', markersize=6, color=colorList(i/dirList.shape[0]), fillstyle='none', label=labelName + dirList[i])
        #center = np.mean(data[:,0])
        #x = np.abs(data[:,0]-center)
        #y = data[np.argsort(x),1]
        #yerr = data[np.argsort(x),2]
        #x = np.sort(x)
        #ax.errorbar(x, y, yerr, lw=1, marker='o', markersize=6, color=colorList(i/dirList.shape[0]), fillstyle='none', label=labelName + dirList[i])
    ax.set_xlabel("$x$", fontsize=16)
    ax.set_ylabel("$\\varphi(x)$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=12, loc='best')
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/profileVSSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPEvaporationRatePDF(dirName, figureName, which='passive', numSamples=30, numBins=20):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='passive'):
        dirList = np.array(['0.30', '0.32', '0.34', '0.36', '0.38', '0.40', '0.42'])
    elif(which=='active'):
        dirList = np.array(['1e-04', '1e-03', '1e-02', '1e-01'])
    elif(which=='compare'):
        dirList = np.array(['0.30', '1e-03'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    for d in range(dirList.shape[0]):
        if(which=='passive'):
            dirType = "/langevin-lj/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirType = "/langevin-lj/T0.30/active-lj/T0.30-tp" + dirList[d] + "-f01/dynamics/"
            labelName = "$\\tau_p=$"
        elif(which=='compare'):
            if(d==0):
                dirType = "/langevin-lj/T" + dirList[d] + "/dynamics/"
                labelName = "$passive,$ $T=$"
            else:
                dirType = "/langevin-lj/T0.30/active-lj/T0.30-tp" + dirList[d] + "-f01/dynamics/"
                labelName = "$active,$ $T=0.30$ $f_0=1$ $\\tau_p=$"
        rate = np.empty(0)
        for i in range(numSamples):
            dirSample = dirName + "-" + str(i) + dirType
            if(os.path.exists(dirSample)):
                rate = np.append(rate, np.loadtxt(dirSample + "/evaporationRate.dat"))
        if(rate.shape[0] != 0):
            pdf, edges = np.histogram(rate, bins=np.linspace(np.min(rate), np.max(rate), numBins), density=True)
            edges = (edges[1:] + edges[:-1])/2
            ax.plot(edges[pdf>0], pdf[pdf>0], marker='o', markersize=8, lw=1, fillstyle='none', markeredgewidth=1.5, color=colorList(d/dirList.shape[0]), label=labelName + dirList[d])
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlabel("$Evaporation$ $rate,$ $R_e = N_e / \\tau_{ISF}$", fontsize=16)
    ax.set_ylabel("$PDF(R_e)$", fontsize=16)
    figureName = "/home/francesco/Pictures/soft/mips/evaporatePDF-" + figureName
    plt.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterExchangeVSTemp(dirName, figureName, which='lj', Ly='1', numBins=20):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        dirList = np.array(['0.35', '0.36', '0.37', '0.38', '0.39', '0.40', '0.41'])
    elif(which=='active'):
        dirList = np.array(['1e-02', '9e-03', '8e-03', '7e-03', '6e-03', '5.5e-03', '5e-03', '4.5e-03', '4e-03', '3.5e-03', '3.3e-03', '3e-03',
                            '2.8e-03', '2.6e-03', '2.4e-03', '2.2e-03', '2e-03'])
    else:
        print("Please specify the sample type")
    colorList = cm.get_cmap('viridis', dirList.shape[0]+2)
    temp = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    condense = np.zeros((dirList.shape[0],2))
    evaporate = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(which=='lj'):
            dirSample = dirName + "box6-" + Ly + "/0.238/langevin-lj/T" + dirList[d] + "/dynamics/"
            labelName = "$T=$"
        elif(which=='active'):
            dirSample = dirName + "box/0.30/active-lj/T0.35-Dr" + dirList[d] + "-f08/dynamics/"
            labelName = "$D_r=$"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "condensationTime.dat")):
            interface.computeExchangeTimes(dirSample)
        data = np.loadtxt(dirSample + "condensationTime.dat")
        condense[d,0] = np.mean(data)
        condense[d,1] = np.std(data)
        pdf, edges = np.histogram(data, bins=np.geomspace(np.min(data), np.max(data), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        ax.loglog(edges[pdf>0], pdf[pdf>0], marker='$C$', markersize=6, lw=0.8, fillstyle='none', color=colorList(d/dirList.shape[0]), label=labelName + dirList[d])
        data = np.loadtxt(dirSample + "evaporationTime.dat")
        evaporate[d,0] = np.mean(data)
        evaporate[d,1] = np.std(data)
        pdf, edges = np.histogram(data, bins=np.geomspace(np.min(data), np.max(data), numBins), density=True)
        edges = (edges[1:] + edges[:-1])/2
        ax.loglog(edges[pdf>0], pdf[pdf>0], marker='$E$', markersize=6, lw=0.8, fillstyle='full', color=colorList(d/dirList.shape[0]))
        if not(os.path.exists(dirSample + "clusterTemperature.dat")):
            interface.computeClusterTemperatureVSTime(dirSample)
        data = np.loadtxt(dirSample + "clusterTemperature.dat")
        temp[d] = np.mean(data[:,1])
    #ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=14)
    #ax.legend(loc='best', fontsize=10, ncol=2)
    ax.set_xlabel("$time$", fontsize=16)
    ax.set_ylabel("$Distribution$", fontsize=16)
    figure1Name = "/home/francesco/Pictures/soft/mips/exchangeVSTemp-" + figureName
    plt.tight_layout()
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(which=='lj'):
        x = temp
        xlabel = "$Temperature,$ $T$"
    elif(which=='active'):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Average$ $time$", fontsize=16)
    ax.errorbar(x, condense[:,0], condense[:,1], lw=1.2, marker='$C$', markersize=15, color='k', fillstyle='none', capsize=3, label="$Condensation$")
    ax.errorbar(x, evaporate[:,1], evaporate[:,1], lw=1.2, marker='$E$', markersize=15, color='k', fillstyle='none', capsize=3, label="$Evaporation$")
    ax.legend(loc='best', fontsize=12, ncol=2)
    plt.tight_layout()
    figure2Name = "/home/francesco/Pictures/soft/mips/exchangeTimeVSTemp-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

################################################################################
############################### Cluster dynamics ###############################
################################################################################
def plotSPClusterTemperature(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['100', '10', '5', '1', '5e-01', '1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['100', '10', '5', '1', '5e-01', '1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['thermal25', 'thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        dirList = np.array(['0.25', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    temp = np.zeros((dirList.shape[0],2))
    tempIn = np.zeros((dirList.shape[0],2))
    tempOut = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-long/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            #phi[d] = utils.readFromParams(dirSample, "phi")
            phi[d] = np.loadtxt(dirSample + 'localDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
            #print(phi[d], utils.readFromParams(dirSample, "phi"))
        data = cluster.computeClusterTemperatureVSTime(dirSample)
        temp[d,0] = np.mean(data[:,0])
        temp[d,1] = np.std(data[:,0])
        tempIn[d,0] = np.mean(data[:,1])
        tempIn[d,1] = np.std(data[:,1])
        tempOut[d,0] = np.mean(data[:,2])
        tempOut[d,1] = np.std(data[:,2])
        print(dirList[d], "temp:", temp[d,0], "in:", tempIn[d,0], "out:", tempOut[d,0])
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterTemp-vsPhi-" + figureName
        ax.plot(x, phi, color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterTemp-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Temperature,$ $T$", fontsize=18)
    ax.errorbar(x, temp[:,0], temp[:,1], color='k', lw=1.2, marker='o', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Liquid + gas$')
    ax.errorbar(x, tempIn[:,0], tempIn[:,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Liquid$')
    ax.errorbar(x, tempOut[:,0], tempOut[:,1], color='g', lw=1.2, marker='v', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Gas$')
    if(fixed!="Dr"):
        ax.set_xscale('log')
        #ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterFluctuationsVSTime(dirName, figureName, fixed=False, which='10'):
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
        figureName = "/home/francesco/Pictures/soft/mips/pClusterTime-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        label = "$D_r$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterTime-vsDr-" + figureName + "-iod" + which
    else:
        label = "$m/\\gamma$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterTime-vsDamping-" + figureName + "-Dr" + which
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
    shape = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-long/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            #phi[d] = utils.readFromParams(dirSample, "phi")
            phi[d] = np.loadtxt(dirSample + 'localDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
            #print(phi[d], utils.readFromParams(dirSample, "phi"))
            if(d==0):
                numParticles = utils.readFromParams(dirSample, "numParticles")
        if(os.path.exists(dirSample + "clusterShape.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            data = np.loadtxt(dirSample + "clusterShape.dat")
            shape[d,0] = np.mean(data)
            shape[d,1] = np.std(data)
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterShape-vsPhi-" + figureName
        ax.plot(x, phi, color='k', lw=1.2, ls='--')
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterShape-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Liquid$ $shape$", fontsize=18)
    ax.errorbar(x, shape[:,0], shape[:,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', elinewidth=1, capsize=4)
    if(fixed!="Dr"):
        ax.set_xscale('log')
        #ax.set_yscale('log')
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
    figureName = "/home/francesco/Pictures/soft/mips/pClusterShapeVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterMixingTime(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(8,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '1e-02', '3e-03', '1e-03', '2e-04', '1e-04', '3e-05', '1e-05', '1e-06', '1e-07'])
            #dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['0.26', '0.28', '0.30', '0.32', '0.35', '0.40', '0.45', '0.50'])#, '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    mixingTime = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    tau = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-log/"
            sigma = utils.readFromDynParams(dirSample, 'sigma')
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*sigma)
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/short/"
            phi[d] = np.loadtxt(dirSample + 'localDensity-N16-stats.dat')[0]
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        # first get timescale from velocity correlation
        if(d==0):
            if not(os.path.exists(dirSample + "clusterVelCorr.dat")):
                cluster.computeClusterVelCorr(dirSample, 10, 7)
            data = np.loadtxt(dirSample + "clusterVelCorr.dat")
            #data[:,0] /= sigma
            tauScale = utils.computeTau(data, index=1)
        # get the cluster mixing correlation and extract lifetime
        if not(os.path.exists(dirSample + "clusterMixing.dat")):
            cluster.computeClusterMixing(dirSample, 10, 7)
        data = np.loadtxt(dirSample + "clusterMixing.dat")
        data[:,0] /= tauScale
        if(fixed=='phi'):
            label = "$D_r=$" + dirList[d]
        elif(fixed=='Dr'):
            label = "$\\varphi=$" + dirList[d]
        ax.errorbar(data[:,0], data[:,1], data[:,2], color=colorList(d/dirList.shape[0]), lw=1, marker='o', markersize=5, capsize=3, elinewidth=0.8, fillstyle='none', label=label)
        tau[d] = utils.computeTau(data, index=1, normalized=False)
        if(fixed=="phi"):
            taup[d] /= tauScale
    ax.set_xscale('log')
    if(fixed=="Dr"):
        #x = phi
        #xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pMixingTime-vsPhi-" + figureName
    elif(fixed=="phi"):
        #x = taup
        #xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pMixingTime-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Elapsed$ $time,$ $\\Delta t / \\tau_{vv}^{Gas}$", fontsize=16)
    ax.set_ylabel("$C_{mix}(\\Delta t)$", fontsize=16)
    ax.set_ylim(-0.05,1.05)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    label = "$\\frac{\\tau_p}{\\tau_{vv}^{Gas}}$"#"$\\Delta \\varphi^2_{16}}$"
    min = np.min(taup)
    max = np.max(taup)
    cb.set_ticks(np.linspace(0,1,5))
    cb.ax.tick_params(labelsize=12, length=0)
    ticks = np.geomspace(min, max, 5)
    ticklabels = []
    for i in range(ticks.shape[0]):
        ticklabels.append(np.format_float_scientific(ticks[i], 2))
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=24, labelpad=20, rotation='horizontal')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    # plot relaxation time
    fig, ax = plt.subplots(figsize = (6,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.semilogx(taup, tau, color='b', marker='o', markersize=8, fillstyle='none', lw=1)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p / \\tau_{vv}^{Gas}$", fontsize=16)
    ax.set_ylabel("$Cluster$ $lifetime,$ $\\tau_{mix} / \\tau_{vv}^{Gas}$", fontsize=16)
    fig.tight_layout()
    fig.savefig(figureName + "-tau-mix.png", transparent=True, format = "png")
    plt.show()

def plotSPClusterVelTimeCorr(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(8,5), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '1e-02', '3e-03', '1e-03', '2e-04', '1e-04', '3e-05', '1e-05', '1e-06', '1e-07'])
            #dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['0.26', '0.28', '0.30', '0.32', '0.35', '0.40', '0.45', '0.50'])#, '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    mixingTime = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    tau = np.zeros(dirList.shape[0])
    gaslp = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-log/"
            sigma = utils.readFromDynParams(dirSample, 'sigma')
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*sigma)
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/short/"
            phi[d] = np.loadtxt(dirSample + 'localDensity-N16-stats.dat')[0]
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "clusterVelCorr.dat")):
            cluster.computeClusterVelCorr(dirSample, 10, 7)
        data = np.loadtxt(dirSample + "clusterVelCorr.dat")
        data[:,0] /= sigma
        if(fixed=='phi'):
            label = "$D_r=$" + dirList[d]
        elif(fixed=='Dr'):
            label = "$\\varphi=$" + dirList[d]
        ax.errorbar(data[:,0], data[:,1], data[:,2], color=colorList(d/dirList.shape[0]), lw=1, marker='o', markersize=5, capsize=3, elinewidth=0.8, fillstyle='none', label=label)
        tau[d] = utils.computeTau(data, index=1)
    ax.set_xscale('log')
    if(fixed=="Dr"):
        #x = phi
        #xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pGasVelCorr-vsPhi-" + figureName
    elif(fixed=="phi"):
        #x = taup
        #xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pGasVelCorr-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Elapsed$ $time,$ $\\Delta t$", fontsize=18)
    ax.set_ylabel("$C_{vv}^{Gas}(\\Delta t)$", fontsize=18)
    ax.set_ylim(-0.05,1.05)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    label = "$\\tau_p$"#"$\\Delta \\varphi^2_{16}}$"
    min = np.min(taup)
    max = np.max(taup)
    cb.set_ticks(np.linspace(0,1,5))
    cb.ax.tick_params(labelsize=12, length=0)
    ticks = np.geomspace(min, max, 5)
    ticklabels = []
    for i in range(ticks.shape[0]):
        ticklabels.append(np.format_float_scientific(ticks[i], 2))
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=18, labelpad=20, rotation='horizontal')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    # plot relaxation time
    fig, ax = plt.subplots(figsize = (6,4), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.semilogx(taup, tau, color='k', marker='o', markersize=8, fillstyle='none', lw=1)
    #ax.plot(np.linspace(np.min(taup), np.max(taup), 100), np.ones(100)*tmax, ls='dotted', color='k')
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    ax.set_ylabel("$\\tau_{vv}^{Gas}$", fontsize=16)
    fig.tight_layout()
    fig.savefig(figureName + "-tau.png", transparent=True, format = "png")
    plt.show()

def plotSPClusterPersistence(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '1e-02', '3e-03', '1e-03', '2e-04', '1e-04', '3e-05', '1e-05', '1e-06', '1e-07'])
            #dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['0.26', '0.28', '0.30', '0.32', '0.35', '0.40', '0.45', '0.50'])#, '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    liquidlp = np.zeros((dirList.shape[0],2))
    gaslp = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-log/"
            sigma = utils.readFromDynParams(dirSample, 'sigma')
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*sigma)
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/short/"
            phi[d] = np.loadtxt(dirSample + 'localDensity-N16-stats.dat')[0]
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not (os.path.exists(dirSample + "clusterDecayLength.dat")):
            cluster.computeClusterVelCorr(dirSample, 10, 7)
        data = np.loadtxt(dirSample + os.sep + "clusterDecayLength.dat")
        gaslp[d,0] = data[0]
        gaslp[d,1] = data[1]
        liquidlp[d,0] = data[2]
        liquidlp[d,1] = data[3]
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pPers-vsPhi-" + figureName
    elif(fixed=="phi"):
        x = taup
        ax.set_xscale('log')
        ax.set_yscale('log')
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pPers-vsDr-" + figureName
        #t = np.linspace(2e03,3e06,100)
        #ax.plot(t, 0.9*t, ls='dashed', color='k', lw=1.3)
        #t = np.linspace(2e-03,2,100)
        #ax.plot(t, t**1.2, ls='dashdot', color='k', lw=1.3)
    ax.errorbar(x, liquidlp[:,0], liquidlp[:,1], capsize=3, color='b', marker='s', markersize=8, lw=1.2, fillstyle='none', label='$Liquid$')
    ax.errorbar(x, gaslp[:,0], gaslp[:,1], capsize=3, color=[1,0.5,0], marker='o', markersize=8, lw=1.2, fillstyle='none', label='$Gas$')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Persistence$ $length,$ $l_p$", fontsize=18)
    #ax.set_ylabel("$Typical$ $speed,$ $\\langle s \\rangle$", fontsize=18)
    ax.legend(fontsize=14, loc='best')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterPersistenceVSSystemSize(dirName, figureName, which='1e-05'):
    fig, ax = plt.subplots(figsize = (7,3.5), dpi = 120)
    numParticles = np.array([1024, 2048, 4096, 8192])#, 16384, 32768])
    dirList = np.array(['1024', '2048', '4096', '8192'])#, '16384', '32768'])
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$'])#, '$N = 16384$', '$N = 32768$'])
    gaslp = np.zeros((dirList.shape[0],2))
    liquidlp = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/thermal45/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics-log/"
        if(os.path.exists(dirSample)):
            sigma = utils.readFromDynParams(dirSample, 'sigma')
            if not (os.path.exists(dirSample + "clusterDecayLength.dat")):
                cluster.computeClusterVelCorr(dirSample, 10, 7)
            data = np.loadtxt(dirSample + os.sep + "clusterDecayLength.dat")
            print(dirSample)
            gaslp[d,0] = data[0]
            gaslp[d,1] = data[1]
            liquidlp[d,0] = data[2]
            liquidlp[d,1] = data[3]
    ax.errorbar(numParticles[gaslp[:,0]>0], liquidlp[gaslp[:,0]>0,0], liquidlp[gaslp[:,0]>0,1], capsize=3, color='b', marker='s', markersize=8, lw=1.2, fillstyle='none', label='$Liquid$')
    ax.errorbar(numParticles[gaslp[:,0]>0], gaslp[gaslp[:,0]>0,0], gaslp[gaslp[:,0]>0,1], capsize=3, color=[1,0.5,0], marker='o', markersize=8, lw=1.2, fillstyle='none', label='$Gas$')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xscale('log', basex=2)
    ax.set_xlabel('$System$ $size,$ $N$', fontsize=16)
    ax.set_ylabel('$Persistence$ $length,$ $l_p$', fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.set_xticks(numParticles)
    ax.set_xticklabels(dirList)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pPers-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusteringTime(dirName, figureName, fixed=False, which='1e-03', numBins=100):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7,6), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '1e-02', '3e-03', '1e-03', '2e-04', '1e-04', '3e-05', '1e-05', '1e-06', '1e-07'])
            #dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['0.26', '0.28', '0.30', '0.32', '0.35', '0.40', '0.45', '0.50'])#, '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    vapor = np.zeros((dirList.shape[0],2))
    condense = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-log/"
            sigma = utils.readFromDynParams(dirSample, 'sigma')
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*sigma)
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/short/"
            phi[d] = np.loadtxt(dirSample + 'localDensity-N16-stats.dat')[0]
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not (os.path.exists(dirSample + "evaporationTime.dat")):
            cluster.computeClusterRate(dirSample, 10, 7)
        data = np.loadtxt(dirSample + os.sep + "evaporationTime.dat")
        vapor[d,0] = np.mean(data)
        vapor[d,1] = np.std(data)
        #data = np.sort(np.unique(data))
        #pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), numBins), density=True)
        #edges = (edges[1:] + edges[:-1])/2
        #ax[0].semilogx(edges, pdf, marker='v', markersize=6, lw=0.8, fillstyle='none', color=colorList(d/dirList.shape[0]))
        ax[0].semilogy(np.arange(1, data.shape[0]+1, 1), np.sort(data), marker='v', markersize=4, lw=0.8, fillstyle='none', color=colorList(d/dirList.shape[0]))
        data = np.loadtxt(dirSample + os.sep + "condensationTime.dat")
        condense[d,0] = np.mean(data)
        condense[d,1] = np.std(data)
        #data = np.sort(np.unique(data))
        #pdf, edges = np.histogram(data, bins=np.linspace(np.min(data), np.max(data), numBins), density=True)
        #edges = (edges[1:] + edges[:-1])/2
        #ax[1].semilogx(edges, pdf, marker='o', markersize=6, lw=0.8, fillstyle='none', color=colorList(d/dirList.shape[0]))
        ax[1].semilogy(np.arange(1, data.shape[0]+1, 1), np.sort(data), marker='v', markersize=4, lw=0.8, fillstyle='none', color=colorList(d/dirList.shape[0]))
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/soft/mips/pClusteringPDF-vsPhi-" + figureName
        figure2Name = "/home/francesco/Pictures/soft/mips/pClustering-vsPhi-" + figureName
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/soft/mips/pClusteringPDF-vsDr-" + figureName
        figure2Name = "/home/francesco/Pictures/soft/mips/pClustering-vsDr-" + figureName
    for i in range(2):
        ax[i].tick_params(axis='both', labelsize=12)
        #ax[i].set_xlabel("$Time,$ $t$", fontsize=16)
        ax[i].set_xlabel("$Index$", fontsize=16)
    #ax[0].set_ylabel("$P(t_{vapor})$", fontsize=16)
    #ax[1].set_ylabel("$P(t_{condense})$", fontsize=16)
    ax[0].set_ylabel("$t_{vapor}$", fontsize=16)
    ax[1].set_ylabel("$t_{condense}$", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar, ax=ax, pad=0, aspect=30)
    label = "$\\tau_p$"
    min = np.min(taup)
    max = np.max(taup)
    cb.set_ticks(np.linspace(0,1,5))
    cb.ax.tick_params(labelsize=10, length=0)
    ticks = np.geomspace(min, max, 5)
    ticklabels = []
    for i in range(ticks.shape[0]):
        ticklabels.append(np.format_float_scientific(ticks[i], 1))
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=16, labelpad=5, rotation='horizontal', y=0.48)
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    # second plot
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    ax.errorbar(x, vapor[:,0], vapor[:,1], capsize=3, color='b', marker='s', markersize=8, lw=1.2, fillstyle='none', label='$Evaporation$')
    ax.errorbar(x, condense[:,0], condense[:,1], capsize=3, color=[1,0.5,0], marker='o', markersize=8, lw=1.2, fillstyle='none', label='$Condensation$')
    if(fixed=='phi'):
        ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Typical$ $timescale$", fontsize=18)
    ax.legend(fontsize=14, loc='best')
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterRateBySize(dirName, figureName, numBlocks=1, blockPower=6, blockFreq=1e05):
    #print("taup:", 1/(utils.readFromDynParams(dirName, "Dr")*utils.readFromDynParams(dirName, "sigma")))
    fig, ax = plt.subplots(figsize=(6,4), dpi = 120)
    if not(os.path.exists(dirName + "rateBySize.dat")):
        cluster.computeClusterRateBySize(dirName, numBlocks, blockPower, blockFreq)
    sigma = utils.readFromDynParams(dirName, "sigma")
    data = np.loadtxt(dirName + "rateBySize.dat")
    data = data[data[:,1]>0,:]
    data[:,1:] /= sigma
    ax.errorbar(data[data[:,4]>0,0], data[data[:,4]>0,1], data[data[:,4]>0,2], capsize=3, color=[1,0.5,0], marker='o', markersize=6, lw=0, elinewidth=0.9, fillstyle='none', label='$Condensation$')
    ax.errorbar(data[data[:,4]>0,0], data[data[:,4]>0,3], data[data[:,4]>0,4], capsize=3, color='b', marker='s', markersize=6, lw=0, elinewidth=0.9, fillstyle='none', label='$Evaporation$')
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Cluster$ $size,$ $A_c/A_{box}$", fontsize=16)
    ax.set_ylabel("$Time,$ $t/\\sigma$", fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.set_ylim(-1.2e04, 1.9e05)
    ax.set_xlim(1.5e-05, 0.47)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pClusterRate-" + figureName
    fig.savefig(figureName + ".png", transparent=False, format = "png")
    plt.show()

def plotSPClusterVel(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '1e-02', '3e-03', '1e-03', '2e-04', '1e-04', '3e-05', '1e-05', '1e-06', '1e-07'])
            #dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['0.26', '0.28', '0.30', '0.32', '0.35', '0.40', '0.45', '0.50'])#, '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    liquidlp = np.zeros((dirList.shape[0],2))
    gaslp = np.zeros((dirList.shape[0],2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-log/"
            sigma = utils.readFromDynParams(dirSample, 'sigma')
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*sigma)
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/short/"
            phi[d] = np.loadtxt(dirSample + 'localDensity-N16-stats.dat')[0]
        sigma = utils.readFromDynParams(dirSample, 'sigma')
        if not(os.path.exists(dirSample + "clusterVel.dat")):
            persistence = cluster.computeDelaunayClusterVel(dirSample)#*taup[d]
        else:
            persistence = np.loadtxt(dirSample + os.sep + "clusterVel.dat")#*taup[d]
        liquidlp[d,0] = np.mean(persistence[:,2])
        liquidlp[d,1] = np.std(persistence[:,2])
        gaslp[d,0] = np.mean(persistence[:,3])
        gaslp[d,1] = np.std(persistence[:,3])
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pPers-vsPhi-" + figureName
    elif(fixed=="phi"):
        x = taup
        ax.set_xscale('log')
        ax.set_yscale('log')
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pPers-vsDr-" + figureName
        t = np.linspace(2e03,3e06,100)
        ax.plot(t, 0.9*t, ls='dashed', color='k', lw=1.3)
        t = np.linspace(2e-03,2,100)
        ax.plot(t, t**1.2, ls='dashdot', color='k', lw=1.3)
    ax.errorbar(x, liquidlp[:,0], liquidlp[:,1], capsize=3, color='b', marker='s', markersize=8, lw=1.2, fillstyle='none', label='$Liquid$')
    ax.errorbar(x, gaslp[:,0], gaslp[:,1], capsize=3, color=[1,0.5,0], marker='o', markersize=8, lw=1.2, fillstyle='none', label='$Gas$')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    #ax.set_ylabel("$Persistence$ $length,$ $l_p$", fontsize=18)
    ax.set_ylabel("$Typical$ $speed,$ $\\langle s \\rangle$", fontsize=18)
    ax.legend(fontsize=14, loc='best')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPCollisionPersistence(dirName, figureName, fixed=False, which='10'):
    fig, ax = plt.subplots(figsize = (7, 5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67',  'thermal72', 'thermal78', 'thermal83', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$\\varphi = 0.45$', '$\\varphi = 0.58$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.83$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-02', '5e-02', '1e-01', '5e-01', '1', '2', '5', '10', '50', '100', '1000'])
        labelList = np.array(['$D_r = 0.01$', '$D_r = 0.05$', '$D_r = 0.1$', '$D_r = 0.5$', '$D_r = 1$', '$D_r = 2$', '$D_r = 5$', '$D_r = 10$', '$D_r = 50$', '$D_r = 100$', '$D_r = 1000$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod700', 'iod900', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 700$', '$\\beta \\sigma = 900$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 700, 900, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    tauc = np.zeros(dirList.shape[0])
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="iod"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod" + which + "/active-langevin/Dr1e-03-f0200/dynamics-col/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod" + which + "/active-langevin/Dr" + dirList[d] + "-f0200/dynamics-col/"
            Dr[d] = utils.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "-f0200/dynamics-col/"
            damping[d] = utils.readFromDynParams(dirSample, "damping")
        if(os.path.exists(dirSample + "/contactCollision.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            if(os.path.exists(dirSample + "/inClusterCollision.dat")):
                collision = np.loadtxt(dirSample + "/outClusterCollisionIntervals.dat")
                print('cluster')
            else:
                collision = np.loadtxt(dirSample + "/contactCollisionIntervals.dat")
                print('contact')
            collision, counts = np.unique(collision, return_counts=True)
            cdf = np.cumsum(counts)/np.sum(counts)
            cdf = cdf[1:]
            collision = collision[1:]
            failed = False
            try:
                popt, pcov = curve_fit(curveCumSum, collision, cdf, bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == False):
                ax.plot(collision, curveCumSum(collision, *popt), color='k', lw=2, linestyle='--')
                tauc[d] = 1/popt[0]
            else:
                tauc[d] = collision[np.argwhere(cdf>0.9)[0,0]]
            print(dirList[d], " timescale: ", tauc[d])
            ax.semilogx(collision, cdf, color=colorList(d/dirList.shape[0]), lw=1, marker='o', markersize=4, label=labelList[d], fillstyle='none')
            #ax.plot(np.ones(50)*taup[d], np.linspace(-0.1,1.1,50), color=colorList(d/dirList.shape[0]), ls='dotted', lw=1)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.set_ylim(-0.07, 1.07)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$\\Delta_c$", fontsize=18)
    ax.set_ylabel("$CDF(\\Delta_c)$", fontsize=18)
    #ax.set_ylim(2.3e-05, 162)
    fig.tight_layout()
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/soft/cluster/pCollision-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/cluster/pTaus-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = Dr
        xlabel = "$Rotational$ $diffusion,$ $D_r$"
        figure1Name = "/home/francesco/Pictures/soft/cluster/pCollision-vsDr-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/cluster/pTaus-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/soft/cluster/pCollision-vsDamping-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/soft/cluster/pTaus-vsDamping-" + figureName + "-Dr" + which
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize = (6.5, 3.5), dpi = 120)
    ax.tick_params(axis='both', labelsize=12)
    ax.semilogy(x, tauc, color='b', marker='s', markersize=8, fillstyle='none', lw=1)
    ax.semilogy(x, taup, color='k', marker='v', markersize=8, fillstyle='none', lw=1)
    if(fixed=='phi'):
        ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("$Timescales$", fontsize=16)
    ax.legend(("$Collision$ $rate$ $\\tau_c$", "$Persistence$ $time$ $\\tau_p$"), fontsize=12, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelSpaceCorr(dirName, figureName, fixed=False, which='10'):
    fig, ax = plt.subplots(figsize = (6.5, 5.5), dpi = 120)
    if(fixed=="iod"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67', 'thermal72', 'thermal78', 'thermal83',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$\\varphi = 0.45$', '$\\varphi = 0.58$', '$\\varphi = 0.67$', '$\\varphi = 0.72$', '$\\varphi = 0.78$', '$\\varphi = 0.83$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        dirList = np.array(['1e-05', '1e-04', '5e-04', '1e-03', '5e-03', '7e-03', '1e-02', '1.2e-02', '1.5e-02', '5e-02', '1e-01', '5e-01', '1', '5'])
        labelList = np.array(['$D_r = 0.00001$', '$D_r = 0.0001$', '$D_r = 0.0005$', '$D_r = 0.0007$', '$D_r = 0.001$', '$D_r = 0.0012$', '$D_r = 0.0015$', '$D_r = 0.005$', '$D_r = 0.01$', '$D_r = 0.05$', '$D_r = 0.1$', '$D_r = 0.5$', '$D_r = 1$', '$D_r = 5$'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod700', 'iod900', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 700$', '$\\beta \\sigma = 900$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 700, 900, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
        damping = np.zeros(dirList.shape[0])
    diff = np.zeros((dirList.shape[0],3))
    corrlength = np.zeros((dirList.shape[0],2))
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
        if(os.path.exists(dirSample + "spaceVelCorrInCluster.dat")):
            meanRad = np.mean(np.loadtxt(dirSample + "/particleRad.dat"))
            data = np.loadtxt(dirSample + "/spaceVelCorrInCluster.dat")
            if(d > 9):
                data = np.loadtxt(dirSample + "/spaceVelCorrOutCluster.dat")
            data[:,0] /= meanRad
            data = data[data[:,0]<122]
            #data = data[data[:,1]>0]
            ax.plot(data[1:,0], data[1:,1], color=colorList((dirList.shape[0]-d)/dirList.shape[0]), lw=1.2, label=labelList[d])
            #ax.plot(data[1:,0], data[1:,2], color=colorList((dirList.shape[0]-d)/dirList.shape[0]), lw=1.2, ls='--')
            failed = False
            try:
                popt, pcov = curve_fit(curveCvv, data[:,0], data[:,1])
            except RuntimeError:
                print("Error - curve_fit failed")
                failed = True
            if(failed == False and d < 12):
                corrlength[d,0] = 1/popt[0]
                corrlength[d,1] = utils.computeTau(data, index=1, threshold=np.exp(-1)*data[1,1], normalized=False)
                ax.plot(data[1:,0], curveCvv(data[1:,0], *popt), color=colorList((dirList.shape[0]-d)/dirList.shape[0]), lw=0.9, linestyle='--')
            #data = data[data[:,0]<80,:]
            data[:,1] /= data[0,1]
            diff[d,0] = np.sqrt(np.sum(data[:,0]*data[:,1]**2))
            diff[d,1] = np.sqrt(np.sum(data[:,0]*data[:,2]**2))
            diff[d,2] = np.sqrt(np.sum(data[:,0]*data[:,3]**2))
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Distance,$ $r \\, / \\, \\sigma$", fontsize=18)
    ax.set_ylabel("$Velocity$ $correlation,$ $C_{vv}^\\parallel(r)$", fontsize=18)
    fig.tight_layout()
    if(fixed=="iod"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/soft/cluster/pSpaceVelCorr-vsPhi-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/cluster/pSpaceDiff-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = 1/Dr
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/soft/cluster/pSpaceVelCorr-vsDr-" + figureName + "-iod" + which
        figure2Name = "/home/francesco/Pictures/soft/cluster/pSpaceDiff-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figure1Name = "/home/francesco/Pictures/soft/cluster/pSpaceVelCorr-vsDamping-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/soft/cluster/pSpaceDiff-vsDamping-" + figureName + "-Dr" + which
    fig.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig, ax = plt.subplots(figsize=(6.5,4.5), dpi = 120)
    ax.plot(x, diff[:,0], color='k', lw=1.2, marker='s', markersize=8, fillstyle='none', label="$\\sqrt{\\int | C_{vv}^{\\parallel}(r) |^2 dr}$")
    #ax.plot(x, diff[:,1], color='g', lw=1.2, marker='v', markersize=8, fillstyle='none', label="$\\sqrt{\\int | C_{vv}^{\\perp}(r) |^2 dr}$")
    #ax.plot(x, diff[:,2], color='b', lw=1.2, marker='o', markersize=8, fillstyle='none', label="$\\sqrt{\\int | C_{vv}^{cross}(r) |^2 dr}$")
    if(fixed!="phi" and fixed!="iod"):
        ax.set_yscale('log')
    #ax.set_ylim(-0.12, 2.52)
    #ax.set_xlim(5.8e-06, 2.8e03)
    if(fixed=="phi"):
        ax.set_xscale('log')
    ax.legend(fontsize=14, loc='best')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Velocity$ $correlations$", fontsize=18, labelpad=5)
    fig.tight_layout()
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelPhiPDF(dirName, figureName, fixed=False, which='1.5e-04'):
    fig, ax = plt.subplots(1,2, figsize=(12,5), dpi = 120)
    if(fixed=="Dr"):
        dirList = np.array(['thermal45',  'thermal58', 'thermal67',  'thermal78', 'thermal85',  'thermal88',  'thermal94', 'thermal1'])#, 'thermal1'])
        labelList = np.array(['$\\varphi = 0.45$', '$\\varphi = 0.58$', '$\\varphi = 0.67$', '$\\varphi = 0.78$', '$\\varphi = 0.85$', '$\\varphi = 0.88$', '$\\varphi = 0.94$', '$\\varphi = 1$'])#, '$\\varphi = 1.00$'])
        colorList = cm.get_cmap('viridis', dirList.shape[0]+1)
        phi = np.zeros(dirList.shape[0])
    elif(fixed=="phi"):
        if(which == '0.45'):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        colorList = cm.get_cmap('plasma', dirList.shape[0]+1)
        Dr = np.zeros(dirList.shape[0])
    else:
        dirList = np.array(['iod1', 'iod2', 'iod5', 'iod10', 'iod20', 'iod50', 'iod100', 'iod200', 'iod500', 'iod700', 'iod900', 'iod1000'])
        labelList = np.array(['$\\beta \\sigma = 1$', '$\\beta \\sigma = 2$', '$\\beta \\sigma = 5$', '$\\beta \\sigma = 10$', '$\\beta \\sigma = 20$', '$\\beta \\sigma = 50$', '$\\beta \\sigma = 100$', '$\\beta \\sigma = 200$', '$\\beta \\sigma = 500$', '$\\beta \\sigma = 700$', '$\\beta \\sigma = 900$', '$\\beta \\sigma = 1000$'])
        iod = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 700, 900, 1000])
        colorList = cm.get_cmap('cividis', dirList.shape[0])
    damping = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr" + which + "/dynamics/"
            phi[d] = utils.readFromParams(dirSample, "phi")
        elif(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
            Dr[d] = utils.readFromDynParams(dirSample, "Dr")
        else:
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
        if(os.path.exists(dirSample + "velPDFInCluster.dat")):
            if(d==0):
                damping[d] = utils.readFromDynParams(dirSample, "damping")
                f0 = utils.readFromDynParams(dirSample, "f0")
                ax[1].plot(np.ones(50)*f0/(2*damping[d]), np.linspace(-1,20,50), color='k', ls='dashed', lw=1)
            data = np.loadtxt(dirSample + "/velPDFInCluster.dat")
            ax[1].plot(data[:,0], data[:,1], color=colorList(d/dirList.shape[0]), lw=1.2)#, label=labelList[d])
        if(os.path.exists(dirSample + "velPDFOutCluster.dat")):
            data = np.loadtxt(dirSample + "/velPDFOutCluster.dat")
            ax[1].plot(data[:,0], data[:,1], color=colorList(d/dirList.shape[0]), lw=1.2, ls='--')
        if(os.path.exists(dirSample + "localVoroDensity-N16.dat")):
            data = np.loadtxt(dirSample + "localVoroDensity-N16.dat")
            ax[0].semilogy(data[1:,0], data[1:,1], color=colorList(d/dirList.shape[0]), lw=1.2)#, label=labelList[d])
            #ax[0].plot(data[data[:,1]>0,0], -np.log(data[data[:,1]>0,1]), color=colorList(d/dirList.shape[0]), lw=1.2, label=labelList[d])
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[1].set_xlabel("$Speed,$ $|\\vec{v}|$", fontsize=16)
    ax[1].set_ylabel("$Distribution,$ $P(|\\vec{v}|)$", fontsize=16)
    #ax[1].set_xscale('log')
    ax[1].set_ylim(-0.8,19.2)
    ax[0].set_xlabel("$Local$ $density,$ $\\varphi_{local}$", fontsize=16)
    ax[0].set_ylabel("$P(\\varphi_{local})$", fontsize=16)
    #ax[0].set_xlabel("$Local$ $density,$ $\\varphi$", fontsize=18)
    #ax[0].set_ylabel("$Free$ $energy,$ $F(\\varphi)$", fontsize=18)
    colorBar = cm.ScalarMappable(cmap=colorList)
    cb = plt.colorbar(colorBar)
    label = "$D_r$"
    cb.set_ticks([0, 1])
    cb.ax.tick_params(labelsize=12, length=0)
    #cb.ax.invert_yaxis()
    ticklabels = [dirList[0], dirList[-1]]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=16, labelpad=-10, rotation='horizontal')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.22)
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pVelPhiPDF-vsPhi-" + figureName + "-iod" + which
    elif(fixed=="phi"):
        x = 1/Dr
        xlabel = "$Persistent$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pVelPhiPDF-vsDr-" + figureName + "-iod" + which
    else:
        x = damping
        xlabel = "$Damping,$ $\\gamma$"
        figureName = "/home/francesco/Pictures/soft/mips/pVelPhiPDF-vsDamping-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPVelPDF(dirName, figureName):
    fig, ax = plt.subplots(figsize=(6,3.5), dpi = 120)
    if(os.path.exists(dirName + "velPDFInCluster.dat")):
        data = np.loadtxt(dirName + "/velPDFInCluster.dat")
        ax.plot(data[:,0], data[:,1], color='b', lw=1.2, label="$Fluid$")
    print("Temperature in the fluid:", 0.5*data[np.argmax(data[:,1]),0]**2)
    ax.set_ylim(-1,np.max(data[:,1]+2))
    if(os.path.exists(dirName + "velPDFOutCluster.dat")):
        data = np.loadtxt(dirName + "/velPDFOutCluster.dat")
        ax.plot(data[:,0], data[:,1], color='g', lw=1.2, ls='--', label="$Gas$")
        damping = utils.readFromDynParams(dirName, "damping")
        f0 = utils.readFromDynParams(dirName, "f0")
        s0 = f0/(2*damping)
        ax.plot(np.ones(50)*s0, np.linspace(-5,20,50), color='k', ls='dashed', lw=1)
        print("Temperature in the gas:", 0.5*data[np.argmax(data[:,1]),0]**2, "active speed:", s0)
    #ax.set_xlim(-0.012,0.47)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Speed,$ $|\\vec{v}|$", fontsize=16)
    ax.set_ylabel("$Distribution,$ $P(|\\vec{v}|)$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/cluster/pVelPDF-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPressurePDF(dirName, figureName):
    fig, ax = plt.subplots(figsize=(6,3.5), dpi = 120)
    if(os.path.exists(dirName + "pressurePDFInCluster.dat")):
        data = np.loadtxt(dirName + "/pressurePDFInCluster.dat")
        ax.plot(data[:,0], data[:,1], color='g', lw=1.2, label="$Virial$ $inside$ $cluster$")
        ax.plot(data[:,2], data[:,3], color='b', lw=1.2, label="$Thermal$ $inside$ $cluster$")
        ax.plot(data[:,4], data[:,5], color='r', lw=1.2, label="$Active$ $inside$ $cluster$")
    if(os.path.exists(dirName + "pressurePDFOutCluster.dat")):
        data = np.loadtxt(dirName + "/pressurePDFOutCluster.dat")
        ax.plot(data[:,0], data[:,1], color='g', lw=1.2, ls='--', label="$Virial$ $outside$ $cluster$")
        ax.plot(data[:,2], data[:,3], color='b', lw=1.2, ls='--', label="$Thermal$ $outside$ $cluster$")
        ax.plot(data[:,4], data[:,5], color='r', lw=1.2, ls='--', label="$Active$ $outside$ $cluster$")
    #ax.set_ylim(-0.8,17.2)
    #ax.set_xlim(-0.012,0.47)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Pressure,$ $p$", fontsize=16)
    ax.set_ylabel("$Distribution,$ $P(p)$", fontsize=16)
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/cluster/pPressurePDF-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

################################################################################
######################### Cluster pressure properties ##########################
################################################################################
def plotSPClusterArea(dirName, figureName, fixed=False, which='1e-03'):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        taup = np.zeros(dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['0.25', '0.26', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        colorList = cm.get_cmap('viridis', dirList.shape[0])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    fluidArea = np.zeros((dirList.shape[0],2))
    gasArea = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-long/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
            if(os.path.exists(dirSample + "delaunayDensity.dat")):
                data = np.loadtxt(dirSample + "delaunayDensity.dat")
                phi[d] = np.mean(data[:,3])
        if(os.path.exists(dirSample + "delaunayArea.dat")):
            data = np.loadtxt(dirSample + "delaunayArea.dat")
            fluidArea[d,0] = np.mean(data[:,1])
            fluidArea[d,1] = np.std(data[:,1])
            gasArea[d,0] = np.mean(data[:,2])
            gasArea[d,1] = np.std(data[:,2])
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterArea-vsPhi-" + figureName
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterArea-vsDr-" + figureName
        ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Liquid$ $area,$ $A_{liquid}$", fontsize=18)
    ax.errorbar(x[fluidArea[:,0]>0], fluidArea[fluidArea[:,0]>0,0], fluidArea[fluidArea[:,0]>0,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Fluid$')
    #ax.errorbar(x[gasArea[:,0]>0], gasArea[gasArea[:,0]>0,0], gasArea[gasArea[:,0]>0,1], color='g', lw=1.2, marker='o', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Gas$')
    #ax.legend(fontsize=14, loc='best')
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterAreaVSTime(dirName, figureName, which=False):
    fig, ax = plt.subplots(figsize=(7.5,4), dpi = 120)
    data = np.loadtxt(dirName + "delaunayArea.dat")
    dt = utils.readFromParams(dirName, 'dt')
    if(which == 'fluid'):
        figureName = figureName + '-' + which
        ax.plot(data[:,0]*dt, data[:,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', label='$Fluid$')
    elif(which == 'gas'):
        figureName = figureName + '-' + which
        ax.plot(data[:,0]*dt, data[:,2], color='g', lw=1.2, marker='o', markersize = 8, fillstyle='none', label='$Gas$')
    else:
        ax.plot(data[:,0]*dt, data[:,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', label='$Fluid$')
        ax.plot(data[:,0]*dt, data[:,2], color='g', lw=1.2, marker='o', markersize = 8, fillstyle='none', label='$Gas$')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $time,$ $t$", fontsize=18)
    ax.set_ylabel("$Area$ $fraction$", fontsize=18)
    ax.legend(fontsize=14, loc='best')
    figureName = "/home/francesco/Pictures/soft/mips/pArea-vsTime-" + figureName
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterDensity(dirName, figureName, fixed=False, which='1e-03', inter='inter'):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        #dirList = np.array(['thermal25', 'thermal30', 'thermal35', 'thermal40', 'thermal45', 'thermal52', 'thermal58', 'thermal62', 'thermal67', 'thermal72',  'thermal78', 'thermal85',  'thermal94', 'thermal1'])
        dirList = np.array(['0.25', '0.26', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
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
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-long/"
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
            if(fixed=="Dr"):
                phi[d] = np.mean(data[:,3])
                print(dirList[d], phi[d], fluidDensity[d,0], gasDensity[d,0])
    if(fixed=="Dr"):
        x = phi
        xlabel = "$Volume$ $fraction,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterPhi-vsPhi-" + figureName
        ax.plot(x, x, color='k', lw=1.2, ls='--')
        if(inter=='inter'):
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
            # another way to compute the lower bound that matches the first
            ax.set_ylim(-0.18, 1.02)
            y = np.linspace(-0.2, 1.05, 100)
            dirSample = dirName + os.sep + dirList[10+index-1] + "/active-langevin/Dr" + which + "/dynamics/"
            phi1 = np.mean(np.loadtxt(dirSample + "delaunayDensity.dat")[:,3])
            dirSample = dirName + os.sep + dirList[10+index] + "/active-langevin/Dr" + which + "/dynamics/"
            phi2 = np.mean(np.loadtxt(dirSample + "delaunayDensity.dat")[:,3])
            ax.plot(np.ones(100)*(phi1+phi2)/2, y, ls='dotted', color='k', lw=1)
            phiup = (phi1+phi2)/2
            print("phiup from density avarage:", phiup)
            print("phiup from interpolation:", phiupper)
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
            # another way to compute the lower bound that matches the first
            ax.set_ylim(-0.18, 1.02)
            y = np.linspace(-0.2, 1.05, 100)
            dirSample = dirName + os.sep + dirList[index-1] + "/active-langevin/Dr" + which + "/dynamics/"
            phi1 = np.mean(np.loadtxt(dirSample + "delaunayDensity.dat")[:,3])
            dirSample = dirName + os.sep + dirList[index] + "/active-langevin/Dr" + which + "/dynamics/"
            phi2 = np.mean(np.loadtxt(dirSample + "delaunayDensity.dat")[:,3])
            ax.plot(np.ones(100)*(phi1+phi2)/2, y, ls='dotted', color='k', lw=1)
            philow = (phi1+phi2)/2
            print("phidown from density avarage:", philow)
            print("phidown from interpolation:", philower)
            np.savetxt(dirName + "MIPSBounds.dat", np.column_stack((numParticles, philower, phiupper, philow, phiup)))
    elif(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pClusterPhi-vsDr-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Liquid$ $Delaunay$ $density$", fontsize=18)
    ax.errorbar(x[fluidDensity[:,0]>0], fluidDensity[fluidDensity[:,0]>0,0], fluidDensity[fluidDensity[:,0]>0,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Fluid$')
    if(fixed=="Dr"):
        ax.errorbar(x[gasDensity[:,0]>0], gasDensity[gasDensity[:,0]>0,0], gasDensity[gasDensity[:,0]>0,1], color='g', lw=1.2, marker='o', markersize = 8, fillstyle='none', elinewidth=1, capsize=4, label='$Gas$')
    #ax.legend(fontsize=14, loc='best')
    if(fixed!="Dr"):
        ax.set_xscale('log')
    #ax.set_xlim(5.8e-06, 2.8e03)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterDensityVSTime(dirName, figureName, which=False):
    fig, ax = plt.subplots(figsize=(7.5,4), dpi = 120)
    data = np.loadtxt(dirName + "delaunayDensity.dat")
    dt = utils.readFromParams(dirName, 'dt')
    if(which == 'fluid'):
        figureName = figureName + '-' + which
        ax.plot(data[:,0]*dt, data[:,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', label='$Fluid$')
    elif(which == 'gas'):
        figureName = figureName + '-' + which
        ax.plot(data[:,0]*dt, data[:,2], color='g', lw=1.2, marker='o', markersize = 8, fillstyle='none', label='$Gas$')
    else:
        ax.plot(data[:,0]*dt, data[:,1], color='b', lw=1.2, marker='s', markersize = 8, fillstyle='none', label='$Fluid$')
        ax.plot(data[:,0]*dt, data[:,2], color='g', lw=1.2, marker='o', markersize = 8, fillstyle='none', label='$Gas$')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Simulation$ $time,$ $t$", fontsize=18)
    ax.set_ylabel("$Density$", fontsize=18)
    ax.legend(fontsize=14, loc='best')
    figureName = "/home/francesco/Pictures/soft/mips/pDensity-vsTime-" + figureName
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPAreaVSRemoved(dirName, figureName, which='fluid', logx='logx'):
    fig, ax = plt.subplots(figsize=(7,3.5), dpi = 120)
    Nlist = np.array(['4096', '8192', '16384'])
    sampleList = np.array(['0.28', '0.27', '0.27'])
    dirList = np.array([['10', '50', '100', '200', '300', '400', '500', '1000', '1500'],
                        ['10', '100', '500', '1000', '1500', '2100', '2200', '2300', '2500'],
                        ['100', '500', '1000', '2000', '3000', '4000', '4500', '5000', '5500', '6000', '6100', '6200', '6300', '6500', '7500']], dtype=object)
    markerList = np.array(['v', 'o', 's'])
    lsList = np.array(['dotted', 'dashdot', 'solid'])
    for n in range(Nlist.shape[0]):
        fluidArea = []
        gasArea = []
        phi = []
        dirInit = dirName + os.sep + Nlist[n] + "-2d" + "/densitySweep/" + sampleList[n] + "/active-langevin/Dr2e-04/dynamics/"
        data = np.loadtxt(dirInit + os.sep + "delaunayDensity.dat")
        initialPhi = np.mean(data[:,3])
        print("N", Nlist[n], "initial density:", initialPhi, "+-", np.std(data[:,3]))
        lengthscale = utils.readFromDynParams(dirInit, "sigma")
        for d in range(len(dirList[n])):
            #print(Nlist[n], dirList[n][d])
            dirSample = dirName + os.sep + Nlist[n] + "-2d" + "/densitySweep/" + sampleList[n] + "/active-langevin/Dr2e-04/" + dirList[n][d] + "removed/dynamics/"
            if(os.path.exists(dirSample)):
                if(os.path.exists(dirSample + os.sep + "delaunayArea.dat")):
                    delaunayArea = np.loadtxt(dirSample + os.sep + "delaunayArea.dat")
                else:
                    cluster.computeClusterDelaunayArea(dirSample)
                    delaunayArea = np.loadtxt(dirSample + os.sep + "delaunayArea.dat")
                fluidArea.append(np.mean(delaunayArea[:,1]))
                gasArea.append(np.mean(delaunayArea[:,2]))
                if(os.path.exists(dirSample + os.sep + "delaunayDensity.dat")):
                    delaunayDensity = np.loadtxt(dirSample + os.sep + "delaunayDensity.dat")
                else:
                    cluster.computeClusterDelaunayDensity(dirSample)
                    delaunayDensity = np.loadtxt(dirSample + os.sep + "delaunayDensity.dat")
                phi.append(np.mean(delaunayDensity[:,3]))
                #print(Nlist[n], dirList[n][d], phi[d])
        phi = np.array(phi)
        fluidArea = np.array(fluidArea)
        gasArea = np.array(gasArea)
        if(which == 'fluid'):
            if(logx == 'logx'):
                ax.semilogx(initialPhi - phi, fluidArea, markersize=8, marker=markerList[n], ls=lsList[n], label="$N = $" + Nlist[n], lw=1.2, fillstyle='none')
            else:
                ax.plot(phi, fluidArea, markersize=8, marker=markerList[n], ls=lsList[n], label="$N = $" + Nlist[n], lw=1.2, fillstyle='none')
            ax.set_ylabel("$Liquid$ $area$", fontsize=18)
        elif(which == 'gas'):
            if(logx == 'logx'):
                ax.semilogx(initialPhi - phi, gasArea, markersize=8, marker=markerList[n], ls=lsList[n], label="$N = $" + Nlist[n], lw=1.2, fillstyle='none')
            else:
                ax.plot(phi, gasArea, markersize=8, marker=markerList[n], ls=lsList[n], label="$N = $" + Nlist[n], lw=1.2, fillstyle='none')
            ax.set_ylabel("$Gas$ $area$", fontsize=18)
        #ax.plot(phi, fluidArea, markersize=8, marker=markerList[n], label="$N = $" + Nlist[n], lw=1.2, fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    if(logx == 'logx'):
        ax.set_xlabel("$Density$ $difference,$ $\\varphi^{low}$ - $\\varphi$", fontsize=18)
    else:
        ax.set_xlabel("$Volume$ $fraction,$ $\\varphi$", fontsize=18)
    ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    if(logx == 'logx'):
        figureName = "/home/francesco/Pictures/soft/mips/pRemoved-" + figureName + "-logx"
    else:
        figureName = "/home/francesco/Pictures/soft/mips/pRemoved-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPressureProfile(dirName, figureName, shiftx=0, which='delta'):
    fig, ax = plt.subplots(figsize = (8,3.5), dpi = 120)
    fileName = "pressureProfile.dat"
    if not(os.path.exists(dirName + os.sep + fileName)):
        cluster.averageLinearPressureProfile(dirName, shiftx=shiftx)
    data = np.loadtxt(dirName + os.sep + fileName)
    sigma = float(utils.readFromDynParams(dirName, 'sigma'))
    binWidth = (data[1,0] - data[0,0])
    vel = np.loadtxt(dirName + os.sep + "particleVel.dat")
    temp = np.mean(np.linalg.norm(vel,axis=1)**2) / 2
    gamma = 0.5 * np.sum((data[:,1] + data[:,3] + data[:,5] - data[:,2] - data[:,4] - data[:,6]) * binWidth)
    print("surface tension: ", gamma, " over temperature: ", gamma / temp)
    if(which=='component'):
        ax.plot(data[:,0], data[:,1], lw=1.2, color='r', ls='solid', label='$\\sigma_N^{Thermal}$')
        ax.plot(data[:,0], data[:,2], lw=1.2, color='r', ls='dashed', label='$\\sigma_T^{Thermal}$')
        ax.plot(data[:,0], data[:,3], lw=1.2, color='k', ls='solid', label='$\\sigma_N^{Steric}$')
        ax.plot(data[:,0], data[:,4], lw=1.2, color='k', ls='dashed', label='$\\sigma_T^{Steric}$')
        ax.plot(data[:,0], data[:,5], lw=1.2, color=[0.1,0.8,0.2], ls='solid', label='$\\sigma_N^{Active}$')
        ax.plot(data[:,0], data[:,6], lw=1.2, color=[0.1,0.8,0.2], ls='dashed', label='$\\sigma_T^{Active}$')
    elif(which=='delta'):
        gamma = np.cumsum((data[:,1] + data[:,3] + data[:,5] - data[:,2] - data[:,4] - data[:,6]) * binWidth)
        ax.plot(data[:,0], data[:,1] + data[:,3] + data[:,5] - data[:,2] - data[:,4] - data[:,6], lw=1.2, color='k', ls='solid', label='$\\sigma_N - \\sigma_T$')
        ax.plot(data[:,0], gamma, lw=1.5, color='b', ls='dashdot', label='$\int_0^L (\\sigma_N - \\sigma_T) dx$')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Position,$ $x$", fontsize=16)
    if(which=="component"):
        #ax.set_ylabel("$Pressure,$ $p \\sigma^2$", fontsize=16)
        ax.set_ylim(1.3*np.min(data[:,6]), 1.3*np.max(data[:,6]))
        figureName = "/home/francesco/Pictures/soft/mips/pProfile-" + figureName + ".png"
        ax.legend(fontsize=11, loc='upper left', ncol=3)
    elif(which=="delta"):
        #ax.set_ylabel("$\\Delta p^\\ast = (p_N - p_T) \\sigma^2$", fontsize=16)
        #ax.set_ylim(-2.2,0.8)
        figureName = "/home/francesco/Pictures/soft/mips/pcompProfile-" + figureName + ".png"
        ax.legend(fontsize=12, loc='best')
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPSurfaceTension(dirName, figureName, fixed='phi', which='2e-04'):
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    if(fixed=="phi"):
        dirList = np.array(['3e-04', '2e-04', '1e-04', '9e-05', '7e-05'])
        taup = np.zeros(dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['0.20', '0.22', '0.24', '0.26', '0.28', '0.30'])
        phi = np.zeros(dirList.shape[0])
    else:
        print('please specify fixed parameter')
    gamma = np.zeros(dirList.shape[0])
    temp = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr" + which + "/dynamics/"
        if not(os.path.exists(dirSample + os.sep + "pressureProfile.dat")):
            cluster.averageLinearPressureProfile(dirSample)
        data = np.loadtxt(dirSample + os.sep + "pressureProfile.dat")
        sigma = float(utils.readFromDynParams(dirSample, 'sigma'))
        binWidth = (data[1,0] - data[0,0])
        gamma[d] = 0.5 * np.sum((data[:,1] + data[:,3] + data[:,5] - data[:,2] - data[:,4] - data[:,6]) * binWidth)
        vel = np.loadtxt(dirSample + os.sep + "particleVel.dat")
        temp[d] = np.mean(np.linalg.norm(vel,axis=1)**2) / 2
        #if not(os.path.exists(dirSample + os.sep + "clusterTemperature.dat")):
        #    cluster.computeClusterTemperatureVSTime(dirSample)
        #data = np.loadtxt(dirSample + os.sep + "clusterTemperature.dat")
        #temp[d,0] = np.mean(data[:,0])
        #temp[d,1] = np.std(data[:,0])
        print(dirList[d], "surface tension: ", gamma[d], " temperature: ", temp[d])
    if(fixed=='phi'):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pGammaVSDr-" + figureName + ".png"
    elif(fixed=='Dr'):
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pGammaVSPhi-" + figureName + ".png"
    ax.plot(taup, gamma/temp, lw=1.2, marker='o', markersize=8, color='k', fillstyle='none')
    #ax.errorbar(temp[:,0], gamma/temp[:,0], temp[:,1]*gamma, lw=1.2, marker='o', markersize=8, color='k', fillstyle='none')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(xlabel, fontsize=16)
    #ax.set_xlabel("$Temperature,$ $T$", fontsize=16)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=16)
    #ax.set_ylabel("$Line$ $tension,$ $\\gamma$", fontsize=16)
    #ax.set_xticks((0.0022, 0.0026, 0.0030, 0.0034))
    ax.set_ylabel("$\\frac{\\gamma L}{T}$", fontsize=22, rotation='horizontal', labelpad=15)
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPRadialPressureProfile(dirName, figureName, which='delta'):
    fig, ax = plt.subplots(figsize = (6.5,5), dpi = 120)
    fileName = "pressureProfile.dat"
    if not(os.path.exists(dirName + os.sep + fileName)):
        cluster.averageRadialPressureProfile(dirName)
    data = np.loadtxt(dirName + os.sep + fileName)
    data = data[1:,:]
    sigma = float(utils.readFromDynParams(dirName, 'sigma'))
    binWidth = (data[1,0] - data[0,0])
    print("surface tension: ", 0.5 * np.sum((data[:,1] - data[:,2]) * binWidth) / (np.pi*sigma))
    if(which=='pressure'):
        ax.plot(data[:,0], data[:,3], lw=1.2, color='k', ls='--', label='$Steric$')
        ax.plot(data[:,0], data[:,4], lw=1.2, color='r', ls='dotted', label='$Thermal$')
        ax.plot(data[:,0], data[:,5], lw=1.2, color=[1,0.5,0], label='$Active$')
        ax.plot(data[:,0], np.sum(data[:,3:],axis=1), lw=1.2, color='b', ls='dashdot', label='$Total$')
    elif(which=='component'):
        #gamma = np.cumsum((data[:,1]-data[:,2])*binWidth)
        ax.plot(data[:,0], data[:,1], lw=1.2, color='k', ls='solid', label='$p_N(r)$')
        x = (data[1:,0] + data[:-1,0])/2
        y = (data[1:,1] - data[:-1,1])/(data[1:,0] - data[:-1,0])
        ax.plot(x, y, lw=1.2, color='k', ls='dashed', label='$dp_N(r) / dr$')
        #ax.plot(data[:,0], data[:,2], lw=1.2, color='g', ls='dashdot', label='$p_T$')
        #ax.plot(data[:,0], data[:,1] - data[:,2], lw=1.2, color='b', ls='solid', label='$p_N - p_T$')
        #ax.plot(data[:,0], gamma, lw=1.2, color='r', ls='solid')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Position,$ $x$", fontsize=16)
    if(which=="pressure"):
        ax.set_ylabel("$Pressure,$ $p \\sigma^2$", fontsize=16)
        ax.set_ylim(np.min(data[:,9])-0.2, np.max(data[:,11])+0.6)
        figureName = "/home/francesco/Pictures/soft/mips/pRadProfile-" + figureName + ".png"
    elif(which=="component"):
        ax.set_ylabel("$Normal$ $pressure,$  $p_N \\sigma^2$", fontsize=16)
        #ax.set_ylim(np.min(data[:,2])-0.4, np.max(data[:,1])+0.6)
        figureName = "/home/francesco/Pictures/soft/mips/pRadDeltaProfile-" + figureName + ".png"
    ax.legend(fontsize=14, loc='best', ncol=4)
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPLJPressureProfile(dirName, figureName, shift=0, which='delta'):
    fig, ax = plt.subplots(figsize = (11,4), dpi = 120)
    fileName = "pressureProfile.dat"
    if not(os.path.exists(dirName + os.sep + fileName)):
        cluster.averageLinearLJPressureProfile(dirName, shiftx=shift)
    data = np.loadtxt(dirName + os.sep + fileName)
    sigma = float(utils.readFromDynParams(dirName, 'sigma'))
    binWidth = (data[1,0] - data[0,0])
    data = data[:-1,:]
    print("surface tension: ", np.sum((data[:,2] - data[:,3]) * binWidth))
    if(which=='component'):
        ax.plot(data[:,0], data[:,1], lw=1.2, color='r', ls='dotted', label='$Thermal$')
        ax.plot(data[:,0], data[:,2], lw=1.2, color='k', ls='solid', label='$Steric - xx$')
        ax.plot(data[:,0], data[:,3], lw=1.2, color='k', ls='dashed', label='$Steric - yy$')
        #ax.plot(data[:,0], data[:,4], lw=1.2, color='g', ls='solid', label='$Repulsive - xx$')
        #ax.plot(data[:,0], data[:,5], lw=1.2, color='g', ls='dashed', label='$Repulsive - yy$')
        #ax.plot(data[:,0], data[:,6], lw=1.2, color=[1,0.5,0], ls='solid', label='$Attractive - xx$')
        #ax.plot(data[:,0], data[:,7], lw=1.2, color=[1,0.5,0], ls='dashed', label='$Attractive - yy$')
        ax.plot(data[:,0], 0.5*data[:,1] + data[:,2] + data[:,3], lw=1.5, color='b', ls='dashdot', label='$Total$')
    elif(which=='delta'):
        gamma = np.cumsum((data[:,2] - data[:,3]) * binWidth)
        ax.plot(data[:,0], data[:,1] + data[:,2], lw=1.2, color='g', ls='dashdot', label='$p_N$')
        ax.plot(data[:,0], data[:,1] + data[:,3], lw=1.2, color='b', ls='dashed', label='$p_T$')
        ax.plot(data[:,0], data[:,2] - data[:,3], lw=1.2, color='k', ls='solid', label='$p_N - p_T$')
        ax.plot(data[:,0], gamma, lw=1.5, color='r', ls='dotted', label='$\int_0^L (p_N - p_T) dx$')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Position,$ $x$", fontsize=16)
    if(which=="component"):
        ax.set_ylabel("$Pressure,$ $p \\sigma^2$", fontsize=16)
        #ax.set_ylim(np.min(data[:,7])-0.1, np.max(data[:,8])+0.2)
        figureName = "/home/francesco/Pictures/soft/mips/pLJProfile-" + figureName + ".png"
    elif(which=="delta"):
        #ax.set_ylabel("$\\Delta p^\\ast = (p_N - p_T) \\sigma^2$", fontsize=16)
        #ax.set_ylim(np.min(data[:,1] - data[:,2])-0.1, np.max(data[:,1] - data[:,2])+0.2)
        figureName = "/home/francesco/Pictures/soft/mips/pcompLJProfile-" + figureName + ".png"
    ax.legend(fontsize=10, loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPLJRadialPressureProfile(dirName, figureName, which='pressure'):
    fig, ax = plt.subplots(figsize = (6.5,5), dpi = 120)
    fileName = "pressureProfile.dat"
    if not(os.path.exists(dirName + os.sep + fileName)):
        cluster.averageRadialLJPressureProfile(dirName)
    data = np.loadtxt(dirName + os.sep + fileName)
    print(data.shape[0])
    data = data[2:30,:]
    sigma = float(utils.readFromDynParams(dirName, 'sigma'))
    if(which=='pressure'):
        ax.plot(data[:,0], data[:,3], lw=1.2, color='k', ls='--', label='$Steric$')
        ax.plot(data[:,0], data[:,4], lw=1.2, color='r', ls='dotted', label='$Thermal$')
        ax.plot(data[:,0], np.sum(data[:,3:],axis=1), lw=1.2, color='b', ls='dashdot', label='$Total$')
    elif(which=='component'):
        #gamma = np.cumsum((data[:,1]-data[:,2])*binWidth)
        ax.plot(data[:,0], data[:,1], lw=1.2, color='k', ls='solid', label='$p_N(r)$')
        x = (data[1:,0] + data[:-1,0])/2
        y = (data[1:,1] - data[:-1,1])/(data[1:,0] - data[:-1,0])
        ax.plot(x, y, lw=1.2, color='k', ls='dashed', label='$dp_N(r) / dr$')
        #ax.plot(data[:,0], data[:,2], lw=1.2, color='g', ls='dashdot', label='$p_T$')
        #ax.plot(data[:,0], data[:,1] - data[:,2], lw=1.2, color='b', ls='solid', label='$p_N - p_T$')
        #ax.plot(data[:,0], gamma, lw=1.2, color='r', ls='solid')
        centers = (x[1:] + x[:-1])/2
        binWidth = centers[1] - centers[0]
        print("surface tension: ", 0.5 * np.sum((y[1:] - y[:-1]) * centers * binWidth) / sigma)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("$Position,$ $x$", fontsize=16)
    if(which=="pressure"):
        ax.set_ylabel("$Pressure,$ $p \\sigma^2$", fontsize=16)
        ax.set_ylim(np.min(data[:,7])-0.1, np.max(data[:,8])+0.2)
        figureName = "/home/francesco/Pictures/soft/mips/pRadLJProfile-" + figureName + ".png"
    elif(which=="component"):
        ax.set_ylabel("$Normal$ $pressure,$  $p_N \\sigma^2$", fontsize=16)
        #ax.set_ylim(np.min(data[:,1])-0.05, np.max(data[:,1])+0.)
        figureName = "/home/francesco/Pictures/soft/mips/pRadDeltaLJProfile-" + figureName + ".png"
    ax.legend(fontsize=14, loc='best', ncol=4)
    fig.tight_layout()
    fig.savefig(figureName, transparent=True, format = "png")
    plt.show()

def plotSPClusterSurfaceTension(dirName, figureName, fixed='Dr'):
    fig, ax = plt.subplots(figsize=(6,5), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
        taup = np.zeros(dirList.shape[0])
    elif(fixed=="Dr"):
        dirList = np.array(['0.25', '0.26', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    tension = np.zeros((dirList.shape[0], 2))
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
            taup[d] = 1/(utils.readFromParams(dirSample, "Dr") * utils.readFromParams(dirSample, "sigma"))
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr2e-04/dynamics/"
            phi[d] = np.loadtxt(dirSample + 'localVoroDensity-N16-stats.dat')[0]#'localDensity-N16-stats.dat'
        if(os.path.exists(dirSample + "/borderEnergy.dat")):
            data = np.loadtxt(dirSample + "/surfaceWork.dat")
            tension[d,0] = np.mean((data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]))
            tension[d,1] = np.std((data[1:,3] - data[:-1,3]) / (data[1:,4] - data[:-1,4]))
    ax.tick_params(axis='both', labelsize=14)
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figureName = "/home/francesco/Pictures/soft/mips/pSurfaceTension-vsDr-" + figureName
        ax.set_xscale('log')
        #ax.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pSurfaceTension-vsPhi-" + figureName
    ax.errorbar(x[tension[:,0]>0], tension[tension[:,0]>0,0], tension[tension[:,0]>0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Surface$ $tension,$ $\\gamma$", fontsize=18, labelpad=15)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterBorderWorkVSTime(dirName, figureName):
    #numParticles = int(utils.readFromParams(dirName, "numParticles"))
    data = np.loadtxt(dirName + os.sep + "borderWork.dat")
    data = data[1:,:]
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    ax.plot(data[:,0], data[:,1]/data[:,4], linewidth=1.2, color='k')
    ax.plot(data[:,0], data[:,2]/data[:,4], linewidth=1.2, ls='dotted', color='r')
    ax.plot(data[:,0], data[:,3]/data[:,4], linewidth=1.2, ls='dashed', color=[1,0.5,0])
    ax.plot(data[:,0], (data[:,1]+data[:,2]+data[:,3])/data[:,4], linewidth=1.2, ls='dashdot', color='b')
    ax.set_yscale('log')
    # plotting settings
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax.set_ylabel("$Line$ $tension,$ $W / L_c$", fontsize=15)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pLineTensionVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterLJBorderWorkVSTime(dirName, figureName):
    #numParticles = int(utils.readFromParams(dirName, "numParticles"))
    data = np.loadtxt(dirName + os.sep + "borderWork.dat")
    fig, ax = plt.subplots(figsize=(7,4), dpi = 120)
    ax.plot(data[:,0], data[:,1]/data[:,3], linewidth=1.2, color='k')
    ax.plot(data[:,0], data[:,2]/data[:,3], linewidth=1.2, ls='dotted', color='r')
    ax.plot(data[:,0], (data[:,1]+data[:,2])/data[:,3], linewidth=1.2, ls='dashdot', color='b')
    # plotting settings
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Simulation$ $step$", fontsize=15)
    ax.set_ylabel("$Line$ $tension,$ $W / L_c$", fontsize=15)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pLineTensionLJVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPClusterSizeVSTime(dirName, figureName):
    timeStep = float(utils.readFromParams(dirName, "dt"))
    data = np.loadtxt(dirName + os.sep + "clusterSizeVSTime.dat")
    fig, ax = plt.subplots(figsize=(7,5), dpi = 120)
    #ax.errorbar(data[1:,0]*timeStep, data[1:,1], data[1:,2], linewidth=1.2, color='k', capsize=3)
    ax.plot(data[1:,0]*timeStep, data[1:,1], linewidth=1.2, color='k', marker='o', fillstyle='none')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # plotting settings
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel("$Time,$ $t$", fontsize=15)
    ax.set_ylabel("$Cluster$ $size,$ $\\langle l_c \\rangle$", fontsize=15)
    plt.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pClusterSizeVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPForceVelMagnitude(dirName, figureName, fixed=False):
    fig, ax = plt.subplots(figsize=(7.5,5), dpi = 120)
    if(fixed=="phi"):
        if(which == '0.45'):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        dirList = np.array(['0.25', '0.26', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
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
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr2e-04/dynamics/"
            data = np.loadtxt(dirSample + "delaunayDensity.dat")
            phi[d] = np.mean(data[:,3])
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
        figureName = "/home/francesco/Pictures/soft/mips/pVelMagnitude-vsDr-" + figureName
        ax.set_xscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pVelMagnitude-vsPhi-" + figureName
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

def plotSPTotalPressure(dirName, figureName, fixed='Dr'):
    fig, ax = plt.subplots(figsize = (7,4), dpi = 120)
    if(fixed=="phi"):
        phi = utils.readFromParams(dirName, "phi")
        if(phi == 0.45):
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '3e-03', '2e-03', '1.5e-03', '1.2e-03', '1e-03', '7e-04', '5e-04', '3e-04', '2e-04', '1.5e-04', '1e-04', '7e-05', '5e-05', '3e-05', '2e-05', '1.5e-05', '1e-05', '5e-06', '2e-06', '1.5e-06', '1e-06', '5e-07', '2e-07', '1.5e-07', '1e-07'])
        else:
            dirList = np.array(['1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    elif(fixed=="Dr"):
        dirList = np.array(['0.25', '0.26', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
        phi = np.zeros(dirList.shape[0])
    else:
        print("please specify the fixed parameter")
    p = np.zeros((dirList.shape[0], 4, 2))
    taup = np.zeros(dirList.shape[0])
    for d in range(dirList.shape[0]):
        if(fixed=="phi"):
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics/"
        elif(fixed=="Dr"):
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr2e-04/dynamics/"
            data = np.loadtxt(dirSample + "delaunayDensity.dat")
            phi[d] = np.mean(data[:,3])
        if(os.path.exists(dirSample + "/pressure.dat")):
            taup[d] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            #print(dirList[d], taup[d])
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
        figureName = "/home/francesco/Pictures/soft/mips/pTotPressure-vsDr-" + figureName
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figureName = "/home/francesco/Pictures/soft/mips/pTotPressure-vsPhi-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    # pressure components
    ax.errorbar(x[p[:,0,0]>0], p[p[:,0,0]>0,0,0], p[p[:,0,0]>0,0,1], lw=1.2, color='k', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Steric$')
    ax.errorbar(x[p[:,1,0]>0], p[p[:,1,0]>0,1,0], p[p[:,1,0]>0,1,1], lw=1.2, color='r', marker='v', markersize=8, fillstyle='none', capsize=3, label='$Thermal$')
    ax.errorbar(x[p[:,2,0]>0], p[p[:,2,0]>0,2,0], p[p[:,2,0]>0,2,1], lw=1.2, color=[0.1,0.8,0.2], marker='s', markersize=8, fillstyle='none', capsize=3, label='$Actve$')
    #ax.errorbar(x[p[:,3,0]>0], p[p[:,3,0]>0,3,0], p[p[:,3,0]>0,3,1], lw=1.2, color='b', marker='o', markersize=8, fillstyle='none', capsize=3, label='$Total$')
    #ax.set_xlim(5.8e-06, 2.8e03)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("$Pressure$ $components$", fontsize=18)
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
        dirList = np.array(['0.25', '0.26', '0.27', '0.28', '0.29', '0.30', '0.31', '0.32', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.82', '0.84', '0.86', '0.88', '0.90', '0.92', '0.94', '0.96'])
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
            dirSample = dirName + os.sep + "iod10/active-langevin/Dr" + dirList[d] + "/dynamics-long/"
        elif(fixed=="Dr"):
            #dirSample = dirName + os.sep + dirList[d] + "/langevin/T0.001/iod10/active-langevin/Dr1.5e-04/dynamics/"
            dirSample = dirName + os.sep + dirList[d] + "/active-langevin/Dr2e-04/dynamics/"
            data = np.loadtxt(dirSample + "delaunayDensity.dat")
            phi[d] = np.mean(data[:,3])
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
                phic = cluster.averageLocalVoronoiDensity(dirSample)
            print("interpolation indices:", index-1, index, "tauc:", taupc, "density:", phic)
            np.savetxt(dirName + "iod10/active-langevin/" + which + "Tradeoff.dat", np.array([phic, taupc]))
    if(fixed=="phi"):
        x = taup
        xlabel = "$Persistence$ $time,$ $\\tau_p$"
        figure1Name = "/home/francesco/Pictures/soft/mips/pGasFluid-vsDr-" + figureName
        figure2Name = "/home/francesco/Pictures/soft/mips/pPressures-vsDr-" + figureName
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        figure1Name = "/home/francesco/Pictures/soft/mips/pGasFLuid-vsPhi-" + figureName + "-Dr" + which
        figure2Name = "/home/francesco/Pictures/soft/mips/pPressures-vsPhi-" + figureName + "-Dr" + which
        #ax1.set_yscale('log')
        ax2.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)
    # dense and dilute pressure
    #ax1.errorbar(x[ptotIn[:,0]>0], ptotIn[ptotIn[:,0]>0,0], ptotIn[ptotIn[:,0]>0,1], color='b', marker='s', markersize=8, fillstyle='none', lw=1.2, capsize=3, label='$Fluid$')
    #ax1.errorbar(x[ptotOut[:,0]>0], ptotOut[ptotOut[:,0]>0,0], ptotIn[ptotIn[:,0]>0,1], color=[1,0.5,0], marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3, label='$Gas$')
    deltaP = ptotIn[:,0] - ptotOut[:,0]
    errorDeltaP = np.sqrt(ptotIn[:,1]**2 + ptotOut[:,1]**2)
    ax1.errorbar(x, np.abs(deltaP), errorDeltaP, color='k', marker='o', markersize=8, fillstyle='none', lw=1.2, capsize=3)
    #ax1.legend(fontsize=14, loc='best')
    ax1.set_ylabel("$Pressure,$ $p$", fontsize=18)
    ax1.set_ylabel("$|\\Delta p|$", fontsize=18)
    ax1.set_yscale('log')
    #ax1.set_ylabel("$\\Delta p = p_{liquid} - p_{gas}$", fontsize=18)
    # pressure components
    if(fixed=='phi'):
        ax2.errorbar(x[pOut[:,0,0]>0], pOut[pOut[:,0,0]>0,0,0], pOut[pOut[:,0,0]>0,0,1], lw=1.1, color='k', marker='o', markersize=8, markeredgewidth=1.3, fillstyle='none', capsize=3, label='$P^{Steric}_{gas}$')
        ax2.errorbar(x[pOut[:,2,0]>0], pOut[pOut[:,2,0]>0,2,0], pOut[pOut[:,2,0]>0,2,1], lw=1.1, color=[0.1,0.8,0.2], marker='s', markersize=8, markeredgewidth=1.3, fillstyle='none', capsize=3, label='$P^{Active}_{gas}$')
        ax2.errorbar(x[pOut[:,1,0]>0], pOut[pOut[:,1,0]>0,1,0], pOut[pOut[:,1,0]>0,1,1], lw=1.1, color='r', marker='v', markersize=8, markeredgewidth=1.3, fillstyle='none', capsize=3, label='$P^{Thermal}_{gas}$')
        ax2.errorbar(x[pIn[:,0,0]>0], pIn[pIn[:,0,0]>0,0,0], pIn[pIn[:,0,0]>0,0,1], lw=1.1, color='k', marker='o', markersize=6, fillstyle='full', capsize=3, label='$P^{Steric}_{liquid}$')
        ax2.errorbar(x[pIn[:,2,0]>0], pIn[pIn[:,2,0]>0,2,0], pIn[pIn[:,2,0]>0,2,1], lw=1.1, color=[0.1,0.8,0.2], marker='s', markeredgecolor='k', markeredgewidth=0.9, markersize=6, fillstyle='full', capsize=3, label='$P^{Active}_{liquid}$')
        ax2.errorbar(x[pIn[:,1,0]>0], pIn[pIn[:,1,0]>0,1,0], pIn[pIn[:,1,0]>0,1,1], lw=1.1, color='r', marker='v', markeredgecolor='k', markeredgewidth=0.9, markersize=6, fillstyle='full', capsize=3, label='$P^{Thermal}_{liquid}$')
    else:
        ax2.errorbar(x, pOut[:,0,0], pOut[:,0,1], lw=1.1, color='k', marker='o', markersize=8, markeredgewidth=1.5, fillstyle='none', capsize=3)
        ax2.errorbar(x, pOut[:,1,0], pOut[:,1,1], lw=1.1, color='r', marker='v', markersize=8, markeredgewidth=1.5, fillstyle='none', capsize=3)
        ax2.errorbar(x, pOut[:,2,0], pOut[:,2,1], lw=1.1, color=[0.1,0.8,0.2], marker='s', markersize=8, markeredgewidth=1.5, fillstyle='none', capsize=3)
        ax2.errorbar(x, pIn[:,0,0], pIn[:,0,1], lw=1.1, color='k', marker='o', markersize=6, fillstyle='full', capsize=3, label='$Steric$')
        ax2.errorbar(x, pIn[:,1,0], pIn[:,1,1], lw=1.1, color='r', marker='v', markeredgecolor='k', markersize=6, fillstyle='full', capsize=3, label='$Thermal$')
        ax2.errorbar(x, pIn[:,2,0], pIn[:,2,1], lw=1.1, color=[0.1,0.8,0.2], marker='s', markeredgecolor='k', markersize=6, fillstyle='full', capsize=3, label='$Active$')
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
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel("$Pressure$ $components$", fontsize=18)
    ax2.legend(fontsize=12, loc='best', ncol=2)
    fig1.tight_layout()
    fig1.savefig(figure1Name + ".png", transparent=True, format = "png")
    fig2.tight_layout()
    fig2.savefig(figure2Name + ".png", transparent=True, format = "png")
    plt.show()

def plotClusterPressureVSTime(dirName, figureName, bound=False):
    #numParticles = int(utils.readFromParams(dirName, "numParticles"))
    pressure = np.loadtxt(dirName + os.sep + "delaunayPressure.dat")
    fig, ax = plt.subplots(1, 2, sharey=True, figsize = (11, 5), dpi = 120)
    if(bound == "bound"):
        ax[0].plot(pressure[:,0], pressure[:,1], linewidth=1.2, color='g', label="$wall$")
        ax[1].plot(pressure[:,0], pressure[:,1], linewidth=1.2, color='g', label="$wall$")
    # dense particles
    ax[0].plot(pressure[:,0], pressure[:,2], linewidth=1.2, color='k', ls='solid', label="$Steric$")
    ax[0].plot(pressure[:,0], pressure[:,3], linewidth=1.2, color='r', ls='dashed', label="$Thermal$")
    ptot = pressure[:,2] + pressure[:,3]
    ax[0].plot(pressure[:,0], pressure[:,4], linewidth=1.2, color=[0.1,0.8,0.2], ls='dashdot', label="$Active$")
    ptot += pressure[:,4]
    ax[0].plot(pressure[:,0], ptot, linewidth=1.2, color='b', ls='dotted', label="$Total$")
    # dilute particles
    ax[1].plot(pressure[:,0], pressure[:,5], linewidth=1.2, color='k', ls='solid', label="$Steric$")
    ax[1].plot(pressure[:,0], pressure[:,6], linewidth=1.2, color='r', ls='dashed', label="$Thermal$")
    ptot = pressure[:,5] + pressure[:,6]
    ax[1].plot(pressure[:,0], pressure[:,7], linewidth=1.2, color=[0.1,0.8,0.2], ls='dashdot', label="$Active$")
    ptot += pressure[:,7]
    ax[1].plot(pressure[:,0], ptot, linewidth=1.2, color='b', ls='dotted', label="$Total$")
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    # plotting settings
    ax[0].tick_params(axis='both', labelsize=12)
    ax[1].tick_params(axis='both', labelsize=12)
    ax[0].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[1].set_xlabel("$Simulation$ $step$", fontsize=15)
    ax[0].set_ylabel("$Pressure$", fontsize=15)
    #ax[0].legend(fontsize=12, loc='best', title='$Liqud$', title_fontsize=15)
    #ax[1].legend(fontsize=12, loc='best', title='$Gas$', title_fontsize=15)
    ax[1].legend(fontsize=12, loc='best')
    #ax.set_ylim(50, 700)
    plt.tight_layout()
    fig.subplots_adjust(wspace=0)
    figureName = "/home/francesco/Pictures/soft/mips/pClusterPressureVSTime-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
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
    figureName = "/home/francesco/Pictures/soft/mips/pClusterTradeoff-" + figureName
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Density$", fontsize=18)
    ax.set_xlabel("$Tradeoff$ $time,$ $\\tau_p^*$", fontsize=18)
    fig.tight_layout()
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPPhaseDiagram(dirName, numBins, figureName, variance='density'):
    fig, ax = plt.subplots(figsize=(8,6), dpi = 120)
    phi = np.array(['67', '72', '78', '82', '85', '88', '91'])#, '94'])
    Dr = np.array(['100', '10', '5', '1', '5e-01', '1e-01', '5e-02', '2e-02', '1e-02', '5e-03', '2e-03', '1e-03', '7e-04', '5e-04', '2e-04', '1e-04', '7e-05', '5e-05', '2e-05', '1e-05', '5e-06', '2e-06', '1e-06', '5e-07', '2e-07', '1e-07'])
    taup = np.zeros((phi.shape[0], Dr.shape[0]))
    # load the data
    deltaPhi = np.zeros((phi.shape[0], Dr.shape[0]))
    meanPhi = np.zeros((phi.shape[0], Dr.shape[0]))
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            dirSample = dirName + 'thermal' + phi[i] + '/langevin/T0.001/iod10/active-langevin/Dr' + Dr[j] + '/dynamics/'
            taup[i,j] = 1/(utils.readFromDynParams(dirSample, 'Dr')*utils.readFromDynParams(dirSample, 'sigma'))
            vel = np.loadtxt(dirSample + os.sep + "particleVel.dat")
            if(variance=='density'):
                deltaFile = dirSample + 'localDensity-N16-stats.dat'
                if not(os.path.exists(deltaFile)):
                    cluster.averageLocalDensity(dirSample, numBins=int(which))
                data = np.loadtxt(deltaFile)
                deltaPhi[i,j] = np.sqrt(data[1])/data[0]
            elif(variance=='cluster'):
                deltaFile = dirSample + 'simplexDistribution.dat'
                if not(os.path.exists(deltaFile + '!')):
                    cluster.computeSimplexClusterDistribution(dirSample)
                data = np.loadtxt(deltaFile)
                deltaPhi[i,j] = np.std(data[:,1]) / np.mean(data[:,1])
            elif(variance=='temperature'):
                deltaPhi[i,j] = np.std(np.linalg.norm(vel,axis=1)**2) / np.mean(np.linalg.norm(vel,axis=1)**2)
            # compute density values from average of delaunay density
            phiFile = dirSample + 'delaunayDensity.dat'
            if not(os.path.exists(phiFile)):
                cluster.computeClusterDelaunayDensity(dirSample)
            data = np.loadtxt(phiFile)
            meanPhi[i,j] = np.mean(data[:,3])
    # assign color based on deltaPhi
    colorId = np.zeros((phi.shape[0], Dr.shape[0]))
    min = np.min(deltaPhi)
    max = np.max(deltaPhi)
    bins = np.linspace(min, max+0.001, numBins)
    print('minimum intensity: ', min, ' maximum intensity: ', max)
    colorMap = cm.get_cmap('inferno', numBins)
    for i in range(phi.shape[0]):
        for j in range(Dr.shape[0]):
            for k in range(numBins-1):
                if(deltaPhi[i,j] > bins[k] and deltaPhi[i,j] < bins[k+1]):
                    ax.semilogx(taup[i,j], meanPhi[i,j], color=colorMap((numBins-k)/numBins), marker='s', markersize=15, lw=0)
                    #print(phi[i])
                    #plt.pause(0.1)
    #data = np.loadtxt(dirName + "/gasFluidTradeoff.dat")
    #ax.plot(data[:,1], data[:,0], color='k', marker='o', markersize=6, markeredgewidth=1.2, fillstyle='none', lw=1)
    #data = np.loadtxt(dirName + "/stericActiveTradeoff.dat")#[0.3,0.7,0]
    #ax.plot(data[:,1], data[:,0], color='k', marker='v', markersize=8, markeredgewidth=1.2, fillstyle='none', lw=1)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("$Local$ $density,$ $\\langle \\varphi_{local} \\rangle$", fontsize=18)
    ax.set_xlabel("$Persistence$ $time,$ $\\tau_p$", fontsize=18)
    ax.set_ylim(0.21, 0.915)
    ax.plot(np.ones(50)*1e06, np.linspace(0,1.2,50), ls='dotted', color='k', lw=0.7)
    colorBar = cm.ScalarMappable(cmap=colorMap)
    cb = plt.colorbar(colorBar)
    label = "$\\frac{\\Delta \\varphi_{local}}{\\langle \\varphi_{local} \\rangle}$"#"$\\Delta \\varphi^2_{16}}$"
    cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cb.ax.tick_params(labelsize=14, length=0)
    cb.ax.invert_yaxis()
    dvar = (max - min)/5
    ticklabels = [np.format_float_positional(max, 2), np.format_float_positional(max - dvar,2), np.format_float_positional(max - 2*dvar,2), np.format_float_positional(max - 3*dvar,2), np.format_float_positional(max - 4*dvar,2), np.format_float_positional(min, 2)]
    cb.set_ticklabels(ticklabels)
    cb.set_label(label=label, fontsize=28, labelpad=40, rotation='horizontal')
    fig.tight_layout()
    figureName = "/home/francesco/Pictures/soft/mips/pPhaseDiagram-" + figureName
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
    figure1Name = "/home/francesco/Pictures/soft/mips/pDeltaPressureVSPhi-" + figureName
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
    figure2Name = "/home/francesco/Pictures/soft/mips/pDeltaPressurePhaseDiagram-" + figureName
    fig.savefig(figure2Name + ".png", transparent=True, format = "png")
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
    figureName = "/home/francesco/Pictures/soft/mips/pDeltaP-vsPhi-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

################################################################################
############################# System size analysis #############################
################################################################################
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
    figureName = "/home/francesco/Pictures/soft/mips/pDeltaP-vsSystemSize-" + figureName + "-Dr" + which
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

def plotSPMIPSBoundsVSSystemSize(dirName, figureName, which='up'):
    fig, ax = plt.subplots(figsize = (7,3.5), dpi = 120)
    numParticles = np.array([1024, 2048, 4096, 8192, 16384, 32768])
    dirList = np.array(['1024', '2048', '4096', '8192', '16384', '32768'])
    labelList = np.array(['$N = 1024$', '$N = 2048$', '$N = 4096$', '$N = 8192$', '$N = 16384$', '$N = 32768$'])
    bounds = np.zeros((dirList.shape[0],2))
    for d in range(dirList.shape[0]):
        dirSample = dirName + os.sep + dirList[d] + "-2d/densitySweep/"
        if(os.path.exists(dirSample + "/MIPSBounds.dat")):
            data = np.loadtxt(dirSample + "/MIPSBounds.dat")
            bounds[d,0] = data[1]
            bounds[d,1] = data[2]
            print(dirList[d], bounds[d,0], bounds[d,1])
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
    figureName = "/home/francesco/Pictures/soft/mips/pBounds-vsSystemSize-" + figureName
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
    figureName = "/home/francesco/Pictures/soft/mips/pClusterTension-vsSystemSize-" + figureName
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
    figureName = "/home/francesco/Pictures/soft/mips/pCluster-vsSystemSize-" + figureName + "-Dr" + which
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
    figureName = "/home/francesco/Pictures/soft/mips/pDynamics-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()

################################################################################
######################### Attractive droplet analysis ##########################
################################################################################
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
        figureName = "/home/francesco/Pictures/soft/mips/pProfile-" + figureName + ".png"
    elif(which=="delta"):
        ax.set_ylabel("$\\Delta p^\ast = p_{xx} - p_{yy} \\sigma^2$", fontsize=18)
        ax.set_ylim(np.min(data[:,2] - data[:,3])-0.2, np.max(data[:,2] - data[:,3])+0.6)
        figureName = "/home/francesco/Pictures/soft/mips/deltaProfile-" + figureName + ".png"
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
        figure1Name = "/home/francesco/Pictures/soft/mips/pDropletPressure-vsT-" + figureName + "-phi" + which
        figure2Name = "/home/francesco/Pictures/soft/mips/pTotalDropletPressure-vsT-" + figureName + "-phi" + which
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        x = phi
        xlabel = "$Density,$ $\\varphi$"
        loc1 = 'lower right'
        loc2 = 'upper left'
        figure1Name = "/home/francesco/Pictures/soft/mips/pDropletPressure-vsPhi-" + figureName + "-T" + which
        figure2Name = "/home/francesco/Pictures/soft/mips/pTotalDropletPressure-vsPhi-" + figureName + "-T" + which
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
    figureName = "/home/francesco/Pictures/soft/mips/pDropletTension-vsSystemSize-" + figureName
    fig.savefig(figureName + ".png", transparent=True, format = "png")
    plt.show()



if __name__ == '__main__':
    dirName = sys.argv[1]
    whichPlot = sys.argv[2]

############################ local density analysis ############################
    if(whichPlot == "simplex"):
        figureName = sys.argv[3]
        threshold = float(sys.argv[4])
        which = sys.argv[5]
        logy = sys.argv[6]
        plotSimplexDensity(dirName, figureName, threshold, which, logy)

    elif(whichPlot == "simplexlabel"):
        figureName = sys.argv[3]
        threshold = float(sys.argv[4])
        filter = sys.argv[5]
        logy = sys.argv[6]
        plotSimplexLabels(dirName, figureName, threshold, filter, logy)

    elif(whichPlot == "fitphi"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
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
        numBins = int(sys.argv[4])
        which = sys.argv[5]
        cdf = sys.argv[6]
        plotSPClusterPDF(dirName, figureName, numBins, which, cdf)

    elif(whichPlot == "shapepdf"):
        figureName = sys.argv[3]
        numBins = int(sys.argv[4])
        cdf = sys.argv[5]
        plotSPClusterShapePDF(dirName, figureName, numBins, cdf)

    elif(whichPlot == "clusterdistro"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        numBins = int(sys.argv[5])
        which = sys.argv[6]
        plotSPClusterDistribution(dirName, figureName, fixed, numBins, which)

    elif(whichPlot == "clustersize"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        simplex = sys.argv[6]
        plotSPClusterSize(dirName, figureName, fixed, which, simplex)

########################### Interface fluctuations #############################
    elif(whichPlot == "heightflu"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        qmax = float(sys.argv[5])
        plotSPClusterHeightFluctuations(dirName, figureName, which, qmax)

    elif(whichPlot == "heighttemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        qmax = float(sys.argv[5])
        plotSPClusterHeightVSTemp(dirName, figureName, which, qmax)

    elif(whichPlot == "heightcorr"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterHeightCorrelation(dirName, figureName, which)

    elif(whichPlot == "profile"):
        figureName = sys.argv[3]
        plotSPDensityProfile(dirName, figureName)

    elif(whichPlot == "width"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        param = sys.argv[5]
        plotSPClusterWidth(dirName, figureName, which, param)

    elif(whichPlot == "widthtemp"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterWidthVSTemp(dirName, figureName, which)

    elif(whichPlot == "energytemp"):
        figureName = sys.argv[3]
        plotSPActiveEnergyVSTemp(dirName, figureName)

    elif(whichPlot == "isf"):
        figureName = sys.argv[3]
        compare = sys.argv[4]
        plotSPClusterISF(dirName, figureName, compare)

    elif(whichPlot == "walltime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        index = int(sys.argv[5])
        numSamples = int(sys.argv[6])
        plotSPWallForceVSTime(dirName, figureName, which, index, numSamples)

    elif(whichPlot == "wallstrain"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        index = int(sys.argv[5])
        plotSPWallForceVSStrain(dirName, figureName, which, index)

    elif(whichPlot == "sampletime"):
        figureName = sys.argv[3]
        type = sys.argv[4]
        which = sys.argv[5]
        index = int(sys.argv[6])
        numSamples = int(sys.argv[7])
        plotSampleWallForceVSTime(dirName, figureName, type, which, index, numSamples)

    elif(whichPlot == "sampledelta"):
        figureName = sys.argv[3]
        type = sys.argv[4]
        numSamples = int(sys.argv[5])
        maxStrain = float(sys.argv[6])
        plotSampleWallDeltaForceVSStrain(dirName, figureName, type, numSamples, maxStrain)

    elif(whichPlot == "samplestrain"):
        figureName = sys.argv[3]
        type = sys.argv[4]
        which = sys.argv[5]
        index = int(sys.argv[6])
        numSamples = int(sys.argv[7])
        limit = int(sys.argv[8])
        maxStrain = float(sys.argv[9])
        temp = sys.argv[10]
        taup = sys.argv[11]
        plotSampleWallForceVSStrain(dirName, figureName, type, which, index, numSamples, limit, maxStrain, temp, taup)

    elif(whichPlot == "gammatau"):
        figureName = sys.argv[3]
        plotGammaVSActivity(dirName, figureName)

    elif(whichPlot == "walltemp"):
        figureName = sys.argv[3]
        sample = sys.argv[4]
        which = sys.argv[5]
        plotSPWallForceVSTemp(dirName, figureName, sample, which)

    elif(whichPlot == "profilesize"):
        figureName = sys.argv[3]
        plotSPProfileVSSystemSize(dirName, figureName)

    elif(whichPlot == "rate"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        numSamples = int(sys.argv[5])
        numBins = int(sys.argv[6])
        plotSPEvaporationRatePDF(dirName, figureName, which, numSamples, numBins)

    elif(whichPlot == "exchange"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        Ly = sys.argv[5]
        numBins = int(sys.argv[6])
        plotSPClusterExchangeVSTemp(dirName, figureName, which, Ly, numBins)

############################### Cluster dynamics ###############################
    elif(whichPlot == "clustertemp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterTemperature(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterflutime"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterFluctuationsVSTime(dirName, figureName, fixed, which)

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

    elif(whichPlot == "clustervcorr"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterVelTimeCorr(dirName, figureName, fixed, which)

    elif(whichPlot == "clustervel"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterVel(dirName, figureName, fixed, which)

    elif(whichPlot == "clusterpers"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterPersistence(dirName, figureName, fixed, which)

    elif(whichPlot == "clusteringtime"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        numBins = int(sys.argv[6])
        plotSPClusteringTime(dirName, figureName, fixed, which, numBins)

    elif(whichPlot == "ratebysize"):
        figureName = sys.argv[3]
        plotSPClusterRateBySize(dirName, figureName)

    elif(whichPlot == "perssize"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterPersistenceVSSystemSize(dirName, figureName, which)

    elif(whichPlot == "colpers"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPCollisionPersistence(dirName, figureName, fixed, which)

    elif(whichPlot == "spacevel"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPVelSpaceCorr(dirName, figureName, fixed, which)

    elif(whichPlot == "velphi"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPVelPhiPDF(dirName, figureName, fixed, which)

    elif(whichPlot == "velpdf"):
        figureName = sys.argv[3]
        plotSPVelPDF(dirName, figureName)

    elif(whichPlot == "pressurepdf"):
        figureName = sys.argv[3]
        plotSPPressurePDF(dirName, figureName)

######################### cluster pressure properties ##########################
    elif(whichPlot == "clusterptime"):
        figureName = sys.argv[3]
        bound = sys.argv[4]
        plotClusterPressureVSTime(dirName, figureName, bound)

    elif(whichPlot == "pprofile"):
        figureName = sys.argv[3]
        shiftx = float(sys.argv[4])
        which = sys.argv[5]
        plotSPPressureProfile(dirName, figureName, shiftx, which)

    elif(whichPlot == "pgamma"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPSurfaceTension(dirName, figureName, fixed, which)

    elif(whichPlot == "radprofile"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPRadialPressureProfile(dirName, figureName, which)

    elif(whichPlot == "ljprofile"):
        figureName = sys.argv[3]
        shiftx = float(sys.argv[4])
        which = sys.argv[5]
        plotSPLJPressureProfile(dirName, figureName, shiftx, which)

    elif(whichPlot == "radljprofile"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPLJRadialPressureProfile(dirName, figureName, which)

    elif(whichPlot == "remove"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        logx = sys.argv[5]
        plotSPAreaVSRemoved(dirName, figureName, which, logx)

    elif(whichPlot == "clusterarea"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        plotSPClusterArea(dirName, figureName, fixed, which)

    elif(whichPlot == "areatime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterAreaVSTime(dirName, figureName, which)

    elif(whichPlot == "clusterphi"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        which = sys.argv[5]
        inter = sys.argv[6]
        plotSPClusterDensity(dirName, figureName, fixed, which, inter)

    elif(whichPlot == "phitime"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterDensityVSTime(dirName, figureName, which)

    elif(whichPlot == "worktime"):
        figureName = sys.argv[3]
        plotSPClusterBorderWorkVSTime(dirName, figureName)

    elif(whichPlot == "ljworktime"):
        figureName = sys.argv[3]
        plotSPClusterLJBorderWorkVSTime(dirName, figureName)

    elif(whichPlot == "sizetime"):
        figureName = sys.argv[3]
        plotSPClusterSizeVSTime(dirName, figureName)

    elif(whichPlot == "clustergamma"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        plotSPClusterSurfaceTension(dirName, figureName, fixed)

    elif(whichPlot == "forcevel"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        plotSPForceVelMagnitude(dirName, figureName, fixed)

    elif(whichPlot == "totalp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        plotSPTotalPressure(dirName, figureName, fixed)

    elif(whichPlot == "clusterp"):
        figureName = sys.argv[3]
        fixed = sys.argv[4]
        inter = sys.argv[5]
        which = sys.argv[6]
        plotSPClusterPressure(dirName, figureName, fixed, inter, which)

    elif(whichPlot == "tradeoff"):
        figureName = sys.argv[3]
        which = sys.argv[4]
        plotSPClusterTradeoff(dirName, figureName, which)

    elif(whichPlot == "phased"):
        numBins = int(sys.argv[3])
        figureName = sys.argv[4]
        variance = sys.argv[5]
        plotSPPhaseDiagram(dirName, numBins, figureName, variance)

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

    elif(whichPlot == "clustersystem"):
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
