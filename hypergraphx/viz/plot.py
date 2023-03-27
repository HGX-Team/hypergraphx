import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

from hypergraphx import Hypergraph
from hypergraphx.measures import degree_distribution

def distr_bin(data, n_bin=30, logbin=True):
    ###This is a very old function copied from my c++ library. It's ugly but works :)
    """ Logarithmic binning of raw positive data;
        Input:
            data = np array,
            bins= number if bins,
            logbin = if true log bin
        Output (array: bins, array: hist) 
        bins: centred bins 
        hist: histogram value / (bin_length*num_el_data) [nonzero]
    """
    if len(data)==0:
        print( "Error empty data\n")
    min_d = float(min(data))
    if logbin and min_d<=0:
        print ("Error nonpositive data\n")
    n_bin = float(n_bin)            #ensure float values
    bins = np.arange(n_bin+1)
    if logbin:
        data = np.array(data)/min_d
        base= np.power(float(max(data)) , 1.0/n_bin)
        bins = np.power(base,bins)
        bins = np.ceil(bins)                   #to avoid problem for small ints
    else:
        data = np.array(data) + min_d          #to include negative data
        delta = (float(max(data)) - float(min(data)))/n_bin
        bins = bins*delta + float(min(data))
    n_bin = int(n_bin)
    #print ('first bin: ', bins[0], 'first data:', min(data), 'max bin:', bins[n_bin], 'max data', float(max(data)))
    hist = np.histogram(data, bins)[0]
    ii = np.nonzero(hist)[0]            #take non zero values of histogram
    bins = bins[ii]
    hist = hist[ii]
    bins=np.append(bins,float(max(data)))          #append the last bin
    bin_len = np.diff(bins)
    bins =  bins[:-1] + bin_len/2.0     #don't return last bin, centred boxes
    if logbin:
        hist = hist/bin_len                 #normalize values
        bins = bins*min_d                   #restore original bin values
    else:
        bins = bins - min_d 
    res = list(zip(bins, hist/float(sum(hist))))    #restore original bin values, norm hist
    return list(zip(*res))


def plot_degree_distributions(hg: Hypergraph, max_size=5, ax=None, **kwargs):
    """
    Plots the degree distributions of the hypergraph at orders 1, 2, 3 and 4.

    Parameters
    ----------
    hg : Hypergraph
        The hypergraph of interest.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure is created.
    **kwargs
        Additional keyword arguments to pass to the plot function.

    Returns
    -------
    matplotlib.axes.Axes
        The axes on which the plot was made.
    """
    if ax is None:
        fig, ax = plt.subplots()

    for size in range(2, max_size+1):
        degrees = list(hg.degree_sequence(size=size).values())
        degrees = [d for d in degrees if d > 0]
        degrees = distr_bin(degrees, n_bin=20, logbin=True)[1]
        print(size)
        print(degrees)
        #sn.scatterplot(x=degrees.keys(), y=degrees.values(), label="Size: {}".format(size), ax=ax, **kwargs)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    sn.despine()
    return ax

