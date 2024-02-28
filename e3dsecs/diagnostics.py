#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 09:56:02 2023

@author: jone

File that contain functions for producing various diagnostic plots

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gemini3d.grid.convert import geomag2geog, geog2geomag
from secsy import cubedsphere
import gemini3d.read as read
from gemini3d.grid.gridmodeldata import geog2dipole
import scipy
import matplotlib

try:
    from . import visualization
    from . import coordinates
except:
    import visualization
    import coordinates    
RE = 6371.2 # Earth radius in km


def compare_input_jperp(dat, jjj2, inside, savesuff, secs_grid, alts_grid, 
                        dim=2, sliceindex=None, gif=False, maph=None, param='jperpphi', 
                        pdf=False, geomag=False):
    # Plot maps to compare where the estimated current differs from the GEMINI one
    # savesuff string indicated the method used for making the input jperp dataset,
    # to be compared with jperp directly from GEMINI
    # jjj2 is a (3,N) 2D array of jperp current density expressed using Lompe (r, theta, phi)
    
    # if sum(use) != use.size:
    #     print('Can not reconstruct the data cube for visualization')
    #     print(1/0)
    
    # Input parameters
    # dat : instance of the data class
    # jjj2 : current density estimated
    # param = 'jperpphi' # the quantity to plot
    # dim = 2 # which dimension the slice will be from
    if geomag:
        j_perp = np.hstack((dat.jperpu_gm,-dat.jperpn_gm,dat.jperpe_gm))
    else:
        j_perp = np.hstack((dat.jperpu,-dat.jperpn,dat.jperpe))
    N = dat.lat.size
    if sliceindex is None:
        slices = dat.shape[dim]
    else:
        slices = [sliceindex]
    clim = 20e-6
    if param == 'jperpr':
        txt = '$\\mathbf{j}_{\perp,r}$'
        ppp1 = j_perp[0:N].reshape(dat.shape)
        ppp1[~inside.reshape(dat.shape)] = np.nan
        _ppp2 = jjj2[0,:]
        ppp2 = np.zeros(dat.shape)
        ppp2[~inside.reshape(dat.shape)] = np.nan
        ppp2[inside.reshape(dat.shape)] = _ppp2
    elif param == 'jperptheta':
        txt = '$\\mathbf{j}_{\perp,\\theta}$'
        ppp1 = j_perp[N:2*N].reshape(dat.shape)
        ppp1[~inside.reshape(dat.shape)] = np.nan
        _ppp2 = jjj2[1,:]
        ppp2 = np.zeros(dat.shape)
        ppp2[~inside.reshape(dat.shape)] = np.nan
        ppp2[inside.reshape(dat.shape)] = _ppp2
    elif param == 'jperpphi':
        txt = '$\\mathbf{j}_{\perp,\phi}$'
        ppp1 = j_perp[2*N:3*N].reshape(dat.shape)
        ppp1[~inside.reshape(dat.shape)] = np.nan
        _ppp2 = jjj2[2,:]
        ppp2 = np.zeros(dat.shape)
        ppp2[~inside.reshape(dat.shape)] = np.nan
        ppp2[inside.reshape(dat.shape)] = _ppp2
        
        # diffs = 100 * (ppp1[0:8,:,sliceindex] - ppp2[0:8,:,sliceindex])/(ppp1[0:8,:,sliceindex])
        # plt.hist(diffs.flatten(), range=(-100,100), bins=40)
    
    for sliceindex in slices:
        fig = plt.figure(figsize = (30, 10))
        #GEMINI part to the left
        ax = fig.add_subplot(131, projection='3d')
        visualization.plot_slice(ax, secs_grid, alts_grid, dat.lat.reshape(dat.shape), 
                                 dat.lon.reshape(dat.shape), 
                                 dat.alt.reshape(dat.shape), 
                                 ppp1, dim = dim, sliceindex = sliceindex, maph=maph, 
                                 parameter=txt + ' from GEMINI', clim=clim, dipole_lompe=geomag,
                                 coastlines=False)
        #vi-ve part in middle
        ax = fig.add_subplot(132, projection='3d')
        visualization.plot_slice(ax, secs_grid, alts_grid, dat.lat.reshape(dat.shape), 
                                 dat.lon.reshape(dat.shape), 
                                 dat.alt.reshape(dat.shape), 
                                 ppp2, dim = dim, sliceindex = sliceindex, maph=maph, 
                                 parameter=txt + ' estimated with '+savesuff[1:-1], 
                                 clim=clim, dipole_lompe=geomag, coastlines=False)            
        #difference in right panel
        ax = fig.add_subplot(133, projection='3d')
        visualization.plot_slice(ax, secs_grid, alts_grid, dat.lat.reshape(dat.shape), 
                                 dat.lon.reshape(dat.shape), 
                                 dat.alt.reshape(dat.shape), 
                                 ppp1-ppp2, dim = dim, sliceindex = sliceindex, maph=maph,
                                 parameter='GEMINI - estimate', clim=clim, dipole_lompe=geomag,
                                 coastlines=False)
        #Colorbar
        cbarax = plt.axes((0.37,0.2,0.3,0.02))
        # cbarax = plt.subplot2grid((1,3), (0, 1), rowspan = 1, colspan = 1)
        cmap = plt.cm.bwr
        norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
        cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
        cb1.ax.tick_params(labelsize=16)
        cb1.ax.xaxis.get_offset_text().set_fontsize(16)
        cb1.set_label('[A/m$^2$]', fontsize=16)
        # if pdf:
        #     savename = './comparison_'+param+'_dim='+str(dim)+savesuff+'%03i.pdf' % sliceindex
        # else:
        #     savename = './plots/uniform_sampling_slices/comparison_'+param+'_dim='+str(dim)+savesuff+'%03i.png' % sliceindex
        # fig.savefig(savename, dpi=250)
    if gif:
        import glob
        files = glob.glob('./plots/uniform_sampling_slices/comparison_'+param+'_dim='+str(dim)+savesuff+'*.png')
        files.sort()
        gifname = './plots/uniform_sampling_slices/comparison_'+param+'_dim='+str(dim)+savesuff[:-1]+'.gif'  
        visualization.make_gif(files, filename=gifname, delete=True, duration = 1)
    
    return fig

def compare_potentials(dat, grid, alts_grid, datadict, inside, model, dipole_lompe=True):
    # Plot GEMINI and Lompe potentials on top of each other
    # GEMINI potential
    #First define a fine evaluation grid
    alts__ = np.arange(datadict['maph'],datadict['maph']+1,1)
    # Sampling at separate CS grids at each height
    xi_edge = np.linspace(grid.xi_mesh[0,0]-2*grid.dxi,grid.xi_mesh[0,-1]+2*grid.dxi,40)
    xis = xi_edge[0:-1] + np.diff(xi_edge)/2 # center locations
    eta_edge = np.linspace(grid.eta_mesh[0,0]-2*grid.deta,grid.eta_mesh[-1,0]+2*grid.deta,40)
    etas = eta_edge[0:-1] + np.diff(eta_edge)/2 # center locations
    alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, etas, xis, indexing='ij')
    sh = alt_ev.shape
    lon_ev = np.zeros(sh)
    lat_ev = np.zeros(sh)
    le_ev = np.zeros((sh[0],sh[1]*sh[2],sh[1]*sh[2]))
    ln_ev = np.zeros((sh[0],sh[1]*sh[2],sh[1]*sh[2]))
    for i,_a in enumerate(alts__):
        _grid = cubedsphere.CSgrid(grid.projection, 0, 0, 0, 0, R = (RE+_a)*1e3, edges = (xi_edge,eta_edge))
        _lon_ev, _lat_ev = _grid.projection.cube2geo(xi_ev[i,:,:], eta_ev[i,:,:])
        lon_ev[i,:,:] = _lon_ev
        lat_ev[i,:,:] = _lat_ev
        le, ln = _grid.get_Le_Ln()
        le_ev[i,:,:] = le
        ln_ev[i,:,:] = ln
    
    k0 = secs3d.get_alt_index(alts_grid, alt_ev, returnfloat=True).flatten()
    inside =  grid.ingrid(lon_ev.flatten(), lat_ev.flatten(), ext_factor = -1) & \
                (k0 >= 0) & (k0<alts_grid.size-1)
    if dipole_lompe: #convert back to geographic coordinates
        lon_ev, lat_ev = geomag2geog(np.radians(lon_ev), np.pi/2 - np.radians(lat_ev))
    path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/issi_team/gemini_output/"
    var = ["v1", "v2", "v3", "Phi", "J1", "J2", "J3", "ne"]
    xg = read.grid(path)
    # cfg = read.config(path)
    # timeindex = -1 # This should not be hard codeded, so check this value
    # times = cfg["time"][timeindex:]
    # t = times[0]
    # dat = read.frame(path, t, var=var)
    lx2 = xg["lx"][1]
    lx3 = xg["lx"][2]
    # inds1 = range(2, lx1 + 2)
    inds2 = range(2, lx2 + 2)
    inds3 = range(2, lx3 + 2)
    # x1 = xg["x1"][inds1]
    x2 = xg["x2"][inds2]
    x3 = xg["x3"][inds3]
    x1i, x2i, x3i = geog2dipole(alt_ev*1e3, lon_ev, lat_ev)
    xi = np.array((x2i.ravel(), x3i.ravel())).transpose()
    phitop = scipy.interpolate.interpn(
        points=(x2, x3),
        values=dat.Phitop.values,
        xi=xi,
        method="linear",
        bounds_error=False,
        fill_value=np.NaN).reshape(sh)
    gem_pot = phitop[0,:,:]
    gem_pot[~inside.reshape(sh)[0,:,:]] = np.nan
    gem_lon = lon_ev[0,:,:]
    gem_lat = lat_ev[0,:,:]

    
    # Need to set correct altitude index for a meaningful comparison
    # lll = 38
    # gem_pot = datadict['Phitop'].reshape(datadict['shape'])[lll,:,:]
    # gem_pot[~inside.reshape(datadict['shape'])[lll,:,:]] = np.nan
    # gem_lon = datadict['lon'].reshape(datadict['shape'])[lll,:,:]
    # gem_lat = datadict['lat'].reshape(datadict['shape'])[lll,:,:]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    dv = 2
    ax.contour(gem_lon, gem_lat, gem_pot, np.arange(np.round(np.nanmin(gem_pot)/1000),np.round(np.nanmax(gem_pot)/1000),dv)*1000+600, label='GEMINI', colors='black')
    # Lompe potential
    lompe_phi, lompe_theta = geog2geomag(gem_lon, gem_lat)
    lompe_pot = model.E_pot(lon=np.degrees(lompe_phi), lat=90-np.degrees(lompe_theta)).reshape(lompe_phi.shape)
    lompe_pot[~inside.reshape(sh)[0,:,:]] = np.nan
    ax.contour(gem_lon, gem_lat, lompe_pot, np.arange(np.round(np.nanmin(lompe_pot)/1000),np.round(np.nanmax(lompe_pot)/1000),dv)*1000, label='Lompe', colors='red')
    ax.text(20,68.2, 'GEMINI', color='black', fontsize=18)
    ax.text(20,68, 'Lompe', color='Red', fontsize=18)
    ax.set_xlabel('glon')
    ax.set_ylabel('glat')
    ax.set_title('Electric potential', fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(19.5,29)
    ax.set_ylim(65,68.5)
    fig.savefig('./plots/uniform_sampling_slices/potential-comparison.png', dpi=250)
    
    
def plot_analysis_grid(datadict, grid, alts_grid, lat_ev, lon_ev, alt_ev, dipole_lompe=False, 
                       data=True, eiscat=True, _d=400, boxgrid=False, q='fac', clim=2e-5, 
                       cmap='bwr', diverging=True, ax=None, maxh=None):
    '''
    Make a plot of the inversion grid and data selection grid, and possibly also some data
    to aid the tuning of the grids to be used
    
    Parameters
    ----------
    datadict : dictionary
        The GEMINI output in ENU components (in 3D).    
    grid : CS grid object
        The grid we use to compute gridded indices of data/evaluation locations
        Dimension is (I,J). Assumes secs_grid.A refer to bottom layer.
    alts_grid : array-like, 1D or 2D
        Altitude grid to use together with secs_grid. Length is K. Values 
        represent the height in km of the centre of the voxels of each layer. 
        Should be in increasing order.    
    lat_ev : array-like (K,I,J) shape
        latitude array of 3D data cube, in degrees
    lon_ev : array-like (K,I,J) shape
        latitude array of 3D data cube, in degrees
    alt_ev : array-like (K,I,J) shape
        altitude array of 3D data cube, in km.
    dipole_lompe : boolean, optional
        Whether Lompe is used in dipole mode. The default is True.
    data : boolean, optional
        Wether to plt a surface of input data. Does not work with EISCAT
        beam sampling, must be a regular grid
    eiscat : boolean, optional
        If True, do scatterplot instead of surface plot
    _d : boolean, optional
        number of measurements along each beam (for EISCAT FOV plotting)
    boxgrid : boolean, optional
        If plotting the 3D box of where to evaluate 3D model
    q : str
        The quantity to plot, must be a valid key in datadict, such as je, jn, ju, fac ...
    clim : float
        the bounds of the colorscale. Default is 2e-5. Implementation depends on
        value of diverging.
    cmap : str
        The colormap to use. Default us bwr.
    diverging : boolean
        Whether to use a diverging colormap. Default is true-
    maxh : bool
        The maximum altitude to plot (in the E3D scatterplot case). Default is None
        leading to a scatterplor of all data. Alternatively an int or float could be
        passed, representing the maximum altitude in km to show in the plot.
    

    Returns
    -------
    matplotlib axis object.

    '''
    
    import matplotlib
    if not eiscat:
        sh = datadict['shape']
    
    ###################
    # Before proceeding to inversion, check that the analysis grid looks reasonable
    if ax == None:
        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    ax.view_init(azim=-26, elev=7)
    if boxgrid:
        visualization.spherical_grid(ax, lat_ev, lon_ev, alt_ev, color='blue', maph=datadict['maph'])
    visualization.field_aligned_grid(ax, grid, alts_grid, color='green', dipoleB=dipole_lompe, coastlines=False)
    kwargs={'linewidth':3}
    # cmap = plt.cm.bwr
    # clim = 2e-5
    if diverging:
        norm = matplotlib.colors.Normalize(vmin=-clim, vmax=clim)
    else:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=clim)
    # for kk in range(lat_ev[0,-1,:].size):
    #     visualization.plot_field_line(ax, lat_ev[0,-1,kk], lon_ev[0,-1,kk], 
    #                               alts__, color='orange', **kwargs, dipole=True)
    if data:
        x, y, z = coordinates.sph_to_car((RE+datadict['alt'].flatten(), 90-datadict['lat'].flatten(), 
                              datadict['lon'].flatten()), deg=True)

        if eiscat:
            if maxh is None: 
                maxh = alts_grid[-1]
            keep = datadict['alt'] <= maxh
            x = x[keep]
            y = y[keep]
            z = z[keep]
            c = datadict[q][keep]
            ax.scatter(x, y, z, c = c, cmap=cmap, norm=norm)
        else:
            
            p = ax.plot_surface(x.reshape(sh)[-1,:,:], y.reshape(sh)[-1,:,:], 
                                z.reshape(sh)[-1,:,:], alpha=0.5,
                                facecolors=cmap(norm(datadict[q].reshape(sh)[-1,:,:])), 
                                rcount = np.max(sh[1:]), ccount = np.max(sh[1:]), cmap=cmap)
            p = ax.plot_surface(x.reshape(sh)[:,:,sh[2]//2], y.reshape(sh)[:,:,sh[2]//2], 
                                z.reshape(sh)[:,:,sh[2]//2], alpha=0.5,
                                facecolors=cmap(norm(datadict[q].reshape(sh)[:,:,sh[2]//2])), 
                                rcount = np.max(sh), ccount = np.max(sh),cmap=cmap)
                                # facecolors=cmap(norm(np.ones(sh)[:,:,sh[2]//2]*1e-6)),cmap=cmap)
    if dipole_lompe:
        lon_, lat_ = geomag2geog(np.radians(grid.projection.position[0]), np.pi/2 - np.radians(grid.projection.position[1])) # returns in degrees
        x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-lat_, lon_), deg=True)
    else:
        x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-grid.projection.position[1], grid.projection.position[0]), deg=True)   
    # if eiscatfov:
    #     lat0 = datadict['lat'][::_d][12:-3]
    #     lon0 = datadict['lon'][::_d][12:-3]
    #     alt0 = datadict['alt'][::_d][12:-3]
    #     lat1 = datadict['lat'][_d-1:][::_d][12:-3]
    #     lon1 = datadict['lon'][_d-1:][::_d][12:-3]
    #     alt1 = datadict['alt'][_d-1:][::_d][12:-3]
    #     visualization.plot_e3dfov(ax, lat0, lon0, alt0, lat1, lon1, alt1, **kwargs)
    
    # Fix viewing geometry
    lat_ = grid.projection.position[1] # in degrees
    lon_ = grid.projection.position[0] # in degrees
    L = grid.L*1e-3
    Lres = grid.Lres*1e-3
    x_, y_, z_ = coordinates.sph_to_car((RE, 90-lat_, lon_), deg=True)
    xlim = (x_[0]-L+4*Lres, x_[0]+L-3*Lres) 
    ylim = (y_[0]-L+6*Lres, y_[0]+L-3*Lres) 
    zlim = (z_[0]+ 0.15*alts_grid[-1], z_[0]+ 0.75*alts_grid[-1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)             
    # range_ =  alts_grid[-1]*0.3    
    # ax.set_xlim(x0-range_, x0+range_)
    # ax.set_ylim(y0-0.5*range_, y0+range_)
    # ax.set_zlim(z0+0.5*range_, z0+2.5*range_)
    ax.set_title(q + ' from GEMINI')
    #########################
    
    return ax

def resolution_analysis(GTG, GT, Reg, grid, alts_grid, CF=True, k=42, i=2, j=6, clim=5e-2):
    '''
    Compute resolution matrix and do some plotting

    Parameters
    ----------
    GTG : 2D array
        GTG matrix.
    GT : 2D array
        Transpose of G matrix.
    Reg : 2D array
        The 0th order Tikhonov regularization matrix.
    grid : CS grid object
        The base grid.
    alts_grid : 1D array
        Altitude grid.
    CF : boolean, optional
        True means that the CF parameters are investigated. The default is True.
    k : int, optional
        altitude index of parameter to investigate. The defaukt is 42.
    i : int, optional
        i index of parameter to investigate. The defaukt is 2.
    j : int, optional
        j index of parameter to investigate. The defaukt is 6.        
    clim : float, optional
        limit colorbar

    Returns
    -------
    None.

    '''
    #Grid dimensions
    K = alts_grid.shape[0] #Number of vertival layers
    I = grid.shape[0] #Number of cells in eta direction
    J = grid.shape[1]  #Number of cells in xi direction 
    
    # Make resolution matrix R
    Cd  = np.diag(np.ones(GT.shape[1])) #Data covariance matrix
    Gt = np.linalg.pinv(GTG+Reg).dot(GT)
    R = Gt.dot(GT.T)

    # Investigate resolution matrix
    position = grid.projection.position
    # k, i, j = secs3d.get_indices_kij(grid, alts_grid, position[1], position[0], np.array([500]))
    # kij = np.ravel_multi_index((k[0],i[0],j[0]), (K,I,J))
    kij = np.ravel_multi_index((k,i,j), (K,I,J))
    if CF:
        NN=0 # 0 if CF parameters, 1 if DF parameters
    else:
        NN=1
    psf = R[:,NN*K*I*J+kij] #psf
    # psf = R[kij,:] # averaging function
    absolute=True
    
    fig = plt.figure(figsize = (30, 10))
    ax = fig.add_subplot(131, projection='3d')
    ax.set_axis_off()
    visualization.plot_resolution(ax, grid, alts_grid, kij, psf[NN*K*I*J:(NN+1)*K*I*J], clim=clim, 
                                 planes=[0], absolute=absolute)
    ax = fig.add_subplot(132, projection='3d')
    ax.set_axis_off()
    visualization.plot_resolution(ax, grid, alts_grid, kij, psf[NN*K*I*J:(NN+1)*K*I*J], clim=clim, 
                                 planes=[1], az=40, el=5, absolute=absolute)
    ax = fig.add_subplot(133, projection='3d')
    ax.set_axis_off()
    visualization.plot_resolution(ax, grid, alts_grid, kij, psf[NN*K*I*J:(NN+1)*K*I*J], clim=clim, 
                                 planes=[2], absolute=absolute)    


def model_amplitude_analysis(grid, alts_grid, m, clim=1e-1, k=47, i = 5, j = 5, dipoleB=True):
    '''
    Compute resolution matrix and do some plotting

    Parameters
    ----------
    grid : CS grid object
        The base grid.
    alts_grid : 1D array
        Altitude grid.
    m : 1D array
        model aplitude vector
    clim : float, optional
        limit colorbar
    k : int
        index along k dimension
    i : int
        index along i dimension        
    j : int
        index along j dimension        

    Returns
    -------
    None.

    '''
    #Grid dimensions
    K = alts_grid.shape[0] #Number of vertival layers
    I = grid.shape[0] #Number of cells in eta direction
    J = grid.shape[1]  #Number of cells in xi direction 
    

    # Investigate resolution matrix
    position = grid.projection.position
    # k, i, j = secs3d.get_indices_kij(grid, alts_grid, position[1], position[0], np.array([500]))
    # kij = np.ravel_multi_index((k[0],i[0],j[0]), (K,I,J))
    kij = np.ravel_multi_index((k,i,j), (K,I,J))      
    
    fig = plt.figure(figsize = (30, 10))
    ax = fig.add_subplot(131, projection='3d')
    ax.set_axis_off()
    visualization.plot_resolution(ax, grid, alts_grid, kij, m, clim=clim, 
                                 planes=[0], az=40, el=70, absolute=False, dipoleB=dipoleB)
    ax = fig.add_subplot(132, projection='3d')
    ax.set_axis_off()
    visualization.plot_resolution(ax, grid, alts_grid, kij, m, clim=clim, 
                                 planes=[1], az=40, el=5, absolute=False, dipoleB=dipoleB)
    ax = fig.add_subplot(133, projection='3d')
    ax.set_axis_off()
    visualization.plot_resolution(ax, grid, alts_grid, kij, m, clim=clim, 
                                 planes=[2], absolute=False, dipoleB=dipoleB)       
    

def reconsrtruction_performance(dat, grid, alts_grid, lat_ev, lon_ev, alt_ev, full_j, jpar, dipolekw=True, 
                                cut='k', single=None, gif=True, inputmode='vi', clim = 2e-5):
   
    import matplotlib
    N = jpar.size
    
    datadict = dat.__dict__
    
    datadict['jtheta'] = -datadict['jn']
    
    #Make gifs of performance in different slices
    ##################################3
    # Performance plotting
    # jjj = 1 # which cut to show
    shape = lat_ev.shape
    if type(single) == int:
        TTT = np.arange(single,single+1,1)
    elif cut=='k':
        TTT = np.arange(0,shape[0],1)
    elif cut=='i':
        TTT = np.arange(0,shape[1],1)
    else:
        TTT = np.arange(0,shape[2],1)
        
    for ttt in TTT:
        cmap = plt.cm.bwr
        norm = matplotlib.colors.Normalize(vmin=-clim, vmax=clim)
        fig = plt.figure(figsize=(15,10))
        ax3 = plt.subplot2grid((20, 32), (0, 0), rowspan = 10, colspan = 10, projection='3d')
        ax2 = plt.subplot2grid((20, 32), (0, 10), rowspan = 10, colspan = 10, projection='3d')
        ax1 = plt.subplot2grid((20, 32), (0, 20), rowspan = 10, colspan = 10, projection='3d')
        axs = [ax1,ax2,ax3]
        plotparams = ['je', 'jtheta','ju']
        plot_titles = ['$\\mathbf{j}_{r}$', '$\\mathbf{j}_{\\theta}$', '$\\mathbf{j}_{\phi}$']
        plot_titles.reverse()
        kwargs={'linewidth':1}
        
        #GEMINI on top
        for pp, ax in enumerate(axs):
            ax.set_axis_off()
            ax.view_init(azim=-26, elev=7)
            # visualization.spherical_grid(ax, lat_ev, lon_ev, alt_ev, color='blue')
            visualization.field_aligned_grid(ax, grid, alts_grid, color='green', dipoleB=dipolekw,
                                             coastlines=False,**kwargs)
            for kk in range(lat_ev[0,-1,:].size):
                visualization.plot_field_line(ax, lat_ev[0,-1,kk], lon_ev[0,-1,kk], 
                                          alts_grid, color='orange', **kwargs, dipoleB=dipolekw)
            x, y, z = coordinates.sph_to_car((RE+alt_ev.flatten(), 90-lat_ev.flatten(), 
                                  lon_ev.flatten()), deg=True)
            if cut=='k':
                p = ax.plot_surface(x.reshape(shape)[ttt,:,:], y.reshape(shape)[ttt,:,:], 
                                    z.reshape(shape)[ttt,:,:], alpha=0.5,
                                    facecolors=cmap(norm(datadict[plotparams[pp]].reshape(shape)[ttt,:,:])), 
                                    rcount = np.max(shape), ccount = np.max(shape),cmap=cmap)
            elif cut=='i':
                p = ax.plot_surface(x.reshape(shape)[:,ttt,:], y.reshape(shape)[:,ttt,:], 
                                    z.reshape(shape)[:,ttt,:], alpha=0.5,
                                    facecolors=cmap(norm(datadict[plotparams[pp]].reshape(shape)[:,ttt,:])), 
                                    rcount = np.max(shape[0:]), ccount = np.max(shape[0:]), cmap=cmap)
            else:
                p = ax.plot_surface(x.reshape(shape)[:,:,ttt], y.reshape(shape)[:,:,ttt], 
                                    z.reshape(shape)[:,:,ttt], alpha=0.5,
                                    facecolors=cmap(norm(datadict[plotparams[pp]].reshape(shape)[:,:,ttt])), 
                                    rcount = np.max(shape), ccount = np.max(shape),cmap=cmap)
            if dipolekw:
                lon_, lat_ = geomag2geog(np.radians(grid.projection.position[0]), np.pi/2 - np.radians(grid.projection.position[1])) # returns in degrees
                x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-lat_, lon_), deg=True)
            else:
                x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-grid.projection.position[1], grid.projection.position[0]), deg=True)
            # Fix viewing geometry
            lat_ = grid.projection.position[1] # in degrees
            lon_ = grid.projection.position[0] # in degrees
            L = grid.L*1e-3
            Lres = grid.Lres*1e-3
            x_, y_, z_ = coordinates.sph_to_car((RE, 90-lat_, lon_), deg=True)
            xlim = (x_[0]-L+5*Lres, x_[0]+L-3*Lres) 
            ylim = (y_[0]-L+6*Lres, y_[0]+L-3*Lres) 
            zlim = (z_[0]+ 0.15*alts_grid[-1], z_[0]+ 0.75*alts_grid[-1])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)             
            
            # range_ =  alts_grid[-1]*0.3
            # ax.set_xlim(x0-range_, x0+range_)
            # ax.set_ylim(y0-range_, y0+range_)
            # ax.set_zlim(z0, z0+2*range_)
            
            ax.set_title(plot_titles[pp], fontsize=16)
            
            if pp==2:
                x_, y_, z_ = coordinates.sph_to_car((RE+0, 90-74, 37), deg=True)
                ax.text(x_[0], y_[0], z_[0], 'GEMINI', fontsize=16)
        
        # Lower row is the reconstruction
        ax6 = plt.subplot2grid((20, 32), (10, 0), rowspan = 10, colspan = 10, projection='3d')   
        ax5 = plt.subplot2grid((20, 32), (10, 10), rowspan = 10, colspan = 10, projection='3d')   
        ax4 = plt.subplot2grid((20, 32), (10, 20), rowspan = 10, colspan = 10, projection='3d')   
        axs =[ax4,ax5,ax6]
        for pp, ax in enumerate(axs):
            ax.set_axis_off()
            ax.view_init(azim=-26, elev=7)
            # visualization.spherical_grid(ax, lat_ev, lon_ev, alt_ev, color='blue') #Data locations
            visualization.field_aligned_grid(ax, grid, alts_grid, color='green', dipoleB=dipolekw, 
                                             coastlines=False, **kwargs) # Base SECS grid
            # for kk in range(lat.reshape(shape)[0,-1,:].size): # Plot some field-lines
            #     visualization.plot_field_line(ax2, lat_ev.reshape(shape)[0,-1,kk], 
            #                 lon_ev.reshape(shape)[0,-1,kk], alt_ev.reshape(shape)[:,-1,0], 
            #                 color='orange', **kwargs, dipole=True)
                # visualization.plot_field_line(ax, lat_ev[0,8,kk], lon_ev[0,8,kk], 
                                          # alts__)
            x, y, z = coordinates.sph_to_car((RE+alt_ev.flatten(), 90-lat_ev.flatten(), 
                                  lon_ev.flatten()), deg=True)
            # p = ax2.plot_surface(x.reshape(shape)[kkk,:,:], y.reshape(shape)[kkk,:,:], 
            #                     z.reshape(shape)[kkk,:,:], alpha=0.5,
            #                     facecolors=cmap(norm((-full_j[N:2*N]).reshape(shape)[kkk,:,:])), cmap=cmap)
            # p = ax2.plot_surface(x.reshape(shape)[:,iii,:], y.reshape(shape)[:,iii,:], 
            #                     z.reshape(shape)[:,iii,:], alpha=0.5,
            #                     facecolors=cmap(norm((jpar).reshape(shape)[:,iii,:])), cmap=cmap)
            if plotparams[pp] =='fac':
                ppp = jpar
            elif plotparams[pp] == 'je':
                ppp = full_j[2*N:3*N]
            elif plotparams[pp] == 'jtheta':
                ppp = full_j[1*N:2*N]
            elif plotparams[pp] == 'ju':
                ppp = full_j[0*N:1*N]
            
            if cut=='k':              
                p = ax.plot_surface(x.reshape(shape)[ttt,:,:], y.reshape(shape)[ttt,:,:], 
                                    z.reshape(shape)[ttt,:,:], alpha=0.5,
                                    facecolors=cmap(norm((ppp).reshape(shape)[ttt,:,:])), cmap=cmap)
            elif cut=='i':
                p = ax.plot_surface(x.reshape(shape)[:,ttt,:], y.reshape(shape)[:,ttt,:], 
                                    z.reshape(shape)[:,ttt,:], alpha=0.5,
                                    facecolors=cmap(norm((ppp).reshape(shape)[:,ttt,:])), cmap=cmap)        
            else:
                p = ax.plot_surface(x.reshape(shape)[:,:,ttt], y.reshape(shape)[:,:,ttt], 
                                    z.reshape(shape)[:,:,ttt], alpha=0.5,
                                    facecolors=cmap(norm((ppp).reshape(shape)[:,:,ttt])), cmap=cmap)
            if dipolekw:
                lon_, lat_ = geomag2geog(np.radians(grid.projection.position[0]), np.pi/2 - np.radians(grid.projection.position[1])) # returns in degrees
                x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-lat_, lon_), deg=True)
            else:
                x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-grid.projection.position[1], grid.projection.position[0]), deg=True)

            # Fix viewing geometry
            lat_ = grid.projection.position[1] # in degrees
            lon_ = grid.projection.position[0] # in degrees
            L = grid.L*1e-3
            Lres = grid.Lres*1e-3
            x_, y_, z_ = coordinates.sph_to_car((RE, 90-lat_, lon_), deg=True)
            xlim = (x_[0]-L+5*Lres, x_[0]+L-3*Lres) 
            ylim = (y_[0]-L+6*Lres, y_[0]+L-3*Lres) 
            zlim = (z_[0]+ 0.15*alts_grid[-1], z_[0]+ 0.75*alts_grid[-1])
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)             

            # range_ =  alts_grid[-1]*0.3
            # ax.set_xlim(x0-range_, x0+range_)
            # ax.set_ylim(y0-range_, y0+range_)
            # ax.set_zlim(z0, z0+2*range_)
            ax.set_title(plot_titles[pp], fontsize=16)
            
            if pp==2:
                x_, y_, z_ = coordinates.sph_to_car((RE+0, 90-74, 37), deg=True)
                ax.text(x_[0], y_[0], z_[0], '3D reconstruction', fontsize=16)        
        # Colorbar
        cbarax = plt.subplot2grid((20,32), (5, 31), rowspan = 10, colspan = 1)
        cmap = plt.cm.bwr
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
        cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
        cb1.set_label('[A/m$^2$]', fontsize=16)
        cb1.ax.yaxis.set_label_coords(-1.3, 0.5)
            
        sss = ''
        if inputmode == 'vi':
            sss = sss + '_vi-ve_lompe'
        elif inputmode == 'vi_ohmslaw':
            sss = sss + '_vi-ve_lompe_ohmslaw'
        else:
            sss = sss + '_' + inputmode
        
        if cut=='k':
            savename = './plots/uniform_sampling_slices/gemini_comparison_alt-slice'+sss+'_%03i.png' % ttt     
        elif cut=='i':
            savename = './plots/uniform_sampling_slices/gemini_comparison_ew-slice'+sss+'_%03i.png' % ttt 
        else:
            savename = './plots/uniform_sampling_slices/gemini_comparison_ns-slice'+sss+'_%03i.pdf' % ttt 
        fig.savefig(savename, dpi=250,bbox_inches='tight')
    
    if single is None and gif:
        import glob
        if cut=='k':
            files = glob.glob('./plots/uniform_sampling_slices/gemini_comparison_alt-slice'+sss+'*.png')
            gifname = './plots/uniform_sampling_slices/gemini_comparison_alt-slice'+sss+'.gif'  
        elif cut=='i':
            files = glob.glob('./plots/uniform_sampling_slices/gemini_comparison_ew-slice'+sss+'*.png')
            gifname = './plots/uniform_sampling_slices/gemini_comparison_ew-slice'+sss+'.gif'  
        else:
            files = glob.glob('./plots/uniform_sampling_slices/gemini_comparison_ns-slice'+sss+'*.png')
            gifname = './plots/uniform_sampling_slices/gemini_comparison_ns-slice'+sss+'.gif'  
        files.sort()
        visualization.make_gif(files, filename=gifname, delete=True, duration = 1)

    return fig
    


def scatterplot_reconstruction(grid, alts_grid, dat, lon, lat, alt, full_j, jpar, 
                               dipolekw=False, inout=False):
    '''
    

    Parameters
    ----------
    grid : CS grid object
        The CS grid we used in reconstruction of j. Dimension is (I,J). 
    alts_grid : array-like, 1D
        Altitude grid to use together with grid. Length is K. Values 
        represent the height in km of the centre of the voxels of each layer. 
        Should be in increasing order.    
    dat : instance of data class
        The GEMINI output in ENU components (in 3D).
    lon : array-like 1D, 
        longitude of evaluation locations, in degrees
    lat : array-like 1D, 
        latitude of evaluation locations, in degrees
    alt : array-like 1D, 
        altitude of evaluation locations, in km
    dipolekw : boolean, optional
        Whether Lompe is used in dipole mode. The default is True.    
    full_j : array-like 1D
        Reconsrtucted 3D current density, shape 1x3N.
    jpar : array-like 1D
        Reconstructed field aligned current
    dipolekw : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    '''

    N = lon.size
    # Scatterplots
    #Residual scatterplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    from gemini3d.grid import convert
    if dipolekw:
        mlon_ = np.radians(lon)
        mtheta_ = np.radians(90-lat) 
        j_gg_enu = np.vstack((dat.je, dat.jn, dat.ju)).T
        j_gm_enu = gemini_tools.enugg2enugm(j_gg_enu, dat.lon, dat.lat)
    else:
        mlon_, mtheta_ = convert.geog2geomag(lon,lat)
    m_theta = np.arcsin(np.sqrt((RE+alts_grid[0])/(RE+alt))*np.sin(mtheta_)) #sjekk - ok!
    m_mlon = mlon_
    if dipolekw:
        ax.scatter(1e6*j_gm_enu[:,2],1e6*full_j[0:N], s=1, label='$j_r$', color='C0')
        inside = grid.ingrid(np.degrees(m_mlon), 90-np.degrees(m_theta), ext_factor=-2)
        ax.scatter(-1e6*j_gm_enu[:,1],1e6*full_j[N:2*N], s=1, label='$j_\\theta$', color='C1')
        ax.scatter(1e6*j_gm_enu[:,0],1e6*full_j[2*N:3*N], s=1, label='$j_\phi$', color='C2')
    else:    
        m_glon, m_glat = convert.geomag2geog(m_mlon, m_theta)
        inside = grid.ingrid(m_glon, m_glat, ext_factor=-2)
        print(type(dat))
        ax.scatter(1e6*dat.ju,1e6*full_j[0:N], s=1, label='$j_r$', color='C0')
        ax.scatter(-1e6*dat.jn,1e6*full_j[N:2*N], s=1, label='$j_\\theta$', color='C1')
        ax.scatter(1e6*dat.je,1e6*full_j[2*N:3*N], s=1, label='$j_\phi$', color='C2')
        
    highalt = alt > 200
    ax.scatter(1e6*dat.fac[highalt],1e6*jpar[highalt], s=1, label='FAC>200km', color='C3')
    if inout:
        ax.scatter(1e6*dat.fac[inside],1e6*jpar[inside], s=1, label='FAC_inside', color='C4')
        ax.scatter(1e6*dat.fac[~inside],1e6*jpar[~inside], s=1, label='FAC_outside', color='C5')
        r_outside = np.corrcoef(1e6*dat.fac[~inside],1e6*jpar[~inside])[0,1]
        r_inside = np.corrcoef(1e6*dat.fac[inside],1e6*jpar[inside])[0,1]
        ax.text(0.1,0.3, 'correlation inside: %4.2f' % r_inside, color='C3', transform = ax.transAxes)
        ax.text(0.1,0.2, 'correlation outside: %4.2f' % r_outside, color='C4', transform = ax.transAxes)
    lgnd = ax.legend(frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    # lgnd.legendHandles[0]._legmarker.set_markersize(6)
    # lgnd.legendHandles[1]._legmarker.set_markersize(6)
    ax.set_xlabel('GEMINI $[\mu A/m^2]$')
    ax.set_ylabel('3D SECS $[\mu A/m^2]$')
    ax.set_ylim(-140,70)
    ax.set_xlim(-140,70)
    
    RMSE_e = np.sqrt(np.mean((dat.je*1e6 - 1e6*full_j[2*N:3*N])**2))
    corr_e = np.corrcoef(dat.je, full_j[2*N:3*N])[0,1]
    RMSE_n = np.sqrt(np.mean((dat.jn*1e6 + 1e6*full_j[1*N:2*N])**2))
    corr_n = np.corrcoef(dat.jn, -full_j[1*N:2*N])[0,1]
    RMSE_u = np.sqrt(np.mean((dat.ju*1e6 - 1e6*full_j[0*N:1*N])**2))
    corr_u = np.corrcoef(dat.ju, full_j[0*N:1*N])[0,1]
    RMSE_fac = np.sqrt(np.mean((dat.fac[highalt]*1e6 - 1e6*jpar[highalt])**2))
    corr_fac = np.corrcoef(dat.fac[highalt], jpar[highalt])[0,1]

    ax.text(15, -60, 'RMSE $[\mu A/m^2], r$', color='black', ha='left')   
    ax.text(50, -75, '%4.1f,   %4.2f' % (RMSE_u, corr_u), color='C0', ha='right')
    ax.text(50, -90, '%4.1f,   %4.2f' % (RMSE_n, corr_n), color='C1', ha='right')
    ax.text(50, -105, '%4.1f,   %4.2f' % (RMSE_e, corr_e), color='C2', ha='right')
    ax.text(50, -120, '%4.1f,   %4.2f' % (RMSE_fac, corr_fac), color='C3', ha='right')

    return fig
    
    # sum(np.abs(residual_r/d[0:N])<0.2)/N
    # sum(np.abs(residual_r/d[0:N])[inside]<0.2)/sum(inside)
    # sum(np.abs(residual_r/d[0:N])[~inside]<0.2)/sum(~inside)        
        
############
# Dipole mapping investigation notes
# Investigate the two ways of mapping. The dipole lib mappig seems to flatten out the mapped
# velocities better than the sofko/walker mapping. Why? This suggests that the dipole lib
# mapping is better I think. Have also tried converting the mapped velocities (ENU) back to GEMINI 
# bases. Both methods leads to distrobitions around 0 m/s, but the dipole lib mapping has std of
# 2 m/s compared to S/W with 4 m/s spread. This also favors dipole lib mapping I think. 
# Lastly, from computung vmapped dot B, the orthogonality is actually best satisfied with
# S/W mapping, where the std of the angle between vmapped and B is 1e-6 degrees, while 0.5 degrees for
# dipole lib mapping. Both dists are centered on 90 deg (89.8 for dipole lib). This is
# a bit strange considerging the larger values of v1mapped for S/W mapping. But could maybe
# be due to inaccuracies in dipole pole location? Nevertheless, we proceed with dipole lib mapping
# plt.pcolormesh(dat0.vperpmappedn[:,:,100]- dat0.vperpmappedn[0,:,100], cmap='seismic', vmin=-10,vmax=10)
# plt.pcolormesh(dat.vperpmappedn[:,:,100]- dat.vperpmappedn[0,:,100], cmap='seismic', vmin=-10,vmax=10)
# use = xg['alt']*1e-3>=maph
# plt.scatter(dat.vperpmappede.values[use], dat0.vperpmappede.values[use])
# #Convert dipole lib mapped velovities to gemini base
# enu_vec = np.vstack((dat.vperpmappede.values[use],dat.vperpmappedn.values[use],dat.vperpmappedu.values[use])).T
# enu_vec0 = np.vstack((dat0.vperpmappede.values[use],dat0.vperpmappedn.values[use],dat0.vperpmappedu.values[use])).T
# v1mapped, v2mapped, v3mapped = gemini_tools.enu_vec_2_gemini_vec(enu_vec, dat.mappedglon.values[use], dat.mappedglat.values[use])
# v1mapped0, v2mapped0, v3mapped0 = gemini_tools.enu_vec_2_gemini_vec(enu_vec0, dat0.mappedglon.values[use], dat0.mappedglat.values[use])
# vdotb = dat.vperpmappede.values[use]*dat.Be.values[use] + dat.vperpmappedn.values[use]*dat.Bn.values[use] + dat.vperpmappedu.values[use]*dat.Bu.values[use]
# vdotb0 = dat0.vperpmappede.values[use]*dat0.Be.values[use] + dat0.vperpmappedn.values[use]*dat0.Bn.values[use] + dat0.vperpmappedu.values[use]*dat0.Bu.values[use]
# bmags = xg['Bmag'][use]
# vmappedmag = np.sqrt(enu_vec[:,0]**2+enu_vec[:,1]**2+enu_vec[:,2]**2)
# vmappedmag0 = np.sqrt(enu_vec0[:,0]**2+enu_vec0[:,1]**2+enu_vec0[:,2]**2)
# plt.hist(np.degrees(np.arccos(vdotb/(bmags*vmappedmag))))
# plt.hist(np.degrees(np.arccos(vdotb0/(bmags*vmappedmag))))    

def scatterplot_lompe(ax, sim, dat, conv, gr):
    '''
    Functionto produce plot of how the lompe fit reproduces the F-region ion drift
    velocities perp to B in GEMINI

    Parameters
    ----------
    ax : matplotlib axis object
        To plot on
    sim : instance of simulation class
        To hold the GEMINI data and grid        
    dat : instane of data class
        Containing the GEMINI data sampled at the same altitude as the lompe model
        is represented at.        
    conv : instance of convection class
        Contain lompe representation. conv.data also contain the instance of the
        data class of the E3D like samples used to make the lompe fit, but also contain
        all samples also below maph.
    gr : instance of the grid class
        Contains both lompe and 3D grids


    Returns
    -------
    axis object

    '''
    
    use = gr.grid.ingrid(lon=dat.mappedglon, lat=dat.mappedglat)

    enugg_vec = conv.get_E_from_lmodel(sim, dat, returnvperp=True).T
    
    RMSE_e = np.sqrt(np.mean((dat.vperpmappede[use] - enugg_vec[use,0])**2))
    RMSE_n = np.sqrt(np.mean((dat.vperpmappedn[use] - enugg_vec[use,1])**2))
    RMSE_u = np.sqrt(np.mean((dat.vperpmappedu[use] - enugg_vec[use,2])**2))
    RMSE = np.mean(np.array([RMSE_e, RMSE_n, RMSE_u]))
    residuals_e = dat.vperpmappede[use] - enugg_vec[use,0]
    residuals_n = dat.vperpmappedn[use] - enugg_vec[use,1]
    residuals_u = dat.vperpmappedu[use] - enugg_vec[use,2]
    residuals = np.sqrt(residuals_e**2 + residuals_n**2 + residuals_u**2)
    large = residuals > 100
    frac = 100 * sum(large)/residuals.size
    print('%2i percent of residuals > 100 m/s' % frac)

    ax.scatter(dat.vperpmappede[use], enugg_vec[use,0], label='$v_\perp$ east', alpha=0.1, color='C0')
    ax.scatter(dat.vperpmappedn[use], enugg_vec[use,1], label='$v_\perp$ north', alpha=0.1, color='C1')
    ax.scatter(dat.vperpmappedu[use], enugg_vec[use,2], label='$v_\perp$ up', alpha=0.1, color='C2')
    ax.legend(frameon=False)
    # ax.set_xlabel('True value, no noise [m/s]')
    # ax.set_ylabel('Estimate from noisy data [m/s]')
    minx = np.min(np.hstack((dat.vperpe[use],dat.vperpn[use])))
    maxx = np.max(np.hstack((dat.vperpe[use],dat.vperpn[use])))
    miny = np.min(np.hstack((enugg_vec[use,0],enugg_vec[use,1])))
    maxy = np.max(np.hstack((enugg_vec[use,0],enugg_vec[use,1])))
    # minx = -1900
    # maxx = 550
    # miny = -1900
    # maxy = 450
    xx = np.linspace(np.min((minx,miny)), np.max((maxx, maxy)),10)
    # yy = np.linspace(miny,maxy,10)
    ax.plot(xx,xx, color='black')
    
    # Residuals:
    ax.text(-1800,-150, 'RMSE [m/s]:')
    ax.text(-1825,-250, '%3i' % RMSE_e, color='C0')
    ax.text(-1825,-350, '%3i' % RMSE_n, color='C1')
    ax.text(-1825,-450, '%3i' % RMSE_u, color='C2')

    
    # extent = np.max(np.abs(np.hstack((ax.get_xlim(), ax.get_ylim()))))
    ax.set_aspect('equal')
    
    # ax.plot([-extent, extent], [-extent, extent], 'k-', zorder = 7)
    # ax.set_xlim(minx, maxx)
    # ax.set_ylim(miny, maxy)
    
    ax.plot([0, 0], [0.5*miny, 0.5*maxy], linestyle = '--', color = 'black', zorder = 7)
    ax.text(50, -1450, 'Estimate from noisy data [m/s]', rotation=90)
    ax.plot([0.5*minx, 0.5*maxx], [0, 0], linestyle = '--', color = 'black', zorder = 7)
    ax.text(-1200, 100, 'True value, no noise [m/s]')
    
    scale = 1000
    xloc = minx+125
    dx = 25
    yloc = miny + scale/2 + 350
    
    ax.plot([xloc,xloc], [yloc-scale/2, yloc+scale/2], 'k-', zorder = 7)
     # scale
    ax.plot([xloc-dx, xloc+dx], [yloc-scale/2, yloc-scale/2], 'k-', zorder = 7)
    ax.plot([xloc-dx, xloc+dx], [yloc+scale/2, yloc+scale/2], 'k-', zorder = 7)
    ax.text(xloc - 5*dx, yloc-200, '%4i m/s' % scale, rotation=90)

    ax.set_axis_off()
    
    return ax

def snr_output_plot(covar_j, meshgrid, dat, grid, alts_grid, clim=2e-5, 
                    cut='j', ind=5, transmitter=('ski_mod', 67.5, 23.7), 
                    receivers=[('ski_mod', 67.5, 23.7),
                     ('krs_mod', 66.55, 25.92), ('kai_mod', 66.48, 22.54)],
                    climsnr=1):

    datadict = dat.__dict__

    # ind = 5 # slice index to show
    # cut = 'k' # slice dimension

    alt_ev = meshgrid[0]
    lat_ev = meshgrid[1]
    lon_ev = meshgrid[2]
    
    # Native 3D SECS grid
    alt_ev_, eta_ev_, xi_ev_ = np.meshgrid(alts_grid, grid.eta[:,0], grid.xi[0,:], indexing='ij')    
    lon_ev_, lat_ev_ = grid.projection.cube2geo(xi_ev_, eta_ev_)
    lon_ev_, lat_ev_ = geomag2geog(np.radians(lon_ev_), np.radians(90-lat_ev_))


    N = covar_j.shape[0] // 3
    K = alts_grid.size
    I = grid.shape[0]
    J = grid.shape[1]
    KIJ = K*I*J

    # Convert current densiry to dipole components: Not needed anymore since we now 
    # do evertyhing in geographic components/coordinates
    # jgg_enu = np.vstack((datadict['je'],datadict['jn'], datadict['ju'])).T
    # jgm_enu = secs3d.gemini_tools.enugg2enugm(jgg_enu, datadict['mappedglon'], \
    #         datadict['mappedglat']) # Convert ion perp velocity (local ENU) from geographic to geomag components    
    # datadict['je_gm'] = jgm_enu[:,0]
    # datadict['jn_gm'] = jgm_enu[:,1]
    # datadict['ju_gm'] = jgm_enu[:,2]
    fig = plt.figure(figsize=(15,10))
    
    # Top row showing 3D reconstruction uncertainties. Bottom row show SNR
    ax3 = plt.subplot2grid((20, 32), (0, 0), rowspan = 10, colspan = 10, projection='3d')
    ax2 = plt.subplot2grid((20, 32), (0, 10), rowspan = 10, colspan = 10, projection='3d')
    ax1 = plt.subplot2grid((20, 32), (0, 20), rowspan = 10, colspan = 10, projection='3d')
    axs = [ax1,ax2,ax3]
    ax6 = plt.subplot2grid((20, 32), (10, 0), rowspan = 10, colspan = 10, projection='3d')
    ax5 = plt.subplot2grid((20, 32), (10, 10), rowspan = 10, colspan = 10, projection='3d')
    ax4 = plt.subplot2grid((20, 32), (10, 20), rowspan = 10, colspan = 10, projection='3d')
    axs2 = [ax4,ax5,ax6] 
    plotparams = ['je', 'jn','ju']
    plot_titles = ['$\\sigma_{j_{\phi}}$', '$\\sigma_{j_{\\theta}}$', '$\\sigma_{j_{r}}$']
    kwargs={'linewidth':1}
    plot_titles2 = ['$|j_{\phi, GEMINI}| / \\sigma_{j_{\phi}}$', '$|j_{\\theta, GEMINI}| / \\sigma_{j_{\\theta}}$', '$|j_{r, GEMINI}| / \\sigma_{j_{r}}$']
    
    for pp, ax in enumerate(axs):
        if pp == 0:
            pp2 = 2
        elif pp == 1:
            pp2 = 1
        elif pp ==2:
            pp2 = 0
        noise = np.sqrt(np.diag(covar_j[pp2*N:(pp2+1)*N,pp2*N:(pp2+1)*N])).reshape(alt_ev.shape)

        signal = datadict[plotparams[pp]].reshape(alt_ev.shape)

        # noise_cf = np.sqrt(np.diag(Cmpost[0:KIJ,:KIJ]).reshape((K,I,J)))
        # noise_df = np.sqrt(np.diag(Cmpost[1*KIJ:2*KIJ,1*KIJ:2*KIJ]).reshape((K,I,J)))
        ax2 = axs2[pp]
        ax = visualization.field_aligned_grid(ax, grid, alts_grid, color='green', dipoleB=False, **kwargs, coastlines=False)
        ax.set_title(plot_titles[pp], fontsize=16)
        ax2 = visualization.field_aligned_grid(ax2, grid, alts_grid, color='green', dipoleB=False, **kwargs, coastlines=False)
        ax2.set_title(plot_titles2[pp], fontsize=16)
        if pp==2:
            x_, y_, z_ = coordinates.sph_to_car((RE+0, 90-74, 37), deg=True)
            ax.text(x_[0], y_[0], z_[0], 'Uncertainty', fontsize=16)
            ax2.text(x_[0], y_[0], z_[0], 'SNR', fontsize=16)
        if type(cut) == str:
            ax = visualization.plotslice(ax,(alt_ev,lat_ev,lon_ev), noise, cut=cut, ind=ind, clim=clim)
            ax2 = visualization.plotslice(ax2,(alt_ev,lat_ev,lon_ev), np.abs(signal)/noise, cut=cut, 
                                     ind=ind, clim=5, diverging=False, cmap='viridis')  
        else:
            for ii, c in enumerate(cut):
                ax = visualization.plotslice(ax,(alt_ev,lat_ev,lon_ev), noise, cut=c, ind=ind[ii], clim=clim)
                ax2 = visualization.plotslice(ax2,(alt_ev,lat_ev,lon_ev), np.abs(signal)/noise, cut=c, 
                                         ind=ind[ii], clim=climsnr, diverging=False, cmap='viridis')
                # Plot the radar site locations
                alts = np.ones(3)*0.01
                glats = np.array([receivers[0][1], receivers[1][1], receivers[2][1]])
                glons = np.array([receivers[0][2], receivers[1][2], receivers[2][2]])
                x_, y_, z_ = coordinates.sph_to_car((RE+alts, 90-glats, glons), deg=True)
                ax.scatter(x_, y_, z_, marker='*', c=['C0', 'C1', 'C2'])
                ax2.scatter(x_, y_, z_, marker='*', c=['C0', 'C1', 'C2'])
                # Print height of hor cuts
                # if (c=='k') & (pp==2):
                #     alt = int(alt_ev[ind[ii],0,0])
                #     # glon, glat = geomag2geog(np.radians(grid.lon[-1,0]), np.radians(90-grid.lat[-1,0]))
                #     x_, y_, z_ = coordinates.sph_to_car((RE+alt, 90-grid.lat[-1,0]-0.4, grid.lon[-1,0]), deg=True)
                #     ax.text(x_[0], y_[0], z_[0], str(alt)+' km', fontsize=10)
                #     ax2.text(x_[0], y_[0], z_[0], str(alt)+' km', fontsize=10)
                    
                
    # Colorbar SNR
    cbarax = plt.subplot2grid((20,32), (10, 31), rowspan = 8, colspan = 1)
    cmap = plt.cm.viridis
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=climsnr)
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label('[SNR]', fontsize=16)
    cb1.ax.yaxis.set_label_coords(-1.3, 0.5)                
    
    # Colorbar error
    cbarax = plt.subplot2grid((20,32), (0, 31), rowspan = 8, colspan = 1)
    cmap = plt.cm.bwr
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
    cb2 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb2.set_label('[A/m$^2$]', fontsize=16)
    cb2.ax.yaxis.set_label_coords(-1.3, 0.5)                
    
    return fig
    
    # fig.savefig('./3d_uncertainty.pdf',bbox_inches='tight', dpi=300)
    

def performance_plot(reconstructed_j, meshgrid, dat, grid, alts_grid, clim=2e-5, 
                    cut='j', ind=5, transmitter=('ski_mod', 67.5, 23.7), 
                    receivers=[('ski_mod', 67.5, 23.7),
                     ('krs_mod', 66.55, 25.92), ('kai_mod', 66.48, 22.54)]):

    # Function that make a plot of reconstructed j and compare with ground truth
    # from GEMINI. This function is on the same form as the SNR plot function above


    datadict = dat.__dict__

    # ind = 5 # slice index to show
    # cut = 'k' # slice dimension

    alt_ev = meshgrid[0]
    lat_ev = meshgrid[1]
    lon_ev = meshgrid[2]
    
    # Native 3D SECS grid
    alt_ev_, eta_ev_, xi_ev_ = np.meshgrid(alts_grid, grid.eta[:,0], grid.xi[0,:], indexing='ij')    
    lon_ev_, lat_ev_ = grid.projection.cube2geo(xi_ev_, eta_ev_)
    lon_ev_, lat_ev_ = geomag2geog(np.radians(lon_ev_), np.radians(90-lat_ev_))


    N = reconstructed_j.shape[0] // 3
    K = alts_grid.size
    I = grid.shape[0]
    J = grid.shape[1]
    KIJ = K*I*J

    # Convert current densiry to dipole components: Not needed anymore since we now 
    # do evertyhing in geographic components/coordinates
    # jgg_enu = np.vstack((datadict['je'],datadict['jn'], datadict['ju'])).T
    # jgm_enu = secs3d.gemini_tools.enugg2enugm(jgg_enu, datadict['mappedglon'], \
    #         datadict['mappedglat']) # Convert ion perp velocity (local ENU) from geographic to geomag components    
    # datadict['je_gm'] = jgm_enu[:,0]
    # datadict['jn_gm'] = jgm_enu[:,1]
    # datadict['ju_gm'] = jgm_enu[:,2]
    fig = plt.figure(figsize=(15,10))
    
    # Top row showing 3D reconstruction uncertainties. Bottom row show SNR
    ax3 = plt.subplot2grid((20, 32), (0, 0), rowspan = 10, colspan = 10, projection='3d')
    ax2 = plt.subplot2grid((20, 32), (0, 10), rowspan = 10, colspan = 10, projection='3d')
    ax1 = plt.subplot2grid((20, 32), (0, 20), rowspan = 10, colspan = 10, projection='3d')
    axs = [ax1,ax2,ax3]
    ax6 = plt.subplot2grid((20, 32), (10, 0), rowspan = 10, colspan = 10, projection='3d')
    ax5 = plt.subplot2grid((20, 32), (10, 10), rowspan = 10, colspan = 10, projection='3d')
    ax4 = plt.subplot2grid((20, 32), (10, 20), rowspan = 10, colspan = 10, projection='3d')
    axs2 = [ax4,ax5,ax6] 
    plotparams = ['je', 'jn','ju']
    plot_titles2 = ['$j_{\phi}$', '$j_{\\theta}$', '$j_{r}$']
    kwargs={'linewidth':1}
    plot_titles = ['$j_{\phi}$', '$j_{\\theta}$', '$j_{r}$']
    
    for pp, ax in enumerate(axs):
        if pp == 0:
            pp2 = 2
        elif pp == 1:
            pp2 = 1
        elif pp ==2:
            pp2 = 0
        _j = reconstructed_j[pp2*N:(pp2+1)*N].reshape(alt_ev.shape)
        signal = datadict[plotparams[pp]].reshape(alt_ev.shape)
        if pp == 1:
            signal = -signal

        # noise_cf = np.sqrt(np.diag(Cmpost[0:KIJ,:KIJ]).reshape((K,I,J)))
        # noise_df = np.sqrt(np.diag(Cmpost[1*KIJ:2*KIJ,1*KIJ:2*KIJ]).reshape((K,I,J)))
        ax2 = axs2[pp]
        ax = visualization.field_aligned_grid(ax, grid, alts_grid, color='green', dipoleB=False, **kwargs, coastlines=False)
        ax.set_title(plot_titles[pp], fontsize=16)
        ax2 = visualization.field_aligned_grid(ax2, grid, alts_grid, color='green', dipoleB=False, **kwargs, coastlines=False)
        ax2.set_title(plot_titles2[pp], fontsize=16)
        if pp==2:
            x_, y_, z_ = coordinates.sph_to_car((RE+0, 90-74, 37), deg=True)
            ax.text(x_[0], y_[0], z_[0], 'GEMINI', fontsize=16, zorder=10)
            ax2.text(x_[0], y_[0], z_[0], '3D reconstruction', fontsize=16, zorder=10)
        if type(cut) == str:
            ax = visualization.plotslice(ax,(alt_ev,lat_ev,lon_ev), signal, cut=cut, ind=ind, clim=clim)
            ax2 = visualization.plotslice(ax2,(alt_ev,lat_ev,lon_ev), _j, cut=cut, clim=clim) 
        else:
            for ii, c in enumerate(cut):
                ax = visualization.plotslice(ax,(alt_ev,lat_ev,lon_ev), signal, cut=c, ind=ind[ii], clim=clim)
                ax2 = visualization.plotslice(ax2,(alt_ev,lat_ev,lon_ev), _j, cut=c, ind=ind[ii], clim=clim)
                # Plot the radar site locations
                alts = np.ones(3)*0.01
                glats = np.array([receivers[0][1], receivers[1][1], receivers[2][1]])
                glons = np.array([receivers[0][2], receivers[1][2], receivers[2][2]])
                x_, y_, z_ = coordinates.sph_to_car((RE+alts, 90-glats, glons), deg=True)
                ax.scatter(x_, y_, z_, marker='*', c=['C0', 'C1', 'C2'])
                ax2.scatter(x_, y_, z_, marker='*', c=['C0', 'C1', 'C2'])
                # Print alatitude of hor cuts
                # if (c=='k') & (pp==2):
                #     alt = int(alt_ev[ind[ii],0,0])
                #     # glon, glat = geomag2geog(np.radians(grid.lon[-1,0]), np.radians(90-grid.lat[-1,0]))
                #     x_, y_, z_ = coordinates.sph_to_car((RE+alt, 90-grid.lat[-1,0]-0.4, grid.lon[-1,0]), deg=True)
                #     ax.text(x_[0], y_[0], z_[0], str(alt)+' km', fontsize=10)
                #     ax2.text(x_[0], y_[0], z_[0], str(alt)+' km', fontsize=10)
                
    # Colorbar
    cbarax = plt.subplot2grid((20,32), (5, 31), rowspan = 10, colspan = 1)
    cmap = plt.cm.bwr
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label('[A/m$^2$]', fontsize=16)
    cb1.ax.yaxis.set_label_coords(-1.3, 0.5)            
    
    return fig


def output_plot(reconstructed_p, meshgrid, dat, grid, alts_grid, clim=2e-5, 
                    cut='j', ind=5, transmitter=('ski_mod', 67.5, 23.7), 
                    receivers=[('ski_mod', 67.5, 23.7),
                     ('krs_mod', 66.55, 25.92), ('kai_mod', 66.48, 22.54)],
                    plot_titles = ['$j_{\phi}$', '$j_{\\theta}$', '$j_{r}$'],
                    unit='[m/s]'):

    # Function that make a plot of reconstructed j and compare with ground truth
    # from GEMINI. This function is on the same form as the SNR plot function above


    datadict = dat.__dict__

    # ind = 5 # slice index to show
    # cut = 'k' # slice dimension

    alt_ev = meshgrid[0]
    lat_ev = meshgrid[1]
    lon_ev = meshgrid[2]
    
    # Native 3D SECS grid
    alt_ev_, eta_ev_, xi_ev_ = np.meshgrid(alts_grid, grid.eta[:,0], grid.xi[0,:], indexing='ij')    
    lon_ev_, lat_ev_ = grid.projection.cube2geo(xi_ev_, eta_ev_)
    lon_ev_, lat_ev_ = geomag2geog(np.radians(lon_ev_), np.radians(90-lat_ev_))


    N = reconstructed_p.shape[0] // 3
    K = alts_grid.size
    I = grid.shape[0]
    J = grid.shape[1]
    KIJ = K*I*J

    # Convert current densiry to dipole components: Not needed anymore since we now 
    # do evertyhing in geographic components/coordinates
    # jgg_enu = np.vstack((datadict['je'],datadict['jn'], datadict['ju'])).T
    # jgm_enu = secs3d.gemini_tools.enugg2enugm(jgg_enu, datadict['mappedglon'], \
    #         datadict['mappedglat']) # Convert ion perp velocity (local ENU) from geographic to geomag components    
    # datadict['je_gm'] = jgm_enu[:,0]
    # datadict['jn_gm'] = jgm_enu[:,1]
    # datadict['ju_gm'] = jgm_enu[:,2]
    fig = plt.figure(figsize=(15,6))
    
    # Top row showing 3D reconstruction uncertainties. Bottom row show SNR
    ax3 = plt.subplot2grid((10, 32), (0, 0), rowspan = 10, colspan = 10, projection='3d')
    ax2 = plt.subplot2grid((10, 32), (0, 10), rowspan = 10, colspan = 10, projection='3d')
    ax1 = plt.subplot2grid((10, 32), (0, 20), rowspan = 10, colspan = 10, projection='3d')
    axs = [ax1,ax2,ax3]
    kwargs={'linewidth':1}
    
    for pp, ax in enumerate(axs):
        if pp == 0:
            pp2 = 2
        elif pp == 1:
            pp2 = 1
        elif pp ==2:
            pp2 = 0
        _j = reconstructed_p[pp2*N:(pp2+1)*N].reshape(alt_ev.shape)


        # noise_cf = np.sqrt(np.diag(Cmpost[0:KIJ,:KIJ]).reshape((K,I,J)))
        # noise_df = np.sqrt(np.diag(Cmpost[1*KIJ:2*KIJ,1*KIJ:2*KIJ]).reshape((K,I,J)))
        ax = visualization.field_aligned_grid(ax, grid, alts_grid, color='green', dipoleB=False, **kwargs, coastlines=False)
        ax.set_title(plot_titles[pp], fontsize=16)
        if pp==2:
            x_, y_, z_ = coordinates.sph_to_car((RE+0, 90-74, 37), deg=True)
            ax.text(x_[0], y_[0], z_[0], '3D reconstruction', fontsize=16, zorder=10)
        if type(cut) == str:
            ax = visualization.plotslice(ax,(alt_ev,lat_ev,lon_ev), _j, cut=cut, clim=clim) 
        else:
            for ii, c in enumerate(cut):
                ax = visualization.plotslice(ax,(alt_ev,lat_ev,lon_ev), _j, cut=c, ind=ind[ii], clim=clim)
                # Plot the radar site locations
                alts = np.ones(3)*0.01
                glats = np.array([receivers[0][1], receivers[1][1], receivers[2][1]])
                glons = np.array([receivers[0][2], receivers[1][2], receivers[2][2]])
                x_, y_, z_ = coordinates.sph_to_car((RE+alts, 90-glats, glons), deg=True)
                ax.scatter(x_, y_, z_, marker='*', c=['C0', 'C1', 'C2'])
                # Print alatitude of hor cuts
                # if (c=='k') & (pp==2):
                #     alt = int(alt_ev[ind[ii],0,0])
                #     # glon, glat = geomag2geog(np.radians(grid.lon[-1,0]), np.radians(90-grid.lat[-1,0]))
                #     x_, y_, z_ = coordinates.sph_to_car((RE+alt, 90-grid.lat[-1,0]-0.4, grid.lon[-1,0]), deg=True)
                #     ax.text(x_[0], y_[0], z_[0], str(alt)+' km', fontsize=10)
                #     ax2.text(x_[0], y_[0], z_[0], str(alt)+' km', fontsize=10)
                
    # Colorbar
    cbarax = plt.subplot2grid((10,32), (1, 31), rowspan = 7, colspan = 1)
    cmap = plt.cm.bwr
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label(unit, fontsize=16)
    cb1.ax.yaxis.set_label_coords(-1.3, 0.5)            
    
    return fig    