#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 06:31:08 2024

@author: jone

High level script that applies the lompe fit and do the 3D reconstruction.
This script is intended to produce the figures for the 3D reconstruction paper
using extended strategies to solve for the model parameters, such as specifyin
radial/FAC pattern on top, and/or requiring the vertical profile of the current
density to follow a specific shape.

Adjust the two paths to fit your system:
1) sys.path to contain the path to the e3dsecs directory
2) path variable to point to gemini output datafiles. These are also stored on the DAG 
    Dropbox Data folder
"""

nrec = False # For testing to use with nrec virtual machine
import sys
if nrec:
    sys.path.append('/home/ubuntu/git/e3dsecs')
    path = '/mnt/data2/gemini/gemini_data/e3dpaper_datafiles/'
else:
    sys.path.append('/Users/jone/Dropbox (Personal)/uib/researcher/git/e3dsecs')
    path = '/Users/jone/Documents/uib/gemini_output/'# Adjust to fit your system
from e3dsecs import gemini_tools, diagnostics, secs3d, uncertainty
import xarray as xr
import numpy as np
import lompe
from secsy import cubedsphere
import h5py

########################################
# Run options
maph            = 200   # height of mapping of ion velocity, in km.
extend          = 5     # How many padding "frames" used for lompe grid around 3D hor. grid
inputmode =     'vi'    # How jperp is estimated. Must be either:
                        #   'vi'            : Use samples of ion velocities and (vi-ve)
                        #   'vi_ohmslaw'    : Use samples of ion velocities and Ohms law
                        #   'phitop'        : Use potential at top from GEMINI and (vi-ve)
                        #   'phitop_ohmslaw': Use potential at top from GEMINI and Ohms law
                        #   'jperp'         : Use jperp directly sampled from GEMINI
l1_lompe        = 1e-1  # Regularization parameter for Lompe representation
l2_lompe        = 0     # Regularization parameter for Lompe representation
l1              = 10**(-2.3)  # Regularization parameter for 3D reconstruction inversion
intsec          = 5*60  # Integrationtime in seconds used in E3DOUBT
factop          = True  # Provide values of vertical current at top of domain
vert_profile    = 150   # Regularize vertical Hall and Pedersen profile based on 
                        # GEMINI profile. None or # km from where it is applied above
vert_dprofile   = False # Regularize vertical profile of gradient of H/P currents 
                        # based on electron density profile. NOT WORKING
gcv             = True  # Determine 3D model reg. parameter using cross-validation
overwrite       = True  # Overwrites exisring 3D model coefficient file
e3doubt_        = True  # Estimate sample (co)variances of ne and v with E3DOUBT
addnoise        = True  # Adds noise to data based on the E3DOUBT variances
diagnostic      = True  # Wheter to make diagnostic plots
########################################


########################################
# Load GEMINI grid and data
try: # look for saved file including some of the needed types of data    
    dat = xr.open_dataset(path + 'gemini_dataset.nc')
    xg = gemini_tools.read_hdf5_to_dict(path + 'gemini_grid.h5')
except: # make the datafiles from reading GEMINI output
    xg, dat = gemini_tools.read_gemini(path, timeindex=-1, maph=maph)
    dat.attrs={}
    dat.to_netcdf(path + 'gemini_dataset.nc')
    del xg['glatctr']
    del xg['glonctr']
    del xg['filename']
    with h5py.File(path + 'gemini_grid.h5', 'w') as file:
        for key, value in xg.items():
            file.create_dataset(key, data=value)
xgdat = (xg, dat)

########################################
# Define grids for 3D model
# Altitude grid
alts_grid = np.concatenate((np.arange(90,140,5),np.arange(140,170,10), 
                            np.arange(170,230,20),np.arange(230,500,50)))
altres = np.diff(alts_grid)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
# Horizontal CS grid
grid, grid_l = gemini_tools.make_csgrid(xg, maph=maph, h0=alts_grid[0], 
                                    crop_factor=0.2, resolution_factor=0.45, 
                                    extend=extend, dlat = 0.2)
#Grid dimensions
K = alts_grid.shape[0] #Number of vertival layers
I = grid.shape[0] #Number of cells in eta direction, north-south, W dimension
J = grid.shape[1]  #Number of cells in xi direction, east-west, L dimension
KIJ = K*I*J
IJ = I*J

########################################
# Step 0: Sample some data in 3D at specified beams
min_alt = 90    # of where to sample along beams
max_alt = 500 
dr      = 4          # altitude resolution of sampling
az      = None       # If None, use default values
el      = None       # If None, use default values
sitelat = 67.7       # geo lat of transmitter. Skibotn: 69.39
sitelon = 23.       # geo lon of transmitter. Skibotn: 20.27
dlat = 69.39- sitelat
dlon = 20.27 - sitelon
lats0 = np.array([69.39, 68.44, 68.37])
lons0 = np.array([20.26, 22.48, 19.10])
lats = np.array([sitelat, lats0[1]-dlat, lats0[2]-dlat])
lons = np.array([sitelon, lons0[1]-dlon, lons0[2]-dlon])
datadict = gemini_tools.sample_eiscat(xg, dat, min_alt=min_alt, max_alt=max_alt, 
                        dr=dr, sitelat=sitelat, sitephi=sitelon, az=az, el=el)
datadict['maph'] = maph
if e3doubt_:
    transmitter=('ski_mod',lats[0],lons[0])
    receivers=[('ski_mod',lats[0],lons[0]), ('krs_mod',lats[1],lons[1]), 
                ('kai_mod',lats[2],lons[2])]
    try: #Try to use an existing file, since the e3doubt calculations take a while
        datadict = np.load('./inversion_coefs/datadict_temp.npy', allow_pickle=True).item()
        datadict_backup = datadict.copy()
    except:
        datadict = uncertainty.get_datacov_e3doubt(datadict, intsec=intsec, 
                            transmitter=transmitter, receivers=receivers)
        datadict = uncertainty.remove_bad(datadict)
        datadict_backup = datadict.copy()
        np.save('./inversion_coefs/datadict_temp.npy', datadict)
    if addnoise:
        datadict = uncertainty.add_noise(datadict, maph, alternative=True)


########################################
# Step 1: Make v_perp representation at maph if specified by inputmode
filename, filename_lompe = secs3d.make_filenames(grid.projection.position, 
                                inputmode, factop=factop, vert_profile=vert_profile)
if (inputmode=='vi') or (inputmode=='vi_ohmslaw'):
    # The make_lompe function also calculates all the necessary covariances
    datadict, lmodel = gemini_tools.make_lompe(grid_l, datadict, inputmode, 
                            maph, e3doubt_=e3doubt_, l1_lompe=l1_lompe, l2_lompe=l2_lompe, 
                            intsec=intsec, filename_lompe=filename_lompe)
else:
    lmodel = None
        
#####################################
# Step 2: Inversion
filename, filename_lompe = secs3d.run_inversion(grid, alts_grid, datadict, 
                    inputmode=inputmode,lmodel=lmodel, factop=factop, 
                    vert_profile=vert_profile, vert_dprofile=vert_dprofile, 
                    l1=l1, diagnostic=diagnostic, overwrite=overwrite, gcv=gcv, 
                    xgdat=xgdat, e3doubt_=e3doubt_)


##########################################################
###### Analysis of performance########
# Load inversion results
filename, filename_lompe = secs3d.make_filenames(grid.projection.position, 
                                inputmode, factop=factop, vert_profile=vert_profile)
m_ = np.load(filename, allow_pickle=True).item()
m = m_['m']
if (inputmode=='vi') or (inputmode=='vi_ohmslaw'):
    lmodel = lompe.load_model(filename_lompe)

# Make new evaluation grid that does not contain locations outside 3D grid    
grid_tuple = m_['grid']
grid = cubedsphere.CSgrid(cubedsphere.CSprojection(grid_tuple[0], grid_tuple[1]), 
                grid_tuple[2], grid_tuple[3], grid_tuple[4], grid_tuple[5], R = grid_tuple[6])
alts_grid = m_['alts_grid'] # alts_grid[1:]-altres[1:]
altres = np.diff(alts_grid)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
alts__ = alts_grid[1:]-altres[1:]
xi_e  = grid.xi[0,1:] - grid.dxi/2 
eta_e = grid.eta[1:,0]- grid.deta/2
alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, eta_e, xi_e, indexing='ij')
lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev)  
shape = lon_ev.shape
# Alternative way to make even finer evaluation grid
# Nxi = int(grid.shape[1]*1.5) #sample resolution, corresponding to direction of "j" index
# Neta = int(grid.shape[0]*1.5) #sample resolution, corresponding to direction of "i" index
# xi_edge = np.linspace(grid.xi_mesh[0,1]+0.01*grid.dxi,grid.xi_mesh[0,-2]-0.01*grid.dxi,Nxi)
# xis = xi_edge[0:-1] + np.diff(xi_edge)/2 # center locations
# eta_edge = np.linspace(grid.eta_mesh[1,0]+0.01*grid.deta,grid.eta_mesh[-2,0]-0.01*grid.deta,Neta)
# etas = eta_edge[0:-1] + np.diff(eta_edge)/2 # center locations
# xi_ev, eta_ev = np.meshgrid(xis, etas, indexing = 'xy')
# alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, etas, xis, indexing='ij')
# lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev)

# First get the true values from GEMINI, and what the input values to the 3D 
# inversion would have been given the lmodel at the evaluating locations, to
# use for later performance evaluation
datadict = gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)
if not 'phitop' in inputmode:
    vperp = gemini_tools.get_E_from_lmodel(lmodel, datadict, xgdat, returnvperp=True)
    datadict['vperp_electron'] = vperp
inputdict = gemini_tools.make_inputdict(datadict, grid, alts_grid,
                    inputmode=inputmode, ext_factor=-1, hp_from_brekke=False)
d = np.hstack((inputdict['jperp'][2,:], -inputdict['jperp'][1,:], 
               inputdict['jperp'][0,:])) # (r, theta, phi components)

# Evaluate model of current denisty
#Make G to evaluate for full j based on the model made above
G = secs3d.make_G(grid, m_['alts_grid'], lat_ev, lon_ev, alt_ev, ext_factor=0)
full_j = G.dot(m)
#Get the jperp and fac of the ful_j expressed by the model, to be compared to input
br, btheta, bphi = secs3d.make_b_unitvectors(datadict['Bu'], 
                -datadict['Bn'], datadict['Be'])
N = br.size
B = secs3d.make_B(br, btheta, bphi)
P = secs3d.make_P(N)
j_perp = P.T.dot(B.dot(P.dot(full_j)))
jpar = np.sum(np.array([full_j[0:N], full_j[N:2*N], full_j[2*N:3*N]]) * 
                np.array([br, btheta, bphi]), axis=0)



##################################################
# Diagnostics plotting
#####################################################
if diagnostic:
    # Compute GEMINI values at the evaluation locations
    datadict['fac'] = np.sum(np.array([datadict['ju'], -datadict['jn'], datadict['je']]) * 
                    np.array([br, btheta, bphi]), axis=0)
    datadict['shape'] = shape
    
    # Scatterplot of reconstruction performance
    fig = diagnostics.scatterplot_reconstruction(grid, alts_grid, datadict, lon_ev.flatten(), \
                                lat_ev.flatten(), alt_ev.flatten(), full_j, jpar, \
                                dipolekw=False, inout=False)
    fig.savefig('./plots/3d_reconstruction_extended_scatterplot.pdf', dpi=250,bbox_inches='tight')    
    
    #Make gifs of performance in different slices
    clim = 2e-5
    fig = diagnostics.reconsrtruction_performance(datadict, grid, alts_grid, lat_ev, 
                                lon_ev, alt_ev, full_j, jpar, dipolekw=False, cut='j', 
                                clim=clim, single=4, gif=False, inputmode=inputmode)
    fig.savefig('./plots/3d_reconstruction_extended.pdf', dpi=250,bbox_inches='tight')
    
    diagnostics.model_amplitude_analysis(grid,alts_grid, m[1*K*I*J:2*K*I*J], clim=0.5, 
                                         dipoleB=False, k=20)    
    
    ##############
    # Make SNR plot of reconstruction
    Cmpost = m_['Cmpost']
    covar_j = G.dot(Cmpost).dot(G.T) # Nothe the r,theta,phi order
    meshgrid = (alt_ev,lat_ev,lon_ev)
    fig = diagnostics.snr_output_plot(covar_j, meshgrid, datadict, grid, 
                                alts_grid, Cmpost, clim=clim, cut=['j','k','k'], 
                                ind=[5,5,18], transmitter=transmitter,
                                receivers=receivers)
    fig.savefig('./plots/3d_uncertainty_extended.pdf',bbox_inches='tight', dpi=300)