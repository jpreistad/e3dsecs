#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:45:47 2023

@author: jone

High level script that should demonstrate the use of e3dsecs to reconstruct 
3D electric current field in a specified region above the radar facility.

"""

import xarray as xr
import gemini3d.read as read
import numpy as np
import lompe
from secsy import cubedsphere
try:
    # from . import visualization
    from . import secs3d
    from . import gemini_tools
    from . import uncertainty
    from . import diagnostics
except:
    # import visualization
    import secs3d
    import uncertainty
    import gemini_tools
    import diagnostics


########################################
# Run options
maph            = 200   # height of mapping of ion velocity, in km.
extend          = 5     # How many padding "frames" used for lompe grid around 3D hor. grid
inputmode = 'vi'        # How jperp is estimated. Must be either:
                        #   'vi'            : Use samples of ion velocities and (vi-ve)
                        #   'vi_ohmslaw'    : Use samples of ion velocities and Ohms law
                        #   'phitop'        : Use potential at top from GEMINI and (vi-ve)
                        #   'phitop_ohmslaw': Use potential at top from GEMINI and Ohms law
                        #   'jperp'         : Use jperp directly sampled from GEMINI
l1_lompe        = 1e-1  # Regularization parameter for Lompe representation
l2_lompe        = 1e0   # Regularization parameter for Lompe representation
l1              = 1e-2  # Regularization parameter for 3D reconstruction inversion
intsec          = 5*60  # Integrationtime in seconds used in E3DOUBT
factop          = True  # Provide values of vertical current at top of domain
vert_profile    = None  # Regularize vertical Hall and Pedersen profile based on 
                        # GEMINI profile. None or # km from where it is applied above
vert_dprofile   = False # Regularize vertical profile of gradient of H/P currents 
                        # based on electron density profile. NOT WORKING
gcv             = False # Determine 3D model reg. parameter using GCV score
overwrite       = True  # Overwrites exisring 3D model coefficient file
e3doubt_        = True  # Estimate sample (co)variances of ne and v with E3DOUBT
addnoise        = True  # Adds noise to data based on the E3DOUBT variances
diagnostic      = True  # Wheter to make diagnostic plots
########################################


########################################
# Load GEMINI grid and data
try: # look for saved file including some of the needed types of data    
    try:
        path = "/home/ubuntu/gemini_data/run1/gemini_run_initial_mini/"
        dat = xr.open_dataset(path+'temp_dat.nc')
    except:
        path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/" + \
                "issi_team/gemini_output/"
        dat = xr.open_dataset('/Users/jone/BCSS-DAG Dropbox/Jone Reistad/' + \
                'tmpfiles/temp3_dat.nc')
    xg = read.grid(path)
except: # make the datafiles from reading GEMINI output
    path = "/Users/jone/BCSS-DAG Dropbox/Jone Reistad/projects/eiscat_3d/" + \
                "issi_team/gemini_output/"
    xg, dat = gemini_tools.read_gemini(path, timeindex=-1, maph=maph)
    dat.attrs={}
    dat.to_netcdf('/Users/jone/BCSS-DAG Dropbox/Jone Reistad/tmpfiles/temp3_dat.nc')
xgdat = (xg, dat)

########################################
# Define grids for 3D model
# Altitude grid
alts_grid = np.concatenate((np.arange(90,140,5),np.arange(140,170,10), 
                            np.arange(170,230,20),np.arange(230,500,50)))
altres = np.diff(alts_grid)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
# Horizontal CS grid
grid, grid_l = gemini_tools.make_csgrid(xg, maph=maph, h0=alts_grid[0], crop_factor=0.2,
                                    resolution_factor=0.45, extend=extend, dlat = 0.2)
#Grid dimensions
K = alts_grid.shape[0] #Number of vertival layers
I = grid.shape[0] #Number of cells in eta direction, north-south, W dimension
J = grid.shape[1]  #Number of cells in xi direction, east-west, L dimension
KIJ = K*I*J
IJ = I*J

########################################
# Sample some data in 3D at specified beams
min_alt = 90 # of where to sample along beams
max_alt = 500 
dr = 4
az = None # np.array([20])# None
el = None # np.array([37])# None
sitelat = 67.2 # geo lat of transmitter. Skibotn: 69.39
sitephi=23.7 # geo lon of transmitter. Skibotn: 20.27
datadict = gemini_tools.sample_eiscat(xg, dat, min_alt=min_alt, max_alt=max_alt, 
                        dr=dr, sitelat=sitelat, sitephi=sitephi, az=az, el=el)
datadict['maph'] = maph

########################################
# Estimate variances of the oberved values
# datadict = datadict_backup.copy()
if e3doubt_:
    datadict = uncertainty.get_datacov_e3doubt(datadict, intsec=intsec)
    datadict = uncertainty.remove_bad(datadict)
    datadict_backup = datadict.copy()
    if addnoise:
        datadict = uncertainty.add_noise(datadict, maph)
    lompedata = uncertainty.make_datacov_lompe(datadict.copy(), grid_l, maph)
    # Do Lompe fit (actually, just to initialise model object)
    lmodel = gemini_tools.lompe_fit(lompedata, grid_l, l1=l1_lompe, l2=l2_lompe, 
                            altlim = maph, e3doubt_=e3doubt_)
    # Do the lompe inversion and calculate model covariance matrix
    m, Cmpost = uncertainty.make_cmpost(lmodel, lompedata, l1=l1_lompe, l2=l2_lompe)
    lmodel.m = m.copy()
    # Calculate the covariance matrix of lompe representation of v_perp at maph
    # at the locations that map to each observation, 'covVlompe'
    datadict = uncertainty.make_lompe_v_cov(lmodel, datadict, Cmpost)
    # Calculate covariance of predicted electron perp velocity when mapped to
    # measurement locations, 
    datadict = uncertainty.make_ve_cov(lmodel, datadict)
    # Calculate covariance of jperp based on (vi_perp-ve_perp)
    datadict = uncertainty.make_cov_jperp(datadict)

else:
    # lmodel = None
    if (inputmode == 'vi') or (inputmode == 'vi_ohmslaw'):
        # Do Lompe fit
        l1_lompe = 0.5
        l2_lompe = 5
        lmodel = gemini_tools.lompe_fit(datadict.copy(), grid_l, l1=l1_lompe,
                                        l2=l2_lompe, altlim = maph, 
                                        e3doubt_=e3doubt_)    
    else:
        lmodel=None
        
#####################################
# Inversion
filename, filename_lompe = secs3d.run_inversion(grid, alts_grid, datadict, 
                    inputmode=inputmode,lmodel=lmodel, factop=factop, 
                    vert_profile=vert_profile, vert_dprofile=vert_dprofile, 
                    l1=l1, diagnostic=diagnostic, overwrite=overwrite, gcv=gcv, 
                    xgdat=xgdat, e3doubt_=e3doubt_)


##########################################################
###### Analysis of performance########
# Load inversion results
filename, filename_lompe = secs3d.make_filenames(grid.projection.position, inputmode)
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

# Compute GEMINI values at the evaluation locations
datadict['fac'] = np.sum(np.array([datadict['ju'], -datadict['jn'], datadict['je']]) * 
                np.array([br, btheta, bphi]), axis=0)
datadict['shape'] = shape

# Scatterplot of reconstruction performance
diagnostics.scatterplot_reconstruction(grid, alts_grid, datadict, lon_ev.flatten(), \
                            lat_ev.flatten(), alt_ev.flatten(), full_j, jpar, \
                            dipolekw=False, inout=False)
    
#Make gifs of performance in different slices
clim = 1e-5
diagnostics.reconsrtruction_performance(datadict, grid, alts_grid, lat_ev, lon_ev, alt_ev, full_j, jpar, 
                                        dipolekw=False, cut='j', clim=clim, 
                                        single=4, gif=False, inputmode=inputmode)

diagnostics.model_amplitude_analysis(grid,alts_grid, m[1*K*I*J:2*K*I*J], clim=1e-1, 
                                     dipoleB=False, k=20)    


##############
# Make SNR plot of reconstruction
Cmpost = m_['Cmpost']
covar_j = G.dot(Cmpost).dot(G.T) # Nothe the r,theta,phi order
meshgrid = (alt_ev,lat_ev,lon_ev)
diagnostics.snr_output_plot(covar_j, meshgrid, datadict, grid, alts_grid, Cmpost, 
                            clim=clim, cut=['j','k','k'], ind=[5,5,18])

# ###############################
# #Make plot of variation from multiple realizations of the same noise
# fulljs = []
# for nn in range(NN):
#     fullj = G.dot(ms[nn])
#     fulljs.append(fullj)
# fulljs = np.array(fulljs)
# std = np.std(fulljs, axis=0)
# meshgrid = (alt_ev,lat_ev,lon_ev)
# diagnostics.snr_output_plot(np.diag(std**2), meshgrid, datadict, grid, alts_grid, Cmpost, 
#                             clim=clim, cut=['j','k','k'], ind=[5,5,18])

# #########################################3
# # Performance testing of vertical current profile regularization
# __lat = grid.lat_mesh[I//2,J//2]
# __lon = grid.lon_mesh[I//2,J//2]
# N = 100
# _lat = np.ones(N) * __lat
# _lon = np.ones(N) * __lon
# _alt = np.linspace(90,200,N)

# # Hall/pedersen profile
# if not estimate_E_field:
#     _phitop =xgdat
# else:
#     _phitop = lmodel
# JP, inds = secs3d.make_JHP(grid, alts_grid, _lat, _lon, _alt, _phitop, hop='p', pederseniscf=True, hallisdf=True)
# JH, inds = secs3d.make_JHP(grid, alts_grid, _lat, _lon, _alt, _phitop, hop='h', pederseniscf=True, hallisdf=True)
# _glon, _glat = geomag2geog(np.radians(_lon), np.pi/2 - np.radians(_lat))
# _ddict = gemini_tools.sample_points(xg, dat, _glat[inds==1], _glon[inds==1], _alt[inds==1], estimate_E_field=True) 
# _ddict['maph'] = maph
# if not estimate_E_field:
#     _phitop = not estimate_E_field
# else:
#     _phitop = lmodel
# jh, jp = gemini_tools.calc_hall_pedersen(_ddict, _phitop)  
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(JP.dot(m),_alt, label='model Pedersen profile')
# plt.plot(jp,_alt, label='GEMINI Pedersen profile') 
# plt.legend()
# plt.figure()
# plt.plot(JH.dot(m),_alt, label='model Hall profile')
# plt.plot(jh,_alt, label='GEMINI Hall profile')
# plt.legend()

# G = secs3d.make_G(grid, m_['alts_grid'], _lat, _lon, _alt, ext_factor=0)
# full_j = G.dot(m)
# plt.figure()
# if dipole_lompe:
#     j_gg_enu = np.vstack((_ddict['je'], _ddict['jn'], _ddict['ju'])).T
#     j_gm_enu = gemini_tools.enugg2enugm(j_gg_enu, _ddict['lon'], _ddict['lat'])    
#     plt.plot(j_gm_enu[:,1], _alt, label='GEMINI jn')
#     plt.plot(-full_j[1*N:2*N], _alt, label='model jn')
#     plt.plot(j_gm_enu[:,0], _alt, '--', label='GEMINI je')
#     plt.plot(full_j[2*N:3*N], _alt, '--', label='model je')
# else:
#     plt.plot(_ddict['jn'], _alt, label='GEMINI jn')
#     plt.plot(-full_j[1*N:2*N], _alt, label='model jn')
#     plt.plot(_ddict['je'], _alt, '--', label='GEMINI je')
#     plt.plot(full_j[2*N:3*N], _alt, '--', label='model je')
# plt.legend()
# plt.xlabel('$[A/m^2]$')
# plt.ylabel('[km]')

# datadict = datadict_backup.copy()
# 

# #Derivative of Hall/Pedersen profile
# DP, inds = secs3d.make_D(lmodel, grid, alts_grid, np.ones(100)*__lat, np.ones(100)*__lon, np.linspace(90,200,100), hop='p', pederseniscf=True, hallisdf=True)
# DH, inds = secs3d.make_D(lmodel, grid, alts_grid, np.ones(100)*__lat, np.ones(100)*__lon, np.linspace(90,200,100), hop='h', pederseniscf=True, hallisdf=True)
# plt.figure()
# plt.plot(DP.dot(m),np.linspace(90,200,100))
# _hdiff, _pdiff = secs3d.sigma_profile(np.linspace(90,200,100), lat=np.ones(100)*__lat, lon=np.ones(100)*__lon, derivative=True, iri=False, xgdat=(xg,dat))
# plt.plot(_pdiff,np.linspace(90,200,100)) 
# plt.figure()
# plt.plot(DH.dot(m),np.linspace(90,200,100))
# plt.plot(_hdiff,np.linspace(90,200,100)) 

#scp ubuntu@nrec:/home/ubuntu/git/secs_3d/inversion_coefs/3Dreconstruction_eiscat_lompe_ohmslaw_118.20_64.06.npy ./inversion_coefs/