#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:58:48 2024

@author: jone

Produce additional plots for the paper

"""

import xarray as xr
import gemini3d.read as read
import sys
sys.path.append('/Users/jone/BCSS-DAG Dropbox/Jone Reistad')
import git.e3dsecs as e3dsecs
import numpy as np
from secsy import cubedsphere
import lompe
import matplotlib.pyplot as plt


# Run options
maph            = 200   # height of mapping of ion velocity, in km.
extend          = 5     # How many padding "frames" used for lompe grid around 3D hor. grid
inputmode =     'vi'    # How jperp is estimated. Must be either:
                        #   'vi'            : Use samples of ion velocities and (vi-ve)
                        #   'vi_ohmslaw'    : Use samples of ion velocities and Ohms law
                        #   'phitop'        : Use potential at top from GEMINI and (vi-ve)
                        #   'phitop_ohmslaw': Use potential at top from GEMINI and Ohms law
                        #   'jperp'         : Use jperp directly sampled from GEMINI

########################################
# Load GEMINI grid and data
path = "/Users/jone/BCSS-DAG Dropbox/Data/E3D_GEMINI_paper/" # Adjust to fit your system
try: # look for saved file including some of the needed types of data    
    dat = xr.open_dataset(path + 'temp3_dat.nc')
    xg = read.grid(path)
except: # make the datafiles from reading GEMINI output
    xg, dat = e3dsecs.gemini_tools.read_gemini(path, timeindex=-1, maph=maph)
    dat.attrs={}
    dat.to_netcdf('/Users/jone/BCSS-DAG Dropbox/Jone Reistad/tmpfiles/temp3_dat.nc')
xgdat = (xg, dat)

# Define grids for 3D model
# Altitude grid
alts_grid = np.concatenate((np.arange(90,140,5),np.arange(140,170,10), 
                            np.arange(170,230,20),np.arange(230,500,50)))
altres = np.diff(alts_grid)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
# Horizontal CS grid
grid, grid_l = e3dsecs.gemini_tools.make_csgrid(xg, maph=maph, h0=alts_grid[0], 
                                    crop_factor=0.2, resolution_factor=0.45, 
                                    extend=extend, dlat = 0.2)

#Grid dimensions
K = alts_grid.shape[0] #Number of vertival layers
I = grid.shape[0] #Number of cells in eta direction, north-south, W dimension
J = grid.shape[1]  #Number of cells in xi direction, east-west, L dimension
KIJ = K*I*J
IJ = I*J

# Open the minimal model
factop = False
vert_profile = None
filename, filename_lompe = e3dsecs.secs3d.make_filenames(grid.projection.position, 
                                inputmode, factop=factop, vert_profile=vert_profile)
m_ = np.load(filename, allow_pickle=True).item()
m_minimal = m_['m']
if (inputmode=='vi') or (inputmode=='vi_ohmslaw'):
    lmodel = lompe.load_model(filename_lompe)

# Open the extended model
factop = True
vert_profile = None
filename, filename_lompe = e3dsecs.secs3d.make_filenames(grid.projection.position, 
                                inputmode, factop=factop, vert_profile=vert_profile)
m_ = np.load(filename, allow_pickle=True).item()
m_extended = m_['m']
if (inputmode=='vi') or (inputmode=='vi_ohmslaw'):
    lmodel = lompe.load_model(filename_lompe)


# Make evaluation grid to test performance
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


# Model performance
Gh, inds = e3dsecs.secs3d.make_JHP(grid, alts_grid, lat_ev.flatten(), lon_ev.flatten(), 
                   alt_ev.flatten(), inputmode='vi', hop='h', 
                   return_indices=True, xgdat=xgdat, lmodel=lmodel)
Gp, inds = e3dsecs.secs3d.make_JHP(grid, alts_grid, lat_ev.flatten(), lon_ev.flatten(), 
                   alt_ev.flatten(), inputmode='vi', hop='p', 
                   return_indices=True, xgdat=xgdat, lmodel=lmodel)

j_hall_minimal = Gh.dot(m_minimal)
j_pedersen_minimal = Gp.dot(m_minimal)
j_hall_extended = Gh.dot(m_extended)
j_pedersen_extended = Gp.dot(m_extended)

# Make ground truth dataset
datadict = e3dsecs.gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)
jh, jp = e3dsecs.gemini_tools.calc_hall_pedersen(datadict, inputmode='vi', 
                                                 lmodel=lmodel, xgdat=xgdat) 
jh = jh[inds==1]
jp = jp[inds==1]

# Make figure that looks at the performance of Hall and Perdersen currents in the two models
clim = 10e-5
fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(jh, j_hall_minimal, label='Minimal Hall current', alpha=0.1)
ax.scatter(jh, j_hall_extended, label='Extended Hall current', alpha=0.1)
ax.set_xlim(1e-5,clim)
ax.set_ylim(1e-5,clim)
ax.legend(frameon=False)

ax = fig.add_subplot(122)
ax.scatter(jp, j_pedersen_minimal, label='Minimal Pedersen current', alpha=0.3)
ax.scatter(jp, j_pedersen_extended, label='Extended Pedersen current', alpha=0.3)
ax.set_xlim(1e-5,clim)
ax.set_ylim(1e-5,clim)
ax.legend(frameon=False)

large = jp > 2e-5
np.corrcoef(jp[large], j_pedersen_minimal[large])
np.corrcoef(jp[large], j_pedersen_extended[large])
np.linalg.norm(jp[large]-j_pedersen_minimal[large])
np.linalg.norm(jp[large]-j_pedersen_extended[large])



