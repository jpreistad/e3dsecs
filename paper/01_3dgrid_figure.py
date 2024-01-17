#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:28:57 2023

@author: jone
Make a figure of an example of a 3D grid

"""

nrec = True
import sys
if nrec:
    sys.path.append('/home/ubuntu/git/e3dsecs')
else:
    sys.path.append('/Users/jone/Dropbox (Personal)/uib/researcher/git/e3dsecs')
from e3dsecs import gemini_tools, coordinates, visualization
import numpy as np
from gemini3d.grid.convert import geomag2geog
import gemini3d.read as read
import xarray as xr
import matplotlib.pyplot as plt

#######################
# Set global attributes and variables
maph = 90
dipole_lompe = False
RE = 6371.2 #Earth radius in km


########################################
# Load GEMINI grid and data
if nrec:
    path = '/home/ubuntu/gemini_data/e3dpaper_datafiles/'
else:
    path = '/Users/jone/Documents/uib_lagacy/gemini_output/'# Adjust to fit your system
try: # look for saved file including some of the needed types of data    
    dat = xr.open_dataset(path + 'gemini_dataset.nc')
    xg = np.load(path + 'gemini_grid.npy', allow_pickle=True).item()
except: # make the datafiles from reading GEMINI output
    xg, dat = gemini_tools.read_gemini(path, timeindex=-1, maph=maph)
    dat.attrs={}
    dat.to_netcdf(path + 'gemini_dataset.nc')
    del xg['glatctr']
    del xg['glonctr']
    np.save(path + 'gemini_grid.npy', xg)
xgdat = (xg, dat)

########################################
# Define grids
# Altitude grid
alts_grid = np.concatenate((np.arange(90,140,5),np.arange(140,170,10), 
                            np.arange(170,230,20),np.arange(230,500,50)))
altres = np.diff(alts_grid)*0.5
altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
# Horizontal CS grid
extend=5
grid, grid_l = gemini_tools.make_csgrid(xg, maph=maph, h0=alts_grid[0], crop_factor=0.2,
                                    resolution_factor=0.45, extend=extend, dlat = 0.2)
#Grid dimensions
K = alts_grid.shape[0] #Number of vertival layers
I = grid.shape[0] #Number of cells in eta direction
J = grid.shape[1]  #Number of cells in xi direction 
KIJ = K*I*J
IJ = I*J
######################################

mlats0 = grid.lat_mesh
mlons0 = grid.lon_mesh
cs_scaling = (grid.xi.max()-grid.xi.min())/grid.L

glats = grid.lat_mesh[np.newaxis]
glons = grid.lon_mesh[np.newaxis]

mlats = mlats0[np.newaxis]
mlons = mlons0[np.newaxis]


# Plot grid and coastlines:
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
visualization.field_aligned_grid(ax, grid, alts_grid, color='green', showlayers=True, 
                       showbase=True, fullbox=False, verticalcorners=False, 
                       dipoleB=dipole_lompe)
kwargs={'linewidth':3}
lat = grid.lat_mesh[-1,:]
lon = grid.lon_mesh[-1,:]
for kk in range(lat.size):
    visualization.plot_field_line(ax, lat[kk], lon[kk], 
                              alts_grid, color='orange', **kwargs, dipoleB=True)

# Plot vertical height scale
alts = np.array([0,100,200,300,400,500])
N = len(alts)
lats = np.array([grid.lat_mesh[-1,0]]*N)
lons = np.array([grid.lon_mesh[-1,0]]*N)
L = grid.L*1e-3
Lres = grid.Lres*1e-3
pos = grid.projection.position
if dipole_lompe:
    lons, lats = geomag2geog(np.radians(lats), np.pi/2 - np.radians(lons)) # returns in degrees
x_, y_, z_ = coordinates.sph_to_car((RE+alts, 90-lats, lons), deg=True)
ax.plot(x_, y_, z_, color='black')
for (ii,aa) in enumerate(alts):
    ax.text(x_[ii], y_[ii], z_[ii], str(aa)+' km', ha='right')
    
# Plot projecte frame on ground level
lats = np.hstack((grid.lat_mesh[:,-1],grid.lat_mesh[-1,:],grid.lat_mesh[:,0][::-1],grid.lat_mesh[0,:]))
lons = np.hstack((grid.lon_mesh[:,-1],grid.lon_mesh[-1,:],grid.lon_mesh[:,0][::-1],grid.lon_mesh[0,:]))
if dipole_lompe:
    lons, lats = geomag2geog(np.radians(lats), np.pi/2 - np.radians(lons)) # returns in degrees
x_, y_, z_ = coordinates.sph_to_car((RE+0, 90-lats, lons), deg=True)
ax.plot(x_, y_, z_, '--', color='black')

# Add E3D sites
sitelat = 67.7 #67.5       # geo lat of transmitter. Skibotn: 69.39
sitelon = 23.       # geo lon of transmitter. Skibotn: 20.27
dlat = 69.39- sitelat
dlon = 20.26 - sitelon
lats0 = np.array([69.39, 68.44, 68.37])
lons0 = np.array([20.26, 22.48, 19.10])
lats = np.array([sitelat, lats0[1]-dlat, lats0[2]-dlat])
lons = np.array([sitelon, lons0[1]-dlon, lons0[2]-dlon])
alts = np.array([0,0,0])
# sites = ['Skibotn', 'Karesuando', 'Kaiseniemi']
sites = ['Tx/Rx1', 'Rx2', 'Rx3']
x, y, z = coordinates.sph_to_car((RE+alts, 90-lats, lons), deg=True)
for i,s in enumerate(sites):
    ax.scatter(x[i], y[i], z[i], label=s, marker='*', s=55)
ax.legend(frameon=False)

# Add grid dimensions
alt = int(alts_grid[0]) - 20
sh = grid.shape
# glon, glat = geomag2geog(np.radians(grid.lon[sh[0]//2,0]), np.radians(90-grid.lat[sh[0]//2,0]))
glon = grid.lon[sh[0]//2+5,0] 
glat = grid.lat[sh[0]//2,0]
x_, y_, z_ = coordinates.sph_to_car((RE+alt, 90-glat, glon), deg=True)
w = grid.W/1000
ax.text(x_[0], y_[0], z_[0], '%3i km' % w, 'y', fontsize=10) 

glon = grid.lon[0,sh[1]//2] 
glat = grid.lat[0,sh[1]//2-1]
x_, y_, z_ = coordinates.sph_to_car((RE+alt, 90-glat, glon), deg=True)
l = grid.L/1000
ax.text(x_[0], y_[0], z_[0], '%3i km' % l, fontsize=10) 

# Fix viewing geometry
lat_ = grid.projection.position[1] # in degrees
lon_ = grid.projection.position[0] # in degrees
L = grid.L*1e-3
Lres = grid.Lres*1e-3
x_, y_, z_ = coordinates.sph_to_car((RE, 90-lat_, lon_), deg=True)
xlim = (x_[0]-L+3*Lres, x_[0]+L-3*Lres) 
ylim = (y_[0]-L+3*Lres, y_[0]+L-3*Lres) 
# zlim = (RE, RE+alts_grid[-1]+1)
zlim = (z_[0], z_[0]+ 0.7*alts_grid[-1])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# ax.set_zlim(5600,6400)
ax.set_zlim(zlim)
ax.view_init(azim=-15, elev=0)
fig.savefig('./plots/3dgrid_figure.pdf', dpi=250,bbox_inches='tight')

