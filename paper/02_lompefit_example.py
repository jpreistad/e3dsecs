#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:39:38 2023

@author: jone

Script to produce figures for the E3D reconstruction paper

This script is designated to produce Figures describing the Lompe fit part of the analysis

"""



import sys
sys.path.append('/Users/jone/BCSS-DAG Dropbox/Jone Reistad/git/DAG/src')
sys.path.append('/Users/jone/BCSS-DAG Dropbox/Jone Reistad')
import git.e3dsecs as e3dsecs
import numpy as np
import apexpy
from gemini3d.grid.convert import geomag2geog, geog2geomag
import gemini3d.read as read
import xarray as xr
import time
import lompe
import matplotlib.pyplot as plt
import secsy as cs


startTime = time.time()


########################################
# Run options
maph            = 200   # height of mapping of ion velocity, in km.
diagnostic      = True  # Wheter to make diagnostic plots
e3doubt_        = True  # Spencers uncertainty 
addnoise        = True  # based on e3doubt covariances
intsec          = 5*60  # Integrationtime in seconds used in E3DOUBT
l1_lompe        = 1e-2  # Regularization parameter for Lompe representation
l2_lompe        = 1e-2  # Regularization parameter for Lompe representation
extend          = 5     # How many padding "frames" used for lompe grid around 3D hor. grid
inputmode       = 'vi'  # How jperp is estimated. Must be either:
                        #   'vi'            : Use samples of ion velocities and (vi-ve)
                        #   'vi_ohmslaw'    : Use samples of ion velocities and Ohms law
                        #   'phitop'        : Use potential at top from GEMINI and (vi-ve)
                        #   'phitop_ohmslaw': Use potential at top from GEMINI and Ohms law
                        #   'jperp'         : Use jperp directly sampled from GEMINI
########################################

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


########################################
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
dlon = 20.26 - sitelon
lats0 = np.array([69.39, 68.44, 68.37])
lons0 = np.array([20.26, 22.48, 19.10])
lats = np.array([sitelat, lats0[1]-dlat, lats0[2]-dlat])
lons = np.array([sitelon, lons0[1]-dlon, lons0[2]-dlon])
datadict = e3dsecs.gemini_tools.sample_eiscat(xg, dat, min_alt=min_alt, max_alt=max_alt, 
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
        datadict = e3dsecs.uncertainty.get_datacov_e3doubt(datadict, intsec=intsec, 
                            transmitter=transmitter, receivers=receivers)
        datadict = e3dsecs.uncertainty.remove_bad(datadict)
        datadict_backup = datadict.copy()
        np.save('./inversion_coefs/datadict_temp.npy', datadict)
    if addnoise:
        datadict = e3dsecs.uncertainty.add_noise(datadict, maph, alternative=True)

########################################
# Step 1: Make v_perp representation at maph if specified by inputmode
filename, filename_lompe = e3dsecs.secs3d.make_filenames(grid.projection.position, inputmode)
if (inputmode=='vi') or (inputmode=='vi_ohmslaw'):
    datadict, lmodel = e3dsecs.gemini_tools.make_lompe(grid_l, datadict, inputmode, 
                            maph, e3doubt_=e3doubt_, l1_lompe=l1_lompe, l2_lompe=l2_lompe, 
                            intsec=intsec, filename_lompe=filename_lompe)
    datadict_backup = datadict.copy()
else:
    lmodel = None

    
########################################################
# PLOTING
fig = plt.figure(figsize=(10,5))
ax1 = plt.subplot2grid((11, 20), (0, 0), rowspan = 10, colspan = 10)
ax2 = plt.subplot2grid((11, 20), (0, 10), rowspan = 11, colspan = 10)
apex = apexpy.Apex(2023)
lompe.visualization.format_ax(ax1, lmodel, apex = apex)
lompe.visualization.plot_quiver(ax1, lmodel, 'convection')
lompe.visualization.plot_potential(ax1, lmodel)
lompe.visualization.plot_datasets(ax1, lmodel, 'convection')
ax1.set_title('Estimate $\\mathbf{v}_{\perp}$ with E3D sampling > ' + str(maph) + 'km')
xi_corners = [grid_l.xi_mesh[0,extend],grid_l.xi_mesh[0,-extend-1],
              grid_l.xi_mesh[0,-extend-1],grid_l.xi_mesh[0,extend],grid_l.xi_mesh[0,extend]]
eta_corners = [grid_l.eta_mesh[extend,0], grid_l.eta_mesh[extend,0], 
               grid_l.eta_mesh[-extend-1,0], grid_l.eta_mesh[-extend-1,0], 
               grid_l.eta_mesh[extend,0]]
ax1.plot(xi_corners, eta_corners, color='green')

# Right panel is showing a scatterplot of the performance of the lompe fit on data from a uniform
# mesh that was not used to make the model inthe first place
# xi_e  = grid.xi[0,1:] - grid.dxi/2 
# eta_e = grid.eta[1:,0]- grid.deta/2
alts__ = alts_grid[15:]-altres[15:]
xi_e  = grid.xi[0,:] - grid.dxi/2 
eta_e = grid.eta[:,0]- grid.deta/2
alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, eta_e, xi_e, indexing='ij')
# eta_ev, xi_ev = np.meshgrid(eta_e, xi_e, indexing='ij')
lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev) 
# alt_ev = np.ones(lon_ev.shape) * maph
datadict = e3dsecs.gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)
datadict['maph'] = maph

axs2 = e3dsecs.diagnostics.scatterplot_lompe(ax2, lmodel, datadict, xgdat)
axs2.set_title('Performance of estimated $\\mathbf{v}_{\perp}$ > ' + str(maph)+ 'km')

arrowax = plt.subplot2grid((11, 20), (10, 0), rowspan = 1, colspan = 10)
arrowax.quiver(.2, .001, 1, 0, scale = 2, scale_units = 'inches')
arrowax.text(0.45,0.7, '1000 m/s', transform=arrowax.transAxes)
arrowax.set_axis_off()

xi_map, eta_map = grid.projection.geo2cube(datadict['mappedglon'], datadict['mappedglat'])
ax1.scatter(xi_map, eta_map, alpha=1, color='green', s=1.4)
plt.tight_layout()
fig.savefig('./plots/lompefit.pdf')

###################################################################

########################
# Make figure that shows the performance of computing currents with Lompe fit vs current from GEMINI
# inputmode = 'vi_ohmslaw'
alts__ = alts_grid[1:]-altres[1:]
xi_e  = grid.xi_mesh[0,1:] - grid.dxi/2 
eta_e = grid.eta_mesh[1:,0]- grid.deta/2
alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, eta_e, xi_e, indexing='ij')
lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev) 
shape = lon_ev.shape
datadict = e3dsecs.gemini_tools.sample_points(xg, dat, lat_ev, lon_ev, alt_ev)
# if e3doubt_:
#     datadict = e3dsecs.uncertainty.get_datacov_e3doubt(datadict, intsec=intsec)
#     datadict = e3dsecs.uncertainty.remove_bad(datadict)
#     datadict_backup = datadict.copy()
#     if addnoise:
#         datadict = e3dsecs.uncertainty.add_noise(datadict, maph)
#         datadict_backup = datadict.copy()
datadict['shape'] = shape
datadict['maph'] = maph
N = datadict['lat'].size
jjj = np.hstack((datadict['jperpu'],-datadict['jperpn'],datadict['jperpe'])) # jperp from GEMINI, in geo (r,theta,phi)
ext_factor = 0
if not 'phitop' in inputmode:
    vperp = e3dsecs.gemini_tools.get_E_from_lmodel(lmodel, datadict, xgdat, returnvperp=True)
    datadict['vperp_electron'] = vperp
inputdict = e3dsecs.gemini_tools.make_inputdict(datadict, grid, alts_grid,
                    inputmode=inputmode, ext_factor=0, hp_from_brekke=False)
jperp_enu = np.vstack((inputdict['jperp'][0,:],inputdict['jperp'][1,:],inputdict['jperp'][2,:])).T

jjj2 = np.zeros((3,N))
jjj2[0,:] = jperp_enu[:,2]
jjj2[1,:] = -jperp_enu[:,1]
jjj2[2,:] = jperp_enu[:,0]

if inputmode=='vi':
    savesuff = '_' + inputmode + '-ve_'    
elif inputmode=='vi_ohmslaw':
    savesuff = '_Ohmslaw_'
else:
    savesuff = '_' + inputmode + '_'
inside = np.array([True]*N)
fig = e3dsecs.diagnostics.compare_input_jperp(datadict, jjj2, inside, 
                savesuff, grid, alts_grid, sliceindex=3, 
                maph=maph, dim=2, param='jperpphi', pdf=True)
fig.savefig('./plots/input_jperp_'+inputmode+'.pdf', dpi=250)
          

###########################################
# Uncertainty plotting
# Investigate the magnitude of the uncertainty in the "observed" jperp with relative
# to jperp in GEMINI
if e3doubt_:
    datadict = datadict_backup.copy()

    # Plotting
    fig = plt.figure(figsize=(12,10))
    ax1 = plt.subplot2grid((20, 21), (0, 0), rowspan = 10, colspan = 10, projection='3d')
    ax2 = plt.subplot2grid((20, 21), (0, 10), rowspan = 10, colspan = 10, projection='3d')
    ax3 = plt.subplot2grid((20, 21), (10, 0), rowspan = 9, colspan = 10)
    ax4 = plt.subplot2grid((20, 21), (10, 10), rowspan = 10, colspan = 10, projection='3d')
    
    datadict['dje'] =  np.sqrt(datadict['cov_jperp'][0,0,:])
    datadict['SNRe'] = np.abs(datadict['jperpe']) / np.sqrt(datadict['cov_jperp'][0,0,:])
    datadict['SNRn'] = np.abs(datadict['jperpn']) / np.sqrt(datadict['cov_jperp'][1,1,:])
    # datadict['sigma_ve'] = np.sqrt(datadict['cov_ve'][0,0,:])
    # datadict['sigma_vn'] = np.sqrt(datadict['cov_ve'][1,1,:])
    # datadict['sigma_vu'] = np.sqrt(datadict['cov_ve'][2,2,:])

    clim = 2e-5  
    ax1 = e3dsecs.diagnostics.plot_analysis_grid(datadict, grid, alts_grid, 
                    1, 1, 1, dipole_lompe=False, data=True, eiscat=True, _d=400, 
                    q='jperpe', cmap='bwr', clim=clim, diverging=True, ax=ax1)
    ax1.set_title('$j_{\perp, \phi}$ from GEMINI', fontsize=16)
    ax1.text(1900,850, 6200, 'A', fontsize=16)
    ax2 = e3dsecs.diagnostics.plot_analysis_grid(datadict, grid, alts_grid, 
                    1, 1, 1, dipole_lompe=False, data=True, eiscat=True, _d=400, 
                    q='dje', cmap='bwr', clim=clim, diverging=True, ax=ax2) 
    ax2.set_title('Uncertainty of $j_{\perp, \phi}$', fontsize=16)
    ax2.text(1900,850, 6200, 'B', fontsize=16)

    #Colorbar upper row
    cbarax = plt.subplot2grid((20,21), (1, 20), rowspan = 8, colspan = 1)
    cmap = plt.cm.bwr
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label('[A/m$^2$]', fontsize=16)
    
    clim=1
    ax4 = e3dsecs.diagnostics.plot_analysis_grid(datadict, grid, alts_grid, 
                    1, 1, 1, dipole_lompe=False, data=True, eiscat=True, _d=400, 
                    q='SNRe', cmap='viridis', clim=clim, diverging=False, ax=ax4) 
    ax4.set_title('SNR of $j_{\perp,\phi}$', fontsize=16)
    ax4.text(1900,850, 6200, 'D', fontsize=16)

    #Colorbar lower row
    cbarax = plt.subplot2grid((20,21), (11, 20), rowspan = 8, colspan = 1)
    cmap = plt.cm.viridis
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0, vmax=clim)
    cb2 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb2.set_label('[SNR]', fontsize=16)
    
    # SNR line plot    
    n =3
    d=-8
    nn = datadict['alts'].size
    ax3.plot(datadict['SNRe'][n*nn+d:n*nn+20+d], datadict['alt'][n*nn+d:n*nn+20+d], label='SNR $j_{\perp,\phi}$, el=$35^\circ$')
    ax3.plot(np.abs(1e5*datadict['jperpe'][n*nn+d:n*nn+20+d]), datadict['alt'][n*nn+d:n*nn+20+d], label='1e5*abs(GEMINI $j_{\perp,\phi}$)')
    ax3.legend(frameon=False)
    ax3.set_xlabel('SNR and 1e5 $j_{\perp,\phi}$')
    ax3.set_ylabel('Alt. [km]')
    ax3.spines[['right', 'top']].set_visible(False)
    ax3.text(0.1,0.9, 'C', fontsize=16, transform=ax3.transAxes)
    fig.savefig('./plots/jperp_uncertainty.pdf',bbox_inches='tight')
