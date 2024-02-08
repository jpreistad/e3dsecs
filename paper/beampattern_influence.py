# Script where I want to test the influence on beam configuration for teh E3DSECS reconstruction

# Set the working directory and make python find the e3dsecs package
import os
notebook_file_path = os.path.realpath('__file__')
os.chdir(os.path.dirname(notebook_file_path))
import sys
parent_directory = os.path.dirname(notebook_file_path)
e3dsecsdir = os.path.abspath(os.path.join(parent_directory, '..'))
sys.path.append(e3dsecsdir)

# Import the packages needed
from e3dsecs import simulation, grid, data, convection, model, diagnostics
import numpy as np
import matplotlib.pyplot as plt
import apexpy
from gemini3d.grid.convert import geog2geomag
import lompe

# Set various options
path            = '/Users/jone/Documents/uib/gemini_output/'
                       # Adjust to fit your system to find the GEMINI files
maph            = 200  # km, mapping altitude. Ions assumed fully magnetized above this alt.
extend          = 5     # How many padding "frames" used for lompe grid around 3D hor. grid
alt_layers      = np.concatenate((np.arange(90,140,5),np.arange(140,170,10), 
                                    np.arange(170,230,20),np.arange(230,500,50)))
                       # The vertical layers to be used
inputmode       = 'vi' # See documentation e.g. in model.py for allowed values. 
                       # Specifies how jperp estimates are obtained
e3doubt_        = True # Wheter to use realistic variances of the observations, and add 
                       # corresponding noise
intsec          = 5*60  # Integrationtime in seconds used in E3DOUBT                       
sitelat         = 67.36# Gegraphic latitude of transmitter site location
sitephi         = 23.0 # Gegraphic longitude of transmitter site location            
min_alt         = 90   # km, start altitude of where to sample along beams
max_alt         = 500  # km, stop altitude of where to sample along beams
dr              = 4    # altitude resolution of sampling
l1_lompe        = 1e-1 # Lompe regularization parameter 1 (model norm)
l2_lompe        = 0    # Lompe regularization parameter 2 (east-west structure)

factop          = True # Provide values of vertical current at top of domain
vert_profile    = 150  # Regularize vertical Hall and Pedersen profile based on 
                       # GEMINI profile. None or # km from where it is applied above
vert_dprofile   = False# Regularize vertical profile of gradient of H/P currents 
                       # based on electron density profile. NOT WORKING
l1              = 10**(-2.3) # Regularization parameter for 3D reconstruction inversion
diagnostic      = False# Wheter to make diagnostic plots
overwrite       = False# Overwrites exisring 3D model coefficient file
crossval        = False# Determine 3D model reg. parameter using cross-validation
beamfile        = 'datadict_spencer_beams.npy' # Specifies which beam sampling file to use


# Make custom beam pattern
# Nrings = 3#18
# __el = np.linspace(57.5,80,Nrings)
# Naz = 10
# __az = np.arange(0,360,360/Naz)
# el = np.tile(__el,Naz)
# _az = []
# _daz = 360/Naz/2
# daz = np.array([0,_daz,0]) # alter the az value every other ring
# # daz = np.array([i*360/Naz/Nrings for i in np.arange(5)])
# for a in __az:
#     _az.append(daz+a)
# az = np.array(_az).flatten()        
# el = np.hstack((el,90)) # add a vertical beam
# az = np.hstack((az,0)) # add a vertical beam
az = np.array([-1.34999651e+02, -1.46310647e+02, -1.61567265e+02,  1.79996264e+02,
        1.61560540e+02,  1.46305474e+02,  1.34995915e+02, -1.23688759e+02,
       -1.34999745e+02, -1.53437096e+02,  1.79994396e+02,  1.53428130e+02,
        1.34994141e+02,  1.23685310e+02, -1.08432278e+02, -1.16562514e+02,
       -1.34999838e+02,  1.79988791e+02,  1.34988631e+02,  1.16558032e+02,
        1.08430037e+02, -8.99959810e+01, -8.99942067e+01, -8.99886946e+01,
        4.49946880e+01,  8.99886989e+01,  8.99942077e+01,  8.99959815e+01,
       -7.15604316e+01, -6.34281417e+01, -4.49887643e+01,  1.12048524e-02,
        4.49999713e+01,  6.34326251e+01,  7.15626734e+01, -5.63055030e+01,
       -4.49942742e+01, -2.65582326e+01,  5.60295200e-03,  2.65671977e+01,
        4.49998777e+01,  5.63089514e+01, -4.49960481e+01, -3.36855276e+01,
       -1.84303052e+01,  3.73540252e-03,  1.84370291e+01,  3.36906999e+01,
        4.49997838e+01])
el = np.array([78.01252633, 79.77334637, 81.00970123, 81.46427611, 81.00935497,
       79.77274322, 78.01176626, 79.77440053, 81.94726543, 83.61886038,
       84.287919  , 83.61836496, 81.94648762, 79.77349589, 81.01156262,
       83.6195233 , 85.95537653, 87.13775169, 85.95458768, 83.61853259,
       81.01052408, 81.46693485, 84.28911023, 87.1380482 , 89.99920747,
       87.13693016, 84.28800012, 81.4658377 , 81.01309353, 83.62037109,
       85.95544326, 87.13724   , 85.95465473, 83.61938066, 81.01205524,
       79.77804781, 81.94986154, 83.62055609, 84.28921743, 83.62006095,
       81.94908413, 79.77714359, 78.01918137, 79.7788174 , 81.01429419,
       81.46853414, 81.01394819, 79.77821467, 78.0184218 ])


# Make objects containing simulation data (sim), grid (gr), sampled data(dat), 
# fitted convection (conv), and the 3d model (mod)

sim = simulation.simulation(path, maph=maph)
gr = grid.grid(sim, alts=alt_layers, extend=extend, dlat=0.3, dlon=0, crop_factor=0.1, resolution_factor=0.75)
dat = data.data(gr, sim, beams=True, sitelat=sitelat, sitephi=sitephi,
                 az=az, el=el, uniformmesh=False, lat_ev=None, lon_ev=None, 
                 alt_ev=None, e3doubt_=e3doubt_, intsec = intsec, min_alt=min_alt, 
                 max_alt=max_alt, dr=dr, tempfile=beamfile)
conv = convection.convection(gr.grid_l, dat, inputmode=inputmode, e3doubt_=e3doubt_, l1_lompe=l1_lompe, 
                 l2_lompe=l2_lompe, maph=maph)
mod = model.model(gr, inputmode=inputmode, 
                           factop=factop, vert_profile=vert_profile, 
                           vert_dprofile=vert_dprofile, l1=l1, 
                           diagnostic=diagnostic, overwrite=overwrite, 
                           crossval=crossval, e3doubt_=e3doubt_)

############################
# Plot the error along beams
if not e3doubt_:
    print('This analysis can not be done unless E3DOUBT is used')
    breakpoint()    
fig = plt.figure(figsize=(12,10))
ax1 = plt.subplot2grid((20, 21), (0, 0), rowspan = 10, colspan = 10, projection='3d')
ax2 = plt.subplot2grid((20, 21), (0, 10), rowspan = 10, colspan = 10, projection='3d')
ax3 = plt.subplot2grid((20, 21), (10, 0), rowspan = 9, colspan = 10)
ax4 = plt.subplot2grid((20, 21), (10, 10), rowspan = 10, colspan = 10, projection='3d')

dat.dje =  np.sqrt(dat.cov_jperp[0,0,:])
dat.djn =  np.sqrt(dat.cov_jperp[1,1,:])
dat.SNRe = np.abs(dat.jperpe) / np.sqrt(dat.cov_jperp[0,0,:])
dat.SNRn = np.abs(dat.jperpn) / np.sqrt(dat.cov_jperp[1,1,:])
# datadict['sigma_ve'] = np.sqrt(datadict['cov_ve'][0,0,:])
# datadict['sigma_vn'] = np.sqrt(datadict['cov_ve'][1,1,:])
# datadict['sigma_vu'] = np.sqrt(datadict['cov_ve'][2,2,:])

clim = 2e-5  
ax1 = diagnostics.plot_analysis_grid(dat.__dict__, gr.grid, gr.alts_grid, 
                1, 1, 1, dipole_lompe=False, data=True, eiscat=True, _d=400, 
                q='jperpe', cmap='bwr', clim=clim, diverging=True, ax=ax1)
ax1.set_title(r'$j_{\perp, \phi}$ from GEMINI', fontsize=16)
ax1.text(1900,850, 6200, 'A', fontsize=16)
ax2 = diagnostics.plot_analysis_grid(dat.__dict__, gr.grid, gr.alts_grid, 
                1, 1, 1, dipole_lompe=False, data=True, eiscat=True, _d=400, 
                q='dje', cmap='bwr', clim=clim, diverging=True, ax=ax2) 
ax2.set_title(r'Uncertainty of $j_{\perp, \phi}$', fontsize=16)
ax2.text(1900,850, 6200, 'B', fontsize=16)

#Colorbar upper row
cbarax = plt.subplot2grid((20,21), (1, 20), rowspan = 8, colspan = 1)
cmap = plt.cm.bwr
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=-clim, vmax=clim)
cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                            norm=norm,
                            orientation='vertical')
cb1.set_label(r'[A/m$^2$]', fontsize=16)

clim=1
ax4 = diagnostics.plot_analysis_grid(dat.__dict__, gr.grid, gr.alts_grid, 
                1, 1, 1, dipole_lompe=False, data=True, eiscat=True, _d=400, 
                q='SNRe', cmap='pink_r', clim=clim, diverging=False, ax=ax4) 
ax4.set_title(r'SNR of $j_{\perp,\phi}$', fontsize=16)
ax4.text(1900,850, 6200, 'D', fontsize=16)

#Colorbar lower row
cbarax = plt.subplot2grid((20,21), (11, 20), rowspan = 8, colspan = 1)
cmap = plt.cm.pink_r
import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=clim)
cb2 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                            norm=norm,
                            orientation='vertical')
cb2.set_label('[SNR]', fontsize=16)

# SNR line plot
use = np.where((dat.az_all==288) & (dat.el_all==80))[0]
N = dat.el_all.size
_az = dat.az_all[use[0]]
_el = dat.el_all[use[0]]
# _az = datadict['az_all'][N-nn]#[n*nn+d]
# _el = datadict['el_all'][N-nn]#[n*nn+d]    
s_str = r'SNR $j_{\perp,\phi}$, el=$%3.1f^\circ$, az=$%3.1f^\circ$' % (_el,_az)
ax3.plot(dat.SNRe[use[0]:use[0]+20], dat.alt[use[0]:use[0]+20], label=s_str)
# ax3.plot(datadict['SNRe'][N-nn:N-nn+20], datadict['alt'][N-nn:N-nn+20], label=s_str)
ax3.plot(np.abs(1e5*dat.jperpe[use[0]:use[0]+20]), dat.alt[use[0]:use[0]+20], label=r'1e5*abs(GEMINI $j_{\perp,\phi}$)')
# ax3.plot(np.abs(1e5*datadict['jperpe'][N-nn:N-nn+20]), datadict['alt'][N-nn:N-nn+20], label='1e5*abs(GEMINI $j_{\perp,\phi}$)')
ax3.legend(frameon=False)
ax3.set_xlabel('SNR and 1e5 $j_{\perp,\phi}$')
ax3.set_ylabel('Alt. [km]')
ax3.spines[['right', 'top']].set_visible(False)
ax3.text(0.1,0.9, 'C', fontsize=16, transform=ax3.transAxes)


##################################################
# Do the 3D inversion if file not exists
if not mod.file_exists or overwrite:
    mod.run_inversion(gr, conv, sim)
    
### Analysis of performance  
# Load inversion results. To save time, we reload the inversion file here, so that the inversion above dont have to be called
m_ = np.load(mod.filename, allow_pickle=True).item()
m = m_['m']
if (inputmode=='vi') or (inputmode=='vi_ohmslaw'):
    lmodel = conv.lmodel
    
# Make new evaluation grid that does not contain locations outside 3D grid
alts__ = gr.alts_grid[1:]-gr.altres[1:]
xi_e  = gr.grid.xi[0,1:] - gr.grid.dxi/2 
eta_e = gr.grid.eta[1:,0]- gr.grid.deta/2
alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, eta_e, xi_e, indexing='ij')
lon_ev, lat_ev = gr.grid.projection.cube2geo(xi_ev, eta_ev)  
shape = lon_ev.shape

# First, get the true values from GEMINI on the evaluation grid, and compute what the input values to the 3D inversion 
# (jperp) would have been given the lmodel at the evaluating locations, to use for performance evaluation
_data = data.data(gr, sim, beams=False, uniformmesh=True, 
                  lat_ev=lat_ev, lon_ev=lon_ev, alt_ev=alt_ev, 
                  e3doubt_=False)
if not 'phitop' in inputmode:
    vperp = conv.get_E_from_lmodel(sim, _data, returnvperp=True)
    _data.vperp_electron = vperp
inputdict = mod.make_inputdict(_data, gr, inputmode=inputmode, ext_factor=-1, 
                    hp_from_brekke=False)    
d = np.hstack((inputdict['jperp'][2,:], -inputdict['jperp'][1,:], 
               inputdict['jperp'][0,:])) # (r, theta, phi components)

# Evaluate the 3D model of current denisty on the uniform evaluation grid
#Make G to evaluate for full j based on the model made above
G = mod.make_G(gr.grid, m_['alts_grid'], lat_ev, lon_ev, alt_ev, ext_factor=0)
full_j = G.dot(m)
#Get the jperp and fac of the ful_j expressed by the model, to be compared to input
br, btheta, bphi = mod.make_b_unitvectors(_data.Bu, 
                -_data.Bn, _data.Be)
N = br.size
B = mod.make_B(br, btheta, bphi)
P = mod.make_P(N)
j_perp = P.T.dot(B.dot(P.dot(full_j)))
jpar = np.sum(np.array([full_j[0:N], full_j[N:2*N], full_j[2*N:3*N]]) * 
                np.array([br, btheta, bphi]), axis=0)

###################
# Plotting
clim = 2e-5
meshgrid = (alt_ev,lat_ev,lon_ev)
fig = diagnostics.performance_plot(full_j, meshgrid, _data, gr.grid, gr.alts_grid, 
                            clim=clim, cut=['j','k','k'], ind=[5,2,18], transmitter=conv.data.transmitter,
                            receivers=conv.data.receivers)
