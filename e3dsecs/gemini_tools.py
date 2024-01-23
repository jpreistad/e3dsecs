#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 02:19:17 2023

@author: jone

Tools for working with output of GEMINI model. Tools are developed to enable 
benchmarking the secs3d representation of the current density field.

"""


from gemini3d.grid.convert import unitvecs_geographic
import numpy as np
import lompe
from lompe.data_tools.dataloader import getbearing
import great_circle_calculator.great_circle_calculator as gcc
import xarray as xr
from gemini3d.grid.gridmodeldata import model2pointsgeogcoords, geog2dipole
from gemini3d.grid.convert import geomag2geog
import pandas as pd
from secsy import cubedsphere
import gemini3d.read as read
import dipole
import scipy
import matplotlib.pyplot as plt
import h5py

RE = 6371.2 #Earth radius in km

from . import secs3d
from . import coordinates
from . import uncertainty


def read_gemini(path, timeindex=-1, maph=200):
    """
    Function that reads output files from GEMINI simulations, and return an 
    xarray dataset, including some additional quantities estiated in this function.

    Parameters
    ----------
    path : str
        Path to data files.
    timeindex : int, optional
        integer index describing which timestep in simulation to return. The 
        default is -1 corresponding to the last timestep.
    maph : int or float, optional
        The height to map ion velocities to. Default is 100 km.
    

    Returns
    -------
    xg, dat : tuple
        First element is GEMINI grid dictionary, second element is an xarray 
        Dataset of the simulation dataset at timeindex.
    """    
    var = ["v1", "v2", "v3", "Phi", "J1", "J2", "J3", "ne" ,"Te", "Ti"]
    cfg = read.config(path)
    xg = read.grid(path)
    times = cfg["time"][timeindex:]
    t = times[0]
    dat = read.frame(path, t, var=var)
    sshh = dat.v1.values.shape
    dat = calc_conductivities(xg, dat)
    dat = compute_enu_components(xg, dat)            
    
    # Do mapping of observation to maph altitude
    dip = dipole.Dipole(dipole_pole=((90-11),289)) # These locations found in pygemini doc.
    d1, d2, d3, _1, _2, _3 = dip.get_apex_base_vectors_geo(xg['glon'].flatten(), 
                        xg['glat'].flatten(), xg['alt'].flatten()*1e-3+RE, R=RE)
    v = np.vstack((dat.ve.values.flatten(),dat.vn.values.flatten(),dat.vu.values.flatten()))
    #Calculate the quantities that is constant along the field-lines
    ve1 = (d1[0,:]*v[0,:] + d1[1,:]*v[1,:] + d1[2,:]*v[2,:]).reshape(sshh)
    ve2 = (d2[0,:]*v[0,:] + d2[1,:]*v[1,:] + d2[2,:]*v[2,:]).reshape(sshh)
    # Calculate the mapped geographic locations. Map from observed location (2) 
    # to the maph height (1) using dipole formula
    r_2 = xg['alt']*1e-3 + RE
    r_1 = np.ones(r_2.shape)*(RE+maph)
    colat_1 = np.arcsin(np.sin(xg['theta']) * np.sqrt(r_1/r_2))
    mlon_1 = xg['phi']# in radians
    mappedglon, mappedglat = geomag2geog(mlon_1, colat_1) #returns in degrees
    # Calculate basis vectors at the mapped locations
    _1, _2, _3, e1, e2, e3 = dip.get_apex_base_vectors_geo(mappedglon.flatten(), 
                        mappedglat.flatten(), r_1.flatten(), R=RE)
    #Calculate the mapped velocity using eq 4.17 in Richmond 1995. geographic components, ENU            
    vperpmappede = (ve1.flatten()*e1[0,:] + ve2.flatten()*e2[0,:]).reshape(sshh)
    vperpmappedn = (ve1.flatten()*e1[1,:] + ve2.flatten()*e2[1,:]).reshape(sshh)
    vperpmappedu = (ve1.flatten()*e1[2,:] + ve2.flatten()*e2[2,:]).reshape(sshh)
    dat['mappedglat'] = xr.DataArray(mappedglat, dims=('x1','x2','x3'))
    dat['mappedglon'] = xr.DataArray(mappedglon, dims=('x1','x2','x3'))
    dat['vperpmappede'] = xr.DataArray(vperpmappede, dims=('x1','x2','x3'))
    dat['vperpmappedn'] = xr.DataArray(vperpmappedn, dims=('x1','x2','x3'))
    dat['vperpmappedu'] = xr.DataArray(vperpmappedu, dims=('x1','x2','x3'))
 
    return xg, dat


def gemini_gradient(xg, dat, q='Phitop'):
    '''
    Compute the gradient of a scalar field, e.g. electric potential defined in GEMINI's
    curvlinear coordinates. Input arrays must be 2D or 3D arrays

    Parameters
    ----------
    xg : dict
        GEMINI grid dict
    dat : xarray dataset
        Containing GEMINI output variables
    q : string
        Pointing to the scalar field in dat to be differentiated
    Returns
    -------
    dat : xarray dataset.
        Containting the components of the gradient
    '''
    
    RE = 6371.2e3 #Earth radius in m

    #Metric factors defined in eqs 114-116 in GEMINI documentation
    # h1 = xg['r']**3/(RE**2*np.sqrt(1+3*(np.cos(xg['theta']))**2))
    h2 = RE*(np.sin(xg['theta']))**3/np.sqrt(1+3*(np.cos(xg['theta']))**2)
    h3 = xg['r'] * np.sin(xg['theta'])
    
    ndim = len(dat.Phitop.shape)

    if ndim == 2:
        x2x2, x3x3 = np.meshgrid(xg['x2i'][1:], xg['x3i'][1:], indexing='ij')
        q2 = 1/h2 * coordinates.diff2d(x2x2, dat[q].values, axis=0)
        q3 = 1/h3 * coordinates.diff2d(x3x3, dat[q].values, axis=1)
        dat['grad'+q+'_x2'] = xr.DataArray(q2, dims=('x1','x2','x3'))
        dat['grad'+q+'_x3'] = xr.DataArray(q3, dims=('x1','x2','x3'))
    if ndim ==3:
       print('Not implemented')
       print(1/0)
       
    return dat


def calc_conductivities(xg, dat):
    '''
    Function that calculates the Hall and Pedersen conductivities used in GEMINI
    based on the currents and E-fields provided in the output files. This function
    also stores the E-field values in the native GEMINI grid in the dat structure.

    Parameters
    ----------
    xg : dict
        GEMINI grid dict
    dat : xarray dataset
        Containing GEMINI output variables

    Returns
    -------
    dat : xarray dataset
        The updated data object
    '''
    
    dat = gemini_gradient(xg, dat, q='Phitop')
    shape = dat.J1.shape
    E2 = -dat.gradPhitop_x2
    E3 = -dat.gradPhitop_x3
    Emag = np.sqrt(E2**2 + E3**2)
    ehat = np.stack((np.zeros(shape), E2/Emag, E3/Emag))
    j_ped_mag = dat.J2 * ehat[1,:,:,:] + dat.J3 * ehat[2,:,:,:]
    sp = j_ped_mag / Emag
    j_hall_mag = -dat.J2*ehat[2,:,:,:] + dat.J3*ehat[1,:,:,:]
    sh = j_hall_mag / Emag
    
    dat['sp'] = xr.DataArray(sp, dims=('x1','x2','x3'))
    dat['sh'] = xr.DataArray(sh, dims=('x1','x2','x3'))
    dat['E1'] = xr.DataArray(np.zeros(E2.shape), dims=('x1','x2','x3'))
    dat['E2'] = xr.DataArray(E2, dims=('x1','x2','x3'))
    dat['E3'] = xr.DataArray(E3, dims=('x1','x2','x3'))
    
    return dat

def compute_enu_components(xg, dat):
    """
    Add ENU components (geographic) of V, J and B to xarray dataset

    Parameters
    ----------
    xg : GEMINI grid object
        Read by gemini read function from config file
    dat : GEMINI data object (xarray dataset)
        Containing GEMINI output data for specified variables at specified time.

    Returns
    -------
    xarray dataset where the geographic ENU components of V, J and B is added.

    """
    
    #Convert velocity and current to grographic components, use ENU notation
    vu, ve, vn = model_vec2geo_vec(xg, dat, param='v')
    vperpu, vperpe, vperpn = model_vec2geo_vec(xg, dat, param='v', perp=True)
    jperpu, jperpe, jperpn = model_vec2geo_vec(xg, dat, param='J', perp=True)
    ju, je, jn = model_vec2geo_vec(xg, dat, param='J')
    
    #B vectors from model output, project on geo ENU frame
    [egalt,eglon,eglat]=unitvecs_geographic(xg)    
    Be = np.sum(xg["e1"]*eglon*xg['Bmag'][...,np.newaxis],3)
    Bn = np.sum(xg["e1"]*eglat*xg['Bmag'][...,np.newaxis],3)
    Bu = np.sum(xg["e1"]*egalt*xg['Bmag'][...,np.newaxis],3)
    
    # E-field
    Eu, Ee, En = model_vec2geo_vec(xg, dat, param='E', perp=True)

    # Add to dat structure
    dat['Ee'] = Ee
    dat['En'] = En
    dat['Eu'] = Eu
    dat['ve'] = ve
    dat['vn'] = vn
    dat['vu'] = vu
    dat['vperpe'] = vperpe
    dat['vperpn'] = vperpn
    dat['vperpu'] = vperpu
    dat['jperpe'] = jperpe
    dat['jperpn'] = jperpn
    dat['jperpu'] = jperpu
    dat['je'] = je
    dat['jn'] = jn
    dat['ju'] = ju
    dat['Be'] = xr.DataArray(Be, dims=('x1','x2','x3'))
    dat['Bn'] = xr.DataArray(Bn, dims=('x1','x2','x3'))
    dat['Bu'] = xr.DataArray(Bu, dims=('x1','x2','x3'))
    
    return dat


def model_vec2geo_vec(xg, dat, param='v', perp=False):
    '''
    Function to convert model vector components into geographic conponents. 
    Code provided by M. Zettergren, and put into this function by JPR.

    Parameters
    ----------
    xg : GEMINI grid object
        contain all info about the grid used in the simulation
    dat : GAMERA data object (xarray) at specific time
        As returned from the read.frame() function.
    param : 'str'
        'v' (default) or 'J', refering to velocity or current density
    perp : Boolean
        Specifies if only the perpendicular component (2 and 3) of param is to
        be projected to (r, theta, phi) components. Default is False.
        
    Returns
    -------
    (radial, east, north) geographic components of velocity

    '''
     
    [egalt,eglon,eglat]=unitvecs_geographic(xg)     #up, east, north
    #^ returns a set of geographic unit vectors on xg; these are in ECEF geomag
    # comps like all other unit vectors in xg

    # each of the components in models basis projected onto geographic unit 
    # vectors
    if perp:
        vgalt=(np.sum(xg["e2"]*egalt,3)*dat[param+"2"] + 
               np.sum(xg["e3"]*egalt,3)*dat[param+"3"] )
        vglat=(np.sum(xg["e2"]*eglat,3)*dat[param+"2"] +
               np.sum(xg["e3"]*eglat,3)*dat[param+"3"] )
        vglon=(np.sum(xg["e2"]*eglon,3)*dat[param+"2"] + 
               np.sum(xg["e3"]*eglon,3)*dat[param+"3"] )
    else:
        vgalt=( np.sum(xg["e1"]*egalt,3)*dat[param+"1"] + 
               np.sum(xg["e2"]*egalt,3)*dat[param+"2"] + 
               np.sum(xg["e3"]*egalt,3)*dat[param+"3"] )
        vglat=( np.sum(xg["e1"]*eglat,3)*dat[param+"1"] + 
               np.sum(xg["e2"]*eglat,3)*dat[param+"2"] +
               np.sum(xg["e3"]*eglat,3)*dat[param+"3"] )
        vglon=( np.sum(xg["e1"]*eglon,3)*dat[param+"1"] + 
               np.sum(xg["e2"]*eglon,3)*dat[param+"2"] + 
               np.sum(xg["e3"]*eglon,3)*dat[param+"3"] )
    
    return [vgalt, vglon, vglat] # (up, east, north)


def make_csgrid(xg, maph = 200, h0 = 90, crop_factor = 0.6, resolution_factor = 0.5, 
                extend = 1, dlon = 0., dlat = 0., asymres=1,
                extend_ew=1):
    '''
    Put a CubedSphere grid inside GEMINI model domain at specified height. The 
    CS grid is made in geographic coordinates, like everything else in this library.
    This means that one should not use the Lompe representation to express the 
    E-field, since the IGRF field would then be used. This is not a problem as
    long as only the velocity field is represented with lompe, estimated only
    with velocity as input.
    
    The grid is oriented along the dimensions of the GEMINI naive grid, so that
    its east-west extent will be approx magnetic east west.

    Parameters
    ----------
    xg : GEMINI grid object
        contain all info about the grid used in GEMINI simulation
    maph : float or int
        Height in km of where to make the CS grid to use with Lompe.
    h0 : float or int    
        Height in km of where to make the CS grid to use with 3D reconstruction. 
        Should be at the bottom layer of the 3D grid.
    crop_factor : float
        How much to reduce the CS grid compared to GEMINI grid
    resolution_factor : float
        How much to reduce the spatial resolution compared to GEMINI. 0.5 will 
        double the spacing of grid cells
    extend : int
        How many secs poles to pad with on each side compared in lompe grid compared
        to the inner grid to be used in 3D representation
    dlon : float
        How much to shift the centre longitude of grid, in degrees. Default is 0
    dlat : float
        How much th shift the centre latitude of grid, in degrees. Default is 0    
    asymres : int or float
        Determine if the grid cells should be larger in one direction, compared
        to what the extent of the grid prescribes. Leads to grid cells
        with sizes Wres = asymres*Wres. Hence, asymres>1 leads to grid cells that 
        is relatively larger in east-west grid direction
    extend_ew : int or float
        Factor that grid.L is multiplied with. Will extend magnetic east-west extent
        relative to north-south extent of the grid.
    
    Returns
    -------
    Tuple containing two CS grid object, first the grid at the base altitude to
    be used for the 3D reconstruction, second is the grid used in the Lompe 
    representation of the velocity field at maph height, which is the sane grid
    but padded with # of frames determined by extend keyword.

    '''

    #GEMINI grid is field-aligned and orthogonal. These limits are thus not 
    # exactly the limits at the fixed height
    # The crop factor is used to make the CS grid fit inside the model domain. 
    # Should be changed to something more robust and intuitive in the future
    
    # Find index of height matching the desired height in centre of grid
    dims = xg['lx']
    diff = xg['alt'][:,dims[1]//2, dims[2]//2] - h0*1e3
    ii = np.argmin(np.abs(diff))
   
    #Find the orientation of the model grid
    x0 = (xg['glat'][ii,dims[1]//2,dims[2]//2], 
          xg['glon'][ii,dims[1]//2,dims[2]//2])
    x1 = (xg['glat'][ii,1+dims[1]//2,dims[2]//2], 
          xg['glon'][ii,1+dims[1]//2,dims[2]//2])
    orientation = np.degrees(getbearing(np.array([x0[0]]), np.array([x0[1]]), 
                                        np.array([x1[0]]), np.array([x1[1]])))
    
    #Centre location of CS grid
    position = (xg['glon'][ii,dims[1]//6,dims[2]//2] + dlon, 
                xg['glat'][ii,(dims[1]//6),dims[2]//2] + dlat) #Added the //6 hack             

    #Dimensions of CS grid
    p0 = (xg['glon'][ii,0,dims[2]//2],xg['glat'][ii,0,dims[2]//2])
    p0lon = np.rad2deg(np.arctan2(np.sin(np.deg2rad(p0[0])), np.cos(np.deg2rad(p0[0]))))
    p0 = (p0lon,p0[1])
    p1 = (xg['glon'][ii,-1,dims[2]//2],xg['glat'][ii,-1,dims[2]//2])
    p1lon = np.rad2deg(np.arctan2(np.sin(np.deg2rad(p1[0])), np.cos(np.deg2rad(p1[0]))))
    p1 = (p1lon,p1[1])
    # d2: distance in dimension 2 (magntic northsouth ish)
    d2 = gcc.distance_between_points(p0, p1) 
    W = d2 * crop_factor
    Wres = d2/dims[1] / resolution_factor # For secs grid (grid)
    p0 = (xg['glon'][ii,dims[1]//2,0],xg['glat'][ii,dims[1]//2,0])
    p0lon = np.rad2deg(np.arctan2(np.sin(np.deg2rad(p0[0])), np.cos(np.deg2rad(p0[0]))))
    p0 = (p0lon,p0[1])
    p1 = (xg['glon'][ii,dims[1]//2,-1],xg['glat'][ii,dims[1]//2,-1])
    p1lon = np.rad2deg(np.arctan2(np.sin(np.deg2rad(p1[0])), np.cos(np.deg2rad(p1[0]))))
    p1 = (p1lon,p1[1])
    # d3 is distance in dimension 3 (magntic east ish)    
    d3 = gcc.distance_between_points(p0, p1) 
    L = d3 * crop_factor * extend_ew
    Lres = d3*asymres/dims[2] / resolution_factor # For secs grid (grid)

    #Make CS grid object
    #Base grid for 3D model  
    grid = cubedsphere.CSgrid(cubedsphere.CSprojection(position, 
                -orientation[0]), L, W, Lres, Wres, R = (RE+h0)*1e3)
    
    # Extended grid for Lompe fit of E-filed   
    xi_e  = np.hstack((grid.xi_mesh[0,0]-np.flip([i*grid.dxi for i in range(1,extend+1)]),
                       grid.xi_mesh[0], grid.xi_mesh[0,-1] + np.array([i*grid.dxi 
                        for i in range(1,extend+1)]) ))
    eta_e = np.hstack((grid.eta_mesh[0,0]-np.flip([i*grid.deta for i in range(1,extend+1)]),
                       grid.eta_mesh[:, 0], grid.eta_mesh[-1,   0] + np.array([i*grid.deta 
                        for i in range(1,extend+1)]) )) 
    
    # outer grid padded with extend number of extra frames, intended to use for
    # the Lompe fit to avoid boundary effects when evaluated iniside the inner grid
    lgrid = cubedsphere.CSgrid(cubedsphere.CSprojection(grid.projection.position,
                grid.projection.orientation), grid.L + extend*2*grid.Lres, 
                grid.W + extend*2*grid.Wres, grid.Lres, grid.Wres, 
                edges = (xi_e, eta_e), R = (RE+maph)*1e3)

    return grid, lgrid



def sample_eiscat(xg, dat, dr = 1, az=None, el=None, sitelat=67.36, sitephi=23., 
                  min_alt=100, max_alt=500, estimate_E_field=False):
    """
    Function that sample from GEMINI output (linear interpolation from its 
    native grid) along prescribed beams (az,el).

    Parameters
    ----------
    xg : dict
        Gemini grid object
    dat : xarrat Dataset
        GEMINI data.
    dr : int or float, optional
        altitude sampling separation. This does not take into account elevation 
        angle, which the real EISCAT does. Unit is km. The default is 1 km.
    az : array like, optional
        array of azimuth angle, in degrees for each beam. Positive towards east.
        The default is None, using a beam pattern found in an E3D presentation 
        by Ogawa.
    el : array like, optional
        array of elevation angle, in degrees for each beam. Positive above horizon.
        The default is None, using a beam pattern found in an E3D presentation 
        by Ogawa.
    sitelat : float, optional
        geographic latitude of radar site. The default is 67.36.
    sitephi : float, optional
        geographic longitude of radar site. The default is 23.
    min_alt : int or float, optional
        lowest altitude for sampling, in km. Default is 100 km.
    max_alt : int or float, optional
        highest altitude for sampling, in km. Default is 500 km.  
    estimate_E_field : Boolean, optional
        Specify if the E-field should be represented (e.g. using Lompe) from 
        ion velocities sampled from the model above some height, and used to 
        specify perp velocity of electrons when j_perp is caluculated using
        j_perp = ne(v_i_perp - v_e_perp). This is how we intend to use the 
        EISCAT data. Default is False.
    Returns
    -------
    datadict : dict
        Containing all sampled quantities and locations.

    """

    alt_grid = np.arange(min_alt, max_alt, dr)

    if (az is None ) | (el is None):
        # Make a lot of beams
        Nrings = 5#18
        __el = np.linspace(35,80,Nrings)
        Naz = 10
        __az = np.arange(0,360,360/Naz)
        el = np.tile(__el,Naz)
        _az = []
        _daz = 360/Naz/2
        daz = np.array([0,_daz,0,_daz,0]) # alter the az value every other ring
        # daz = np.array([i*360/Naz/Nrings for i in np.arange(5)])
        for a in __az:
            _az.append(daz+a)
        az = np.array(_az).flatten()  

        # Nrings = 5#18
        # __el = np.linspace(35,80,Nrings)
        # Naz = 10
        # __az = np.linspace(0,340,Naz)
        # _az, _el = np.meshgrid(__az, __el, indexing='ij')
        # az = _az.flatten()
        # el = _el.flatten()          
        
        # add a vertical beam
        el = np.hstack((el,90))
        az = np.hstack((az,0))

        # # Implement 27 beam (monostatic config) configuration as sketched by Ogawa (2021)
        # el1 = np.array([64,61,60,58,57,55,54,54,57,59,61,61])
        # az1 = np.array([0,35,69,101,130,156,180,204,231,258,288,323])
        # el2 = np.array([30,30,30,30,30,30,30,30,30,30,30,30])
        # az2 = np.array([0,30,60,90,120,150,180,210,240,270,300,330])
        # el3 = np.array([66,77.8,90])
        # az3 = np.array([180,180,180])
        # el = np.hstack((el1,el2,el3)) #deg
        # az = np.hstack((az1,az2,az3)) #deg

    sitetheta = 90-sitelat
    O_sph = np.array([RE, 0, 0]) #site location vector
    R = coordinates.sph2car(sitetheta,sitephi) #sph2ecef rotation matrix cor vector components
    O_ecef = R.dot(O_sph) #Site location vecor in ecef
 
    # Initiate arrays/lists to hold the locations of where to sample from
    r = []
    theta = []
    phi = []
    lx = []
    ly = []
    lz = []
    __el = []
    __az = []
    
    # Find the intersection points of the beams and the given altitudes
    for i in range(len(az)):
        uhat_sph = coordinates.get_uhat(az[i],el[i]) # beam direction unit vector in sph
        uhat_ecef = R.dot(uhat_sph) #Beam direction unit vector in ecef
        lx.append([uhat_ecef[0]]*alt_grid.size)
        ly.append([uhat_ecef[1]]*alt_grid.size)
        lz.append([uhat_ecef[2]]*alt_grid.size)
        #Line sphere intersecion formula from 
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        dot = uhat_ecef.dot(O_ecef)# RE * np.cos(np.radians(90+el[i]))
        root = np.sqrt(dot**2 - (RE**2-(RE+alt_grid)**2))
        #d0 = -dot - root # through earth
        d = -dot + root # the distances from site corresponding to selected altitudes
        deltas = (uhat_ecef.reshape(3,1) * d) #Vector to be added to O_ecef
        pos = O_ecef[:,np.newaxis] + deltas
        r_, theta_, phi_ = coordinates.car2sph(pos[0,:], pos[1,:], pos[2,:])
        r.append(r_)
        theta.append(theta_)
        phi.append(phi_)
        __el.append(np.ones(r_.size)*el[i])
        __az.append(np.ones(r_.size)*az[i])
    poss = np.vstack((np.array(r).flatten(), np.array(theta).flatten(),
                      np.array(phi).flatten())).T
    
    # Now we can sample from GEMINI at the identified locations
    je = model2pointsgeogcoords(xg, dat['je'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    jn = model2pointsgeogcoords(xg, dat['jn'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    ju = model2pointsgeogcoords(xg, dat['ju'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    Be = model2pointsgeogcoords(xg, dat['Be'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    Bn = model2pointsgeogcoords(xg, dat['Bn'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    Bu = model2pointsgeogcoords(xg, dat['Bu'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])

    # Initialize the datadict object
    datadict = {'lat':90-poss[:,1], 'lon':poss[:,2], 'alt':poss[:,0]-RE, 
                'je':je, 'jn':jn, 'ju':ju, 'Be':Be, 'Bn':Bn, 'Bu':Bu, 
                'az':np.array(az), 'el':np.array(el), 'alts':alt_grid, 
                'az_all':np.array(__az).flatten(), 
                'el_all':np.array(__el).flatten()}
    
    # Here we add some additonal quantities
    datadict['vperpe'] = model2pointsgeogcoords(xg, dat['vperpe'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['vperpn'] = model2pointsgeogcoords(xg, dat['vperpn'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['vperpu'] = model2pointsgeogcoords(xg, dat['vperpu'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['jperpe'] = model2pointsgeogcoords(xg, dat['jperpe'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['jperpn'] = model2pointsgeogcoords(xg, dat['jperpn'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['jperpu'] = model2pointsgeogcoords(xg, dat['jperpu'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['sp'] = model2pointsgeogcoords(xg, dat['sp'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['sh'] = model2pointsgeogcoords(xg, dat['sh'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['Ee'] = model2pointsgeogcoords(xg, dat['Ee'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['En'] = model2pointsgeogcoords(xg, dat['En'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['Eu'] = model2pointsgeogcoords(xg, dat['Eu'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['ne'] = model2pointsgeogcoords(xg, dat['ne'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['Te'] = model2pointsgeogcoords(xg, dat['Te'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['Ti'] = model2pointsgeogcoords(xg, dat['Ti'], 
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['vperpmappede'] = model2pointsgeogcoords(xg, dat['vperpmappede'],
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['vperpmappedn'] = model2pointsgeogcoords(xg, dat['vperpmappedn'],
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['vperpmappedu'] = model2pointsgeogcoords(xg, dat['vperpmappedu'],
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['mappedglat'] = model2pointsgeogcoords(xg, dat['mappedglat'],
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
    datadict['mappedglon'] = model2pointsgeogcoords(xg, dat['mappedglon'],
                            (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])              
    datadict['ve'] = model2pointsgeogcoords(xg, dat['ve'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    datadict['vn'] = model2pointsgeogcoords(xg, dat['vn'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    datadict['vu'] = model2pointsgeogcoords(xg, dat['vu'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    datadict['v1'] = model2pointsgeogcoords(xg, dat['v1'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    datadict['v2'] = model2pointsgeogcoords(xg, dat['v2'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])
    datadict['v3'] = model2pointsgeogcoords(xg, dat['v3'],(poss[:,0]-RE)*1e3,
                                     poss[:,2],90-poss[:,1])

    if 'divjperp' in dat.keys():
        datadict['divjperp'] = model2pointsgeogcoords(xg, dat['divjperp'],
                                (poss[:,0]-RE)*1e3, poss[:,2],90-poss[:,1])
        
    br, btheta, bphi = secs3d.make_b_unitvectors(datadict['Bu'], 
                    -datadict['Bn'], datadict['Be'])
    datadict['fac'] = np.sum(np.array([datadict['ju'], -datadict['jn'], datadict['je']]) * 
                    np.array([br, btheta, bphi]), axis=0)

    # Line-of-sight sampling: NOT FULLY IMPLEMENTED YET
    # Project the mapped Vperp, at the mapped locations, onto line-of-sight 
    # direction of each measurement
    # Should try to implement an option of subtracting field aligned compoenet 
    # using a field aligned beam.
    # In reality, we will need to make such an assumption, or similar.
    # Must convert vperp into ECEF frame before doing dot product
    # enu_vec = np.vstack((datadict['vperpmappede'],datadict['vperpmappedn'], 
    #                      datadict['vperpmappedu'])).T
    # xyz_vec = coordinates.enu2xyz(enu_vec, datadict['mappedglon'], datadict['mappedglat'])
    # datadict['vlos'] = xyz_vec[:,0]*np.array(lx).flatten() + \
    #         xyz_vec[:,1]*np.array(ly).flatten() + \
    #         xyz_vec[:,2]*np.array(lz).flatten()
    # # convert cartesian LOS unit vector to local ENU form (at measurement location)
    # l_xyz = np.vstack((np.array(lx).flatten(),np.array(ly).flatten(),np.array(lz).flatten())).T
    # l_enu = coordinates.xyz2enu(l_xyz, datadict['mappedglon'], datadict['mappedglat'])
    # # Horizontal part of LOS direction
    # hormag = np.sqrt(l_enu[:,0]**2 + l_enu[:,1]**2)
    # datadict['l_hor_e'] =l_enu[:,0]/hormag
    # datadict['l_hor_n'] = l_enu[:,1]/hormag
        
    return datadict


def sample_points(xg, dat, lat, lon, alt):
    '''
    Sample GEMINI at input locations. Input must be geographic locations.

    Parameters
    ----------
    xg : dict
        Gemini grid object
    dat : xarrat Dataset
        GEMINI data.
    lat : array-like
        May have multiple dimensions. In degrees, geographic.
    lon : array-like
        May have multiple dimensions. In degrees, geographic.
    alt : array-like
        May have multiple dimensions. In km above surface.

    Returns
    -------
    datadict : dict
        Contain the relevant GEMINI data at sample locations.

    '''

    j1 = model2pointsgeogcoords(xg, dat['J1'], alt*1e3, lon, lat)
    je = model2pointsgeogcoords(xg, dat['je'], alt*1e3, lon, lat)
    jn = model2pointsgeogcoords(xg, dat['jn'], alt*1e3, lon, lat)
    ju = model2pointsgeogcoords(xg, dat['ju'], alt*1e3, lon, lat)
    Be = model2pointsgeogcoords(xg, dat['Be'], alt*1e3, lon, lat)
    Bn = model2pointsgeogcoords(xg, dat['Bn'], alt*1e3, lon, lat)
    Bu = model2pointsgeogcoords(xg, dat['Bu'], alt*1e3, lon, lat)
    
    datadict = {'lat':lat.flatten(), 'lon':lon.flatten(), 'alt':alt.flatten(), 
                'je':je, 'jn':jn, 'ju':ju, 'Be':Be, 'Bn':Bn, 'Bu':Bu, 'J1':j1}

    datadict['ne'] = model2pointsgeogcoords(xg, dat['ne'],alt*1e3, lon, lat)       
    datadict['ve'] = model2pointsgeogcoords(xg, dat['ve'],alt*1e3, lon, lat)
    datadict['vn'] = model2pointsgeogcoords(xg, dat['vn'],alt*1e3, lon, lat)
    datadict['vu'] = model2pointsgeogcoords(xg, dat['vu'],alt*1e3, lon, lat)
    datadict['vperpe'] = model2pointsgeogcoords(xg, dat['vperpe'],alt*1e3, lon, lat)
    datadict['vperpn'] = model2pointsgeogcoords(xg, dat['vperpn'],alt*1e3, lon, lat)
    datadict['vperpu'] = model2pointsgeogcoords(xg, dat['vperpu'],alt*1e3, lon, lat)
    datadict['sp'] = model2pointsgeogcoords(xg, dat['sp'], alt*1e3, lon, lat)
    datadict['sh'] = model2pointsgeogcoords(xg, dat['sh'], alt*1e3, lon, lat)
    datadict['Ee'] = model2pointsgeogcoords(xg, dat['Ee'], alt*1e3, lon, lat)
    datadict['En'] = model2pointsgeogcoords(xg, dat['En'], alt*1e3, lon, lat)
    datadict['Eu'] = model2pointsgeogcoords(xg, dat['Eu'], alt*1e3, lon, lat)
    datadict['jperpe'] = model2pointsgeogcoords(xg, dat['jperpe'], alt*1e3, lon, lat)
    datadict['jperpn'] = model2pointsgeogcoords(xg, dat['jperpn'], alt*1e3, lon, lat)
    datadict['jperpu'] = model2pointsgeogcoords(xg, dat['jperpu'], alt*1e3, lon, lat)

    # Sample electric potential
    # lx1 = xg["lx"][0]
    lx2 = xg["lx"][1]
    lx3 = xg["lx"][2]
    # inds1 = range(2, lx1 + 2)
    inds2 = range(2, lx2 + 2)
    inds3 = range(2, lx3 + 2)
    # x1 = xg["x1"][inds1]
    x2 = xg["x2"][inds2]
    x3 = xg["x3"][inds3]
    x1i, x2i, x3i = geog2dipole(alt*1e3, lon, lat)
    xi = np.array((x2i.ravel(), x3i.ravel())).transpose()
    if len(dat.Phitop.shape) == 2:
        datadict['Phitop'] = scipy.interpolate.interpn(
            points=(x2, x3),
            values=dat.Phitop.values,
            xi=xi,
            method="linear",
            bounds_error=False,
            fill_value=np.NaN)
    if len(dat.Phitop.shape) == 3: #Not sure what this means?
        datadict['Phitop'] = scipy.interpolate.interpn(
            points=(x2, x3),
            values=dat.Phitop.values[0,:,:],
            xi=xi,
            method="linear",
            bounds_error=False,
            fill_value=np.NaN)

    datadict['vperpmappede'] = model2pointsgeogcoords(xg, dat['vperpmappede'],
                            alt*1e3, lon, lat)
    datadict['vperpmappedn'] = model2pointsgeogcoords(xg, dat['vperpmappedn'],
                            alt*1e3, lon, lat)
    datadict['vperpmappedu'] = model2pointsgeogcoords(xg, dat['vperpmappedu'],
                            alt*1e3, lon, lat)              
    datadict['mappedglat'] = model2pointsgeogcoords(xg, dat['mappedglat'],
                            alt*1e3, lon, lat)
    datadict['mappedglon'] = model2pointsgeogcoords(xg, dat['mappedglon'],
                            alt*1e3, lon, lat)        
    
    return datadict


def lompe_fit(datadict, grid_l, l1=1, l2=1, altlim = 200, e3doubt_=False, 
              diagnostic=False):

    # making conductance tuples (not used in this case, just placeholders)
    def _hp(lo, la):
        return np.ones(lo.shape)
    SH = lambda lon = grid_l.lon, lat = grid_l.lat: _hp(lon, lat)
    SP = lambda lon = grid_l.lon, lat = grid_l.lat: _hp(lon, lat)

    # Create new grid so that the SECS nodes used to represent Lompe model parameters
    # aligns with the nodes in the input grid (may or may not be the same as used 
    # for current density reconstruction. Should in any case cover a larger
    # region to avoid boundary effects from the Lompe fit in the interior of the
    # 3D modelling region
    
    xi_e  = grid_l.xi[0,:]
    eta_e = grid_l.eta[:,0]
    _grid = cubedsphere.CSgrid(cubedsphere.CSprojection(grid_l.projection.position,
                            grid_l.projection.orientation), grid_l.L - grid_l.Lres,
                            grid_l.W - grid_l.Wres, grid_l.Lres, grid_l.Wres, 
                            edges = (xi_e, eta_e), R = grid_l.R) # outer

    # Create Emodel object. Pass grid and Hall/Pedersen conductance functions
    model = lompe.Emodel(_grid, Hall_Pedersen_conductance = (SH, SP), epoch=2000)
    
    #####
    # Add data
    minalt = altlim #km, only use velocities above to make the Lompe model
    use = datadict['alt']>=minalt
    data_coords = np.vstack((datadict['mappedglon'][use], datadict['mappedglat'][use]))
    vperpgg_enu = np.vstack((datadict['vperpmappede'][use],
                             datadict['vperpmappedn'][use],
                             datadict['vperpmappedu'][use])).T
    v_data = vperpgg_enu.T[0:2,:] # only use the east,north component in lompe
    if e3doubt_: # At present, off-diagonal terms are not used in Lompe
        error = np.vstack((datadict['cov_viperp_mapped'][0,0,use], 
                           datadict['cov_viperp_mapped'][1,1,use]))
        bad = (~np.isfinite(error)) | (error<-1e6) | (error>1e6)
        error[bad] = 1000**2
    else:
        error = 10**2 # variance
    gemini_data  = lompe.Data(v_data, data_coords, datatype = 'convection', 
                              iweight=1, error=np.sqrt(error))
    model.add_data(gemini_data)

    # if los:
    #     v_data = datadict['vlos'][use]
    #     lhor_gg_enu = np.vstack((datadict['l_hor_e'][use],
                    # datadict['l_hor_n'][use],np.zeros(sum(use)))).T 
    #     lhor_gm_enu = enugg2enugm(lhor_gg_enu, datadict['mappedglon'][use],
    #           datadict['mappedglat'][use]) # Convert horizontal part of LOS unit 
    #           vector at each observation location mapped to maph (local ENU) 
    #           from geographic to geomag components
    #     _LOS = np.vstack((lhor_gm_enu[:,0],lhor_gm_enu[:,1]))
    #     gemini_data  = lompe.Data(v_data, data_coords, datatype = 'convection', 
    #                               iweight=1, error=0.001, LOS=_LOS)  
    #     model.add_data(gemini_data)

    # Run inversion
    model.run_inversion(l1 = l1, l2 = l2, perimeter_width = 0)

    if diagnostic: # Validate the lompe representation
        fig = plt.figure()
        ax = fig.add_subplot(111)
        import apexpy
        apex = apexpy.Apex(2023)
        lompe.visualization.format_ax(ax, model, apex = apex)
        lompe.visualization.plot_quiver(ax, model, 'convection')
        lompe.visualization.plot_potential(ax, model)
        lompe.visualization.plot_datasets(ax, model, 'convection')
        ax.set_title('Convection velocity and electric potential')        
        # lompe.lompeplot(model,include_data=True, quiverscales={'convection':1500})
        # lompe.visualization.model_data_scatterplot(model, perimiter_width=-2)
        # diagnostics.compare_potentials(dat, secs_grid, alts_grid, datadict, 
                                       # inside, model)
    return model


def make_inputdict(datadict, grid, alts_grid, inputmode='vi', ext_factor=-1, 
                   hp_from_brekke=False):
    '''
    Prepare sampled GEMINI data to input in 3D inversion (jperp). Remove nans and 
    data outside the inner grid region, controlled by the ext_factor keyword. Its
    default value is -1, meaning that no observations are included from the 
    perimiter of the SECS grid. This is to enhance the ability og the perimiter
    SECS nodes in the 3D model to capture laplacian parts of the field in the 
    interior domain. Hence, the model amplitues should not try to fit the data
    on the perimiter, and one should not try to evaluate the model on the 
    perimiter for the same reasons.
    
    Parameters
    ----------
    datadict : dictionary
        The GEMINI output in ENU components (in 3D), and other derived useful
        quantities
    grid : CS grid object
        The grid we use to compute gridded indices of data/evaluation locations
        Dimension is (I,J). 
    alts_grid : 1D array
        Altitude grid to use together with grid. Length is K. Values 
        represent the height in km of the centre of the voxels of each layer. 
        Should be in increasing order.
    inputmode : str
        Determines how the input jperp to the 3D model is obtained. Must be one of:
            vi             : use sampled ion velocities, and jperp=ne(vi_perp-ve_perp)
            vi_ohmslaw     : use sampled ion velocities, but estimate jperp using
                             Ohms law, meaning that a conductance estimate must also
                             be used
            phitop         : Use potential at top from GEMINI and (vi-ve)                             
            phitop_ohmslaw : Use potential at top from GEMINI for E, and Ohms law
            jperp          : Use jperp directly sampled from GEMINI
    ext_factor : int, optional
        To control how to filter out obsrvations based on their proximity to the 
        grid. The default is -1, removing points closer than 1 grid cell from 
        the edge of the grid (mesh), on the interior side. If positive, allow
        points outside the grid. See notes at top of this function. 
    hp_from_brekke : boolean
        If inputmode uses Ohms law, this keyword specifies if conductivity is
        estimated with formulaes from Brekke book. If not, it uses the self-
        consistent values from GEMINI. Default is False. This has no influence
        when using vi-ve to estimate perp.
    
    Returns
    -------
    inputdict : dict
        Datadict containting only the observations that will be used
        in 3D reconstruction. Also includes 'inds', a 1D array containting the 
        mapping of which obserations of the input datadict that has been selscted 
        for use.
    '''
    
    e = 1.6e-19 # elementary charge
    if inputmode == 'vi':
        jperp_e = datadict['ne'] * e * (datadict['vperpe'] - 
                                        datadict['vperp_electron'][0,:])
        jperp_n = datadict['ne'] * e * (datadict['vperpn'] - 
                                        datadict['vperp_electron'][1,:])
        jperp_u = datadict['ne'] * e * (datadict['vperpu'] - 
                                        datadict['vperp_electron'][2,:])
        jperp = np.vstack((jperp_e, jperp_n, jperp_u))
            
    elif inputmode == 'jperp':
        jperp = np.vstack((datadict['jperpe'],datadict['jperpn'],datadict['jperpu']))
    
    elif inputmode == 'phitop':
        E = np.vstack((datadict['Ee'],datadict['En'],datadict['Eu']))
        B = np.vstack((datadict['Be'], datadict['Bn'], datadict['Bu']))
        Bmag = np.linalg.norm(B, axis=0)

        vperp_electron = np.cross(E, B, axis=0) / Bmag**2

        # Calculate jperp from velocity difference (geographic ENU coords)
        jperp_e = datadict['ne'] * e * (datadict['vperpe'] - vperp_electron[0,:])
        jperp_n = datadict['ne'] * e * (datadict['vperpn'] - vperp_electron[1,:])
        jperp_u = datadict['ne'] * e * (datadict['vperpu'] - vperp_electron[2,:])
        jperp = np.vstack((jperp_e, jperp_n, jperp_u))
            
    else: #using ohms law, either vi or phitop based E-field
        if hp_from_brekke: # Estimate conductances using atmosphere model
            #Calculate conductivity
            kn2 = 4.34e-16
            ko2 = 4.28e-16
            ko = 2.44e-16
            e = 1.6e-19 # electron charge
            mp = 1.67e-27 #proton mass in kg
            ZN = 7
            ZO = 8
            # m_n2 = 2*2*ZN * mp
            m_no = (2*ZN + 2*ZO) * mp
            Bmag = np.linalg.norm(B, axis=0)
            omega_no = e*Bmag/(m_no) #gyro freq
            #Open msis and iri profiles (CCMC output)
            names = ['year', 'month', 'day', 'doy', 'unknown', 'Height', 'glat', 
                     'glon', 'O', 'N2', 'O2', 'rho', 'Temperature_neutral', 'u1', 
                     'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 
                     'u12', 'u13', 'u14']
            msis = pd.read_csv('./utils/nrlmsis_2.0.txt', header=0, names=names, 
                   delim_whitespace=True)
            f = scipy.interpolate.interp1d(msis.Height, msis.N2)
            N2_msis = f(datadict['alt'].flatten())
            f = scipy.interpolate.interp1d(msis.Height, msis.O2)
            O2_msis = f(datadict['alt'].flatten())
            f = scipy.interpolate.interp1d(msis.Height, msis.O)
            O_msis = f(datadict['alt'].flatten())
            #eq 7.12 in Brekke book. Assumes average ion mass of 30.7.
            c_brekke = kn2 * N2_msis*1e6 + ko2 * O2_msis*1e6 + ko * O_msis*1e6 

            sp = e*datadict['ne'].flatten()/Bmag * \
                    (c_brekke * omega_no)/(omega_no**2 + c_brekke**2)
            sh = e*datadict['ne'].flatten()/Bmag * \
                    (c_brekke**2)/(omega_no**2 + c_brekke**2)
        else: # use conductances from GEMINI
            sp = datadict['sp']
            sh = datadict['sh']
            
        B = np.vstack((datadict['Be'], datadict['Bn'], datadict['Bu']))
        bhat_r, bhat_theta, bhat_phi = secs3d.make_b_unitvectors(datadict['Bu'], 
                                            -datadict['Bn'], datadict['Be'])
        bhat_enu = np.vstack((bhat_phi, -bhat_theta, bhat_r))
        
        if inputmode == 'vi_ohmslaw':
            E = -np.cross(datadict['vperp_electron'].T, B.T, axis=1).T
        elif inputmode == 'phitop_ohmslaw':
            E = np.vstack((datadict['Ee'],datadict['En'],datadict['Eu']))
        else:
            assert 1 == 0, 'Inputmode keyword is not valid.'
    
        j_pedersen = sp * E
        j_hall = sh * np.cross(bhat_enu.T, E.T).T
        jperp = j_pedersen + j_hall
        
    # Now, figure out which observations that are inside the interior of 3D model
    # domain
    #identify obs inside grid horizontally, and any nans
    use = grid.ingrid(datadict['lon'], datadict['lat'], ext_factor = ext_factor) & \
                np.isfinite(datadict['jperpe'])

    # Remove data/evaluation points outside grid in vertical direction
    k0 = secs3d.get_alt_index(alts_grid, datadict['alt'][use], returnfloat=True)
    inside = (k0 >= 0) & (k0<alts_grid.size-1)
    inds = np.zeros(use.size).astype(int)
    temp = np.zeros(sum(use))
    temp[inside] = 1
    inds[use] = temp

    inputdict = {'lon':datadict['lon'][inds==1], 'lat':datadict['lat'][inds==1], 
                 'alt':datadict['alt'][inds==1], 'Be':datadict['Be'][inds==1],
                 'Bn':datadict['Bn'][inds==1], 'Bu':datadict['Bu'][inds==1],
                 'jperp':jperp[:,inds==1], 'inds':inds}

    return inputdict

def get_E_from_lmodel(lmodel, ddict, xgdat, returnvperp=False):
    '''
    Use the Lompe representation of vperp at maph height, and compute 
    its mapped velocity vector at each (lat,lon,alt) location, and then
    estimate the corresponding E-field, using the magnetic field value.
    
    Note that E-field must not be computed with Lompe matrices, since that
    may involve a B-field that is inconsistent with what is used in GEMINI.
    Here we rather use E = - v x B with B from GEMINI.

    Parameters
    ----------
    lmodel : lompe object
        The Lompe representation of horizontal part of vperp in the domain
    ddict : dict
        Containing the information of the locations, velocities and B field
        sampled from GEMINI
    xgdat : tuple
        First element is the GEMINI grid object, second is the GEMINI data object 
    returnvperp : boolean
        Instead, return vperp at the input locations

    Returns
    -------
    E, electric field at each location (ENU), a (3 x N) 2D array [V/m].

    '''
    
    # Get hor part of vperp from Lompe
    ve, vn = lmodel.v(lon=ddict['mappedglon'], lat=ddict['mappedglat'])
    # Get vu of vperp from vperp dot B = 0
    # First need to bet B field at the maph altitude of Lompe
    Be = model2pointsgeogcoords(xgdat[0], xgdat[1]['Be'], lmodel.grid_E.R-RE*1e3, ddict['mappedglon'], ddict['mappedglat'])
    Bn = model2pointsgeogcoords(xgdat[0], xgdat[1]['Bn'], lmodel.grid_E.R-RE*1e3, ddict['mappedglon'], ddict['mappedglat'])
    Bu = model2pointsgeogcoords(xgdat[0], xgdat[1]['Bu'], lmodel.grid_E.R-RE*1e3, ddict['mappedglon'], ddict['mappedglat'])
    vu = -(ve*Be + vn*Bn)/Bu
    vperp = np.vstack((ve ,vn, vu))
    dip = dipole.Dipole(dipole_pole=((90-11),289)) # These locations found in pygemini doc.
    d1, d2, d3, _1, _2, _3 = dip.get_apex_base_vectors_geo(ddict['mappedglon'], 
                        ddict['mappedglat'], lmodel.grid_E.R*1e-3, R=RE)
    #Calculate the quantities that is constant along the field-lines
    ve1 = (d1[0,:]*vperp[0,:] + d1[1,:]*vperp[1,:] + d1[2,:]*vperp[2,:])
    ve2 = (d2[0,:]*vperp[0,:] + d2[1,:]*vperp[1,:] + d2[2,:]*vperp[2,:])
    # Calculate basis vectors at the mapped locations
    _1, _2, _3, e1, e2, e3 = dip.get_apex_base_vectors_geo(ddict['lon'], 
                        ddict['lat'], ddict['alt']+RE, R=RE)
    #Calculate the mapped velocity using eq 4.17 in Richmond 1995. geographic components, ENU            
    vperpmappede = (ve1.flatten()*e1[0,:] + ve2.flatten()*e2[0,:])
    vperpmappedn = (ve1.flatten()*e1[1,:] + ve2.flatten()*e2[1,:])
    vperpmappedu = (ve1.flatten()*e1[2,:] + ve2.flatten()*e2[2,:])
    
    v = np.vstack((vperpmappede, vperpmappedn, vperpmappedu))
    B = np.vstack((ddict['Be'], ddict['Bn'], ddict['Bu']))
    E = -np.cross(v, B, axis=0)
    
    if returnvperp:
        return v
    else:
        return E


def ehat(lmodel, ddict, xgdat):
    '''
    Function that return unit vector along E-field as represented by Lompe
    at the locations (lat, lon, alt) contained in ddict. 
    
    Parameters
    ----------
    lmodel : lompe object
        The Lompe representation of horizontal part of vperp in the domain
    ddict : dict
        Containing the information of the locations, velocities and B field
        sampled from GEMINI
    xgdat : tuple
        First element is the GEMINI grid object, second is the GEMINI data object          
    Returns
    -------
    3 element tuple (ehate, ehatn, ehatu).

    '''
    
    E = get_E_from_lmodel(lmodel, ddict, xgdat)
    
    Emag = np.linalg.norm(E, axis=0)
    ehate = E[0,:]/Emag
    ehatn = E[1,:]/Emag
    ehatu = E[2,:]/Emag

    return (ehate, ehatn, ehatu)


def calc_hall_pedersen(datadict, inputmode='vi', lmodel=None, xgdat=None):
    '''
    Calculate the magnitude of Hall and Pedersen current at the locations in datadict.

    Parameters
    ----------
    datadict : dict
        GEMINI data.
    inputmode : str
        Determines how the input jperp to the 3D model is obtained. Must be one of:
            vi             : use sampled ion velocities, and jperp=ne(vi_perp-ve_perp)
            vi_ohmslaw     : use sampled ion velocities, but estimate jperp using
                             Ohms law, meaning that a conductance estimate must also
                             be used
            phitop         : Use potential at top from GEMINI and (vi-ve)                             
            phitop_ohmslaw : Use potential at top from GEMINI for E, and Ohms law
            jperp          : Use jperp directly sampled from GEMINI        
    lmodel : lompe object
        The Lompe representation of horizontal part of vperp in the domain        

    Returns
    -------
    tuple (jh_mag, jp_mag).

    '''
    if 'phitop' in inputmode:
        E_enu = np.vstack((datadict['Ee'],datadict['En'],datadict['Eu']))
    else:
        E_enu = get_E_from_lmodel(lmodel, datadict, xgdat)
    
    br, btheta, bphi = secs3d.make_b_unitvectors(datadict['Bu'],-datadict['Bn'],datadict['Be'])
    bhat_enu = np.vstack((bphi,-btheta,br))

    # ENU gg conponnets of Hall and Pedersen current
    jp = datadict['sp'] * E_enu
    jh = datadict['sh'] * np.cross(bhat_enu,E_enu, axis=0)
    
    jp_mag = np.sqrt(jp[0,:]**2 + jp[1,:]**2 + jp[2,:]**2)
    jh_mag = np.sqrt(jh[0,:]**2 + jh[1,:]**2 + jh[2,:]**2)

    return (jh_mag, jp_mag)


def dipole_B(theta, height = 500):
    '''
    Calculate the centered dipole magnetic field strength in magnetic up and north directions.
    These values does not mathc exactly with the ones from the GEMINI output, likely due to 
    differences in dipole moment used.

    Parameters
    ----------
    theta : scalar or array like
        Centered dipole colatitude, in radians.
    height : scalar or array-like, optional
        Altitude in km of point to evaluate. If array provided, it must be of the same shape
        as theta. The default is 500.

    Returns
    -------
    2 element tuple (Bn, Bu) in T, wiht same shape as input theta.
    '''
    

    mu0 = 4*np.pi * 10**(-7)
    m = 7.94e22 #magnetic dipole moment, taken from GEMINI documentation
    RE = 6371.2 * 1e3
    Bu = -2 * mu0 * m * np.cos(theta) / (4*np.pi*(RE+height*1e3)**3)
    Btheta =  -mu0 * m * np.sin(theta) / (4*np.pi*(RE+height*1e3)**3)
    Bn = -Btheta
    return (Bn, Bu)


def make_lompe(grid_l, datadict, inputmode, maph, e3doubt_=True, 
               l1_lompe=1e-2, l2_lompe=1e-2, intsec=5*60, 
               filename_lompe='./inversion_coefs/lmodel.npy'):
    '''
    Make the lompe object that is used for the continous vperp description
    at maph altitude.

    Parameters
    ----------
    grid_l : SC grid object
        The grid object to be used for the Lompe fit, which should be an extended
        verstion of the one used in the 3D model.
    datadict : dict
        Contain the input data sampled from GEMINI.
    inputmode : str
        Determines how the input jperp to the 3D model is obtained. Must be one of:
            vi             : use sampled ion velocities, and jperp=ne(vi_perp-ve_perp)
            vi_ohmslaw     : use sampled ion velocities, but estimate jperp using
                             Ohms law, meaning that a conductance estimate must also
                             be used
            phitop         : Use potential at top from GEMINI and (vi-ve)                             
            phitop_ohmslaw : Use potential at top from GEMINI for E, and Ohms law
            jperp          : Use jperp directly sampled from GEMINI
    maph : int or float
        Height in km of altitude of where to make the Lompe representation. Also 
        the altitude of where ion obs are assumed to be not affected by collissions.
    e3doubt_ : bool, optional
        Wheter to estimate realistic uncertainties in the sampling.. The default is True.
    l1_lompe : bool, optional
        Lompe l1 regularization parameter. The default is 1e-2.
    l2_lompe : bool, optional
        Lompe l2 regularization parameter. The default is 1e-2.
    intsec : float or int, optional
        Number of seconds of integration when estimating unertainties with E3DOUBT. 
        The default is 5*60.
    filename_lompe : str, optional
        Filename of the saved lompe object. The default is './inversion_coefs/lmodel.npy'.

    Returns
    -------
    2 element tuple
    
    datadict : dict
        With updated values.
    lmodel : lompe object
        The estimated Lompe model.
    '''
    
    if (inputmode == 'vi') or (inputmode == 'vi_ohmslaw'):    
        if e3doubt_:
            # Make the data covariance matrix for input data to lompe
            lompedata = uncertainty.make_datacov_lompe(datadict.copy(), grid_l, maph)
            
            # Initialise Lompe model object (will not use the fit done here)
            lmodel = lompe_fit(lompedata, grid_l, l1=l1_lompe, l2=l2_lompe, 
                                    altlim = maph, e3doubt_=True)
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
            # datadict_backup = datadict.copy()
        else:
            lmodel = lompe_fit(datadict.copy(), grid_l, l1=l1_lompe,
                                            l2=l2_lompe, altlim = maph, 
                                            e3doubt_=e3doubt_)
           # Make up some covatiance values
            _cov = np.zeros((3,3,datadict['lat'].size))
            _cov[0,0,:] = 1e-6
            _cov[1,1,:] = 1e-6
            _cov[2,2,:] = 1e-7
            datadict['cov_jperp'] = _cov
            
        ds = lmodel.save()
        ds.to_netcdf(filename_lompe)
        print('Saved Lompe file.')

    else:
        lmodel=None
    
    return (datadict, lmodel)


def read_hdf5_to_dict(file_path):
    '''
    Function that read a HDF file and puts it into a dictionary.
    Intended to use on the gemini_gri.h5 file to make it back
    into the xg grid dictionary.
    '''

    # Initialize an empty dictionary to store the data
    data_dict = {}

    # Open the HDF5 file in read mode
    with h5py.File(file_path, 'r') as file:
        # Iterate over the datasets in the HDF5 file
        for key in file.keys():
            # Retrieve the data for each dataset
            data = file[key][()]

            # Add the data to the dictionary
            data_dict[key] = data

    return data_dict