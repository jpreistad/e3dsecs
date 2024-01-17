#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 07:45:47 2023

@author: jone
"""

import numpy as np
import pandas as pd
import secs3d
from gemini3d.grid.convert import unitvecs_geographic
from gemini3d.grid.gridmodeldata import model2geogcoords, model2pointsgeogcoords, geog2dipole
from gemini3d.grid.convert import geomag2geog, geog2geomag
import gemini3d.grid.convert as convert
import coordinates
import xarray as xr
from gemini_tools import RE as RE


def grid_e_field_at_alt(datadict, grid, return_B=False):
    """
    Compute E filed on CS grid from the interpolated Phi values by using the 
    differentiation matrices in the grid class.
    
    Parameters
    ----------
    datadict : dictionary
        The GEMINI output in ENU components at specific altitude.
    grid : CS grid object
        The field-aligned CS grid, defined at the top boundary. Should be the 
        evaluation grid (grid_ev object made with the function above in this 
        file).
    return_B : bool
        Use to rather returned the gridded B-field

    Returns
    -------
    Tuple containing the ENU components of the E-field deduced from model 
    output of Phi, gridded on the CS grid.

    """
    
    use = np.isfinite(datadict['ju'].flatten()) & grid.ingrid(datadict['glonmesh'].flatten(),
                    datadict['glatmesh'].flatten()) 
    ii, jj = grid.bin_index(datadict['glonmesh'].flatten()[use],datadict['glatmesh'].flatten()[use])
    ii1d = grid._index(ii, jj)
    df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 'phi': datadict['phi'].flatten()[use], 
                    'Be': datadict['Be'].flatten()[use], 'Bn': datadict['Bn'].flatten()[use], 
                    'Bu': datadict['Bu'].flatten()[use]})
    gdf = df.groupby('i1d').mean()
    df_sorted = gdf.reindex(index=pd.Series(np.arange(0,len(grid.lon.flatten()))), method='nearest', tolerance=0.1)
    phi = df_sorted.phi.values.reshape(grid.shape) #The "top boundary condition" for layerded SECS below
    Be = df_sorted.Be.values.reshape(grid.shape)
    Bn = df_sorted.Bn.values.reshape(grid.shape)
    Bu = df_sorted.Bu.values.reshape(grid.shape)
    Le, Ln = grid.get_Le_Ln()
    
    Ee = -Le.dot(phi.flatten()).reshape(grid.shape)    
    En = -Ln.dot(phi.flatten()).reshape(grid.shape)
    Eu = (Ee*Be + En*Bn)/Bu #Using E.dot(B) = 0
    
    if not return_B:
        datadict['Ee'] = Ee
        datadict['En'] = En
        datadict['Eu'] = Eu
        
        return (Ee, En, Eu)
    else:
        return (Be, Bn, Bu)
    
    
    
def grid_param_at_alt(datadict, grid, param='v'):
    """
    
    Parameters
    ----------
    datadict : dictionary
        The GEMINI output in ENU components (in 3D).
    grid : CS grid object
        The field-aligned CS grid, defined at the top boundary. Should be the 
        evaluation grid (grid_ev object made with the function above in this 
        file).
    param : str
        Select type of parameter to grid. : v, j, n

    Returns
    -------
    Tuple containing the ENU components of the param from model 
    output, gridded on the CS grid. Each component is a 3D array.

    """
    alts_ = datadict['altmesh'][:,0,0]
    use = np.isfinite(datadict['ju'][0,:,:].flatten()) & grid.ingrid(datadict['glonmesh'][0,:,:].flatten(),
                datadict['glatmesh'][0,:,:].flatten())
    if param == 'n':
        p = []
    else:
        pe = []
        pn = []
        pu = []

    for (i,alt) in enumerate(alts_):
        ii, jj = grid.bin_index(datadict['glonmesh'][i,:,:].flatten()[use],datadict['glatmesh'][i,:,:].flatten()[use])
        ii1d = grid._index(ii, jj)
        if param == 'n':
            df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 'n': datadict['n'].flatten()[use]})
        else:
            df = pd.DataFrame({'i1d':ii1d, 'i':ii, 'j':jj, 
                               param+'e': datadict[param+'e'][i,:,:].flatten()[use], 
                               param+'n': datadict[param+'n'][i,:,:].flatten()[use], 
                               param+'u': datadict[param+'u'][i,:,:].flatten()[use]})        
        gdf = df.groupby('i1d').mean()
        df_sorted = gdf.reindex(index=pd.Series(np.arange(0,len(grid.lon.flatten()))), method='nearest', tolerance=0.1)
        
        if param == 'n':
            p.append(df_sorted[param].values.reshape(grid.shape))
        else:
            pe.append(df_sorted[param+'e'].values.reshape(grid.shape))
            pn.append(df_sorted[param+'n'].values.reshape(grid.shape))
            pu.append(df_sorted[param+'u'].values.reshape(grid.shape))
        
    if param == 'n':
        return np.array(p)
    else:
        return (np.array(pe),np.array(pn), np.array(pu))    


def divergence(xg, dat, param = 'j_perp'):
    """
    

    Parameters
    ----------
    xg : GEMINI grid object
        Read by gemini read function from config file
    dat : GEMINI data object
        Containing GEMINI output data for specified variables at specified time.
    param : STR
        Which vector field to compute divergence of. Default is j_perp

    Returns
    -------
    Divergence of param on GEMINI grid.

    """
    RE = 6371.2e3 #Earth radius in m
    dims = xg['lx']


    #Metric factors defined in eqs 114-116 in GEMINI documentation
    h1 = xg['r']**3/(RE**2*np.sqrt(1+3*(np.cos(xg['theta']))**2))
    h2 = RE*(np.sin(xg['theta']))**3/np.sqrt(1+3*(np.cos(xg['theta']))**2)
    h3 = xg['r'] * np.sin(xg['theta'])
    scale = 1./(h1*h2*h3)
    
    #The three components (1,2,3 gemini grid) of the vector field to work with
    if param == 'j_perp':
        a1 = np.zeros(dat.J1.values.shape)
        a2 = dat.J2.values
        a3 = dat.J3.values
    elif param == 'j':
        a1 = dat.J1.values
        a2 = dat.J2.values
        a3 = dat.J3.values    
      
    else:
        print(param + ' not implemented yet.')
        print(1/0)
    
    #Differentials
    dx1__ = np.diff(dat.x1)
    dx1_ = np.concatenate((dx1__,[dx1__[-1]]))
    dx1 =  np.array([np.ones(dims[1:])*i for i in dx1_])
    dx2__ = np.diff(dat.x2)
    dx2_ = np.concatenate((dx2__,[dx2__[-1]]))
    dx2 =  np.array([np.ones([dims[0],dims[2]])*i for i in dx2_])
    dx2 = np.swapaxes(dx2,1,0)
    dx3__ = np.diff(dat.x3)
    dx3_ = np.concatenate((dx3__,[dx3__[-1]]))
    dx3 =  np.array([np.ones([dims[0],dims[1]])*i for i in dx3_])
    dx3 = np.swapaxes(dx3,1,0)
    dx3 = np.swapaxes(dx3,1,2)


    #The three terms of the divergence
    d1_ = np.append(np.diff(h2*h3*a1, axis=0),np.zeros([1,dims[1],dims[2]]), axis=0)
    d1 = scale*d1_/dx1
    d2_ = np.append(np.diff(h1*h3*a2, axis=1),np.zeros([dims[0],1,dims[2]]), axis=1)
    d2 = scale*d2_/dx2
    d3_ = np.append(np.diff(h1*h2*a3, axis=2),np.zeros([dims[0],dims[1],1]), axis=2)
    d3 = scale*d3_/dx3
    
    #Add the three terms to make the divergence of param
    d = d1 + d2 + d3

    return d


def gemini_fac_from_divergence(xg, dat):
    """
    

    Parameters
    ----------
    xg : GEMINI grid object
        Read by gemini read function from config file
    dat : GEMINI data object
        Containing GEMINI output data for specified variables at specified time.

    Returns
    -------
    Computed FAC based on divergence, on GEMINI grid.

    """    
    
    RE = 6371.2e3 #Earth radius in m
    dims = xg['lx']


    #Metric factors defined in eqs 114-116 in GEMINI documentation
    h1 = xg['r']**3/(RE**2*np.sqrt(1+3*(np.cos(xg['theta']))**2))
    h2 = RE*(np.sin(xg['theta']))**3/np.sqrt(1+3*(np.cos(xg['theta']))**2)
    h3 = xg['r'] * np.sin(xg['theta'])
    
    # FAC calculation based on equation 76 in the GEMINI documentation, which is
    # derived from equation 50 by setting div(j) = 0.
    J1_boundary = np.zeros(dims) # the FAC at the boundary, here assumed to be 0 at max x1 location
    divjperp = divergence(xg, dat, param = 'j_perp')
    dx1 = (np.diff(xg['x1'], axis=0, append=0)[2:-2])[:,np.newaxis,np.newaxis] * np.ones(dims)

    h2h3J1 = h2*h3*J1_boundary + np.flip(np.cumsum(np.flip(h1, axis=0) * np.flip(h2, axis=0) * 
                np.flip(h3, axis=0) * np.flip(divjperp, axis=0) * 
                np.flip(dx1, axis=0), axis=0), axis=0)    
    
    fac = h2h3J1 / (h2 * h3)
    
    return fac


def divergence_spherical(sampledict, hor=False, alt = None, perp=False):
    """
    

    Parameters
    ----------
    sampledict : dictionary
        Dictoinary containing coordinates and data sampled with sampling 
        function in this file (spherical components). Lat/lon in degrees is 
        required, and altitude in km. Dictionary also contain EN(U) components 
        of the vector field which must be named je, jn, (and ju).
    perp : boolean
        Whether only horizontal (lon/lat) divergence is computed
    alt : int
        Altitude in km at where divergence is evaluated. Needed when input is 2D.

    Returns
    -------
    Divergence of param on computed from spherical components.

    """
    RE = 6371.2e3 #Earth radius in m
    
    if 'shape' in sampledict.keys():
        dims = sampledict['shape']
        sampledict['glon'] =sampledict['lon'].reshape(dims)
        sampledict['glat'] =sampledict['lat'].reshape(dims)
        sampledict['alt'] =sampledict['alt'].reshape(dims)
        sampledict['je'] =sampledict['je'].reshape(dims)
        sampledict['jn'] =sampledict['jn'].reshape(dims)
        sampledict['ju'] =sampledict['ju'].reshape(dims)
    else:
        dims = sampledict['glat'].shape
    ndims = len(dims)
    if ndims == 2: #Does not use np.gradient(). Should be implemented.
        if alt is None:
            print('Must specify altitude')
            print(1/0)
        r = RE + alt*1e3

        #Vector components
        jtheta = -sampledict['jn'][0,:,:]
        jphi = sampledict['je'][0,:,:]
        
        #Grid differentials
        glons = np.radians(sampledict['glonmesh'])
        glats = np.radians(sampledict['glatmesh'])
        thetas = np.pi/2. - glats
        dphi_ = np.diff(glons, axis=0)
        dphi = np.concatenate((dphi_,[dphi_[-1]]))
        dtheta_ = np.diff(thetas, axis=1)
        dtheta = np.hstack((dtheta_,dtheta_[:,-1][:,np.newaxis]))
        
        #vector part differentials
        cphi_ = np.diff(jphi, axis=0)
        cphi = np.concatenate((cphi_,[cphi_[-1]]))
        ctheta_ = np.diff(jtheta*np.sin(thetas), axis=1)
        ctheta = np.hstack((ctheta_,ctheta_[:,-1][:,np.newaxis]))
        
        #Divergence terms
        d1 = 1./(r*np.sin(thetas)) * (ctheta / dtheta)
        d2 = 1./(r*np.sin(thetas)) * (cphi / dphi)
        
        return d1 + d2
    
    if ndims == 3:

        # Rearrange arrays so that differentiation is done in correct direction
        #Note, input is not cheked if the flip is really needed! It depends on how
        # the input was created.
        r = np.flip(RE + sampledict['alt']*1e3, axis=1)
        
        #Vector components
        if type(sampledict['ju']) != int: 
            jr = np.flip(sampledict['ju'], axis=1)
        else:
            jr = sampledict['ju']
        jtheta = np.flip(-sampledict['jn'], axis=1)
        jphi = np.flip(sampledict['je'],axis=1)
        
        # Grid representation
        rs = RE + sampledict['alt'][:,0,0]*1e3
        glats = np.flip(np.radians(sampledict['glat']), axis=1)
        thetas = np.pi/2 - glats
        glons = np.flip(np.radians(sampledict['glon']), axis=1)
        # rs = RE + sampledict['alt'][:,0,0]*1e3
        # glats = np.radians(sampledict['glat'][0,:,:])
        # thetas = np.pi/2 - glats
        # glons = np.radians(sampledict['glon'][0,:,:])
        
        # #Alternative way
        # glats = np.radians(sampledict['glat'])
        # thetas = np.pi/2 - glats
        # glons = np.radians(sampledict['glon'])
        # dphi_ = np.diff(np.radians(sampledict['glon'][0,0,:]))#[0]
        # dphi = np.concatenate((dphi_,[dphi_[-1]]))
        # dtheta_ = np.diff(np.radians(90.-sampledict['glat'][0,:,0]))#[0]
        # dtheta = np.concatenate((dtheta_,[dtheta_[-1]]))
        # rs = RE + sampledict['alt'][:,0,0]*1e3
        # dr_ = np.diff(rs)
        # dr = np.concatenate((dr_,[dr_[-1]]))

        # d0 = (1/r**2) * np.gradient(r**2 * jr, dr, axis=0)
        # d1 = 1./(r*np.sin(thetas)) * np.gradient(jtheta * np.sin(thetas), dtheta, axis=1)
        # d2 = 1./(r*np.sin(thetas)) * np.gradient(jphi, dphi, axis=2)
        
        #Grid differentials
        dphi_ = np.diff(glons, axis=1)
        dphi = np.hstack((dphi_,dphi_[:,-1][:,np.newaxis]))
        dtheta_ = np.diff(thetas, axis=1)
        # dtheta = np.vstack((dtheta_,dtheta_[-1,:][np.newaxis,:]))
        dtheta = np.hstack((dtheta_,dtheta_[:,-1,:][:,np.newaxis,:]))
        dr_ = np.diff(rs)
        dr = np.concatenate((dr_,[dr_[-1]]))
        
        #vector part differentials
        cphi_ = np.diff(jphi, axis=2)
        cphi = np.dstack((cphi_,cphi_[:,:,-1][:,:,np.newaxis]))
        ctheta_ = np.diff(jtheta*np.sin(thetas), axis=1)
        ctheta = np.hstack((ctheta_,ctheta_[:,-1,:][:,np.newaxis,:]))
        cr_ = np.diff(r**2 * jr, axis=0)
        cr = np.vstack((cr_, cr_[-1,:,:][np.newaxis,:,:]))
        
        #Divergence terms
        d0 = (1./r**2) * (cr / dr[:,np.newaxis, np.newaxis])
        d1 = 1./(r*np.sin(thetas)) * (ctheta / dtheta)
        d2 = 1./(r*np.sin(thetas)) * (cphi / dphi)
        if hor:
            return np.flip(d1 + d2, axis=1)
        else:
            if perp: # Have a look at this before use!
                print('Have a look at this before use!')
                br, btheta, bphi = secs3d.make_b_unitvectors(sampledict['Bu'], 
                                    -sampledict['Bn'], sampledict['Be'])
                return np.flip(br*d0 + btheta*d1 + bphi*d2, axis=1)
            else: 
                return np.flip(d0 + d1 + d2, axis=1)


from scipy.interpolate import RectBivariateSpline
def get_interpolated_amplitudes(m, secs_grid, grid_ev, alts_grid):
    # instead of using the 
    m_cf = m[0:m.size//2].reshape((alts_grid.size, 
                        secs_grid.lat.shape[0],secs_grid.lat.shape[1]))
    x = secs_grid.eta[:,0]
    y = secs_grid.xi[0,:]    
    
    K = alts_grid.size
    for k in range(K):
        f = RectBivariateSpline(x, y, m_cf[k,:,:], kx = 2, ky = 2, s = 0)
        eval_xi = grid_ev.xi[0,:]
        eval_eta = grid_ev.eta[:,0]
        gridded = f(eval_eta, eval_xi)   


def map_velocity(mlon_2, mlat_2, r_2, v2_2, v3_2, maph=100):
    """
    Map velolity component from (2) to (1). If maph is a float or int, (1) will
    all be at this height. If maph is an array of same length as mlon_2/mlat_2, 
    r_2 must be provided as an int or float to be applied to all input locations.

    Parameters
    ----------
    mlon_2 : array-like
        magnetic longitude in radians of points to map.
    mlat_2 : array-like
        magnetic latitude in radians of points to map.
    r_2 : array-like or float/int
        radius in km (RE+alt) of points to map. If maph is an array, r_2 must be
        an int or float.
    v2_2 : array-like
        velocity component 2 (gemini grid, north-south ish) of points to map.
    v3_2 : array-like
        velocity component 3 (gemini grid, east-west ish) of points to map.
    maph : int, float, or array-like, optional
        height to map to. The default is 100 km. This is different from the height
        used to determine the E-field from observations, which need to be in
        F-layer. If array is provided, it must have the same shape as the locations.
        Then each location is mapped to the given altitude.

    Returns
    -------
    v2_1 : array-like
        mapped velocity comopnent.
    v3_1 : array-like
        mapped velocity comopnent.
    mappedglon : array-like
        geographic longitude of where the point map to maph, in degrees.
    mappedglat : array-like
        geographic latitude of where the point map to maph, in degrees.

    """
    # dims = dat.v1.shape
    # KK = dims[0] # vertical
    # II = dims[1] # north-south
    # JJ = dims[2] # east-west
    
    # df = 5 #altitude of shell at maph, in km,
    # vdiff = xg['alt'] - maph*1e3
    # diffs_h = np.min(np.abs(vdiff),axis=0) # differences at maph, in meter
    # kk = np.argmin(np.abs(vdiff),axis=0) # location of differences
    # ii_ = np.arange(0,II)
    # jj_ = np.arange(0,JJ)
    # ii, jj = np.meshgrid(ii_, jj_, indexing='ij')
    # kij = np.ravel_multi_index((kk,ii,jj), (KK,II,JJ))
    
    # use = diffs_h <= df*1e3 #only use locations within df distance of maph
    # v2_1 = dat.v2.values.flatten()[kij]
    # v2_1[~use] = np.nan
    # v3_1 = dat.v3.values.flatten()[kij]
    # v3_1[~use] = np.nan
    

    # mapping from (2) to (1) using dipole equation
    if (type(maph) == int) or (type(maph) == float): 
        r_1 = np.ones(r_2.shape) * (RE+maph)
    else: # maph should be an array
        r_1 = RE + maph
        r_2 = np.ones(mlon_2.shape) * r_2
    colat_1 = np.arcsin(np.sin(np.pi/2 - mlat_2) * np.sqrt(r_1/r_2))
    mlat_1 = np.radians(90) - colat_1 #in radians
    mlon_1 = mlon_2# xg['phi'] # in radians
    mappedglon, mappedglat = geomag2geog(mlon_1, colat_1) #returns in degrees

    # Apply mapping formulas for velocity from 
    # https://angeo.copernicus.org/articles/34/55/2016/angeo-34-55-2016.pdf
    v2_ratio = (np.cos(mlat_2)**3 / np.cos(mlat_1)**3) * \
        np.sqrt((1+3*np.sin(mlat_1)**2) / (1+3*np.sin(mlat_2)**2))
    v2_1 = v2_2 / v2_ratio
    v3_ratio = (r_2/r_1)**(1.5)
    # _v3_ratio = (np.cos(mlat_2)**3) / (np.cos(mlat_1)**3) # should be the same
    v3_1 = v3_2 / v3_ratio        

    return v2_1, v3_1, mappedglon, mappedglat


def rotvec_gg2gm(e):
    egg=convert.Rgg2gm()@e.T   
    return egg.T



def unitvecs_geographic_general(glon, glat, dipole=True):
    '''
    Return a set of unit vectors in the geographic directions; components in 
    cartesian ECEF. Code copied from pygemini convert.py. Note that if input arguments
    are not geographic, e.g. geomagnetic, and dipole=False, then this function will
    return the ECEF components of the ENU directions with refernce to that (geomagnetic)
    coordinate system.
    
    Parameters
    ----------
    glon : array-like
        geographic longitude in degrees.
    glat : array-like
        grographic latitude in degrees.
    dipole : boolean
        specifies whether the cartesian ECEF components should be in geographic
        or geomagnetic (dipole) frame. If false, z is aligned with earths rotation.
        If True (default) z is along dipole axis. This is the typical use with GEMINI.

    Returns
    -------
    egalt : array-like
        cartesian ECEF components of local radial direction.
    eglon : array-like
        cartesian ECEF components of local geo east direction.
    eglat : array-like
        cartesian ECEF components of local geo north direction.

    '''
    thetagg=np.pi/2-glat.flatten()*np.pi/180
    phigg=glon.flatten()*np.pi/180
    shh = glon.shape
    lx = glon.size
    ergg=np.empty((lx,3))
    ethetagg=np.empty((lx,3))
    ephigg=np.empty((lx,3))
    
    # unit vectors in ECEF Cartesian geographic
    ergg[:,0]=np.sin(thetagg)*np.cos(phigg)
    ergg[:,1]=np.sin(thetagg)*np.sin(phigg)
    ergg[:,2]=np.cos(thetagg)
    ethetagg[:,0]=np.cos(thetagg)*np.cos(phigg)
    ethetagg[:,1]=np.cos(thetagg)*np.sin(phigg)
    ethetagg[:,2]=-np.sin(thetagg)
    ephigg[:,0]=-np.sin(phigg)
    ephigg[:,1]=np.cos(phigg)
    ephigg[:,2]=np.zeros(thetagg.shape)
    
    if dipole:
        # rotate into geomagnetic components (as used in grid dictionary)
        egalt=rotvec_gg2gm(ergg)
        eglon=rotvec_gg2gm(ephigg)
        eglat=-1*rotvec_gg2gm(ethetagg)
    else:
        egalt = ergg
        eglon = ephigg
        eglat = -ethetagg
    
    return eglon,eglat,egalt


def gemini_vec_2_enu_vec(gemini_vec, lon, lat, geographic=True):
    '''
    Convert a gemini vector (given along e1,e2,e3 directions) to local 
    ENU components at the input lon, lat locations. geographic keyword control whether
    the returned components are in geographic ENU (default) or geomagnetic ENU.
    
    This function can replace the 
    above compute_enu_components() function, as this is more general and not locked 
    to the GEMINI grid. 
    I dont understand why altitude is not part of this calculation. Probably 
    because the e1 vector has the same angle to the main B field regardless
    of the distance r along the field line. The results
    obtained here reproduces the ones provided in the xg grid dict, so it should 
    be correct. Note that a factor 3 was added to the z-component in eq 124 in the
    gemini doc. This was a typo in their derivation.
    
    Parameters
    ----------
    gemini_vec : array_like
        (N,3) array of N vectors represented along e1, e2, e3 directions
    lon : array-like
        longitude of the vector to convert, in degrees.
    lat : array-like
        latitude of the vector to convert, in degrees.
    alt : array-like
        altitude in km of the vector to convert.

    Returns
    -------
    (N,3) shape array of ENU components of input vector.

    '''
    
    # Get cartesian geomagnetic ECEF components of local unit vector at (lon, lat)
    egmlon, egmlat, egmalt = unitvecs_geographic_general(lon, lat, dipole=geographic)
    if geographic:
        phi, theta = geog2geomag(lon, lat) # degrees input, radians out
    else:
        phi = np.radians(lon)
        theta = np.pi/2 - np.radians(lat)
    
    # Get cartesian geomagnetic components of local (e1, e2, e3) unit vector
    # at (lat, lon)
    # Will use eqs 123-125 in the GEMINI document. 
    sf = np.sqrt(1+3*(np.cos(theta))**2)
    sfm = 1-3*(np.cos(theta))**2
    e1 = np.array([-3*np.cos(theta)*np.sin(theta)*np.cos(phi)/sf, 
                   -3*np.cos(theta)*np.sin(theta)*np.sin(phi)/sf, 
                   sfm/sf]).T
    e2 = np.array([np.cos(phi)*sfm/sf, np.sin(phi)*sfm/sf, 
                   3*np.cos(theta)*np.sin(theta)/sf]).T
    e3 = np.array([-np.sin(phi), np.cos(phi), np.zeros(phi.size)]).T
    
    # Project each GEMINI component (1,2,3) of gemini_vec onto the local ENU directions
    vgalt=( np.sum(e1*egmalt,1)*gemini_vec[:,0] + 
           np.sum(e2*egmalt,1)*gemini_vec[:,1] + 
           np.sum(e3*egmalt,1)*gemini_vec[:,2] )
    vglat=( np.sum(e1*egmlat,1)*gemini_vec[:,0] + 
           np.sum(e2*egmlat,1)*gemini_vec[:,1] +
           np.sum(e3*egmlat,1)*gemini_vec[:,2] )
    vglon=( np.sum(e1*egmlon,1)*gemini_vec[:,0] + 
           np.sum(e2*egmlon,1)*gemini_vec[:,1] + 
           np.sum(e3*egmlon,1)*gemini_vec[:,2] )   
    
    return vglon, vglat, vgalt


def enu_vec_2_gemini_vec(enu_vec, lon, lat, geographic = True):
    '''input must be in degrees, and in same coords (geographic or magnetic) as 
    enu_vec components are provided in. enu_vec must be
    of shape Nx3, where the second dimesion reflect E,N,U components'''
    if geographic:
        egmlon, egmlat, egmalt = unitvecs_geographic_general(lon, lat)
        phi, theta = geog2geomag(lon, lat) # degrees input, radians out

    else:
        phi = np.radians(lon)
        theta = np.pi/2 - np.radians(lat)
        egmlon, egmlat, egmalt = unitvecs_geographic_general(lon, lat, dipole=False)

    # Get cartesian geomagnetic components of local (e1, e2, e3) unit vector
    # at (glat, glon). Will use eqs 123-125 in the GEMINI document, with typo in
    # z-comp in eq 124 corrected (added a factor 3)
    sf = np.sqrt(1+3*(np.cos(theta))**2)
    sfm = 1-3*(np.cos(theta))**2
    e1 = np.array([-3*np.cos(theta)*np.sin(theta)*np.cos(phi)/sf, 
                   -3*np.cos(theta)*np.sin(theta)*np.sin(phi)/sf, 
                   sfm/sf]).T
    e2 = np.array([np.cos(phi)*sfm/sf, np.sin(phi)*sfm/sf, 
                   3*np.cos(theta)*np.sin(theta)/sf]).T
    e3 = np.array([-np.sin(phi), np.cos(phi), np.zeros(phi.size)]).T
    
     
   
    # Project each ENU component enu_vec onto the gemini (1,2,3) directions
    v1= ( np.sum(e1*egmlon,1)*enu_vec[:,0] + 
           np.sum(e1*egmlat,1)*enu_vec[:,1] + 
           np.sum(e1*egmalt,1)*enu_vec[:,2] )
    v2= ( np.sum(e2*egmlon,1)*enu_vec[:,0] + 
           np.sum(e2*egmlat,1)*enu_vec[:,1] + 
           np.sum(e2*egmalt,1)*enu_vec[:,2] )
    v3= ( np.sum(e3*egmlon,1)*enu_vec[:,0] + 
           np.sum(e3*egmlat,1)*enu_vec[:,1] + 
           np.sum(e3*egmalt,1)*enu_vec[:,2] )    
    
    return v1, v2, v3


def enugm2enugg(enugm_vec, glon, glat):
    """
    Convert vector with ENU geomagnetic components, located at glon, glat to 
    ENU geographic components.

    Parameters
    ----------
    enugm_vec : 2D array
        The ENU geomag vectors to convert. Shape must be N,3
    glon : 1D array
        Geographic longitudes of the location of the ENU vectors of size N
    glat : 1D array
        geographic latitudes of the location of the ENU vectors of size N.

    Returns
    -------
    enugg_vec, a 2D array of size N,3 of the converted vectors

    """
    
    # 1) Represent vetor as ECEF geomag
    phi, theta = geog2geomag(glon, glat) # degrees input, radians out 
    mlon = np.degrees(phi)
    mlat = 90 - np.degrees(theta)
    ex, ey, ez = coordinates.unitvec_rthetaphi(mlon, mlat) 
    vx = ex[:,0]*enugm_vec[:,2] - ex[:,1]*enugm_vec[:,1] + ex[:,2]*enugm_vec[:,0]
    vy = ey[:,0]*enugm_vec[:,2] - ey[:,1]*enugm_vec[:,1] + ey[:,2]*enugm_vec[:,0]
    vz = ez[:,0]*enugm_vec[:,2] - ez[:,1]*enugm_vec[:,1] + ez[:,2]*enugm_vec[:,0]
    # 2) rotate gm->gg
    v_ecef_gg = (convert.Rgm2gg()@(np.vstack((vx,vy,vz)))).T
    # 3) go back to geographic east,north components from ECEF
    er, etheta, ephi = coordinates.unitvec_xyz(glon, glat)
    # er, etheta, ephi = unitvec_xyz(datadict['lon'][inside], datadict['lat'][inside])
    vr = er[:,0]*v_ecef_gg[:,0] + er[:,1]*v_ecef_gg[:,1] + er[:,2]*v_ecef_gg[:,2]
    vtheta = etheta[:,0]*v_ecef_gg[:,0] + etheta[:,1]*v_ecef_gg[:,1] + etheta[:,2]*v_ecef_gg[:,2]
    vphi = ephi[:,0]*v_ecef_gg[:,0] + ephi[:,1]*v_ecef_gg[:,1] + ephi[:,2]*v_ecef_gg[:,2]
    enugg_vec = np.vstack((vphi,-vtheta,vr)).T

    return enugg_vec      


def enugg2enugm(enugg_vec, glon, glat):
    """
    Convert vector with ENU geographic components, located at glon, glat to 
    ENU geomagnetic components.

    Parameters
    ----------
    enugg_vec : 2D array
        The vectors with ENU geographic components to convert. Shape must be N,3
    glon : 1D array
        Geographic longitudes of the location of the ENU vectors of size N
    glat : 1D array
        geographic latitudes of the location of the ENU vectors of size N.

    Returns
    -------
    enugm_vec, a 2D array of size N,3 of the converted vectors

    """

    #Convert locations to dipole coordinate system
    phi, theta = geog2geomag(glon, glat) # degrees input, radians out

    # Convert geographic components (east,north) to dipole (east,north) components
    # This is done by 1) computing the ECEF geographic components of the vecor,
    # 2) rotate this into ECEF geomagnetic basis using matrices from pyGEMINI,
    # 3) compute the corresponding ENU components (geomagnetic east, north)
    # egglon, egglat, eggalt = unitvecs_geographic_general(datadict['mappedglon'][use], 
    #                                     datadict['mappedglat'][use], dipole=False)
    # 1) Project each geographic component (east,north, up) of the mapped velovity onto 
    # the geographical ECEF directions (x,y,z)
    xhat, yhat, zhat = coordinates.unitvec_rthetaphi(glon, glat)
    vx = xhat[:,0]*enugg_vec[:,2] - xhat[:,1]*enugg_vec[:,1] + xhat[:,2]*enugg_vec[:,0]
    vy = yhat[:,0]*enugg_vec[:,2] - yhat[:,1]*enugg_vec[:,1] + yhat[:,2]*enugg_vec[:,0]
    vz = zhat[:,0]*enugg_vec[:,2] - zhat[:,1]*enugg_vec[:,1] + zhat[:,2]*enugg_vec[:,0]
    # 2) rotate gg->gm
    v_ecef_gm = (convert.Rgg2gm()@(np.vstack((vx,vy,vz)))).T
    # 3) go back to dipole east,north components    
    rhat, thetahat, phihat = coordinates.unitvec_xyz(np.degrees(phi), 90-np.degrees(theta))
    vr = rhat[:,0]*v_ecef_gm[:,0] + rhat[:,1]*v_ecef_gm[:,1] + rhat[:,2]*v_ecef_gm[:,2]
    vtheta = (thetahat[:,0]*v_ecef_gm[:,0] + thetahat[:,1]*v_ecef_gm[:,1] + thetahat[:,2]*v_ecef_gm[:,2])
    vphi = phihat[:,0]*v_ecef_gm[:,0] + phihat[:,1]*v_ecef_gm[:,1] + phihat[:,2]*v_ecef_gm[:,2]
    enugm_vec = np.vstack((vphi,-vtheta,vr)).T
    
    return enugm_vec


def gemini_gradient(xg, dat, q='Phitop'):
    '''
    Compute the gradient of a scalar field, e.g. electric potential defined in GEMINIS
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
    dims = xg['lx']


    #Metric factors defined in eqs 114-116 in GEMINI documentation
    h1 = xg['r']**3/(RE**2*np.sqrt(1+3*(np.cos(xg['theta']))**2))
    h2 = RE*(np.sin(xg['theta']))**3/np.sqrt(1+3*(np.cos(xg['theta']))**2)
    h3 = xg['r'] * np.sin(xg['theta'])
    scale = 1./(h1*h2*h3)
    
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


def R_ecef_enu(lon, lat,):
    # Make rotation matrix that rotate a vector at spherical location lon,lat
    # with components in x,y,z directions, to local ENU components
    # input in degrees
    
    phi = np.radians(lon)
    theta = np.radians(90-lat)
    
    # ehat = np.array([-np.sin(phi),                np.cos(phi),               0])
    # nhat = np.array([-np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)])
    # uhat = np.array([np.sin(theta)*np.cos(phi),   np.sin(theta)*np.sin(phi), np.cos(theta)])

    if isinstance(lon, np.ndarray):
        ehat = np.vstack((-np.sin(phi),                np.cos(phi),          np.zeros(phi.size)))
        nhat = np.vstack((-np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)))
        uhat = np.vstack((np.sin(theta)*np.cos(phi),   np.sin(theta)*np.sin(phi), np.cos(theta)))
        R = np.concatenate((ehat[np.newaxis,:,:], nhat[np.newaxis,:,:], uhat[np.newaxis,:,:]), axis=0)

    else:
        ehat = np.vstack((-np.sin(phi),                np.cos(phi),               0))
        nhat = np.vstack((-np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)))
        uhat = np.vstack((np.sin(theta)*np.cos(phi),   np.sin(theta)*np.sin(phi), np.cos(theta))) 
        R = np.vstack((ehat.T, nhat.T, uhat.T))
    
    return R


def R_gg_gm():
    # Rotation matrix that operate on vectors of ECEF components in geographic directions
    # to produce ECEF components in geomagnetic (centered dipole)
    
    
    thetan = np.radians(11)
    phin = np.radians(289)
    
    # make unit vector pointing along centered magnetic dipole axis, in geo basis
    mhat = np.array([np.sin(thetan)*np.cos(phin),np.sin(thetan)*np.sin(phin),np.cos(thetan)])
    zhat = mhat
    yhat = np.cross(np.array([0,0,1]), zhat)/np.linalg.norm(np.cross(np.array([0,0,1]), zhat))
    xhat = np.cross(yhat, zhat)
    
    R = np.vstack((xhat.T, yhat.T, zhat.T))

    return R

def R_enugg_enugm(glon, glat):
    # Make rotation matrix that convert vectors at lon,lat of geographic enu components
    # to geomagnetic (CD) ENU components
    
    _R_ecef_enu_gg = R_ecef_enu(glon, glat)
    #Find magnetic coordinates of the locations when rotating into the magnetic ENU components
    phi, theta = geog2geomag(glon, glat)
    _R_ecef_enu_gm = R_ecef_enu(np.degrees(phi), 90-np.degrees(theta))
    
    # R = _R_ecef_enu.transpose(1,0,2) * R_gg_gm().reshape(3,3,1)*_R_ecef_enu)
    
    if isinstance(glon, np.ndarray):
        N = glon.size
        R = []
        for i in range(N):
            R.append(_R_ecef_enu_gm[:,:,i] @ R_gg_gm() @ _R_ecef_enu_gg[:,:,i].T)
        R = np.array(R)
        R = np.swapaxes(np.swapaxes(np.array(R),1,0),2,1)
    else:
        R = _R_ecef_enu_gm @ R_gg_gm() @ _R_ecef_enu_gg.T

    # Testing
    # v_enu_gg = np.array([1,0,0])
    # glon = 10
    # glat = 70
    # _R_ecef_enu_gg = R_ecef_enu(glon, glat)
    # v_ecef_gg = _R_ecef_enu_gg.T.dot(v_enu_gg)
    # v_ecef_gm = R_gg_gm().dot(v_ecef_gg)
    # #Find magnetic coordinates of the locations when rotating into the magnetic ENU components
    # phi, theta = geog2geomag(glon, glat)
    # _R_ecef_enu_gm = R_ecef_enu(np.degrees(phi), 90-np.degrees(theta))
    # v_enu_gm = _R_ecef_enu_gm.dot(v_ecef_gm)
    
    R = _R_ecef_enu_gm @ R_gg_gm() @ _R_ecef_enu_gg.T
    # v_enu_gm = R.dot(v_enu_gg)
    

    return R
            