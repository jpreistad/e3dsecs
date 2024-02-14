#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:32:35 2023

@author: jone

This file contain functions that preform various coordinate transformations and
numerical tools relevant for working with GEMINI and the 3D SECS model for 
estimating 3D current densities, but not directly linked to the GEMINI grid.

"""

import numpy as np
from gemini3d.grid.convert import geog2geomag
import gemini3d.grid.convert as convert


def _diff(_x, _s):
    '''
    Compute derivatives with central differencing on 1D array.
    One sided difference on edges, leaving the output of same size as input.

    Parameters
    ----------
    _x : 1D array
        The grid.
    _s : 1D array
        The values.

    Returns
    -------
    _derivative : 1D array
        The derivative of _s on the _x grid.
    '''
    _derivative = np.zeros(_x.size)
    # Compute central difference for interior points
    for i in range(1, len(_x) - 1):
        _derivative[i] = (_s[i + 1] - _s[i - 1]) / (_x[i + 1] - _x[i - 1])
    # Compute one-sided differences for the edges
    _derivative[0] = (_s[1] - _s[0]) / (_x[1] - _x[0])  # One-sided difference at the left edge
    _derivative[-1] = (_s[-1] - _s[-2]) / (_x[-1] - _x[-2])  # One-sided difference at the right edge
    return _derivative
    

def diff2d(_x, _y, axis=0):
    '''
    Compute derivatives with central differencing on a 2D mesh grid, but in the
    direction specified with the axis keyword. 
        
    IMPROVE DOCUMENTATION


    Parameters
    ----------
    _x : TYPE
        DESCRIPTION.
    _y : TYPE
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    _derivative : TYPE
        DESCRIPTION.
    '''
    _derivative = np.zeros(_x.shape)
    # Compute central difference for interior points
    if axis == 0:
        I = _x.shape[0]
        for i in range(1, I - 1):
            _derivative[i,:] = (_y[i + 1,:] - _y[i - 1,:]) / (_x[i + 1,:] - _x[i - 1,:])
        # Compute one-sided differences for the edges
        _derivative[0,:] = (_y[1,:] - _y[0,:]) / (_x[1,:] - _x[0,:])  # One-sided difference at the left edge
        _derivative[-1,:] = (_y[-1,:] - _y[-2,:]) / (_x[-1,:] - _x[-2,:])  # One-sided difference at the right edge
    elif axis == 1:
        I = _x.shape[1]
        for i in range(1, I - 1):
            _derivative[:,i] = (_y[:,i + 1] - _y[:,i - 1]) / (_x[:,i + 1] - _x[:,i - 1])
        # Compute one-sided differences for the edges
        _derivative[:,0] = (_y[:,1] - _y[:,0]) / (_x[:,1] - _x[:,0])  # One-sided difference at the left edge
        _derivative[:,-1] = (_y[:,-1] - _y[:,-2]) / (_x[:,-1] - _x[:,-2])  # One-sided difference at the right edge
    else:
        print('Not implemented')
        print(1/0)
    return _derivative


def sph2car(theta, phi):
    #Theta and phi in degrees. Make rotation matrix to convert a vector with 
    # spherical components at location (theta,phi) to ECEF cartesian components
    theta = np.radians(theta)
    phi = np.radians(phi)
    R = np.array([[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)], 
                  [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi) ],
                  [np.cos(theta),             -np.sin(theta),            0           ]])
    return R


def get_uhat(az, el):
    # Unit vector of beam direction defined by azimuth and elevation angle, 
    # expressed in spherical components. Refer to the radar site location.
    # input az, el in degrees
    norm = np.sqrt(1-(np.sin(np.radians(el)))**2)
    ur = np.sin(np.radians(el))
    utheta = -np.cos(np.radians(az))*norm
    uphi = np.sin(np.radians(az))*norm
    return np.array([ur, utheta, uphi])

def car2sph(x,y,z):
    # Convert the cartesian ECEF location to spherical (r, theta, phi) coordinates
    # Returns components in degrees.
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.degrees(np.arccos(z/r))
    phi = np.degrees(np.arctan2(y,x))
    return (r, theta, phi)


def unitvec_rthetaphi(lon, lat):
    '''
    Provide the r,theta,phi components of ECEF (x,y,z) unit vectors at the 
    input locations. Will work on both geographic and geomagnetic bases.

    Parameters
    ----------
    lon : array-like
        in degrees., shape N
    lat : array-like
        in degrees, shape N

    Returns
    -------
    ex, ey, ez : tuple of 3 arrays, each of shape 3 x N

    '''
    _theta=np.pi/2-lat.flatten()*np.pi/180
    _phi=lon.flatten()*np.pi/180
    lx = lon.size
    ex=np.empty((lx,3))
    ey=np.empty((lx,3))
    ez=np.empty((lx,3))
    
    # unit vectors in ECEF Cartesian geographic
    ex[:,0]=np.sin(_theta)*np.cos(_phi)
    ex[:,1]=np.cos(_theta)*np.cos(_phi)
    ex[:,2]=-np.sin(_phi)
    ey[:,0]=np.sin(_theta)*np.sin(_phi)
    ey[:,1]=np.cos(_theta)*np.sin(_phi)
    ey[:,2]=np.cos(_phi)
    ez[:,0]=np.cos(_theta)
    ez[:,1]=-np.sin(_theta)
    ez[:,2]=np.zeros(_theta.shape)    
    
    return (ex, ey, ez)


def enu2xyz(enu_vec, lon, lat):
    """
    Convert vector with ENU components, located at lon, lat to 
    cartesian components (ECEF). Both input locations and components must be in the
    same coordinate system (e.g. geographic or geomagnetic). The returned cartesian
    components will be in the same system

    Parameters
    ----------
    enu_vec : 2D array
        The vectors with ENU components to convert. Shape must be N,3
    lon : 1D array
        longitudes [degrees] of the location of the ENU vectors of size N.
    lat : 1D array
        latitudes [degrees] of the location of the ENU vectors of size N.

    Returns
    -------
    xyz_vec, a 2D array of size N,3 of the converted vectors

    """

    # Project each component (east,north, up) onto the ECEF directions (x,y,z)
    xhat, yhat, zhat = unitvec_rthetaphi(lon, lat)
    vx = xhat[:,0]*enu_vec[:,2] - xhat[:,1]*enu_vec[:,1] + xhat[:,2]*enu_vec[:,0]
    vy = yhat[:,0]*enu_vec[:,2] - yhat[:,1]*enu_vec[:,1] + yhat[:,2]*enu_vec[:,0]
    vz = zhat[:,0]*enu_vec[:,2] - zhat[:,1]*enu_vec[:,1] + zhat[:,2]*enu_vec[:,0]

    xyz_vec = np.vstack((vx,vy,vz)).T
    
    return xyz_vec


def xyz2enu(xyz_vec, lon, lat):
    """
    Convert vector with cartesian components, located at lon, lat to 
    local ENU components. Both input locations and components must be in the
    same coordinate system (e.g. geographic or geomagnetic). The returned ENU
    components will be in the same system

    Parameters
    ----------
    xyz_vec : 2D array
        The vectors with xyz components to convert. Shape must be N,3
    lon : 1D array
        longitudes [degrees] of the location of the ENU vectors of size N.
    lat : 1D array
        latitudes [degrees] of the location of the ENU vectors of size N.

    Returns
    -------
    enu_vec, a 2D array of size N,3 of the converted vectors

    """

    # Project each component (xyz) onto the local ENU directions
    er, etheta, ephi = unitvec_xyz(lon, lat)
    vr = er[:,0]*xyz_vec[:,0] + er[:,1]*xyz_vec[:,1] + er[:,2]*xyz_vec[:,2]
    vtheta = etheta[:,0]*xyz_vec[:,0] + etheta[:,1]*xyz_vec[:,1] + etheta[:,2]*xyz_vec[:,2]
    vphi = ephi[:,0]*xyz_vec[:,0] + ephi[:,1]*xyz_vec[:,1] + ephi[:,2]*xyz_vec[:,2]
    enu_vec = np.vstack((vphi,-vtheta,vr)).T

    return enu_vec 

def unitvec_xyz(lon, lat):
    '''
    Provide the x,y,z components of r, theta, phi (x,y,z) unit vectors at the 
    input locations. Will work on both geographic and geomagnetic bases.

    Parameters
    ----------
    lon : array-like
        in degrees., shape N
    lat : array-like
        in degrees, shape N

    Returns
    -------
    rhat, thetahat, phihat : tuple of 3 arrays, each of shape 3 x N

    '''
    _theta=np.pi/2-lat.flatten()*np.pi/180
    _phi=lon.flatten()*np.pi/180
    lx = lon.size
    er=np.empty((lx,3))
    et=np.empty((lx,3))
    ep=np.empty((lx,3))
    
    # unit vectors in ECEF Cartesian geographic
    er[:,0]=np.sin(_theta)*np.cos(_phi)
    er[:,1]=np.sin(_theta)*np.sin(_phi)
    er[:,2]=np.cos(_theta)
    et[:,0]=np.cos(_theta)*np.cos(_phi)
    et[:,1]=np.cos(_theta)*np.sin(_phi)
    et[:,2]=-np.sin(_theta)
    ep[:,0]=-np.sin(_phi)
    ep[:,1]=np.cos(_phi)
    ep[:,2]=np.zeros(_theta.shape)    
    
    return (er, et, ep)


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


def enu2gemini_rot(lon, lat, geographic = True):
    '''
    Make 3x3 matrix G such that G.dot(v_enu), where v_enu is a 3 element enu vector
    gives the 3 element vector in GEMINI basis (x1,x2,x3) where x1 is the field
    aligend direction, and x2 and x3 span the perp plane.
    
    Input must be in degrees.
    
    Parameters
    -----------
    lat : 1D array
        Latitudes in degrees
    lon : 1D array
        Longitudes in degrees
    geographic : boolean
        Specifies if input latitudes are in geographic coordinates. If not, they
        must be in centered dipole coords. In both cases, the returned matrix will
        take a vector of ENU components into GEMINI components. Actually not sure
        if this is a redundant feature.
        
    Return
    ---------
    3x3xN array containing the N rotation matrices
    
    '''
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
    
    Gs = []
    N = lat.size
    for i in range(N):
        G_ecef = np.vstack((egmlon[i,:],egmlat[i,:],egmalt[i,:]))
        G_g = np.hstack((e1[i,:][:,np.newaxis],e2[i,:][:,np.newaxis],e3[i,:][:,np.newaxis]))
        G = (G_ecef @ G_g).T
        Gs.append(G)
    Gs = np.swapaxes(np.swapaxes(np.array(Gs),1,0),2,1)    
    
    return Gs


def sph_to_car(sph, deg = True):
    """ convert from spherical to cartesian coordinates

        input: 3 X N array:
           [r1    , r2    , ..., rN    ]
           [colat1, colat2, ..., colatN]
           [lon1  , lon2  , ..., lonN  ]

        output: 3 X N array:
           [x1, x2, ... xN]
           [y1, y2, ... yN]
           [z1, z2, ... zN]

        deg = True if lat and lon are given in degrees, 
              False if radians
    """
    d2r = np.pi/180

    r, theta, phi = sph

    if deg == False:
        conv = 1.
    else:
        conv = d2r


    return np.vstack((r * np.sin(theta * conv) * np.cos(phi * conv), 
                      r * np.sin(theta * conv) * np.sin(phi * conv), 
                      r * np.cos(theta * conv)))


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
    xhat, yhat, zhat = unitvec_rthetaphi(glon, glat)
    vx = xhat[:,0]*enugg_vec[:,2] - xhat[:,1]*enugg_vec[:,1] + xhat[:,2]*enugg_vec[:,0]
    vy = yhat[:,0]*enugg_vec[:,2] - yhat[:,1]*enugg_vec[:,1] + yhat[:,2]*enugg_vec[:,0]
    vz = zhat[:,0]*enugg_vec[:,2] - zhat[:,1]*enugg_vec[:,1] + zhat[:,2]*enugg_vec[:,0]
    # 2) rotate gg->gm
    v_ecef_gm = (convert.Rgg2gm()@(np.vstack((vx,vy,vz)))).T
    # 3) go back to dipole east,north components    
    rhat, thetahat, phihat = unitvec_xyz(np.degrees(phi), 90-np.degrees(theta))
    vr = rhat[:,0]*v_ecef_gm[:,0] + rhat[:,1]*v_ecef_gm[:,1] + rhat[:,2]*v_ecef_gm[:,2]
    vtheta = (thetahat[:,0]*v_ecef_gm[:,0] + thetahat[:,1]*v_ecef_gm[:,1] + thetahat[:,2]*v_ecef_gm[:,2])
    vphi = phihat[:,0]*v_ecef_gm[:,0] + phihat[:,1]*v_ecef_gm[:,1] + phihat[:,2]*v_ecef_gm[:,2]
    enugm_vec = np.vstack((vphi,-vtheta,vr)).T
    
    return enugm_vec