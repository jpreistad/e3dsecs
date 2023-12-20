#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:32:35 2023

@author: jone

This file contain functions that preform various coordinate transformations and
numerical tools relevant for working with GEMINI and the 3D SECS model for 
estimating 3D current densities, but not directly linked to the GEMINI grid.

"""

from gemini3d.grid.convert import unitvecs_geographic
import numpy as np
from gemini3d.grid.gridmodeldata import model2geogcoords, model2pointsgeogcoords, geog2dipole
from gemini3d.grid.convert import geomag2geog, geog2geomag
import gemini3d.grid.convert as convert
import gemini_tools

RE = gemini_tools.RE


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