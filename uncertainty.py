#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:48:55 2023

@author: jone

This file contain functions used to calculate and propagate uncertainties related
to EISCAT_3D like measurements from the GEMINI model. The measurement uncertainties
at its respective location is first obtained from the E3DOUBT package, and then
further propagated for use with the 3DSECS technique. 

"""

import e3doubt 
import numpy as np
import dipole
import scipy
from scipy.stats import multivariate_normal

try:
    from . import secs3d
    from . import gemini_tools
    from . import coordinates
    RE = gemini_tools.RE

except:
    import secs3d
    from gemini_tools import RE as RE
    import coordinates

RE = 6371.2 #Earth radius in km


def get_datacov_e3doubt(ddict, intsec = 5*60):
    '''
    Thus function calls E3DOUBT to estimate datacovariance matrix at each
    sample location in ddict.

    Parameters
    ----------
    ddict : dict
        The dict containing the sampling from GEMINI. Must contain the
        elevation, azimuth, and altitude information of the sampling.
    intsec : int or float, optional
        Number of seconds of integration time to be shared for all beams. 
        The default is 5*60 sec.

    Returns
    -------
    ddict : dict
        Updated to containt the 3x3 covariance matrices for all N 
        obsercations, stored as a 3x3xN array. Also contain the 
        variances of the electron density.
    '''
    
    exp = e3doubt.Experiment(el=ddict['el'], az = ddict['az'],h=ddict['alts'], 
                             transmitter=('ski_mod', 67.2,23.7), 
                             receivers=[('ski_mod', 67.2,23.7), ('kai_mod', 66.18,22.54), ('krs_mod', 66.25,25.92)])

    # exp.set_ionos('nuin',0)

    exp.run_models()
    
    #replace nans
    nans = ~np.isfinite(ddict['ne'])
    ddict['ne'][nans] = 1e9
    ddict['Te'][nans] = 1000
    ddict['Ti'][nans] = 1000

    exp.set_ionos('ne',ddict['ne'])    
    exp.set_ionos('Te',ddict['Te'])    
    exp.set_ionos('Ti',ddict['Ti'])    

    uncert =exp.calc_uncertainties(integrationsec=intsec)

    cov = exp.get_velocity_cov_matrix()

    ddict['cov_vi'] = cov
    ddict['var_ne'] = uncert.dnemulti.values

    return ddict    

def remove_bad(ddict):
    '''
    Remove observation that have undefined covariaces from E3DOUBT

    Parameters
    ----------
    ddict : dict
        Contain the GEMINI data and E3DOUBT covariances.

    Returns
    -------
    ddict, removed for undefined variances.

    '''
    N = ddict['lat'].size
    keys = ddict.keys()
    updatekeys = []
    updateshapes = []
    for key in keys:
        if (type(ddict[key]) is not int) and (type(ddict[key]) is not bool):
            updatekeys.append(key)
            updateshapes.append(ddict[key].shape)
    
    bads = np.zeros(N)
    for n in range(N):
        nn = np.abs(ddict['cov_vi'][:,:,n])
        if np.max(nn) > 1e10:
            bads[n] = 1
            
    use = np.isfinite(ddict['cov_vi'][0,0,:]) & (bads==0)

    for i, key in enumerate(updatekeys):
        sh = updateshapes[i]
        ss = np.where(np.array(sh)==N)[0]
        if ss.size == 1:
            if ss[0] == 0:
                ddict[key] = ddict[key][use]
            if ss[0] == 2:
                ddict[key] = ddict[key][:,:,use]
                                
    return ddict
            


def add_noise(ddict, minalt):
    '''
    Add noise to the samples from GEMINI by using the covariace matrix obtained 
    from E3DOUBT.
    
    Noise must be added to the following two parameters in ddict: 
        1) vperpmapped(e,n,u): This is what the lompe-representation uses. Here 
        noise is added to all ovservations, also in the E-region. These will later
        be ignored when estimating the lompe E-field.
        
        2) vperp(e,n,u): This is what is used as input to the 3D reconsrtuction to make
        j_perp estimate, together with vperp_electrons obtained from 1)
    
        3) A flag in ddict is added to secify that noise has been added
    
    This function assumes that all vector quantities in ddict are geographic
    components unless explicitly stated something else.
    
    Since we want to generate the noise by sampling from the multivariate normal
    distribution with the prescribed covariance, we get an apparent issue for
    the _perp quantities when first propagaing the 3D covariance from E3DOUBT (ENU)
    into cov_perp_enu. E.g. let v_perp = G.dot(v). cov(v) is given from E3DOUBT.
    cov(v_perp) = cov(G.dot(v)) = G @ cov(v) @ G.T
    However, since G is the projection matrix removing the field-aligned component,
    the 3x3 element G matrix will only have rank=2, and hence the 3x3 cov(v_perp) 
    will also be rank deficient. This leads to the multivariate_normal() function 
    complaining about singular matrix when using the 3x3 cov(v_perp) as argument.
    JPR is not completely sure about the implications of having a rank deficient
    covariance matrix, but I see no obvious reasons why it can not be used as we
    intend to propagate the variances through our model. SH suggested that to 
    get around the issue with applying the multivariate_normal() function, we 
    instead apply this function to generate the desired noise in only the perp
    direction, hence, providing it with a 2x2 cov(v_perp) matrix whose dimensions
    are e.g. the x2 and x3 dimensions used in the GEMINI grid, spanning the _perp
    plane. Then, the new obsevations, with the added noise, are projected back
    into 3D ENU space, again leaving its associated covatiance matrix rank 
    deficient. A consequence of this is that not all information from the original
    cov matrix is used when generating the noise, and that no noise will be added
    in the field-aligned direction.
    

    Parameters
    ----------
    ddict : dict
        Contain the GEMINI observarions and the covariances.
    minalt : int or float
        The altitude above where we assume ExB drift.

    Returns
    -------
    ddict with updated values for the vperpmapped and vperp ion velocities.

    '''
    
    N = ddict['lat'].size
    
    # 0) obtain matrices that map the covairances:
    Gperpmapped = _make_vperpmappedG(ddict, minalt)
    Gperp = _make_vperpG(ddict)
    Gs = coordinates.enu2gemini_rot(ddict['lon'], ddict['lat']) #rotation matrix from ENU2GEMINI
    _perp = np.eye(3)
    _perp[0,0] = 0 # select only perp components in GEMINI basis
  
    # Arrays to hold the result
    vperp_enu_noisy = np.zeros((N,3))
    vperpmapped_enu_noisy = np.zeros((N,3))

    for i in range(N):     
        cov = ddict['cov_vi'][:,:,i]

        # 1) Add noise to vperpmapped
        # 1.1: obtain covariance of vperpmapped(e,n,u) observations:
        Gi = Gperpmapped[:,:,i]
        # Propagate covariance of v into the mapped vperp at minalt
        cov_vperpmapped_enu = Gi.dot(cov).dot(Gi.T) 
        # Map vperpmaped_enu into GEMINI basis
        vperpmapped_enu = np.hstack((ddict['vperpmappede'][i],ddict['vperpmappedn'][i],ddict['vperpmappedu'][i]))
        vperpmapped_g = Gs[:,:,i].dot(vperpmapped_enu)
        # Map covariance of vperpmapped into GEMINI basis
        cov_vperpmapped_g = _perp @ (Gs[:,:,i] @ cov_vperpmapped_enu @ Gs[:,:,i].T) @ _perp.T
        # Make some noise
        _noise = multivariate_normal(mean=np.zeros(2), cov=cov_vperpmapped_g[1:,1:])
        noisy_obs = np.hstack((0,vperpmapped_g[1:] + _noise.rvs()))
        # Project back into ENU space. Inverse transformation is the transpose
        # cov_vperp_enu = Gs[:,:,i].T @ cov @ Gs[:,:,i]
        vperpmapped_enu_noisy[i,:] = Gs[:,:,i].T.dot(noisy_obs)
        
        # 2) Add noise to vperp
        # 2.1: obtain covariance of vperp(e,n,u) observations:
        Gi = Gperp[:,:,i]
        # Propagate covariance of v into vperp 
        cov_vperp_enu = Gi.dot(cov).dot(Gi.T)
        # Map vperp_enu into GEMINI basis
        vperp_enu = np.hstack((ddict['vperpe'][i],ddict['vperpn'][i],ddict['vperpu'][i]))
        vperp_g = Gs[:,:,i].dot(vperp_enu)
        # vperp_g = np.hstack((0,ddict['v2'][i],ddict['v3'][i])) # true value
        # Map covariance of vperp into GEMINI basis
        cov_vperp_g = _perp @ (Gs[:,:,i] @ cov_vperp_enu @ Gs[:,:,i].T) @ _perp.T
        # Make some noise
        _noise = multivariate_normal(mean=np.zeros(2), cov=cov_vperp_g[1:,1:])
        noisy_obs = np.hstack((0,vperp_g[1:] + _noise.rvs()))
        # Project back into ENU space. Inverse transformation is the transpose
        # cov_vperp_enu = Gs[:,:,i].T @ cov @ Gs[:,:,i]
        vperp_enu_noisy[i,:] = Gs[:,:,i].T.dot(noisy_obs)
        
    # update ddict with the added noise
    ddict['vperpmappede'] = vperpmapped_enu_noisy[:,0]
    ddict['vperpmappedn'] = vperpmapped_enu_noisy[:,1]
    ddict['vperpmappedu'] = vperpmapped_enu_noisy[:,2]
    ddict['vperpe'] = vperp_enu_noisy[:,0]
    ddict['vperpn'] = vperp_enu_noisy[:,1]
    ddict['vperpu'] = vperp_enu_noisy[:,2]
    

    # 3) Add the noise flag
    ddict['noise_added'] = True
    
    return ddict

def _make_vperpmappedG(ddict, minalt):
    # Make the (3,3) matrix that act on a velocity measurement v=(ve,vn,vu) in
    # geographic components to produce
    # the value of the velocity perpendicular to B, mapped to the altitude minalt [km]
    # assuming no collissions and that E_par = 0: vperpmapped(e,n,u). The matrices are
    # stacked in the third dimension for each observation.
    # minalt is the altitude above where we assume ExB drift
    
    N = ddict['lat'].size

    # First, calculate d vectors at the observation locations of v
    dip = dipole.Dipole(dipole_pole=((90-11),289))
    d1, d2, d3, _1, _2, _3 = dip.get_apex_base_vectors_geo(ddict['lon'], 
                                ddict['lat'], ddict['alt']+RE, R=RE)
    
    # Calculate e vectors at the mapped locations
    _1, _2, _3, e1, e2, e3 = dip.get_apex_base_vectors_geo(ddict['mappedglon'], 
                        ddict['mappedglat'], np.ones(N)*(RE+minalt), R=RE)

    Gs = []
    for i in range(N):
        Gi = np.zeros((3, 3))
        Gi[0,:] = np.vstack((d1[0,i]*e1[0,i]+d2[0,i]*e2[0,i], d1[1,i]*e1[0,i]+d2[1,i]*e2[0,i], d1[2,i]*e1[0,i]+d2[2,i]*e2[0,i])).T
        Gi[1,:] = np.vstack((d1[0,i]*e1[1,i]+d2[0,i]*e2[1,i], d1[1,i]*e1[1,i]+d2[1,i]*e2[1,i], d1[2,i]*e1[1,i]+d2[2,i]*e2[1,i])).T
        Gi[2,:] = np.vstack((d1[0,i]*e1[2,i]+d2[0,i]*e2[2,i], d1[1,i]*e1[2,i]+d2[1,i]*e2[2,i], d1[2,i]*e1[2,i]+d2[2,i]*e2[2,i])).T
        Gs.append(Gi)
    Gs = np.swapaxes(np.swapaxes(np.array(Gs),1,0),2,1)    

    return Gs

def _make_vperpG(ddict):
    # Make the (3,3) matrix that act on a velocity measurement v=(ve,vn,vu) in
    # geographic components to produce
    # the value of the velocity perpendicular to B: vperp(e,n,u). The matrices are
    # stacked in the third dimension for each observation
    
    N = ddict['lat'].size

    # First, calculate d vectors at the observation locations of v
    dip = dipole.Dipole(dipole_pole=((90-11),289))
    d1, d2, d3, e1, e2, e3 = dip.get_apex_base_vectors_geo(ddict['lon'], 
                                ddict['lat'], ddict['alt']+RE, R=RE)
    Gs = []
    for i in range(N):
        Gi = np.zeros((3, 3))
        Gi[0,:] = np.vstack((d1[0,i]*e1[0,i]+d2[0,i]*e2[0,i], d1[1,i]*e1[0,i]+d2[1,i]*e2[0,i], d1[2,i]*e1[0,i]+d2[2,i]*e2[0,i])).T
        Gi[1,:] = np.vstack((d1[0,i]*e1[1,i]+d2[0,i]*e2[1,i], d1[1,i]*e1[1,i]+d2[1,i]*e2[1,i], d1[2,i]*e1[1,i]+d2[2,i]*e2[1,i])).T
        Gi[2,:] = np.vstack((d1[0,i]*e1[2,i]+d2[0,i]*e2[2,i], d1[1,i]*e1[2,i]+d2[1,i]*e2[2,i], d1[2,i]*e1[2,i]+d2[2,i]*e2[2,i])).T
        Gs.append(Gi)
    Gs = np.swapaxes(np.swapaxes(np.array(Gs),1,0),2,1)    

    return Gs


def make_datacov_lompe(ddict, grid_l, minalt):
    # This function will map the covariance of v from the observation locations
    # as contained in datadict['cov_vi'] and obtained wiht E3DOUBT, to covariances
    # of vperp at maph height, and put it into datadict['cov_viperp_mapped']
    # minalt is the altitude above where we assume ExB drift


    N = ddict['lat'].size

    Gs = _make_vperpmappedG(ddict, minalt)

    mapped_cov = []

    for i in range(N):
        Gi = Gs[:,:,i]
        
        # Propagate covariance of v into of the mapped vperp at minalt
        cov = ddict['cov_vi'][:,:,i]
        mapped_cov_i = Gi.dot(cov).dot(Gi.T)
        
        mapped_cov.append(mapped_cov_i)
        
    mapped_cov = np.swapaxes(np.swapaxes(np.array(mapped_cov),1,0),2,1)    
    
    ddict['cov_viperp_mapped'] = mapped_cov
    
    # Remove observations below minalt from ddict, and nans in mapped locations
    lon = ddict['mappedglon']
    lat = ddict['mappedglat'] 
    use = np.isfinite(ddict['mappedglat']) & \
            (np.isfinite(ddict['cov_viperp_mapped'][0,0,:])) & \
            (grid_l.ingrid(lon, lat)) & (ddict['alt']>=minalt)
    keys = list(ddict.keys())
    drops = ['maph', 'az', 'el', 'alts', 'cov_vi', 'cov_viperp_mapped', 'noise_added']
    __ = [keys.remove(i) for i in drops]
    for key in keys:
        ddict[key] = ddict[key][use]
    ddict['cov_vi'] = ddict['cov_vi'][:,:,use]
    ddict['cov_viperp_mapped'] = ddict['cov_viperp_mapped'][:,:,use]

    return ddict    

def make_cmpost(model, ddict, l1=1, l2=1):
    '''
    Make posterior model covatiance matrix based on data covatiance. This function 
    is needed as Lompe at present does not include off-diagonal elements in data
    covatiance matrix.

    Parameters
    ----------
    model : lompe model object
        DESCRIPTION.
    datadict : dict
        The data and locations used to constrain the Lompe representation.

    Returns
    -------
    Cmpost.

    '''
    ndim = 3    # use also vertival component in inversion, to take advantage of
                # the knowledge of covariance. The covariance is also favorable
                # in this direction
    
    # Lompe describe horizontal part of v_perp.
    lon = ddict['mappedglon']
    lat = ddict['mappedglat']        
    
    # skip data points that are outside biggrid:
    use = model.biggrid.ingrid(lon, lat)
    lon = lon[use]
    lat = lat[use]
    coords = {'lon':lon, 'lat':lat}
    N = sum(use)

    #Data vector and covariance
    vperpgg_enu = np.vstack((ddict['vperpmappede'][use],ddict['vperpmappedn'][use], \
                             ddict['vperpmappedu'][use])).T   
    cov = ddict['cov_viperp_mapped'][:,:,use]
    Cd = np.zeros((N*ndim,N*ndim))
    fudge = 1#0.99 # To make the invesion not fail (Spencers fix)
        
    
    # First build full data covariance matrix
    # Spatial weighting scheme is part of data covariance matrix
    data_density_weight = True
    if data_density_weight:
        bincount = model.biggrid.count(lon, lat)
        i, j = model.biggrid.bin_index(lon, lat)
        spatial_weight = 1. / np.maximum(bincount[i, j], 1)
        spatial_weight[i == -1] = 1
        spatial_weight = np.tile(spatial_weight, ndim)
    else:
        spatial_weight = np.ones(N*ndim)
    # Make the SECS G matrix
    Gen = model.matrix_func['convection'](**coords)
    if ndim == 3:
        Be, Bn, Bu = ddict['Be'][use], ddict['Bn'][use], ddict['Bu'][use]
        # Use v_perp.dot(B) = 0 to get up component of vperp
        Gu =  -np.einsum('ij,i->ij', Gen[0], Be/Bu) - \
                np.einsum('ij,i->ij', Gen[1], Bn/Bu)
        G = np.vstack((Gen[0], Gen[1], Gu))
        d = np.hstack((vperpgg_enu[:,0],vperpgg_enu[:,1], vperpgg_enu[:,2]))
        
        covar_ee = cov[0,0,:]
        covar_nn = cov[1,1,:]
        covar_uu = cov[2,2,:]
        covar_en = cov[0,1,:]
        covar_ne = cov[1,0,:]
        covar_eu = cov[0,2,:]
        covar_ue = cov[2,0,:]
        covar_nu = cov[1,2,:]
        covar_un = cov[2,1,:]
        assert np.all(np.isclose(covar_en,covar_ne)) # sanity check
        assert np.all(np.isclose(covar_eu,covar_ue)) # sanity check
        assert np.all(np.isclose(covar_nu,covar_un)) # sanity check
        Cd[:N,:N] = np.diag(covar_ee)# upper left, all e components
        Cd[N:2*N,N:2*N] = np.diag(covar_nn)# center block, all n components
        Cd[2*N:3*N,2*N:3*N] = np.diag(covar_uu)# bottom right block, all u components
        Cd[:N,N:2*N] = np.diag(covar_en)# * fudge# upper center block, e-n covariance
        Cd[N:2*N,0:N] = np.diag(covar_en)# middle row, left block, n-e covariance
        Cd[:N,2*N:3*N] = np.diag(covar_eu)# upper right block, e-u covariance
        Cd[2*N:3*N,0:N] = np.diag(covar_eu)# lower row, left block, u-e covariance
        Cd[N:2*N,2*N:3*N] = np.diag(covar_nu)# middle row, right block, n-u covariance
        Cd[2*N:3*N,N:2*N] = np.diag(covar_nu)# lower row, center block, u-n covariance  
        Cdinv = np.linalg.pinv(Cd)
        # It turns out that Cd is often singular, most likely due to multiple 
        # observations being near identical in locations and value. According
        # to ChatGPT, it is common to use an apporach like pinv() in such cases.
        # If not, one could try to indicate the duplicate observations and remove
        # them. SH suggested to tweak Cd just a tiny bit to make it non-singular, and 
        # use the inv() function instead. I have seen situations when the result of
        # the two approaches is vastly different. This is apparently not an issue
        # when using all 3 components.
        
    elif ndim == 2:
        G = np.vstack((Gen[0], Gen[1]))
        d = np.hstack((vperpgg_enu[:,0],vperpgg_enu[:,1]))
        covar_ee = cov[0,0,:]
        covar_nn = cov[1,1,:]
        covar_en = cov[0,1,:]
        covar_ne = cov[1,0,:]
        assert np.all(np.isclose(covar_en,covar_ne)) # sanity check
        Cd[:N,:N] = np.diag(covar_ee) # upper left, all e components
        Cd[N:2*N,N:2*N] = np.diag(covar_nn) # lower right, all n components
        Cd[:N,N:2*N] = np.diag(covar_en) # upper right block, e-n covariance
        Cd[N:2*N,0:N] = np.diag(covar_en) # bottom row, left block, n-e covariance
        Cdinv = np.linalg.inv(Cd)
    else:
        print('ndim must be 2 or 3')
        print(1/0)
    

    # i_neg = Cd<-1e6
    # i_pos = Cd>1e6
    # Cd[i_neg] = -1e6 
    # Cd[i_pos] = 1e6 

    # Apply the spatial weighting scheme
    Cdinv = np.einsum('ij,i->ij', Cdinv, spatial_weight)
    
    # Regularization
    GTG = G.T@Cdinv@G
    gtg_mag = np.median(np.diagonal(GTG))
    ltl_mag = np.median(model.LTL.diagonal())   
    GG = GTG + l1*gtg_mag * np.eye(GTG.shape[0]) + l2 * gtg_mag / ltl_mag * model.LTL
    GTd = G.T.dot(Cdinv).dot(d)
    
    # Posterior model covariance matrix
    Cmpost = scipy.linalg.lstsq(GG, np.eye(GG.shape[0]))[0]
    
    # Solution
    m = Cmpost.dot(GTd)

    return (m,Cmpost)

def make_lompe_v_cov(model, ddict, Cmpost, dipole_lompe=True):
    '''
    Estimate covariances of the v_perp as predicted by lompe, at maph altitude 

    Parameters
    ----------
    model : lompe model object
        DESCRIPTION.
    datadict: dict
        The datadict of the all observed locations to evaluate Lompe with 

    Returns
    -------
    updated datadict.

    '''
    N = ddict['mappedglon'].size
    
    # Lompe describe horizontal part of v_perp.
    lon = ddict['mappedglon']
    lat = ddict['mappedglat']        
    coords = {'lon':lon, 'lat':lat}
    
    # Covariance of v_perp field as described by Lompe
    Gen = model.matrix_func['convection'](**coords)  # Ge and Gn matrices
    Be, Bn, Bu = ddict['Be'], ddict['Bn'], ddict['Bu']
    # Use v_perp.dot(B) = 0 to get up component of vperp
    Gu =  -np.einsum('ij,i->ij', Gen[0], Be/Bu) - \
            np.einsum('ij,i->ij', Gen[1], Bn/Bu)
    # Project lompe model cov into lompe vperp prediction covariance
    G = np.vstack((Gen[0],Gen[1],Gu)) # this matrix relates lompe model parameters
                                      # to enu components of vperp
    v_lompe_cov = G.dot(Cmpost).dot(G.T)
    v_cov = np.zeros((3,3,N))
    v_cov[0,0,:] = np.diag(v_lompe_cov[0:N,0:N])
    v_cov[1,0,:] = np.diag(v_lompe_cov[N:2*N,0:N])
    v_cov[0,1,:] = np.diag(v_lompe_cov[0:N,N:2*N])
    v_cov[1,1,:] = np.diag(v_lompe_cov[N:2*N,N:2*N])
    v_cov[2,0,:] = np.diag(v_lompe_cov[2*N:3*N,0:N])
    v_cov[0,2,:] = np.diag(v_lompe_cov[0:N,2*N:3*N])
    v_cov[2,2,:] = np.diag(v_lompe_cov[2*N:3*N,2*N:3*N])
    v_cov[2,1,:] = np.diag(v_lompe_cov[2*N:3*N,N:2*N])
    v_cov[1,2,:] = np.diag(v_lompe_cov[N:2*N,2*N:3*N])
    
    v_lompe = G.dot(model.m)
    ddict['v_lompe'] = np.vstack((v_lompe[0:N],v_lompe[N:2*N],v_lompe[2*N:3*N]))
    ddict['covVlompe'] = v_cov
    
    return ddict
    

def make_ve_cov(model, ddict):
    '''
    Make covaiance matrix of v_perp as represented by Lompe, but mapped back to its
    original observation location. This is therefore intended to represent the 
    covariance of the estimated electron perp velocity of each observation, to 
    be used to estimate j_perp with (vi_perp - ve_perp).

    Parameters
    ----------
    model : lompe model object
        DESCRIPTION.
    datadict : dict
        Data sampled from Lompe, here also including uncertaintied from E3DOUBT.

    Returns
    -------
    Updated ddict.

    '''

    N = ddict['lat'].size

    # First, calculate d vectors at the layer of Lompe representation
    dip = dipole.Dipole(dipole_pole=((90-11),289))
    d1, d2, d3, _1, _2, _3 = dip.get_apex_base_vectors_geo(ddict['lon'], 
                                ddict['lat'], np.ones(N)*(RE+ddict['maph']), R=RE)
    
    # Calculate e vectors at the original measurement locations
    _1, _2, _3, e1, e2, e3 = dip.get_apex_base_vectors_geo(ddict['mappedglon'], 
                        ddict['mappedglat'], ddict['alt']+RE, R=RE)
    
    cov_ve = []
    vmappede = []
    vmappedn = []
    vmappedu = []
    for i in range(N):
        Gi = np.zeros((3, 3))
        Gi[0,:] = np.vstack((e1[0,i]*d1[0,i]+e2[0,i]*d2[0,i], 
                             e1[1,i]*d1[0,i]+e2[1,i]*d2[0,i], 
                             e1[2,i]*d1[0,i]+e2[2,i]*d2[0,i])).T
        Gi[1,:] = np.vstack((e1[0,i]*d1[1,i]+e2[0,i]*d2[1,i], 
                             e1[1,i]*d1[1,i]+e2[1,i]*d2[1,i], 
                             e1[2,i]*d1[1,i]+e2[2,i]*d2[1,i])).T
        Gi[2,:] = np.vstack((e1[0,i]*d1[2,i]+e2[0,i]*d2[2,i], 
                             e1[1,i]*d1[2,i]+e2[1,i]*d2[2,i], 
                             e1[2,i]*d1[2,i]+e2[2,i]*d2[2,i])).T
        # Gs.append(Gi)
        vi = np.vstack((ddict['v_lompe'][0,i],ddict['v_lompe'][1,i],
                        ddict['v_lompe'][2,i]))
        # vmappedi = np.einsum('ij,jk->i', Gi, vi)
        vmappedi = Gi.dot(vi)
        vmappede.append(vmappedi[0,0])
        vmappedn.append(vmappedi[1,0])
        vmappedu.append(vmappedi[2,0])
        
        # Propagate covariance of V_lompe at maph altitude into covariance of v at
        # observation locations, cov_ve, representing the covariance of the
        # perpendicular electron velocity vector of each observation
        cov_lompe = ddict['covVlompe'][:,:,i]
        mapped_cov_ve = Gi.dot(cov_lompe).dot(Gi.T)
        cov_ve.append(mapped_cov_ve)
        
    cov_ve = np.swapaxes(np.swapaxes(np.array(cov_ve),1,0),2,1)   
    ddict['cov_ve'] = cov_ve
    ddict['vperp_electron'] = np.vstack((vmappede,vmappedn,vmappedu))
    
    return ddict


def make_cov_vi_perp(ddict):
    # Map the covariance matrix from E3DOUBT to covariance of 
    # vi_perp, which is what is needed when estimating covariance
    # of j_perp using (vi-ve)_perp

    N = ddict['lat'].size
    
    Gperp = _make_vperpG(ddict) # matrices that extract perp part of v
    
    cov_vi_perp = []
    
    for i in range(N):
        Gperp_i = Gperp[:,:,i]
        cov_vi_perp.append(Gperp_i @ ddict['cov_vi'][:,:,i] @ Gperp_i.T)
    cov_vi_perp = np.swapaxes(np.swapaxes(np.array(cov_vi_perp),1,0),2,1)    
    ddict['cov_vi_perp'] = cov_vi_perp
    return ddict


def make_cov_jperp(ddict):
    '''
    Make covariance matrix of jperp based on variance of n and covariance of
    vi_perp and ve_perp using jperp = ne(vi_perp - ve_perp)

    Parameters
    ----------
    ddict : dict
        Containing all relevant data sampled from GEMINI, including its variances.

    Returns
    -------
    Updated ddict.

    '''  
    
    #Covariance of vi - ve = cov(vi) + cov(ve)
    ddict = make_cov_vi_perp(ddict)
    cov_vive = ddict['cov_vi_perp'] + ddict['cov_ve']
    e = 1.6e-19 #electron charge
    cov_jperp = []
    # cov_jperp2 = []
    
    N = ddict['lat'].size
    dve = ddict['vperpe'] - ddict['vperp_electron'][0,:]
    dvn = ddict['vperpn'] - ddict['vperp_electron'][1,:]
    dvu = ddict['vperpu'] - ddict['vperp_electron'][2,:]
    for i in range(N):
        # mu_mat = np.array([[dve[i]*dve[i], dve[i]*dvn[i], dve[i]*dvu[i]], 
        #                    [dvn[i]*dve[i], dvn[i]*dvn[i], dvn[i]*dvu[i]], 
        #                    [dvu[i]*dve[i], dvu[i]*dvn[i], dvu[i]*dvu[i]]])
        dv = np.vstack((dve[i],dvn[i],dvu[i]))
        _cov_jperp = e**2 * (ddict['var_ne'][i] * (cov_vive[:,:,i] + dv.dot(dv.T)) +
                             ddict['ne'][i]**2 * cov_vive[:,:,i])
        cov_jperp.append(_cov_jperp)        

    ddict['cov_jperp'] = np.swapaxes(np.swapaxes(np.array(cov_jperp),1,0),2,1)
    
    return ddict