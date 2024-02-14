import numpy as np
from secsy import cubedsphere
import lompe
import matplotlib.pyplot as plt
import scipy
import dipole
from gemini3d.grid.gridmodeldata import model2pointsgeogcoords
from . import model


class convection:
    def __init__(self, grid_l, data, inputmode='vi', e3doubt_=True, l1_lompe=1e-2, 
                 l2_lompe=1e-2, maph=200) -> None:
        '''
        Make the convection object that is used for the continous vperp description
        at maph altitude.

        Parameters
        ----------
        grid_l : SC grid object
            The grid object to be used for the Lompe fit, which should be an extended
            verstion of the one used in the 3D model.
        data : instane of data class
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
            Wheter to estimate realistic uncertainties in the sampling. The default is True.
        l1_lompe : bool, optional
            Lompe l1 regularization parameter. The default is 1e-2.
        l2_lompe : bool, optional
            Lompe l2 regularization parameter. The default is 1e-2.
        
        Returns
        -------
        2 element tuple
        
        data : instance of the data class
            With updated values.
        lmodel : lompe object
            The estimated Lompe model.
        '''
        
        self.maph = maph
        self.RE = data.RE
        filename_lompe = self.make_filename(grid_l.projection.position, inputmode)
        
        if (inputmode == 'vi') or (inputmode == 'vi_ohmslaw'):   
            if e3doubt_:
                # Make the data covariance matrix for input data to lompe
                data = self.make_datacov_lompe(data, grid_l)
                # Initialise Lompe model object
                lmodel = self.lompe_fit(data, grid_l, l1_lompe=l1_lompe, l2_lompe=l2_lompe, 
                                    e3doubt_=e3doubt_)                 
                # Do the lompe inversion and calculate model covariance matrix
                m, Cmpost = self.make_cmpost(lmodel, data, l1_lompe=l1_lompe, l2_lompe=l2_lompe)
                lmodel.m = m.copy()
                
                # Calculate the covariance matrix of lompe representation of v_perp at maph
                # at the locations that map to each observation, 'covVlompe'
                data = self.make_lompe_v_cov(lmodel, data, Cmpost)
                # Calculate covariance of predicted electron perp velocity when mapped to
                # measurement locations, 
                data = self.make_ve_cov(data)
                # Calculate covariance of jperp based on (vi_perp-ve_perp)
                data = self.make_cov_jperp(data)

            else:
                # Initialise Lompe model object
                lmodel = self.lompe_fit(data, grid_l, l1_lompe=l1_lompe, l2_lompe=l2_lompe, 
                                    e3doubt_=e3doubt_)                                 
                # Make up some covatiance values
                _cov = np.zeros((3,3,data.lat.size))
                _cov[0,0,:] = 1e-6
                _cov[1,1,:] = 1e-6
                _cov[2,2,:] = 1e-7
                data.cov_jperp = _cov
                
            ds = lmodel.save()
            ds.to_netcdf(filename_lompe)
            print('Saved Lompe file.')

        else:
            lmodel=None
        
        self.lmodel = lmodel
        self.data = data


    def make_datacov_lompe(self, data, grid_l):
        # This function will map the covariance of v from the observation locations
        # as contained in datadict['cov_vi'] and obtained wiht E3DOUBT, to covariances
        # of vperp at maph height, and put it into datadict['cov_viperp_mapped']
        # maph is the altitude above where we assume ExB drift

        N = data.lat.size

        Gs = data._make_vperpmappedG()

        mapped_cov = []

        for i in range(N):
            Gi = Gs[:,:,i]
            
            # Propagate covariance of v into of the mapped vperp at maph
            cov = data.cov_vi[:,:,i]
            mapped_cov_i = Gi.dot(cov).dot(Gi.T)
            
            mapped_cov.append(mapped_cov_i)
            
        mapped_cov = np.swapaxes(np.swapaxes(np.array(mapped_cov),1,0),2,1)    
        
        data.cov_viperp_mapped = mapped_cov
        
        # Identify observations below maph from data class, and nans in mapped locations
        mappedlon = data.mappedglon
        mappedlat = data.mappedglat 
        use = np.isfinite(mappedlat) & \
                (np.isfinite(data.cov_viperp_mapped[0,0,:])) & \
                (grid_l.ingrid(mappedlon, mappedlat)) & (data.alt>=self.maph)
        data.uselompe = use
        
        return data
    

    def lompe_fit(self, data, grid_l, l1_lompe=1e-2, l2_lompe=1e-2, e3doubt_=False, 
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
        use = (data.uselompe)
        data_coords = np.vstack((data.mappedglon[use], data.mappedglat[use]))
        vperpgg_enu = np.vstack((data.vperpmappede[use],
                                data.vperpmappedn[use],
                                data.vperpmappedu[use])).T
        v_data = vperpgg_enu.T[0:2,:] # only use the east,north component in lompe
        if e3doubt_: # At present, off-diagonal terms are not used in Lompe
            error = np.vstack((data.cov_viperp_mapped[0,0,use], 
                            data.cov_viperp_mapped[1,1,use]))
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
        model.run_inversion(l1 = l1_lompe, l2 = l2_lompe, perimeter_width = 0)

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


    def make_cmpost(self, model, data, l1_lompe=1e-2, l2_lompe=1e-2):
        '''
        Make posterior model covatiance matrix based on data covatiance. This function 
        is needed as Lompe at present does not include off-diagonal elements in data
        covatiance matrix. Also compute the model parameters (the solution.)

        Parameters
        ----------
        model : lompe model object
            DESCRIPTION.
        data : instance of the data class
            The data and locations used to constrain the Lompe representation, 
            including the covariances from E3DOUBT.

        Returns
        -------
        m, Cmpost.
        '''
        
        ndim = 2    # if 3, use also vertival component in inversion, to take advantage of
                    # the knowledge of covariance. The covariance is also favorable
                    # in this direction. Turns out that the data covariance matrix 
                    # becomes singular when also using the vertial component. Unclear
                    # if this is an issue. Proceed with using ndim=2.
        
        # Lompe describe horizontal part of v_perp.
        usel = data.uselompe
        lon = data.mappedglon[usel]
        lat = data.mappedglat[usel]        
        
        # skip data points that are outside biggrid:
        use = model.biggrid.ingrid(lon, lat)
        lon = lon[use]
        lat = lat[use]
        coords = {'lon':lon, 'lat':lat}
        N = sum(use)

        #Data vector and covariance
        vperpgg_enu = np.vstack((data.vperpmappede[usel][use],data.vperpmappedn[usel][use], \
                                data.vperpmappedu[usel][use])).T   
        cov = data.cov_viperp_mapped[:,:,usel]
        cov = cov[:,:,use]
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
            Be, Bn, Bu = data.Be[usel][use], data.Bn[usel][use], data.Bu[usel][use]
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
        GG = GTG + l1_lompe*gtg_mag * np.eye(GTG.shape[0]) + l2_lompe * gtg_mag / ltl_mag * model.LTL
        GTd = G.T.dot(Cdinv).dot(d)
        
        # Posterior model covariance matrix
        Cmpost = scipy.linalg.lstsq(GG, np.eye(GG.shape[0]))[0]
        
        # Solution
        m = Cmpost.dot(GTd)

        return (m,Cmpost)
    
    
    def make_lompe_v_cov(self, model, data, Cmpost):
        '''
        Estimate covariances of the v_perp as predicted by lompe, at maph altitude 

        Parameters
        ----------
        model : lompe model object
            DESCRIPTION.
        data: instance of data class
            The data from all observed locations to evaluate Lompe with 
        Cmpost: 2D array
            Lompe posterior model covariance matrix

        Returns
        -------
        updated data class instance.

        '''
        N = data.mappedglon.size
        
        # Lompe describe horizontal part of v_perp.
        lon = data.mappedglon
        lat = data.mappedglat        
        coords = {'lon':lon, 'lat':lat}
        
        # Covariance of v_perp field as described by Lompe
        Gen = model.matrix_func['convection'](**coords)  # Ge and Gn matrices
        Be, Bn, Bu = data.Be, data.Bn, data.Bu
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
        data.v_lompe = np.vstack((v_lompe[0:N],v_lompe[N:2*N],v_lompe[2*N:3*N]))
        data.covVlompe = v_cov
        
        return data
    

    def make_ve_cov(self, data):
        '''
        Make covaiance matrix of v_perp as represented by Lompe, but mapped back to its
        original observation location. This is therefore intended to represent the 
        covariance of the estimated electron perp velocity of each observation, to 
        be used to estimate j_perp with (vi_perp - ve_perp).

        Parameters
        ----------
        data: instance of data class
            The data from all observed locations to evaluate Lompe with

        Returns
        -------
        Updated instance of data class.

        '''

        N = data.lat.size

        # First, calculate d vectors at the layer of Lompe representation
        dip = dipole.Dipole(dipole_pole=((90-11),289))
        d1, d2, d3, _1, _2, _3 = dip.get_apex_base_vectors_geo(data.lon, 
                                    data.lat, np.ones(N)*(data.RE+self.maph), R=data.RE)
        
        # Calculate e vectors at the original measurement locations
        _1, _2, _3, e1, e2, e3 = dip.get_apex_base_vectors_geo(data.mappedglon, 
                            data.mappedglat, data.alt+data.RE, R=data.RE)
        
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
            vi = np.vstack((data.v_lompe[0,i],data.v_lompe[1,i],
                            data.v_lompe[2,i]))
            # vmappedi = np.einsum('ij,jk->i', Gi, vi)
            vmappedi = Gi.dot(vi)
            vmappede.append(vmappedi[0,0])
            vmappedn.append(vmappedi[1,0])
            vmappedu.append(vmappedi[2,0])
            
            # Propagate covariance of V_lompe at maph altitude into covariance of v at
            # observation locations, cov_ve, representing the covariance of the
            # perpendicular electron velocity vector of each observation
            cov_lompe = data.covVlompe[:,:,i]
            mapped_cov_ve = Gi.dot(cov_lompe).dot(Gi.T)
            cov_ve.append(mapped_cov_ve)
            
        cov_ve = np.swapaxes(np.swapaxes(np.array(cov_ve),1,0),2,1)   
        data.cov_ve = cov_ve
        data.vperp_electron = np.vstack((vmappede,vmappedn,vmappedu))
        
        return data
    

    def make_cov_jperp(self, data):
        '''
        Make covariance matrix of jperp based on variance of n and covariance of
        vi_perp and ve_perp using jperp = ne(vi_perp - ve_perp)

        Parameters
        ----------
        data: instance of data class
            The data from all observed locations to evaluate Lompe with

        Returns
        -------
        Updated ddict.

        '''  
        
        #Covariance of vi - ve = cov(vi) + cov(ve)
        data = self.make_cov_vi_perp(data)
        cov_vive = data.cov_vi_perp + data.cov_ve
        e = 1.6e-19 #electron charge
        cov_jperp = []
        # cov_jperp2 = []
        
        N = data.lat.size
        dve = data.vperpe - data.vperp_electron[0,:]
        dvn = data.vperpn - data.vperp_electron[1,:]
        dvu = data.vperpu - data.vperp_electron[2,:]
        for i in range(N):
            # mu_mat = np.array([[dve[i]*dve[i], dve[i]*dvn[i], dve[i]*dvu[i]], 
            #                    [dvn[i]*dve[i], dvn[i]*dvn[i], dvn[i]*dvu[i]], 
            #                    [dvu[i]*dve[i], dvu[i]*dvn[i], dvu[i]*dvu[i]]])
            dv = np.vstack((dve[i],dvn[i],dvu[i]))
            _cov_jperp = e**2 * (data.var_ne[i] * (cov_vive[:,:,i] + dv.dot(dv.T)) +
                                data.ne[i]**2 * cov_vive[:,:,i])
            cov_jperp.append(_cov_jperp)        

        data.cov_jperp = np.swapaxes(np.swapaxes(np.array(cov_jperp),1,0),2,1)
        
        return data
    
    
    def make_cov_vi_perp(self, data):
        # Map the covariance matrix from E3DOUBT to covariance of 
        # vi_perp, which is what is needed when estimating covariance
        # of j_perp using (vi-ve)_perp

        N = data.lat.size
        
        Gperp = data._make_vperpG() # matrices that extract perp part of v
        
        cov_vi_perp = []
        
        for i in range(N):
            Gperp_i = Gperp[:,:,i]
            cov_vi_perp.append(Gperp_i @ data.cov_vi[:,:,i] @ Gperp_i.T)
        cov_vi_perp = np.swapaxes(np.swapaxes(np.array(cov_vi_perp),1,0),2,1)    
        data.cov_vi_perp = cov_vi_perp
        return data

    def make_filename(self, position, inputmode):
        '''
        Make the filenames of the coefficient and lompe model files    

        Parameters
        ----------
        position : tuple
            Position tuple as contained in the CS projection class (lon,lat) in degrees.
        inputmode : str
            As defined in above functions.

        Returns
        -------
        str, filename (with relative path) of lompe object to be saved.

        '''
        sss = ''
        if inputmode == 'vi':
            sss = sss + '_vi-ve_lompe'
        elif inputmode == 'vi_ohmslaw':
            sss = sss + '_vi-ve_lompe_ohmslaw'
        else:
            sss = sss + '_' + inputmode
        
        filename_lompe = './inversion_coefs/3Dreconstruction_lompeobject'+sss+ \
                    '_%4.2f_%4.2f.npy' % (position[0],position[1])    
                    
        return filename_lompe


    def ehat(self, simulation, data):
        '''
        Function that return unit vector along E-field as represented by Lompe
        at the locations (lat, lon, alt) contained in ddict. 
        
        Parameters
        ----------
        simulation : instance of simulation class
            Contains the GEMINI grid and data object        
        data : instance of the data class
            The E-field is computed on the locations contained here.        

        Returns
        -------
        3 element tuple (ehate, ehatn, ehatu).

        '''
        
        E = self.get_E_from_lmodel(simulation, data)
        
        Emag = np.linalg.norm(E, axis=0)
        ehate = E[0,:]/Emag
        ehatn = E[1,:]/Emag
        ehatu = E[2,:]/Emag

        return (ehate, ehatn, ehatu)
    

    def get_E_from_lmodel(self,simulation, data, returnvperp=False):
        '''
        Use the Lompe representation of vperp at maph height, and compute 
        its mapped velocity vector at each (lat,lon,alt) location, and then
        estimate the corresponding E-field, using the magnetic field value.
        
        Note that E-field must not be computed with Lompe matrices, since that
        may involve a B-field that is inconsistent with what is used in GEMINI.
        Here we rather use E = - v x B with B from GEMINI.

        Parameters
        ----------
        simulation : instance of simulation class
            Contains the GEMINI grid and data object
        data : instance of the data class
            The E-field is computed on the locations contained here.
        returnvperp : boolean
            Instead, return vperp at the input locations

        Returns
        -------
        E, electric field at each location (ENU), a (3 x N) 2D array [V/m]. 
        Or vperp with same shape.

        '''
        
        # Get hor part of vperp from Lompe
        ve, vn = self.lmodel.v(lon=data.mappedglon, lat=data.mappedglat)
        # Get vu of vperp from vperp dot B = 0
        # First need to bet B field at the maph altitude of Lompe
        Be = model2pointsgeogcoords(simulation.xg, simulation.dat.Be, self.lmodel.grid_E.R-self.RE*1e3, 
                                    data.mappedglon, data.mappedglat)
        Bn = model2pointsgeogcoords(simulation.xg, simulation.dat.Bn, self.lmodel.grid_E.R-self.RE*1e3, 
                                    data.mappedglon, data.mappedglat)
        Bu = model2pointsgeogcoords(simulation.xg, simulation.dat.Bu, self.lmodel.grid_E.R-self.RE*1e3, 
                                    data.mappedglon, data.mappedglat)
        vu = -(ve*Be + vn*Bn)/Bu
        vperp = np.vstack((ve ,vn, vu))
        dip = dipole.Dipole(dipole_pole=((90-11),289)) # These locations found in pygemini doc.
        d1, d2, d3, _1, _2, _3 = dip.get_apex_base_vectors_geo(data.mappedglon, 
                            data.mappedglat, self.lmodel.grid_E.R*1e-3, R=self.RE+110)
        #Calculate the quantities that is constant along the field-lines
        ve1 = (d1[0,:]*vperp[0,:] + d1[1,:]*vperp[1,:] + d1[2,:]*vperp[2,:])
        ve2 = (d2[0,:]*vperp[0,:] + d2[1,:]*vperp[1,:] + d2[2,:]*vperp[2,:])
        # Calculate basis vectors at the mapped locations
        _1, _2, _3, e1, e2, e3 = dip.get_apex_base_vectors_geo(data.lon, 
                            data.lat, data.alt+self.RE, R=self.RE+110)
        #Calculate the mapped velocity using eq 4.17 in Richmond 1995. geographic components, ENU            
        vperpmappede = (ve1.flatten()*e1[0,:] + ve2.flatten()*e2[0,:])
        vperpmappedn = (ve1.flatten()*e1[1,:] + ve2.flatten()*e2[1,:])
        vperpmappedu = (ve1.flatten()*e1[2,:] + ve2.flatten()*e2[2,:])
        
        v = np.vstack((vperpmappede, vperpmappedn, vperpmappedu))
        B = np.vstack((data.Be, data.Bn, data.Bu))
        E = -np.cross(v, B, axis=0)
        
        if returnvperp:
            return v
        else:
            return E
        
        
    def calc_hall_pedersen(self, simulation, data, inputmode='vi'):
        '''
        Calculate the magnitude of Hall and Pedersen current at the locations in this
        instance of the data class.

        Parameters
        ----------
        simulation : instance of simulation class
            Contains the GEMINI grid and data object        
        data : instance of the data class
            The E-field is computed on the locations contained here. 
        inputmode : str
            Must have same values as it use to, but is here used to override the value used
            to make this instnace of the convection class, and applies to how the E-field
            in the provided data class is computed.      

        Returns
        -------
        tuple (jh_mag, jp_mag).

        '''
        if 'phitop' in inputmode:
            E_enu = np.vstack((data.Ee,data.En,data.Eu))
        else:
            E_enu = self.get_E_from_lmodel(simulation, data)
        
        br, btheta, bphi = data.make_b_unitvectors()
        bhat_enu = np.vstack((bphi,-btheta,br))

        # ENU gg conponnets of Hall and Pedersen current
        jp = data.sp * E_enu
        jh = data.sh * np.cross(bhat_enu,E_enu, axis=0)
        
        jp_mag = np.sqrt(jp[0,:]**2 + jp[1,:]**2 + jp[2,:]**2)
        jh_mag = np.sqrt(jh[0,:]**2 + jh[1,:]**2 + jh[2,:]**2)

        return (jh_mag, jp_mag)