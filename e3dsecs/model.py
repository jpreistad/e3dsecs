import numpy as np
import os
import scipy
import pandas as pd
from secsy import utils as secsy
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

from . import diagnostics
from . import data
from . import coordinates

class model:
    def __init__(self, grid, inputmode='vi', 
                factop=True, vert_profile=150, vert_dprofile=False, l1=1e-2,
                diagnostic=False, overwrite=False, crossval=False, 
                e3doubt_ = False) -> None:
        '''
        grid : instance of grid class object
            Contains the grids we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J). Assumes grid.grid.A refer to bottom layer.
            Contains also grid.alts_grid : array-like, 1D: Altitude grid to use together 
            with grid.grid. Length is K. Values 
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
        factop : boolean, optional
            Wheter to include observations of FAC at top boundary. 
            The default is True.
        vert_profile : int, optional
            Specify altitude in km from where above the vertical profile of Hall and 
            Pedersen currents in the solution will be encouraged to be similar to the
            GEMINI profile. Or could implement any other profile. If None (default) 
            this feature is disabled.
            with no other observations. The default is True.
        vert_dprofile : boolean
            Whether to encourage solutions that has a similar altitude profile of the
            derivative of Hall and Pedersen currents as it has in the GEMINI model.
        l1 : float, optional
            Regularization parameter, normalized to the median of the diagonal of GTG. 
            The default is 1e-2.
        diagnostic : boolean
            Make some diagnostic plots. The default is False
        overwrite : boolean
            Whether one should overwrite existing coef file if it exists. Default is
            False
        crossval : boolean
            Whether to select regularization parameters from crossvalidation analysis.
        e3doubt_ : boolean
            Specify if covariances from E3D is estimated and to be used in the inversion.
            The default is False
        '''            
        
        
        # Define filename and see if it exists
        filename = self.make_filename(grid.grid.projection.position, inputmode, 
                            factop=factop, vert_profile=vert_profile)
        
        # Set keyword attributes
        self.filename = filename
        self.inputmode = inputmode
        self.factop = factop
        self.vert_profile = vert_profile
        self.vert_dprofile = vert_dprofile
        self.l1 = l1
        self.diagnostic = diagnostic
        self.overwrite = overwrite
        self.crossval = crossval
        self.e3doubt_ = e3doubt_        
        self.RE = grid.RE
        self.file_exists = False
        
        # Check if coef files exists
        exists = os.path.isfile(filename)
        if (inputmode == 'vi') or (inputmode == 'vi_ohmslaw'):
            if exists and not overwrite:
                self.file_exists = True
                print('Model coef file exists')
        else:
            if exists and not overwrite:
                self.file_exists = True
                print('Model coef file exists')
          
    
    def run_inversion(self, grid, convection, simulation):
        '''
        Solve for 3D model coefficeints.

        Parameters
        ----------
        grid : instance of grid class object
            Contains the grids we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J). Assumes grid.grid.A refer to bottom layer.
            Contains also grid.alts_grid : array-like, 1D: Altitude grid to use together 
            with grid.grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.  
        convection : instance of convection class
            Contain lompe representation. convection.data also contain the instance of the
            data class of the E3D like samples used to make the lompe fit, but also contain
            all samples also below maph.
        simulation : instance of simulation class
            contain the GEMINI data and grid object used
        '''
   
        # Prepare the sampled GEMINI data for inversion
        # Thus function pulls out jperp from the instance of the data class that
        # is in line with the method inputmode specified, and contain only
        # observations from within the reconstruction region, as specified with
        # the ext_factor keyword.
        
        # Set the input options
        inputmode = self.inputmode
        factop = self.factop
        vert_profile = self.vert_profile
        vert_dprofile = self.vert_dprofile
        l1 = self.l1
        diagnostic = self.diagnostic
        overwrite = self.overwrite
        crossval = self.crossval
        e3doubt_ = self.e3doubt_
        
        
        inputdict = self.make_inputdict(convection.data, grid,
                            inputmode='vi', ext_factor=-1, hp_from_brekke=False)
        
        ###############################################
        # Start building matricies and arrays used in inversion
        # Up to now, ENU componsnts has been used. In the 3D model, we use 
        # r, theta, phi components (still geographic).
        d = np.hstack((inputdict['jperp'][2,:], -inputdict['jperp'][1,:], 
                    inputdict['jperp'][0,:])) # (r, theta, phi components)
        dcopy = d.copy()
        
        # Initialize matrices
        GTGs = []
        GTds = []
        
        if d.size > 0:
            br, btheta, bphi = self.make_b_unitvectors(inputdict['Bu'],inputdict['Bn'],
                                                inputdict['Be'])
            B = self.make_B(br, btheta, bphi)
            G = self.make_G(grid.grid, grid.alts_grid, inputdict['lat'], inputdict['lon'], 
                    inputdict['alt'], jperp=B)
            Gcopy = G.copy()
            N = inputdict['lat'].size
            
            # Before proceeding to inversion, check that the analysis grid looks reasonable
            if diagnostic:
                #Make evaluation grid to see what will be good to use later
                Nxi = int(grid.J*1.5) #sample resolution, corresponding to direction of "j" index
                Neta = int(grid.I*1.5) #sample resolution, corresponding to direction of "i" index
                alts__ = grid.alts_grid[:-1] # alts_grid[1:]-altres[1:]
                etas = np.linspace(grid.grid.eta_mesh[1,0]+0.01*grid.grid.deta,
                                grid.grid.eta_mesh[-2,0]-0.01*grid.grid.deta,Neta)
                xis = np.linspace(grid.grid.xi_mesh[0,1]+0.01*grid.grid.dxi,
                                grid.grid.xi_mesh[0,-2]-0.01*grid.grid.dxi,Nxi)
                xi_ev, eta_ev = np.meshgrid(xis, etas, indexing = 'xy')
                alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, etas, xis, indexing='ij')
                lon_ev, lat_ev = grid.grid.projection.cube2geo(xi_ev, eta_ev)
                diagnostics.plot_analysis_grid(convection.data.__dict__, grid.grid, grid.alts_grid, lat_ev, 
                                    lon_ev, alt_ev, data=True, eiscat=True)
            
            # Weighting business
            data_density_weight = True # do the spatial weighting ala Lompe
            iweight = 1 # Relative weight of the GEMINI observations    
            if data_density_weight:
                bincount = self.count_kij(grid.grid, grid.alts_grid, inputdict['lon'], inputdict['lat'], 
                                    inputdict['alt'])
                k, i, j = self.get_indices_kij(grid.grid, grid.alts_grid, inputdict['lat'], 
                                        inputdict['lon'], inputdict['alt'])
                i, j = grid.grid.bin_index(inputdict['lon'], inputdict['lat'])
                spatial_weight = 1. / np.maximum(bincount[k, i, j], 1)
                spatial_weight[i == -1] = 1
                # stack weights for each component in dataset.values:
                spatial_weight = np.tile(spatial_weight, 3)
            else:
                spatial_weight = np.ones(3*N)        
            if e3doubt_:
                # Build data covariance matrix
                cov = convection.data.cov_jperp[:,:,inputdict['inds']==1] # ENU, not r,theta,phi
                # Fix nans in cov by setting the cov to a large number
                # nans = ~np.isfinite(cov[0,0,:])
                # cov[:,:,nans] = 1e-8 # seems like a large number 
                large = np.abs(cov) > 1e-8
                cov[large] = 1e-8 # Crazy values make the inversion of Cd fail.
                # fudge = 0.99 # To make the invesion not fail (Spencers fix)
                Cd = np.zeros((N*3,N*3)) # Data covariance matrix
                Cd[:N,:N] = np.diag(cov[2,2,:])# upper left, corresponds to all r components
                Cd[N:2*N,N:2*N] = np.diag(cov[1,1,:])# center block, corresponds to 
                                                        # all theta components
                Cd[2*N:3*N,2*N:3*N] = np.diag(cov[0,0,:]) # bottom right block, 
                                                # corresponds to all phi components
                Cd[:N,N:2*N] = np.diag(cov[2,1,:])# * fudge # upper center block, corresponds 
                                                    # to r-theta covariance
                Cd[:N,2*N:3*N] = np.diag(cov[2,0,:])# upper right block, corresponds 
                                                    # to r-phi covariance
                Cd[N:2*N,0:N] = np.diag(cov[1,2,:])# middle row, left block, 
                                                    # corresponds to theta-r covariance
                Cd[N:2*N,2*N:3*N] = np.diag(cov[1,0,:])# middle row, right block, 
                                                        # corresponds to theta-phi covariance
                Cd[2*N:3*N,0:N] = np.diag(cov[0,2,:])# lower row, left block, 
                                                    # corresponds to phi-r covariance
                Cd[2*N:3*N,N:2*N] = np.diag(cov[0,1,:])# lower row, center block, 
                                                    # corresponds to phi-theta covariance
                # Cdinv = np.linalg.inv(Cd)
                Cdinv = np.linalg.pinv(Cd)
                w_i = spatial_weight * Cdinv * iweight
                GTG_i = G.T.dot(w_i).dot(G)
                GTd_i = G.T.dot(w_i).dot(d)
            else:
                Cdinv = 1/((1e-6)**2) # error in the jperp data as sampled from GEMINI [A/m^2]
                w_i = spatial_weight * Cdinv * iweight
                GTG_i = G.T.dot(np.diag(w_i)).dot(G)
                GTd_i = G.T.dot(np.diag(w_i)).dot(d)  

            GTGs.append(GTG_i)
            GTds.append(GTd_i)
        else:
            print('Warning: No E3D observations are being used from inside volume.')
        
        # Regularize based on FAC value on top of domain
        if factop:
            G = self.make_L(grid.grid, grid.alts_grid)
            _data = data.data(grid, simulation, beams=False, uniformmesh=True, lat_ev=grid.grid.lat, 
                              lon_ev=grid.grid.lon, 
                              alt_ev=np.ones(grid.grid.shape)*grid.alts_grid[-1]*.999, 
                              e3doubt_=False)
            d = self.make_F(grid.grid, _data.ju.reshape(grid.grid.shape), 
                    grid.grid.lat_mesh[1:-1,1:-1], grid.grid.lon_mesh[1:-1,1:-1])
            N = d.size
            spatial_weight = np.ones(N) # no spaial weighting since FACs are sampled on secs grid
            error = 1e-6 # error in the ju data as sampled from GEMINI [A/m^2]
            iweight = 1 # Relative weight of the GEMINI observations
            w_i = spatial_weight * 1/(error**2) * iweight
            GTG_i = G.T.dot(np.diag(w_i)).dot(G)
            GTd_i = G.T.dot(np.diag(w_i)).dot(d)
            GTGs.append(GTG_i)
            GTds.append(GTd_i)

        # Regularize based on vert profile of hall/ped currents from GEMINI
        if vert_profile is not None:
            # Make new grid for the model padding from IRI of vertical H/P currents     
            use = grid.alts_grid[1:] >= vert_profile
            alts__ = grid.alts_grid[1:][use]-grid.altres[1:][use]
            xi_e  = grid.grid.xi[0,1:] - grid.grid.dxi/2 
            eta_e = grid.grid.eta[1:,0]- grid.grid.deta/2
            alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, eta_e, xi_e, indexing='ij')
            lon_ev, lat_ev = grid.grid.projection.cube2geo(xi_ev, eta_ev)        
            G, inds = self.make_JHP(convection, simulation, grid, lat_ev.flatten(), lon_ev.flatten(), 
                            alt_ev.flatten(), inputmode='vi', hop='p', 
                            return_indices=True)
            _data = data.data(grid, simulation, beams=False, uniformmesh=True, 
                              lat_ev=lat_ev.flatten()[inds==1], lon_ev=lon_ev.flatten()[inds==1], 
                              alt_ev=alt_ev.flatten()[inds==1], e3doubt_=False)

            jh, jp = convection.calc_hall_pedersen(simulation, _data, inputmode=inputmode)
            # d = jp
            d = np.zeros(jp.size)
            N = d.size
            spatial_weight = np.ones(N) # no spaial weighting here, since FACs are samped on secs grid
            error = 1e-6#0.1*d.max()#6e-5 # error in the jp "data" from IRI/MSIS model [A/m^2]
            iweight = 0.1 # Relative weight of the GEMINI observations
            w_i = spatial_weight * 1/(error**2) * iweight
            GTG_i = G.T.dot(np.diag(w_i)).dot(G)
            GTd_i = G.T.dot(np.diag(w_i)).dot(d)
            GTGs.append(GTG_i)
            GTds.append(GTd_i)
            G, inds = self.make_JHP(convection, simulation, grid, lat_ev.flatten(), lon_ev.flatten(), 
                            alt_ev.flatten(), inputmode='vi', hop='h', 
                            return_indices=True)
            # d = jh
            d = np.zeros(jh.size)
            N = d.size
            spatial_weight = np.ones(N) # no spaial weighting here, since FACs are samped on secs grid
            error = 1e-6#0.1*d.max() # error in the jp "data" from IRI/MSIS model [A/m^2]
            iweight = 0.1 # Relative weight of the GEMINI observations
            w_i = spatial_weight * 1/(error**2) * iweight
            GTG_i = G.T.dot(np.diag(w_i)).dot(G)
            GTd_i = G.T.dot(np.diag(w_i)).dot(d)
            GTGs.append(GTG_i)
            GTds.append(GTd_i)

        # Regularize based on gradient of vertical profile of hall/ped currents from IRI
        if vert_dprofile:
            # Make new grid for the model padding from IRI of vertical gradients in H/P currents        
            alts__ = grid.alts_grid[1:]-grid.altres[1:]
            xi_e  = grid.grid.xi[0,1:] - grid.grid.dxi/2 
            eta_e = grid.grid.eta[1:,0]- grid.grid.deta/2
            alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, eta_e, xi_e, indexing='ij')
            lon_ev, lat_ev = grid.projection.cube2geo(xi_ev, eta_ev)        
            G, inds = self.make_D(grid, simulation, convection, lat_ev.flatten(), lon_ev.flatten(), alt_ev.flatten(), hop='p')
            _h, _p = self.sigma_profile(grid, simulation, alt_ev.flatten()[inds==1], lat=lat_ev.flatten()[inds==1], 
                                lon=lon_ev.flatten()[inds==1], derivative=True, iri=False)
            d = _p
            N = d.size
            spatial_weight = np.ones(N) # no spaial weighting here, since FACs are samped on secs grid
            error = 0.1*d.max()#6e-9 # error in the jp "data" from IRI/MSIS model [A/m^2]
            iweight = 1 # Relative weight of the GEMINI observations
            w_i = spatial_weight * 1/(error**2) * iweight
            GTG_i = G.T.dot(np.diag(w_i)).dot(G)
            GTd_i = G.T.dot(np.diag(w_i)).dot(d)
            GTGs.append(GTG_i)
            GTds.append(GTd_i)
            G, inds = self.make_D(grid, simulation, convection, lat_ev.flatten(), lon_ev.flatten(), alt_ev.flatten(), hop='h')
            d = _h
            N = d.size
            spatial_weight = np.ones(N) # no spaial weighting here, since FACs are samped on secs grid
            error = 0.1*d.max()#1e-9 # error in the jp "data" from IRI/MSIS model [A/m^2]
            iweight = 1 # Relative weight of the GEMINI observations
            w_i = spatial_weight * 1/(error**2) * iweight
            GTG_i = G.T.dot(np.diag(w_i)).dot(G)
            GTd_i = G.T.dot(np.diag(w_i)).dot(d)
            GTGs.append(GTG_i)
            GTds.append(GTd_i)
        GTG = np.sum(np.array(GTGs), axis=0)
        GTd = np.sum(np.array(GTds), axis=0)
        
        # regularization
        gtg_mag = np.median(np.diagonal(GTG)) # Using l1,l2, alpha 1/2 similar as described in Lompe paper 
        # Altreg is compensating for the irregular vertical spacing, making the model 
        # amplitudes obtaining different values
        altreg = np.ones(2*grid.KIJ)
        for i,_a in enumerate(grid.altres):
            altreg[i*grid.IJ:(i+1)*grid.IJ] = _a
            altreg[grid.KIJ+i*grid.IJ:grid.KIJ+(i+1)*grid.IJ] = _a
        if crossval:
            # resnorm, modelnorm, ls = Lcurve(GTG, GTd, altreg, Gcopy, dcopy, steps=10)   
            resnorm, modelnorm, ls = self.crossvalidation(grid, simulation, convection, GTG, GTd, altreg,
                                        inputmode=inputmode, steps=15)
            print('In the following, l1 = %5f is used.' % ls[np.argmin(resnorm)])
            l1 = 10**(ls[np.argmin(resnorm)])
        l1star = l1 * gtg_mag
        Reg = np.diag(altreg) * l1star
        GG = GTG + Reg
        Cmpost = scipy.linalg.solve(GG, np.eye(GG.shape[0]))
        # Cmpost = scipy.linalg.lstsq(GG, np.eye(GG.shape[0]))[0] 
        m = Cmpost.dot(GTd)

        # Save result of inversion
        grid_tuple = (grid.grid.projection.position, grid.grid.projection.orientation, grid.grid.L, 
                    grid.grid.W, grid.grid.Lres, grid.grid.Wres, grid.grid.R)
        keep = {'alts_grid':grid.alts_grid, 'grid':grid_tuple, 'm':m, 'Cmpost':Cmpost}    
        if diagnostic:
            Rmatrix = Cmpost.dot(GTG)
            keep['Rmatrix'] = Rmatrix
        np.save(self.filename, keep)
        print('Saved coefficient file.')

   
    def make_filename(self, position, inputmode, factop=False, vert_profile=False):
        '''
        Make the filenames of the coefficient and lompe model files    

        Parameters
        ----------
        position : tuple
            Position tuple as contained in the CS projection class (lon,lat) in degrees.
        inputmode : str
            As defined in above functions.
        factop : bool
            If FAC pattern on top is given as input.
        vert_profile : None or int
            If a specific vertical profile of Hall and Pedersen currents are
            encouraged.

        Returns
        -------
        str, containing the filename.

        '''
        sss = ''
        if inputmode == 'vi':
            sss = sss + '_vi-ve_lompe'
        elif inputmode == 'vi_ohmslaw':
            sss = sss + '_vi-ve_lompe_ohmslaw'
        else:
            sss = sss + '_' + inputmode
        
        if factop:
            sss = sss+'_factop'
        else:
            sss = sss + '_nofactop'
            
        if vert_profile is None:
            sss = sss+'_no_vert_profile'
        else:
            sss = sss + '_vert_profile'

        filename = './inversion_coefs/3Dreconstruction'+sss+'_%4.2f_%4.2f.npy' % \
                    (position[0],position[1])
                    
        return filename
    

    def make_inputdict(self, data, grid, inputmode='vi', ext_factor=-1, 
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
        data : instance of the data class
            where to the data is stored.
        grid : instance of the grid class
            Contain grid.grid, which is the grid we use to compute gridded indices 
            of data/evaluation locations. Dimension is (I,J). 
            Also contain grid.alts_grid : 1D array
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
        data : instance of data class
            Updated instance of data class containting only the observations that will be used
            in 3D reconstruction. Also includes 'inds', a 1D array containting the 
            mapping of which obserations of the input datadict that has been selscted 
            for use.
        '''
        
        e = 1.6e-19 # elementary charge
        if inputmode == 'vi':
            jperp_e = data.ne * e * (data.vperpe - 
                                            data.vperp_electron[0,:])
            jperp_n = data.ne * e * (data.vperpn - 
                                            data.vperp_electron[1,:])
            jperp_u = data.ne * e * (data.vperpu - 
                                            data.vperp_electron[2,:])
            jperp = np.vstack((jperp_e, jperp_n, jperp_u))
                
        elif inputmode == 'jperp':
            jperp = np.vstack((data.jperpe,data.jperpn,data.jperpu))
        
        elif inputmode == 'phitop':
            E = np.vstack((data.Ee,data.En,data.Eu))
            B = np.vstack((data.Be, data.Bn, data.Bu))
            Bmag = np.linalg.norm(B, axis=0)

            vperp_electron = np.cross(E, B, axis=0) / Bmag**2

            # Calculate jperp from velocity difference (geographic ENU coords)
            jperp_e = data.ne * e * (data.vperpe - vperp_electron[0,:])
            jperp_n = data.ne * e * (data.vperpn - vperp_electron[1,:])
            jperp_u = data.ne * e * (data.vperpu - vperp_electron[2,:])
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
                N2_msis = f(data.alt.flatten())
                f = scipy.interpolate.interp1d(msis.Height, msis.O2)
                O2_msis = f(data.alt.flatten())
                f = scipy.interpolate.interp1d(msis.Height, msis.O)
                O_msis = f(data.alt.flatten())
                #eq 7.12 in Brekke book. Assumes average ion mass of 30.7.
                c_brekke = kn2 * N2_msis*1e6 + ko2 * O2_msis*1e6 + ko * O_msis*1e6 

                sp = e*data.ne.flatten()/Bmag * \
                        (c_brekke * omega_no)/(omega_no**2 + c_brekke**2)
                sh = e*data.ne.flatten()/Bmag * \
                        (c_brekke**2)/(omega_no**2 + c_brekke**2)
            else: # use conductances from GEMINI
                sp = data.sp
                sh = data.sh
                
            B = np.vstack((data.Be, data.Bn, data.Bu))
            bhat_r, bhat_theta, bhat_phi = self.make_b_unitvectors(data.Bu, 
                                                -data.Bn, data.Be)
            bhat_enu = np.vstack((bhat_phi, -bhat_theta, bhat_r))
            
            if inputmode == 'vi_ohmslaw':
                E = -np.cross(data.vperp_electron.T, B.T, axis=1).T
            elif inputmode == 'phitop_ohmslaw':
                E = np.vstack((data.Ee,data.En,data.Eu))
            else:
                assert 1 == 0, 'Inputmode keyword is not valid.'
        
            j_pedersen = sp * E
            j_hall = sh * np.cross(bhat_enu.T, E.T).T
            jperp = j_pedersen + j_hall
            
        # Now, figure out which observations that are inside the interior of 3D model
        # domain
        #identify obs inside grid horizontally, and any nans
        use = grid.grid.ingrid(data.lon, data.lat, ext_factor = ext_factor) & \
                    np.isfinite(data.jperpe)

        # Remove data/evaluation points outside grid in vertical direction
        k0 = self.get_alt_index(grid.alts_grid, data.alt[use], returnfloat=True)
        inside = (k0 >= 0) & (k0<grid.alts_grid.size-1)
        inds = np.zeros(use.size).astype(int)
        temp = np.zeros(sum(use))
        temp[inside] = 1
        inds[use] = temp

        inputdict = {'lon':data.lon[inds==1], 'lat':data.lat[inds==1], 
                    'alt':data.alt[inds==1], 'Be':data.Be[inds==1],
                    'Bn':data.Bn[inds==1], 'Bu':data.Bu[inds==1],
                    'jperp':jperp[:,inds==1], 'inds':inds}

        return inputdict


    def make_b_unitvectors(self, Br, Btheta, Bphi):
        """
        Parameters
        ----------
        Br : array-like
            1D array of radial magnetic field strength.
        Btheta : array-like
            1D array of magnetic field strength in theta direction (towards south).
        Bphi : array-like
            1D array of magnetic field strength in eta direction (towards east).

        Returns
        -------
        tuple containing the following:
        
        br : array-like
            Radial component of unit vector of magnetic field.
        btheta : array-like
            Theta component of unit vector of magnetic field.
        bphi : array-like
            Phi component of unit vector of magnetic field.

        """
        Bmag = np.sqrt(Br**2 + Btheta**2 + Bphi**2)
        br = Br/Bmag
        btheta = Btheta/Bmag
        bphi = Bphi/Bmag
        return (br, btheta, bphi)
    
    
    def get_alt_index(self, alts_grid, alt, returnfloat=False):
        """
        Parameters
        ----------
        alts_grid : array-like, 1D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the
            voxels of each layer. Should be in increasing order.
        alt : array-like, 1D
            altitude [km] of the data/evaluation locations. Flattened to 1D of 
            length N
        returnfloat : Boolean
            If one wants to know how close it is to a neighboring layer. Default is
            False
        Returns
        -------
        Array of length N of the index in vertical direction of each altitude in 
        alt. Altitudes outside the range spefified by alts_grid is given index -1

        """
        altres = np.diff(alts_grid)*0.5
        altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))

        edges = np.concatenate((alts_grid-altres, np.array([alts_grid[-1]+altres[-1]])))
        k = (np.digitize(alt, edges) - 1)
        overthetop = k == alts_grid.shape[0]
        k[overthetop] = -1
        
        if returnfloat:
            use = k != -1
            res_ = (alt[use] - edges[k[use]])/(edges[k[use]+1]-edges[k[use]])
            # res_ = (alt[use] - alts_grid[k[use]])/(2*altres[k[use]])
            k = k.astype(float)
            k[use] = k[use] - 0.5 + res_

        return k
    
    
    def make_B(self, br, btheta, bphi, ENU=False):
        """
        Make matrix that project 3D vectors (r, theta, phi components) to the plane
        perpendicular to the magnetic field as given in input. The returned 
        matrix need to act on a 1D array of N vectors (length of 3N) sorted 
        vectorwise, i.e. [r1, theta1, phi1, ... rN, thetaN, phiN].
        
        Parameters
        ----------
        br : array-like
            Radial component of unit vector of magnetic field.
        btheta : array-like
            Theta component of unit vector of magnetic field.
        bphi : array-like
            Phi component of unit vector of magnetic field.
        ENU : boolean
            If True, the returned matrix refer to ENU directions, in contrast to the
            default r,theta,phi

        Returns
        -------
        B : Projection matrix. 3Nx3N array, N is length of input array
        
        """
        br = br.flatten()
        btheta = btheta.flatten()
        bphi = bphi.flatten()
        N = br.flatten().size
        
        # The nine components of the projection matrix
        brr = btheta**2+bphi**2
        brtheta = -br*btheta
        brphi = -br*bphi
        bthetar = -br*btheta
        bthetatheta = br**2+bphi**2
        bthetaphi = -btheta*bphi
        bphir = -br*bphi
        bphitheta = -btheta*bphi
        bphiphi = br**2+btheta**2
        # from scipy import sparse    
        # B = sparse.csc_matrix((3*N,3*N))
        B = np.zeros((3*N,3*N))
        if ENU:
            for n in range(N):
                B[3*n,3*n] =  bphiphi[n]
                B[3*n,3*n+1] = -bphitheta[n]
                B[3*n,3*n+2] = bphir[n]
                B[3*n+1,3*n] = bthetaphi[n]
                B[3*n+1,3*n+1] = -bthetatheta[n]
                B[3*n+1,3*n+2] = bthetar[n]
                B[3*n+2,3*n] = brphi[n]
                B[3*n+2,3*n+1] = -brtheta[n]
                B[3*n+2,3*n+2] = brr[n]        
        else:
            for n in range(N):
                B[3*n,3*n] = brr[n]
                B[3*n,3*n+1] = brtheta[n]
                B[3*n,3*n+2] = brphi[n]
                B[3*n+1,3*n] = bthetar[n]
                B[3*n+1,3*n+1] = bthetatheta[n]
                B[3*n+1,3*n+2] = bthetaphi[n]
                B[3*n+2,3*n] = bphir[n]
                B[3*n+2,3*n+1] = bphitheta[n]
                B[3*n+2,3*n+2] = bphiphi[n]        
        
        return B


    def make_G(self, grid, alts_grid, lat, lon, alt, interpolate=True, ext_factor=-1, 
            jperp=None):
        '''
        Make the matrix that relate the 3D model parameters to the 3D current density.
        In the inverse problem jperp is typically what is used. In the forward problem,
        one typially will use jperp=False since the full current is what one wish to
        express.

        Parameters
        ----------
        grid : CS grid object
            The grid we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J). Assumes grid.A refer to bottom layer.
        alts_grid : array-like, 1D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order. 
        lat : 1D array
            latitudes of data/evaluation locations, in degrees.
        lon : TYPE
            longitudes of data/evaluation locations, in degrees.
        alt : TYPE
            altitudes of data/evaluation locations, in km.
        interpolate : boolean, optional
            Whether to use (bi)linear interpolation scheme. Should be used. The default 
            is True.
        ext_factor : int, optional
            DTo control how to filter out obsrvations based on their proximity to the 
            grid. The default is -1, removing points closer than 1 grid cell from 
            the edge of the grid (mesh), on the interior side. If positive, allow
            points outside the grid. See notes at top of this function. The default is -1.
        jperp : , optional
            If not None, jperp is the 3N x 3N magnetic field projection matrix. 
            The default is None. This must be provided when jperp is to be related
            to the model parameters, which is the case in the inverse problem, but
            typically not wanted when using the model in the forward problem.

        Returns
        -------
        G : 2D array
            Shape N x 2*K*I*J matrix that relate j or jperp to model parameters through
            j = G.dot(m)
        '''
        Ge_cf, Gn_cf, Ge_df, Gn_df = self.get_SECS_J_G_matrices_3D(grid, alts_grid, 
                        lat, lon, alt, interpolate=interpolate, 
                        singularity_limit=grid.Lres, ext_factor=ext_factor)
        S = self.get_jr_matrix(grid, alts_grid, lat, lon, alt, interpolate=interpolate, 
                        ext_factor=ext_factor)
        O = np.zeros(S.shape)
        Gcf = np.vstack((S, -Gn_cf, Ge_cf))
        Gdf = np.vstack((O, -Gn_df, Ge_df))
        G = np.hstack((Gcf, Gdf))
        
        if jperp is not None:
            B = jperp.copy()
            P = self.make_P(lat.size)
            G = P.T.dot(B.dot(P.dot(G)))
        
        return G
    
    
    def get_SECS_J_G_matrices_3D(self, grid, alts_grid, lat, lon, alt, constant = 
                                1./(4.*np.pi), singularity_limit=None, 
                                interpolate=False, ext_factor=-1):
        ''' 
        Calculate SECS J_G matrices for 3D representation using CS grid at fixed
        (lat,lon) locations at different heights. For now we assume sampling 
        in each altitude layer is identical, allowing us to scale the SECS G 
        matrices with height. 

        (I, J, K) shapes: I is number of SECS nodes in eta (CS) direction, 
        J is number of SECS poles in xi (CS) direction, and K is number of hor. 
        layers.

        Parameters
        ----------
        grid : CS grid object
            The grid we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J)
        alts_grid : array-like, 1D or 2D
            Altitude grid to use together with secs_grid. Length is K. Values represent 
            the height in km of the centre of the
            voxels of each layer. Should be in increasing order.
        lat : array-like, 1D or 2D
        latitudes [deg] of the data/evaluation locations. Flattened to 1D.
        lon : array-like, 1D
        longitudes [deg] of the data/evaluation locations. Flattened to 1D.       
        alt : array-like, 1D
        altitude [km] of the data/evaluation locations. Flattened to 1D.              
        constant : float, optional
            Passed to the underlying secsy function. The default is 1./(4.*np.pi).
        singularity_limit : float, optional
            Passed to the underlying secsy function. The default is 0. [m] meaning
            that the modified SECS functions are not used. Typically ~half a CS grid 
            cell is used. Default (None) will use 0.5*grid.Lres
        interpolate : Boolean, optional
            If True: Each row in G (observation/evaloation, N)
            will affect the two closest layers of SECS nodes (above and below its 
            location). Its vertical placement reflect the relative weight of the 
            influence of each layer. S matrix should be constructed with the same
            option. Default is False
        ext_factor : int, optional
            To control how to filter out locations based on their proximity to the 
            grid. The default is -1, removing points closer than 1 grid cell from 
            the edge of the grid (mesh).        

        Returns
        -------
        SECS CF and DF G matrices stacked in the appropriate way to represent its
        value in the 3D domain.

        '''
        
        # Evaluation locations test
        if not lat.shape[0]==lon.shape[0]==alt.shape[0]:
            print('Inconsistent dimensions of data/evaluation locations (lat,lon,alt)')
            print(1/0)  
        RE = 6371.2 #Earth radius in km
        
        if singularity_limit == None:
            singularity_limit=grid.Lres*0.5
            
        # Remove data/evaluation points outside the 3D grid
        lat, lon, alt = self.remove_outside(grid, alts_grid, lat, lon, alt, 
                                    ext_factor=ext_factor)  

        #Grid dimensions
        K = alts_grid.shape[0] #Number of vertival layers
        I = grid.shape[0] #Number of cells in eta direction
        J = grid.shape[1]  #Number of cells in xi direction  
        IJK = I*J*K
        N = lat.shape[0] # Number of data/evaluation points  
        
        #Compute SECS matrices at bottom layer, using all observational locations (N). 
        # We will apply the singularity threshold
        # defined as a length on the bottom layer, and use the same corresponding
        # theta0 = singularity_limit/RI on every layer, although this corresponds
        # to a different lenght. I think this makes the mose sense, and is easier
        # to implement (we dont need to keep track of which nodes needs to be modified
        # for each layer).
        alt_ = alts_grid[0]
        Ge_cf_, Gn_cf_ = secsy.get_SECS_J_G_matrices(lat.flatten(), 
                    lon.flatten(), grid.lat.flatten(), 
                    grid.lon.flatten(), constant = 1./(4.*np.pi), 
                    RI=RE * 1e3 + alt_ * 1e3, current_type = 'curl_free', 
                    singularity_limit=singularity_limit)
        Ge_df_, Gn_df_ = secsy.get_SECS_J_G_matrices(lat.flatten(), 
                    lon.flatten(), grid.lat.flatten(), 
                    grid.lon.flatten(), constant = 1./(4.*np.pi), 
                    RI=RE * 1e3 + alt_ * 1e3, current_type = 'divergence_free', 
                    singularity_limit=singularity_limit)
        
        #Indices of each evaluation point in 3D
        k, i, j = self.get_indices_kij(grid, alts_grid, lat, lon, alt, 
                                returnfloat=interpolate)  
        # kij = np.ravel_multi_index((k,i,j), (K,I,J)) #flattened index

        
        if interpolate:
            k_under = np.floor(k).astype(int)
            k_over = np.ceil(k).astype(int)
            same = k_over == k_under
            k_over[same] = k_over[same] + 1
            overthetop = k_over >= len(alts_grid)
            k_over[overthetop] = len(alts_grid)-1
            kij_start_under = np.ravel_multi_index((k_under, 
                                                    np.zeros(len(k)).astype(int),
                                                    np.zeros(len(k)).astype(int)), 
                                                    (K,I,J)) #flattened
            kij_stop_under = kij_start_under + I*J
            kij_start_over = np.ravel_multi_index((k_over, 
                                                    np.zeros(len(k)).astype(int),
                                                    np.zeros(len(k)).astype(int)), 
                                                    (K,I,J)) #flattened
            kij_stop_over = kij_start_over + I*J
            k_frac = k % 1
            w_over = k_frac
            w_under = 1 - w_over        

        #Make stacked SECS G matrices, not corrected for altitude nor vertical 
        # observational / evaluation location
        Ge_cf_k = np.tile(Ge_cf_,K)
        Gn_cf_k = np.tile(Gn_cf_,K)
        Ge_df_k = np.tile(Ge_df_,K)
        Gn_df_k = np.tile(Gn_df_,K)

        #Do the altitude correction and vertical interpolation
        altmask = []
        r_ = (RE + alt_)/(RE + alts_grid) #Altitude correction factors        
        for kkk in range(N):
            mask_ = np.zeros(IJK)
            if interpolate:
                mask_[kij_start_under[kkk]:kij_stop_under[kkk]] = w_under[kkk] * \
                                r_[k_under[kkk]]   
                if not overthetop[kkk]:
                    mask_[kij_start_over[kkk]:kij_stop_over[kkk]] = w_over[kkk] * \
                                r_[k_over[kkk]]    
            else:
                kij_start = np.ravel_multi_index((k[kkk],0,0), (K,I,J)) #flattened
                kij_stop = kij_start + I*J
                mask_[kij_start:kij_stop] = 1 * r_[k[kkk]]
            altmask.append(mask_)
        altmask = np.array(altmask)
        Ge_cf = Ge_cf_k * altmask
        Gn_cf = Gn_cf_k * altmask
        Ge_df = Ge_df_k * altmask
        Gn_df = Gn_df_k * altmask
                            
        # #Kalles broadcasting way
        # r, theta, phi = np.meshgrid(r1d, theta1d, phi1d, indexing = 'ij')
        # r = np.array([np.meshgrid(np.ones(secs_grid.shape[0])*alts_grid[i], np.ones(secs_grid.shape[1])*alts_grid[i])[0] for i in range(len(alts_grid))])
        # Ge_cf_kalle = (Ge_cf_.reshape((N, -1, secs_grid.lon.shape[0] * secs_grid.lon.shape[1])) / 
        #          r.reshape((-1, alts_grid.size, secs_grid.lon.shape[0] * secs_grid.lon.shape[1]))).reshape(N, I*J*K)        

        return (Ge_cf, Gn_cf, Ge_df, Gn_df)
    
    
    def remove_outside(self, grid, alts_grid, lat, lon, alt, params=None, 
                    ext_factor=-1, return_indices=False):
        """
        Parameters
        ----------
        grid : CS grid object
            The grid we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J). Assumes secs_grid.A refer to bottom layer.
        alts_grid : array-like, 1D or 2D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.
        lat : array-like
            Output latitude after filtering. Flattened, in degrees.
        lon : array-like
            Output longitude after filtering. Flattened, in degrees.
        alt : array-like
            Output altitude after filtering. Flattened, in km.
        params : tuple, optional
            Apply the same filtering also to the variables as contained in this
            tuple. Must be of same size as lat/lon/alt. The default is None.
        ext_factor : int, optional
            To control how to filter out locations based on their proximity to the 
            grid. The default is -1, removing points closer than 1 grid cell from 
            the edge of the grid (mesh).
        return_indices : boolean
            Return the indices of which elements has been kept. 1 means kept, 0 is 
            dropped.

        Returns
        -------
        Tuple.
            Filtered lat,lon,alt arrays, flattened. If params is provided, the first
            element in the returned tuple will be tuple of the different params 
            provided.
        """
        # Remove data/evaluation points outside the perimiter of secs nodes
        use = grid.ingrid(lon.flatten(), lat.flatten(), ext_factor = ext_factor)
        lat = lat.flatten()[use]
        lon = lon.flatten()[use]
        alt = alt.flatten()[use]
        if params is not None:
            ps = []
            for p in params:
                p_ = p[use]
                ps.append(p_)
                
        # Remove data/evaluation points outside grid in vertical direction
        k0 = self.get_alt_index(alts_grid, alt, returnfloat=True)
        inside = (k0 >= 0) & (k0<alts_grid.size-1)
        lat = lat[inside]
        lon = lon[inside]
        alt = alt[inside]
        inds = np.zeros(use.size).astype(int)
        temp = np.zeros(sum(use))
        temp[inside] = 1
        inds[use] = temp
        if params is not None:
            pss = []
            for p in ps:
                p__ = p[inside]
                pss.append(p__)
            if return_indices:
                return (pss, lat, lon, alt, inds)
            else:
                return (pss, lat, lon, alt)
        
        if return_indices:
            return (lat, lon, alt, inds)
        else:
            return (lat, lon, alt)
        
        
    def get_indices_kij(self, grid, alts_grid, lat, lon, alt, returnfloat=False):
        """
        Parameters
        ----------
        grid : CS grid object
            The grid we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J)
        alts_grid : array-like, 1D or 2D
            Altitude grid to use together with secs_grid. Length is K. Values represent the height in km of the centre of the
            voxels of each layer. Should be in increasing order.
        lat : array-like, 1D or 2D
        latitudes [deg] of the data/evaluation locations. Length N when 
        flattened to 1D. Must be inside secs_grid (not checked).
        lon : array-like, 1D or 2D
        longitudes [deg] of the data/evaluation locations. Length N when 
        flattened to 1D. Must be inside secs_grid (not checked).
        alt : array-like, 1D or 2D
            altitude [km] of the data/evaluation locations. Length N when 
            flattened to 1D.
        returnfloat : boolean, optional
            return the exact index of the input location. Defailt is False.
        Returns
        -------
        Tuple of indices (k,i,j). k is index in altitude, i is index in CS eta 
        direction, j is index in CS xi direction, refering to alts_grid. (k,i,j)
        is flattened before return.

        """
        
        if (alt.flatten().shape[0] != lon.flatten().shape[0]) | (alt.flatten().shape[0] != lat.flatten().shape[0]):
            print('Dimension mismathc in evaluation locations')
            print (1/0)
        
        binnumber = grid.bin_index(lon, lat)
        k = self.get_alt_index(alts_grid, alt, returnfloat=returnfloat).flatten()
        
        i = binnumber[0].flatten()
        j = binnumber[1].flatten()
        if returnfloat:
            xi_obs, eta_obs = grid.projection.geo2cube(lon, lat)
            xi_grid = grid.xi[i,j]
            eta_grid = grid.eta[i,j]
            i_frac = (eta_obs-eta_grid)/grid.deta
            j_frac = (xi_obs-xi_grid)/grid.dxi
            return (k, i+i_frac, j+j_frac)    
        else:
            return (k, i, j)
        
        
    def get_jr_matrix(self, grid, alts_grid, lat, lon, alt, interpolate=None, 
                    ext_factor=-1):
        """
        Parameters
        ----------
        grid : CS grid object
            The grid we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J). Assumes secs_grid.A refer to bottom layer.
        alts_grid : array-like, 1D or 2D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.
        lat : array-like, 1D or 2D
        latitudes [deg] of the data/evaluation locations. Flattened to 1D of 
        length N
        lon : array-like, 1D or 2D
        longitudes [deg] of the data/evaluation locations. Flattened to 1D of 
        length N
        alt : array-like, 1D or 2D
            altitude [km] of the data/evaluation locations. Flattened to 1D of 
            length N. Need to be within the range specified by alts_grid. Points 
            outside will be removed.
        highres : Boolean, optional
            Specifies whether to take into account the vertical placement of the
            data/evaluation location, and distribute the effect of the data into
            two layers: The layer it falls within, and the closest one. 
            Default is False
        interpolate : Boolean, optional
            Affect two things if True: 1) Vertical location of observation/evaluation
            location is implemented as a weighted contribution of the above and below
            node in the construction of S. 2) When making S, the closest
            four grid cells (in each layer) are considered, weighted by their 
            distance to the location in question (n) using bilinear interpolation. 
            Default is False.
        ext_factor : int, optional
            To control how to filter out locations based on their proximity to the 
            grid. The default is -1, removing points closer than 1 grid cell from 
            the edge of the grid (mesh).        
        Returns
        -------
        Tuple of indices (i,j,k). i is index in CS eta direction, j is index in 
        CS xi direction, k is index in altitude, refering to alts_grid.

        """
        # Remove data/evaluation points outside (2D) secs_grid, using the ingrid function
        use = grid.ingrid(lon.flatten(), lat.flatten(), ext_factor=ext_factor)
        lat = lat.flatten()[use]
        lon = lon.flatten()[use]
        alt = alt.flatten()[use]

        # Remove data/evaluation points outside grid in vertical direction
        k0 = self.get_alt_index(alts_grid, alt, returnfloat=True)
        inside = (k0 >= 0) & (k0<alts_grid.size-1)
        lat = lat.flatten()[inside]
        lon = lon.flatten()[inside]
        alt = alt.flatten()[inside]  
        
        # Evaluation locations.
        if not lat.shape[0]==lon.shape[0]==alt.shape[0]:
            print('Inconsistent dimensions of data/evaluation locations (lat,lon,alt)')
            print(1/0)        
            
        # Dimensions of 3D SECS grid
        N = lat.shape[0]
        I = grid.shape[0]
        J = grid.shape[1]
        K = alts_grid.shape[0]
        KIJ = K*I*J
        
        #Vertical resolution
        altres = np.diff(alts_grid)
        altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
        
        # Horizontal area of each 3D grid cell
        A0 = grid.A #in m2
        
        #Kalles broadcasting stuff. Maybe try this later to speed up
        # A = (A0.reshape((N, -1, I*J)))# / r.reshape((-1, r1d.size, phi1d.size * theta1d.size))).reshape(N, r.size)    
        # Ge = (Ge.reshape((N, -1, phi1d.size * theta1d.size)) / r.reshape((-1, r1d.size, phi1d.size * theta1d.size))).reshape(N, r.size)        


        #Indices of each evaluation point in 3D
        k, i, j = self.get_indices_kij(grid, alts_grid, lat, lon, alt, 
                                returnfloat=interpolate)
        # Corresponding flattened index    
        kij = np.ravel_multi_index((np.round(k).astype(int),np.round(i).astype(int),
                                    np.round(j).astype(int)), (K,I,J)) #flattened index
        
        # Make the vertical integration matrix S. S is very sparse. TO BE IMPLEMENTED
        S = np.zeros((N, KIJ)) #The final integration matrix
        for (counter,idx) in enumerate(kij): #each evaluation/data point  
            # print(counter)#k[counter], i[counter], j[counter])      
            temp = np.zeros(KIJ) #The part of S corresponding to observation idx
            if interpolate:
                k_n = np.arange(0, np.ceil(k[counter])+1).astype(int)
                if k[counter] % 1 == 0: #add above layer, but set to 0 weight
                    k_n = np.hstack((k_n,k_n[-1]+1))
                ks = k_n.size
                alt_n = alts_grid[0:np.floor(k[counter]).astype(int)+2]            
                if (i[counter]<=1) | (i[counter]>=I-1) | (j[counter]<=1) | (j[counter]>=J-1):
                    # Do not interpolate horizontally on points close to edge
                    i_n = np.ones(k_n.size).astype(int)*np.round(i[counter]).astype(int)
                    j_n = np.ones(k_n.size).astype(int)*np.round(j[counter]).astype(int)
                    fill =  np.ravel_multi_index((k_n, i_n, j_n), (K, I, J)) #flattened index   
                    dr = altres[k_n] * 1e3 #altitude range of layer in m
                    temp[fill] = -dr/(A0[i_n,j_n] * (alt_n+self.RE)/(alt_n[0]+self.RE))
                    #negative sign due to the sign convention of amplituded and FAC defined by Amm     

                    #Apply the linear interpolation scheme in vertical direction
                    k_frac = k[counter] % 1
                    w_over = k_frac
                    w_under = 1 - w_over
                    under_ = np.take(fill,np.array([ks-2]))
                    over_ = np.take(fill,np.array([ks-1]))
                    temp[under_] = temp[under_] * w_under
                    temp[over_] = temp[over_] * w_over
                else: #Point is in interior             
                    # Identify the four neighboring secs nodes. Since the four points share
                    # xi, eta coordinates, only two points are needed to be computed
                    xi1 = grid.xi[np.floor(i[counter]).astype(int),
                                    np.floor(j[counter]).astype(int)]
                    eta1 = grid.eta[np.floor(i[counter]).astype(int),
                                    np.floor(j[counter]).astype(int)]            
                    # xi2 = grid.xi[np.ceil(i[counter]).astype(int),
                                    # np.floor(j[counter]).astype(int)]
                    eta2 = grid.eta[np.ceil(i[counter]).astype(int),
                                    np.floor(j[counter]).astype(int)]                        
                    # xi3 = grid.xi[np.ceil(i[counter]).astype(int),
                                    # np.ceil(j[counter]).astype(int)]
                    # eta3 = grid.eta[np.ceil(i[counter]).astype(int),
                                    # np.ceil(j[counter]).astype(int)]                        
                    xi4 = grid.xi[np.floor(i[counter]).astype(int),
                                    np.ceil(j[counter]).astype(int)]
                    # eta4 = grid.eta[np.floor(i[counter]).astype(int),
                                    # np.ceil(j[counter]).astype(int)]
                    xi_obs, eta_obs = grid.projection.geo2cube(lon[counter], lat[counter])
                    #Bilinear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
                    w1 = (xi4-xi_obs)*(eta2-eta_obs) / ((xi4-xi1)*(eta2-eta1)) #w11
                    w2 = (xi4-xi_obs)*(eta_obs-eta1) / ((xi4-xi1)*(eta2-eta1)) #w12
                    w3 = (xi_obs-xi1)*(eta_obs-eta1) / ((xi4-xi1)*(eta2-eta1)) #w22
                    w4 = (xi_obs-xi1)*(eta2-eta_obs) / ((xi4-xi1)*(eta2-eta1)) #w21
                    wij = np.hstack((np.tile(w1,ks),np.tile(w2,ks),np.tile(w3,ks),np.tile(w4,ks)))
                    #Where to fill in temp array for this observation/evaluation location
                    #Here all layers are treated the same way, just scaled with area
                    k_ns = np.tile(k_n,4) 
                    i_ns = np.hstack((np.ones(k_n.size).astype(int)* 
                                    np.floor(i[counter]).astype(int),
                                    np.ones(k_n.size).astype(int)*
                                    np.ceil(i[counter]).astype(int),
                                    np.ones(k_n.size).astype(int)*
                                    np.ceil(i[counter]).astype(int),
                                    np.ones(k_n.size).astype(int)*
                                    np.floor(i[counter]).astype(int)))
                    j_ns = np.hstack((np.ones(k_n.size).astype(int)*
                                    np.floor(j[counter]).astype(int),
                                    np.ones(k_n.size).astype(int)*
                                    np.floor(j[counter]).astype(int),
                                    np.ones(k_n.size).astype(int)*
                                    np.ceil(j[counter]).astype(int),
                                    np.ones(k_n.size).astype(int)*
                                    np.ceil(j[counter]).astype(int)))       
                    fill =  np.ravel_multi_index((k_ns, i_ns, j_ns), (K, I, J)) #flattened index   
                    dr = np.tile(altres[k_n],4) * 1e3 #altitude range of layer in m
                    temp[fill] = -dr*wij/(A0[i_ns,j_ns] * (np.tile(alt_n,4)+self.RE)/(alt_n[0]+self.RE))            
                    # Negative sign due to the sign convention of amplituded and FAC 
                    # defined by Amm
        
                    #Apply the linear interpolation scheme in vertical direction
                    k_frac = k[counter] % 1
                    w_over = k_frac
                    w_under = 1 - w_over
                    under_ = np.take(fill,np.array([ks-2, 2*ks-2, 3*ks-2, 4*ks-2]))
                    over_ = np.take(fill,np.array([ks-1, 2*ks-1, 3*ks-1, 4*ks-1]))
                    temp[under_] = temp[under_] * w_under
                    temp[over_] = temp[over_] * w_over          

            else:
                k_n = np.arange(0, k[counter]+1)
                alt_n = alts_grid[0:np.floor(k[counter]).astype(int)+1]
                i_n = np.ones(k_n.size).astype(int)*np.round(i[counter]).astype(int)
                j_n = np.ones(k_n.size).astype(int)*np.round(j[counter]).astype(int)
                fill =  np.ravel_multi_index((k_n, i_n, j_n), (K, I, J)) #flattened index   
                dr = altres[k_n] * 1e3 #altitude range of layer in m
                temp[fill] = -dr/(A0[i[counter],j[counter]] * (alt_n+self.RE)/(alt_n[0]+self.RE))
                #negative sign due to the sign convention of amplituded and FAC defined by Amm

            S[counter,:] = temp # populate the row
            
        return S
    
    
    def count_kij(self, grid, alts_grid, lon, lat, alt, **kwargs):
        """ 
        Count number of points in each grid cell

        Parameters
        ----------
        lon : array
            array of longitudes [degrees]. Must have same size as lat
        lat : array
            array of latitudes [degrees]. Must have same size as lon
        kwargs : dict, optional
            passed to numpy.histogramdd. Use this if you want density, 
            normed, or weighted histograms for example. 


        Returns
        -------
        count : array
            array with count of how many of the coordinates defined
            by lon, lat, alt are in each grid cell. Same shape as lat,
            lon, and alt.
        """

        lon, lat, alt = lon.flatten(), lat.flatten(), alt.flatten()
        xi, eta = grid.projection.geo2cube(lon, lat)

        xi_edges, eta_edges = grid.xi_mesh[0, :], grid.eta_mesh[:, 0]
        altres = np.diff(alts_grid)*0.5
        altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
        alts_edges = np.concatenate((alts_grid - altres, [alts_grid[-1]+altres[-1]]))
        count, xi_ = np.histogramdd((alt, eta, xi), bins=(alts_edges, eta_edges, 
                                                        xi_edges), **kwargs)

        return count
    
    
    def make_P(self, N):
        """
        Function to make permutation matrix that act on a 1D array of length 3N 
        with a vectorwise representation i.e. 
        v = [r1, theta1, phi1, ... rN, thetaN, phiN]. Result of P.dot(v) is a 3N 1D
        array with componentwise representation, 
        i.e. [r1...rN, theta1...thetaN, phi1...phiN]. P is orthogonal, hence 
        P**-1 = P.T

        Parameters
        ----------
        N : int
            Number of measurements.

        Returns
        -------
        P : 2D array
            Permutation matrix that will produce component wise representation
            when acting on a vectorized representation.

        """
        P = np.zeros((3*N,3*N))
        for n in range(N):
            P[3*n+0, n+0] = 1
            P[3*n+1, n+0+N] = 1
            P[3*n+2, n+0+2*N] = 1
        return P


    def make_L(self, grid, alts_grid):
        '''
        Make the matrix L with size (N x 2*K*I*J) that has the property that when 
        multiplied with the  3D SECS model vector m (size 2*K*I*J), it produces 
        the 1D (shape N) array where each element corresponds to the radial current 
        density described by the 3D model on top of the domain.
            
        Parameters
        ----------
        grid : CS grid object
            The grid we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J). 
        alts_grid : array-like, 1D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.

        Returns
        -------
        L (N x 2KIJ) matrix.

        '''
        Sfac = self.get_jr_matrix(grid, alts_grid, grid.lat_mesh[1:-1,1:-1].flatten(), 
                grid.lon_mesh[1:-1,1:-1].flatten(), 
                np.ones(grid.lon_mesh[1:-1,1:-1].flatten().size)*alts_grid[-1]*.999, 
                interpolate=True, ext_factor=0) 
        O = np.zeros(Sfac.shape)
        L = np.hstack((Sfac, O))
        
        return L
    
    
    def make_F(self, grid, jr, lat, lon):
        '''
        This is an interpolation function to evaluate jr at any (lon,lat) location
        within the top face of the 3D domain, based on its values as represented on
        the 2D grid. The result (F) is a 1D array of length N, being the length of the
        input arrays lat and lon, of the values of the prescribed jr at the top of
        the 3D domain at each of the observation locations (lat,lon).
        
        Why use this function and not just directly use the sampled values from GEMINI?
            - This function allows for evaluation at desirable locations, that is on the
            grid.mesh_lat/lon grid, to avoid singularity effects. This is generally 
            recommended, but could also be done in the call to the 
            gemini_tools.sample_points() function. 
            - This function removes datapoints outside the grid. Hence, this function
            is not strictly needed as long as one keep these points in mind.

        Parameters
        ----------
        grid : CS grid object
            The grid we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J).
        jr : 2D array
            Prescribed jr at top layer in A/m2, shape (IxJ) 
        alts_grid : array-like, 1D
            Altitude grid to use together with grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.
        lat : array-like, 1D or 2D
        latitudes [deg] of the data/evaluation locations. Flattened to 1D of 
        length N. Must be in same coordinate system as grid object
        lon : array-like, 1D or 2D
        longitudes [deg] of the data/evaluation locations. Flattened to 1D of 
        length N. Must be in same coordinate system as grid object.


        Returns
        -------
        F : 1D array
            The FAC magnitude at top of 3D domain for each observation. Shape N.

        '''
        inside =  grid.ingrid(lon.flatten(), lat.flatten(), ext_factor = 0)
        f = RectBivariateSpline(grid.eta[:,0], grid.xi[0,:], jr, kx=1, ky=1)
        xi_obs, eta_obs = grid.projection.geo2cube(lon.flatten()[inside], lat.flatten()[inside])
        F = f(eta_obs,xi_obs, grid=False)
        
        return F
    
    
    def make_JHP(self, convection, simulation, grid, lat, lon, alt, inputmode='vi', hop='p', 
                return_indices=True):
        '''
        Make the matrix JHP with shape (N x 2*K*I*J) that when multiplied with m
        give the magnitude of the Hall or Pedersen current,
        as specified with the hop keyword, at the measurement locations (lat, lon, alt). 
        We use a model sigmah/p profile to obtain estimats of this quantity at 
        (lat, lon, alt). These locations should be a uniform mesh inside the domain 
        to help the inversion in regions with no other observations. The lat,lon locations
        should ideally be at the midpoint between SECS nodes.
                
        Parameters
        ----------  
        convection : instance of convection class
            Contain lompe representation. convection.data also contain the instance of the
            data class of the E3D like samples used to make the lompe fit, but also contain
            all samples also below maph.
        simulation : instance of simulation class
            To hold the GEMINI data and grid
        grid : instance of grid class
            contain, among other things, the grid we use to compute gridded indices of
            data/evaluation locations, as grid.grid, with dimension (I,J). 
            Also contains  alts_grid : array-like, 1D or 2D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.
        lat : array-like
            latitudes [deg] of the data/evaluation locations. Flattened to 1D of 
            length N
        lon : array-like
            longitudes [deg] of the data/evaluation locations. Flattened to 1D of 
            length N
        alt : array-like
            altitude in km of the data/evaluation locations. Flattened to 1D of 
            length N
        inputmode : str
            Determines how the input jperp to the 3D model is obtained. Must be one of:
                vi             : use sampled ion velocities, and jperp=ne(vi_perp-ve_perp)
                vi_ohmslaw     : use sampled ion velocities, but estimate jperp using
                                Ohms law, meaning that a conductance estimate must also
                                be used
                phitop         : Use potential at top from GEMINI and (vi-ve)                             
                phitop_ohmslaw : Use potential at top from GEMINI for E, and Ohms law
                jperp          : Use jperp directly sampled from GEMINI       
        
        hop : str
            Specifies whether the JHP matrix should compute the Hall or Pedersen current. 
        return_indices : boolean
            Return the indices of which elements has been kept. 1 means kept, 0 is 
            dropped.
        xgdat : tuple
            First element is the GEMINI grid object, second is the GEMINI data object        
        lmodel : lompe model object
            Must be provided if inputmode is vi or vi_ohmslaw
            
        Returns
        -------
        JHP, 2D array with shape (N x 2*K*I*J)
        '''
        
        lat = lat.flatten()
        lon = lon.flatten()
        alt = alt.flatten()
        _lat, _lon, _alt, inds = self.remove_outside(grid.grid, grid.alts_grid, lat, lon, alt, 
                                                ext_factor=-1, return_indices=True)  

        _data = data.data(grid, simulation, beams=False, uniformmesh=True, lat_ev=_lat, 
                            lon_ev=_lon, alt_ev=_alt, e3doubt_=False)
        

        # Get the E-filed unit vectors
        if (inputmode=='phitop') or (inputmode=='phitop_ohmslaw'):
            Ee = _data.Ee
            En = _data.En
            Eu = _data.Eu
            Emag = np.sqrt(Ee**2 + En**2 + Eu**2)
            ehatr = Eu/Emag
            ehattheta = -En/Emag
            ehatphi = Ee/Emag    
        else:
            # Obtain unit vectors of E field at observation locations. This involve
            # mapping the Lompe representation to each observation in the below function       
            ehate, ehatn, ehatu = convection.ehat(simulation, _data)
            nans = ~np.isfinite(ehate)
            ehate[nans] = 1 # not sure why there is nans. only appear in one corner?
            ehatn[nans] = 0        
            ehatu[nans] = 0    
            ehatr = ehatu
            ehattheta = -ehatn
            ehatphi = ehate
        
        # SECS matrices
        br, btheta, bphi = self.make_b_unitvectors(_data.Bu, -_data.Bn, _data.Be)
        B = self.make_B(br, btheta, bphi)
        G = self.make_G(grid.grid, grid.alts_grid, _lat, _lon, _alt, interpolate=True, 
                    jperp=B)
        N = _lat.size
        Gr = G[0:N,:]
        Gtheta = G[N:2*N,:]
        Gphi = G[2*N:3*N,:]
        # To calculate Hall and Pederse currents from the model, one projects the modelled
        # jperp along ehat and bhat x ehat
        if hop.lower() == 'p':
            JHP = np.multiply(ehatr[:,np.newaxis], Gr) + \
                        np.multiply(ehattheta[:,np.newaxis], Gtheta) + \
                        np.multiply(ehatphi[:,np.newaxis], Gphi)
        if hop.lower() == 'h':
            JHP = np.multiply((btheta*ehatphi)[:,np.newaxis], Gr) - \
                        np.multiply((br*ehatphi)[:,np.newaxis], Gtheta) + \
                        np.multiply((br*ehatphi-btheta*ehatr)[:,np.newaxis], Gphi)    

        # The following code was used earlier, trying to take advantage of knowledge of
        # Hall and Pedersen currents being DF and CF. May be a relevant assumption to
        # put back in again.
        # Ge_cf, Gn_cf, Ge_df, Gn_df = get_SECS_J_G_matrices_3D(
        #         secs_grid, alts_grid, _lat, _lon, _alt, interpolate=True, 
        #         singularity_limit=secs_grid.Lres, ext_factor=-1)
        # if hop.lower() == 'p':
        #     if pederseniscf:
        #         Ge = np.hstack((Ge_cf, np.zeros(Ge_cf.shape)))
        #         Gn = np.hstack((Gn_cf, np.zeros(Ge_cf.shape)))
        #     else:
        #         Ge = np.hstack((Ge_cf, Ge_df))
        #         Gn = np.hstack((Gn_cf, Gn_df))
        #     JHP = np.multiply(ehate[:,np.newaxis], Ge) + \
        #                 np.multiply(ehatn[:,np.newaxis], Gn)
        # if hop.lower() == 'h':
        #     if hallisdf:
        #         Ge = np.hstack((np.zeros(Ge_cf.shape), Ge_df))
        #         Gn = np.hstack((np.zeros(Ge_cf.shape), Gn_df))
        #     else:
        #         Ge = np.hstack((Ge_cf, Ge_df))
        #         Gn = np.hstack((Gn_cf, Gn_df))        
        #     Bn, Bu = gemini_tools.dipole_B(np.radians(90-_lat), height=_alt)
        #     Be = np.zeros(Bn.shape)
        #     br, btheta, bphi = make_b_unitvectors(Bu, -Bn, Be)
        #     JHP = np.multiply((-ehatn*br)[:,np.newaxis], Ge) + \
        #                  np.multiply((ehate*br)[:,np.newaxis], Gn)   
            
        if return_indices:
            return (JHP, inds)
        else:
            return JHP
        
        
    def make_D(self, grid, simulation, convection, lat, lon, alt, hop='p', return_indices=True, 
            pederseniscf=True, hallisdf=True):
        '''
        Make the matrix D with shape (N x 2*K*I*J) that when multiplied with m
        give the magnitude of the derivative in the vertical direction of the Hall 
        or Pedersen current, as specified with the hop keyword, at the measurement 
        locations (lat, lon, alt). We use a model sigmah/p profile to obtain 
        estimats of this quantity at (lat, lon, alt). These locations should be a 
        uniform mesh inside the domain to help the inversion in regions with no 
        other observations. The altitudes should be at the centre point between 
        the layers.
        
        A central differencing scheme is used based only on the two closest layers
        in the model (vertical direction.)

        Parameters
        ----------
        grid : instance of grid class
            contain, among other things, the grid we use to compute gridded indices of
            data/evaluation locations, as grid.grid, with dimension (I,J). 
            Also contains  alts_grid : array-like, 1D or 2D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.        
        simulation : instance of simulation class
            To hold the GEMINI data and grid
        convection : instance of convection class
            Contain lompe representation. convection.data also contain the instance of the
            data class of the E3D like samples used to make the lompe fit, but also contain
            all samples also below maph.                 
        lat : array-like
            latitudes [deg] of the data/evaluation locations. Flattened to 1D of 
            length N
        lon : array-like
            longitudes [deg] of the data/evaluation locations. Flattened to 1D of 
         length N
        alt : array-like
            altitude in km of the data/evaluation locations. Flattened to 1D of 
            length N
        hop : str
            Specifies whether the D matrix should compute the vertical gradient of the
            Hall or Pedersen current. 
        return_indices : boolean
            Return the indices of which elements has been kept. 1 means kept, 0 is 
            dropped.
        pederseniscf : boolean
            Make the assuption that the pedersen current is curl-free. Default is True.
        hallisdf : boolean
            Make the assuption that the hall current is divergence-free. Default is True.
            
        Returns
        -------
        D, 2D array with shape (N x 2*K*I*J)
        
        '''
        
            
        # Prepare input locations: above layer
        k, i, j = self.get_indices_kij(grid.grid, grid.alts_grid, lat, lon, alt, 
                                returnfloat=True)
        ks = np.ceil(k).astype(int)
        moduluses = k % 1
        ks[moduluses==0] =ks[moduluses==0]+1
        __alt = grid.alts_grid[ks]
        _lat, _lon, _alt, inds = self.remove_outside(grid.grid, grid.alts_grid, lat, lon, __alt, 
                                                ext_factor=-1, return_indices=True)  
        _data = data.data(grid, simulation, beams=False, uniformmesh=True, lat_ev=_lat, 
                            lon_ev=_lon, alt_ev=_alt, e3doubt_=False)        
        
        # Get the E-filed unit vectors    
        ehate, ehatn, ehatu = convection.ehat(simulation, _data)
        nans = ~np.isfinite(ehate)
        ehate[nans] = 1 # not sure why there is nans. only appear in one corner?
        ehatn[nans] = 0
        
        # SECS matrices of above
        Ge_cf, Gn_cf, Ge_df, Gn_df = self.get_SECS_J_G_matrices_3D(
                grid.grid, grid.alts_grid, _lat, _lon, _alt, interpolate=True, 
                singularity_limit=grid.Lres, ext_factor=-1)
        if hop.lower() == 'p':
            if pederseniscf:
                Ge = np.hstack((Ge_cf, np.zeros(Ge_cf.shape)))
                Gn = np.hstack((Gn_cf, np.zeros(Ge_cf.shape)))
            else:
                Ge = np.hstack((Ge_cf, Ge_df))
                Gn = np.hstack((Gn_cf, Gn_df))
            Gabove = np.multiply(ehate[:,np.newaxis], Ge) + \
                        np.multiply(ehatn[:,np.newaxis], Gn)
        if hop.lower() == 'h':
            if hallisdf:
                Ge = np.hstack((np.zeros(Ge_cf.shape), Ge_df))
                Gn = np.hstack((np.zeros(Ge_cf.shape), Gn_df))
            else:
                Ge = np.hstack((Ge_cf, Ge_df))
                Gn = np.hstack((Gn_cf, Gn_df))        
            br, btheta, bphi = self.make_b_unitvectors(_data.Bu, -_data.Bn, _data.Be)
            Gabove = np.multiply((-ehatn*br)[:,np.newaxis], Ge) + \
                        np.multiply((ehate*br)[:,np.newaxis], Gn)     

        # Prepare input locations: below layer
        kabove, iabove, jabove = self.get_indices_kij(grid.grid, grid.alts_grid, _lat, _lon, _alt, 
                                returnfloat=True)
        _alt = self.alts_grid[kabove.astype(int)-1]
        
        # SECS matrices of below
        Ge_cf, Gn_cf, Ge_df, Gn_df = self.get_SECS_J_G_matrices_3D(
                grid.grid, grid.alts_grid, _lat, _lon, _alt, interpolate=True, 
                singularity_limit=grid.Lres, ext_factor=-1)
        if hop.lower() == 'p':
            if pederseniscf:
                Ge = np.hstack((Ge_cf, np.zeros(Ge_cf.shape)))
                Gn = np.hstack((Gn_cf, np.zeros(Ge_cf.shape)))
            else:
                Ge = np.hstack((Ge_cf, Ge_df))
                Gn = np.hstack((Gn_cf, Gn_df))
            Gbelow = np.multiply(ehate[:,np.newaxis], Ge) +  \
                        np.multiply(ehatn[:,np.newaxis], Gn)
        if hop.lower() == 'h':
            if hallisdf:
                Ge = np.hstack((np.zeros(Ge_cf.shape), Ge_df))
                Gn = np.hstack((np.zeros(Ge_cf.shape), Gn_df))
            else:
                Ge = np.hstack((Ge_cf, Ge_df))
                Gn = np.hstack((Gn_cf, Gn_df))  
            br, btheta, bphi = self.make_b_unitvectors(_data.Bu, -_data.Bn, _data.Be)
            Gbelow = np.multiply((-ehatn*br)[:,np.newaxis], Ge) + \
                        np.multiply((ehate*br)[:,np.newaxis], Gn)  
        
        # Altitude differnce
        altres = np.diff(grid.alts_grid*1000)
        altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
        altdiff = altres[kabove.astype(int)]
        
        # Do the differencing between layers
        _D = Gabove - Gbelow
        D = np.multiply(1/altdiff[:,np.newaxis], _D)
        
        if return_indices:
            return (D, inds)
        else:
            return D
        
        
    def sigma_profile(self, grid, simulation, alt, lat = None, lon = None, iri = True, 
                    derivative=False, xgdat=None):
        '''
        Return values of the Hall and Pedersen conductivities at input locations.
        Conductivities are calculated based on reference MSIS density profiles, and
        IRI electron density profile if IRI=True.
        If derivative=True it returns the vertical gradient of Hall and Pedersen
        currents. MSIS is used for the neutral atmosphere. IRI (if set to True) or 
        the GEMINI simulation can be used for the electron density. The purpose is 
        to inform the 3D current density model about the vertical structure of the 
        currents when no other measurements are present.

        Parameters
        ----------
        grid : instance of grid class
            contain, among other things, the grid we use to compute gridded indices of
            data/evaluation locations, as grid.grid, with dimension (I,J). 
            Also contains  alts_grid : array-like, 1D or 2D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.        
        simulation : instance of simulation class
            To hold the GEMINI data and grid        
        alt : array-like
            Contains the input altitudes in km.
        lat : None or array-like
            If None, use the Skibotn transmitter site as location of model profile.
            Alternatively, an array of values, correspoinding to alt can be provided,
            in degrees.
        lon : None or array-like
            If None, use the Skibotn transmitter site as location of model profile.
            Alternatively, an array of values, correspoinding to alt can be provided,
            in degrees.        
        iri : boolean
            if True, IRI is used for electron density instead of the model values from
            GEMINI inside datadict
        derivative : boolean
            Specifies if the returned H/P profiles should be the derivative or not.
            Default is False.
        xgdat : tuple
            First element is the GEMINI grid object, second is the GEMINI data object.        

        Returns
        -------
        Tuple of two 1D (flattened input) arrays of modelled hall and pedersen 
        conductivity [S/m^2], or its derivative.

        '''
        
        #Constants for conductivity estimation
        kn2 = 4.34e-16
        ko2 = 4.28e-16
        ko = 2.44e-16
        e = 1.6e-19 # electron charge
        mp = 1.67e-27 #proton mass in kg
        ZN = 7
        ZO = 8
        # m_n2 = 2*2*ZN * mp
        m_no = (2*ZN + 2*ZO) * mp        

        # Load MSIS vertical neutral densiry profiles (from CCMC)
        # msis = pd.read_csv('./inclination_model/msis-90.txt', header=7, names=['Height','O', 'N2', 'O2', 
        #      'Temperature_neutral', 'Temperature_exospheric'], 
        #        delim_whitespace=True)
        names = ['year', 'month', 'day', 'doy', 'unknown', 'Height', 'glat', 
                'glon', 'O', 'N2', 'O2', 'rho', 'Temperature_neutral', 'u1', 
                'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 
                'u12', 'u13', 'u14']
        msis = pd.read_csv('./utils/nrlmsis_2.0.txt', header=0, names=names, 
            delim_whitespace=True)
        
        # Vertical profile of magnetic field
        height = np.arange(1,1000,1)
        Bn, Bu = self.dipole_B(np.radians(90-63), height=height)
        Bmag = np.sqrt(Bn**2 + Bu**2)
        omega_no = e*Bmag/(m_no) #gyro freq
        c_brekke = kn2 * msis.N2.values[0:-1]*1e6 + ko2 * msis.O2.values[0:-1]*1e6 + \
                ko * msis.O.values[0:-1]*1e6 #eq 7.12 in Brekke book. Assumes average ion mass of 30.7.    

        if iri == True:
            # Read the text file
            with open('./utils/iri2020.txt', 'r') as file:
                lines = file.readlines()
            
            # Initialize empty lists to store data
            data = {
                'H (km)': [],
                'ELECTRON DENSITY (Ne/cm-3)': [],
                'TEMPERATURES (Tn/K)': [],
                'ION PERCENTAGES[%]*10': [],
                '1E16m-2': [],
            }
            
            # Set a flag to start reading data
            start_reading_data = False
            
            # Iterate through the lines and extract data
            for line in lines:
                if "    km    Ne" in line:
                    start_reading_data = True  # Start reading data when this line is encountered
                    continue
                if start_reading_data:
                    if line.strip() == '-' or not line.strip():
                        continue  # Skip empty lines or lines with just a dash
                    parts = line.split()
                    data['H (km)'].append(float(parts[0].split('-')[0]))  # Extract the lower limit of the altitude range
                    data['ELECTRON DENSITY (Ne/cm-3)'].append(float(parts[1]))
                    data['TEMPERATURES (Tn/K)'].append(float(parts[3]))
                    data['ION PERCENTAGES[%]*10'].append(int(parts[-1]))
                    data['1E16m-2'].append(float(parts[-2]))
            
            # Create a pandas DataFrame
            iri = pd.DataFrame(data)
            ne = iri['ELECTRON DENSITY (Ne/cm-3)'].values[1:]*1e6
        elif lat is None:
            # Sample vertical profile of electron density above Tromsø
            glat = 67.3
            glon = 23.6
            _data = data.data(grid, simulation, beams=False, uniformmesh=True, lat_ev=np.ones(height.size)*glat, 
                                lon_ev=np.ones(height.size)*glon, alt_ev=height, e3doubt_=False)              
            ne = _data.ne
        elif lat is not None:
            # lat and lon must be an array of locations to evaluate vertical ne profile
            lat = lat.flatten()
            lon = lon.flatten()
            _sps = []
            _shs = []
            for i in range(len(lat.flatten())):
                _data = data.data(grid, simulation, beams=False, uniformmesh=True, lat_ev=np.ones(height.size)*lat[i], 
                                    lon_ev=np.ones(height.size)*lon[i], alt_ev=height, e3doubt_=False)                 
                ne = _data.ne
                sp = e*ne/Bmag * (c_brekke * omega_no)/(omega_no**2 + c_brekke**2)
                sh = e*ne/Bmag * (c_brekke**2)/(omega_no**2 + c_brekke**2)
                _sps.append(sp)
                _shs.append(sh)
        else:
            print('Input parameters are not suppored')
            print(1/0)
        
        if lat is None:
            sp = e*ne/Bmag * (c_brekke * omega_no)/(omega_no**2 + c_brekke**2)
            sh = e*ne/Bmag * (c_brekke**2)/(omega_no**2 + c_brekke**2)

            if derivative:
                diff_sp = coordinates._diff(height*1000, sp)
                diff_sh = coordinates._diff(height*1000, sh)
        
                # Evaluate the vertical gradient at the input locations
                f_diffp = scipy.interpolate.interp1d(height, diff_sp)
                diffp = f_diffp(alt)
                f_diffh = scipy.interpolate.interp1d(height, diff_sh)
                diffh = f_diffh(alt)
            
                return (diffh, diffp)
            else:
                # Evaluate conductivities at the input locations
                f_p = scipy.interpolate.interp1d(height, sp)
                p = f_p(alt)
                f_h = scipy.interpolate.interp1d(height, sh)
                h = f_h(alt)
                
                return (h, p)            

        else:
            sps = []
            shs = []
            for i in range(len(lat.flatten())):
                sp = _sps[i]
                sh = _shs[i]
                
                if derivative:
                    diff_sp = coordinates._diff(height*1000, sp)
                    diff_sh = coordinates._diff(height*1000, sh)
            
                    # Evaluate the vertical gradient at the input locations
                    f_diffp = scipy.interpolate.interp1d(height, diff_sp)
                    diffp = f_diffp(alt[i])
                    f_diffh = scipy.interpolate.interp1d(height, diff_sh)
                    diffh = f_diffh(alt[i])
                    sps.append(diffp)
                    shs.append(diffh)
                else:
                    # Evaluate conductivities at the input locations
                    f_p = scipy.interpolate.interp1d(height, sp)
                    p = f_p(alt[i])
                    f_h = scipy.interpolate.interp1d(height, sh)
                    h = f_h(alt[i])                
                    sps.append(p)
                    shs.append(h)
            return (np.array(shs), np.array(sps))
        
        
    def dipole_B(self, theta, height = 500):
        '''
        Calculate the centered dipole magnetic field strength in magnetic up and north directions.
        These values does not match exactly with the ones from the GEMINI output, likely due to 
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
    
    
    def crossvalidation(self, grid, simulation, convection, GTG, GTd, altreg, 
                        inputmode='vi', steps=10):
        '''
        Do the Generalized Cross Validation analysis to determine the l1 regularization
        parameter for 3D inversion.

        Parameters
        ----------
        grid : instance of grid class
            contain, among other things, the grid we use to compute gridded indices of
            data/evaluation locations, as grid.grid, with dimension (I,J). 
            Also contains  alts_grid : array-like, 1D or 2D
            Altitude grid to use together with secs_grid. Length is K. Values 
            represent the height in km of the centre of the voxels of each layer. 
            Should be in increasing order.           
        simulation : instance of simulation class
            To hold the GEMINI data and grid
        convection : instance of convection class
            Contain lompe representation. convection.data also contain the instance of the
            data class of the E3D like samples used to make the lompe fit, but also contain
            all samples also below maph.                    
        GTG : 2D array
            GTG matrix of inverse problem.
        GTd : 1D array
            The data array.
        altreg : 1D array
            Scaling the 0th order Tikhonov regularization, due to varying alt. diff
            between SECS layers.
        xgdat : tuple
            First element is the GEMINI grid object, second is the GEMINI data object
        grid : CS grid object
            The grid we use to compute gridded indices of data/evaluation locations
            Dimension is (I,J). Assumes secs_grid.A refer to bottom layer.
        alts_grid : array-like, 1D
            Altitude grid to use together with secs_grid. Length is K. Values 
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
        steps : int, optional
            How many different l1 values to use. The default is 10.
        lmodel : lompe model object
            Must be provided if inputmode is vi or vi_ohmslaw

        Returns
        -------
        3 element tuple of the following:
        resnorm : 1D array
            norm of the reidual vector for each step.
        modelnorm : 1D array
            norm of the model vector for each step.
        ls : 1D array
            The l1 value used for each step.
        '''
        
        # Define evaluation grid used to judge performance
        altres = np.diff(grid.alts_grid)*0.5
        altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
        alts__ = grid.alts_grid[1:]-altres[1:]
        xi_e  = grid.grid.xi[0,1:] - grid.grid.dxi/2 
        eta_e = grid.grid.eta[1:,0]- grid.grid.deta/2
        alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, eta_e, xi_e, indexing='ij')
        lon_ev, lat_ev = grid.grid.projection.cube2geo(xi_ev, eta_ev)
        _data = data.data(grid, simulation, beams=False, uniformmesh=True, lat_ev=lat_ev, 
                            lon_ev=lon_ev, alt_ev=alt_ev, e3doubt_=False)
        if not 'phitop' in inputmode:
            vperp = convection.get_E_from_lmodel(simulation, _data, returnvperp=True)
            _data.vperp_electron = vperp
        inputdict = self.make_inputdict(_data, grid, inputmode=inputmode, ext_factor=-1, hp_from_brekke=False)
        d = np.hstack((inputdict['jperp'][2,:], -inputdict['jperp'][1,:], 
                    inputdict['jperp'][0,:])) # (r, theta, phi components)    

        br, btheta, bphi = self.make_b_unitvectors(inputdict['Bu'],-inputdict['Bn'],inputdict['Be'])
        B = self.make_B(br, btheta, bphi)
        G = self.make_G(grid.grid, grid.alts_grid, lat_ev.flatten()[inputdict['inds']==1], 
                lon_ev.flatten()[inputdict['inds']==1], 
                alt_ev.flatten()[inputdict['inds']==1], ext_factor=0, jperp=B)

        # Inversion
        gtg_mag = np.median(np.diagonal(GTG))
        ls = np.linspace(-7,4,steps)
        resnorm = []
        modelnorm = []
        print('Printing: lambda1, residual norm, model norm')
        for l in ls:
            Reg = 10**l * gtg_mag * np.diag(altreg)
            GG = GTG + Reg
            Cmpost = scipy.linalg.solve(GG, np.eye(GG.shape[0]))
            m = Cmpost.dot(GTd)
            # m = lstsq(GG, GTd, cond=0.)[0]
            res = (d - G.dot(m))
            resnorm.append(np.sqrt(np.sum(res**2)))
            modelnorm.append(np.sqrt(np.sum(m**2)))
            print(l, np.sqrt(np.sum(res**2)), np.sqrt(np.sum(m**2)))
        plt.figure()
        print(resnorm)
        print(modelnorm)
        plt.plot(ls, resnorm)
        plt.scatter(ls, resnorm)
        # plt.ylim(0,0.1)
        for i in range(steps):
            sss = '%4.1f' % ls[i]
            plt.text(ls[i], resnorm[i], sss)
        # plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('l1')
        plt.ylabel('residual norm')   
        
        return (resnorm, modelnorm, ls)    