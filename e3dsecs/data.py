import numpy as np
from . import coordinates
from gemini3d.grid.gridmodeldata import model2pointsgeogcoords, geog2dipole
import scipy
import e3doubt
from scipy.stats import multivariate_normal
import dipole

class data:

    def __init__(self, grid, simulation, beams=True, sitelat=67.36, sitephi=23.,
                 az=None, el=None, uniformmesh=False, lat_ev=None, lon_ev=None, 
                 alt_ev=None, e3doubt_=True, intsec = 5*60, min_alt=90, 
                 max_alt=500, dr=4) -> None:
        """_summary_

        Args:
            grid (grid object): The grid object
            simulation (silmulation object): The simulation object
            beams (bool, optional): Whether to sample along beams. Defaults to True.
        """ 
        
        self.RE = grid.RE
        self.intsec = intsec
        self.maph = simulation.maph
        self.sitelat = sitelat      # geo lat of transmitter. Skibotn: 69.39
        self.sitephi = sitephi      # geo lon of transmitter. Skibotn: 20.27
        
        if beams:
            self.sample_eiscat(simulation, sitelat=sitelat, sitephi=sitephi, az=az, el=el,
                               min_alt=min_alt, max_alt=max_alt, dr=dr)
        elif uniformmesh == True:
            self.sample_mesh(grid, simulation, lat_ev=lat_ev, lon_ev=lon_ev, alt_ev=alt_ev)
            
        if e3doubt_:
            try: #Try to use an existing file, since the e3doubt calculations take a while
                loaded = np.load('./inversion_coefs/datadict_temp.npy', allow_pickle=True).item()
                # Copy attributes from the loaded object to self
                self.__dict__.update(loaded.__dict__)               
                print('Using existing sampling file with estimated variances from E3DOUBT')
            except:            
                self.get_variances()
            self.add_noise()
        
            
    def sample_eiscat(self, simulation, dr = 4, az=None, el=None, sitelat=67.36, sitephi=23., 
                    min_alt=90, max_alt=500):
        """
        Function that sample from GEMINI output (linear interpolation from its 
        native grid) along prescribed beams (az,el).

        Parameters
        ----------
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
        O_sph = np.array([self.RE, 0, 0]) #site location vector
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
            root = np.sqrt(dot**2 - (self.RE**2-(self.RE+alt_grid)**2))
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
    
        # Populate beam configuration info
        self.alts = alt_grid
        self.az = np.array(az)
        self.el = np.array(el)        
        self.az_all = np.array(__az).flatten()
        self.el_all = np.array(__el).flatten()    
        self.shape = poss.shape[0]
        self.beams = True
        
        # Populate with values from GEMINI
        self.sample_gemini(simulation, poss)
        
        
    def sample_mesh(self, grid, simulation, lat_ev=None, lon_ev=None, alt_ev=None):
        '''
        Sample GEMINI at input locations. Input must be geographic locations.

        Parameters
        ----------
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
        
        if lat_ev is None:
            alts__ = grid.alts_grid[1:]-grid.altres[1:]
            xi_e  = grid.grid.xi[0,1:] - grid.grid.dxi/2 
            eta_e = grid.grid.eta[1:,0]- grid.grid.deta/2
            alt_ev, eta_ev, xi_ev = np.meshgrid(alts__, eta_e, xi_e, indexing='ij')
            lon_ev, lat_ev = grid.grid.projection.cube2geo(xi_ev, eta_ev)
        
        self.shape = lon_ev.shape
            
        # Make the array that contain the locations to be sampled
        poss = np.vstack(((alt_ev+self.RE).flatten(), (90-lat_ev).flatten(),
                    lon_ev.flatten())).T   
        
        # Populate with values from GEMINI
        self.sample_gemini(simulation, poss)

        
    def sample_gemini(self, simulation, poss):
        '''
        Populate data object with observations sampled from GEMINI at locations in poss
        '''
        
        # Now we can sample from GEMINI at the identified locations
        j1 = model2pointsgeogcoords(simulation.xg, simulation.dat['J1'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        je = model2pointsgeogcoords(simulation.xg, simulation.dat['je'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        jn = model2pointsgeogcoords(simulation.xg, simulation.dat['jn'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        ju = model2pointsgeogcoords(simulation.xg, simulation.dat['ju'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        Be = model2pointsgeogcoords(simulation.xg, simulation.dat['Be'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        Bn = model2pointsgeogcoords(simulation.xg, simulation.dat['Bn'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        Bu = model2pointsgeogcoords(simulation.xg, simulation.dat['Bu'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])

        # Populate the data object
        self.lat = 90-poss[:,1]
        self.lon = poss[:,2]
        self.alt = poss[:,0]-self.RE
        self.fac = j1
        self.je = je
        self.jn = jn
        self.ju = ju
        self.Be = Be
        self.Bn = Bn
        self.Bu = Bu
        self.vperpe = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpe'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.vperpn = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpn'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.vperpu = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpu'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.jperpe = model2pointsgeogcoords(simulation.xg, simulation.dat['jperpe'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.jperpn = model2pointsgeogcoords(simulation.xg, simulation.dat['jperpn'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.jperpu = model2pointsgeogcoords(simulation.xg, simulation.dat['jperpu'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.sp = model2pointsgeogcoords(simulation.xg, simulation.dat['sp'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.sh = model2pointsgeogcoords(simulation.xg, simulation.dat['sh'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.Ee = model2pointsgeogcoords(simulation.xg, simulation.dat['Ee'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.En = model2pointsgeogcoords(simulation.xg, simulation.dat['En'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.Eu = model2pointsgeogcoords(simulation.xg, simulation.dat['Eu'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.ne = model2pointsgeogcoords(simulation.xg, simulation.dat['ne'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.Te = model2pointsgeogcoords(simulation.xg, simulation.dat['Te'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.Ti = model2pointsgeogcoords(simulation.xg, simulation.dat['Ti'], 
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.vperpmappede = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpmappede'],
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.vperpmappedn = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpmappedn'],
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.vperpmappedu = model2pointsgeogcoords(simulation.xg, simulation.dat['vperpmappedu'],
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.mappedglat = model2pointsgeogcoords(simulation.xg, simulation.dat['mappedglat'],
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        self.mappedglon = model2pointsgeogcoords(simulation.xg, simulation.dat['mappedglon'],
                                (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])              
        self.ve = model2pointsgeogcoords(simulation.xg, simulation.dat['ve'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        self.vn = model2pointsgeogcoords(simulation.xg, simulation.dat['vn'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        self.vu = model2pointsgeogcoords(simulation.xg, simulation.dat['vu'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        self.v1 = model2pointsgeogcoords(simulation.xg, simulation.dat['v1'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        self.v2 = model2pointsgeogcoords(simulation.xg, simulation.dat['v2'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])
        self.v3 = model2pointsgeogcoords(simulation.xg, simulation.dat['v3'],(poss[:,0]-self.RE)*1e3,
                                        poss[:,2],90-poss[:,1])

        if 'divjperp' in simulation.dat.keys():
            self.divjperp = model2pointsgeogcoords(simulation.xg, simulation.dat['divjperp'],
                                    (poss[:,0]-self.RE)*1e3, poss[:,2],90-poss[:,1])
        
        # Sample electric potential
        # lx1 = xg["lx"][0]
        lx2 = simulation.xg["lx"][1]
        lx3 = simulation.xg["lx"][2]
        # inds1 = range(2, lx1 + 2)
        inds2 = range(2, lx2 + 2)
        inds3 = range(2, lx3 + 2)
        # x1 = xg["x1"][inds1]
        x2 = simulation.xg["x2"][inds2]
        x3 = simulation.xg["x3"][inds3]
        x1i, x2i, x3i = geog2dipole((poss[:,0]-self.RE)*1e3, poss[:,2], 90-poss[:,1])
        xi = np.array((x2i.ravel(), x3i.ravel())).transpose()
        if len(simulation.dat.Phitop.shape) == 2:
            self.Phitop = scipy.interpolate.interpn(
                points=(x2, x3),
                values=simulation.dat.Phitop.values,
                xi=xi,
                method="linear",
                bounds_error=False,
                fill_value=np.NaN)
        if len(simulation.dat.Phitop.shape) == 3: #Not sure what this means?
            self.Phitop = scipy.interpolate.interpn(
                points=(x2, x3),
                values=simulation.dat.Phitop.values[0,:,:],
                xi=xi,
                method="linear",
                bounds_error=False,
                fill_value=np.NaN)

        # Line-of-sight sampling: NOT FULLY IMPLEMENTED YET
        # Project the mapped Vperp, at the mapped locations, onto line-of-sight 
        # direction of each measurement
        # Should try to implement an option of subtracting field aligned compoenet 
        # using a field aligned beam.
        # In reality, we will need to make such an assumption, or similar.
        # Must convert vperp into ECEF frame before doing dot product
        # enu_vec = np.vstack((selfdict['vperpmappede'],selfdict['vperpmappedn'], 
        #                      selfdict['vperpmappedu'])).T
        # xyz_vec = coordinates.enu2xyz(enu_vec, selfdict['mappedglon'], selfdict['mappedglat'])
        # selfdict['vlos'] = xyz_vec[:,0]*np.array(lx).flatten() + \
        #         xyz_vec[:,1]*np.array(ly).flatten() + \
        #         xyz_vec[:,2]*np.array(lz).flatten()
        # # convert cartesian LOS unit vector to local ENU form (at measurement location)
        # l_xyz = np.vstack((np.array(lx).flatten(),np.array(ly).flatten(),np.array(lz).flatten())).T
        # l_enu = coordinates.xyz2enu(l_xyz, selfdict['mappedglon'], selfdict['mappedglat'])
        # # Horizontal part of LOS direction
        # hormag = np.sqrt(l_enu[:,0]**2 + l_enu[:,1]**2)
        # selfdict['l_hor_e'] =l_enu[:,0]/hormag
        # selfdict['l_hor_n'] = l_enu[:,1]/hormag
    
    
    def get_variances(self):
        dlat = 69.39- self.sitelat
        dlon = 20.27 - self.sitephi
        lats0 = np.array([69.39, 68.44, 68.37])
        lons0 = np.array([20.26, 22.48, 19.10])
        lats = np.array([self.sitelat, lats0[1]-dlat, lats0[2]-dlat])
        lons = np.array([self.sitephi, lons0[1]-dlon, lons0[2]-dlon])        
        
        transmitter = ('ski_mod',lats[0],lons[0])
        receivers = [('ski_mod',lats[0],lons[0]), ('krs_mod',lats[1],lons[1]), 
                ('kai_mod',lats[2],lons[2])]
        self.transmitter = transmitter
        self.receivers = receivers
        
        self.get_datacov_e3doubt(transmitter=transmitter, receivers=receivers)
        self.remove_bad()
        np.save('./inversion_coefs/datadict_temp.npy', self)
        
    
    def get_datacov_e3doubt(self, transmitter=('ski_mod', 67.2,23.7), 
                            receivers=[('ski_mod', 67.2,23.7), 
                                    ('kai_mod', 66.18,22.54), 
                                    ('krs_mod', 66.25,25.92)]):
        '''
        Thus function calls E3DOUBT to estimate datacovariance matrix at each
        sample location in ddict.
        '''
        
        exp = e3doubt.Experiment(el=self.el, az = self.az,h=self.alts, 
                                transmitter=transmitter, receivers=receivers)

        # exp.set_ionos('nuin',0)
        exp.run_models()
        
        #replace nans
        nans = ~np.isfinite(self.ne)
        self.ne[nans] = 1e9
        self.Te[nans] = 1000
        self.Ti[nans] = 1000

        exp.set_ionos('ne', self.ne)    
        exp.set_ionos('Te', self.Te)    
        exp.set_ionos('Ti', self.Ti)    

        uncert =exp.calc_uncertainties(integrationsec=self.intsec)

        cov = exp.get_velocity_cov_matrix()

        self.cov_vi = cov
        self.var_ne = uncert.dnemulti.values**2 # dnemulti is the standard deviation
        self.noise_added = False
        
        
    def remove_bad(self):
        '''
        Remove observation that have undefined covariaces from E3DOUBT
        '''
        N = self.lat.size
        ddict = vars(self)
        keys = ddict.keys()
        updatekeys = []
        updateshapes = []
        for key in keys:
            if (type(ddict[key]) is not int) and (type(ddict[key]) is not bool) and \
                    (type(ddict[key]) is not float) and (type(ddict[key]) is not tuple \
                         and (type(ddict[key]) is not list)):
                updatekeys.append(key)
                updateshapes.append(ddict[key].shape)
        
        bads = np.zeros(N)
        for n in range(N):
            nn = np.abs(self.cov_vi[:,:,n])
            if np.max(nn) > 1e10:
                bads[n] = 1
                
        use = np.isfinite(self.cov_vi[0,0,:]) & (bads==0)

        for i, key in enumerate(updatekeys):
            sh = updateshapes[i]
            ss = np.where(np.array(sh)==N)[0]
            if ss.size == 1:
                if ss[0] == 0:
                    setattr(self, key, ddict[key][use])
                if ss[0] == 2:
                    setattr(self, key, ddict[key][:,:,use])


    def add_noise(self):
        '''
        Add noise to the samples from GEMINI by using the covariace matrix obtained 
        from E3DOUBT.
        
        Noise is generated using scipy's multivariate_normal function by specifying 
        covarince from E3DOUBT. The noisy vi samples are then projected to vperp
        and vmappedperp. Finally a flag is set to indicate that noise is added.
        
        This function assumes that all vector quantities in this data class are in 
        geographic components unless explicitly stated something else.

        Parameters
        ----------
        minalt : int or float
            The altitude above where we assume ExB drift.
        '''
        N = self.lat.size

        # Arrays to hold the result
        v_enu_noisy = np.zeros((N,3))
        vperp_enu_noisy = np.zeros((N,3))
        vperpmapped_enu_noisy = np.zeros((N,3))
        ne_noisy = np.zeros(N)     
        
        # 0) obtain matrices that map the covairances:
        Gperpmapped = self._make_vperpmappedG()
        Gperp = self._make_vperpG()
        _perp = np.eye(3)
        _perp[0,0] = 0 # select only perp components in GEMINI basis
    
        # Alternative approach: First add noise to full v, then estimate vperp and vmappedperp
        for i in range(N): 
            # Noise in velocity observations   
            cov = self.cov_vi[:,:,i]
            vi = np.hstack((self.ve[i], self.vn[i], self.vu[i]))
            _noise = multivariate_normal(mean=np.zeros(3), cov=cov)
            noisy_obs = np.hstack((vi + _noise.rvs()))
            vperp_noisy = Gperp[:,:,i].dot(noisy_obs)
            vperpmapped_noisy = Gperpmapped[:,:,i].dot(noisy_obs)
            
            v_enu_noisy[i,:] = noisy_obs
            vperp_enu_noisy[i,:] = vperp_noisy
            vperpmapped_enu_noisy[i,:] = vperpmapped_noisy

            # Noise in electron density observations
            _noise = multivariate_normal(mean=0, cov=self.var_ne[i])
            ne_noisy[i] = self.ne[i] + _noise.rvs()
            
        # update ddict with the added noise
        self.vperpmappede = vperpmapped_enu_noisy[:,0]
        self.vperpmappedn = vperpmapped_enu_noisy[:,1]
        self.vperpmappedu = vperpmapped_enu_noisy[:,2]
        self.vperpe = vperp_enu_noisy[:,0]
        self.vperpn = vperp_enu_noisy[:,1]
        self.vperpu = vperp_enu_noisy[:,2]
        self.ne = ne_noisy
        
        # 3) Add the noise flag
        self.noise_added = True
        

    def _make_vperpmappedG(self):
        # Make the (3,3) matrix that act on a velocity measurement v=(ve,vn,vu) in
        # geographic components to produce
        # the value of the velocity perpendicular to B, mapped to the altitude minalt [km]
        # assuming no collissions and that E_par = 0: vperpmapped(e,n,u). The matrices are
        # stacked in the third dimension for each observation.
        # minalt is the altitude above where we assume ExB drift
        
        N = self.lat.size

        # First, calculate d vectors at the observation locations of v
        dip = dipole.Dipole(dipole_pole=((90-11),289))
        d1, d2, d3, _1, _2, _3 = dip.get_apex_base_vectors_geo(self.lon, 
                                    self.lat, self.alt+self.RE, R=self.RE)
        
        # Calculate e vectors at the mapped locations
        _1, _2, _3, e1, e2, e3 = dip.get_apex_base_vectors_geo(self.mappedglon, 
                            self.mappedglat, np.ones(N)*(self.RE+self.maph), R=self.RE)

        Gs = []
        for i in range(N):
            Gi = np.zeros((3, 3))
            Gi[0,:] = np.vstack((d1[0,i]*e1[0,i]+d2[0,i]*e2[0,i], d1[1,i]*e1[0,i]+d2[1,i]*e2[0,i], d1[2,i]*e1[0,i]+d2[2,i]*e2[0,i])).T
            Gi[1,:] = np.vstack((d1[0,i]*e1[1,i]+d2[0,i]*e2[1,i], d1[1,i]*e1[1,i]+d2[1,i]*e2[1,i], d1[2,i]*e1[1,i]+d2[2,i]*e2[1,i])).T
            Gi[2,:] = np.vstack((d1[0,i]*e1[2,i]+d2[0,i]*e2[2,i], d1[1,i]*e1[2,i]+d2[1,i]*e2[2,i], d1[2,i]*e1[2,i]+d2[2,i]*e2[2,i])).T
            Gs.append(Gi)
        Gs = np.swapaxes(np.swapaxes(np.array(Gs),1,0),2,1)    

        return Gs

    def _make_vperpG(self):
        # Make the (3,3) matrix that act on a velocity measurement v=(ve,vn,vu) in
        # geographic components to produce
        # the value of the velocity perpendicular to B: vperp(e,n,u). The matrices are
        # stacked in the third dimension for each observation
        
        N = self.lat.size

        # First, calculate d vectors at the observation locations of v
        dip = dipole.Dipole(dipole_pole=((90-11),289))
        d1, d2, d3, e1, e2, e3 = dip.get_apex_base_vectors_geo(self.lon, 
                                    self.lat, self.alt+self.RE, R=self.RE)
        Gs = []
        for i in range(N):
            Gi = np.zeros((3, 3))
            Gi[0,:] = np.vstack((d1[0,i]*e1[0,i]+d2[0,i]*e2[0,i], d1[1,i]*e1[0,i]+d2[1,i]*e2[0,i], d1[2,i]*e1[0,i]+d2[2,i]*e2[0,i])).T
            Gi[1,:] = np.vstack((d1[0,i]*e1[1,i]+d2[0,i]*e2[1,i], d1[1,i]*e1[1,i]+d2[1,i]*e2[1,i], d1[2,i]*e1[1,i]+d2[2,i]*e2[1,i])).T
            Gi[2,:] = np.vstack((d1[0,i]*e1[2,i]+d2[0,i]*e2[2,i], d1[1,i]*e1[2,i]+d2[1,i]*e2[2,i], d1[2,i]*e1[2,i]+d2[2,i]*e2[2,i])).T
            Gs.append(Gi)
        Gs = np.swapaxes(np.swapaxes(np.array(Gs),1,0),2,1)    

        return Gs