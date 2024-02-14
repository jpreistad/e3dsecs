import xarray as xr
import gemini3d.read as read
import numpy as np
from . import coordinates
from gemini3d.grid.convert import unitvecs_geographic
import dipole
from gemini3d.grid.convert import geomag2geog
import h5py

class simulation:
    def __init__(self, path, maph=200, timeindex=-1) -> None:
        """Initiate the simulation class
         The Function read_sim() is called, and first look for the GEMINI dump
         file 'gemini_dataset.nc' and 'gemini_grid.h5' that can be downloaded
         from the Zenodo repo https://doi.org/10.5281/zenodo.10561479
         
         If not, it tries to read from the full simulation output in the 
         same destination (path) and to generate the same files, based on the
         timeindex keyword.
        

        Args:
            path (str): path to the GEMINI simulation data
            maph (int, optional): Mapping height. Defaults to 200 km.
            timeindex (int, optional): Which time index from the GEMINI simulation 
                to use. Defaults to -1.
        """        
        
        self.path = path
        self.maph = maph
        self.RE = 6371.2 #Earth radius in km
        self.timeindex = timeindex
        self.read_sim()

        
    def read_sim(self):
        # Load GEMINI grid and data
        try: # look for saved file including some of the needed types of data    
            dat = xr.open_dataset(self.path + 'gemini_dataset.nc')
            xg = self.read_hdf5_to_dict(self.path + 'gemini_grid.h5')
            self.dat = dat
            self.xg = xg
        except: # make the datafiles from reading GEMINI output
            self.read_gemini(self.path, timeindex=self.timeindex)
            _dat = self.dat
            _dat.attrs={}
            _dat.to_netcdf(self.path + 'gemini_dataset.nc')
            _xg = self.xg
            del _xg['glatctr']
            del _xg['glonctr']
            del _xg['filename']
            with h5py.File(self.path + 'gemini_grid.h5', 'w') as file:
                for key, value in _xg.items():
                    file.create_dataset(key, data=value)


    def read_hdf5_to_dict(self, file_path):
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
    
    def read_gemini(self, path, timeindex=-1):
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
        

        Returns
        -------
        xg, dat : tuple
            First element is GEMINI grid dictionary, second element is an xarray 
            Dataset of the simulation dataset at timeindex.
        """    
        var = ["v1", "v2", "v3", "Phi", "J1", "J2", "J3", "ne" ,"Te", "Ti"]
        cfg = read.config(path)
        self.xg = read.grid(path)
        times = cfg["time"][timeindex:]
        t = times[0]
        self.dat = read.frame(path, t, var=var)
        sshh = self.dat.v1.values.shape
        self.calc_conductivities()
        self.compute_enu_components()
        
        # Do mapping of observation to maph altitude
        dip = dipole.Dipole(dipole_pole=((90-11),289)) # These locations found in pygemini doc.
        d1, d2, d3, _1, _2, _3 = dip.get_apex_base_vectors_geo(self.xg['glon'].flatten(), 
                            self.xg['glat'].flatten(), self.xg['alt'].flatten()*1e-3+self.RE, 
                            R=self.RE+110)
        v = np.vstack((self.dat.ve.values.flatten(),self.dat.vn.values.flatten(),
                       self.dat.vu.values.flatten()))
        #Calculate the quantities that is constant along the field-lines
        ve1 = (d1[0,:]*v[0,:] + d1[1,:]*v[1,:] + d1[2,:]*v[2,:]).reshape(sshh)
        ve2 = (d2[0,:]*v[0,:] + d2[1,:]*v[1,:] + d2[2,:]*v[2,:]).reshape(sshh)
        # Calculate the mapped geographic locations. Map from observed location (2) 
        # to the maph height (1) using dipole formula
        r_2 = self.xg['alt']*1e-3 + self.RE
        r_1 = np.ones(r_2.shape)*(self.RE+self.maph)
        colat_1 = np.arcsin(np.sin(self.xg['theta']) * np.sqrt(r_1/r_2))
        mlon_1 = self.xg['phi']# in radians
        mappedglon, mappedglat = geomag2geog(mlon_1, colat_1) #returns in degrees
        # Calculate basis vectors at the mapped locations
        _1, _2, _3, e1, e2, e3 = dip.get_apex_base_vectors_geo(mappedglon.flatten(), 
                            mappedglat.flatten(), r_1.flatten(), R=self.RE+110)
        #Calculate the mapped velocity using eq 4.17 in Richmond 1995. geographic components, ENU            
        vperpmappede = (ve1.flatten()*e1[0,:] + ve2.flatten()*e2[0,:]).reshape(sshh)
        vperpmappedn = (ve1.flatten()*e1[1,:] + ve2.flatten()*e2[1,:]).reshape(sshh)
        vperpmappedu = (ve1.flatten()*e1[2,:] + ve2.flatten()*e2[2,:]).reshape(sshh)
        self.dat['mappedglat'] = xr.DataArray(mappedglat, dims=('x1','x2','x3'))
        self.dat['mappedglon'] = xr.DataArray(mappedglon, dims=('x1','x2','x3'))
        self.dat['vperpmappede'] = xr.DataArray(vperpmappede, dims=('x1','x2','x3'))
        self.dat['vperpmappedn'] = xr.DataArray(vperpmappedn, dims=('x1','x2','x3'))
        self.dat['vperpmappedu'] = xr.DataArray(vperpmappedu, dims=('x1','x2','x3'))
    

    def calc_conductivities(self):
        '''
        Function that calculates the Hall and Pedersen conductivities used in GEMINI
        based on the currents and E-fields provided in the output files. This function
        also stores the E-field values in the native GEMINI grid in the dat structure.
        '''
        
        self.gemini_gradient(q='Phitop')
        shape = self.dat.J1.shape
        E2 = -self.dat.gradPhitop_x2
        E3 = -self.dat.gradPhitop_x3
        Emag = np.sqrt(E2**2 + E3**2)
        ehat = np.stack((np.zeros(shape), E2/Emag, E3/Emag))
        j_ped_mag = self.dat.J2 * ehat[1,:,:,:] + self.dat.J3 * ehat[2,:,:,:]
        sp = j_ped_mag / Emag
        j_hall_mag = -self.dat.J2*ehat[2,:,:,:] + self.dat.J3*ehat[1,:,:,:]
        sh = j_hall_mag / Emag
        
        self.dat['sp'] = xr.DataArray(sp, dims=('x1','x2','x3'))
        self.dat['sh'] = xr.DataArray(sh, dims=('x1','x2','x3'))
        self.dat['E1'] = xr.DataArray(np.zeros(E2.shape), dims=('x1','x2','x3'))
        self.dat['E2'] = xr.DataArray(E2, dims=('x1','x2','x3'))
        self.dat['E3'] = xr.DataArray(E3, dims=('x1','x2','x3'))
    
    
    def gemini_gradient(self, q='Phitop'):
        '''
        Compute the gradient of a scalar field, e.g. electric potential defined in GEMINI's
        curvlinear coordinates. Input arrays must be 2D or 3D arrays
        
        q: quantity to differentiate
        
        '''
        
        #Metric factors defined in eqs 114-116 in GEMINI documentation
        # h1 = xg['r']**3/(RE**2*np.sqrt(1+3*(np.cos(xg['theta']))**2))
        h2 = self.RE*1e3*(np.sin(self.xg['theta']))**3/np.sqrt(1+3*(np.cos(self.xg['theta']))**2)
        h3 = self.xg['r'] * np.sin(self.xg['theta'])
        
        ndim = len(self.dat.Phitop.shape)

        if ndim == 2:
            x2x2, x3x3 = np.meshgrid(self.xg['x2i'][1:], self.xg['x3i'][1:], indexing='ij')
            q2 = 1/h2 * coordinates.diff2d(x2x2, self.dat[q].values, axis=0)
            q3 = 1/h3 * coordinates.diff2d(x3x3, self.dat[q].values, axis=1)
            self.dat['grad'+q+'_x2'] = xr.DataArray(q2, dims=('x1','x2','x3'))
            self.dat['grad'+q+'_x3'] = xr.DataArray(q3, dims=('x1','x2','x3'))
        if ndim ==3:
            print('Not implemented')
            print(1/0)


    def compute_enu_components(self):
        """
        Add ENU components (geographic) of V, J and B to xarray dataset
        """
        
        #Convert velocity and current to grographic components, use ENU notation
        vu, ve, vn = self.model_vec2geo_vec(param='v')
        vperpu, vperpe, vperpn = self.model_vec2geo_vec(param='v', perp=True)
        jperpu, jperpe, jperpn = self.model_vec2geo_vec(param='J', perp=True)
        ju, je, jn = self.model_vec2geo_vec(param='J')
        
        #B vectors from model output, project on geo ENU frame
        [egalt,eglon,eglat]=unitvecs_geographic(self.xg)    
        Be = np.sum(self.xg["e1"]*eglon*self.xg['Bmag'][...,np.newaxis],3)
        Bn = np.sum(self.xg["e1"]*eglat*self.xg['Bmag'][...,np.newaxis],3)
        Bu = np.sum(self.xg["e1"]*egalt*self.xg['Bmag'][...,np.newaxis],3)
        
        # E-field
        Eu, Ee, En = self.model_vec2geo_vec(param='E', perp=True)

        # Add to dat structure
        self.dat['Ee'] = Ee
        self.dat['En'] = En
        self.dat['Eu'] = Eu
        self.dat['ve'] = ve
        self.dat['vn'] = vn
        self.dat['vu'] = vu
        self.dat['vperpe'] = vperpe
        self.dat['vperpn'] = vperpn
        self.dat['vperpu'] = vperpu
        self.dat['jperpe'] = jperpe
        self.dat['jperpn'] = jperpn
        self.dat['jperpu'] = jperpu
        self.dat['je'] = je
        self.dat['jn'] = jn
        self.dat['ju'] = ju
        self.dat['Be'] = xr.DataArray(Be, dims=('x1','x2','x3'))
        self.dat['Bn'] = xr.DataArray(Bn, dims=('x1','x2','x3'))
        self.dat['Bu'] = xr.DataArray(Bu, dims=('x1','x2','x3'))


    def model_vec2geo_vec(self, param='v', perp=False):
        '''
        Function to convert model vector components into geographic conponents. 
        Code provided by M. Zettergren, and put into this function by JPR.

        Parameters
        ----------
        param : 'str'
            'v' (default) or 'J', refering to velocity or current density
        perp : Boolean
            Specifies if only the perpendicular component (2 and 3) of param is to
            be projected to (r, theta, phi) components. Default is False.
            
        Returns
        -------
        (radial, east, north) geographic components of velocity

        '''
        
        [egalt,eglon,eglat]=unitvecs_geographic(self.xg)     #up, east, north
        #^ returns a set of geographic unit vectors on xg; these are in ECEF geomag
        # comps like all other unit vectors in xg

        # each of the components in models basis projected onto geographic unit 
        # vectors
        if perp:
            vgalt=(np.sum(self.xg["e2"]*egalt,3)*self.dat[param+"2"] + 
                np.sum(self.xg["e3"]*egalt,3)*self.dat[param+"3"] )
            vglat=(np.sum(self.xg["e2"]*eglat,3)*self.dat[param+"2"] +
                np.sum(self.xg["e3"]*eglat,3)*self.dat[param+"3"] )
            vglon=(np.sum(self.xg["e2"]*eglon,3)*self.dat[param+"2"] + 
                np.sum(self.xg["e3"]*eglon,3)*self.dat[param+"3"] )
        else:
            vgalt=( np.sum(self.xg["e1"]*egalt,3)*self.dat[param+"1"] + 
                np.sum(self.xg["e2"]*egalt,3)*self.dat[param+"2"] + 
                np.sum(self.xg["e3"]*egalt,3)*self.dat[param+"3"] )
            vglat=( np.sum(self.xg["e1"]*eglat,3)*self.dat[param+"1"] + 
                np.sum(self.xg["e2"]*eglat,3)*self.dat[param+"2"] +
                np.sum(self.xg["e3"]*eglat,3)*self.dat[param+"3"] )
            vglon=( np.sum(self.xg["e1"]*eglon,3)*self.dat[param+"1"] + 
                np.sum(self.xg["e2"]*eglon,3)*self.dat[param+"2"] + 
                np.sum(self.xg["e3"]*eglon,3)*self.dat[param+"3"] )
        
        return [vgalt, vglon, vglat] # (up, east, north)