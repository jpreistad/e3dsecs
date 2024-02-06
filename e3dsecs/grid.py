import numpy as np
from lompe.data_tools.dataloader import getbearing
from secsy import cubedsphere
import great_circle_calculator.great_circle_calculator as gcc


class grid:

    def __init__(self, simulation, alts=None, extend=5, dlat=0.3, dlon=0,
                 crop_factor=0.2, resolution_factor=0.45) -> None:
        """Make grid class to contain the analysis grid used in all steps of volumetric 
        reconstruction E3DSECS

        Args:
            simulation (simulation class): holding simulation grid and data. Here is maph set.
            alts (1D array, optional): Altitude grid. Defaults to None.
            maph (int, optional): height of mapping of ion velocity, in km. Defaults to 200.
            extend (int, optional): How many padding "frames" used for lompe grid around 
                3D hor. grid. Defaults to 5.
            dlat (float, optional): How much th shift the centre latitude of grid, 
                in degrees. Default is 0.
            dlon (int, optional):How much to shift the centre longitude of grid, 
                in degrees. Default is 0.
            resolution_factor (float, optional): How much to reduce the spatial resolution 
                compared to GEMINI. 0.5 will double the spacing of grid cells. 
                Defaults to 0.45.
            crop_factor (float): How much to reduce the CS grid compared to GEMINI grid.
                Default is 0.2
        """        
        
        self.RE = simulation.RE
        
        # Altitude grid
        if alts is None:
            alts_grid = np.concatenate((np.arange(90,140,5),np.arange(140,170,10), 
                                    np.arange(170,230,20),np.arange(230,500,50)))            
        else:
            alts_grid = alts

        altres = np.diff(alts_grid)*0.5
        altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
        
        # Horizontal CS grid
        grid, grid_l = self.make_csgrid(simulation.xg, maph=simulation.maph, 
                                            h0=alts_grid[0], crop_factor=crop_factor, 
                                            resolution_factor=0.45, extend=extend, 
                                            dlat = 0.2)
        #Grid dimensions
        K = alts_grid.shape[0] #Number of vertival layers
        I = grid.shape[0] #Number of cells in eta direction, north-south, W dimension
        J = grid.shape[1]  #Number of cells in xi direction, east-west, L dimension
        KIJ = K*I*J
        IJ = I*J

        # Put into class object
        self.extend = extend
        self.alts_grid = alts_grid
        self.altres = altres
        self.grid = grid        # For volumetric reconstruction
        self.grid_l = grid_l    # For convection fitting with Lompe
        self.K = K
        self.I = I
        self.J = J
        self.KIJ = KIJ
        self.IJ = IJ            
        
        
    def make_csgrid(self, xg, maph = 200, h0 = 90, crop_factor = 0.6, resolution_factor = 0.5, 
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
                    -orientation[0]), L, W, Lres, Wres, R = (self.RE+h0)*1e3)
        
        # Extended grid for Lompe fit of E-filed   
        xi_e  = np.hstack((grid.xi_mesh[0,0]-np.flip([i*grid.dxi for i in range(1,extend+1)]),
                        grid.xi_mesh[0], grid.xi_mesh[0,-1] + np.array([i*grid.dxi 
                            for i in range(1,extend+1)]) ))
        eta_e = np.hstack((grid.eta_mesh[0,0]-np.flip([i*grid.deta for i in range(1,extend+1)]),
                        grid.eta_mesh[:, 0], grid.eta_mesh[-1,   0] + np.array([i*grid.deta 
                            for i in range(1,extend+1)]) )) 
        
        # outer grid padded with extend number of extra frames, intended to use for
        # the Lompe fit to avoid boundary effects when evaluated iniside the inner grid
        grid_l = cubedsphere.CSgrid(cubedsphere.CSprojection(grid.projection.position,
                    grid.projection.orientation), grid.L + extend*2*grid.Lres, 
                    grid.W + extend*2*grid.Wres, grid.Lres, grid.Wres, 
                    edges = (xi_e, eta_e), R = (self.RE+maph)*1e3)

        return grid, grid_l
        