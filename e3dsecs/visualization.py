#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:51:24 2022

@author: jone

Functions to produce various types of plots in 3D secs analysis, and its validation
using GEMINI output.

"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import apexpy
import matplotlib
import polplot
from gemini3d.grid.convert import geomag2geog
from gemini3d.grid import convert
import imageio

try:
    from . import coordinates
except:
    import coordinates
RE = 6371.2 # Earth radius in km


def fixed_alt(data, shape = None, crange=(-20,20), cbartitle='Arb.', **kwargs):
    '''
    

    Parameters
    ----------
    data : tuple
        Each element is a dictionary containing what to plot in each panel.
    shape : tuple
        Shape of plot. Default is None, leading to a 1xN layout
    crange : tuple
        Size 2. Values used in color range. Units according to data values.
    cbartitle : str
        Title to go on the colorbar. Should indicate units.

    Returns
    -------
    None.

    '''
    if shape == None:
        shape = (1,len(data))
    ccc = 7
    fig = plt.figure(figsize = (0.5*ccc*shape[1],0.5*ccc*shape[0]))
    plotshape = (ccc*shape[0], ccc*shape[1]+1)
    
    #Colorbar
    cbarax = plt.subplot2grid(plotshape, (0, plotshape[1]-1), rowspan = ccc, colspan = 1)
    cmap = mpl.cm.seismic
    norm = mpl.colors.Normalize(vmin=crange[0], vmax=crange[1])
    cb1 = mpl.colorbar.ColorbarBase(cbarax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label(cbartitle)
 
    row_i = -1
    col_i = -1
    # plt.figure(figsize=figsize)
    for i in range(len(data)):
        if (i % shape[1] == 0) & (shape[1]>1):
            col_i = col_i + 1
            row_i = 0
        elif (i % shape[1] == 0) & (shape[1]==1):
            col_i = 0
            row_i = row_i + 1
        else:
            col_i = col_i + 1
        ax = plt.subplot2grid(plotshape, (ccc*row_i, ccc*col_i), rowspan = ccc, colspan = ccc)
        ddd = data[i]
        # dxi = np.diff(ddd['xi'][0,:])[0]
        # deta = np.diff(ddd['eta'][:,0])[0]
        
        # plt.subplot(shape[0],shape[1],i+1)
        ax.set_axis_off()
        # plt.axis('off')
        ax.pcolormesh(ddd['xi'], ddd['eta'], ddd['values'], cmap='seismic', 
                      vmin=crange[0], vmax=crange[1], shading='nearest')
        c1 = ax.contour(ddd['xi'], ddd['eta'], ddd['glat'], levels=[64,66,68,70], 
                         colors='grey', linewidths=0.5, **kwargs)
        ax.clabel(c1, inline=1, fontsize=10, fmt = '%1.0f$^\circ$')
        c2 = ax.contour(ddd['xi'], ddd['eta'], ddd['glon'], levels=[15,20,25,30], 
                         colors='grey', linewidths=0.5, **kwargs)
        ax.clabel(c2, inline=1, fontsize=10, fmt = '%1.0f$^\circ$')
        ax.set_title(ddd['title'])
        ax.set_xlim(ddd['xirange'][0], ddd['xirange'][1])
        ax.set_ylim(ddd['etarange'][0], ddd['etarange'][1])
        
        if 'plotgrid' in ddd.keys():
            for xi, eta in get_grid_boundaries(ddd['plotgrid'].xi_mesh, 
                        ddd['plotgrid'].eta_mesh, ddd['plotgrid'].NL, 
                        ddd['plotgrid'].NW):
                ax.plot(xi, eta, color = 'grey', linewidth = .4)
    
    plt.savefig(data[0]['filename'])
    


def altitude_profile(m, K, I, J, alts_grid, i = 6, j = 6):
    use_i = np.ones(K).astype(int)*i
    use_j = np.ones(K).astype(int)*j
    use_k = np.arange(K).astype(int)
    kijs = np.ravel_multi_index((use_k, use_i, use_j), (K,I,J))
    kijs_df = kijs + K*I*J
    plt.plot(m[kijs],alts_grid, label='CF')
    plt.plot(m[kijs_df],alts_grid, label='DF')
    plt.xlabel('SECS amplitude [A/m]')
    plt.ylabel('Altitude [km]')
    plt.title('i='+str(i)+', j='+str(j))
    plt.legend()    
    
def get_grid_boundaries(lon, lat, NL, NW):
    """ 
    Get grid boundaries for plotting 
        
    Yields tuples of (lon, lat) arrays that outline
    the grid cell boundaries. 

    Example:
    --------
    for c in obj.get_grid_boundaries():
        lon, lat = c
        plot(lon, lat, 'k-', transform = ccrs.Geocentric())
    """
    x, y = lon, lat

    for i in range(NL + NW + 2):
        if i < NL + 1:
            yield (x[i, :], y[i, :])
        else:
            i = i - NL - 1
            yield (x[:, i], y[:, i])
            

# def get_coastlines(**kwargs):
#     """ generate coastlines in projected coordinates """

#     if 'resolution' not in kwargs.keys():
#         kwargs['resolution'] = '50m'
#     if 'category' not in kwargs.keys():
#         kwargs['category'] = 'physical'
#     if 'name' not in kwargs.keys():
#         kwargs['name'] = 'coastline'

#     shpfilename = shpreader.natural_earth(**kwargs)
#     reader = shpreader.Reader(shpfilename)
#     coastlines = reader.records()
#     multilinestrings = []
#     for coastline in coastlines:
#         if coastline.geometry.geom_type == 'MultiLineString':
#             multilinestrings.append(coastline.geometry)
#             print(coastline.geometry)
#             continue
#         lon, lat = np.array(coastline.geometry.coords[:]).T 
#         # print(lat)
#         yield (lon, lat)

#     for mls in multilinestrings:
#         for ls in mls:
#             lon, lat = np.array(ls.coords[:]).T 
#             yield (lon, lat)    


    # # plt.plot(glon_secs[extend:-extend,extend], glat_secs[extend:-extend,extend], color='black')
    # # plt.plot(glon_secs[extend:-extend,-extend-1], glat_secs[extend:-extend,-extend], color='black')
    # # plt.plot(glon_secs[extend,extend:-extend], glat_secs[extend,extend:-extend], color='black')
    # # plt.plot(glon_secs[-extend-1,extend:-extend], glat_secs[-extend,extend:-extend], color='black')



def plot_field_aligned_segment(ax, mlon, mlat, alts_grid, color='green', **kwargs):
    apex = apexpy.Apex(2022)
    xs = []
    ys = []
    zs = []
    # alts = np.linspace(0,500, 20)
    for alt in alts_grid:
        glat_, glon_, e = apex.apex2geo(mlat, mlon, alt)
        x,y,z = coordinates.sph_to_car((RE+alt, 90-glat_, glon_), deg=True)
        xs.append(x[0])
        ys.append(y[0])
        zs.append(z[0])
    ax.plot(xs,ys,zs, color=color, **kwargs)    

def plot_hor_segment(ax, mlons, mlats, alt, color='green', dipoleB=False, **kwargs):
    # Not implemented realistic field geometry. Hence, dipoleB is a dummy keyword
    if dipoleB:
        glon_, glat_ = geomag2geog(np.radians(mlons), np.pi/2 - np.radians(mlats)) # returns in degrees
    else:
        glon_, glat_ = geomag2geog(np.radians(mlons), np.pi/2 - np.radians(mlats)) # returns in degrees
        # apex = apexpy.Apex(2022)
        # glat_, glon_, e = apex.apex2geo(mlats, mlons, alt)
    x,y,z = coordinates.sph_to_car((RE+alt, 90-glat_, glon_), deg=True)
    ax.plot(x,y,z, color=color, **kwargs)


def field_aligned_grid(ax, grid, alts_grid, color='green', showlayers=False, 
                       showbase=True, fullbox=False, verticalcorners=False, 
                       dipoleB=False, coastlines=True, **kwargs):
    '''
    Make 3D plot of volume spanned by CS grid following a field line from its
    central location
    
    Parameters
    ----------
    ax : matplotlib 3D axis object
        To plot on
    grid : CS grid object
        The CS grid ato top or bottom boundary to extend along field line.
    alts_grid : 1-D array type
        Center location of altitude layers. In km.

    Returns
    -------
    None.

    '''
    # Plot grid and coastlines:
    # fig = plt.figure(figsize = (10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    #ax.set_axis_off()
    
    #Calculate ecef grid boundaries
    apex = apexpy.Apex(2022)
    L = grid.L*1e-3
    Lres = grid.Lres*1e-3
    pos = grid.projection.position
    if dipoleB:
        lon_, lat_ = geomag2geog(np.radians(pos[0]), np.pi/2 - np.radians(pos[1])) # returns in degrees
    else:
        lat_ = pos[1] # in degrees
        lon_ = pos[0] # in degrees
    # site_mlat, site_mlon = apex.geo2apex(lat_, lon_, 0*0.001)
    x_, y_, z_ = coordinates.sph_to_car((RE, 90-lat_, lon_), deg=True)
    xlim = (x_[0]-L-10*Lres, x_[0]+L+10*Lres) 
    ylim = (y_[0]-L-10*Lres, y_[0]+L+10*Lres) 
    zlim = (RE, RE+alts_grid[-1]+1)
    zlim = (z_[0], z_[0]+ alts_grid[-1])
    if coastlines:
        #Plot coastlines in ecef frame
        datapath=polplot.__path__[0]+'/data/'
        resolution = '50m' #or '110m'
        _coastlines = np.load(datapath + 'coastlines_' + resolution + '.npz')
        for cl in _coastlines:
            lat, lon = _coastlines[cl]
            x,y,z = coordinates.sph_to_car((RE, 90-lat, lon), deg=True)
            use = (x > xlim[0]-L/2) & (x < xlim[1]+L/2) & (y > ylim[0]-L/2) & (y < ylim[1]+L/2) & (z > 0)
            ax.plot(x[use], y[use], z[use], color = 'C0', **kwargs)
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_zlim(5600,6400)
    ax.set_zlim(zlim)
    ######
    
    #Plot field-aligned layers of the SECS grid
    if dipoleB:
        mlons0 = grid.lon_mesh 
        mlats0 = grid.lat_mesh
    else:
        mlons0_, mtheta_ = convert.geog2geomag(grid.lon_mesh,grid.lat_mesh)
        mlons0 = np.degrees(mlons0_)
        mlats0 = 90 - np.degrees(mtheta_)
        # mlats0, mlons0 = apex.geo2apex(grid.lat_mesh, grid.lon_mesh, alts_grid[0]) #-1?
    # glats = grid.lat_mesh[np.newaxis]
    # glons = grid.lon_mesh[np.newaxis]
    # mlats = mlats0[np.newaxis]
    # mlons = mlons0[np.newaxis]
    if showbase:
        # if dipoleB:
        glon_, glat_ = geomag2geog(np.radians(mlons0), np.pi/2 - np.radians(mlats0)) # returns in degrees
        # else:
            # glat_,glon_, _ = apex.apex2geo(mlats0, mlons0, alts_grid[0])
        for lon, lat in get_grid_boundaries(glon_, glat_, grid.NL, grid.NW):
            x,y,z = coordinates.sph_to_car((RE+alts_grid[0], 90-lat, lon), deg=True)
            ax.plot(x, y, z, color = 'grey', linewidth = .4)   
    
    if showlayers:
        for alt in alts_grid[::1]:
            if dipoleB:
                print('Show layers not implemented for dipole mode')
                print(1/0)
            glat_ = grid.lat_mesh
            glon_ = grid.lon_mesh
            # glat_,glon_, _ = apex.apex2geo(mlats0, mlons0, alt)
            # glats = np.vstack((glats, glat_[np.newaxis]))
            # glons = np.vstack((glons, glon_[np.newaxis]))
            # mlat_, mlon_ = apex.geo2apex(glat_, glon_, alt)
            # mlats = np.vstack((mlats, mlat_[np.newaxis]))
            # mlons = np.vstack((mlons, mlon_[np.newaxis]))
            counter = 0
            for lon, lat in get_grid_boundaries(glon_, glat_, grid.NL, grid.NW):
                if (counter==0) or (counter==grid.NL) or (counter==grid.NL+1) or (counter==grid.NL+grid.NW+1):
                    x1,y1,z1 = coordinates.sph_to_car((RE+alt, 90-lat[0], lon[0]), deg=True)
                    x2,y2,z2 = coordinates.sph_to_car((RE+alt, 90-lat[-1], lon[-1]), deg=True)
                    ax.plot([x1, x2], [y1, y2], [z1, z2], color='grey', linewidth=0.4)                
                counter = counter + 1
                    # ax.plot(x, y, z, color = 'grey', linewidth = .4)
        

    
    #Horizontal boundary
    plot_hor_segment(ax, mlons0[0,:], mlats0[0,:], alts_grid[0], color=color, dipoleB=dipoleB, **kwargs)
    plot_hor_segment(ax, mlons0[-1,:], mlats0[-1,:], alts_grid[0], color=color, dipoleB=dipoleB, **kwargs)
    plot_hor_segment(ax, mlons0[:,0], mlats0[:,0], alts_grid[0], color=color, dipoleB=dipoleB, **kwargs)
    plot_hor_segment(ax, mlons0[:,-1], mlats0[:,-1], alts_grid[0], color=color, dipoleB=dipoleB, **kwargs)
    
    if fullbox:
        #Horizontal boundary
        plot_hor_segment(ax, mlons0[0,:], mlats0[0,:], alts_grid[-1], color=color, **kwargs, dipoleB=dipoleB)
        plot_hor_segment(ax, mlons0[-1,:], mlats0[-1,:], alts_grid[-1], color=color, dipoleB=dipoleB, **kwargs)
        plot_hor_segment(ax, mlons0[:,0], mlats0[:,0], alts_grid[-1], color=color, dipoleB=dipoleB, **kwargs)
        plot_hor_segment(ax, mlons0[:,-1], mlats0[:,-1], alts_grid[-1], color=color, dipoleB=dipoleB, **kwargs)
    
        #Field-aligned boundary
        plot_field_aligned_segment(ax, mlons0[0,0], mlats0[0,0], alts_grid, color=color)
        plot_field_aligned_segment(ax, mlons0[0,-1], mlats0[0,-1], alts_grid, color=color)
        plot_field_aligned_segment(ax, mlons0[-1,0], mlats0[-1,0], alts_grid, color=color)
        plot_field_aligned_segment(ax, mlons0[-1,-1], mlats0[-1,-1], alts_grid, color=color)
    
    if verticalcorners:
        #alts_grid[0] should be the altitude of the base CS grid
        x0,y0,z0 = coordinates.sph_to_car((RE+alts_grid[0], 90-grid.lat_mesh[0,0], grid.lon_mesh[0,0]), deg=True)
        x1,y1,z1 = coordinates.sph_to_car((RE+alts_grid[-1], 90-grid.lat_mesh[0,0], grid.lon_mesh[0,0]), deg=True)
        ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color='black', linestyle='dotted')
        
        x0,y0,z0 = coordinates.sph_to_car((RE+alts_grid[0], 90-grid.lat_mesh[-1,0], grid.lon_mesh[-1,0]), deg=True)
        x1,y1,z1 = coordinates.sph_to_car((RE+alts_grid[-1], 90-grid.lat_mesh[-1,0], grid.lon_mesh[-1,0]), deg=True)
        ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color='black', linestyle='dotted')

        x0,y0,z0 = coordinates.sph_to_car((RE+alts_grid[0], 90-grid.lat_mesh[0,-1], grid.lon_mesh[0,-1]), deg=True)
        x1,y1,z1 = coordinates.sph_to_car((RE+alts_grid[-1], 90-grid.lat_mesh[0,-1], grid.lon_mesh[0,-1]), deg=True)
        ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color='black', linestyle='dotted')

        x0,y0,z0 = coordinates.sph_to_car((RE+alts_grid[0], 90-grid.lat_mesh[-1,-1], grid.lon_mesh[-1,-1]), deg=True)
        x1,y1,z1 = coordinates.sph_to_car((RE+alts_grid[-1], 90-grid.lat_mesh[-1,-1], grid.lon_mesh[-1,-1]), deg=True)
        ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color='black', linestyle='dotted')
    
    # Plot vertical height scale
    alts = np.array([0,100,200,300,400,500])
    N = len(alts)
    lats = np.array([grid.lat_mesh[-1,0]]*N)
    lons = np.array([grid.lon_mesh[-1,0]]*N)
    L = grid.L*1e-3
    Lres = grid.Lres*1e-3
    pos = grid.projection.position
    x_, y_, z_ = coordinates.sph_to_car((RE+alts, 90-lats, lons), deg=True)
    ax.plot(x_, y_, z_, color='black', zorder=100)
    for (ii,aa) in enumerate(alts):
        ax.text(x_[ii], y_[ii], z_[ii], str(aa)+' km', ha='right', zorder=100)#, clip_on=True)
    
    return ax

def spherical_grid(ax, lat_ev, lon_ev, alt_ev, color='red', maph=300):
    # lat_ev and lon_ev must be in geographic coords
    
    # Vertical lines
    x,y,z = coordinates.sph_to_car((RE+alt_ev[:,0,0], 90-lat_ev[:,0,0], lon_ev[:,0,0]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[:,-1,0], 90-lat_ev[:,-1,0], lon_ev[:,-1,0]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[:,0,-1], 90-lat_ev[:,0,-1], lon_ev[:,0,-1]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[:,-1,-1], 90-lat_ev[:,-1,-1], lon_ev[:,-1,-1]), deg=True)
    ax.plot(x,y,z, color=color)

    #Horizontal lines
    x,y,z = coordinates.sph_to_car((RE+alt_ev[0,:,0], 90-lat_ev[0,:,0], lon_ev[0,:,0]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[0,:,-1], 90-lat_ev[0,:,-1], lon_ev[0,:,-1]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[0,0,:], 90-lat_ev[0,0,:], lon_ev[0,0,:]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[0,-1,:], 90-lat_ev[0,-1,:], lon_ev[0,-1,:]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[-1,:,0], 90-lat_ev[-1,:,0], lon_ev[-1,:,0]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[-1,:,-1], 90-lat_ev[-1,:,-1], lon_ev[-1,:,-1]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[-1,0,:], 90-lat_ev[-1,0,:], lon_ev[-1,0,:]), deg=True)
    ax.plot(x,y,z, color=color)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[-1,-1,:], 90-lat_ev[-1,-1,:], lon_ev[-1,-1,:]), deg=True)
    ax.plot(x,y,z, color=color)
    
    # Horizontal layer where Lompe is used to represent E-field, using observations above this height
    closest_k = np.argmin(np.abs(alt_ev[:,0,0]-maph))
    x,y,z = coordinates.sph_to_car((RE+alt_ev[closest_k,0,:], 90-lat_ev[closest_k,0,:], lon_ev[closest_k,0,:]), deg=True)
    ax.plot(x,y,z, color=color, alpha=0.5)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[closest_k,-1,:], 90-lat_ev[closest_k,-1,:], lon_ev[closest_k,-1,:]), deg=True)
    ax.plot(x,y,z, color=color, alpha=0.5)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[closest_k,:,0], 90-lat_ev[closest_k,:,0], lon_ev[closest_k,:,0]), deg=True)
    ax.plot(x,y,z, color=color, alpha=0.5)
    x,y,z = coordinates.sph_to_car((RE+alt_ev[closest_k,:,-1], 90-lat_ev[closest_k,:,-1], lon_ev[closest_k,:,-1]), deg=True)
    ax.plot(x,y,z, color=color, alpha=0.5)

def plot_e3dfov(ax, lat0, lon0, alt0, lat1, lon1, alt1, color='C1', **kwargs):
    '''
    Plot FOV of EISCAT 3D, single site. Plots a line between the outer edges
    of the FOV at alt0 and alt1, and a vertical line connecting them.

    Parameters
    ----------
    ax : matplotlib 3D axis object
        To plot on
    lat0 : array-like
        Latitude of lower altitude boundary.
    lon0 : array-like
        Longitude of lower altitude boundary.
    alt0 : array-like
        Altitude of lower altitude boundary.
    lat1 : array-like
        Latitude of upper altitude boundary.
    lon1 : array-like
        Longitude of upper altitude boundary.
    alt1 : array-like
        Altitude of upper altitude boundary.
    color : string, optional
        Color of line to plot. The default is 'C1'.

    Returns
    -------
    None.

    '''
    NNN = lat0.size
    for i in range(NNN-1):
        #Horizontal segmetn, bottom
        x0,y0,z0 = coordinates.sph_to_car((RE+alt0[i], 90-lat0[i], lon0[i]), deg=True)
        x1,y1,z1 = coordinates.sph_to_car((RE+alt0[i+1], 90-lat0[i+1], lon0[i+1]), deg=True)
        ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color=color)
        
        #Horizontal segmetn, top
        x0,y0,z0 = coordinates.sph_to_car((RE+alt1[i], 90-lat1[i], lon1[i]), deg=True)
        x1,y1,z1 = coordinates.sph_to_car((RE+alt1[i+1], 90-lat1[i+1], lon1[i+1]), deg=True)
        ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color=color)

        #Vertical segment
        x0,y0,z0 = coordinates.sph_to_car((RE+alt0[i], 90-lat0[i], lon0[i]), deg=True)
        x1,y1,z1 = coordinates.sph_to_car((RE+alt1[i], 90-lat1[i], lon1[i]), deg=True)
        ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color=color, alpha=0.3)

    #Connecting last beam to fist beam
    #Horizontal segmetn, bottom
    x0,y0,z0 = coordinates.sph_to_car((RE+alt0[NNN-1], 90-lat0[NNN-1], lon0[NNN-1]), deg=True)
    x1,y1,z1 = coordinates.sph_to_car((RE+alt0[0], 90-lat0[0], lon0[0]), deg=True)
    ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color=color)

    #Horizontal segmetn, top
    x0,y0,z0 = coordinates.sph_to_car((RE+alt1[NNN-1], 90-lat1[NNN-1], lon1[NNN-1]), deg=True)
    x1,y1,z1 = coordinates.sph_to_car((RE+alt1[0], 90-lat1[0], lon1[0]), deg=True)
    ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color=color)
    
    #Vertical segment
    x0,y0,z0 = coordinates.sph_to_car((RE+alt0[NNN-1], 90-lat0[NNN-1], lon0[NNN-1]), deg=True)
    x1,y1,z1 = coordinates.sph_to_car((RE+alt1[NNN-1], 90-lat1[NNN-1], lon1[NNN-1]), deg=True)
    ax.plot([x0[0],x1[0]],[y0[0],y1[0]],[z0[0],z1[0]], color=color, alpha=0.3, **kwargs)

                

def plot_field_line(ax, glat0, glon0, alts_grid, color='grey', dipoleB=False, **kwargs):
    # here, inlut location must be geographic. dipoleB specified if one shoud
    # plot a dipole line or IGRF if false.
    if dipoleB: 
        from gemini3d.grid import convert
        mlon_, mtheta_ = convert.geog2geomag(glon0,glat0)
        m_theta = np.arcsin(np.sqrt((RE+alts_grid)/(RE+alts_grid[0]))*np.sin(mtheta_))
        m_mlon = np.ones(alts_grid.size)*mlon_
        m_glon, m_glat = convert.geomag2geog(m_mlon, m_theta)
        x,y,z = coordinates.sph_to_car((RE+alts_grid, 90-m_glat, m_glon), deg=True)
        ax.plot(x,y,z, color=color, alpha=0.5, **kwargs)

    else:
        apex = apexpy.Apex(2022)
        mlat0, mlon0 = apex.geo2apex(glat0, glon0, alts_grid[0])
        xs = []
        ys = []
        zs = []
        for alt in alts_grid:
            glat_, glon_, e = apex.apex2geo(mlat0, mlon0, alt)
            x,y,z = coordinates.sph_to_car((RE+alt, 90-glat_, glon_), deg=True)
            xs.append(x[0])
            ys.append(y[0])
            zs.append(z[0])
        ax.plot(xs,ys,zs, color=color, alpha=0.5, **kwargs)


def plot_resolution(ax, grid, alts_grid, kij, psf, az=-26, el=7, clim=1e-6, 
                   planes=[0,1], dipoleB=True, alpha=0.5, absolute=False, range_p=0.3):
    '''
    A problem here with the dipoleB keyword is that it is used for two different
    things: 1) To specify whether one should use a dipole model of B, the intended use
            2) To specify wheter the input coordinates is mag dipole or geo.
    This causes some confusion and inconsistensises that should be cleaned up.
    
    Parameters
    ----------
    ax : matplotlib axis object
        Axis to plot on
    grid : TYPE
        DESCRIPTION.
    alts_grid : TYPE
        DESCRIPTION.
    kij : TYPE
        DESCRIPTION.
    psf : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    #Grid dimensions
    K = alts_grid.shape[0] #Number of vertival layers
    I = grid.shape[0] #Number of cells in eta direction
    J = grid.shape[1]  #Number of cells in xi direction 
    k ,i, j = np.unravel_index(kij, (K,I,J))

    ax.view_init(azim=az, elev=el)
    field_aligned_grid(ax, grid, alts_grid, color='green', dipoleB=dipoleB)
    kwargs={'linewidth':3}

    if absolute:
        cmap = plt.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=0, vmax=clim)
        psf = np.abs(psf)
    else:
        cmap = plt.cm.seismic
        norm = matplotlib.colors.Normalize(vmin=-clim, vmax=clim)
        
    # for kk in range(lat_ev[0,-1,:].size):
    #     visualization.plot_field_line(ax, lat_ev[0,-1,kk], lon_ev[0,-1,kk], 
    #                               alts__, color='orange', **kwargs, dipole=True)
    #     visualization.plot_field_line(ax, lat_ev[0,sh[1]//2,kk], lon_ev[0,sh[1]//2,kk], 
    #                               alts__, color='orange', **kwargs, dipole=True)

    altres = np.diff(alts_grid)*0.5
    altres = np.abs(np.concatenate((np.array([altres[0]]),altres)))
    
    for pp in planes:
        if pp == 0:
            xis = grid.xi_mesh[0,:]# + grid.dxi*0.5
            etas = grid.eta_mesh[:,0]# + grid.deta*0.5
            alt_, etas_, xis_ = np.meshgrid(alts_grid+altres, etas, xis, indexing='ij')
            lon_, lat_ = grid.projection.cube2geo(xis_, etas_)
            if dipoleB:
                lon_, lat_ = geomag2geog(np.radians(lon_), np.pi/2 - np.radians(lat_))
            sh = lon_.shape
            x, y, z = coordinates.sph_to_car((RE+alt_.flatten(), 90-lat_.flatten(), 
                                  lon_.flatten()), deg=True)
            p = ax.plot_surface(x.reshape(sh)[k,:,:], y.reshape(sh)[k,:,:], 
                                z.reshape(sh)[k,:,:], alpha=alpha, zorder=1,
                                facecolors=cmap(norm(psf.reshape(sh[0],sh[1]-1,sh[2]-1)[k,:,:])), cmap=cmap)
    
        if pp == 1:
            xis = grid.xi_mesh[0,:]# + grid.dxi*0.5
            etas = grid.eta_mesh[:,0] + grid.deta*0.5
            alt_, etas_, xis_ = np.meshgrid(alts_grid, etas, xis, indexing='ij')
            lon_, lat_ = grid.projection.cube2geo(xis_, etas_)
            if dipoleB:
                lon_, lat_ = geomag2geog(np.radians(lon_), np.pi/2 - np.radians(lat_))
            sh = lon_.shape
            x, y, z = coordinates.sph_to_car((RE+alt_.flatten(), 90-lat_.flatten(), 
                                  lon_.flatten()), deg=True)        
            p = ax.plot_surface(x.reshape(sh)[:,i,:], y.reshape(sh)[:,i,:], 
                                z.reshape(sh)[:,i,:], alpha=alpha, zorder=3,
                                facecolors=cmap(norm(psf.reshape(sh[0],sh[1]-1,sh[2]-1)[:,i,:])), cmap=cmap)
    
        if pp == 2:
            xis = grid.xi_mesh[0,:] + grid.dxi*0.5
            etas = grid.eta_mesh[:,0]# + grid.deta*0.5
            alt_, etas_, xis_ = np.meshgrid(alts_grid, etas, xis, indexing='ij')
            lon_, lat_ = grid.projection.cube2geo(xis_, etas_)
            if dipoleB:
                lon_, lat_ = geomag2geog(np.radians(lon_), np.pi/2 - np.radians(lat_))
            sh = lon_.shape
            x, y, z = coordinates.sph_to_car((RE+alt_.flatten(), 90-lat_.flatten(), 
                                  lon_.flatten()), deg=True)
            p = ax.plot_surface(x.reshape(sh)[:,:,j], y.reshape(sh)[:,:,j], 
                            z.reshape(sh)[:,:,j], alpha=alpha, zorder=2,
                            facecolors=cmap(norm(psf.reshape(sh[0],sh[1]-1,sh[2]-1)[:,:,j])), cmap=cmap) 
        alt0 = alts_grid[k]+altres[k]
        lon0 = grid.lon[i,j]
        lat0 = grid.lat[i,j]
        if dipoleB:
            lon0, lat0 = geomag2geog(np.radians(lon0), np.pi/2 - np.radians(lat0))
        x0, y0, z0 = coordinates.sph_to_car((RE+alt0, 90-lat0, lon0), deg=True)
        ax.scatter(x0, y0, z0, s=50, marker='*', color='green')
        
        #Field lines
        for kk in range(lat_[0,-1,:].size):     
            plot_field_line(ax, lat_[0,-1,kk], lon_[0,-1,kk], 
                                      alts_grid, color='orange', **kwargs, dipoleB=True)
            
        # Fit limits of the plot
        if dipoleB:
            lon0, lat0 = geomag2geog(np.radians(grid.projection.position[0]), np.pi/2 - np.radians(grid.projection.position[1]))
        else:
            lat0 = grid.projection.position[1]
            lon0 = grid.projection.position[0]
        x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-lat0, lon0), deg=True)
        range_ =  alts_grid[-1]*range_p
        ax.set_xlim(x0-range_, x0+range_)
        ax.set_ylim(y0-range_, y0+range_)
        ax.set_zlim(z0, z0+2*range_)
        ax.set_title('PSF at k='+str(k)+', i='+str(i)+', j='+str(j))
        
        
def plot_slice(ax, grid, alts_grid, lat, lon, alt, data, clim = 5e-5, azim=-3, 
               elev=12, dipole_lompe=True, dim = 0, sliceindex = 0, parameter='No name', 
               maph=None, coastlines=True):
    '''
    Plot a slice on a 3D plot of a choosen quantity from a data-cube
    
    ax : matplotlib axis object
        axis to plot on
    grid : CS grid object
        the 2D grid at the bottom of the analysis region
    alts_grid : 1D array
        altitude grid
    lat : 3D array
        containing the latitude in degrees of the centre of the data-cube values
    lon : 3D array
        containing the longitude in degrees of the centre of the data-cube values
    alt : 3D array
        containing the altitude in km of the centre of the data-cube values
    data : 3D array
        containing the values to display at the specified locations
    azim (float): -20. Projection setting
    elev (float): 7. Projection setting.

    Returns
    -------
    None.

    '''
    shape = alt.shape
    alts__ = alt.reshape(shape)[:,5,0]
    
    cmap = plt.cm.bwr
    norm = matplotlib.colors.Normalize(vmin=-clim*1e6, vmax=clim*1e6)
    
    ax.set_axis_off()
    ax.view_init(azim=azim, elev=elev, roll=8)
    # spherical_grid(ax, lat, lon, alt, color='blue')
    field_aligned_grid(ax, grid, alts__, color='green', dipoleB=dipole_lompe, coastlines=coastlines)
    kwargs={'linewidth':3}
    for kk in range(lat[0,-1,:].size):
        plot_field_line(ax, lat[0,-1,kk], lon[0,-1,kk], 
                                  alts__, color='orange', **kwargs, dipoleB=True)
    x, y, z = coordinates.sph_to_car((RE+alt.flatten(), 90-lat.flatten(), 
                          lon.flatten()), deg=True)
    if dim == 0:
        p = ax.plot_surface(x.reshape(shape)[sliceindex,:,:], y.reshape(shape)[sliceindex,:,:], 
                            z.reshape(shape)[sliceindex,:,:], alpha=0.5,
                            facecolors=cmap(norm(data.reshape(shape)[sliceindex,:,:]*1e6)), 
                            rcount = np.max(shape), ccount = np.max(shape),cmap=cmap)
        if maph is not None:
            N = shape[1]
            intmin = np.argmin(np.abs(maph - alt[:,0,0]))
            x, y, z = coordinates.sph_to_car((RE+np.ones(N)*maph, 90-lat[intmin,:,sliceindex], 
                                  lon[intmin,:,sliceindex]), deg=True)
            ax.plot(x, y, z, color='black')
    elif dim == 1:
        p = ax.plot_surface(x.reshape(shape)[:,sliceindex,:], y.reshape(shape)[:,sliceindex,:], 
                            z.reshape(shape)[:,sliceindex,:], alpha=0.5,
                            facecolors=cmap(norm(data.reshape(shape)[:,sliceindex,:]*1e6)), 
                            rcount = np.max(shape[0:]), ccount = np.max(shape[0:]), cmap=cmap)
        if maph is not None:
            N = shape[2]
            intmin = np.argmin(np.abs(maph - alt[:,0,0]))
            x, y, z = coordinates.sph_to_car((RE+np.ones(N)*maph, 90-lat[intmin,sliceindex,:], 
                                  lon[intmin,sliceindex,:]), deg=True)
            ax.plot(x, y, z, color='black', linewidth=3, alpha=1)
    elif dim == 2:
        p = ax.plot_surface(x.reshape(shape)[:,:,sliceindex], y.reshape(shape)[:,:,sliceindex], 
                            z.reshape(shape)[:,:,sliceindex], alpha=0.5,
                            facecolors=cmap(norm(data.reshape(shape)[:,:,sliceindex]*1e6)), 
                            rcount = np.max(shape), ccount = np.max(shape),cmap=cmap)
        if maph is not None:
            N = shape[1]
            intmin = np.argmin(np.abs(maph - alt[:,0,0]))
            x, y, z = coordinates.sph_to_car((RE+np.ones(N)*maph, 90-lat[intmin,:,sliceindex], 
                                  lon[intmin,:, sliceindex]), deg=True)
            ax.plot(x, y, z, color='black', linewidth=3, alpha=1)        
    if dipole_lompe:
        lon_, lat_ = geomag2geog(np.radians(grid.projection.position[0]), np.pi/2 - np.radians(grid.projection.position[1])) # returns in degrees
        x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-lat_, lon_), deg=True)
    else:
        x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-grid.projection.position[1], grid.projection.position[0]), deg=True)
    # Fix viewing geometry
    lat_ = grid.projection.position[1] # in degrees
    lon_ = grid.projection.position[0] # in degrees
    L = grid.L*1e-3
    Lres = grid.Lres*1e-3
    x_, y_, z_ = coordinates.sph_to_car((RE, 90-lat_, lon_), deg=True)
    xlim = (x_[0]-L+5*Lres, x_[0]+L-3*Lres) 
    ylim = (y_[0]-L+6*Lres, y_[0]+L-3*Lres) 
    zlim = (z_[0]+ 0.15*alts_grid[-1], z_[0]+ 0.75*alts_grid[-1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)      
    # range_ =  alts__[-1]*0.3    
    # ax.set_xlim(x0-range_, x0+range_)
    # ax.set_ylim(y0-range_, y0+range_)
    # ax.set_zlim(z0, z0+2*range_)
    ax.set_title(parameter, fontsize=22)
        

def make_gif(files, filename=None, delete=True, duration = 10):
    #make gif from pre-made pngs
    #files is list of the files to use. File order must be alphabetical. 
    #If filename not specified, it is taken from the input files
    files.sort() # To make sure make sure the order gets right
    frames = []
    for f in files:
        # print(filename)
        frames.append(imageio.imread(f))
    if filename is None:
        filename=files[0]+'.gif'
    imageio.mimsave(filename, frames, 'GIF', duration = duration*1000)
    if delete:
        import os
        for f in files:
            os.remove(f)


def plotslice(ax, meshgrid, q, cut='k', ind=0, clim=1e-5, cmap='bwr', diverging=True):
    '''
    Plot a slice of a quantity q through the provided 3D mesh 

    Parameters
    ----------
    ax : matplotlib axis object
        To plot on.
    meshgrid : tuple
        each element is a 3D grid array, KIJ style. First element is latitudes, in degrees.
        Next element is longitudes in degrees, last element is altitudes in km.
    q : 3D array
        Quantity to plot on the same grid as meshgrid.
    cut : str
        indicating which dimension to make the cut
    ind : int
        index along cut dimension to do the slice

    Returns
    -------
    the plotting axis object.

    '''       
    if diverging:
        norm = matplotlib.colors.Normalize(vmin=-clim, vmax=clim)
    else:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=clim)
    
    cmap = matplotlib.pyplot.get_cmap(name=cmap)
    ax.set_axis_off()
#    ax.view_init(azim=-26, elev=7)
    ax.view_init(azim=-3, elev=12, roll=8)

    # visualization.spherical_grid(ax, lat_ev, lon_ev, alt_ev, color='blue')
    # visualization.field_aligned_grid(ax, grid, alts_grid, color='green', dipoleB=dipolekw)

    alt_ev = meshgrid[0]
    lat_ev = meshgrid[1]
    lon_ev = meshgrid[2]
    shape = lat_ev.shape
        
    x, y, z = coordinates.sph_to_car((RE+alt_ev.flatten(), 90-lat_ev.flatten(), lon_ev.flatten()), deg=True)
    if cut=='k':
        p = ax.plot_surface(x.reshape(shape)[ind,:,:], y.reshape(shape)[ind,:,:], 
                            z.reshape(shape)[ind,:,:], alpha=0.5,
                            facecolors=cmap(norm(q[ind,:,:])), linewidth=0.1, 
                            rcount = np.max(shape), ccount = np.max(shape),cmap=cmap)
    elif cut=='i':
        p = ax.plot_surface(x.reshape(shape)[:,ind,:], y.reshape(shape)[:,ind,:], 
                            z.reshape(shape)[:,ind,:], alpha=0.5,
                            facecolors=cmap(norm(q[:,ind,:])), linewidth=0.1, 
                            rcount = np.max(shape[0:]), ccount = np.max(shape[0:]), cmap=cmap)
    else:
        p = ax.plot_surface(x.reshape(shape)[:,:,ind], y.reshape(shape)[:,:,ind], 
                            z.reshape(shape)[:,:,ind], alpha=0.5,
                            facecolors=cmap(norm(q[:,:,ind])), linewidth=0.1, 
                            rcount = np.max(shape), ccount = np.max(shape),cmap=cmap)    

    x0, y0, z0 = coordinates.sph_to_car((RE+0, 90-lat_ev[0,shape[1]//2,shape[2]//2], lon_ev[0,shape[1]//2,shape[2]//2]), deg=True)

    range_ =  alt_ev[-1,0,0]*0.32
    ax.set_xlim(x0[0]-range_, x0[0]+range_)
    ax.set_ylim(y0[0]-range_, y0[0]+range_)
    ax.set_zlim(z0[0], z0[0]+2*range_) 

    return ax            