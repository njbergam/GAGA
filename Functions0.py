## in this notebook we run dh/dt simulations
import pandas as pd     
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from IPython import embed
import triangle as tr
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import radius_neighbors_graph
from grave import grave
import networkx as nx
from scipy.spatial import Delaunay
import random
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
import random
from scipy.interpolate import LinearNDInterpolator
from matplotlib.animation import FuncAnimation
import math
from ALPS_functions import *
import matplotlib.patches as patches

### IMPORTANT BOUNDING BOXES
r = 0.5
jakobs = [-49.5-r, -49.5+r, 69.10-r, 69.10+r]
helheim = [-38.12-r,-38.12+r, 66.21-r,66.21+r,]
kanger = [-33-r,-33+r, 68.38-r,68.38+r]

# Given points in latitude/longitude,  
# return if they are within the geographical interior of Greenland
def greenland_filter(points):
    temp = []
    countries = gpd.read_file( gpd.datasets.get_path("naturalearth_lowres"))
    for i in range(len(points)):
        if countries['geometry'].contains(Point(points[i]))[22]:
            temp.append(points[i])
    return np.array(temp)

def greenland_filter_ixs(points):
    countries = gpd.read_file( gpd.datasets.get_path("naturalearth_lowres"))
    ixs = []
    for i in range(len(points)):
        if countries['geometry'].contains(Point(points[i]))[22]:
            ixs.append(i)
    return ixs


# run ALPS estimator on each point in the time series
# times and time series
def alps_dhdt(tss,times): 
    p=4
    q=2

    derivatives = []
    uncertainties = []
    preds = []
    
    for i in range(len(tss)):
        ts = tss[i]
        #embed()
        Data = np.vstack([times, ts]).T

        [n,lamb,sigmasq] = full_search_nk(Data,p,q)
        c = n+p
        U = Kno_pspline_opt(Data,p,n)
        B = Basis_Pspline(n,p,U,Data[:,0])
        P = Penalty_p(q,c)
        theta = np.linalg.solve(B.T.dot(B) + lamb*P, B.T.dot(Data[:,1].reshape(-1,1)))

        # only predicting at the given points
        xpred = Data[:,0]

        Bpred_dert = Basis_derv_Pspline(n,p,U,xpred)

        # derivative predictions at the given points
        ypred_derth = Bpred_dert.dot(theta)
        # uncertainty predictions at the given points
        std_th_derv,std_nh_derv = Var_bounds(Data,Bpred_dert,B,theta,P,lamb)

        derivatives.append(ypred_derth)
        uncertainties.append(std_th_derv)
        preds.append(B.dot(theta))

    return derivatives, uncertainties, preds


class Graph:

    def __init__(self, data, bbox=[], label = ''):

        self.data = data
        # global info: saving all the data, e.g. ICESat 2003-2008
        self.x = data['lon']
        self.y = data['lat']
        self.z = data['elev']
        self.t = data['time_day']

        self.bbox = bbox
        if bbox == []:
            self.bbox = [min(self.x),max(self.x),min(self.y),max(self.y)]

        self.label = ''



    def draw_bbox(self, pts=[]):

        fig, ax = plt.subplots(figsize=(6,8))
        countries = gpd.read_file( gpd.datasets.get_path("naturalearth_lowres"))
        gr_shp = countries[countries['name']=='Greenland']
        x,y = gr_shp['geometry'][22].exterior.xy
        plt.plot(x,y)
        # Define the coordinates of the rectangle (x, y, width, height)
        rectangle_coords = [self.bbox[0], self.bbox[2], self.bbox[1]-self.bbox[0],self.bbox[3]-self.bbox[2]]

        # Create the rectangle patch
        rect = patches.Rectangle((rectangle_coords[0], rectangle_coords[1]), rectangle_coords[2], rectangle_coords[3],
                                linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)
        ax.set_title(self.label)
        if len(pts) != 0:
            plt.scatter(pts[:,0],pts[:,1])

        plt.plot()

        

    def get_grid(self, resolution):
        test_x = np.linspace(self.bbox[0], self.bbox[1], resolution)
        test_y = np.linspace(self.bbox[2], self.bbox[3], resolution)
        pairs = []
        for x in test_x:
            for y in test_y:
                pairs.append([x,y])
        pairs = greenland_filter(pairs)
        xy = np.array(pairs)
        return xy


    ### IMPORTANT FUNCTIONALITY NEED TO MAKE THIS MORE 
    # downsample where there are fewer points
    # or just do it randomly for now
    def grid_reduce(self,grid):
        import random
        indices = random.sample(list(range(len(grid))), int(len(grid)/2))
        new_grid = np.take(grid, indices,axis=0)
        print(new_grid.shape)

        return new_grid


    # just use Kriging interpolation over each time step
    def make_grid_ts(self, grid_pts, time_step, type='nn'):
        
        time_series = []

        time_block = [2003, 2003+time_step]

        time_pts = []

        while time_block[1] < 2009:
            time_pts.append(time_block[0])

            data_spec = self.data[self.data['time_day'] > time_block[0]]
            data_spec = data_spec[data_spec['time_day'] < time_block[1]]

            ## reduce the 
            x = data_spec['lon']
            y = data_spec['lat']
            z = data_spec['elev']
            #plt.cla()
            #plt.scatter(x,y,c=z)
            #plt.show()

            if type == 'krige':
                # downsample for now so we can actually run this MF
                if len(x) > 5000:
                    import random
                    indices = random.sample(list(range(len(x))), 5000)
                    x = np.take(x, indices)
                    y = np.take(y, indices)
                    z = np.take(z, indices)

                print('krige with ' + str(len(x)) + ' points')
                
                OK = OrdinaryKriging(x,y,z,variogram_model="spherical")
                zpred, error = OK.execute("points", grid_pts[:,0], grid_pts[:,1])
            
            elif type == 'nn':
                from sklearn.neighbors import KNeighborsRegressor
                neigh = KNeighborsRegressor(n_neighbors=2)
                neigh.fit(np.vstack([x,y]).T, z)
                zpred = neigh.predict(grid_pts)

            time_series.append(zpred)

            time_block = [time_block[0] + time_step, time_block[1] + time_step ]
        
        time_series = np.array(time_series).T
        return time_series.tolist(), time_pts

    def krige_given(self, test_coords):
        x = self.x_spec
        y = self.y_spec
        z = self.z_spec

        if len(x) > 1000:
            indices = random.sample(list(range(len(x))), 1000)
            x = np.take(self.x_spec, indices)
            y = np.take(self.y_spec, indices)
            z = np.take(self.z_spec, indices)

        OK = OrdinaryKriging(x,y,z,variogram_model="spherical")
        zpred, error = OK.execute("points", test_coords[:,0], test_coords[:,1])
        return zpred, error