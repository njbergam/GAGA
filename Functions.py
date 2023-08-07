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

### IMPORTANT BOUNDING BOXES
r = 1
jakobs = [-49.5-r, -49.5+r, 69.10-r, 69.10+r]
helheim = [66.21-r,66.21+r, -38.12-r,-38.12+r]

# Given points in latitude/longitude,  
# return if they are within the geographical interior of Greenland
def greenland_filter(points):
    temp = []
    countries = gpd.read_file( gpd.datasets.get_path("naturalearth_lowres"))
    for i in range(len(points)):
        if countries['geometry'].contains(Point(points[i]))[22]:
            temp.append(points[i])
    return temp

# data structure to process/handle the graph instantiations of the Greenland data
class Graph:

    def __init__(self, data, time_window):
        self.data = data
        # global info: saving all the data, e.g. ICESat 2003-2008
        self.x = data['lon']
        self.y = data['lat']
        self.z = data['elev']
        self.t = data['time_day']

        # local info: the window we use for visualization, etc
        self.time_window = time_window
        data_spec = data[data['time_day'] >= self.time_window[0]]
        data_spec = data_spec[data_spec['time_day'] <= self.time_window[1]]
        self.spec_label = str(time_window)
        self.data_spec = data_spec
        # using the bounding box method, we can refine this to look closer at something
        self.x_spec = self.data_spec['lon']
        self.y_spec = self.data_spec['lat']
        self.z_spec = self.data_spec['elev']
        self.t_spec = self.data_spec['time_day']
        self.bbox = [min(self.x),max(self.x),min(self.y),max(self.y)]


    # output a directed graph on the specialized points

    def digraph(self, tamed=False, gran=100):
        x = np.array(self.x_spec)
        y = np.array(self.y_spec)
        z = np.array(self.z_spec)

        coordinates = np.column_stack([x,y])
        rad=0.05

        if not tamed:
            #A1 = radius_neighbors_graph(coordinates, rad, mode='connectivity', include_self=False).toarray()
            A = kneighbors_graph(coordinates, 1, mode='connectivity', include_self=False).toarray()
            #A = np.minimum(A1,A2)
            for i in tqdm(range(len(A))):
                for j in range(len(A[i])):
                    if A[i,j]!=0 and z[i]-z[j]>0:
                        A[i,j] = A[i,j]*(z[i]-z[j])
                    else:
                        A[i,j] = 0
            print(A)

            return nx.from_numpy_matrix(A, create_using = nx.DiGraph()), z, coordinates


        else:
            test_x = np.linspace(self.bbox[2], self.bbox[3], gran)
            test_y = np.linspace(self.bbox[0], self.bbox[1], gran)
            pairs = []
            for x in test_x:
                for y in test_y:
                    pairs.append([x,y])
            pairs = greenland_filter(pairs)
            xy = np.array(pairs)
            values, error = self.krige_given(xy)

            assert len(values) == len(xy)
            A = kneighbors_graph(xy, 4, mode='connectivity', include_self=False).toarray()
            #embed()
            mat = np.tile(values, (len(A),1) ) - np.tile(values, (len(A),1) ).T
            mat = mat<0
            A = A*mat.astype(int)

            #for i in range(len(A)):
            #    for j in range(len(A[i])):
            #        if A[i,j]!=0 and values[i]-values[j]>0:
            #            A[i,j] = A[i,j]*(values[i]-values[j])
            #        else: 
            #            A[i,j] = 0
                    #if A[i,j] == 0:
                    #    continue
                    #else:
                    #    A[i,j] = values[i]

            return nx.from_numpy_matrix(A, create_using = nx.DiGraph()), values, xy
        





    def info(self):
        print('number of total points:', len(self.data))
        print('number of points in ' + str(self.time_window) + ': ' +  str(len(self.data_spec)))
        print('bbox', self.bbox)

    # (lat_min, lon_min, lat_max, lon_max)
    def bbox_specialize(self,bbox, add_label=''):

        self.spec_label += add_label
        data_temp = self.data
        data_temp = data_temp[data_temp['lat'] >= bbox[0]]
        data_temp = data_temp[data_temp['lat'] <= bbox[1]]
        data_temp = data_temp[data_temp['lon'] >= bbox[2]]
        data_temp = data_temp[data_temp['lon'] <= bbox[3]]

        self.data_spec = data_temp
        self.x_spec = self.data_spec['lon']
        self.y_spec = self.data_spec['lat']
        self.z_spec = self.data_spec['elev']
        self.t_spec = self.data_spec['time_day']
        self.bbox = bbox

    def shift_time_window(self, shift):

        self.time_window = [self.time_window[0]+shift, self.time_window[1] +shift ] # new_window

        data_spec = self.data[self.data['time_day'] >= self.time_window[0]]
        data_spec = data_spec[data_spec['time_day'] <= self.time_window[1]]

        self.data_spec = data_spec
        self.x_spec = self.data_spec['lon']
        self.y_spec = self.data_spec['lat']
        self.z_spec = self.data_spec['elev']
        self.t_spec = self.data_spec['time_day']

        self.bbox_specialize(self.bbox)
        self.spec_label = str(self.time_window)

    # krige on the given test coordinates 
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

    def krige(self,toplot=False, downsample=False ,basis_pts=[], resolution=200,bbox=[]):
        # if bbox is empty we use the default spec

        x = self.x_spec
        y = self.y_spec
        z = self.z_spec

        print('number of points for kriging:', str(len(x)))
        if len(x) > 1000:
            downsample=True

        if downsample==True:
            indices = random.sample(list(range(len(x))), 1000)
            x = np.take(self.x_spec, indices)
            y = np.take(self.y_spec, indices)
            z = np.take(self.z_spec, indices)

        OK = OrdinaryKriging(
            x,y,z,
            variogram_model="spherical",
        )

        test_x = np.linspace(self.bbox[2], self.bbox[3], resolution)
        test_y = np.linspace(self.bbox[0], self.bbox[1], resolution)
        pairs = []
        for s in test_x:
            for t in test_y:
                pairs.append([s,t])

        pairs = greenland_filter(pairs)
        pairs=np.array(pairs)

        zpred, error = OK.execute("points", pairs[:,0], pairs[:,1])

        if toplot:
            plt.close()
            plt.scatter(pairs[:,0], pairs[:,1], c=zpred,s=0.2)
            plt.scatter(x,y,c=z)
            plt.title('kriging')
            plt.savefig('krige_test.png')
            plt.close()

        return pairs, zpred, error

    def krige_tamed(self, saveto, toplot=False, ):
        points, zpred, error = self.krige(resolution=50, downsample=False)
        tri = Delaunay(points)
        f1 = figure(figsize=(6,8))
        x = points[:,0]
        y = points[:,1]

        #plt.triplot(pairs[:,0], pairs[:,1], tri.simplices)

        # higher resolution for the linear interp
        test_x = np.linspace(min(x), max(x), 100)
        test_y = np.linspace(min(y), max(y), 100) 
        pairs = []
        shift = (test_x[1]-test_x[0])/2
        for x in test_x:
            for y in test_y:
                pairs.append([x,y])
        pairs = greenland_filter(pairs)
        pairs=np.array(pairs)
        # filtering out test points that are not inside the convex hull
        pairs = pairs[np.where(tri.find_simplex(pairs)!=-1)]
        interp = LinearNDInterpolator(points, zpred)
        p_values = interp(pairs[:,0], pairs[:,1])


        if toplot:
            pairs = pairs[np.where(p_values>250)]
            p_values = p_values[np.where(p_values>250)]
            
            sc = plt.scatter(pairs[:,0], pairs[:,1], c=p_values, s=10)
            plt.scatter(points[:,0], points[:,1], c=zpred, s=3)
            plt.colorbar(sc) 
            plt.title('Helheim Glacier: '+str(self.time_window))
            plt.savefig('samples/'+saveto+': ' +str(self.time_window) + '.png')
            
        # return the locations and their corresponding predicted value
        return pairs, p_values




    # three types: radius, k-nn, triangle, or normal
    def visualize(self,saveto, tp='radius'):
        x = np.array(self.x_spec)
        y = np.array(self.y_spec)
        z = np.array(self.z_spec)

        if tp == 'knn':
            coordinates = np.column_stack([x,y])
            G = nx.from_numpy_array(kneighbors_graph(coordinates, 12, mode='connectivity', include_self=False).toarray())
            f1 = figure(figsize=(6,8))
            positions = dict(zip(G.nodes, coordinates))
            nx.draw(G, pos=positions, node_size=.5, node_color="b")
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.plot()

        elif tp == 'trisurf':
            ax = plt.figure().add_subplot(projection='3d')
            ixs = np.where(z>200)[0]
            plt.title('Raw Altitude Map of Jakobs')
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            #plt.zlabel('latitude')
            ax.plot_trisurf(x[ixs], y[ixs], z[ixs], linewidth=0.2, antialiased=True)
            plt.show()

        elif tp == 'triangle':
            # downsample: otherwise it will take too long
            #indices = random.sample(list(range(len(x))), 00)
            indices = list(range(len(x)))
            x = np.take(x,indices)
            y = np.take(y,indices)
            points = np.column_stack([x,y])

            tri = Delaunay(points)
            f1 = figure(figsize=(6,8))
            plt.triplot(points[:,0], points[:,1], tri.simplices)
            plt.plot(points[:,0], points[:,1], 'o')
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.title('Greenland, Delaunay Triangulation, ' + str(self.time_window))

        
        elif tp == 'radius':
            coordinates = np.column_stack([x,y])
            G = nx.from_numpy_array(radius_neighbors_graph(coordinates, 0.009, mode='connectivity', include_self=False).toarray())
            f1 = figure(figsize=(6,8))
            positions = dict(zip(G.nodes, coordinates))
            nx.draw(G, pos=positions, node_size=.5, node_color="b")
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.plot()
            
        else: # type == 'normal':
            f1 = figure(figsize=(6,8))
            plt.title('Greenland, IceSAT Measurements, ' + str(self.time_window))
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            sc = plt.scatter(list(x),list(y),c=list(z))  
            plt.colorbar(sc) 
                 
        plt.savefig(saveto+".png")


    # find the crossover points in the image
    def crossovers(self, saveto='crossovers'):

        x = self.x_spec
        y = self.y_spec
        z = self.z_spec

        coord = np.column_stack([x,y])
        coordinates = coord
        rad = 0.05

        A1 = radius_neighbors_graph(coordinates, rad, mode='connectivity', include_self=False).toarray()
        A2 = kneighbors_graph(coordinates, 20, mode='connectivity', include_self=False).toarray()
        A = np.minimum(A1,A2)
        #A = radius_neighbors_graph(coordinates, rad, mode='distance', include_self=False).toarray()


        # argmax with conditions
        ixs = np.argpartition(sum(A), -20)[-20:] #np.where(sum(A)>max(sum(A))-3)[0]
        #print(sum(A))
        #print(ixs)

        pts = coordinates[ixs]
        plt.close()
        
        G=nx.from_numpy_array(A)
        positions = dict(zip(G.nodes, coordinates))

        figure, axes = plt.subplots()
        #nx.draw(G, pos=positions, node_size=.5, node_color="b")
        plt.scatter(x,y,c=z,s=4)

        plt.scatter(pts[:,0], pts[:,1],c='r')
        Drawing_colored_circle = plt.Circle(( -49.5 , 69 ), rad, alpha=0.2 )
        axes.add_artist( Drawing_colored_circle )
        #plt.scatter([-49.5], [69], s=)
        plt.savefig(saveto+'.png')
    
        return pts[:,0], pts[:,1]




def time_interp(z_s, num_times):
    #xnew = #[xy_s[0]]
    z_new = []
    for i in range(z_s.shape[1]):
        t_given = list(range(len(z_s[:,i])))
        t_eval = np.linspace(min(t_given), max(t_given), 100)
        z_new.append( np.interp(t_eval, t_given, z_s[:,i]) )
    return z_new

def alps_interpolate(title, bbox):
    gr = pd.read_pickle("greenland_clean.pkl") 
    # doing periods of half a year
    period = [2003,2004]
    gr = Graph(gr, period)
    gr.bbox_specialize(bbox,add_label='helheim')
    # going to do a specific location for now

    # PICK GRID POINTS ahead of time NOW
    test_x = np.linspace(gr.bbox[2], gr.bbox[3], 500)
    test_y = np.linspace(gr.bbox[0], gr.bbox[1], 500)
    pairs = []
    for x in test_x:
        for y in test_y:
            pairs.append([x,y])
    pairs = greenland_filter(pairs)
    xy = np.array(pairs)

    z_s = []    
    for i in range(5):
        #embed()
        

        ### 1) krige + delaunay
        
        # pairs, values = gr.krige_tamed(saveto='',toplot=False)

        ### 2) krige + alps + delaunay

        # first we get the values from kriging
        #pairs, values, error = gr.krige(resolution=50, downsample=False)
        
        values, error = gr.krige_given(xy)
        # then we run alps on those values

        z_s.append(list(values))
        #period = [period[0]+0.5,period[1]+0.5]
        gr.shift_time_window(0.5)

    # interpolating for a richer animation
    #embed()
    z_s = np.array(z_s)
    #embed()
    #z_s = np.array(time_interp(z_s, 100)).T
    


    animate(title, xy,z_s)


    # we would like to increase the granularity of the time steps, 
    # looking at 0.05 of a year (so around 10) between each half year mark

    # but let's look at things without alps interp for nows

def animate_grid(frames, saveto):
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'ro')
    p = frames

    def init():
        plt.imshow(p[0], cmap='hot', interpolation='nearest')
        return ln,

    def update(frame):
        plt.imshow(p[frame], cmap='hot', interpolation='nearest')
        return ln,
    print('animating!')
    ani = FuncAnimation(fig, update, frames=list(range(len(frames))), init_func=init, blit=True)
    ani.save(saveto, writer='pillow', fps=60)


def animate(title, xy, z_s):
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'ro')

    start = 2003
    end = 2009
    num_steps = len(z_s)

    def init():
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title(title + ", 2003")
        plt.scatter(xy[:,0], xy[:,1], c=z_s[0])
        return ln,

    def update(frame):
        if frame < len(z_s):
            plt.cla()
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.title(title + "," + str(  (end-start)*(frame+1)/num_steps + start   ))

            ixs = np.where(z_s[frame] > 300)
            #embed()
            show = xy[ixs]
            show_z = z_s[frame][ixs]

            plt.scatter(show[:,0], show[:,1], c=show_z)
        else:
            # stop at last one
            plt.cla()
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            plt.title(title + "," + str(  end   ))

            ixs = np.where(z_s[-1] > 300)
            #embed()
            show = xy[ixs]
            show_z = z_s[-1][ixs]

            plt.scatter(show[:,0], show[:,1], c=show_z)
        return ln,

    ani = FuncAnimation(fig, update, frames=list(range(len(z_s)+3)), init_func=init, interval=400,blit=True)
    ani.save(title+'.gif', writer='pillow')#, fps=10000)


#def coarsen_graph():



def main():
    gr = pd.read_pickle("greenland_clean.pkl") 
    period = [2003,2004]
    gr = Graph(gr, period)
    gr.bbox_specialize(jakobs,add_label='jakobs')
    gr.visualize(tp='trisurf', saveto='tests/jakobs_trisurf')

    
    #gr.visualize(tp='knn', saveto='tests/nn_full_greenland')
    #gr.visualize(tp='triangle', saveto='tests/triangle_greenland')
    #gr.visualize(tp='radius', saveto='tests/radius_full_greenland')
    #gr.visualize(tp='normal', saveto='tests/normal_greenland')
    #for i in range(5):
    #    gr.krige_tamed()
    #    gr.shift_time_window(1)
    ## Visualizing...
    # Greenland
    # Jakobshavn Glacier
    # Helheim Glacier

    ## Detecting Crossovers

    ## Interpolation
    # Kriging
    # Delaunay Triangulation
    # Krige-tamed Delaunay

#main()
#alps_interpolate('Whole 2003-2009', bbox=[66.21-10,66.21+10, -38.12-10,-38.12+10])#[-73.297, 60.03676, -12.20855, 83.64513])
