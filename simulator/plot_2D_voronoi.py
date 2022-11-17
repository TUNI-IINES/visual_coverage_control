import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class DrawVoronoi():
    def __init__(self, ax, roi_vertices, region_num, color_idx = None):
        self.__ax = ax
        self.__roi_vertices = roi_vertices.copy()
        self.__region_num = region_num

        # Set color index --> in particular for multiple voronoi
        self.__color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if color_idx is None:
            self.__color_idx = [i for i in range(self.__region_num)] # default color index
        else: self.__color_idx = color_idx

        # Plotting and logging variable
        self.__vorline_plot = []
        self.__density_plot = None
        self.__sensing_plot = [None]*self.__region_num
        # Specific for visual coverage
        self.__triangular_plot = []


    def plot_voronoi_diagram(self, vor_vertices):
        # returns a pyplot of given voronoi data
        for l in self.__vorline_plot: # Clear previous drawn line 
            l.pop(0).remove()
            # l.pop.remove()
        self.__vorline_plot = []

        for i in range(self.__region_num):
            if (vor_vertices[i] is not None):# and (i==3):
                l_handler = self.__ax.plot(vor_vertices[i][:, 0], vor_vertices[i][:, 1], 'k-')
                self.__vorline_plot.append(l_handler)

        if len(self.__vorline_plot) == 0: # no voronoi data, draw the boundary
            l_handler = self.__ax.plot(self.__roi_vertices[:, 0], self.__roi_vertices[:, 1], 'k-')
            self.__vorline_plot.append(l_handler)

    def plot_density_function(self, datapoints, density_val):
        # Draw the density 
        if self.__density_plot is not None:
            self.__density_plot.set_array(density_val)            
        else:
            self.__density_plot = self.__ax.tripcolor(datapoints[:,0], datapoints[:,1], density_val,
                    vmin = 0, vmax = 1, shading='gouraud')

            axins1 = inset_axes(self.__ax, width="25%", height="2%", loc='lower right')
            plt.colorbar(self.__density_plot, cax=axins1, orientation='horizontal', ticks=[0, 0.5, 1])
            axins1.xaxis.set_ticks_position("top")
        

    def plot_density_specified_points(self, specified_points, corresponding_density):
        if self.__density_plot is not None:
            #self.pl_den.set_array(self.pDensity[self.mesh2TargetedPointIdx])
            self.__density_plot.set_array(corresponding_density)
        else:
            self.__density_plot = self.__ax.scatter(specified_points[:,0], specified_points[:,1], \
                c=corresponding_density, cmap='gray_r', vmin = 0, vmax = 1, marker='s', s = 8)

            #axins1 = inset_axes(ax, width="25%", height="2%", loc='lower right')
            #plt.colorbar(self.pl_den, cax=axins1, orientation='horizontal', ticks=[0, 0.5, 1])
            #axins1.xaxis.set_ticks_position("top")

    def plot_sensing_area(self, sensing_area_data):
        # Draw the sensing radius
        for i in range(self.__region_num):
            if self.__sensing_plot[i] is not None: # already drawn
                # self.sensingArea[i] is not None --> should be fulfilled, assuming normal behaviour
                self.__sensing_plot[i].set_data(*sensing_area_data[i].exterior.xy)
            else:
                if sensing_area_data[i] is not None: # first time drawing
                    self.__sensing_plot[i], = self.__ax.plot(*sensing_area_data[i].exterior.xy, 
                        color=self.__color_list[self.__color_idx[i]], linewidth=3)
                # else do nothing

    def plot_viscoverage_voronoi(self, points, triang, trisubgraph, cell_map):
        # Replacement for voronoi plotting for Visual Coverage Control
        for l in self.__triangular_plot:
            l.pop(0).remove()
        self.__triangular_plot = []

        for l in self.__vorline_plot:
            l.pop(0).remove()
        self.__vorline_plot = []

        # Plot the active triangular section
        for i in range(self.__region_num):
            for t in triang[i]:
                mid_ij = 0.5*(points[i] + points[ trisubgraph[i]['j'][t] ])
                mid_ik = 0.5*(points[i] + points[ trisubgraph[i]['k'][t] ])
                
                point_x = [ mid_ij[0], points[i][0], mid_ik[0] ]
                point_y = [ mid_ij[1], points[i][1], mid_ik[1] ]
                self.__triangular_plot.append(self.__ax.plot(point_x, point_y, 'k--'))
        
        # Plot the Voronoi cells
        for segment_list in cell_map.values():
            for edge, (A, U, tmin, tmax) in segment_list:
                len = 20 # TODO: better way to adjust len and fit with ROI
                if tmax is None: tmax = len
                if tmin is None: tmin = -len
                p1, p2 = A + tmin * U, A + tmax * U
                self.__vorline_plot.append(self.__ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-'))


