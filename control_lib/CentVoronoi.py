import numpy as np
from matplotlib import path
from scipy.spatial import Voronoi, distance
from scipy.stats import multivariate_normal
import shapely.geometry as geom


class CentVoronoi():
    def __init__(self, pos, roi_vert=None, sensingRadius = None, grid_size = 0.05):

        self._pos = pos[:,0:2] # Initial position (can be outside of ROI) --> only store the xy
        self._region_num = np.shape(pos)[0] # number of region to be generated when all is inside ROI

        # Set ROI + generate mesh
        self._grid_size = grid_size
        if roi_vert is not None:
            self._set_roi_vertices(roi_vert) # store vertices + generate bounding box
        else: self._roi_vert = None # Initialization for checking if it is not assigned yet
        
        # Set sensing radius
        if sensingRadius is None:
            try: largeSens = 3*max(self._roi_width, self._roi_height) 
            except: largeSens = 1000
            self._sens_rad = [largeSens]*self._region_num # By default set large value 2 times ROI max width
        else:
            assert len(sensingRadius) == self._region_num, "Mismatched number of points and sensing radius"    
            self._sens_rad = sensingRadius
        self._sensing_region = {i:None for i in range(len(self._pos))} # For plotting        


    # SUPPORTING FUNCTION to modify ROI and POINTS
    # ----------------------------------------------------------------------
    def _set_roi_vertices(self, roi_vert):
        #  Input must be 2 dimensional numpy with 2 column
        assert isinstance(roi_vert, np.ndarray) \
            and (roi_vert.ndim == 2) \
            and (roi_vert.shape[1] == 2), \
            f"set_roi_vertices. The ROI vertices must be 2 dimensional numpy with 2 column. Input {roi_vert}"
        # End row must be identical with first row (otherwise append it)
        if not np.array_equal(roi_vert[0], roi_vert[-1]):
            roi_vert = np.append(roi_vert, np.array([roi_vert[0]]), axis=0)
        # generate roi vertices
        self._roi_vert = roi_vert
        self._roi_poly = geom.Polygon(self._roi_vert)
        self._roi_mpl_poly = path.Path(self._roi_vert)
        self._roi_cent = np.resize(self._roi_poly.centroid.xy, (1,2))
        # generate bounding box from vertices
        self._roi_xmin, self._roi_ymin, self._roi_xmax, self._roi_ymax = self._roi_poly.bounds
        self._roi_width = self._roi_xmax - self._roi_xmin # x width
        self._roi_height = self._roi_ymax - self._roi_ymin # y width
        # self.roi_bbox = [self.roi_b_xmin, self.roi_b_ymin, self.roi_b_xmax, self.roi_b_ymax]
        # Generate mesh from the given ROI
        self._generate_mesh()

    def set_centroid(self, new_centroid): 
        self._roi_cent = new_centroid

    # MAIN COMPUTATION
    # ----------------------------------------------------------------------
    def _generate_mesh(self):
        # generate mesh for centroid computing and density recording
        m = int( self._roi_width//self._grid_size ) + 1
        n = int( self._roi_height//self._grid_size ) + 1
        gx, gy  = np.meshgrid(np.linspace(self._roi_xmin, self._roi_xmax, m), np.linspace(self._roi_ymin, self._roi_ymax, n))
        lenRow, lenCol = gx.shape
        self._grid_polygon = [] # Fill the grid only within ROI (to accomodate non-square ROI)
        for i in range(lenRow-1): 
            for j in range(lenCol-1):
                check_points = np.array([ [gx[i,j],    gy[i,j]], 
                                          [gx[i,j+1],  gy[i,j+1]], 
                                          [gx[i+1,j+1],gy[i+1,j+1]], 
                                          [gx[i+1,j],  gy[i+1,j]],
                                          [gx[i,j],    gy[i,j]] ])
                is_intersect = self._roi_mpl_poly.contains_points(check_points)
                if sum(is_intersect) > 0: # The grid intersects if there is any vertices of grid within ROI
                #if sum(is_intersect) == 5: # The grid intersects if ALL vertices of grid within ROI
                    self._grid_polygon += [ geom.Polygon(check_points) ] 
        # store the total number of grid, grid's centroid
        self._grid_num = len(self._grid_polygon)
        self._grid_points = np.empty([self._grid_num, 2])
        for i in range(self._grid_num): 
            self._grid_points[i] = np.array([self._grid_polygon[i].centroid.x, self._grid_polygon[i].centroid.y])

        # The distribution on the variables X, Y packed into pos.
        self._default_densVal = np.ones([self._grid_num])
        self._grid_densVal = np.ones([self._grid_num]) # Can be modified later


    # this function only called the first time in the beginning to register the initial density function
    # TODO: option to add base value
    # fix max value based on base value
    def set_density_from_points(self, points, sigma=[[0.25, 0], [0, 0.25]], mode='sum'):
        # store targetedPoints and mapping mesh for display in clean version
        self._points_of_importance = points
        self._poi_grid_idx = [distance.cdist([points[i]], self._grid_points).argmin() for i in range(len(points))]

        # Generate density function
        temp = np.zeros(self._grid_densVal.shape)
        for p in points:
            rv = multivariate_normal([p[0], p[1]], sigma)
            if mode == 'max': temp = np.maximum(temp, rv.pdf(self._grid_points))
            else: temp += rv.pdf(self._grid_points) # defaulting to sum of pdf
        # normalize to 0 and 1
        if mode == 'max': 
            # It should cancel the original gain of 1 / [ 2 pi sqrt(det(Sigma)) ]
            beta = 2. * np.pi * np.sqrt(np.linalg.det(sigma)) 
        else: beta = ( 1./np.max(temp) ) # normalize with maximum value
        temp *= beta # set maximum to 1
        if temp.any() > 1.0: assert False, f"There exist density function greater than 1: {temp}"

        # Initialize default value and time-varying value of density function
        self._default_densVal = np.copy(temp)
        self._grid_densVal = np.copy(temp) # Can be modified later


    # Generates a bounded voronoi diagram with finite regions
    def _update_bounded_voronoi(self):
        # Select points inside the bounding box to be computed for voronoi
        self._pos_in_roi = self._roi_mpl_poly.contains_points(self._pos)
        points_center = self._pos[self._pos_in_roi, :]

        vor = None
        if len(points_center) > 0: 
            # Add Large boundaries to make the region finite without affecting the voronoi within ROI
            offset_x = self._roi_width*10
            offset_y = self._roi_height*10
            vor_bounds_vertices = np.array([
                [ self._roi_xmin - offset_x, self._roi_ymin - offset_y ], 
                [ self._roi_xmax + offset_x, self._roi_ymin - offset_y ], 
                [ self._roi_xmin - offset_x, self._roi_ymax + offset_y ],
                [ self._roi_xmax + offset_x, self._roi_ymax + offset_y ], 
            ])
            vor_points = np.append(points_center, vor_bounds_vertices, axis=0)

            # Compute Voronoi
            vor = Voronoi(vor_points)
            #vor_centroid = self.points.copy() # List of points according to original index from points
            self._vor_vertices = {} # (to be filled) List of vertices according to original index from points
            # Intersect the computed voronoi with ROI
            idx_vor = 0
            for i in range(len(self._pos)):
                if self._pos_in_roi[i]:
                    poly = [vor.vertices[v] for v in vor.regions[vor.point_region[idx_vor]]]
                    i_cell = self._roi_poly.intersection(geom.Polygon(poly))
                    #self.vor_centroid[i] = i_cell.centroid.coords[0]
                    x, y = i_cell.exterior.xy
                    self._vor_vertices[i] = np.transpose(np.array([x, y]))
                    idx_vor += 1
                else:
                    #self.vor_centroid[i] = np.nan
                    self._vor_vertices[i] = None

        self._vor_obj = vor

    # Compute the center of mass
    def _compute_center_mass(self, vertices, i):
        # Only compute points within sensing region
        dx = self._grid_points[:,0] - self._pos[i][0]
        dy = self._grid_points[:,1] - self._pos[i][1]
        dist = np.hypot(dx, dy)
        is_within_sensing = dist < self._sens_rad[i]*1.1 # give some margin

        # Calculate center of mass. TODO: optimize by defining default circle. Here we only shift its value.
        theta = np.linspace(0, np.pi*2, 100) # divide circle into 100 vertices
        sensR_xy = np.stack([
            self._pos[i][0] + np.cos(theta)*self._sens_rad[i], 
            self._pos[i][1] + np.sin(theta)*self._sens_rad[i]
            ], 1)
        # intersect ROI with sensing radius
        mass_dist = geom.Polygon(vertices).intersection(geom.Polygon(sensR_xy)) 

        R = np.zeros(2) 
        M = 0
        for j in range(self._grid_num): 
            if is_within_sensing[j]:
                m = (self._grid_polygon[j].intersection(mass_dist).area / self._grid_polygon[j].area) * self._grid_densVal[j]
                M += m # sum mass (density) over region
                R += m * self._grid_points[j] # sum point*mass
        mass = M*self._grid_size*self._grid_size # multiply with the grid size
        cent = geom.Point(R / M)

        return np.array([[cent.x, cent.y]]), mass, mass_dist
        # TODO: add correction term --> with variable b (improve nominal control)
        # TODO: add gamma persistence level


    def update_voronoi_data(self, points): 
        self._pos = points[:,0:2]  # TODO: some checking condition
        # Compute the bounded voronoi tesselation
        self._update_bounded_voronoi() 

    # Calculate a centroidal voronoi diagram and compute nominal control
    def compute_nominal_control(self): # TODO: change the name
        # set default values
        nomControl = np.zeros(self._pos.shape)
        if self._roi_vert is not None: # point toward centre of ROI
            c = self._roi_cent
            centroids = np.repeat(c, self._region_num, axis=0)
        else: # point toward oneselves
            centroids = self._pos.copy()

        # update with voronoi computation & compute nominal control
        kOut = 1. # of sensing radius distance
        nomControl = np.zeros(self._pos.shape)
        self._sensing_region = {i:None for i in range(len(self._pos))} # For plotting

        # Compute nominal control
        for i in range(len(self._pos)):
            if (self._vor_obj is not None) and (self._vor_vertices[i] is not None): # if points is within ROI
                centroids[i,:], mass, self._sensing_region[i] = self._compute_center_mass(self._vor_vertices[i], i)
                gain = 2*mass
                distC = centroids[i] - self._pos[i]                

            else: # set limited gain based on distance so it is moving within sensing radius
                distC = centroids[i] - self._pos[i]
                gain = kOut*self._sens_rad[i]/np.hypot(distC[0], distC[1])
                self._sensing_region[i] = None

            nomControl[i] = gain*distC

        return np.array(centroids), nomControl


    # EXTRACT DATA to be used for plotting
    # When this does not work, that means some names are changed in CentVoronoi
    # ----------------------------------------------------------------------
    def extract_voronoi_vertices(self): 
        if self._vor_obj is not None:
            return self._vor_vertices
        else: return [None]*self._region_num

    def extract_mesh_density_data(self): 
        return self._grid_points, self._grid_densVal

    def extract_specified_density_data(self): 
        # Special function to show the density value near the specified point 
        # that was used to create the original density function
        return self._points_of_importance, self._grid_densVal[self._poi_grid_idx]

    def extract_sensing_data(self): return self._sensing_region



class CentVoronoiTVDensity(CentVoronoi):
    # Class for Time Varying Density 
    # The density value is decreased when the area is being sensed by the robot
    # and increasing again when the area is outside robot's sensing area
    def __init__(self, pos, roi_vert=None, sensingRadius = None, grid_size = 0.05):
        super().__init__(pos[:,:2], roi_vert, sensingRadius, grid_size) 
        # Set parameter for time-varying density, by default it is 0 (time-invariant)
        self.set_update_density_rate()

    def set_update_density_rate(self, sensing_rate = 0., decay_rate = 0.): 
        self._sense_rate = sensing_rate
        self._decay_rate = decay_rate

    def update_voronoi_data(self, points, dt = 0.02): 
        self._pos = points[:,0:2]  # TODO: some checking condition
        # Compute the bounded voronoi tesselation
        self._update_bounded_voronoi() 

        # update density function (time-varying) if either sensing rate and decay rate is set
        if (self._sense_rate != 0.0) or (self._decay_rate != 0.0):
            self._update_density_function(dt)


    def _update_density_function(self, dt): 
        # should be called after 
        if self._roi_vert is not None:
            # update density function
            sensed_idx = None
            for i in range(self._region_num):
                # get distance from each grid to oneself
                pos = self._pos[i]
                dx = self._grid_points[:,0] - pos[0]
                dy = self._grid_points[:,1] - pos[1]
                dist = np.hypot(dx, dy)
                # Only consider the grind is sensed if points is within ROI and within sensing radius
                i_sensed = dist < self._sens_rad[i] if self._pos_in_roi[i] else dist < 0.
                # mark all points in ROI that is within sensing range
                if sensed_idx is not None:
                    sensed_idx = np.logical_or(sensed_idx, (i_sensed) )
                else:
                    sensed_idx = i_sensed

            out_idx = np.logical_not(sensed_idx)
            self._grid_densVal[sensed_idx] -= dt * self._sense_rate * self._grid_densVal[sensed_idx] # within sensing range
            self._grid_densVal[out_idx] += dt * self._decay_rate * (1 - self._grid_densVal[out_idx]) # decaying information
            self._grid_densVal = np.clip(self._grid_densVal, 0, self._default_densVal) # Prohibit decaying above the set values

