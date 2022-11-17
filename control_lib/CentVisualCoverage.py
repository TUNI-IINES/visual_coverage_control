import numpy as np
import cvxopt
import shapely.geometry as geom
from control_lib.CentVoronoi import CentVoronoi
from control_lib.laguerre_voronoi_2d import get_power_triangulation, get_voronoi_cells
from matplotlib.collections import LineCollection

# Replicated from the following Paper:
# [1] ICRA2019 - Visual Coverage Control for Teams of Quadcopters via Control Barrier Functions, Riku Funada et.al.
# [2] ICRA2020 - Visual Coverage Maintrnance for Quadcopters using Nonsmooth Barrier Functions, Riku Funada et.al.

class CentVisualCoverage(CentVoronoi):

    def __init__(self, state, roi_vert=None, sensingRadius = None, grid_size = 0.05, ):
        # Set ROI, generate Mesh, and set sensing radius
        super().__init__(state[:,:2], roi_vert, sensingRadius, grid_size) 

        # Store robot's altitude
        self.altitudes = state[:,2] 
        self.lambdas = state[:,3]

        # if TRUE, This is used to debug, by showing (most of) the grid in computation
        self.SHOW_COMPUTED_AREA = False 

        # Additional plotting variable
        self.active_triangCBF = [{i:None} for i in range(self._region_num)]
        self.l_triangHCBF = []
        self.l_triangVor = []

    def init_fov_parameters(self, rList=None, kappa=1., sigma=1., des_R=1., overlap_m = 0., overlap_m_ratio = None):
        # Store radius of image plane
        if rList is None:
            self.rList = 0.0034 * np.tan( np.deg2rad(50/2) ) * np.ones(self._region_num) 
        else:
            assert np.shape(rList)[0] ==  self._region_num, "Mismatched number of points and image plane radiuses" 
            self.rList = rList

        self.kappa = kappa
        self.sigma = sigma
        self.des_R = des_R

        self.overlap_m_ratio = overlap_m_ratio
        self.overlap_m = overlap_m

        self.is_sensing_grid = {}
        self.f_pers = {} # Perspective quality
        self.f_res = {} # Loss of resolution

        # derivative term for f_pers
        self.df_pers_dxi = {}
        self.df_pers_dyi = {}
        self.df_pers_dzi = {}
        self.df_pers_dli = {}
        # derivative term for f_res
        self.df_res_dxi = {}
        self.df_res_dyi = {}
        self.df_res_dzi = {}
        self.df_res_dli = {}


    def update_voronoi_data(self, state, dt = 0.02): 
        self.points = state[:,:2]  # TODO: some checking condition
        self.altitudes = state[:,2] 
        self.lambdas = state[:,3]

        # Select points inside the bounding box to be computed for voronoi
        self.idxInVor = self._roi_mpl_poly.contains_points(self.points)

        for i in range(self._region_num):
            self._sens_rad[i] = self.rList[i] * self.altitudes[i] / self.lambdas[i]


    # MAIN COMPUTATION
    # ----------------------------------------------------------------------
    def update_fov_function(self):

        for i in range(self._region_num):
            # Reset all value in the beginning
            self.f_pers[i] = np.zeros(self._grid_num)
            self.f_res[i] = np.zeros(self._grid_num)

            # derivative term for f_pers
            self.df_pers_dxi[i] = np.zeros(self._grid_num)
            self.df_pers_dyi[i] = np.zeros(self._grid_num)
            self.df_pers_dzi[i] = np.zeros(self._grid_num)
            self.df_pers_dli[i] = np.zeros(self._grid_num)
            # derivative term for f_res
            self.df_res_dxi[i] = np.zeros(self._grid_num)
            self.df_res_dyi[i] = np.zeros(self._grid_num)
            self.df_res_dzi[i] = np.zeros(self._grid_num)
            self.df_res_dli[i] = np.zeros(self._grid_num)

            # Compute f_pers, f_res, and the computed region for control input
            if self.idxInVor[i]: 
                # Only update points within sensing region
                dx = self._grid_points[:,0] - self.points[i][0]
                dy = self._grid_points[:,1] - self.points[i][1]
                dist = np.hypot(dx, dy)
                self.is_sensing_grid[i] = dist < self._sens_rad[i]*1.1 # give some margin

                for g in range(self._grid_num): 
                    if self.is_sensing_grid[i][g]:

                        # dummy term to ease the computation
                        sqrt_lam_r = np.sqrt( self.lambdas[i]**2 + self.rList[i]**2 )
                        qmp_x = self._grid_points[g][0] - self.points[i][0]
                        qmp_y = self._grid_points[g][1] - self.points[i][1]
                        qmp_z = - self.altitudes[i]
                        dist_q_min_p = np.sqrt( qmp_x**2 + qmp_y**2 + qmp_z**2 )

                        p_a = sqrt_lam_r / (sqrt_lam_r - self.lambdas[i]) # 1 / ( 1 - np.cos( np.atan( self.rList[i] / self.lambdas[i] )))
                        p_b = self.altitudes[i] / dist_q_min_p
                        p_c = self.lambdas[i] / sqrt_lam_r # np.cos( np.atan( self.rList[i] / self.lambdas[i] ))

                        r_a = p_c**self.kappa
                        r_b = - ( (dist_q_min_p - self.des_R)**2 ) / ( 2 * (self.sigma**2) )
                        
                        # Final computation for perspective and resolution function
                        self.f_pers[i][g] = p_a * (p_b - p_c)
                        self.f_res[i][g] = r_a * np.exp( r_b )
                        
                        # Derivative Computation
                        B = p_c
                        A = ( self.altitudes[i] / ( (1 - B) * (dist_q_min_p**3) ) )
                        d = (dist_q_min_p - self.des_R) / ( 2 * (self.sigma**2) * dist_q_min_p )
                        C = 2*(B**self.kappa)*np.exp( r_b ) * d

                        dfp_dli_a = (self.altitudes[i] - dist_q_min_p) / dist_q_min_p
                        dfp_dli_b = (self.rList[i]**2) / (sqrt_lam_r * ((sqrt_lam_r - self.lambdas[i])**2) )

                        dfr_dli_c = ( self.kappa * (self.rList[i]**2) * (self.lambdas[i]**self.kappa) )
                        dfr_dli_d = ( self.lambdas[i] * (sqrt_lam_r**(self.kappa + 2)) )

                        # derivative term for f_pers
                        self.df_pers_dxi[i][g] = A * qmp_x
                        self.df_pers_dyi[i][g] = A * qmp_y
                        self.df_pers_dzi[i][g] = A * qmp_z + (1 / ((1 - B)*dist_q_min_p) ) 
                        self.df_pers_dli[i][g] = dfp_dli_a * dfp_dli_b
                        # derivative term for f_res
                        self.df_res_dxi[i][g] = C * qmp_x
                        self.df_res_dyi[i][g] = C * qmp_y
                        self.df_res_dzi[i][g] = C * qmp_z
                        self.df_res_dli[i][g] = (dfr_dli_c / dfr_dli_d) * np.exp( r_b ) 


        # Check intersection between 2 agents
        # Note: diagonal will always be False
        self.fov_overlap_table = np.full( (self._region_num,self._region_num), False)
        for i in range(self._region_num):
            for j in range(i+1, self._region_num):
                dist = np.linalg.norm(self.points[i] - self.points[j])
                check = dist < ( self._sens_rad[i] + self._sens_rad[j] )
                if check: # FOVs are overlap, set value to True
                    self.fov_overlap_table[i,j] = check
                    self.fov_overlap_table[j,i] = check
        # print(self.fov_overlap_table)

        # The structure to store the triangular subgraph list
        # - trisubgraph_for_each[i]: An array with each element contains a dictionary of triangular subgraph for each agent
        #   --> ['n'] : number of triangular subgraph stored
        #   --> ['j'] : list of index for j
        #   --> ['k'] : list of index for k
        self.trisubgraph_for_each = [{'n':0, 'j':[], 'k':[]} for _ in range(self._region_num)]
        # Create triangular subgraph with respect to each agent
        self.triang_list, self.triang_V = get_power_triangulation(self.points, np.array(self._sens_rad))
        for tri in self.triang_list:
            for i in range(3):
                j, k = (i+1)%3, (i+2)%3
                # Store for each 
                self.trisubgraph_for_each[tri[i]]['n'] += 1
                self.trisubgraph_for_each[tri[i]]['j'] += [tri[j]]
                self.trisubgraph_for_each[tri[i]]['k'] += [tri[k]]

    # Calculate a centroidal voronoi diagram and compute nominal control
    def compute_viscov_nominal_control(self): # TODO: change the name

        # update all fov functions
        self.update_fov_function()

        # TODO: check if this is still compatible
        # update density function (time-varying) if either sensing rate and decay rate is set
        # if (self.sense_rate != 0.0) or (self.decay_rate != 0.0):
        #     self.update_density_function()

        # set default values
        nomControl = np.zeros(self.points.shape)
        if self._roi_vert is not None: # point toward centre of ROI
            c = self._roi_cent
            centroids = np.repeat(c, self._region_num, axis=0)
        else: # point toward oneselves
            centroids = self.points.copy()

        # update with voronoi computation & compute nominal control
        kOut = 1. # of sensing radius distance
        nomControl = np.zeros((self._region_num, 4))
        dq = self._grid_size**2
        self._sensing_region = {} # For plotting

        # Compute nominal control
        for i in range(self._region_num):
            if self.idxInVor[i]: # if points is within ROI

                # Create sensing circle. TODO: optimize by defining default circle. Here we only shift its value.
                theta = np.linspace(0, np.pi*2, 100) # divide circle into 100 vertices
                sensR_xy = np.stack([
                    self.points[i][0] + np.cos(theta)*self._sens_rad[i], 
                    self.points[i][1] + np.sin(theta)*self._sens_rad[i]
                    ], 1)
                # intersect ROI with sensing circle
                fov_in_roi = self._roi_poly.intersection(geom.Polygon(sensR_xy))
                computed_roi = fov_in_roi
                
                # Integral over for the nominal input
                df_dpi = np.zeros(4)
                for g in range(self._grid_num): # for each grid
                    if self.is_sensing_grid[i][g]: # within sensing range * 1.1
                        is_inside_conic_voronoi = True
                        # Check if the grid is inside conic voronoi
                        i_f_value = self.f_pers[i][g]*self.f_res[i][g]
                        for j in range(self._region_num): # where i_f_value is greater than neighbor robot k's f function
                            if self.fov_overlap_table[i,j]: 
                                is_inside_conic_voronoi = is_inside_conic_voronoi and ( i_f_value > self.f_pers[j][g]*self.f_res[j][g] )

                        if is_inside_conic_voronoi:
                            k_step_pos = 0.5 # 7e-1 --> previous good value
                            k_step_lam = 1e-11 # 1e-6 --> previous good value
                            df_dpi[0] = ( ( self.df_pers_dxi[i][g] * self.f_res[i][g] ) + ( self.df_res_dxi[i][g] * self.f_pers[i][g] ) )*k_step_pos
                            df_dpi[1] = ( ( self.df_pers_dyi[i][g] * self.f_res[i][g] ) + ( self.df_res_dyi[i][g] * self.f_pers[i][g] ) )*k_step_pos
                            df_dpi[2] = ( ( self.df_pers_dzi[i][g] * self.f_res[i][g] ) + ( self.df_res_dzi[i][g] * self.f_pers[i][g] ) )*k_step_pos
                            df_dpi[3] = ( ( self.df_pers_dli[i][g] * self.f_res[i][g] ) + ( self.df_res_dli[i][g] * self.f_pers[i][g] ) )*k_step_lam
                            # It is different in Funada's case as well, he use step size. TODO: why? what rationalization?

                            area_ratio = (self._grid_polygon[g].intersection(fov_in_roi).area / self._grid_polygon[g].area)
                            nomControl[i] = nomControl[i] + df_dpi * (area_ratio * self._grid_densVal[g] * dq)
                        
                        else:
                            computed_roi = computed_roi - self._grid_polygon[g]

                if self.SHOW_COMPUTED_AREA:
                    if computed_roi.geom_type == 'MultiPolygon':
                        self._sensing_region[i] = computed_roi[0] # TODO: 
                    else: self._sensing_region[i] = computed_roi
                else: 
                    self._sensing_region[i] = fov_in_roi
                # gain = 0.1
                # nomControl[i] = gain*nomControl[i]

            else: # set limited gain based on distance so it is moving within sensing radius
                distC = centroids[i] - self.points[i]
                gain = kOut*self._sens_rad[i]/np.hypot(distC[0], distC[1])
                self._sensing_region[i] = None

                nomControl[i,:2] = gain*distC

        return nomControl


    #---------------------------------------------------------------------------
    # BARRIER FUNCTION COMPUTATION
    def compute_h_triang(self, id_i, id_j, id_k):
        # Deconstruct the information
        xi, yi = self.points[id_i]
        xj, yj = self.points[id_j]
        xk, yk = self.points[id_k]
        radi, radj, radk = self._sens_rad[id_i], self._sens_rad[id_j], self._sens_rad[id_k]
        zi, lami = self.altitudes[id_i], self.lambdas[id_i]

        # The coordinates of radical center v_{ijk}
        #------------------------------------------------
        Aij, Bij, Cij = ( 2*(xj - xi) ),  ( 2*(yj - yi) ),  ( xi**2 - xj**2 + yi**2 - yj**2 + radj - radi )
        Aik, Bik, Cik = ( 2*(xk - xi) ),  ( 2*(yk - yi) ),  ( xi**2 - xk**2 + yi**2 - yk**2 + radk - radi )

        Xcro_num, Xcro_den = ( Bij*Cik - Bik*Cij ), ( Aij*Bik - Aik*Bij )
        Ycro_num, Ycro_den = ( Aik*Cij - Aij*Cik ), ( Aij*Bik - Aik*Bij )
        Xcro = Xcro_num / Xcro_den
        Ycro = Ycro_num / Ycro_den

        # general derivative over state i --> xi, yi, zi, lami
        dXi_d = np.array([ 1, 0, 0, 0 ]) # d( xi + any )/dstate
        dYi_d = np.array([ 0, 1, 0, 0 ]) # d( yi + any )/dstate
        # Compute derivative of radical center over state i --> xi, yi, zi, lami
        dCiany_d = np.array([ 2*xi, 2*yi, -radi/zi, radi/lami ])

        dXcro_num_d = (-2*dYi_d*Cik + Bij*dCiany_d  ) - (-2*dYi_d*Cij + Bik*dCiany_d  )
        dXcro_den_d = (-2*dXi_d*Bik + Aij*(-2*dYi_d)) - (-2*dXi_d*Bij + Aik*(-2*dYi_d))
        dXcro_d = ( dXcro_num_d * Xcro_den - Xcro_num * dXcro_den_d ) / ( Xcro_den**2 )

        dYcro_num_d = (-2*dXi_d*Cij + Aik*dCiany_d  ) - (-2*dXi_d*Cik + Aij*dCiany_d  )
        dYcro_den_d = (-2*dXi_d*Bik + Aij*(-2*dYi_d)) - (-2*dXi_d*Bij + Aik*(-2*dYi_d))
        dYcro_d = ( dYcro_num_d * Ycro_den - Ycro_num * dYcro_den_d ) / ( Ycro_den**2 )


        # Compute CBF for triangle
        #------------------------------------------------
        h_triang_array = np.zeros(4)
        h_triang_deriv = np.zeros([4,4]) # Store the derivative of each h function
        # Note: the derivatives are computed using chain rules

        # CBF: Is in the triangle IJK?
        # e_z^T ( [x1 y1 z1]^T x [x2 y2 z2]^T ) = x1*y2 - y1*x2   --> e_z^T means only take the k vector
        k = 20 # further adjustment to normalize with h_fov
        
        # hijk = ( e_z^T (IJ x IVcro) ) / ( e_z^T (IJ x IK) ) 
        h_ijk_num = (xj-xi)*(Ycro-yi) - (yj-yi)*(Xcro-xi) # e_z^T (IJ x IVcro)
        h_ijk_den = (xj-xi)*(yk-yi)   - (yj-yi)*(xk-xi) # e_z^T (IJ x IK)
        h_triang_array[0] = -k * ( h_ijk_num ) / ( h_ijk_den )
        # derivative of -h_ijk
        dh_ijk_num_d = ( -dXi_d*(Ycro-yi) + (xj-xi)*(dYcro_d-dYi_d) ) - ( -dYi_d*(Xcro-xi) + (yj-yi)*(dXcro_d-dXi_d) )
        dh_ijk_den_d = ( -dXi_d*(yk-yi)   + (xj-xi)*(-dYi_d) )        - ( -dYi_d*(xk-xi)   + (yj-yi)*(-dXi_d) )
        h_triang_deriv[0] = -( dh_ijk_num_d * h_ijk_den - h_ijk_num * dh_ijk_den_d ) / ( h_ijk_den**2 )

        # hjki = ( e_z^T (JK x JVcro) ) / ( e_z^T (JK x JI) ) 
        h_jki_num = (xk-xj)*(Ycro-yj) - (yk-yj)*(Xcro-xj) # e_z^T (JK x JVcro)
        h_jki_den = (xk-xj)*(yi-yj) - (yk-yj)*(xi-xj) # e_z^T (JK x JI)
        h_triang_array[1] = -k * ( h_jki_num ) / ( h_jki_den )
        # derivative of -h_jki
        dh_jki_num_d = (xk-xj)*dYcro_d - (yk-yj)*dXcro_d
        dh_jki_den_d = (xk-xj)*dYi_d   - (yk-yj)*dXi_d
        h_triang_deriv[1] = -( dh_jki_num_d * h_jki_den - h_jki_num * dh_jki_den_d ) / ( h_jki_den**2 )

        # hkij = ( e_z^T (KI x KVcro) ) / ( e_z^T (KI x KJ) ) 
        h_kij_num = (xi-xk)*(Ycro-yk) - (yi-yk)*(Xcro-xk) # e_z^T (KJ x KVcro)
        h_kij_den = (xi-xk)*(yj-yk) - (yi-yk)*(xj-xk) # e_z^T (KI x KJ)
        h_triang_array[2] = -k * ( h_kij_num ) / ( h_kij_den )
        # derivative of -h_kij
        dh_kij_num_d = ( dXi_d*(Ycro-yk) + (xi-xk)*dYcro_d ) - ( dYi_d*(Xcro-xk) + (yi-yk)*dXcro_d )
        dh_kij_den_d = ( dXi_d*(yj-yk) ) - ( dYi_d*(xj-xk) )
        h_triang_deriv[2] = -( dh_kij_num_d * h_kij_den - h_kij_num * dh_kij_den_d ) / ( h_kij_den**2 )

        # CBF: Is in FOV_i?
        # Note that in [1] the radi computation is shited by small margin
        if self.overlap_m_ratio is not None: # By default prioritize ratio over fixed M
            comp_radi = radi*(1 - self.overlap_m_ratio)
        else: comp_radi = radi - self.overlap_m
        h_triang_array[3] = (comp_radi)**2 - (yi - Ycro)**2 - (xi - Xcro)**2
        # derivative of FOV_i
        h_triang_deriv[3] = 2*(comp_radi)*np.array([ 0, 0, radi/zi, -radi/lami ]) \
            - 2*(yi - Ycro)*(dYi_d - dYcro_d) - 2*(xi - Xcro)*(dXi_d - dXcro_d)

        return h_triang_array, h_triang_deriv
    

    # Compute Hybrid CBF to ensure no hole being created on the intersections of FOVs
    def compute_viscov_HCBF_nohole(self):
        
        nomControl = self.compute_viscov_nominal_control()
        hcbf_control = np.copy(nomControl) # By default the value is nominal control input
        
        # TODO: allow editing from main function
        epsilon = 0.2
        gamma = 0.5
        alpha = 3
        ignore_threshold = 500 
        weight_lam = 1e10 # 1e6 --> previous good value

        self.active_triangCBF = {i:[] for i in range(self._region_num)}
 
        # Note: the triangle subgraphs are computed when calling nominal control
        # --> trisubgraph_for_each[i] ['n'], ['j'][0 - n], ['k'][0 - n]
        for i in range(self._region_num):

            triang_num = self.trisubgraph_for_each[i]['n']
            if triang_num > 0:
                # Number of all computed CBF is triang_num x 4
                table_h = np.zeros((triang_num, 4))
                dict_h_deriv = {}

                for t in range(triang_num):
                    j = self.trisubgraph_for_each[i]['j'][t]
                    k = self.trisubgraph_for_each[i]['k'][t]

                    table_h[t], dict_h_deriv[t] = self.compute_h_triang(i, j, k)

                # Find maximum for the h function within triangular and minimum over all triangular
                h_mi = np.nanmax(table_h, -1)
                h_i = np.nanmin( h_mi )

                # Evaluate almost active set of functions
                is_h_compute = np.full( table_h.shape, False)
                is_mi_in_Imi = np.abs(h_mi - h_i) < epsilon
                for t in range(triang_num):
                    if is_mi_in_Imi[t]: 
                        is_h_compute[t] = (np.abs(table_h[t] - h_mi[t]) < epsilon) #\
                            # & (table_h[t] < ignore_threshold)
                        if np.any( is_h_compute[t] ): self.active_triangCBF[i] += [t]
                # NOTE 1: condition (table_h[t] < ignore_threshold) is originally in Funada-san's code
                # to circumvent division by NaN in the computation --> here we use nanmin and nanmax to circumvent it

                row = np.sum(is_h_compute)
                if row > 0:
                    # Construct QP-based Optimization
                    weight = np.array([ 1, 1, 1, weight_lam ])
                    P_mat = 2 * cvxopt.matrix( np.diag(weight), tc='d')
                    q_mat = -2 * cvxopt.matrix( weight * nomControl[i], tc='d')

                    # initialize G and h, Then fill it afterwards
                    G = np.zeros([row, 4])
                    h = np.ones([row, 1])

                    cur_row = 0
                    for t in range(triang_num):
                        for ell in range(4):
                            if is_h_compute[t,ell]: 
                                G[cur_row] = dict_h_deriv[t][ell]
                                # h[cur_row] = gamma*(h_i**alpha)
                                h[cur_row] = gamma*(table_h[t,ell]**alpha)
                                cur_row += 1

                    # Resize the G and H into appropriate matrix for optimization
                    G_mat = cvxopt.matrix( -G, tc='d') 
                    h_mat = cvxopt.matrix( h, tc='d')

                    # Solving Optimization
                    cvxopt.solvers.options['show_progress'] = False
                    sol = cvxopt.solvers.qp(P_mat, q_mat, G_mat, h_mat, verbose=False)
                    
                    if sol['status'] == 'optimal':
                        hcbf_control[i] = np.array([sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3]])
                    else: 
                        print( 'WARNING QP SOLVER id-' + str(i) + ' status: ' + sol['status'] + ' --> use nominal instead' )

                    # Update the drawing information
            
        return hcbf_control, nomControl


    def extract_viscoverage_voronoi(self):
        voronoi_cell_map = get_voronoi_cells(self.points, self.triang_V, self.triang_list)
        return self.active_triangCBF, self.trisubgraph_for_each, voronoi_cell_map



