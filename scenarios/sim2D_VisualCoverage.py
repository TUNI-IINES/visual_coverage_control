import numpy as np
from control_lib.CentVisualCoverage import CentVisualCoverage
# from control_lib.cbf_single_integrator import cbf_si

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from simulator.dynamics import SI_DroneVision
from simulator.plot_2D_pointSI import draw2DPointSI
from simulator.plot_2D_voronoi import DrawVoronoi
from simulator.data_logger import dataLogger
from simulator.naive_stitch import NaiveStitch

# MAIN COMPUTATION
#------------------------------------------------------------------------------
class SceneSetup(): 
    # General variable needed to run the controller
    # Can be adjusted later by set new value on the class variable

    # Set Initial state --> x, y, z, lambda
    init_state = np.array([ \
        [-4. , 4., 15., 0.0034], [0,  4., 15., 0.0034], [-2.,  0., 15., 0.0031]
        #[-7. , 6., 11., 0.0034], [-1,  6., 11., 0.0034], [5.,  6., 11., 0.0034], \
        #[-5.,  0., 11., 0.0031], [ 1,  0., 11., 0.0033], [7.,  0., 11., 0.0033], \
        #[-7., -6., 11., 0.0035], [-1, -6., 11., 0.0035], [5., -6., 11., 0.0034], \
        ])
    robot_num = init_state.shape[0]

    # Parameter for Visual Coverage 
    field_x_range, field_y_range = 55, 55
    field_x = [-15, -15+field_x_range]
    field_y = [-40, -40+field_y_range]
    field_z = [0, 15]

    roi_vertices = np.array([
                [field_x[0], field_y[0]], 
                [field_x[1], field_y[0]], 
                [field_x[1], field_y[1]], 
                [field_x[0], field_y[1]],
                [field_x[0], field_y[0]]
            ])
    dens_pos = np.array([[12., -5.], [20., -25.], [25., 0.], [8., -20.]])

    r_list = 0.0034 * np.tan( np.deg2rad(50/2) ) * np.ones(robot_num) # 50 deg camera view with lambda 0.0034

    kappa = 4
    sigma = 3
    R = 15.
    param_M = 2.
    param_Mratio = 0.4 # ratio of overlap over the current radius

    # Parameters for CBF 
    ds_dyn = 0.15
    # sr = [0.5]*robot_num # sensing range in meter (to ease optimization) 
    cbf_gam = 10
    cbf_pow = 1

    speed_limit = 0.15


# General class for computing the controller input
class Controller():
    def __init__(self): # INITIALIZE CONTROLLER
        self.vor = CentVisualCoverage(SceneSetup.init_state, roi_vert=SceneSetup.roi_vertices, grid_size = 1.)
        self.vor.set_density_from_points(SceneSetup.dens_pos, sigma=[[75, 0], [0, 75]], mode='sum')
        self.vor.init_fov_parameters(rList=SceneSetup.r_list, kappa=SceneSetup.kappa, 
            sigma=SceneSetup.sigma, des_R=SceneSetup.R, overlap_m_ratio=SceneSetup.param_Mratio)

    def compute_control(self, feedback, computed_control): # INITIALIZE CONTROLLER
        # get all robot position
        states = feedback.get_all_robot_state()
        # Do computation with Voronoi
        self.vor.update_voronoi_data(states, dt = feedback.dt)
        # centroid, vorNomControl = self.vor.compute_nominal_control()
        viscovControl, nominalControl = self.vor.compute_viscov_HCBF_nohole()

        # Reset monitor properties to log data
        computed_control.reset_monitor()

        for i in range(SceneSetup.robot_num):
            # Implementation of Control
            # ------------------------------------------------
            # Extract the controller
            u_nom = nominalControl[i,:]
            u = viscovControl[i,:]

            # if SceneSetup.speed_limit > 0.:
            #     # set speed limit
            #     norm = np.hypot(u_nom[0], u_nom[1])
            #     if norm > SceneSetup.speed_limit: u_nom = SceneSetup.speed_limit* u_nom / norm # max 
            #     # self.cbf[i].add_velocity_bound(SceneSetup.speed_limit)

            # # Directly use nominal input
            # u = u_nom

            # Store command
            # ------------------------------------------------
            computed_control.set_i_input(i, u)
            # store information to be monitored/plot
            computed_control.save_monitored_info( "u_nom_x_"+str(i), u_nom[0] )
            computed_control.save_monitored_info( "u_nom_y_"+str(i), u_nom[1] )
            computed_control.save_monitored_info( "u_nom_alt_"+str(i), u_nom[2] )
            computed_control.save_monitored_info( "u_nom_lam_"+str(i), u_nom[3] )
            computed_control.save_monitored_info( "u_x_"+str(i), u[0] )
            computed_control.save_monitored_info( "u_y_"+str(i), u[1] )
            computed_control.save_monitored_info( "u_alt_"+str(i), u[2] )
            computed_control.save_monitored_info( "u_lam_"+str(i), u[3] )
            computed_control.save_monitored_info( "pos_x_"+str(i), states[i,0] )
            computed_control.save_monitored_info( "pos_y_"+str(i), states[i,1] )
            computed_control.save_monitored_info( "alt_x_"+str(i), states[i,2] )
            computed_control.save_monitored_info( "lam_y_"+str(i), states[i,3] )

        # Below is a temporary way to pass data for the drawing
        # TODO: think of a tidier way for this
        computed_control.store_voronoi_object(self.vor)

#-----------------------------------------
# CLASS FOR CONTROLLER'S INPUT AND OUTPUT
#-----------------------------------------
class ControlOutput():
    # Encapsulate the control command to be passed 
    # from controller into sim/experiment
    def __init__(self):
        # Initialize the formation array
        self.__all_velocity_input_xyzl = np.zeros([SceneSetup.robot_num, 4])
    
    def get_all_input(self): return self.__all_velocity_input_xyzl[:,:]

    def get_i_input(self, ID): return self.__all_velocity_input_xyzl[ID,:]
    def set_i_input(self, ID, input_xyzl):
        self.__all_velocity_input_xyzl[ID,:] = input_xyzl

    # Special case to extract the object within Controller
    def store_voronoi_object(self, obj): self.__vor_obj = obj 
    def get_voronoi_object(self): return self.__vor_obj

    # Allow the options to monitor state / variables over time
    def reset_monitor(self): self.__monitored_signal = {}
    def save_monitored_info(self, label, value): 
        # NOTE: by default name the label with the index being the last
        # example p_x_0, p_y_0, h_form_1_2, etc.
        self.__monitored_signal[label] = value
    # Allow retrieval from sim or experiment
    def get_all_monitored_info(self): return self.__monitored_signal


class FeedbackInformation():
    # Encapsulate the feedback information to be passed 
    # from sim/experiment into controller
    def __init__(self):
        # Set the value based on initial values
        self.set_feedback(SceneSetup.init_state, 0.)

    # To be assigned from the SIM or EXP side of computation
    def set_feedback(self, all_robots_state, dt):
        self.dt = dt # to allow discrete computation within internal controller
        # update all robots position and theta
        self.__all_robot_state = all_robots_state.copy()

    # To allow access from the controller computation
    def get_robot_i_state(self, i):   return self.__all_robot_state[i,:]
    # get all robots information
    def get_all_robot_state(self):   return self.__all_robot_state


# ONLY USED IN SIMULATION
#-----------------------------------------------------------------------
class SimSetup():

    Ts = 0.1 # in second. Determine Visualization and dynamic update speed
    tmax = 30 # simulation duration in seconds (only works when save_animate = True)
    save_animate = False # True: saving but not showing, False: showing animation but not real time
    save_data = False # log data using pickle

    sim_defname = 'animation_result/sim2D_VisualCoverage/sim_'
    sim_fname_output = r''+sim_defname+'.gif'
    trajectory_trail_lenTime = tmax # Show all trajectory
    sim_fdata_vis = sim_defname + '_vis.pkl'

    timeseries_window = 5 # in seconds, for the time series data

    field_x = [0, 3]
    field_y = [0, 3]


# General class for drawing the plots in simulation
class SimulationCanvas():
    def __init__(self):
        self.__cur_time = 0.

        # Initiate the robot
        self.__robot_dyn = [None]*SceneSetup.robot_num
        for i in range(SceneSetup.robot_num):
            self.__robot_dyn[i] = SI_DroneVision(SimSetup.Ts, SceneSetup.init_state[i])

        # Initiate data_logger
        self.log = dataLogger( round(SimSetup.tmax / SimSetup.Ts) )
        # Initiate the plotting
        self.__initiate_plot()


    def update_simulation(self, control_input, feedback):
        # Store data to log
        self.log.store_dictionary( control_input.get_all_monitored_info() )
        self.log.time_stamp( self.__cur_time )
        # Update plot
        self.__update_plot( feedback, control_input )

        self.__cur_time += SimSetup.Ts
        # Set array to be filled
        all_robots_state = np.zeros( SceneSetup.init_state.shape )
        # IMPORTANT: advance the robot's dynamic, and update feedback information
        for i in range(SceneSetup.robot_num):
            self.__robot_dyn[i].set_input(control_input.get_i_input(i), "u")
            state = self.__robot_dyn[i].step_dynamics() 
            all_robots_state[i] = state['q'][:]

        # UPDATE FEEDBACK for the controller
        feedback.set_feedback(all_robots_state, SimSetup.Ts)


    # PROCEDURES RELATED TO PLOTTING - depending on the scenarios
    #---------------------------------------------------------------------------------
    def __initiate_plot(self):
        # Initiate the plotting
        # For now plot 2D with 2x2 grid space, to allow additional plot later on
        rowNum, colNum = 2, 4
        self.fig = plt.figure(figsize=(4*colNum-2, 3*rowNum), dpi= 100)
        gs = GridSpec( rowNum, colNum, figure=self.fig)

        # MAIN 2D PLOT FOR UNICYCLE ROBOTS
        # ------------------------------------------------------------------------------------
        ax_2D = self.fig.add_subplot(gs[0:2,0:2]) # Always on
        # Only show past several seconds trajectory
        trajTail_datanum = int(SimSetup.trajectory_trail_lenTime/SimSetup.Ts) 

        self.__drawn_2D = draw2DPointSI( ax_2D, SceneSetup.init_state,
            field_x = SceneSetup.field_x, field_y = SceneSetup.field_y, pos_trail_nums=trajTail_datanum )

        # Display simulation time
        self.__drawn_time = ax_2D.text(0.78, 0.99, 't = 0 s', color = 'k', fontsize='large', 
            horizontalalignment='left', verticalalignment='top', transform = ax_2D.transAxes)

        # Initiate voronoi diagram
        self.__drawn_vor = DrawVoronoi(ax_2D, SceneSetup.roi_vertices, SceneSetup.robot_num)
        self.ax_2D = ax_2D

        # ADDITIONAL PLOT
        # ------------------------------------------------------------------------------------
        ax_stitch = self.fig.add_subplot(gs[0:2,2:])
        # # TODO: add stitched image
        imageFile = 'simulator/b1.JPG'
        img = plt.imread(imageFile)
        img_array_dim = img.shape
        img_height, img_width = img_array_dim[0], img_array_dim[1]
        mtr2pxl = img_height / SceneSetup.field_x_range
        origin_pos_in_pxl_img = ( int(-(-15)*mtr2pxl), int((-40+SceneSetup.field_y_range)*mtr2pxl)  ) 

        ax_2D.imshow(img, extent=(SceneSetup.field_x[0], SceneSetup.field_x[0] + img_width/mtr2pxl, 
            SceneSetup.field_y[0], SceneSetup.field_y[1]))
        self.__drawn_stich = NaiveStitch( img, mtr2pxl, origin_pos_in_pxl_img, 
            SceneSetup.init_state.shape[0])
        self.ax_stitch = ax_stitch

        # # Plot nominal velocity in x- and y-axis
        # self.__ax_unomx = self.fig.add_subplot(gs[0,2])
        # self.__ax_unomy = self.fig.add_subplot(gs[1,2])
        # self.log.plot_time_series_batch( self.__ax_unomx, 'u_nom_x_' ) 
        # self.log.plot_time_series_batch( self.__ax_unomy, 'u_nom_y_' ) 
        # # Plot position in x- and y-axis
        # self.__ax_pos_x = self.fig.add_subplot(gs[2,0])
        # self.__ax_pos_y = self.fig.add_subplot(gs[2,1])
        # self.log.plot_time_series_batch( self.__ax_pos_x, 'pos_x_' ) 
        # self.log.plot_time_series_batch( self.__ax_pos_y, 'pos_y_' ) 

        plt.tight_layout()


    def __update_plot(self, feedback, control_input):
        # UPDATE 2D Plotting: Formation and Robots
        all_state = feedback.get_all_robot_state()
        self.__drawn_2D.update( all_state )
        self.__drawn_time.set_text('t = '+f"{self.__cur_time:.1f}"+' s')

        # Plot the voronoi diagram
        vor_obj = control_input.get_voronoi_object()
        # extract the required data from vor_obj
        # vor_vertices = vor_obj.extract_voronoi_vertices()
        # datapoints, density_val = vor_obj.extract_mesh_density_data()
        sensing_area_data = vor_obj.extract_sensing_data()
        # use the data for drawing
        # self.__drawn_vor.plot_voronoi_diagram(vor_vertices)
        # self.__drawn_vor.plot_density_function(datapoints, density_val)
        self.__drawn_vor.plot_sensing_area(sensing_area_data)

        triang, trisubgraph, cell_map = vor_obj.extract_viscoverage_voronoi()
        self.__drawn_vor.plot_viscoverage_voronoi(all_state[:,:2], triang, trisubgraph, cell_map)
        # Draw stitched image
        self.__drawn_stich.plot_super_naive_stitch(self.ax_stitch, all_state, SceneSetup.r_list)


        # # get data from Log
        # log_data, max_idx = self.log.get_all_data()
        # # Setup for moving window horizon
        # min_idx = 0
        # if (self.__cur_time > SimSetup.timeseries_window): 
        #     min_idx = max_idx - round(SimSetup.timeseries_window/SimSetup.Ts)

        # # update nominal velocity in x- and y-axis
        # self.log.update_time_series_batch( 'u_nom_x_', data_minmax=(min_idx, max_idx)) 
        # self.log.update_time_series_batch( 'u_nom_y_', data_minmax=(min_idx, max_idx)) 

        # # update position in x- and y-axis
        # self.log.update_time_series_batch( 'pos_x_', data_minmax=(min_idx, max_idx)) 
        # self.log.update_time_series_batch( 'pos_y_', data_minmax=(min_idx, max_idx)) 
