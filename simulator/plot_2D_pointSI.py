import numpy as np
import matplotlib.pyplot as plt

class draw2DPointSI():
    def __init__(self, ax, init_pos, field_x = None, field_y = None, pos_trail_nums = 0):
        # The state should be n rows and 3 dimensional column (pos.X, pos.Y, and theta)
        # pos_trail_nums determine the number of past data to plot as trajectory trails
        self.__ax = ax
        self.__ax.set(xlabel="x [m]", ylabel="y [m]")
        self.__ax.set_aspect('equal', adjustable='box', anchor='C')
        # Set field
        if field_x is not None: self.__ax.set(xlim=(field_x[0]-0.1, field_x[1]+0.1))
        if field_y is not None: self.__ax.set(ylim=(field_y[0]-0.1, field_y[1]+0.1))

        self.__robot_num = init_pos.shape[0] # row number
        
        # Plotting variables
        self.__colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # plot placeholder for the position
        self.__pl_pos = [None]*self.__robot_num
        for i in range(self.__robot_num):
            self.__pl_pos[i], = self.__ax.plot(init_pos[i,0], init_pos[i,1], color=self.__colorList[i], marker='X', markersize=10)

        # Prepare buffer for the trail
        self.__pl_trail = [None]*self.__robot_num
        if pos_trail_nums > 0:
            self.__trail_data = {i:None for i in range(self.__robot_num)}
            
            for i in range(self.__robot_num):
                # use initial position to populate all matrix (pos_trail_nums-row x dim-col)
                # theta is not used for plotting the trail
                self.__trail_data[i] = np.tile(init_pos[i], (pos_trail_nums, 1))

                # Plot the first data
                self.__pl_trail[i], = self.__ax.plot(
                    self.__trail_data[i][:,0], self.__trail_data[i][:,1], 
                    '--', color=self.__colorList[i])
    

    def update(self, all_pos):
        for i in range(self.__robot_num):
            self.__pl_pos[i].set_data(all_pos[i,0], all_pos[i,1])

            if self.__pl_trail[i] is not None: # update trail data
                # roll the data, fill the new one from the top and then update plot
                self.__trail_data[i] = np.roll(self.__trail_data[i], self.__trail_data[i].shape[1])
                self.__trail_data[i][0,:] = all_pos[i,:]
                self.__pl_trail[i].set_data(self.__trail_data[i][:,0], self.__trail_data[i][:,1])

