
class NaiveStitch():
    def __init__(self, base_image, mtr2pxl, origin_pos_in_pxl_img = (0,0), num_pov = 1):
        self.base_image = base_image.copy()
        self.mtr2pxl = mtr2pxl
        self.num_pov = num_pov
        self.pos_origin_pxl_x = origin_pos_in_pxl_img[0]
        self.pos_origin_pxl_y = origin_pos_in_pxl_img[1]

    def grab_figure_and_loc(self, state, r):
        point_x, point_y, altitude, lambd = state

        # Compute radius
        m_rad = r * altitude / lambd

        # Compute image width both in m and pixels
        pos_height = 2 * m_rad
        pos_width = 2 * m_rad * 4 / 3
        pxl_height = int(pos_height * self.mtr2pxl)
        pxl_width = int(pos_width * self.mtr2pxl)

        # Compute bottom-left / south-west position of image plane
        pos_x = point_x - (m_rad*4/3)
        pos_y = point_y - m_rad

        # Compute top-left / north-west position of image plane in pixel
        pxl_x = self.pos_origin_pxl_x + int(pos_x*self.mtr2pxl)
        pxl_y = self.pos_origin_pxl_y - int(pos_y*self.mtr2pxl) - pxl_height

        pxl_xc, pxl_yc, pos_xc, pos_yc = pxl_x, pxl_y, pos_x, pos_y
        if pxl_x < 0: pxl_xc, pos_xc = 0, -15
        if pxl_y < 0: pxl_yc = 0

        # Plot image
        fig_data = self.base_image[ pxl_yc:pxl_y+pxl_height, pxl_xc:pxl_x+pxl_width, : ]
        loc = (pos_xc, pos_x+pos_width, pos_yc, pos_y+pos_height)

        return fig_data, loc


    def plot_super_naive_stitch(self, ax, SI_DroneVision_allStates, rList):
        ax.cla() # clear all figure
        for i in range(self.num_pov):
            fig_data, loc = self.grab_figure_and_loc(SI_DroneVision_allStates[i], rList[i])
            ax.imshow(fig_data, extent=loc) # Plot image
        ax.autoscale()

