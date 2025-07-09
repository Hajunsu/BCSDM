from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Arrow3D(FancyArrowPatch):
    def init(self, xs, ys, zs, *args, **kwargs):
        super().init((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def SE3_Visualization(traj, i=0, ax=None, scale=0.1, alpha=1):
    if ax is None:
        fig = plt.figure(figsize=[7, 7])
        ax = fig.add_subplot(projection='3d')
    if len(traj.shape) == 2:
        T = traj
    else:
        T = traj[i]
    origin = T[:3, 3]
    if type(scale) == list:
        scale_x = scale[0]
        scale_y = scale[1]
        scale_z = scale[2]
    else:
        scale_x = scale
        scale_y = scale
        scale_z = scale
    hat_x = (T[:3, 0]*scale_x + origin).detach().cpu().numpy()
    hat_y = (T[:3, 1]*scale_y + origin).detach().cpu().numpy()
    hat_z = (T[:3, 2]*scale_z + origin).detach().cpu().numpy()
    origin = origin.detach().cpu().numpy()
    arrow_prop_dict = dict(mutation_scale=5, arrowstyle='->', shrinkA=0, shrinkB=0)
    a = Arrow3D([origin[0], hat_x[0]], [origin[1], hat_x[1]], [origin[2], hat_x[2]], **arrow_prop_dict, color='r', alpha=alpha)
    ax.add_artist(a)
    a = Arrow3D([origin[0], hat_y[0]], [origin[1], hat_y[1]], [origin[2], hat_y[2]], **arrow_prop_dict, color='b', alpha=alpha)
    ax.add_artist(a)
    a = Arrow3D([origin[0], hat_z[0]], [origin[1], hat_z[1]], [origin[2], hat_z[2]], **arrow_prop_dict, color='g', alpha=alpha)
    ax.add_artist(a)
    # return ax

def plot_traj(
    traj, ax=None, scale=0.1, alpha=1, skip_size=15,
):
    for i in range(0, len(traj), skip_size):
        SE3_Visualization(traj, i=i, ax=ax, scale=scale, alpha=alpha)

def animate_SE3_traj(
    trajs, fig=None, scale=0.1, frame_interval=20, save_path=None, 
    background_trajs=None, background_alpha=0.2, background_skip_size=15,):
    if len(trajs.shape) == 3:
        trajs = trajs.unsqueeze(0)
        
    if fig is None:
        fig = plt.figure(figsize=[7, 7])
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.axes[0]
    def plot_frame(i):
        ax.cla()
        if background_trajs is not None:
            for background_traj in background_trajs:
                plot_traj(
                    background_traj, ax=ax, scale=scale, 
                    alpha=background_alpha, skip_size=background_skip_size)
        for traj in trajs:
            SE3_Visualization(traj, i=i, ax=ax, scale=scale)
    # plot_frame = partial(SE3_Visualization, traj=traj, ax=ax, scale=scale)
    len_traj = len(trajs[0])
    ani = animation.FuncAnimation(fig, plot_frame, frames = range(len_traj), interval = frame_interval)
    if save_path is not None:
        ani.save(save_path)  
    return ani

def plot_SE3(SE3_traj, ax = None, ax_length=1, alpha=1, interval = 10, final = True):
    if ax is None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(projection='3d')
    
    for i in range(int(SE3_traj.shape[0]/interval)):
        iter = i*interval
        ax.quiver(SE3_traj[iter,0,3], SE3_traj[iter,1,3], SE3_traj[iter,2,3],
                  SE3_traj[iter,0,0]*ax_length, SE3_traj[iter,1,0]*ax_length, SE3_traj[iter,2,0]*ax_length,
                  color = 'tab:red',alpha = alpha)
        ax.quiver(SE3_traj[iter,0,3], SE3_traj[iter,1,3], SE3_traj[iter,2,3],
                  SE3_traj[iter,0,1]*ax_length, SE3_traj[iter,1,1]*ax_length, SE3_traj[iter,2,1]*ax_length,
                  color = 'tab:green',alpha = alpha)
        ax.quiver(SE3_traj[iter,0,3], SE3_traj[iter,1,3], SE3_traj[iter,2,3],
                  SE3_traj[iter,0,2]*ax_length, SE3_traj[iter,1,2]*ax_length, SE3_traj[iter,2,2]*ax_length,
                  color = 'tab:blue',alpha = alpha)
    
    if isinstance(SE3_traj.shape[0]/interval, float):
        ax.quiver(SE3_traj[-1,0,3], SE3_traj[-1,1,3], SE3_traj[-1,2,3],
                  SE3_traj[-1,0,0]*ax_length, SE3_traj[-1,1,0]*ax_length, SE3_traj[-1,2,0]*ax_length,
                  color = 'tab:red',alpha = alpha)
        ax.quiver(SE3_traj[-1,0,3], SE3_traj[-1,1,3], SE3_traj[-1,2,3],
                  SE3_traj[-1,0,1]*ax_length, SE3_traj[-1,1,1]*ax_length, SE3_traj[-1,2,1]*ax_length,
                  color = 'tab:green',alpha = alpha)
        ax.quiver(SE3_traj[-1,0,3], SE3_traj[-1,1,3], SE3_traj[-1,2,3],
                  SE3_traj[-1,0,2]*ax_length, SE3_traj[-1,1,2]*ax_length, SE3_traj[-1,2,2]*ax_length,
                  color = 'tab:blue',alpha = alpha)
    
    return None

def plot_traj(SE3_traj, ax = None, color = 'r'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    
    ax.plot(SE3_traj[:,0,3],SE3_traj[:,1,3],SE3_traj[:,2,3], color = color)
    return None


def plot_SE3_traj_general(
    SE3_trajs, SE3_demo_trajs=None, demo_SE3=False,  ax=None, points_plot=None, ax_length=1,
    fig_size=(8,8), axes_lim=None, view_init=None, alpha=0.2, alpha_demo=1, SE3_interval=10):
    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(projection='3d')
    
    # plot SE3 demo trajs
    if SE3_demo_trajs is None:
        pass
    elif type(SE3_demo_trajs) is list:
        for i in range(len(SE3_demo_trajs)):
            plot_traj(SE3_demo_trajs[i], ax = ax,color = 'purple')
            ax.scatter(SE3_demo_trajs[i][-1,0,3],SE3_demo_trajs[i][-1,1,3],SE3_demo_trajs[i][-1,2,3], s=100, color = 'b')
            ax.scatter(SE3_demo_trajs[i][0,0,3],SE3_demo_trajs[i][0,1,3],SE3_demo_trajs[i][0,2,3], s=100, marker = 's', color = 'darkturquoise')
            if demo_SE3:
                plot_SE3(SE3_demo_trajs[i], ax = ax, ax_length=ax_length, alpha = alpha_demo)
    else:
        plot_traj(SE3_demo_trajs, ax = ax,color = 'purple')
        ax.scatter(SE3_demo_trajs[-1,0,3],SE3_demo_trajs[-1,1,3],SE3_demo_trajs[-1,2,3], s=100, color = 'b')
        ax.scatter(SE3_demo_trajs[0,0,3],SE3_demo_trajs[0,1,3],SE3_demo_trajs[0,2,3], s=100, marker = 's', color = 'darkturquoise')
        if demo_SE3:
            plot_SE3(SE3_demo_trajs, ax = ax, ax_length=ax_length, alpha = alpha_demo, interval=10)
    
    
    # plot SE3 trajs
    if type(SE3_trajs) is list:
        for i in range(len(SE3_trajs)):
            plot_SE3(SE3_trajs[i], ax = ax, ax_length=ax_length, alpha = alpha, interval=SE3_interval)
            plot_traj(SE3_trajs[i], ax = ax, color = 'tab:orange')
            ax.scatter(SE3_trajs[i][0,0,3],SE3_trajs[i][0,1,3],SE3_trajs[i][0,2,3], s=80, color = 'g', marker = 'd')
    else:
        plot_SE3(SE3_trajs, ax = ax, ax_length=ax_length, alpha = alpha, interval=SE3_interval)
        plot_traj(SE3_trajs, ax = ax, color = 'tab:orange')
        ax.scatter(SE3_trajs[0,0,3],SE3_trajs[0,1,3],SE3_trajs[0,2,3], s=80, color = 'r', marker = 'd')
    
    if points_plot is not None:
        for point in points_plot:
            point = point.detach().cpu().numpy()
            ax.scatter(point[0,3],point[1,3],point[2,3], s=80, color = 'r', zorder=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if axes_lim is not None:
        ax.set_xlim(axes_lim[0][0],axes_lim[0][1])
        ax.set_ylim(axes_lim[1][0],axes_lim[1][1])
        ax.set_zlim(axes_lim[2][0],axes_lim[2][1])
    if view_init is not None:
        ax.view_init(view_init[0], view_init[1])
    
    return fig, plt


def visualize_SE3_frames(frames, Ttraj=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in frames:
        # Extract translation and rotation matrix from the frame
        translation = frame[:3, 3]
        rotation = frame[:3, :3]

        # Plot the frame origin
        ax.scatter(translation[0], translation[1], translation[2], c='r', marker='o')

        # Plot the frame axes
        ax.quiver(translation[0], translation[1], translation[2], rotation[0, 0], rotation[1, 0], rotation[2, 0], length=0.1, color='r')
        ax.quiver(translation[0], translation[1], translation[2], rotation[0, 1], rotation[1, 1], rotation[2, 1], length=0.1, color='g')
        ax.quiver(translation[0], translation[1], translation[2], rotation[0, 2], rotation[1, 2], rotation[2, 2], length=0.1, color='b')

    if Ttraj is not None:
        ptraj = Ttraj[:, :3, 3]
        ax.plot(ptraj[:, 0], ptraj[:, 1], ptraj[:, 2], color='k')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot limits
    ax.set_xlim([0, 1])
    ax.set_ylim([-1, 0.2])
    ax.set_zlim([0, 1])

    # Show the plot
    plt.show()