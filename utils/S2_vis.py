
import math
import torch
import numpy as np
from copy import deepcopy
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import utils.S2_functions as s2

dtype = torch.float
device_c = torch.device("cpu")
device = device_c
torch.backends.cudnn.deterministic = True

aut = cm.get_cmap('autumn', 128 * 2.5)
new_aut  = ListedColormap(aut(range(256)))


def streamline_local(qtraj, VF, eta=None, rot_view=0, w=[0, 0, 1], nq=101, color_map='winter', a_max=3, clim=None, fig_size=(12, 6),
                      for_sphere=True, for_wandb=False, for_metric=False, sample_trajs=None, ax=None, check_diffeo=False, vector=False, **file_kwargs):
    if ax is None:
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot()
    
    eps = 1e-3
    if w == [0, 0, 1]:
        phimin = ((3/2*math.pi - rot_view) % (2*math.pi)) - 2*math.pi
        if phimin < -math.pi:
            phimin += 2 * math.pi
        if phimin > math.pi:
            phimin -= 2*math.pi
    else:
        phimin = 0
    qmin = [0, phimin]
    qmax = [math.pi, phimin+2*math.pi]
    # plt.axis([phimin, phimin+2*math.pi, math.pi, 0])
    plt.axis([qmin[1], qmax[1], qmax[0], qmin[0]])
    qmin_grid = [eps, round(phimin, 4)]
    qmax_grid = [round(math.pi - eps, 4), round(phimin+2*math.pi, 4)]
    
    q1 = torch.linspace(qmin_grid[0], qmax_grid[0], nq, dtype=torch.double)
    q2 = torch.linspace(qmin_grid[1], qmax_grid[1], nq, dtype=torch.double)
    x, y = torch.meshgrid(q1, q2)
    
    qmesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).clone().to(torch.float)
    qmesh_long = qmesh.reshape(-1, 2).to(qtraj)
    qmesh_long.requires_grad = True
    if for_wandb:
        xmesh_long = s2.q_to_x(qmesh_long)
        if hasattr(VF, 'device'):
            xmesh_long = xmesh_long.to(VF.device)#to('cuda:0')
        if eta is None:
            xdot_long = VF(x=xmesh_long).to(torch.double)
        else:    
            xdot_long = VF(x=xmesh_long, eta=eta).to(torch.double)
    elif for_metric:
        xmesh_long = s2.q_to_x(qmesh_long).to(qtraj)
        if eta is None:
            xdot_long = VF(x_input=xmesh_long)
        else:
            xdot_long = VF(x_input=xmesh_long, eta=eta)
    else:
        if eta is None:
            xdot_long = VF(qmesh_long).to(torch.double)
        else:    
            xdot_long = VF(qmesh_long, eta=eta).to(torch.double)
    
    if xdot_long.shape[-1] == 3:
        qdot_long = s2.xdot_to_qdot(xdot_long, qmesh_long.to(xdot_long))
    else:
        qdot_long = xdot_long
        xdot_long = s2.qdot_to_xdot(qdot_long, qmesh_long.to(qdot_long))
    
    xdotgrid = xdot_long.cpu().reshape(nq, nq, 3)
    qdotgrid = qdot_long.cpu().reshape(nq, nq, 2).detach()
    color_grid = np.clip(xdotgrid.norm(dim=-1).detach().numpy(), a_max=1, a_min=0) # a_max=a_max, a_min=0)
    qtraj = qtraj.detach().cpu()
    qtraj = s2.x_to_q(s2.q_to_x(qtraj)).numpy()
    init_point = deepcopy(qtraj[0])
    last_point = deepcopy(qtraj[-1])
    if init_point[1] < phimin:
        init_point[1] += 2*math.pi
    if last_point[1] < phimin:
        last_point[1] += 2*math.pi
    if init_point[1] > phimin+2*math.pi:
        init_point[1] -= 2*math.pi
    if last_point[1] > phimin+2*math.pi:
        last_point[1] -= 2*math.pi
    ax.plot(init_point[1], init_point[0], 's', color='tab:blue', markersize=10, zorder=3)
    ax.plot(last_point[1], last_point[0], 'o', color='tab:blue', markersize=10, zorder=3)
    low_idx = qtraj[:, 1] < phimin
    high_idx = qtraj[:, 1] > phimin+2*math.pi
    remaining_idx = (qtraj[:, 1] >= phimin) * (qtraj[:, 1] <= phimin+2*math.pi)
    ax.plot(qtraj[low_idx, 1] + 2*math.pi, qtraj[low_idx, 0], 'tab:blue', zorder=2, linewidth=2,)
    ax.plot(qtraj[high_idx, 1] - 2*math.pi, qtraj[high_idx, 0], 'tab:blue', zorder=2, linewidth=2,)
    ax.plot(qtraj[remaining_idx, 1], qtraj[remaining_idx, 0], 'tab:blue', zorder=2, linewidth=2,)
    
    if sample_trajs is not None:
        for i in range(len(sample_trajs)):
            ax.plot(sample_trajs[i][:, 1], sample_trajs[i][:, 0], 'r', zorder=2)

    # Define custom colors and their positions
    colors = [(192/255, 192/255, 192/255),
              (192/255, 192/255, 192/255),
              (192/255, 192/255, 192/255)]  # RGB colors
    positions = [0, 0.5, 1]  # Position of colors along the colormap (0 to 1)
    custom_cmap = LinearSegmentedColormap.from_list("CustomCmap", list(zip(positions, colors)))
    
    if check_diffeo:
        if hasattr(VF, 'diffeo'):
            stable_point = s2.x_to_q(VF.get_stable_point().detach().cpu()).numpy()
            if stable_point[1] < phimin:
                stable_point[1] += 2*math.pi
            if stable_point[1] > phimin+2*math.pi:
                stable_point[1] -= 2*math.pi
            ax.plot(stable_point[1], stable_point[0], 'ro', markersize=10, zorder=4)
    res = ax.streamplot(y.numpy(), x.numpy(), qdotgrid[:, :, 1].numpy(), qdotgrid[:, :, 0].numpy(),
                        density=3, color=color_grid, cmap=custom_cmap, linewidth=1, zorder=1, arrowsize=1, arrowstyle='->')
    if clim is not None:
        plt.clim(clim)
    if w == [0, 0, 1]:
        n = 50
        bound_y = np.linspace(0, math.pi, n)
        bound_x_upper = (phimin+math.pi)*np.ones(n)
        ax.plot(bound_x_upper, bound_y, color='blue', linewidth=4)
        ax.axvspan(phimin, phimin+math.pi, alpha=0.1, color='cyan')
        ax.axvspan(phimin+math.pi, phimin+2*math.pi, alpha=0.1, color='cyan')
    if 'file_name' in file_kwargs.keys():
        if file_kwargs['file_name'] is not None:
            plt.savefig(file_kwargs['file_name']+'_local', dpi=500)
            plt.close()
    
    
    if for_wandb:
        return plt
    elif for_sphere:
        return res
    else:
        plt.show()


def streamline_sphere(qtraj, *args, rot_view=0, w=[0, 0, 1], 
                      for_wandb=False, sample_trajs=None, vector=False, **kwargs):
    background_c = 'cyan'
    demo_c = 'tab:blue'
    demo_start_c = 'tab:blue'
    demo_end_c = 'tab:blue'
    sample_c = 'r'
    sample_start_c = 'r'
    point_size = 55
    linewidth = 1
    
    qtraj = qtraj.cpu()
    if 'res' in kwargs.keys():
        res = kwargs['res']
    else:
        res = streamline_local(qtraj, *args, **kwargs)
    lines = res.lines.get_paths()
    colors = res.lines.get_colors()
    fig = plt.figure(figsize=(10, 4.75))
    ax = fig.add_subplot(121, projection='3d', computed_zorder=False)
    ax.set_xlim3d(-0.65, 0.65)
    ax.set_ylim3d(-0.65, 0.65)
    ax.set_zlim3d(-0.5, 0.5)
    ax.view_init(0, 0)
    i = 0
    N = 200
    stride = 1
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride, alpha=0.1, color=background_c, zorder=1.2)
    # Get rid of the panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])

    R = s2.get_SO3_from_w_theta(w, rot_view)

    for line in lines:
        old_x = line.vertices.T[1]
        old_y = line.vertices.T[0]
        new_x = np.sin(old_x)*np.cos(old_y)
        new_y = np.sin(old_x)*np.sin(old_y)
        new_z = np.cos(old_x)
        new_xyz = R@np.vstack([new_x, new_y, new_z])
        new_x = new_xyz[0]
        new_y = new_xyz[1]
        new_z = new_xyz[2]
        if new_x[0] >= 0:
            ax.plot(new_x, new_y, new_z, color=colors[i], linewidth=linewidth, zorder=1.1) # colors[i]
            if i % 10 == 0:
                ax.quiver(new_x[0], new_y[0], new_z[0],
                          new_x[1]-new_x[0], new_y[1]-new_y[0], new_z[1]-new_z[0], # width=0.01,
                          arrow_length_ratio=5, length=0.1, color=colors[i], zorder=1,
                          ) # colors[i], arrow_length_ratio=5, length=0.1
        i += 1
    new_xt = (np.sin(qtraj[:, 0])*np.cos(qtraj[:, 1])).unsqueeze(-1)
    new_yt = (np.sin(qtraj[:, 0])*np.sin(qtraj[:, 1])).unsqueeze(-1)
    new_zt = (np.cos(qtraj[:, 0])).unsqueeze(-1)
    new_xyzt = R@np.hstack([new_xt, new_yt, new_zt]).T
    new_xt = new_xyzt[0]
    new_yt = new_xyzt[1]
    new_zt = new_xyzt[2]
    xt_plot = new_xt[new_xt >= 0]
    yt_plot = new_yt[new_xt >= 0]
    zt_plot = new_zt[new_xt >= 0]
    
    dist = 0
    last_point = None
    last_index = 0
    xplot_list, yplot_list, zplot_list = [], [], []
    for i in range(len(xt_plot)):
        point = np.array([xt_plot[i], yt_plot[i], zt_plot[i]])
        if last_point is not None:
            dist = np.sqrt(((last_point-point)**2).sum())
        last_point = point
        if dist > 0.1 or i == (len(xt_plot)-1):
            xplot_list.append(xt_plot[last_index:i])
            yplot_list.append(yt_plot[last_index:i])
            zplot_list.append(zt_plot[last_index:i])
            last_index = i
    
    for i in range(len(xplot_list)):
        ax.plot(xplot_list[i], yplot_list[i], zplot_list[i], demo_c, linewidth=2.5, zorder=4.1)
    if new_xt[0] >= 0:
        ax.scatter(new_xt[0], new_yt[0], new_zt[0], c=demo_start_c, marker='s', s=point_size+10, zorder=4.2)
    if new_xt[-1] >= 0:
        ax.scatter(new_xt[-1], new_yt[-1], new_zt[-1], c=demo_end_c, marker='o', s=point_size+15, zorder=4.3)

    ax2 = fig.add_subplot(122, projection='3d', computed_zorder=False)
    ax2.set_xlim3d(-0.65, 0.65)
    ax2.set_ylim3d(-0.65, 0.65)
    ax2.set_zlim3d(-0.5, 0.5)
    ax2.view_init(0, 0)
    i = 0
    N = 200
    stride = 1
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax2.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride, alpha=0.1, color=background_c, zorder=1.2)
    # Get rid of the panes
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax2.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax2.set_xticks([]) 
    ax2.set_yticks([]) 
    ax2.set_zticks([])

    rot_view2 = (rot_view+math.pi) % (2*math.pi)
    R2 = s2.get_SO3_from_w_theta(w, rot_view2)

    for line in lines:

        old_x = line.vertices.T[1]
        old_y = line.vertices.T[0]
        # apply for 2d to 3d transformation here
        new_x = np.sin(old_x)*np.cos(old_y)
        new_y = np.sin(old_x)*np.sin(old_y)
        new_z = np.cos(old_x)

        new_xyz = R2@np.vstack([new_x, new_y, new_z])
        new_x = new_xyz[0]
        new_y = new_xyz[1]
        new_z = new_xyz[2]

        if new_x[0] >= 0:
            ax2.plot(new_x, new_y, new_z, color=colors[i], linewidth=linewidth, zorder=1.1)
            if i % 10 == 0:
                ax2.quiver(new_x[0], new_y[0], new_z[0], 
                           new_x[1]-new_x[0], new_y[1]-new_y[0], new_z[1]-new_z[0],
                           arrow_length_ratio=5, length=0.1, color=colors[i],zorder=1,
                           )
        i += 1
    new_xt = (np.sin(qtraj[:, 0])*np.cos(qtraj[:, 1])).unsqueeze(-1)
    new_yt = (np.sin(qtraj[:, 0])*np.sin(qtraj[:, 1])).unsqueeze(-1)
    new_zt = (np.cos(qtraj[:, 0])).unsqueeze(-1)
    
    new_xyzt = R2@np.hstack([new_xt, new_yt, new_zt]).T
    new_xt = new_xyzt[0]
    new_yt = new_xyzt[1]
    new_zt = new_xyzt[2]
    xt_plot = new_xt[new_xt >= 0]
    yt_plot = new_yt[new_xt >= 0]
    zt_plot = new_zt[new_xt >= 0]
    
    dist = 0
    last_point = None
    last_index = 0
    xplot_list, yplot_list, zplot_list = [], [], []
    for i in range(len(xt_plot)):
        point = np.array([xt_plot[i], yt_plot[i], zt_plot[i]])
        if last_point is not None:
            dist = np.sqrt(((last_point-point)**2).sum())
        last_point = point
        if dist > 0.1 or i == (len(xt_plot)-1):
            xplot_list.append(xt_plot[last_index:i])
            yplot_list.append(yt_plot[last_index:i])
            zplot_list.append(zt_plot[last_index:i])
            last_index = i
    
    for i in range(len(xplot_list)):
        ax2.plot(xplot_list[i], yplot_list[i], zplot_list[i], demo_c, linewidth=2.5, zorder=4.1)
    if new_xt[0] >= 0:
        ax2.scatter(new_xt[0], new_yt[0], new_zt[0], c=demo_start_c, marker='s', s=point_size+10, zorder=4.2)
    if new_xt[-1] >= 0:
        ax2.scatter(new_xt[-1], new_yt[-1], new_zt[-1], c=demo_start_c, marker='o', s=point_size+25, zorder=4.3)
    
    if sample_trajs is not None:
        R_torch = torch.tensor(R).unsqueeze(0)
        R2_torch = torch.tensor(R2).unsqueeze(0)
        for i in range(len(sample_trajs)):
            traj = (R_torch.repeat(len(sample_trajs[i]),1,1) @ sample_trajs[i].unsqueeze(2)).squeeze()
            sample_xt = traj[:,0].detach().numpy()
            sample_yt = traj[:,1].detach().numpy()
            sample_zt = traj[:,2].detach().numpy()
            xt_plot = sample_xt[sample_xt >= 0]
            yt_plot = sample_yt[sample_xt >= 0]
            zt_plot = sample_zt[sample_xt >= 0]
            
            # check discontinuity
            dist = 0
            last_point = None
            last_index = 0
            xplot_list, yplot_list, zplot_list = [], [], []
            for i in range(len(xt_plot)):
                point = np.array([xt_plot[i], yt_plot[i], zt_plot[i]])
                if last_point is not None:
                    dist = np.sqrt(((last_point-point)**2).sum())
                last_point = point
                if dist > 0.1 or i == (len(xt_plot)-1):
                    xplot_list.append(xt_plot[last_index:i])
                    yplot_list.append(yt_plot[last_index:i])
                    zplot_list.append(zt_plot[last_index:i])
                    last_index = i
            for i in range(len(xplot_list)):
                ax.plot(xplot_list[i], yplot_list[i], zplot_list[i], sample_c, linewidth=1.5, zorder=3)
            
            if len(sample_xt[sample_xt >= 0]) > 0:
                ax.scatter(sample_xt[0], sample_yt[0], sample_zt[0], c=sample_start_c, marker='s', s=point_size*0.75, zorder=3)
            
            traj2 = (R2_torch.repeat(len(sample_trajs[i]),1,1) @ sample_trajs[i].unsqueeze(2)).squeeze()
            sample_xt = traj2[:,0].detach().numpy()
            sample_yt = traj2[:,1].detach().numpy()
            sample_zt = traj2[:,2].detach().numpy()
            xt_plot = sample_xt[sample_xt >= 0]
            yt_plot = sample_yt[sample_xt >= 0]
            zt_plot = sample_zt[sample_xt >= 0]
            
            # check discontinuity
            dist = 0
            last_point = None
            last_index = 0
            xplot_list, yplot_list, zplot_list = [], [], []
            for i in range(len(xt_plot)):
                point = np.array([xt_plot[i], yt_plot[i], zt_plot[i]])
                if last_point is not None:
                    dist = np.sqrt(((last_point-point)**2).sum())
                last_point = point
                if dist > 0.1 or i == (len(xt_plot)-1):
                    xplot_list.append(xt_plot[last_index:i])
                    yplot_list.append(yt_plot[last_index:i])
                    zplot_list.append(zt_plot[last_index:i])
                    last_index = i
            for i in range(len(xplot_list)):
                ax2.plot(xplot_list[i], yplot_list[i], zplot_list[i], sample_c, linewidth=1.5, zorder=3)
            if len(sample_xt[sample_xt >= 0]) > 0:
                ax2.scatter(sample_xt[0], sample_yt[0], sample_zt[0], c=sample_start_c, marker='s', s=point_size*0.75, zorder=3)
            
    if 'file_name' in kwargs.keys():
        if kwargs['file_name'] is not None:
            plt.savefig(kwargs['file_name']+'_sphere', dpi=500)
            plt.close()
    
    if for_wandb:
        return plt