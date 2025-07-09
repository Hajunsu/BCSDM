
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

dtype = torch.float
device_c = torch.device("cpu")
device = device_c
torch.backends.cudnn.deterministic = True

aut = cm.get_cmap('autumn', 128 * 2.5)
new_aut  = ListedColormap(aut(range(256)))


def R2_streamline_plot(
    vf_model,
    xtraj, xdottraj, eta, 
    grid_step = 101, density=1.2, ax=None, figsize=(10,10),
    xlim=None, mec=False):
    # set
    eps = 1e-3
    
    # make figure
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    # make grid
    if xlim is None:
        xmin, _ = torch.min(xtraj, axis=0)
        xmax, _ = torch.max(xtraj, axis=0)
        xlength = torch.norm(xmax - xmin)
        offset = xlength / 4
        xlim = [[xmin[0]-offset, xmin[1]-offset],
                [xmax[0]+offset, xmax[1]+offset]]
    x1_linspace = torch.linspace(xlim[0][0], xlim[1][0], grid_step)
    x2_linspace = torch.linspace(xlim[0][1], xlim[1][1], grid_step)
    x1, x2 = torch.meshgrid(x1_linspace, x2_linspace)    
    xmesh = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], dim=-1)
    xmesh_long = xmesh.reshape(-1, 2).to(xtraj)
    
    # get vector field
    xtraj_repeat = xtraj.reshape(1, -1).repeat(len(xmesh_long), 1)
    xdot_long = vf_model(x=xmesh_long, eta=eta, xtraj=xtraj_repeat)
    xdot_grid = xdot_long.cpu().detach().reshape(grid_step, grid_step, 2)
    
    c = 'tab:blue'
    if mec is True:
        ax.plot(xtraj[0, 0], xtraj[0, 1], 's', color=c, markersize=18, zorder=20, mec='k')
        ax.plot(xtraj[-1, 0], xtraj[-1, 1], 'o', color=c, markersize=18, zorder=21, mec='k')
    else:
        ax.plot(xtraj[0, 0], xtraj[0, 1], 's', color=c, markersize=18, zorder=20)
        ax.plot(xtraj[-1, 0], xtraj[-1, 1], 'o', color=c, markersize=18, zorder=21)
    ax.plot(xtraj[:, 0], xtraj[:, 1], c, zorder=4, linewidth=5)
    
    res = ax.streamplot(x1.T.numpy(), x2.T.numpy(), xdot_grid[:, :, 0].T.numpy(), xdot_grid[:, :, 1].T.numpy(),
                        density=density, color='silver', linewidth=2, cmap=new_aut, zorder=1, arrowsize=2.5, arrowstyle='->') #cmap=color_map)
    ax.axis('equal')
    return ax


def R2_solution_trajs(
    vf_model, xtraj, xdottraj, eta, ax=None, x0_list=[], figsize=(10,10), len_traj=1000, dt=0.1):
    # make figure
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    for x0 in x0_list:
        x_prev = x0.reshape(1, 2)
        x_traj = []
        x_traj.append(x_prev)
        for i in range(len_traj):
            xtraj_repeat = xtraj.reshape(1, -1).repeat(len(x_prev), 1)
            xdot = vf_model(x=x_prev, eta=eta, xtraj=xtraj_repeat)
            x = x_prev + xdot * dt
            x_prev = x.to(torch.float)
            x_traj.append(x_prev)
        x_traj = torch.cat(x_traj, dim=0).detach().numpy()
        ax.plot(x_traj[0, 0], x_traj[0, 1], 's', color='r', markersize=10, zorder=25)
        ax.plot(x_traj[:, 0], x_traj[:, 1], 'r', zorder=5, linewidth=3)
        
    return ax


def get_grad(xmesh, xdot_grid, dim, norm=False, direction=False):
    if direction:
        xdot_grid = xdot_grid / torch.norm(xdot_grid, dim=-1, keepdim=True)
    dx = xmesh[1, 0, 0] - xmesh[0, 0, 0]
    dy = xmesh[0, 1, 1] - xmesh[0, 0, 1]
    grad_mesh = torch.zeros_like(xdot_grid)
    if dim == 0:
        grad_mesh[0] = (-3 * xdot_grid[0] + 4*xdot_grid[1] - xdot_grid[2]) / (2*dx)
        grad_mesh[-1] = (3 * xdot_grid[-1] - 4*xdot_grid[-2] + xdot_grid[-3]) / (2*dx)
        grad_mesh[1:-1] = (xdot_grid[2:] - xdot_grid[:-2]) / (2*dx)
    elif dim == 1:
        grad_mesh[:, 0] = (-3 * xdot_grid[:, 0] + 4*xdot_grid[:, 1] - xdot_grid[:, 2]) / (2*dy)
        grad_mesh[:, -1] = (3 * xdot_grid[:, -1] - 4*xdot_grid[:, -2] + xdot_grid[:, -3]) / (2*dy)
        grad_mesh[:, 1:-1] = (xdot_grid[:, 2:] - xdot_grid[:, :-2]) / (2*dy)
    if norm:
        grad_mesh_norm = torch.norm(grad_mesh, dim=-1)
        return grad_mesh_norm
    else:
        return grad_mesh


def get_discontiuous_region(xmesh, xdot_grid, threshold=25):
    grad_x = get_grad(xmesh, xdot_grid, 0, norm=False).unsqueeze(-2)
    grad_y = get_grad(xmesh, xdot_grid, 1, norm=False).unsqueeze(-2)
    grad_all = torch.cat([grad_x, grad_y], dim=-2)
    grad_norm = torch.norm(grad_all, dim=(-1, -2))
    mask = grad_norm > threshold
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    return mask


def paint_discontinuous_region(
    cvf_model, ax, xtraj, xdottraj, eta, xlim=None, grid_step=301, threshold=25, linewidth=3, color='g', s=10, alpha=1):
    # make grid
    if xlim is None:
        xmin, _ = torch.min(xtraj, axis=0)
        xmax, _ = torch.max(xtraj, axis=0)
        xlength = torch.norm(xmax - xmin)
        offset = xlength / 4
        xlim = [[xmin[0]-offset, xmin[1]-offset],
                [xmax[0]+offset, xmax[1]+offset]]
    x1_linspace = torch.linspace(xlim[0][0], xlim[1][0], grid_step)
    x2_linspace = torch.linspace(xlim[0][1], xlim[1][1], grid_step)
    x1, x2 = torch.meshgrid(x1_linspace, x2_linspace)    
    xmesh = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], dim=-1)
    xmesh_long = xmesh.reshape(-1, 2).to(xtraj)
    
    # get vector field
    xdot_long = cvf_model(xmesh_long)
    xdot_grid = xdot_long.cpu().reshape(grid_step, grid_step, 2)
    
    mask = get_discontiuous_region(xmesh, xdot_grid, threshold)
    mask = mask.cpu().numpy()
    discontinuous_xmesh = xmesh[mask]
    ax.scatter(discontinuous_xmesh[:, 0], discontinuous_xmesh[:, 1], c=color, s=s, linewidth=linewidth, alpha=alpha)
    return ax, xmesh, xdot_grid, mask