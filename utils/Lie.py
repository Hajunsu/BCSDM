import math
import torch
dtype = torch.float


def skew(w):
    # input = n,3
    n = w.shape[0]
    # 3x3 skew --> vector
    if w.shape == (n, 3, 3):
        W = torch.cat([-w[:, 1, 2].unsqueeze(-1), w[:, 0, 2].unsqueeze(-1),
                       -w[:, 0, 1].unsqueeze(-1)], dim=1)
    # 3 dim vector --> skew
    else:
        zero1 = torch.zeros(n, 1, 1).to(w)
        w = w.unsqueeze(-1).unsqueeze(-1)
        W = torch.cat([torch.cat([zero1, -w[:,2],  w[:, 1]], dim=2),
                       torch.cat([w[:, 2], zero1,  -w[:, 0]], dim=2),
                       torch.cat([-w[:, 1], w[:, 0],  zero1], dim=2)], dim=1)
    return W


def screw_bracket(V):
    if isinstance(V, str):
        # print(V)
        return 'trace error'
    n = V.shape[0]
    out = 0
    if V.shape == (n, 4, 4):
        out = torch.cat([-V[:, 1, 2].unsqueeze(-1), V[:, 0, 2].unsqueeze(-1),
                         -V[:, 0, 1].unsqueeze(-1), V[:, :3, 3]], dim=1)
    else:
        W = skew(V[:, 0:3])
        out = torch.cat([torch.cat([W, V[:, 3:].unsqueeze(-1)], dim=2),
                         torch.zeros(n, 1, 4).to(V)], dim=1)
        # print(torch.cat([W, V[:, 0:3].unsqueeze(-1)], dim=2))
    return out


def skew_so3(so3):
    nBatch = len(so3)
    if so3.shape == (nBatch, 3, 3):
        return torch.cat([-so3[:, 1, 2].unsqueeze(-1),
                          so3[:, 0, 2].unsqueeze(-1),
                          -so3[:, 0, 1].unsqueeze(-1)], dim=1)
    elif so3.numel() == nBatch * 3:
        w = so3.reshape(nBatch, 3, 1, 1)
        zeroBatch = so3.new_zeros(nBatch, 1, 1)
        output = torch.cat([torch.cat([zeroBatch, -w[:, 2], w[:, 1]], dim=2),
                            torch.cat([w[:, 2], zeroBatch, -w[:, 0]], dim=2),
                            torch.cat([-w[:, 1], w[:, 0], zeroBatch], dim=2)], dim=1)
        return output
    else:
        print(f'ERROR : skew_so3, so3.shape = {so3.shape}')
        exit(1)


def skew_se3(se3):
    nBatch = len(se3)
    if se3.shape == (nBatch, 4, 4):
        output = se3.new_zeros(nBatch, 6)
        output[:, :3] = skew_so3(se3[:, :3, :3])
        output[:, 3:] = se3[:, :3, 3]
        return output
    elif se3.numel() == nBatch * 6:
        se3_ = se3.reshape(nBatch, 6)
        output = se3_.new_zeros(nBatch, 4, 4)
        output[:, :3, :3] = skew_so3(se3_[:, :3])
        output[:, :3, 3] = se3_[:, 3:]
        return output
    else:
        print(f'ERROR : skew_se3, se3.shape = {se3.shape}')
        exit(1)
        

def exp_so3(Input):
    # shape(w) = (3,1)
    # shape(W) = (3,3)
    n = Input.shape[0]
    if Input.shape==(n,3,3):
        W = Input
        w = skew(Input)
    else:
        w = Input
        W = skew(w)
    wnorm_sq = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)\
    wnorm_sq_unsqueezed = wnorm_sq.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1)
    
    wnorm = torch.sqrt(wnorm_sq)  # (dim = n)
    wnorm_unsqueezed = torch.sqrt(wnorm_sq_unsqueezed)  # dim - (n,1)
    
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    w0 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
    w1 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
    w2 = w[:, 2].unsqueeze(-1).unsqueeze(-1)
    eps = 1e-4
    # R = torch.zeros(n,3,3)
    R = torch.cat((torch.cat((cw - ((w0**2)*(cw - 1))/wnorm_sq_unsqueezed,
                              - (w2*sw)/wnorm_unsqueezed - (w0*w1*(cw - 1))/wnorm_sq_unsqueezed,
                              (w1*sw)/wnorm_unsqueezed - (w0*w2*(cw - 1))/wnorm_sq_unsqueezed), dim=2)
                   , torch.cat(((w2*sw)/wnorm_unsqueezed - (w0*w1*(cw - 1))/wnorm_sq_unsqueezed,
                                cw - ((w1**2)*(cw - 1))/wnorm_sq_unsqueezed,
                               - (w0*sw)/wnorm_unsqueezed - (w1*w2*(cw - 1))/wnorm_sq_unsqueezed), dim=2)
                   , torch.cat((-(w1*sw)/wnorm_unsqueezed - (w0*w2*(cw - 1))/wnorm_sq_unsqueezed,
                                (w0*sw)/wnorm_unsqueezed - (w1*w2*(cw - 1))/wnorm_sq_unsqueezed,
                                cw - ((w2**2)*(cw - 1))/wnorm_sq_unsqueezed), dim=2))
                  , dim=1)
    R[wnorm < eps] = torch.eye(3).to(Input) + W[wnorm < eps] + 1/2*W[wnorm < eps]@W[wnorm < eps]
                              
    return R


def exp_so3_from_screw(S):
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3

    T = torch.cat([torch.cat([exp_so3(w), v], dim=2),
                   torch.zeros(n, 1, 4).to(S)], dim=1)
    T[:, -1, -1] = 1
    return T


def exp_so3_T(S):
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3

    T = torch.cat([torch.cat([exp_so3(w), v], dim=2), (torch.zeros(n, 1, 4, device=S.device))], dim=1)
    T[:, -1, -1] = 1
    return T


def Dexp_so3(w):
    R = exp_so3(w)
    N = w.shape[0]
    Id = torch.eye(3).to(w)
    dRdw = torch.zeros(N, 3, 3, 3).to(w)
    wnorm = torch.sqrt(torch.einsum('ni,ni->n', w, w))
    eps = 1e-5
    e_skew = skew(Id)
    if w.shape == (N, 3):
        W = skew(w)
    else:
        W = w
        w = skew(W)
        assert (False)
    temp1 = torch.einsum('ni,njk->nijk', w, W)
    temp2 = torch.einsum('njk,nki->nij', W, (-R + torch.eye(3).to(w)))
    temp2_2 = skew(temp2.reshape(N * 3, 3)).reshape(N, 3, 3, 3)
    wnorm_square = torch.einsum('ni,ni->n', w, w).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    dRdw[wnorm > eps] = (((temp1 + temp2_2) / wnorm_square) @ R.unsqueeze(1))[wnorm > eps]
    dRdw[wnorm < eps] = e_skew

    return dRdw


def exp_se3(S):
    n = S.shape[0]
    if S.shape==(n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3
    wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1)
    wnorm = torch.sqrt(wsqr)  # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  # dim - (n,1)
    wnorm_inv = 1/wnorm_unsqueezed # dim = (n)
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    
    eps = 1e-014
    W = skew(w)
    P = torch.eye(3).to(S) + (1-cw)*(wnorm_inv**2)*W + \
        (wnorm_unsqueezed - sw)*(wnorm_inv**3)*torch.matmul(W, W)  # n,3,3
    P[wnorm < eps] = torch.eye(3).to(S)
    T = torch.cat([torch.cat([exp_so3(w), P@v], dim=2), (torch.zeros(n, 1, 4).to(S))], dim=1)
    T[:, -1, -1] = 1
    return T


def inverse_SE3(T):
    n = T.shape[0]
    R = T[:, 0:3, 0:3]  # n,3,3
    p = T[:, 0:3, 3].unsqueeze(-1)  # n,3,1
    T_inv = torch.cat([torch.cat([R.transpose(1, 2), (-R.transpose(1, 2))@p], dim=2),
                       torch.zeros(n, 1, 4).to(T)], dim=1)
    T_inv[:, -1, -1] = 1
    return T_inv


def large_Ad(T):
    n = T.shape[0]
    R = T[:, 0:3, 0:3]  # n,3,3
    p = T[:, 0:3, 3]  # n,3
    AdT = torch.cat([torch.cat([R, torch.zeros(n, 3, 3).to(T)], dim=2),
                     torch.cat([skew(p)@R, R], dim=2)], dim=1)
    return AdT


def small_ad(V):
    # shape(V) = (n,6)
    n = V.shape[0]
    w = V[:, :3]
    v = V[:, 3:]
    wskew = skew(w)
    vskew = skew(v)
    adV = torch.cat([torch.cat([wskew, torch.zeros(n, 3, 3).to(V)], dim=2),
                     torch.cat([vskew, wskew], dim=2)], dim=1)
    return adV


def Lie_bracket(u, v):
    if u.shape[1:] == (3,):
        u = skew(u)
    elif u.shape[1:] == (6,):
        u = screw_bracket(u)

    if v.shape[1:] == (3,):
        v = skew(v)
    elif v.shape[1:] == (6,):
        v = screw_bracket(v)

    return u @ v - v @ u


def log_SO3(R):
    eps = 1e-4
    
    trace = torch.sum(R[:, range(3), range(3)], dim=1).to(R)
    omega = torch.zeros(R.shape).to(R)
    theta = torch.acos(torch.clip((trace - 1) / 2, -1, 1)).to(R)
    temp = theta.unsqueeze(-1).unsqueeze(-1).to(R)

    omega[(torch.abs(trace + 1) > eps) * (theta > eps)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(1, 2)))[
        (torch.abs(trace + 1) > eps) * (theta > eps)]

    omega_temp = (R[torch.abs(trace + 1) <= eps] - torch.eye(3).to(R)) / 2

    omega_vector_temp = torch.sqrt((omega_temp[:, range(3), range(3)] + torch.ones(3).to(R)).clip(min=0))
    
    A = omega_vector_temp[:, 1] * torch.sign(omega_temp[:, 0, 1])
    B = omega_vector_temp[:, 2] * torch.sign(omega_temp[:, 0, 2])
    C = omega_vector_temp[:, 0]
    omega_vector = torch.cat([C.unsqueeze(1), A.unsqueeze(1), B.unsqueeze(1)], dim=1)
    omega[torch.abs(trace + 1) <= eps] = skew(omega_vector) * math.pi

    return omega


def log_SO3_T(T):
    # dim T = n,4,4
    R = T[:, 0:3, 0:3]  # dim n,3,3
    p = T[:, 0:3, 3].unsqueeze(-1)  # dim n,3,1
    n = T.shape[0]
    W = log_SO3(R.to(T))  # n,3,3

    return torch.cat([torch.cat([W, p], dim=2), torch.zeros(n, 1, 4).to(T)], dim=1)  # n,4,4


def log_SE3(T):
    #dim T = n,4,4
    R = T[:,0:3,0:3] # dim n,3,3
    p = T[:,0:3,3].unsqueeze(-1) # dim n,3,1
    n = T.shape[0]
    W = log_SO3(R) #n,3,3
    w = skew(W) #n,3
    
    wsqr = torch.tensordot(w,w, dims=([1],[1]))[[range(n),range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1) # dim = (n,1)
    wnorm = torch.sqrt(wsqr) # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed) # dim - (n,1)
    wnorm_inv = 1/wnorm_unsqueezed # dim = (n)
    cw = torch.cos(wnorm).view(-1,1).unsqueeze(-1) # (dim = n,1)
    sw = torch.sin(wnorm).view(-1,1).unsqueeze(-1) # (dim = n,1)
    
    P = torch.eye(3).to(T) + (1-cw)*(wnorm_inv**2)*W + (wnorm_unsqueezed - sw) * (wnorm_inv**3) * torch.matmul(W,W) #n,3,3
    v = torch.inverse(P)@p #n,3,1
    return torch.cat([torch.cat([W,v],dim=2),torch.zeros(n,1,4).to(T)],dim=1)


def SO3_to_quaternion(R):
    #dim(R) = n,3,3
    W = log_SO3(R) # n,3,3
    w = skew(W) #n,3
    theta_1dim = torch.sqrt(torch.sum(w**2,dim=1))
    theta = theta_1dim.unsqueeze(-1) # n,1
    w_hat = w/theta # n,3
    w_hat[theta_1dim<1.0e-016] = 0
    return torch.cat([w_hat[:,0].unsqueeze(-1)*torch.sin(theta/2),
                      w_hat[:,1].unsqueeze(-1)*torch.sin(theta/2),
                      w_hat[:,2].unsqueeze(-1)*torch.sin(theta/2),
                      torch.cos(theta/2)], dim=1)


def quaternion_to_SO3(quaternion):
    # input : quaternion(x,y,z,w)
    assert quaternion.shape[1] == 4

    # initialize
    K = quaternion.shape[0]
    R = quaternion.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    x = quaternion[:, 0]
    y = quaternion[:, 1]
    z = quaternion[:, 2]
    w = quaternion[:, 3]
    xx = x ** 2
    yy = y ** 2
    zz = z ** 2
    ww = w ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternion.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * x * y
    xz = s[:, 0] * x * z
    yz = s[:, 0] * y * z
    xw = s[:, 0] * x * w
    yw = s[:, 0] * y * w
    zw = s[:, 0] * z * w

    xx = s[:, 0] * xx
    yy = s[:, 0] * yy
    zz = s[:, 0] * zz

    idxs = torch.arange(K).to(quaternion.device)
    R[idxs, 0, 0] = 1 - yy - zz
    R[idxs, 0, 1] = xy - zw
    R[idxs, 0, 2] = xz + yw

    R[idxs, 1, 0] = xy + zw
    R[idxs, 1, 1] = 1 - xx - zz
    R[idxs, 1, 2] = yz - xw

    R[idxs, 2, 0] = xz - yw
    R[idxs, 2, 1] = yz + xw
    R[idxs, 2, 2] = 1 - xx - yy

    return R