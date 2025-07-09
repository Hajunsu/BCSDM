import numpy as np
from scipy.linalg import logm

def define_SE3(R, p):
    SE3 = np.identity(4)
    SE3[0:3, 0:3] = R
    SE3[0:3, 3] = p
    return SE3

def get_SO3(SE3):
    return SE3[0:3, 0:3]

def get_p(SE3):
    return SE3[0:3, 3]

def change_SO3(SE3, R):
    SE3[0:3, 0:3] = R
    return SE3

def change_p(SE3, p):
    SE3[0:3, 3] = p
    return SE3

def inverse_SE3(SE3):
    R = np.transpose(get_SO3(SE3))
    p = - np.dot(R, get_p(SE3))
    inv_SE3 = define_SE3(R, p)
    return inv_SE3

def transform_point(SE3, p):
    p_h = np.ones(4)
    p_h[0:3] = p
    p_h = np.dot(SE3, p_h)
    return p_h[0:3]

# skew matrix
def skew(w):
    W = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]])
    
    return W

# SO3 exponential
def exp_so3(w):
    if len(w) != 3:
        raise ValueError('Dimension is not 3')
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        R = np.eye(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        R = np.eye(3) + sw * wnorm_inv * W + (1 - cw) * np.power(wnorm_inv,2) * W.dot(W)

    return R

# ------------
def omega_to_SO3(w, theta):
    skew_w = skew(w)
    R = np.eye(3) + skew_w.dot(np.sin(theta)) + skew_w.dot(skew_w).dot(1 - np.cos(theta))
    return R


def screw_to_SE3(screw, theta):
    w = screw[0:3]
    v = screw[3:6]

    if np.linalg.norm(w) < 1e-20:   # w == 0, |v| == 1
        return define_SE3(np.identity(3), v * theta)
    
    else:     # |w| != 0
        if (np.linalg.norm(w)-1) > 1e-8: # |w| != 1
            v = v / np.linalg.norm(w)
            theta *= np.linalg.norm(w)
            w = w / np.linalg.norm(w)  # should be normalized (|w| == 1)
            
        skew_w = skew(w)
        G = np.eye((3)).dot(theta) + skew_w.dot(1 - np.cos(theta)) + skew_w.dot(skew_w).dot(theta - np.sin(theta))
        return define_SE3(omega_to_SO3(w, theta), G.dot(v))

def Adjoint_T(T):
    R = get_SO3(T)
    p = get_p(T)
    skew_p = skew(p)
    
    AdT = np.eye(6)
    AdT[0:3][:, 0:3] = R
    AdT[0:3][:, 3:6] = np.zeros((3,3))
    AdT[3:6][:, 0:3] = skew_p.dot(R)
    AdT[3:6][:, 3:6] = R

    return AdT

def Jac_s(screws, q):
    Js = np.zeros((6, screws.shape[1]))
    M = np.eye(4)
    for joint in range(screws.shape[1]):
        Js[:, joint] = Adjoint_T(M).dot(screws[:, joint])
        M = M.dot(screw_to_SE3(screws[:, joint], q[joint]))
    return Js

def screws_to_SE3(screws, q):
    T = np.eye(4)
    for joint in range(screws.shape[1]):
        T = T.dot(screw_to_SE3(screws[:, joint], q[joint]))
    return T

def forward_kinematics(screws, q, M_EF):
    T = screws_to_SE3(screws, q)
    return T.dot(M_EF)

def forward_kinematics_grasp_target_list(qtraj):
    # grasp target initial SE3
    initial_SE3 = [[0.7071, 0.7071, 0, 0.088],
                   [0.7071, -0.7071, 0, 0],
                   [0, 0, -1,  0.8210],
                   [0, 0, 0, 1]]
    S = franka_screws_zero()
    
    Ttraj = []
    for i in range(len(qtraj)):
        T = forward_kinematics(S, qtraj[i], np.array(initial_SE3))
        Ttraj.append(np.expand_dims(T, axis=0))
    Ttraj = np.concatenate(Ttraj, axis=0)
    return Ttraj

def forward_kinematics_grasp_target(q):
    # grasp target initial SE3
    initial_SE3 = [[0.7071, 0.7071, 0, 0.088],
                   [0.7071, -0.7071, 0, 0],
                   [0, 0, -1,  0.8210],
                   [0, 0, 0, 1]]
    S = franka_screws_zero()
    T = forward_kinematics(S, q, np.array(initial_SE3))
    return T

def inverse_kinematics(screws, q_init, M_EF, T_desired, joint_limit):
    # Stepsize
    k = 0.03

    # Tolerance for stopping iteration
    Tolerance = 0.0001

    # Initialization
    q = q_init
    T = screws_to_SE3(screws, q).dot(M_EF)

    iter = 1
    V = np.zeros((6,1))
    # Iteration algorithm
    while True: # np.linalg.norm(T - T_desired, 'fro') > Tolerance:
        K = logm(T_desired.dot(inverse_SE3(T)))
        # print(T_desired)
        V[0:3, 0] = [K[2, 1], K[0, 2], K[1, 0]]
        V[3:6, 0] = K[0:3, 3]
        delta_q = k * np.squeeze(np.linalg.pinv(Jac_s(screws, q)).dot(V))
        q = q + delta_q
        q = joint_limit_check(joint_limit, q)
        T = screws_to_SE3(screws, q).dot(M_EF)
        iter += 1
        if np.linalg.norm(V) < 1e-8:
            break
        if iter > 5000:
            print('finished without solution')
            return None

    # Simplify results' value to [-pi, pi]
    q = np.mod(q, 2 * np.pi)
    for joint in range(q.shape[0]):
          if q[joint] > np.pi:
               q[joint] -= 2 * np.pi

    return q

def modify_joint_value(q_init, q_final):
    for joint in range(q_init.shape[0]):
        if np.abs(q_final[joint]-q_init[joint]) > np.pi :
            q_final[joint] -= 2 * np.pi

    return q_final

def joint_limit_check(joint_limit, q):
    for joint in range(len(q)):
        if q[joint] < joint_limit['lower_bound'][joint]:
            n = np.int64((joint_limit['lower_bound'][joint] - q[joint]) / (2 * np.pi)) + 1
            q[joint] += 2 * np.pi * n

        elif joint_limit['lower_bound'][joint] <= q[joint] <= joint_limit['upper_bound'][joint]:
            pass

        else:
            n= np.int64((q[joint] - joint_limit['upper_bound'][joint]) / (2 * np.pi)) + 1
            q[joint] -= 2 * np.pi * n

    return q

def q_refine(joint_limit, q):
    mp = np.mean((joint_limit['upper_bound'], joint_limit['lower_bound']), axis=0)
    for joint in range(len(q)): 
        dir = q[joint] - mp[joint]
        dir /= abs(dir)
        joint_cand = q[joint] - 2* np.pi * dir
        if dir < 0 and joint_limit['upper_bound'][joint] - joint_cand > q[joint] - joint_limit['lower_bound'][joint]:
               q[joint] = joint_cand
        if dir > 0 and joint_cand-joint_limit['lower_bound'][joint] > joint_limit['upper_bound'][joint] - q[joint]:
               q[joint] = joint_cand

    return q

def franka_screws_zero():
    w_1 = np.array([0.0, 0.0, 1.0])
    w_2 = np.array([0.0, 1.0, 0.0])
    w_3 = np.array([0.0, 0.0, 1.0])
    w_4 = np.array([0.0, -1.0, 0.0])	
    w_5 = np.array([0.0, 0.0, 1.0])
    w_6 = np.array([0.0, -1.0, 0.0])
    w_7 = np.array([0.0, 0.0, -1.0])

    p_1 = np.array([0.0, 0.0, 0.333])
    p_2 = np.array([0.0, 0.0, 0.0])
    p_3 = np.array([0.0, 0.0, 0.316])
    p_4 = np.array([0.0825, 0.0, 0.0])
    p_5 = np.array([-0.0825, 0.0, 0.384])
    p_6 = np.array([0.0, 0.0, 0.0])
    p_7 = np.array([0.088, 0.0, 0.0])

    q_1 = p_1
    q_2 = p_1 + p_2
    q_3 = p_1 + p_2 + p_3
    q_4 = p_1 + p_2 + p_3 + p_4
    q_5 = p_1 + p_2 + p_3 + p_4 + p_5
    q_6 = p_1 + p_2 + p_3 + p_4 + p_5 + p_6
    q_7 = q_6 + p_7

    S = np.zeros((6,7))
    S[:,0] = np.concatenate((w_1, -np.cross(w_1, q_1)))
    S[:,1] = np.concatenate((w_2, -np.cross(w_2, q_2)))
    S[:,2] = np.concatenate((w_3, -np.cross(w_3, q_3)))
    S[:,3] = np.concatenate((w_4, -np.cross(w_4, q_4)))
    S[:,4] = np.concatenate((w_5, -np.cross(w_5, q_5)))
    S[:,5] = np.concatenate((w_6, -np.cross(w_6, q_6)))
    S[:,6] = np.concatenate((w_7, -np.cross(w_7, q_7)))
    
    return S
