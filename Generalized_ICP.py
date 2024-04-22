from cmath import atan, sqrt
from scipy.optimize import fmin_cg
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_tnc
from scipy.optimize import least_squares
import numpy as np
from Transformation import rot_mat, grad_rot_mat
from Point_cloud import Point_cloud
from icp_1  import *
from icp_2  import *
from scipy.misc import derivative


def best_transform(data, ref, method = "point2point", indexes_d = None, indexes_r = None, verbose = True, ref_points = None, indexes_r0 = None, dist0 = None, VOXEL_SIZE = 0.01, R0 = None, t0 = None, ref_neighs=None, sigma=0.0,  cov_data0=None, cov_ref0=None):
    """
    Returns the best transformation computed for the two aligned point clouds

    params:
        data: point cloud to align (shape n*3)
        ref: reference point cloud (shape n*3)
        method: must be one of : point2point, point2plane, plane2plane
        indexes_d: integer array Indexes and order to take into account in data
        indexes_r: integer array Indexes and order to take into account in ref
        verbose: Whether to plot the result of the iterations of conjugate gradient in plane2plane

    Returns:
        R: a rotation matrix (shape 3*3)
        t: translation (length 3 vector)
    """

    if indexes_d is None:
        indexes_d = np.arange(data.shape[0])
    if indexes_r is None:
        indexes_r = np.arange(ref.shape[0])
    data_points = data.points[indexes_d]
    if ref_points is None:
        ref_points = ref.points[indexes_r]

    assert(indexes_d.shape == indexes_r.shape)
    n = indexes_d.shape[0]
    if method == "point2point":
        x0 = np.zeros(6)
        if R0 is not None and t0 is not None:
            x0[3:] = r2euler(R0,type="XYZ")
            x0[:3] = t0
        M = np.array([np.eye(3) for i in range(n)])
        f = lambda x: loss(x,data_points,ref_points,M)
        df = lambda x: grad_loss(x,data_points,ref_points,M)

        x = x0
        x = fmin_cg(f = f,x0 = x0,fprime = df, disp = False)

        # cpt=0
        # while cpt<3:
        #     r, A = get_res_Jac(x,data_points,ref_points,M)
        #     dx = -np.linalg.inv(A.T@A) @ A.T@r
        #     x = x+dx
        #     cpt += 1

    elif method == "point2plane":
        x0 = np.zeros(6)
        if R0 is not None and t0 is not None:
            x0[3:] = r2euler(R0,type="XYZ")
            x0[:3] = t0
        if indexes_r0 is None or dist0 is None:
            M = ref.get_projection_matrix_point2plane(indexes = indexes_r, VOXEL_SIZE = 1*VOXEL_SIZE)
        else:
            M = ref.get_projection_matrix_point2plane_m(indexes = indexes_r0, dist = dist0, VOXEL_SIZE = 1*VOXEL_SIZE)
        # M = np.array([np.eye(3) for i in range(n)])
        f = lambda x: loss(x,data_points,ref_points,M)
        df = lambda x: grad_loss(x,data_points,ref_points,M)

        x = x0
        x = fmin_bfgs(f = f, x0 = x0, fprime = df, disp = False)

        # cpt=0
        # while True:
        #     r, A = get_res_Jac(x,data_points,ref_points,M)
        #     dx = -np.linalg.inv(A.T@A) @ A.T@r
        #     x = x+dx
        #     if np.sum(r) < 1e-6:
        #         break
        #     elif cpt >= 1:
        #         break
        #     else:
        #         cpt += 1

    elif method == "plane2plane":
        if cov_data0 is None:
            cov_data = data.get_covariance_matrices_plane2plane(indexes = indexes_d,  epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
            if sigma==0.0:
                cov_ref = ref.get_covariance_matrices_plane2plane(indexes = indexes_r, epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
            else:
                cov_ref = ref.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 10*VOXEL_SIZE)
                cov_ref = np.array([valued_mean(cov_ref[ref_neighs[i]], dist = dist0[i], sigma=sigma) for i in np.arange(ref_neighs.shape[0])])
            # print(cov_data.shape)
            # print(cov_ref.shape)
        else:
            cov_data = np.array([cov_data0[indexes_d]]).squeeze(axis=0)
            cov_ref = np.array([cov_ref0[indexes_r]]).squeeze(axis=0)
            # print(cov_data.shape)
            # print(cov_ref.shape)

        last_min = np.inf
        cpt = 0
        n_iter_max = 50
        x = np.zeros(6)
        if R0 is not None and t0 is not None:
            x[3:] = r2euler(R0,type="XYZ")
            x[:3] = t0
        tol = 1e-6
        while True:
            cpt = cpt+1
            R = rot_mat(x[3:])
            M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])

            f = lambda x: loss(x,data_points,ref_points,M)
            df = lambda x: grad_loss(x,data_points,ref_points,M)

            out = fmin_bfgs(f = f, x0 = x, fprime = df, disp = False, full_output = True)
            # out = fmin_bfgs(f = f, x0 = x, fprime = df, disp = False, full_output = True)
            # out = fmin_tnc(func = f, x0 = x, fprime = df, disp = False)
            # out = fmin_cg(f = loss, x0 = x, fprime = grad_loss, args = (data.points[indexes_d],ref.points[indexes_r],M), disp = False, full_output = True)

            # r, A = get_res_Jac(x,data_points,ref_points,M)
            # dx = -np.linalg.inv(A.T@A) @ A.T@r
            # x = x + dx

            x = out[0]
            f_min = out[1]
            if verbose:
                print("\t\t EM style iteration {} with loss {}".format(cpt,f_min))

            if last_min - f_min < tol:
                if verbose:
                    print("\t\t\t Stopped EM because not enough improvement or not at all")
                break
            elif cpt >= n_iter_max:
                if verbose:
                    print("\t\t\t Stopped EM because maximum number of iterations reached")
                break
            else:
                last_min = f_min
        # print(cpt)

    elif method == "slope2slope":
        if cov_data0 is None:
            cov_data = data.get_covariance_matrices_plane2plane(indexes = indexes_d,  epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
            if sigma==0.0:
                cov_ref = ref.get_covariance_matrices_plane2plane(indexes = indexes_r, epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
            else:
                cov_ref = ref.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 10*VOXEL_SIZE)
                cov_ref = np.array([valued_mean(cov_ref[ref_neighs[i]], dist = dist0[i], sigma=sigma) for i in np.arange(ref_neighs.shape[0])])
            # print(cov_data.shape)
            # print(cov_ref.shape)
        else:
            cov_data = np.array([cov_data0[indexes_d]]).squeeze(axis=0)
            cov_ref = np.array([cov_ref0[indexes_r]]).squeeze(axis=0)
            # print(cov_data.shape)
            # print(cov_ref.shape)

        data_eigenvectors = data.get_eigenvectors(radius = 10*VOXEL_SIZE)
        data_normals = data_eigenvectors[:,:,0]
        data_normals = (data_normals / np.linalg.norm(data_normals, axis = 1, keepdims = True))[indexes_d]
        # print(data_normals.shape)
        # print(indexes_d)
        ref_eigenvectors = ref.get_eigenvectors(radius = 10*VOXEL_SIZE)
        ref_normals = ref_eigenvectors[:,:,0]
        ref_normals = (ref_normals / np.linalg.norm(ref_normals, axis = 1, keepdims = True))[indexes_r]
        # print(ref_normals.shape)
        # print(indexes_r)

        last_min = np.inf
        cpt = 0
        n_iter_max = 50
        x = np.zeros(6)
        if R0 is not None and t0 is not None:
            x[3:] = r2euler(R0,type="XYZ")
            x[:3] = t0
        tol = 1e-6
        while True:
            cpt = cpt+1
            R = rot_mat(x[3:])
            M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])

            f = lambda x: loss(x,data_points,ref_points,M,data_normals,ref_normals)
            df = lambda x: grad_loss(x,data_points,ref_points,M,data_normals,ref_normals)

            # out = fmin_bfgs(f = f, x0 = x, disp = False, full_output = True)
            out = fmin_bfgs(f = f, x0 = x, fprime = df, disp = False, full_output = True)
            # out = fmin_bfgs(f = f, x0 = x, fprime = df, disp = False, full_output = True)
            # out = fmin_tnc(func = f, x0 = x, fprime = df, disp = False)
            # out = fmin_cg(f = loss, x0 = x, fprime = grad_loss, args = (data.points[indexes_d],ref.points[indexes_r],M), disp = False, full_output = True)

            # r, A = get_res_Jac(x,data_points,ref_points,M)
            # dx = -np.linalg.inv(A.T@A) @ A.T@r
            # x = x + dx

            x = out[0]
            f_min = out[1]
            if verbose:
                print("\t\t EM style iteration {} with loss {}".format(cpt,f_min))

            if last_min - f_min < tol:
                if verbose:
                    print("\t\t\t Stopped EM because not enough improvement or not at all")
                break
            elif cpt >= n_iter_max:
                if verbose:
                    print("\t\t\t Stopped EM because maximum number of iterations reached")
                break
            else:
                last_min = f_min
        # print(cpt)
    else:
        print("Error, unknown method : {}".format(method))
        return

    t = x[0:3]
    R = x[3:]

    return rot_mat(R),t

def loss(x,a,b,M, data_normals=None, ref_normals=None):
    """
    loss for parameter x

    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)

    returns:
        Value of the loss function
    """
    t = x[:3]
    R = rot_mat(x[3:])
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
    res = np.sum(residual * tmp)

    if data_normals is not None:
        S1 = np.sqrt((data_normals @ R.T)[: ,0]*(data_normals @ R.T)[: ,0] + (data_normals @ R.T)[: ,1]*(data_normals @ R.T)[: ,1])/(data_normals @ R.T)[: ,2]
        S2 = np.sqrt((ref_normals)[: ,0]*(ref_normals)[: ,0] + (ref_normals)[: ,1]*(ref_normals)[: ,1])/(ref_normals)[: ,2]
        D1 = np.arctan((data_normals @ R.T)[: ,0]/(data_normals @ R.T)[: ,1])
        D2 = np.arctan((ref_normals)[: ,0]/(ref_normals)[: ,1])
        residual1 = S1*np.cos(D1)-S2*np.cos(D2) # shape n*d
        tmp = np.sum(residual1*residual1) # shape n*d
        res += 100*tmp
        residual2 = S1*np.sin(D1)-S2*np.sin(D2) # shape n*d
        tmp = np.sum(residual2*residual2) # shape n*d
        res += 100*tmp

    return res

def grad_loss(x, a, b, M, data_normals=None, ref_normals=None):
    """
    Gradient of the loss loss for parameter x

    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)

    returns:
        Value of the gradient of the loss function
    """
    t = x[:3]
    R = rot_mat(x[3:])
    g = np.zeros(6)
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d

    g[:3] = - 2*np.sum(tmp, axis = 0)
    grad_R = - 2* (tmp.T @ a) # shape d*d
    grad_R_euler = grad_rot_mat(x[3:]) # shape 3*d*d
    g[3:] = np.sum(grad_R[None,:,:] * grad_R_euler, axis = (1,2)) # chain rule

    if data_normals is not None:
        S1 = np.sqrt((data_normals @ R.T)[: ,0]*(data_normals @ R.T)[: ,0] + (data_normals @ R.T)[: ,1]*(data_normals @ R.T)[: ,1])/(data_normals @ R.T)[: ,2]
        S2 = np.sqrt((ref_normals)[: ,0]*(ref_normals)[: ,0] + (ref_normals)[: ,1]*(ref_normals)[: ,1])/(ref_normals)[: ,2]
        D1 = np.arctan((data_normals @ R.T)[: ,0]/(data_normals @ R.T)[: ,1])
        D2 = np.arctan((ref_normals)[: ,0]/(ref_normals)[: ,1])
        residual1 = S1*np.cos(D1)-S2*np.cos(D2) # shape n*d
        residual2 = S1*np.sin(D1)-S2*np.sin(D2) # shape n*d
        n_x = (data_normals @ R.T)[: ,0]
        n_y = (data_normals @ R.T)[: ,1]
        n_z = (data_normals @ R.T)[: ,2]
        temp = np.sqrt((data_normals @ R.T)[: ,0]*(data_normals @ R.T)[: ,0] + (data_normals @ R.T)[: ,1]*(data_normals @ R.T)[: ,1])
        for i in range(S1.shape[0]):
            G_S_n = np.array([n_x[i]/n_z[i]/temp[i], n_y[i]/n_z[i]/temp[i], -temp[i]/n_z[i]/n_z[i]]) 
            G_D_n = np.array([n_y[i]/temp[i]/temp[i], -n_x[i]/temp[i]/temp[i], 0])
            afa = np.array([[data_normals[i,0], data_normals[i,1], data_normals[i,2]], [0,0,0], [0,0,0]])
            beta = np.array([[0,0,0], [data_normals[i,0], data_normals[i,1], data_normals[i,2]], [0,0,0]])
            gama = np.array([[0,0,0], [0,0,0], [data_normals[i,0], data_normals[i,1], data_normals[i,2]]])
            G_n_R = np.array([afa, beta, gama])
            grad_R1 = -2*residual1[i]*(G_S_n*np.cos(D1[i])-G_D_n*S1[i]*np.sin(D1[i])) @ G_n_R
            g[3:] +=  100 * np.sum(grad_R1[None,:,:] * grad_R_euler, axis = (1,2))
            grad_R2 = -2*residual2[i]*(G_S_n*np.sin(D1[i])+G_D_n*S1[i]*np.cos(D1[i])) @ G_n_R
            g[3:] += 100 * np.sum(grad_R2[None,:,:] * grad_R_euler, axis = (1,2))

    return g

def get_res_Jac(x, a, b, M):
    n = a.shape[0]
    r = np.zeros(n)
    A = np.zeros((n, 6))
    for i in range(n):
        r[i] = loss(x,a[None, i],b[None, i],M[None, i])
        A[i] = grad_loss(x,a[None,i],b[None,i],M[None,i])
    return r, A

def GICP(data, ref, method, exclusion_radius = 0.5, sampling_limit = None, verbose = True, R0 = None, t0 = None, VOXEL_SIZE = 0.01, k_neigh=1, B_normals=None, sigma=0.0, dataA1=None, dataB1=None, cov_data0=None, cov_ref0=None):
    """
    Full algorithm
    Aligns the two point cloud by iteratively matching the closest points
    params:
        data: point cloud to align (shape N*3)
        ref:
        method: one of point2point, point2plane, plane2plane
        exclusion_radius: threshold to discard pairs of point with too high distance
        sampling_limit: number of point to consider for huge point clouds
        verbose: whether to plot the results of the iterations and verbose of intermediate functions

    returns:
        R: rotation matrix (shape 3*3)
        T: translation (length 3)
        rms_list: list of rms at the end of each ICP iteration
    """

    data_aligned = Point_cloud()
    data_aligned.init_from_transfo(data, R0, t0)

    rms_list = []
    CD_list = []
    LCP_list = []
    cpt = 0
    max_iter = 30
    dist_threshold = exclusion_radius
    RMS_threshold = 1e-15
    diff_thresh = 1e-17
    rms = np.inf
    R = R0
    T = t0

    # coarse rmse
    data_aligned1 = Point_cloud()
    data_aligned1.init_from_transfo(dataA1, R, T)
    dist0, neighbors0 = dataB1.kdtree.query(data_aligned1.points, k=1, return_distance = True)
    dist0 = dist0.flatten()
    neighbors0 = neighbors0.flatten() 
    ref_points0 = np.array(dataB1.points[neighbors0])
    ref_points = ref_points0[dist0 < 10*dist_threshold]
    samples = np.arange(dataA1.n)
    indexes_d = samples[dist0 < 10*dist_threshold]
    diffVect = data_aligned1.points[indexes_d] - ref_points
    regis_err = np.linalg.norm(diffVect, axis=1, keepdims=True) * np.linalg.norm(diffVect, axis=1, keepdims=True)
    new_rms = np.sqrt(np.mean(regis_err))
    rms_list.append(new_rms)

    LCP_sigma = 0.2*VOXEL_SIZE
    LCP = np.sum(np.sqrt(np.sum((diffVect)**2,axis = 1))<LCP_sigma)
    LCP_list.append(LCP)

    CD = Chamfer_Distance(data, ref, R, T, 2*dist_threshold)
    CD_list.append(CD)

    print("coarse::::rmse:{}, LCP:{}, CD:{}".format(new_rms, LCP, CD))

    while(True):
        if sampling_limit is None:
            samples = np.arange(data.n)
        else:
            samples = np.random.choice(data.n,size = sampling_limit,replace = False)

        dist0, neighbors0 = ref.kdtree.query(data_aligned.points[samples], k=k_neigh, return_distance = True)
        dist1, neighbors1 = data_aligned.kdtree.query(data_aligned.points[samples], k=k_neigh, return_distance = True)
        if k_neigh > 1:
            # ref_points0 = [valued_mean(ref.points[neighbors0[i]]) for i in np.arange(neighbors0.shape[0])]
            ref_points0 = [valued_mean(ref.points[neighbors0[i]], dist = dist0[i], sigma=sigma) for i in np.arange(neighbors0.shape[0])]
            ref_points0 = np.array(ref_points0)
            dist = dist0[:, 0]
            neighbors = neighbors0[:, 0]
        else:
            ref_points0 = ref.points[neighbors0[:, 0]]
            ref_points0 = np.array(ref_points0)
            dist = dist0[:, 0]
            neighbors = neighbors0[:, 0]

        dist = dist.flatten()
        neighbors = neighbors.flatten()  # change to: k-neighbor

        dist_threshold = 10*dist_threshold

        indexes_d = samples[dist < dist_threshold]
        indexes_r = neighbors[dist < dist_threshold]
        indexes_r0 = neighbors0[dist < dist_threshold]
        ref_points = ref_points0[dist < dist_threshold] # after local mean

        dist_threshold = 0.1*dist_threshold
        # print("before:{}, after:{}".format(indexes_d.shape,samples.shape))


        if indexes_r.shape[0] <= 3:
            print(5)
            if verbose:
                print("\t Max iter reached")
            break
        initial = np.zeros(6)
        initial[:3] = r2euler(R0,type="XYZ")
        initial[3:] = t0
        R, T = best_transform(data, ref, method, indexes_d, indexes_r, verbose = verbose, ref_points = ref_points, VOXEL_SIZE = VOXEL_SIZE, R0 = R, t0 = T, dist0 = dist0, ref_neighs=neighbors0, sigma=sigma, cov_data0=cov_data0, cov_ref0=cov_ref0) #, indexes_r0 = indexes_r0, 
        # R, T = icp_1(data.points[indexes_d], ref.points[indexes_r], Rt2T(R, T))
        # R, T = icp_point_to_plane_lm(data.points[indexes_d], ref.points[indexes_r], initial, 0, B_normals)
        data_aligned.init_from_transfo(data, R, T)

        # # Accuracy_evaluation
        # new_rms = np.sqrt(np.mean(np.sum((data_aligned.points[samples]-ref.points[neighbors])**2,axis = 1)))
        # new_rms = np.sqrt(np.mean(np.sum((data_aligned.points[samples]-ref_points0)**2,axis = 1)))
        # new_rms = np.sqrt(np.mean(np.sum((data_aligned.points[indexes_d]-ref_points)**2,axis = 1)))

        # # ref_points0 = np.array([valued_mean(ref.points[neighbors0[i]], dist = dist0[i]) for i in np.arange(neighbors0.shape[0])])
        # dist0, neighbors0 = dataB1.kdtree.query(data_aligned.points, k=1, return_distance = True)
        # dist = dist0.flatten()
        # neighbors = neighbors0.flatten() 
        # ref_points0 = np.array(ref.points[neighbors])
        # ref_points = ref_points0[dist < 5*dist_threshold]
        # indexes_d = samples[dist < 10*dist_threshold]
        # diffVect = data_aligned.points[indexes_d] - ref_points

        data_aligned1 = Point_cloud()
        data_aligned1.init_from_transfo(dataA1, R, T)
        dist0, neighbors0 = dataB1.kdtree.query(data_aligned1.points, k=1, return_distance = True)
        dist0 = dist0.flatten()
        neighbors0 = neighbors0.flatten() 
        ref_points0 = np.array(dataB1.points[neighbors0])
        ref_points = ref_points0[dist0 < 10*dist_threshold]
        samples = np.arange(dataA1.n)
        indexes_d = samples[dist0 < 10*dist_threshold]
        diffVect = data_aligned1.points[indexes_d] - ref_points

        # diffVect[:,2] = diffVect[:,2]/10
        regis_err = np.linalg.norm(diffVect, axis=1, keepdims=True) * np.linalg.norm(diffVect, axis=1, keepdims=True)
        new_rms = np.sqrt(np.mean(regis_err))
        print("rms:{}".format(new_rms))
        rms_list.append(new_rms)
        
        LCP_sigma = 0.2*VOXEL_SIZE
        LCP = np.sum(np.sqrt(np.sum((diffVect)**2,axis = 1))<LCP_sigma)
        LCP_list.append(LCP)
        # print("LCP:{}".format(LCP))

        CD = Chamfer_Distance(data, ref, R, T, 2*dist_threshold)
        CD_list.append(CD)
        # print("CD:{}".format(CD))

        if verbose:
            print("Iteration {} of ICP complete with RMS : {}".format(cpt+1,new_rms))

        if new_rms < RMS_threshold :
            if verbose:
                print("\t Stopped because very low rms")
            print(1)
            break
        
        # elif rms - new_rms < 0:
        #     if verbose:
        #         print("\t Stopped because increasing rms")
        #     print(2)
        #     break
            
        elif np.abs(rms-new_rms) < diff_thresh:
            if verbose:
                print("\t Stopped because convergence of the rms")
            # print(3)
            break
            
        elif cpt >= max_iter-1:
            # print(4)
            if verbose:
                print("\t Max iter reached")
            break
        else:
            rms = new_rms
            cpt = cpt+1
    print("rmse:{}, LCP:{}, CD:{}".format(new_rms, LCP, CD))
    return R, T, rms_list

def valued_mean(ref_points, dist=None, sigma=0.0, VOXEL_SIZE = 400):
    if dist is None:
        return np.mean(ref_points, axis=0)
    # res = np.zeros((3))
    res = ref_points[0]-ref_points[0]
    sum_v = 0
    dist = (dist-np.min(dist))
    for i in range(ref_points.shape[0]):
        res += np.exp(-1.0*sigma*dist[i])*ref_points[i]
        sum_v += np.exp(-1.0*sigma*dist[i])
    res = res/sum_v

    return res

def Get_Slope(n_vec):
    S = sqrt(n_vec[0]*n_vec[0]+n_vec[1]*n_vec[1])/n_vec[2]
    D = atan(n_vec[0]/n_vec[1])
    return S, D

def Chamfer_Distance(data, ref, R, T, dist_threshold):
    data_aligned = Point_cloud()
    data_aligned.init_from_transfo(data, R, T)

    # L-R
    samples = np.arange(data.n)
    dist, neighbors = ref.kdtree.query(data_aligned.points, k=1, return_distance = True)
    dist = dist.flatten()
    neighbors = neighbors.flatten()  # change to: k-neighbor

    indexes_d = samples[dist < dist_threshold]
    indexes_r = neighbors[dist < dist_threshold]
    data_points = data_aligned.points[indexes_d]
    ref_points = ref.points[indexes_r]
    diffVector = data_points-ref_points
    # diffVector[:,2] /= 10.0

    CD = np.sum(np.sqrt(np.sum((diffVector)**2,axis = 1)))/data_points.shape[0]

    # R-L
    samples = np.arange(ref.n)
    dist, neighbors = data_aligned.kdtree.query(ref.points, k=1, return_distance = True)
    dist = dist.flatten()
    neighbors = neighbors.flatten()  # change to: k-neighbor

    indexes_r = samples[dist < dist_threshold]
    indexes_d = neighbors[dist < dist_threshold]
    data_points = data_aligned.points[indexes_d]
    ref_points = ref.points[indexes_r]
    diffVector = data_points-ref_points
    # diffVector[:,2] /= 10.0
    CD += np.sum(np.sqrt(np.sum((diffVector)**2,axis = 1)))/data_points.shape[0]

    return CD

