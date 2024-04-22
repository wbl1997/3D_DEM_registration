#coding:utf-8
from matplotlib.font_manager import FontProperties
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from scipy.spatial import cKDTree
import copy
import math
import os

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 20
  pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
  # print(np.mean(pcd2xyz(pcd),axis=1))
  # pcd.orient_normals_towards_camera_location(np.mean(pcd2xyz(pcd),axis=1))
  # pcd.orient_normals_to_align_with_direction((0,0,0))
  pcd.orient_normals_towards_camera_location((0,0,10000000000))
  # o3d.visualization.draw_geometries([pcd],point_show_normal=True)

  radius_feature = voxel_size * 50
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=1000))
  return np.array(fpfh.data).T

def get_iss_keypoints(pcd, pcd_raw):
  keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd_raw)
  pcd_tree = o3d.geometry.KDTreeFlann(pcd)
  index = []
  kps = []
  for i in range(np.asarray(keypoints.points).shape[0]):
    [_, idx, _] = pcd_tree.search_knn_vector_3d(keypoints.points[i], 1)
    index.append(idx[0])
    kps.append(np.asarray(pcd.points[idx[0]]))
  print(len(index))
  # keypoints = pcd.select_by_index(index)
  # keypoints.points = o3d.utility.Vector3dVector(np.array(kps))
  return index, keypoints

def get_iss_keypoints1(pcd, voxel_size = 0.01):
  keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd, salient_radius = 1 * voxel_size,
                                                        non_max_radius = 1 * voxel_size)
#   pcd_tree = o3d.geometry.KDTreeFlann(pcd)
  index = []
  for i in range(np.asarray(keypoints.points).shape[0]):
    [_, idx, _] = pcd_tree.search_knn_vector_3d(keypoints.points[i], 1)
    index.append(idx[0])
  print(len(index))
  return index, keypoints

def get_iss_fpfh(pcd, pcd_raw, voxel_size=0.01):
  # get iss points
  keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd_raw, salient_radius = 0.2 * voxel_size,
                                                        non_max_radius = 0.2 * voxel_size)
  o3d.visualization.draw_geometries([pcd_raw])
  # merge raw_keypoints and pcd
  pcd = pcd + keypoints
  # get index
  pcd_tree = o3d.geometry.KDTreeFlann(pcd)
  index = []
  for i in range(np.asarray(keypoints.points).shape[0]):
    [_, idx, _] = pcd_tree.search_knn_vector_3d(keypoints.points[i], 1)
    index.append(idx[0])
  # print(len(index))

  # calculate fpfh
  feats = extract_fpfh(pcd, 1*voxel_size)
  feats = feats[index, :]
  normals = np.asarray(pcd.normals)[index]

  return keypoints, feats, normals


def preprocess_point_cloud(pcd, voxel_size=0.01):
    "preprocess point cloud"
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def find_correspondences1(feats0, feats1, mutual_filter=True):
    nns01, dist01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=True)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    dist_thresh = 1.5*np.mean(dist01)
    corres01_idx0 = corres01_idx0[dist01 < dist_thresh]
    corres01_idx1 = corres01_idx1[dist01 < dist_thresh]

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10, dist10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=True)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    # corres10_idx1 = corres10_idx1[dist10 < dist_thresh]
    # corres10_idx0 = corres10_idx0[dist10 < dist_thresh]

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0) 
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def find_correspondences2(feats0, feats1, voxel_size, mutual_filter=True):
    nns01,dist01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=True)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10,dist10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=True)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0) 

    for i in range(nns01.size):
        mutual_filter[i] = mutual_filter[i] and  dist01[i] < voxel_size * 300
        #mutual_filter[i] = dist01[i] < voxel_size * 100
    print(nns01.size)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def execute_global_registration(source_down, target_down, source_fpfh_d,
                               target_fpfh_d, voxel_size):
  distance_threshold = voxel_size * 1.0
  #print(":: RANSAC registration on downsampled point clouds.")
  #print("   Since the downsampling voxel size is %.3f," % voxel_size)
  #print("   we use a liberal distance threshold %.3f." % distance_threshold)
  source_fpfh = o3d.pipelines.registration.Feature()
  source_fpfh.data = source_fpfh_d[:,:].T
  target_fpfh = o3d.pipelines.registration.Feature()
  target_fpfh.data = target_fpfh_d[:,:].T
  ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                      source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
                      o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, 
                      # [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                      # o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
                      [],
                      o3d.pipelines.registration.RANSACConvergenceCriteria(8000000, 500))
  return ransac_result

def draw_inliner(A_pcd, B_pcd, A_corr, B_corr):
    # visualize the point clouds together with feature correspondences
    A_pcd_txy =  copy.deepcopy(A_pcd)
    # B_pcd_txy =  copy.deepcopy(B_pcd).translate(+np.mean(B_corr, axis=0))
    # points = np.concatenate((A_corr, B_corr+B_corr.mean(axis=0)), axis=0)
    B_pcd_txy =  copy.deepcopy(B_pcd).translate(+np.array([0, 0.4, 0]))
    points = np.concatenate((A_corr, B_corr+np.array([0, 0.4, 0])), axis=0)
    lines = []
    num_corrs = A_corr.shape[0]
    for i in range(num_corrs):
        lines.append([i, i+num_corrs])
    colors = [[0, 0.651, 0.929] for i in range(int(len(lines)))]
    colors[0:int(len(lines)/2)] = [[0, 0.651, 0.929] for i in range(int(len(lines)/2))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    line_set.colors = o3d.utility.Vector3dVector(colors)

    # voxel_size = 400
    # radius_normal = voxel_size * 10
    # A_pcd_txy.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # A_pcd_txy.orient_normals_towards_camera_location((0,0,10000000000))
    # B_pcd_txy.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    # B_pcd_txy.orient_normals_towards_camera_location((0,0,10000000000))
    o3d.visualization.draw_geometries([A_pcd_txy, B_pcd_txy, line_set], point_show_normal=False)


def execute_global_registration1(source, target, voxel_size=0.01):
    "execute RANSAC global registration"
    dst_source = copy.deepcopy(source)
    DISTANCE_THRES = voxel_size * 1.0
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % DISTANCE_THRES)
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                        source_down, target_down, source_fpfh, target_fpfh, DISTANCE_THRES,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, 
                        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(DISTANCE_THRES)],
                        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    # dst_source.transform(ransac_result.transformation)
    return ransac_result


def ICP(v1, n1, v2, n2, max_iter=5, sample_rate=1, trans_init=np.eye(4), threshold=0.1):
    """Transformation matrix from v2 to v1"""
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(v1.reshape(-1, 3))
    source.normals = o3d.utility.Vector3dVector(n1.reshape(-1, 3))

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(v2.reshape(-1, 3))
    target.normals = o3d.utility.Vector3dVector(n2.reshape(-1, 3))

    # down sample
    source_vertex_num = np.array(source.points).shape[0]
    if int(source_vertex_num * sample_rate) < 100:
        sample_num = np.minimum(100, source_vertex_num)
        source_sample_rate = sample_num / source_vertex_num
    else:
        source_sample_rate = sample_rate
    source_samples = source.voxel_down_sample(voxel_size=source_sample_rate)

    target_vertex_num = np.array(target.points).shape[0]
    if int(source_vertex_num * sample_rate) < 100:
        sample_num = np.minimum(100, target_vertex_num)
        target_sample_rate = sample_num / source_vertex_num
    else:
        target_sample_rate = sample_rate
    target_samples = target.voxel_down_sample(voxel_size=target_sample_rate)

    # # outlier
    # processed_source, outlier_index = source_samples.remove_radius_outlier(nb_points=16, radius=0.5)
    #     # processed_target, outlier_index = target_samples.remove_radius_outlier(nb_points=16, radius=0.5)
    #     # processed_source_vertex_num = np.array(processed_source.points).shape[0]
    #     # print(source_vertex_num)
    #     # if processed_source_vertex_num < 100:
    #     #     processed_source = source_samples
    #     # processed_target_vertex_num = np.array(processed_target.points).shape[0]
    #     # if processed_target_vertex_num < 100:
    #     #     processed_target = target_samples

    #  icp
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_samples, target_samples, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    
    # draw_registration_result(source_samples, target_samples, reg_p2p.transformation)

    return reg_p2p.transformation


def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T 

def T2Rt(T):
    R = np.identity(3)
    t = np.zeros((3,1))
    R = T[:3,:3]
    t = T[:3,3]
    return R, t

def r2euler(R, type="XYZ"):
    R = np.array(R)
    type = str(type).upper()
    err = float(0.001)

    if np.shape(R) != (3,3):
        print("The size of R matrix is wrong")
        return
    else:
        pass


    if type == "XYZ":
        # R[0,2]/sqrt((R[1,2])**2 + (R[2,2])**2) == sin(beta)/|cos(beta)| 
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(R[0,2], math.sqrt((R[1,2])**2 + (R[2,2])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[1,0], R[1,1])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[1,0], R[1,1])
        else:
            alpha = math.atan2(-(R[1,2])/(math.cos(beta)),(R[2,2])/(math.cos(beta)))
            gamma = math.atan2(-(R[0,1])/(math.cos(beta)), (R[0,0])/(math.cos(beta)))


    elif type == "XZY":
        # -R[0,1]/sqrt((R[1,1])**2 + (R[2,1])**2) == sin(beta)/|cos(beta)|
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(-R[0,1], math.sqrt((R[1,1])**2 + (R[2,1])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[1,2], R[1,0])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[1,2], -R[1,0])
        else:
            alpha = math.atan2((R[2,1])/(math.cos(beta)), (R[1,1])/(math.cos(beta)))
            gamma = math.atan2((R[0,2])/(math.cos(beta)), (R[0,0])/(math.cos(beta)))


    elif type == "YXZ":
        # -R[1,2]/sqrt(R[0,2]**2 + R[2,2]**2) == sin(beta)/|cos(beta)|
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(-R[1,2], math.sqrt((R[0,2])**2 + (R[2,2])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        else:
            alpha = math.atan2((R[0,2])/(math.cos(beta)), (R[2,2])/(math.cos(beta)))
            gamma = math.atan2((R[1,0])/(math.cos(beta)), (R[1,1])/(math.cos(beta)))


    elif type == "YZX":
        # R[1,0]/sqrt(R[0,0]**2 + R[2,0]**2) == sin(beta)/|cos(beta)|
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(R[1,0], math.sqrt((R[0,0])**2 + (R[2,0])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], -R[0,1])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,2], R[0,1])
        else:
            alpha = math.atan2(-(R[2,0])/(math.cos(beta)), (R[0,0])/(math.cos(beta)))
            gamma = math.atan2(-(R[1,2])/(math.cos(beta)), (R[1,1])/(math.cos(beta)))


    elif type == "ZXY":
        # R[2,1]/sqrt(R[0,1]**2 + R[1,1]**2) == sin(beta)/|cos(beta)|
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(R[2,1], math.sqrt((R[0,1])**2 + (R[1,1])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        else:
            alpha = math.atan2(-(R[0,1])/(math.cos(beta)), (R[1,1])/(math.cos(beta)))
            gamma = math.atan2(-(R[2,0])/(math.cos(beta)), (R[2,2])/(math.cos(beta)))


    elif type == "ZYX":
        # -R[2,0]/sqrt(R[0,0]**2 + R[1,0]**2) == sin(beta)/|cos(beta)| 
        # ==> beta (-pi/2, pi/2)
        beta = math.atan2(-R[2,0], math.sqrt((R[0,0])**2 + (R[1,0])**2))

        if beta >= math.pi/2-err and beta <= math.pi/2+err:
            beta = math.pi/2
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,1], R[1,2])
        elif beta >= -(math.pi/2)-err and beta <= -(math.pi/2)+err:
            beta = -math.pi/2
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], -R[1,2])
        else:
            alpha = math.atan2((R[1,0])/(math.cos(beta)), (R[0,0])/(math.cos(beta)))
            gamma = math.atan2((R[2,1])/(math.cos(beta)), (R[2,2])/(math.cos(beta)))
    

    elif type == "XYX":
        # sqrt(R[0,1]**2 + R[0,2]**2)/R[0,0] == |sin(beta)|/cos(beta)
        # ==> beta (0, pi)
        beta = math.atan2(math.sqrt((R[0,1])**2 + (R[0,2])**2), R[0,0])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[1,2], R[1,1])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[1,2], R[1,1])
        else:
            alpha = math.atan2((R[1,0])/(math.sin(beta)), -(R[2,0])/(math.sin(beta)))
            gamma = math.atan2((R[0,1])/(math.sin(beta)), (R[0,2])/(math.sin(beta)))


    elif type == "XZX":
        # sqrt(R[1,0]**2 + R[2,0]**2)/R[0,0] == |sin(beta)|/cos(beta)
        # ==> beta (0, pi)
        beta = math.atan2(math.sqrt((R[1,0])**2 + (R[2,0])**2), R[0,0])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[1,2], R[1,1])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[1,2], -R[1,1])
        else:
            alpha = math.atan2((R[2,0])/(math.sin(beta)), (R[1,0])/(math.sin(beta)))
            gamma = math.atan2((R[0,2])/(math.sin(beta)), -(R[0,1])/(math.sin(beta)))


    elif type == "YXY":
        # sqrt(R[0,1]**2 + R[2,1]**2)/R[1,1] == |sin(beta)|/cos(beta)
        # ==> beta(0, pi)
        beta = math.atan2(math.sqrt((R[0,1])**2 + (R[2,1])**2), R[1,1])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        else:
            alpha = math.atan2((R[0,1])/(math.sin(beta)), (R[2,1])/(math.sin(beta)))
            gamma = math.atan2((R[1,0])/(math.sin(beta)), -(R[1,2])/(math.sin(beta)))


    elif type == "YZY":
        # sqrt(R[0,1]**2 + R[2,1]**2)/R[1,1] == |sin(beta)|/cos(beta)
        # ==> beta(0, pi)
        beta = math.atan2(math.sqrt((R[0,1])**2 + (R[2,1])**2), R[1,1])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,2], R[0,0])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,2], -R[0,0])
        else:
            alpha = math.atan2((R[2,1])/(math.sin(beta)), -(R[0,1])/(math.sin(beta)))
            gamma = math.atan2((R[1,2])/(math.sin(beta)), (R[1,0])/(math.sin(beta)))


    elif type == "ZXZ":
        # sqrt(R[0,2]**2 + R[1,2]**2)/R[2,2] == |sin(beta)|/cos(beta)
        # ==> beta(0, pi)
        beta = math.atan2(math.sqrt((R[0,2])**2 + (R[1,2])**2), R[2,2])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        elif beta >= math.pi-err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        else:
            alpha = math.atan2((R[0,2])/(math.sin(beta)), -(R[1,2])/(math.sin(beta)))
            gamma = math.atan2((R[2,0])/(math.sin(beta)), (R[2,1])/(math.sin(beta)))


    elif type == "ZYZ":
        # sqrt(R[0,2]**2 + R[1,2]**2)/R[2,2] == |sin(beta)|/cos(beta)
        # ==> beta(0, pi)
        beta = math.atan2(math.sqrt((R[0,2])**2 + (R[1,2])**2), R[2,2])
        if beta >= 0.0-err and beta <= 0.0+err:
            beta = 0.0
            # alpha + gamma is fixed
            alpha = 0.0
            gamma = math.atan2(-R[0,1], R[0,0])
        elif beta >= math.pi+err and beta <= math.pi+err:
            beta = math.pi
            # alpha - gamma is fixed
            alpha = 0.0
            gamma = math.atan2(R[0,1], -R[0,0])
        else:
            alpha = math.atan2((R[1,2])/(math.sin(beta)), (R[0,2])/(math.sin(beta)))
            gamma = math.atan2((R[2,1])/(math.sin(beta)), -(R[2,0])/(math.sin(beta)))

    return alpha, beta, gamma

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def draw_time_rms():
    methods = ["ICP", "GICP", "VGICP", "Ours", "ICP1"]
    ln = []
    plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xticks(FontProperties='Times New Roman', size=15)
    plt.yticks(FontProperties='Times New Roman', size=15)
    for i in range(4):
        datafile = "./out/" + methods[i] + ".txt"
        x = np.arange(0,51)
        y = np.loadtxt(datafile)
        if y.shape[0]<x.shape[0]:
            temp = np.ones(51-y.shape[0])*y[-1]
            y = np.append(y,temp,axis=0)
        ln1, = plt.plot(x[0:],y[0:])
        ln.append(ln1)
    font = {'family': 'Times New Roman' , 'weight' : 'normal', 'size': 15}
    plt.legend(handles=ln, labels=methods, prop=font) 
    font1 = {'family': 'SimSun' , 'weight' : 'normal', 'size': 15}
    plt.xlabel(u'迭代次数', font1)
    plt.ylabel(u'RMSE/m', font)
    plt.savefig("time_rms.jpg", dpi=600)
    plt.show()

def draw_time_sigma_rms(sigma_list, algorithm_list=["ours"], k_neigh_list=[5]):
    methods = []
    ln = []
    plt.figure()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.xticks(FontProperties='Times New Roman', size=15)
    plt.yticks(FontProperties='Times New Roman', size=15)
    for i in range(len(sigma_list)):
        for j in range(len(algorithm_list)):
            for k in range(len(k_neigh_list)):
                datafile = "./out/"+algorithm_list[j]+"_"+str(sigma_list[i])+"_"+str(k_neigh_list[k])+".txt"
                x = np.arange(0,51)
                if os.path.exists(datafile):
                    y = np.loadtxt(datafile)
                    if y.shape[0]<x.shape[0]:
                        temp = np.ones(51-y.shape[0])*y[-1]
                        y = np.append(y,temp,axis=0)
                    ln1, = plt.plot(x[0:],y[0:])
                    ln.append(ln1)
                    # methods.append("sigma="+str(sigma_list[i]))
                    methods.append(algorithm_list[j]+":r="+str(np.sqrt(k_neigh_list[k]-1)/2)+"$V_s$")
    plt.ylim(ymax=160)
    font = {'family': 'Times New Roman' , 'weight' : 'normal', 'size': 11}
    # plt.legend(handles=ln, labels=methods, ncol=2, fontsize=11) 
    plt.legend(handles=ln, labels=methods, ncol=2, prop=font) 
    # plt.xlabel(u'迭代次数', fontsize=11)
    # plt.ylabel(u'RMSE/m', fontsize=11)
    font1 = {'family': 'SimSun' , 'weight' : 'normal', 'size': 15}
    plt.xlabel(u'迭代次数', font1)
    font2 = {'family': 'Times New Roman' , 'weight' : 'normal', 'size': 15}
    plt.ylabel(u'RMSE/m', font2)
    plt.savefig("time_rms_sigma_kneigh.jpg", dpi=600)
    plt.show()

def Point2Img(point):
    s_point = point
    minX = min(s_point[:,0])
    minY = min(s_point[:,1])
    minH = min(s_point[:,2])
    maxX = max(s_point[:,0])
    maxY = max(s_point[:,1])
    maxH = max(s_point[:,2])

    rows = int(maxX-minX + 10)
    cols = int(maxY-minY + 10)
    img = np.zeros((cols,rows,3), np.uint8)
    # img.fill(255)  # 使用白色填充图片区域,默认为黑色

    im_gray = np.zeros((20,256,3), np.uint8)
    for i in range(256):
        im_gray[:, i] = i
    im_color  = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
    cv2.imwrite("jet.jpg", im_color)

    cnt = []
    for i in range(s_point.shape[0]):
        cnt.append([int(s_point[i, 0]-minX+0.5), int(s_point[i, 1]-minY+0.5)])
        G = int((int(s_point[i, 2])-minH)/(maxH-minH)*255) 
        img[cnt[i][1]-30:cnt[i][1]+30, cnt[i][0]-30:cnt[i][0]+30, :] = im_color[0, G]
    cnt = np.array(cnt)
    # print(cnt.shape)
    rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
    box = np.int0(box)

    cv2.drawContours(img, [box], 0, (0, 0, 255), 20)
    
    # cv2.imshow("minAreaRect", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    cv2.imwrite("1.jpg", img)


def get_Area_Rect(source):
    s_point = np.asarray(source.points).copy()
    minX = min(s_point[:,0])
    minY = min(s_point[:,1])
    minH = min(s_point[:,2])
    maxX = max(s_point[:,0])
    maxY = max(s_point[:,1])
    maxH = max(s_point[:,2])

    rows = int(maxX-minX + 10)
    cols = int(maxY-minY + 10)
    img = np.zeros((cols,rows,3), np.uint8)
    # img.fill(255)  # 使用白色填充图片区域,默认为黑色

    im_gray = np.zeros((20,256,3), np.uint8)
    for i in range(256):
        im_gray[:, i] = i
    im_color  = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)

    cnt = []
    for i in range(s_point.shape[0]):
        cnt.append([int(s_point[i, 0]-minX+0.5), int(s_point[i, 1]-minY+0.5)])
        G = int((int(s_point[i, 2])-minH)/(maxH-minH)*255) 
        img[cnt[i][1]-30:cnt[i][1]+30, cnt[i][0]-30:cnt[i][0]+30, :] = im_color[0, G]
    cnt = np.array(cnt)
    # print(cnt.shape)
    rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
    box = np.int0(box)

    cv2.drawContours(img, [box], 0, (0, 0, 255), 20)

    angle = rect[2]
    # print(angle)
    M = cv2.getRotationMatrix2D((rows/2, cols/2), 90+angle, 1)
    # print(M)
    img = cv2.warpAffine(img, M, (rows, cols))
    
    # cv2.imshow("minAreaRect", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    cv2.imwrite("1.jpg", img)

    return rect

def show_grid_error(source, err=None, method=None):
    # 获取最小外接矩形
    Rect = get_Area_Rect(source)
    angle = -(90+Rect[2])
    # print(angle)
    angle=angle*3.1415926/180

    # 旋转点平面坐标，并计算最大最小值
    M = np.array([[np.cos(angle), np.sin(angle)],[-np.sin(angle), np.cos(angle)]])
    s_point = np.asarray(source.points).copy()
    s_point[:,0:2] = np.dot(s_point[:,0:2], M)
    minX = min(s_point[:,0])
    minY = min(s_point[:,1])
    minH = min(s_point[:,2])
    maxX = max(s_point[:,0])
    maxY = max(s_point[:,1])
    maxH = max(s_point[:,2])
    # print("{} {} {} {} {} {}".format(minX,minY,minH,maxX,maxY,maxH))

    minE = min(err)
    maxE = max(err)
    # print(maxE)

    # 可视化
    Point2Img(s_point)

    # 分块统计配准误差
    im_gray = np.zeros((20,256,3), np.uint8)
    for i in range(256):
        im_gray[:, i] = i
    im_color  = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)

    gridSize = 50.0
    block = 1
    rows = int((maxX-minX)/gridSize)+3
    cols = int((maxY-minY)/gridSize)+3
    img = np.zeros((cols,rows,3), np.uint8)
    count = np.zeros((cols,rows), np.int)
    value = np.zeros((cols,rows), np.float)

    for i in range(s_point.shape[0]):
        r =  int((s_point[i,0]-minX)/gridSize)
        c =  int((s_point[i,1]-minY)/gridSize)
        # G = int((int(s_point[i, 2])-minH)/(maxH-minH)*255) 
        if err[i]<700:
            count[c, r] += 1
            value[c, r] += err[i]
        # G = int((int(err[i])-minE)/(300-minE)*255) 
        # if G>255:
        #     G=255
        # # img[c-block:c+block+1, r-block:r+block+1, :] = im_color[0, G]
    
    err_TM = 0
    err_count = 0
    G_temp=[0, 0, 0]
    for r in range(rows):
        for c in range(cols):
            if count[c, r] > 0:
                value[c, r] /= count[c, r]
                G = int((value[c, r]-minE)/(300-minE)*255) 
                if G>255:
                    G=255
                img[c, r, :] = im_color[0, G]
                G_temp = G
                err_TM += value[c, r]
                err_count += 1
            else:
                img[c, r, :] = [0, 0, 0]

    print("err_TM:{}".format(err_TM/err_count))
    if method is not None:
        cv2.imwrite(method+".jpg", img)
    else:
        cv2.imwrite("2.jpg", img)

    # 作图


