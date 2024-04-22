#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
pc_match.py
"""

__author__  = "wbl"
__version__ = "1.00"
__date__    = "14 Nov 2021"

import copy
import os
import sys

import time
import numpy as np
import open3d as o3d
import open3d_tutorial as o3dtut
import numpy as np 
from helpers import *
from graph_match import *
from Generalized_ICP import *

from icp_1 import *

# VOXEL_SIZE = 0.01
VOXEL_SIZE = 400
VISUALIZE = False

def fpfh_match():
    # Load and visualize two point clouds from 3DMatch dataset
    B_pcd_raw = o3d.io.read_point_cloud('./ply/sample_gundum_01.ply')
    # B_pcd_raw = o3d.io.read_point_cloud('./data/out_my.ply')
    A_pcd_raw = o3d.io.read_point_cloud('./ply/sample_gundum_02.ply')


    cl, ind=A_pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=3)
    # cl, ind=cl.remove_statistical_outlier(nb_neighbors=50,std_ratio=1)
    # cl, ind=A_pcd_raw.remove_radius_outlier(nb_points=20,radius=0.5)
    A_pcd_raw=cl

    # cl,ind=B_pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=0.3)
    # cl,ind=cl.remove_statistical_outlier(nb_neighbors=50,std_ratio=1)
    # cl,ind=A_pcd_raw.remove_radius_outlier(nb_points=20,radius=0.5)
    # B_pcd_raw=cl

    # A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
    # B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
    # if VISUALIZE:
    #     o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

    # fpfh-matching
    # voxel downsample both clouds
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    #cl,ind=A_pcd.remove_statistical_outlier(nb_neighbors=50,std_ratio=2)
    #A_pcd=cl


    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd, B_pcd]) # plot downsampled A and B 

    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M

    keypointsA = o3d.geometry.keypoint.compute_iss_keypoints(A_pcd)
    keypointsB = o3d.geometry.keypoint.compute_iss_keypoints(B_pcd)
    print(np.asarray(keypointsA.points))

    # extract FPFH features
    A_feats = extract_fpfh(A_pcd, VOXEL_SIZE)
    B_feats = extract_fpfh(B_pcd, VOXEL_SIZE)
    print(np.asarray(A_pcd.points).shape)
    print(A_feats.shape)

    # # establish correspondences by nearest neighbour search in feature space
    # # corrs_A, corrs_B = find_correspondences(A_feats, B_feats,mutual_filter=True)
    # corrs_A, corrs_B = find_correspondences1(A_feats, B_feats, VOXEL_SIZE, mutual_filter=True)
    # A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    # B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

    # num_corrs = A_corr.shape[1]
    # print(f'FPFH generates {num_corrs} putative correspondences.')

    # # visualize the point clouds together with feature correspondences
    # points = np.concatenate(((A_corr).T,(B_corr).T),axis=0)
    # points = np.concatenate(((A_corr).T-np.mean(A_corr,axis=1).T,(B_corr).T-np.mean(B_corr,axis=1).T),axis=0)
    # # points = np.concatenate(((A_corr-A_corr.mean(axis=1)).T,(B_corr-B_corr.mean(axis=1)).T),axis=0)
    # # print(A_corr.mean(axis=1))
    # # print(B_corr.mean(axis=1))
    # lines = []
    # for i in range(num_corrs):
    #     lines.append([i,i+num_corrs])
    # colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(points),
    #     lines=o3d.utility.Vector2iVector(lines),
    # )

    # A_pcd_txy =  copy.deepcopy(A_pcd).translate(-np.mean(A_corr,axis=1))
    # B_pcd_txy =  copy.deepcopy(B_pcd).translate(-np.mean(B_corr,axis=1))
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([A_pcd_txy,B_pcd_txy,line_set])

    # robust global registration using RANSAC
    result_ransac = execute_global_registration(A_pcd, B_pcd,
                                            A_feats, B_feats,
                                            VOXEL_SIZE)
    # result_ransac = execute_global_registration1(A_pcd, B_pcd,
    #                                            VOXEL_SIZE)
    result_ransac = result_ransac.transformation
    print(result_ransac)
    draw_registration_result(A_pcd_raw, B_pcd_raw, result_ransac)


    # Visualize the registration results
    A_pcd_T_teaser = copy.deepcopy(A_pcd_raw).transform(result_ransac)
    o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd_raw])


    #VOXEL_SIZE = 0.05
    # local refinement using ICP
    NOISE_BOUND = 6 * VOXEL_SIZE
    #A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    #B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    icp_sol = o3d.pipelines.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, result_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T_icp =icp_sol.transformation
    print(T_icp)

    # visualize the registration after ICP refinement
    A_pcd_T_icp = copy.deepcopy(A_pcd_raw).transform(T_icp)
    o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd_raw])

    # A_pcd_raw_T_icp = copy.deepcopy(A_pcd_raw).transform(T_icp)
    # B_pcd_raw_c=copy.deepcopy(B_pcd_raw)
    # o3d.io.write_point_cloud('./data/out/result1.ply',A_pcd_T_icp)
    # o3d.io.write_point_cloud('./data/out/result2.ply',B_pcd)
    # #o3d.io.write_point_cloud('/home/wbl/out/result1.ply',A_pcd_raw_T_icp)
    # #o3d.io.write_point_cloud('/home/wbl/out/result2.ply',B_pcd_raw)

def iss_fpfh_match():
    # Load and visualize two point clouds from 3DMatch dataset
    B_pcd_raw = o3d.io.read_point_cloud('./ply/sample_gundum_01.ply')
    A_pcd_raw = o3d.io.read_point_cloud('./ply/sample_gundum_02.ply')

    # remove nosie points
    cl, ind=A_pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=3)
    A_pcd_raw=cl
    cl,ind=B_pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=0.3)
    B_pcd_raw=cl

    # show raw data
    A_pcd_raw.paint_uniform_color([1, 0.706, 0]) # show A_pcd in blue
    B_pcd_raw.paint_uniform_color([0, 0.651, 0.929]) # show B_pcd in red
    # if VISUALIZE:
    #     o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

    # voxel downsample
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd, B_pcd]) # plot downsampled A and B 

    # get iss keypoints
    idxA, keypointsA = get_iss_keypoints(A_pcd, A_pcd_raw)
    idxB, keypointsB = get_iss_keypoints(B_pcd, B_pcd_raw)

    # extract FPFH features
    A_feats = extract_fpfh(A_pcd, VOXEL_SIZE)
    B_feats = extract_fpfh(B_pcd, VOXEL_SIZE)
    

    # robust global registration using RANSAC
    A_feats = A_feats[idxA, :]
    B_feats = B_feats[idxB, :]
    result_ransac = execute_global_registration(keypointsA, keypointsB,
                                            A_feats, B_feats,
                                            VOXEL_SIZE)                                        
    # result_ransac = execute_global_registration1(A_pcd, B_pcd,
    #                                            VOXEL_SIZE)
    result_ransac = result_ransac.transformation
    print(result_ransac)
    draw_registration_result(A_pcd_raw, B_pcd_raw, result_ransac)

    # Visualize the iss_fpfh registration results
    A_pcd_T_ransac = copy.deepcopy(A_pcd_raw).transform(result_ransac)
    o3d.visualization.draw_geometries([A_pcd_T_ransac, B_pcd_raw])

    # local refinement using ICP
    NOISE_BOUND = 6 * VOXEL_SIZE
    icp_sol = o3d.pipelines.registration.registration_icp(
        A_pcd, B_pcd, NOISE_BOUND, result_ransac,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T_icp =icp_sol.transformation
    print(T_icp)

    # visualize the registration after ICP refinement
    A_pcd_T_icp = copy.deepcopy(A_pcd_raw).transform(T_icp)
    o3d.visualization.draw_geometries([A_pcd_T_icp, B_pcd_raw])


def iss_fpfh_match1():
    # Load and visualize two point clouds from 3DMatch dataset
    B_pcd_raw = o3d.io.read_point_cloud('./ply/sample_gundum_01.ply')
    A_pcd_raw = o3d.io.read_point_cloud('./ply/sample_gundum_02.ply')

    # remove nosie points
    cl, ind=A_pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=3)
    A_pcd_raw=cl
    cl, ind=B_pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=3)
    B_pcd_raw=cl

    # show raw data
    A_pcd_raw.paint_uniform_color([1, 0.706, 0]) # show A_pcd in blue
    B_pcd_raw.paint_uniform_color([0, 0.651, 0.929]) # show B_pcd in red
    # if VISUALIZE:
    #     o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

    # voxel downsample
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd, B_pcd]) # plot downsampled A and B 

    # # get iss keypoints
    # idxA, keypointsA = get_iss_keypoints(A_pcd, A_pcd_raw)
    # idxB, keypointsB = get_iss_keypoints(B_pcd, B_pcd_raw)
    # # idxA, keypointsA = get_iss_keypoints1(A_pcd)
    # # idxB, keypointsB = get_iss_keypoints1(B_pcd)

    # # extract FPFH features
    # A_feats = extract_fpfh(A_pcd, VOXEL_SIZE)
    # B_feats = extract_fpfh(B_pcd, VOXEL_SIZE)
    # A_feats = A_feats[idxA, :]
    # B_feats = B_feats[idxB, :]
    # A_normals = np.asarray(A_pcd.normals)[idxA]
    # B_normals = np.asarray(B_pcd.normals)[idxB]
    keypointsA, A_feats, A_normals = get_iss_fpfh(A_pcd, A_pcd_raw, VOXEL_SIZE)
    keypointsB, B_feats, B_normals = get_iss_fpfh(B_pcd, B_pcd_raw, VOXEL_SIZE)

    # graph matching
    GM = GraphMatch()
    result_gm, A_corr, B_corr = GM.estimate_Rt(keypointsA, keypointsB, A_feats, B_feats, A_normals, B_normals)
    GM.draw_inliner(A_pcd, B_pcd, A_corr, B_corr)
    print(result_gm)
    draw_registration_result(A_pcd_raw, B_pcd_raw, result_gm)

    
    # robust global registration using RANSAC
    result_ransac = execute_global_registration(keypointsA, keypointsB,
                                            A_feats, B_feats,
                                            VOXEL_SIZE)                                        
    # result_ransac = execute_global_registration1(A_pcd_raw, B_pcd_raw,
    #                                            VOXEL_SIZE)
    result_ransac = result_ransac.transformation
    # print(result_ransac)
    draw_registration_result(A_pcd_raw, B_pcd_raw, result_ransac)

    # # Visualize the iss_fpfh registration results
    # A_pcd_T_ransac = copy.deepcopy(A_pcd_raw).transform(result_ransac)
    # o3d.visualization.draw_geometries([A_pcd_T_ransac, B_pcd_raw])

    # # local refinement using ICP
    # NOISE_BOUND = 6 * VOXEL_SIZE
    # icp_sol = o3d.pipelines.registration.registration_icp(
    #     A_pcd, B_pcd, NOISE_BOUND, result_ransac,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    # T_icp =icp_sol.transformation
    # print(T_icp)

    # # visualize the registration after ICP refinement
    # A_pcd_T_icp = copy.deepcopy(A_pcd_raw).transform(T_icp)
    # o3d.visualization.draw_geometries([A_pcd_T_icp, B_pcd_raw])

def iss_fpfh_match2():
    # Load and visualize two point clouds from 3DMatch dataset
    B_pcd_raw = o3d.io.read_point_cloud('./data/sprit_MOLA_463_30.ply')
    A_pcd_raw = o3d.io.read_point_cloud('./data/sprit_my_16.ply')
    # B_pcd_raw = o3d.io.read_point_cloud('./data/Curiosity_MOLA_463_30.ply')
    # A_pcd_raw = o3d.io.read_point_cloud('./data/Curiosity_my_16.ply')

    # show raw data
    # A_pcd_raw.paint_uniform_color([1, 0.706, 0]) # show A_pcd in blue
    # B_pcd_raw.paint_uniform_color([0, 0.651, 0.929]) # show B_pcd in red
    # if VISUALIZE:
    #     o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

    # voxel downsample
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd, B_pcd]) # plot downsampled A and B 

    # get iss keypoints
    keypointsA, A_feats, A_normals = get_iss_fpfh(A_pcd, A_pcd_raw, VOXEL_SIZE)
    keypointsB, B_feats, B_normals = get_iss_fpfh(B_pcd, B_pcd_raw, VOXEL_SIZE)
    # print(np.asarray(keypointsA.points).shape)
    # print(np.asarray(keypointsB.points).shape)

    # # graph matching
    # GM = GraphMatch(VOXEL_SIZE = VOXEL_SIZE)
    # result_gm, A_corr, B_corr = GM.estimate_Rt(keypointsA, keypointsB, A_feats, B_feats, A_normals, B_normals)
    # GM.draw_inliner(A_pcd, B_pcd, A_corr, B_corr)
    # print(result_gm)
    result_gm = np.array(  [[ 1,  0,  0,  0],
                            [ 0,  1,  0,  0],
                            [ 0,  0,  1,  0],
                            [ 0,  0,  0,  1]] )
    draw_registration_result(A_pcd_raw, B_pcd_raw, result_gm)
    
    R, t = T2Rt(result_gm)
    dataA = Point_cloud()
    dataA.init_from_points(np.asarray(A_pcd_raw.points))
    dataB = Point_cloud()
    dataB.init_from_points(np.asarray(B_pcd_raw.points))
    k_neigh = 1
    dataA_aligned = Point_cloud()
    dataA_aligned.init_from_transfo(dataA, R, t)
    dist, neighbors = dataB.kdtree.query(dataA_aligned.points, k=k_neigh, return_distance = True)
    PointA = dataA_aligned.points
    PointA_ = np.array([valued_mean(dataB.points[neighbors[i]], dist = dist[i], sigma=0.05) for i in np.arange(neighbors.shape[0])])
    PointA_ = PointA_.reshape(-1,3)
    diffVect = PointA - PointA_
    regis_err = np.linalg.norm(diffVect, axis=1, keepdims=True)
    # regis_err = abs(diffVect[:,2])
    show_grid_error(A_pcd_raw, err=regis_err, method="Coarse")

    dataA1 = Point_cloud()
    dataA1.init_from_points(np.asarray(A_pcd.points))
    dataB1 = Point_cloud()
    dataB1.init_from_points(np.asarray(B_pcd.points))

    # GICP refined regis
    threshold = 2 * VOXEL_SIZE
    methods = ["ICP", "GICP", "VGICP", "ours", "ICP1"]
    for i in range(4):
        # B_pcd_raw = o3d.io.read_point_cloud('./data/out_MOLA_200.ply')
        # A_pcd_raw = o3d.io.read_point_cloud('./data/out_my.ply')

        print(methods[i])
        R, t = T2Rt(result_gm)
        if i==0:
            continue
            A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
            B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=900)  # out_mola
            dataA = Point_cloud()
            dataA.init_from_points(np.asarray(A_pcd.points))
            dataB = Point_cloud()
            dataB.init_from_points(np.asarray(B_pcd.points))
            method = "point2point"
            k_neigh=1
            sigma=0.0
            start = time.clock()
            R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
            end = time.clock()
            print('Running time: %s Seconds'%(end-start))

        elif i==1:
            continue
            method = "plane2plane"
            A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
            B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=300)  # out_mola
            dataA = Point_cloud()
            dataA.init_from_points(np.asarray(A_pcd.points))
            dataB = Point_cloud()
            dataB.init_from_points(np.asarray(B_pcd.points))
            k_neigh=1
            sigma=0.0
            start = time.clock()
            R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
            end = time.clock()
            print('Running time: %s Seconds'%(end-start))

        elif i==2:
            # continue
            # method = "point2point"
            method = "plane2plane"
            k_neigh=3
            sigma=0.0
            A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
            B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=300)  # out_mola
            dataA = Point_cloud()
            dataA.init_from_points(np.asarray(A_pcd.points))
            dataB = Point_cloud()
            dataB.init_from_points(np.asarray(B_pcd.points))
            start = time.clock()
            R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
            end = time.clock()
            print('Running time: %s Seconds'%(end-start))

        elif i==3:
            continue
            method = "plane2plane"
            k_neigh=1
            sigma=0.0
            A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
            B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=300)  # out_mola
            dataA = Point_cloud()
            dataA.init_from_points(np.asarray(A_pcd.points))
            dataB = Point_cloud()
            dataB.init_from_points(np.asarray(B_pcd.points))
            cov_data0 = dataA.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
            cov_ref0 = dataB1.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
            dist, neighbors = dataB1.kdtree.query(dataB.points, k=3, return_distance = True)
            cov_ref0 = np.array([valued_mean(cov_ref0[neighbors[i]], dist = dist[i], sigma=sigma) for i in np.arange(neighbors.shape[0])])

            start = time.clock()
            R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1, cov_data0=cov_data0, cov_ref0=cov_ref0)#
            end = time.clock()
            print('Running time: %s Seconds'%(end-start))

        else:
            continue
            A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
            B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=900)  # out_mola
            dataA = Point_cloud()
            dataA.init_from_points(np.asarray(A_pcd.points))
            dataB = Point_cloud()
            dataB.init_from_points(np.asarray(B_pcd.points))
            method = "point2plane"
            k_neigh=1
            sigma=0.0
            start = time.clock()
            R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
            end = time.clock()
            print('Running time: %s Seconds'%(end-start))

        T_icp = Rt2T(R, t)
        draw_registration_result(A_pcd_raw, B_pcd_raw, T_icp)

def Experiment_scripts():
    # Load and visualize two point clouds from 3DMatch dataset
    B_pcd_raw = o3d.io.read_point_cloud('./data/sprit_MOLA_463_30.ply')
    A_pcd_raw = o3d.io.read_point_cloud('./data/sprit_my_16.ply')

    # # remove nosie points
    # cl, ind=A_pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=3)
    # A_pcd_raw=cl
    # cl, ind=B_pcd_raw.remove_statistical_outlier(nb_neighbors=50,std_ratio=3)
    # B_pcd_raw=cl

    # show raw data
    A_pcd_raw.paint_uniform_color([1, 0.706, 0]) # show A_pcd in blue
    B_pcd_raw.paint_uniform_color([0, 0.651, 0.929]) # show B_pcd in red
    # if VISUALIZE:
    #     o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

    # voxel downsample
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd, B_pcd]) # plot downsampled A and B 

    # get iss keypoints
    keypointsA, A_feats, A_normals = get_iss_fpfh(A_pcd, A_pcd_raw, VOXEL_SIZE)
    keypointsB, B_feats, B_normals = get_iss_fpfh(B_pcd, B_pcd_raw, VOXEL_SIZE)
    # print(np.asarray(keypointsA.points).shape)
    # print(np.asarray(keypointsB.points).shape)

    # # graph matching
    GM = GraphMatch(VOXEL_SIZE = VOXEL_SIZE)
    result_gm, A_corr, B_corr = GM.estimate_Rt(keypointsA, keypointsB, A_feats, B_feats, A_normals, B_normals)
    GM.draw_inliner(A_pcd, B_pcd, A_corr, B_corr)
    print(result_gm)
    draw_registration_result(A_pcd_raw, B_pcd_raw, result_gm)
    # result_gm = np.array([[ 9.99327899e-01, 2.72116688e-02, -2.45616511e-02, -6.29148846e+01],
    #                 [-2.68894533e-02,  9.99549199e-01,  1.33549992e-02, -1.82203179e+04],
    #                 [ 2.49139905e-02, -1.26855739e-02,  9.99609108e-01, -6.03394297e+04],
    #                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    
    R, t = T2Rt(result_gm)
    dataA = Point_cloud()
    dataA.init_from_points(np.asarray(A_pcd_raw.points))
    dataB = Point_cloud()
    dataB.init_from_points(np.asarray(B_pcd_raw.points))
    k_neigh = 1
    dataA_aligned = Point_cloud()
    dataA_aligned.init_from_transfo(dataA, R, t)
    dist, neighbors = dataB.kdtree.query(dataA_aligned.points, k=k_neigh, return_distance = True)
    PointA = dataA_aligned.points
    PointA_ = np.array([valued_mean(dataB.points[neighbors[i]], dist = dist[i], sigma=0.05) for i in np.arange(neighbors.shape[0])])
    PointA_ = PointA_.reshape(-1,3)
    diffVect = PointA - PointA_
    regis_err = np.linalg.norm(diffVect, axis=1, keepdims=True)
    # regis_err = abs(diffVect[:,2])
    show_grid_error(A_pcd_raw, err=regis_err, method="Coarse")

    dataA1 = Point_cloud()
    dataA1.init_from_points(np.asarray(A_pcd.points))
    dataB1 = Point_cloud()
    dataB1.init_from_points(np.asarray(B_pcd.points))

    # # GICP refined regis
    # threshold = 2 * VOXEL_SIZE
    # methods = ["ICP", "GICP", "VGICP", "ours", "ICP1"]
    # for i in range(4):
    #     # B_pcd_raw = o3d.io.read_point_cloud('./data/out_MOLA_200.ply')
    #     # A_pcd_raw = o3d.io.read_point_cloud('./data/out_my.ply')

    #     print(methods[i])
    #     R, t = T2Rt(result_gm)
    #     if i==0:
    #         # continue
    #         A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
    #         B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=900)  # out_mola
    #         dataA = Point_cloud()
    #         dataA.init_from_points(np.asarray(A_pcd.points))
    #         dataB = Point_cloud()
    #         dataB.init_from_points(np.asarray(B_pcd.points))
    #         method = "point2point"
    #         k_neigh=1
    #         sigma=0.0
    #         start = time.clock()
    #         R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
    #         end = time.clock()
    #         print('Running time: %s Seconds'%(end-start))

    #     elif i==1:
    #         # continue
    #         A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
    #         B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=300)  # out_mola
    #         dataA = Point_cloud()
    #         dataA.init_from_points(np.asarray(A_pcd.points))
    #         dataB = Point_cloud()
    #         dataB.init_from_points(np.asarray(B_pcd.points))
    #         method = "plane2plane"
    #         k_neigh=1
    #         sigma=0.0
    #         start = time.clock()
    #         R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
    #         end = time.clock()
    #         print('Running time: %s Seconds'%(end-start))

    #     elif i==2:
    #         # continue
    #         method = "plane2plane"
    #         k_neigh=4
    #         sigma=0.0
    #         A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
    #         B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=300)  # out_mola
    #         dataA = Point_cloud()
    #         dataA.init_from_points(np.asarray(A_pcd.points))
    #         dataB = Point_cloud()
    #         dataB.init_from_points(np.asarray(B_pcd.points))
    #         start = time.clock()
    #         R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
    #         end = time.clock()
    #         print('Running time: %s Seconds'%(end-start))

    #     elif i==3:
    #         method = "plane2plane"
    #         k_neigh=4
    #         sigma=0.0
    #         A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
    #         B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=900)  # out_mola
    #         dataA = Point_cloud()
    #         dataA.init_from_points(np.asarray(A_pcd.points))
    #         dataB = Point_cloud()
    #         dataB.init_from_points(np.asarray(B_pcd.points))
    #         cov_data0 = dataA.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 5*VOXEL_SIZE)
    #         cov_ref0 = dataB1.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 5*VOXEL_SIZE)
    #         dist, neighbors = dataB1.kdtree.query(dataB.points, k=3, return_distance = True)
    #         cov_ref0 = np.array([valued_mean(cov_ref0[neighbors[i]], dist = dist[i], sigma=sigma) for i in np.arange(neighbors.shape[0])])

    #         start = time.clock()
    #         R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1, cov_data0=cov_data0, cov_ref0=cov_ref0)#
    #         end = time.clock()
    #         print('Running time: %s Seconds'%(end-start))

    #     else:
    #         # continue
    #         A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
    #         B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=900)  # out_mola
    #         dataA = Point_cloud()
    #         dataA.init_from_points(np.asarray(A_pcd.points))
    #         dataB = Point_cloud()
    #         dataB.init_from_points(np.asarray(B_pcd.points))
    #         method = "point2plane"
    #         k_neigh=1
    #         sigma=0.0
    #         start = time.clock()
    #         R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
    #         end = time.clock()
    #         print('Running time: %s Seconds'%(end-start))

    #     # start = time.clock()
    #     # R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
    #     # end = time.clock()
    #     # print('Running time: %s Seconds'%(end-start))
    #     T_icp = Rt2T(R, t)
    #     # draw_registration_result(A_pcd_raw, B_pcd_raw, T_icp)

    #     # draw_time_rms()
    #     np.savetxt("./out/"+methods[i]+".txt", rms_list, fmt='%.10f',delimiter=',')

    #     # calculate errors
    #     dataA = Point_cloud()
    #     dataA.init_from_points(np.asarray(A_pcd_raw.points))
    #     dataB = Point_cloud()
    #     dataB.init_from_points(np.asarray(B_pcd_raw.points))
    #     # k_neigh = 5
    #     dataA_aligned = Point_cloud()
    #     dataA_aligned.init_from_transfo(dataA, R, t)
    #     dist, neighbors = dataB.kdtree.query(dataA_aligned.points, k=k_neigh, return_distance = True)
    #     PointA = dataA_aligned.points
    #     # PointA_ = np.mean(dataB.points[neighbors],axis=1)
    #     PointA_ = np.array([valued_mean(dataB.points[neighbors[i]], dist = dist[i], sigma=sigma) for i in np.arange(neighbors.shape[0])])
    #     PointA_ = PointA_.reshape(-1,3)
    #     diffVect = PointA - PointA_
    #     # diffVect[:,2] /= 10.0
    #     regis_err = np.linalg.norm(diffVect, axis=1, keepdims=True)
    #     # regis_err = abs(diffVect[:,2])
    #     # print(regis_err)
    #     show_grid_error(A_pcd_raw, err=regis_err, method=methods[i])

    # draw_time_rms()


if __name__ == '__main__':
    iss_fpfh_match2()
    # Experiment_scripts()



