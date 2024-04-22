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

def coarse_match(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd):
    # extract FPFH features
    A_feats = extract_fpfh(A_pcd, VOXEL_SIZE)
    B_feats = extract_fpfh(B_pcd, VOXEL_SIZE)

    # robust global registration using RANSAC
    result_ransac = execute_global_registration(A_pcd, B_pcd,
                                            A_feats, B_feats,
                                            VOXEL_SIZE)
    result_ransac = result_ransac.transformation
    print(result_ransac)
    draw_registration_result(A_pcd_raw, B_pcd_raw, result_ransac)

    return result_ransac

def coarse_match_gm(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd):
    # get iss keypoints
    keypointsA, A_feats, A_normals = get_iss_fpfh(A_pcd, A_pcd_raw, 1*VOXEL_SIZE)
    keypointsB, B_feats, B_normals = get_iss_fpfh(B_pcd, B_pcd_raw, 1*VOXEL_SIZE)
    print(np.asarray(keypointsA.points).shape)
    print(np.asarray(keypointsB.points).shape)

    # # graph matching
    GM = GraphMatch(VOXEL_SIZE = 1*VOXEL_SIZE)
    result_gm, A_corr, B_corr, A_corr0, B_corr0 = GM.estimate_Rt(keypointsA, keypointsB, A_feats, B_feats, A_normals, B_normals)
    draw_inliner(A_pcd_raw, B_pcd_raw, A_corr0, B_corr0)
    draw_inliner(A_pcd_raw, B_pcd_raw, A_corr, B_corr)
    print(result_gm)
    # draw_registration_result(A_pcd_raw, B_pcd_raw, result_gm)

    return result_gm


def refine_match(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd, result_coarse, algorithm=None):
    R, t = T2Rt(result_coarse)

    dataA1 = Point_cloud()
    dataA1.init_from_points(np.asarray(A_pcd.points))
    dataB1 = Point_cloud()
    dataB1.init_from_points(np.asarray(B_pcd.points))

    # GICP refined regis
    threshold = 2 * VOXEL_SIZE
    R, t = T2Rt(result_coarse)
    if algorithm=="ICP":
        method = "point2point"
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        k_neigh=1
        sigma=0.0
        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    elif algorithm=="GICP":
        method = "plane2plane"
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        k_neigh=1
        sigma=0.0
        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    elif algorithm=="VGICP":
        # method = "point2point"
        method = "plane2plane"
        k_neigh=8
        sigma=0.0
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    elif algorithm=="ours":
        method = "plane2plane"
        k_neigh=10
        sigma=0.5
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        cov_data0 = dataA.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
        cov_ref0 = dataB1.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
        dist, neighbors = dataB1.kdtree.query(dataB.points, k=3, return_distance = True)
        cov_ref0 = np.array([valued_mean(cov_ref0[neighbors[i]], dist = dist[i], sigma=sigma) for i in np.arange(neighbors.shape[0])])

        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1, cov_data0=cov_data0, cov_ref0=cov_ref0)#
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    else:
        method = "slope2slope"
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        k_neigh=8
        sigma=0.5
        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    T_icp = Rt2T(R, t)
    draw_registration_result(A_pcd_raw, B_pcd_raw, T_icp)

    # draw_time_rms()
    np.savetxt("./out/"+algorithm+".txt", rms_list, fmt='%.10f',delimiter=',')

    return T_icp


def test_sigma_kneigh(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd, result_coarse, algorithm="ours", sigma=0.0, k_neigh=4):
    R, t = T2Rt(result_coarse)

    dataA1 = Point_cloud()
    dataA1.init_from_points(np.asarray(A_pcd.points))
    dataB1 = Point_cloud()
    dataB1.init_from_points(np.asarray(B_pcd.points))

    # GICP refined regis
    threshold = 2 * VOXEL_SIZE
    R, t = T2Rt(result_coarse)
    if algorithm=="ICP":
        method = "point2point"
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        # k_neigh=1
        # sigma=0.0
        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    elif algorithm=="GICP":
        method = "plane2plane"
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        # k_neigh=1
        # sigma=0.0
        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    elif algorithm=="VGICP":
        # method = "point2point"
        method = "plane2plane"
        # k_neigh=4
        # sigma=0.1
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    elif algorithm=="ours":
        method = "plane2plane"
        # k_neigh=100
        # sigma=0.5
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        cov_data0 = dataA.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
        cov_ref0 = dataB1.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
        dist, neighbors = dataB1.kdtree.query(dataB.points, k=3, return_distance = True)
        cov_ref0 = np.array([valued_mean(cov_ref0[neighbors[i]], dist = dist[i], sigma=sigma) for i in np.arange(neighbors.shape[0])])

        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1, cov_data0=cov_data0, cov_ref0=cov_ref0)#
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    else:
        # method = "point2plane"
        # A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=300)  # out_my
        # B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=900)  # out_mola
        # dataA = Point_cloud()
        # dataA.init_from_points(np.asarray(A_pcd_temp.points))
        # dataB = Point_cloud()
        # dataB.init_from_points(np.asarray(B_pcd_temp.points))
        # # k_neigh=1
        # # sigma=0.0
        # start = time.clock()
        # R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1)
        # end = time.clock()
        # print('Running time: %s Seconds'%(end-start))

        method = "plane2plane"
        # k_neigh=100
        # sigma=0.5
        A_pcd_temp = A_pcd_raw.voxel_down_sample(voxel_size=400)  # out_my
        B_pcd_temp = B_pcd_raw.voxel_down_sample(voxel_size=400)  # out_mola
        dataA = Point_cloud()
        dataA.init_from_points(np.asarray(A_pcd_temp.points))
        dataB = Point_cloud()
        dataB.init_from_points(np.asarray(B_pcd_temp.points))
        cov_data0 = dataA.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
        cov_ref0 = dataB1.get_covariance_matrices_plane2plane(epsilon = 0.001, VOXEL_SIZE = 1*VOXEL_SIZE)
        dist, neighbors = dataB1.kdtree.query(dataB.points, k=3, return_distance = True)
        cov_ref0 = np.array([valued_mean(cov_ref0[neighbors[i]], dist = dist[i], sigma=sigma) for i in np.arange(neighbors.shape[0])])

        start = time.clock()
        R, t, rms_list = GICP(dataA, dataB, method=method, exclusion_radius = threshold, sampling_limit = None, verbose = False, R0 = R, t0 = t, VOXEL_SIZE = VOXEL_SIZE, k_neigh=k_neigh, B_normals=B_pcd.normals, sigma=sigma, dataA1=dataA1, dataB1=dataB1, cov_data0=cov_data0, cov_ref0=cov_ref0)#
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))

    T_icp = Rt2T(R, t)
    # draw_registration_result(A_pcd_raw, B_pcd_raw, T_icp)

    # draw_time_rms()
    np.savetxt("./out/"+algorithm+"_"+str(sigma)+"_"+str(k_neigh)+".txt", rms_list, fmt='%.10f',delimiter=',')

    return T_icp


def compute_regiserror(source, target, algorithm):
    # data
    dataA = Point_cloud()
    dataA.init_from_points(np.asarray(source.points))
    dataB = Point_cloud()
    dataB.init_from_points(np.asarray(target.points))

    dataA_aligned = Point_cloud()
    dataA_aligned.init_from_transfo(dataA)

    # compute regis error
    k_neigh = 5
    sigma=0.1
    dist, neighbors = dataB.kdtree.query(dataA_aligned.points, k=k_neigh, return_distance = True)
    PointA = dataA_aligned.points
    # print(PointA.shape)
    PointA_ = np.array([valued_mean(dataB.points[neighbors[i]], dist = dist[i], sigma=sigma) for i in np.arange(neighbors.shape[0])])
    PointA_ = PointA_.reshape(-1,3)
    diffVect = PointA - PointA_

    # draw result
    regis_err = np.linalg.norm(diffVect, axis=1, keepdims=True)
    show_grid_error(source, err=regis_err, method=algorithm)

    # dHeight
    regis_err = diffVect[:,2]

    return regis_err




if __name__ == '__main__':
    # B_pcd_raw = o3d.io.read_point_cloud('./data/out_MOLA_200.ply')
    # A_pcd_raw = o3d.io.read_point_cloud('./data/out_my.ply')
    # B_pcd_raw = o3d.io.read_point_cloud('./src_data/Sprit_MOLA463_30.ply')
    # A_pcd_raw = o3d.io.read_point_cloud('./src_data/Sprit_my_16.ply')
    B_pcd_raw = o3d.io.read_point_cloud('./src_data/Curiosity_MOLA200_30_cc.ply')
    A_pcd_raw = o3d.io.read_point_cloud('./src_data/Curiosity_my_16_cc.ply')

    # # show raw data
    # A_pcd_raw.paint_uniform_color([1, 0.706, 0]) # show A_pcd in blue
    # B_pcd_raw.paint_uniform_color([0, 0.651, 0.929]) # show B_pcd in red

    # # voxel downsample
    # A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    # B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    # if VISUALIZE:
    #     o3d.visualization.draw_geometries([A_pcd, B_pcd]) # plot downsampled A and B 

    # # # coarse match
    # result_coarse = coarse_match_gm(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd)
    # dz = np.mean(A_pcd.points,axis=0)-np.mean(B_pcd.points,axis=0)
    # # print(dz)
    # result_coarse = np.array(  [[ 1,  0,  0,  0],
    #                             [ 0,  1,  0,  0],
    #                             [ 0,  0,  1,  -dz[2]],
    #                             [ 0,  0,  0,  1]] )
    # result_coarse = np.array(   [[9.98560802e-01,  5.28916579e-02,  8.87681821e-03,  4.98520668e+02],
    #                             [-5.25178231e-02,  9.97893740e-01, -3.80783768e-02, -1.55888278e+03],
    #                             [-1.08721498e-02,  3.75573833e-02,  9.99235327e-01,  3.46782467e+04],
    #                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # result_coarse = np.array(  [[ 9.96298657e-01, -5.74578942e-02,  6.39341635e-02,  4.37548595e+03],
    #                             [ 5.80995908e-02,  9.98276935e-01, -8.22179759e-03, -3.00271162e+02],
    #                             [-6.33515936e-02,  1.19059146e-02,  9.97920250e-01,  3.47932712e+04],
    #                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # result_coarse = np.array(  [[ 9.99502465e-01, -2.31144039e-02,  2.14603624e-02,  1.28678238e+03],
    #                             [ 2.40768985e-02,  9.98663592e-01, -4.57310866e-02, -1.70072873e+03],
    #                             [-2.03746358e-02,  4.62250327e-02,  9.98723245e-01,  3.46964564e+04],
    #                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # result_coarse = np.array(   [[ 9.98970857e-01, -6.50307145e-03, -4.48880477e-02, -2.25002712e+03],
    #                              [ 5.01282529e-03,  9.99435085e-01, -3.32322626e-02, -1.28589619e+03],
    #                              [ 4.50788016e-02,  3.29730459e-02,  9.98439122e-01,  3.46099507e+04],
    #                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    result_coarse = np.array(  [[ 9.99919368e-01,  1.19406830e-02,  4.32168061e-03,  4.99805984e+02],
                                [-1.17563304e-02,  9.99113326e-01, -4.04271128e-02, -1.88781914e+03],
                                [-4.80057603e-03,  4.03730460e-02,  9.99173144e-01,  3.47920950e+04],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # draw_registration_result(A_pcd_raw, B_pcd_raw, result_coarse)
    # # save the transform source points and show regis_err
    # A_pcd_raw_temp = copy.deepcopy(A_pcd_raw)
    # A_pcd_raw_temp.transform(result_coarse)
    # o3d.io.write_point_cloud('./src_data/result/Pointcloud_trans/Curiosity_my_16_trans_coarse.ply',A_pcd_raw_temp)
    # regis_err = compute_regiserror(A_pcd_raw_temp, B_pcd_raw, "coarse")
    # data_err = np.concatenate((np.asarray(A_pcd_raw_temp.points), regis_err.reshape(-1, 1)),axis=1)
    # np.savetxt('./src_data/result/regis_err/Curiosity_regis_err_coarse.txt', data_err, fmt="%.3f", delimiter=' ') 

    # refine match
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=400)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=30)
    algorithms = ["ICP", "GICP", "VGICP", "ours", "ICP1"]
    for i in range(4,5):
        algorithm = algorithms[i]
        print(algorithm)
        result_refine = refine_match(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd, result_coarse, algorithm)

        # save the transform source points and show regis_err
        A_pcd_raw_temp = copy.deepcopy(A_pcd_raw)
        A_pcd_raw_temp.transform(result_refine)
        regis_err = compute_regiserror(A_pcd_raw_temp, B_pcd_raw, algorithm)

        # # o3d.io.write_point_cloud('./src_data/result/Sprit_my_16_trans.ply',A_pcd_raw_temp)
        # o3d.io.write_point_cloud('./src_data/result/Pointcloud_trans/Curiosity_my_16_trans_'+algorithm+'.ply',A_pcd_raw_temp)
        # data_err = np.concatenate((np.asarray(A_pcd_raw_temp.points), regis_err.reshape(-1, 1)),axis=1)
        # # print(data_err.shape)
        # np.savetxt('./src_data/result/regis_err/Curiosity_regis_err_'+algorithm+'.txt', data_err, fmt="%.3f", delimiter=' ') 

    # # draw time rms
    # draw_time_rms()

    # # sigma test
    # sigma = 0.0001
    # sigma_list = []
    # for i in range(6):
    #     result_refine = test_sigma_kneigh(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd, result_coarse, "ours", sigma=sigma)
    #     sigma_list.append(sigma)
    #     result_refine = test_sigma_kneigh(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd, result_coarse, "ours", sigma=5*sigma)
    #     sigma_list.append(5*sigma)
    #     sigma = sigma*10
    # draw_time_sigma_rms(sigma_list)

    # # kneigh test
    # sigma_list = [0, 0.5]
    # algorithm_list = ["VGICP", "Ours"]
    # k_neigh_list = []
    # for i in range(5):
    #     k_neigh = 2*(i+1)*2*(i+1)+1
    #     # print(k_neigh)
    #     # print("VGICP")
    #     # result_refine = test_sigma_kneigh(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd, result_coarse, "VGICP", sigma=0, k_neigh=k_neigh)
    #     # print("ours")
    #     # result_refine = test_sigma_kneigh(A_pcd_raw, B_pcd_raw, A_pcd, B_pcd, result_coarse, "ours", sigma=0.5, k_neigh=k_neigh)
    #     k_neigh_list.append(k_neigh)
    # draw_time_sigma_rms(sigma_list, algorithm_list, k_neigh_list)



