import copy
import os
import sys

import time
import numpy as np
from sklearn.cluster import spectral_clustering
import matplotlib.pyplot as plt


from helpers import *

class GraphMatch():
    
    def __init__(self, VOXEL_SIZE = 0.01):
        self.VOXEL_SIZE = VOXEL_SIZE
        return
    
    def generate_match_graph(self, A_corr, B_corr, A_normals, B_normals):
        "generate match graph"
        c_num = A_corr.shape[0]
        Weight = np.zeros((c_num, c_num))
        EdgeCost = np.zeros((c_num, c_num))

        # calculate the graph matrix: edge dist and vector angle
        for i in range(c_num):
            for j in range(i, c_num):
                if np.linalg.norm(A_corr[i] - A_corr[j]) == 0:
                    Weight[i, j] = 0.001
                else:
                    Weight[i, j] = 1 / np.linalg.norm(A_corr[i] - A_corr[j])
                Weight[i, j] = np.exp(-10*np.linalg.norm(A_corr[i] - A_corr[j]))
                d1 = np.linalg.norm(A_corr[i] - A_corr[j])
                d2 = np.linalg.norm(B_corr[i] - B_corr[j])
                # EdgeCost[i, j] =  (d1 - d2) * (d1 - d2)
                dmin = min(d1, d2)
                dmax = max(d1, d2)
                EdgeCost[i, j] = 1 * np.abs(1 - (dmin+0.00001)/(dmax+0.00001))*np.abs(1 - (dmin+0.00001)/(dmax+0.00001))
                # print("EdgeCost:{}".format(EdgeCost[i, j]))

                va = A_corr[i] - A_corr[j]
                vb = B_corr[i] - B_corr[j]
                AngleCost = 1/4*(np.abs(self.getVectorCos(A_normals[i], va) - self.getVectorCos(B_normals[i], vb))  \
                     + np.abs(self.getVectorCos(A_normals[j], va) - self.getVectorCos(B_normals[j], vb)))
                # AngleCost = (1 + (self.getVectorCos(A_normals[i], va) - self.getVectorCos(B_normals[i], vb)))/2
                EdgeCost[i, j] += 5*AngleCost
                # print("AngleCost:{}".format(AngleCost))

                Weight[j, i] = Weight[i, j]
                EdgeCost[j, i] = EdgeCost[i, j]
                # print("{}  {}  {}  {}".format(d1, d2, EdgeCost[i, j], Weight[i, j]))
        # print(EdgeCost)
        return EdgeCost, Weight

    def calculate_cost(self, EdgeCost, Weight):
        # for each line calculate loss and find outliers
        c_num = EdgeCost.shape[0]
        PointCost = np.zeros((c_num))
        std_ec = np.std(EdgeCost)
        m_ec = np.mean(EdgeCost)
        sum_ec0 = np.sum(EdgeCost, axis=0)
        # # enhance diff
        # for i in range(c_num):
        #     for j in range(c_num):
        #         if EdgeCost[i, j] > m_ec:
        #             EdgeCost[i, j] *= 20
        # sum
        for i in range(c_num):
            # PointCost[i] = np.sum(EdgeCost[i, :] * Weight[i, :]) / np.sum(Weight[i, :])
            for j in range(c_num):
                # encourge good match
                if sum_ec0[i]/sum_ec0[j] > 1.2 and EdgeCost[i, j]/m_ec > 0.9:#
                    PointCost[i] += 20 * EdgeCost[i, j] *sum_ec0[i]/sum_ec0[j] #*Weight[i, j]
                else:
                    PointCost[i] += 1 * EdgeCost[i, j] *sum_ec0[i]/sum_ec0[j]
        x = np.arange(0, c_num)
        plt.scatter(x, PointCost)        
        plt.show()

        return PointCost

    def remove_outlier(self, A_corr, B_corr, A_normals, B_normals):
        EdgeCost, Weight = self.generate_match_graph(A_corr, B_corr, A_normals, B_normals)
        # PointCost = self.calculate_cost(EdgeCost, Weight)
        PointCost = self.HigherOrderDominantClustering(EdgeCost)
        ind0 = np.argsort(PointCost)
        ind = np.tile(ind0, (A_corr.shape[1], 1)).T
        print(PointCost.shape)
        print(ind.shape)
        print(A_corr.shape)
        print(PointCost[ind0[0]])
        print(PointCost[ind0[1]])
        print(PointCost[ind0[2]])
        A_corr = np.take_along_axis(A_corr, ind, axis=0)
        B_corr = np.take_along_axis(B_corr, ind, axis=0)
        num_inlier = round(0.1 * A_corr.shape[0])
        # return A_corr[0:num_inlier,:], B_corr[0:num_inlier,:], ind0[0:num_inlier]
        return ind0[0:num_inlier]

    def remove_outlier_sc(self, A_corr, B_corr):
        EdgeCost, Weight = self.generate_match_graph(A_corr, B_corr)
        self.draw_edgecost(EdgeCost)
        Affinity = np.exp(-2*EdgeCost)
        labels = spectral_clustering(Affinity, n_clusters=3, eigen_solver="arpack")
        print(labels)
        PointCost = self.calculate_cost(EdgeCost, Weight)
        ind0 = np.argsort(labels)
        ind = np.tile(ind0, (A_corr.shape[1], 1)).T
        print(PointCost.shape)
        # print(ind.shape)
        # print(A_corr.shape)
        # print(PointCost[ind0[0]])
        # print(PointCost[ind0[1]])
        # print(PointCost[ind0[2]])
        A_corr = np.take_along_axis(A_corr, ind, axis=0)
        B_corr = np.take_along_axis(B_corr, ind, axis=0)
        num_inlier = round(0.2 * A_corr.shape[0])
        mlabel = max(set(labels.tolist()), key=labels.tolist().count)
        print(mlabel)
        ind0 = np.where(labels==mlabel)[0]
        print(ind0)
        # return A_corr[0:num_inlier,:], B_corr[0:num_inlier,:], ind0[0:num_inlier]
        return ind0 #[0:num_inlier]

    def estimate_Rt(self, keypointsA, keypointsB, A_feats, B_feats, A_normals0, B_normals0):
        # establish correspondences by nearest neighbour search in feature space
        A_xyz = pcd2xyz(keypointsA) 
        B_xyz = pcd2xyz(keypointsB) 
        corrs_A, corrs_B = find_correspondences1(A_feats, B_feats, mutual_filter=True)
        A_corr0 = (A_xyz[:,corrs_A]).T
        B_corr0 = (B_xyz[:,corrs_B]).T 
        A_normals = A_normals0[corrs_A, :]
        B_normals = B_normals0[corrs_B, :]
        print(A_corr0.shape)
        print(B_corr0.shape)

        # remove_outlier: input = coordinates of matched points (A_corr.size = B_corr.size)
        ind = self.remove_outlier(A_corr0, B_corr0, A_normals, B_normals)
        A_corr = A_corr0[ind, :]
        B_corr = B_corr0[ind, :]
        A_pcd = o3d.geometry.PointCloud()
        B_pcd = o3d.geometry.PointCloud()
        A_pcd.points = o3d.utility.Vector3dVector(A_corr)
        B_pcd.points = o3d.utility.Vector3dVector(B_corr)
        A_pcd.normals = o3d.utility.Vector3dVector(A_normals[ind, :])
        B_pcd.normals = o3d.utility.Vector3dVector(B_normals[ind, :])

        corrs = np.arange(0, A_corr.shape[0])
        corrs = np.tile(corrs, (2, 1)).T
        print(corrs.shape)

        # solve Rt
        loss = o3d.pipelines.registration.TukeyLoss(k=0.2)
        # Rt_E = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        Rt_E = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        # result = Rt_E.compute_transformation(A_pcd, B_pcd, o3d.utility.Vector2iVector(corrs))

        new_result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(           
            A_pcd, B_pcd, o3d.utility.Vector2iVector(corrs), 2*self.VOXEL_SIZE, Rt_E, 4, 
            o3d.pipelines.registration.RANSACConvergenceCriteria(8000000, 500))
        result = new_result.transformation


        return result, A_corr, B_corr, A_corr0, B_corr0

    def draw_inliner(self, A_pcd, B_pcd, A_corr, B_corr):
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
        colors = [[0, 1, 0] for i in range(int(len(lines)))]
        colors[0:int(len(lines)/2)] = [[0, 1, 0] for i in range(int(len(lines)/2))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )

        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([A_pcd_txy, B_pcd_txy, line_set])

    def draw_edgecost(self, EdgeCost):
        c_num = EdgeCost.shape[0]
        x = []
        y = []
        for i in range(c_num):
            for j in range(c_num):
                x.append(i*c_num+j)
                y.append(EdgeCost[i,j])
        plt.scatter(x, y)        
        plt.show()

    def getVectorCos(self, v1, v2):
        if v1.dot(v1)==0 or v2.dot(v2)==0:
            return 0
        return v1.dot(v2)/np.sqrt(v1.dot(v1))/np.sqrt(v2.dot(v2))

    def myNorm(self, input):
        mx = np.max(input)
        mn = np.min(input)
        out = (input-mn)/(mx-mn)
        return out
    
    def HigherOrderDominantClustering(self, EdgeCost, MaxIter=10, sigma3=0.05):
        EdgeCost = np.exp(-EdgeCost*EdgeCost/sigma3)
        # EdgeCost = self.myNorm(EdgeCost)
        # EdgeCost = 1 - EdgeCost*EdgeCost
        sum_ec0 = np.sum(EdgeCost, axis=0)
        m_ec = np.mean(EdgeCost)
        c_num = EdgeCost.shape[0]
        x = np.ones((c_num)) / c_num
        iter = 0
        while iter < MaxIter:
            print("iter:{}".format(iter))
            Fx = np.zeros((c_num))
            for i in range(c_num):
                for j in range(c_num):
                    tmp = x[i] * x[j] * EdgeCost[i, j]
                    if tmp > 0:
                        Fx[i] = Fx[i] + tmp/x[i] #*sum_ec0[i]/sum_ec0[j]
                        Fx[j] = Fx[j] + tmp/x[j] #*sum_ec0[j]/sum_ec0[i]
            x = x*Fx
            xFx = np.sum(x)
            if xFx == 0:
                return
            x = x/xFx
            iter += 1   
        print(x.shape)
        return 1-x 
