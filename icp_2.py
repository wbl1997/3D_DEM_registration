import math
import numpy as np
import re
import transformations as transform
from helpers import *
from Transformation import *

def icp_point_to_point_lm(source_points, dest_points, initial, loop=0):
    """
    Point to point matching using Gauss-Newton
    
    source_points:  nx3 matrix of n 3D points
    dest_points: nx3 matrix of n 3D points, which have been obtained by some rigid deformation of 'source_points'
    initial: 1x6 matrix, denoting alpha, beta, gamma (the Euler angles for rotation and tx, ty, tz (the translation along three axis). 
                this is the initial estimate of the transformation between 'source_points' and 'dest_points'
    loop: start with zero, to keep track of the number of times it loops, just a very crude way to control the recursion            
                
    """
    
    J = []
    e = []
    while loop<50:
        for i in range (0,dest_points.shape[0]-1):
            #print dest_points[i][3],dest_points[i][4],dest_points[i][5]
            dx = dest_points[i][0]
            dy = dest_points[i][1]
            dz = dest_points[i][2]
            
            sx = source_points[i][0]
            sy = source_points[i][1]
            sz = source_points[i][2]
            
            alpha = initial[0][0]
            beta = initial[1][0]
            gamma = initial[2][0]
            tx = initial[3][0]        
            ty = initial[4][0]
            tz = initial[5][0]
            #print alpha
            
            a1 = (-2*beta*sx*sy) - (2*gamma*sx*sz) + (2*alpha*((sy*sy) + (sz*sz))) + (2*((sz*dy) - (sy*dz))) + 2*((sy*tz) - (sz*ty))
            a2 = (-2*alpha*sx*sy) - (2*gamma*sy*sz) + (2*beta*((sx*sx) + (sz*sz))) + (2*((sx*dz) - (sz*dx))) + 2*((sz*tx) - (sx*tz))
            a3 = (-2*alpha*sx*sz) - (2*beta*sy*sz) + (2*gamma*((sx*sx) + (sy*sy))) + (2*((sy*dx) - (sx*dy))) + 2*((sx*ty) - (sy*tx))
            a4 = 2*(sx - (gamma*sy) + (beta*sz) +tx -dx)
            a5 = 2*(sy - (alpha*sz) + (gamma*sx) +ty -dy)
            a6 = 2*(sz - (beta*sx) + (alpha*sy) +tz -dz)
            
            _residual = (a4*a4/4)+(a5*a5/4)+(a6*a6/4)
            
            _J = np.array([a1, a2, a3, a4, a5, a6])
            _e = np.array([_residual])
            
            J.append(_J)
            e.append(_e)
            
        jacobian = np.array(J)
        residual = np.array(e)
        
        update = -np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(jacobian),jacobian)),np.transpose(jacobian)),residual)
        
        #print update, initial
        
        initial = initial + update
        
        print(np.transpose(initial))
        
        loop = loop + 1

        if np.sum(np.abs(update))<0.0001:
            break
    
    initial = np.squeeze(initial)
    R = rot_mat(initial[:3])
    t = initial[3:]
    return R, t


def icp_point_to_plane_lm(source_points, dest_points, initial, loop, dest_normals=None):
    """
    Point to plane matching using Gauss Newton
    
    source_points:  nx3 matrix of n 3D points
    dest_points: nx6 matrix of n 3D points + 3 normal vectors, which have been obtained by some rigid deformation of 'source_points'
    initial: 1x6 matrix, denoting alpha, beta, gamma (the Euler angles for rotation and tx, ty, tz (the translation along three axis). 
                this is the initial estimate of the transformation between 'source_points' and 'dest_points'
    loop: start with zero, to keep track of the number of times it loops, just a very crude way to control the recursion            
                
    """
    while loop<10:
        J = []
        e = []
        for i in range (0,dest_points.shape[0]-1):
            dx = dest_points[i][0]
            dy = dest_points[i][1]
            dz = dest_points[i][2]
            nx = dest_normals[i][0]
            ny = dest_normals[i][1]
            nz = dest_normals[i][2]
            
            sx = source_points[i][0]
            sy = source_points[i][1]
            sz = source_points[i][2]
            
            alpha = initial[0]
            beta = initial[1]
            gamma = initial[2]
            tx = initial[3]      
            ty = initial[4]
            tz = initial[5]
            
            a1 = (nz*sy) - (ny*sz)
            a2 = (nx*sz) - (nz*sx)
            a3 = (ny*sx) - (nx*sy)
            a4 = nx
            a5 = ny
            a6 = nz
            
            _residual = (alpha*a1) + (beta*a2) + (gamma*a3) + (nx*tx) + (ny*ty) + (nz*tz) - (((nx*dx) + (ny*dy) + (nz*dz)) - ((nx*sx) + (ny*sy) + (nz*sz)))
        
            _J = np.array([a1, a2, a3, a4, a5, a6])
            _e = np.array([_residual])
            
            J.append(_J)
            e.append(_e)
            
        A = np.array(J)
        r = np.array(e)
        # print(A.shape)
        # print(r.shape)

        update = np.squeeze(-np.linalg.inv(A.T@A) @ A.T@r)
        # print(update)
    
        initial = initial + update
        loop = loop + 1

        if np.sqrt(r.T@r)/r.shape[0]<0.01:
            print(loop)
            break
    
    R = rot_mat(initial[:3])
    t = initial[3:]
    return R, t