from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
from numba import jit
from pytorch3d.ops import box3d_overlap
from scipy.spatial import ConvexHull

@jit
def iou3d_points(x, y, z, l, w, h, yaw):
    return np.array([[x-(l/2)*np.cos(yaw) - (w/2)*np.sin(yaw), y - (l/2)*np.sin(yaw) + (w/2)*np.cos(yaw), z+h/2],
                       [x-(l/2)*np.cos(yaw) + (w/2)*np.sin(yaw), y - (l/2)*np.sin(yaw) - (w/2)*np.cos(yaw), z+h/2],
                       [x-(l/2)*np.cos(yaw) + (w/2)*np.sin(yaw), y - (l/2)*np.sin(yaw) - (w/2)*np.cos(yaw), z-h/2],
                       [x-(l/2)*np.cos(yaw) - (w/2)*np.sin(yaw), y - (l/2)*np.sin(yaw) + (w/2)*np.cos(yaw), z-h/2],
                       [x+(l/2)*np.cos(yaw) - (w/2)*np.sin(yaw), y + (l/2)*np.sin(yaw) + (w/2)*np.cos(yaw), z+h/2],
                       [x+(l/2)*np.cos(yaw) + (w/2)*np.sin(yaw), y + (l/2)*np.sin(yaw) - (w/2)*np.cos(yaw), z+h/2],
                       [x+(l/2)*np.cos(yaw) + (w/2)*np.sin(yaw), y + (l/2)*np.sin(yaw) - (w/2)*np.cos(yaw), z-h/2],
                       [x+(l/2)*np.cos(yaw) - (w/2)*np.sin(yaw), y + (l/2)*np.sin(yaw) + (w/2)*np.cos(yaw), z-h/2]]).reshape(1,8,3)
@jit
def dist_2d(det_pos, track_pos):
    # 2D Euclidean distance between positions
    return np.linalg.norm(det_pos[0:2] - track_pos[0:2])

@jit
def dist_3d(det_pos, track_pos):
    # 3D Euclidean distance between positions
    return np.linalg.norm(det_pos - track_pos)

@jit
def iou_3d(det_points, trk_points):
    int_vol, iou3d = box3d_overlap(torch.as_tensor(det_points,dtype=torch.float), torch.as_tensor(trk_points,dtype=torch.float))
    return int_vol, iou3d

@jit
def union_vol(det_l, det_w, det_h, trk_l, trk_w, trk_h, int_vol):
    # if int_vol==0.:
    #     return det_l*det_w*det_h + trk_l*trk_w*trk_h
    # else:
    #     return int_vol/iou
    return (det_l*det_w*det_h + trk_l*trk_w*trk_h) - int_vol


def greedy_matching(cost_matrix):
    # association in the greedy manner
    # refer to https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking/blob/master/main.py
    # and also https://github.com/xinshuoweng/AB3DMOT/blob/61f3bd72574093e367916c757b4747ca445f978c/AB3DMOT_libs/matching.py 

    num_dets, num_trks = cost_matrix.shape[0], cost_matrix.shape[1]

    # sort all costs and then convert to 2D
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    # assign matches one by one given the sorting, but first come first serves
    det_matches_to_trk = [-1] * num_dets
    trk_matches_to_det = [-1] * num_trks
    matched_indices = []
    for sort_i in range(index_2d.shape[0]):
        det_id = int(index_2d[sort_i][0])
        trk_id = int(index_2d[sort_i][1])

        # if both id has not been matched yet
        if trk_matches_to_det[trk_id] == -1 and det_matches_to_trk[det_id] == -1:
            trk_matches_to_det[trk_id] = det_id
            det_matches_to_trk[det_id] = trk_id
            matched_indices.append([det_id, trk_id])

    return np.asarray(matched_indices)

def hungarian_matching(cost_matrix):
    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    # tracker.det_asgn_idx, tracker.trk_asgn_idx = list(row_idx), list(col_idx)
    return np.stack((row_idx, col_idx), axis=1)

def compute_cost_matrix(tracker,detector_name):

    tracker.cost_matrix = np.zeros((len(tracker.dets),len(tracker.trks)))
    
    for ii,det in enumerate(tracker.dets):
        for jj,trk in enumerate(tracker.trks):
            
			# If classes don't match, add mismatch penalty and continue
            if trk.obj_class_str != det.obj_class_str:
                tracker.cost_matrix[ii,jj] += tracker.mismatch_penalty
                continue

            # If classes do match, compute cost/affinity as appropriate and assign to cost matrix
            if tracker.detectors[detector_name]['detection_params'][trk.obj_class_str]['sim_metric']=='dist_2d':
                tracker.cost_matrix[ii,jj] += dist_2d(det.pos[:,0], trk.spatial_state.mean()[0:3])
                
            elif tracker.detectors[detector_name]['detection_params'][trk.obj_class_str]['sim_metric']=='dist_3d':
                tracker.cost_matrix[ii,jj] += dist_3d(det.pos[:,0], trk.spatial_state.mean()[0:3])
            
            elif tracker.detectors[detector_name]['detection_params'][trk.obj_class_str]['sim_metric']=='iou_3d':
                det_points = iou3d_points(det.pos[0,0],det.pos[1,0],det.pos[2,0],det.size[0,0],det.size[1,0],det.size[2,0], det.yaw[0,0])
                trk_points = iou3d_points(trk.spatial_state.mean()[0],trk.spatial_state.mean()[1],trk.spatial_state.mean()[2],trk.spatial_state.mean()[4],trk.spatial_state.mean()[5],trk.spatial_state.mean()[6], trk.spatial_state.mean()[3])
                _, iou3d = box3d_overlap(torch.as_tensor(det_points,dtype=torch.float), torch.as_tensor(trk_points,dtype=torch.float))
                tracker.cost_matrix[ii,jj] -= iou3d
                
            elif tracker.detectors[detector_name]['detection_params'][trk.obj_class_str]['sim_metric']=='giou_3d':
                det_points = iou3d_points(det.pos[0,0],det.pos[1,0],det.pos[2,0],det.size[0,0],det.size[1,0],det.size[2,0], det.yaw[0,0])
                trk_points = iou3d_points(trk.spatial_state.mean()[0],trk.spatial_state.mean()[1],trk.spatial_state.mean()[2],trk.spatial_state.mean()[4],trk.spatial_state.mean()[5],trk.spatial_state.mean()[6], trk.spatial_state.mean()[3])
                int_vol, iou3d = box3d_overlap(torch.as_tensor(det_points,dtype=torch.float,device='cuda:0'), torch.as_tensor(trk_points,dtype=torch.float,device='cuda:0'))
                
                un_vol = union_vol(det.size[0,0], det.size[1,0], det.size[2,0], trk.spatial_state.mean()[4], trk.spatial_state.mean()[5], trk.spatial_state.mean()[6], int_vol[0,0].item())
                
                conv_hull = ConvexHull(np.vstack((det_points[0,:,:],trk_points[0,:,:])))
                tracker.cost_matrix[ii,jj] -= (iou3d - (conv_hull.volume - un_vol)/conv_hull.volume)

            else:
                raise TypeError('Invalid similarity metric: %s' % tracker.detectors[detector_name]['detection_params'][trk.obj_class_str]['sim_metric'])

def solve_cost_matrix(tracker, detector_name):
    
    # Compute assignment vectors: each vector contains indices of matched detections and matched tracks
    if tracker.assignment_algo == 'hungarian':
        tracker.matched_indices = hungarian_matching(tracker.cost_matrix) 	
        
    elif tracker.assignment_algo == 'greedy':
        tracker.matched_indices = greedy_matching(tracker.cost_matrix) 		
    
    else:
        raise TypeError('Invalid assignment algorithm: %s' % tracker.assignment_algo)

	# Do not consider matches with cost >= object cost threshold
    # cost matrix: rows = detection index, columns = track index
    tracker.matches = []
    for m in tracker.matched_indices:
        
        if tracker.trks[m[1]].obj_class_str not in tracker.detectors[detector_name]['detection_classes']:
            continue          

        if (tracker.cost_matrix[m[0], m[1]] < tracker.detectors[detector_name]['detection_params'][tracker.trks[m[1]].obj_class_str]['match_thresh']):
            tracker.matches.append(m.reshape(1, 2))
            
    if len(tracker.matches) == 0: 
        tracker.matches = np.empty((0, 2),dtype=int)
    
    else: tracker.matches = np.concatenate(tracker.matches, axis=0)

def compute_assignment(tracker,detector_name):    
    compute_cost_matrix(tracker,detector_name)
    solve_cost_matrix(tracker,detector_name)