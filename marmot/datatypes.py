#!/usr/bin/env python3

import numpy as np
import gtsam

from rclpy.time import Time

from marmot.utils import roty

class Detection():
    def __init__(self, trkr, dets_msg, det_msg, detector_name):
        # Admin
        self.timestamp = Time.from_msg(dets_msg.header.stamp)
        self.detector_name = detector_name

        # Semantic Properties
        self.metadata = det_msg.metadata
        self.det_class_str = det_msg.class_string
        self.obj_class_str = trkr.detectors[detector_name]['detection_params'][self.det_class_str]['obj_class']
        self.class_conf = det_msg.class_confidence
       
        # Spatial properties
        self.pose = det_msg.pose
        self.pos = np.array([[det_msg.pose.position.x], [det_msg.pose.position.y], [det_msg.pose.position.z]],dtype=np.float64)
        self.yaw = np.array([[np.arctan2(2*det_msg.pose.orientation.w*det_msg.pose.orientation.z, 1-2*det_msg.pose.orientation.z**2)]],dtype=np.float64)

        # If bbox info available, use it. Otherwise use default values from yaml
        if trkr.detectors[detector_name]['detector_type'] in ['pos_bbox_3d']:
            self.size = np.array([[det_msg.bbox.size.x], [det_msg.bbox.size.y], [det_msg.bbox.size.z]],dtype=np.float64)
        else:           
            self.size = np.array([[trkr.obj_props[self.obj_class_str]['length']], 
                                  [trkr.obj_props[self.obj_class_str]['width']], 
                                  [trkr.obj_props[self.obj_class_str]['height']]],dtype=np.float64)
            
        # Visual properties
        if det_msg.image_available:
            self.image_available = True
            self.image = det_msg.image
        else:
            self.image_available = False
            self.image = None

class Track():
    def __init__(self, trkr, det):

        # Admin
        self.timestamp = det.timestamp
        self.trk_id = trkr.trk_id_count
        self.metadata = det.metadata
        self.time_created = det.timestamp
        self.time_updated = det.timestamp

        # Track management
        self.track_conf = det.class_conf            

        # Semantic
        self.det_class_str = det.det_class_str # TODO - remove this if it isn't needed later on
        self.obj_class_str = det.obj_class_str
        self.class_conf = det.class_conf

        # Spatial
        self.pose = det.pose
        self.pos = det.pos
        self.yaw = det.yaw
        self.size = det.size

        # Visual
        self.image_available = det.image_available
        self.image = det.image

        # Initialize state and process model
        if trkr.obj_props[self.obj_class_str]['model_type'] in ['cp']:

            # Kalman filter & state
            self.kf = gtsam.KalmanFilter(7) # pos_x, pos_y, pos_z, yaw, length, width, height
            self.cov = np.diag(np.concatenate((trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['pos_obs_var'], 
                                               trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['yaw_obs_var'], 
                                               trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['size_obs_var'])))**2 
            self.spatial_state = self.kf.init(np.vstack((self.pos, self.yaw, self.size)), self.cov)

            # Build initial process model and noise
            self.proc_model = np.diag(np.ones(7))
            self.proc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate((trkr.obj_props[self.obj_class_str]['pos_proc_var'], 
                                                               trkr.obj_props[self.obj_class_str]['yaw_proc_var'],
                                                               trkr.obj_props[self.obj_class_str]['size_proc_var'])))

        elif trkr.obj_props[self.obj_class_str]['model_type'] in ['cvcy','cvcy_obj']:

            # Kalman filter & state
            self.kf = gtsam.KalmanFilter(10) # pos_x, pos_y, pos_z, yaw, length, width, height, vel_x, vel_y, vel_z
            self.cov = np.diag(np.concatenate((trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['pos_obs_var'], 
                                               trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['yaw_obs_var'], 
                                               trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['size_obs_var'],
                                               trkr.obj_props[self.obj_class_str]['vel_proc_var'])))**2

            self.spatial_state = self.kf.init(np.vstack((self.pos, self.yaw, self.size, np.array([[0], [0], [0]]))), self.cov)

            # Build initial process model and noise
            self.proc_model = np.diag(np.ones(10))
            self.proc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate(([0,0,0],
                                                               trkr.obj_props[self.obj_class_str]['yaw_proc_var'],
                                                               trkr.obj_props[self.obj_class_str]['size_proc_var'],
                                                               trkr.obj_props[self.obj_class_str]['vel_proc_var'])))

        elif trkr.obj_props[self.obj_class_str]['model_type'] in ['ctra']:

            # Kalman filter & state
            self.kf = gtsam.KalmanFilter(10) # pos_x, pos_y, pos_z, yaw, length, width, height, vel_x, acc_x, omega
            self.cov = np.diag(np.concatenate(([0.,0.,0.], 
                                               [0.], 
                                               trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['size_obs_var'],
                                               [0.],
                                               trkr.obj_props[self.obj_class_str]['acc_proc_var'],
                                               trkr.obj_props[self.obj_class_str]['omega_proc_var'])))**2
            self.spatial_state = self.kf.init(np.vstack((self.pos, self.yaw, self.size, np.array([[0], [0], [0]]))), self.cov)

            # Build initial process model and noise
            self.proc_model = np.diag(np.ones(10))
            self.proc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate(([0,0,0],
                                                               trkr.obj_props[self.obj_class_str]['yaw_proc_var'],
                                                               trkr.obj_props[self.obj_class_str]['size_proc_var'],
                                                               [0.],
                                                               trkr.obj_props[self.obj_class_str]['acc_proc_var'],
                                                               trkr.obj_props[self.obj_class_str]['omega_proc_var'])))

        elif trkr.obj_props[self.obj_class_str]['model_type'] in ['ack']:

            # Kalman filter & state
            self.kf = gtsam.KalmanFilter(9) # pos_x, pos_y, pos_z, yaw, length, width, height, vel_x, vel_y, vel_z
            self.cov = np.diag(np.concatenate((trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['pos_obs_var'], 
                                               trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['yaw_obs_var'], 
                                               trkr.detectors[det.detector_name]['detection_params'][self.det_class_str]['size_obs_var'],
                                               trkr.obj_props[self.obj_class_str]['vel_proc_var'],
                                               trkr.obj_props[self.obj_class_str]['curv_proc_var'])))**2
            self.spatial_state = self.kf.init(np.vstack((self.pos, self.yaw, self.size, np.array([[0], [0]]))), self.cov)

            # Build initial process model and noise
            self.proc_model = np.diag(np.ones(9))
            self.proc_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate(([0,0,0],
                                                               trkr.obj_props[self.obj_class_str]['yaw_proc_var'],
                                                               trkr.obj_props[self.obj_class_str]['size_proc_var'],
                                                               trkr.obj_props[self.obj_class_str]['vel_proc_var'],
                                                               trkr.obj_props[self.obj_class_str]['curv_proc_var'])))
        else:
            raise TypeError('No process model for type: %s' % trkr.obj_props[self.obj_class_str]['model_type'])

    def compute_proc_model(self,trkr):

        if trkr.obj_props[self.obj_class_str]['model_type'] in ['cvcy']:
            self.proc_model[0,7], self.proc_model[1,8], self.proc_model[2,9]  = self.dt, self.dt, self.dt

        elif trkr.obj_props[self.obj_class_str]['model_type'] in ['cvcy_obj']:
            self.proc_model[0,7] = np.cos(self.spatial_state.mean()[3])*self.dt
            self.proc_model[0,8] = -np.sin(self.spatial_state.mean()[3])*self.dt
            self.proc_model[1,7] = np.sin(self.spatial_state.mean()[3])*self.dt
            self.proc_model[1,8] = np.cos(self.spatial_state.mean()[3])*self.dt
            self.proc_model[2,9] = self.dt

        elif trkr.obj_props[self.obj_class_str]['model_type'] in ['ctra']:
            self.proc_model[0,7] = np.cos(self.spatial_state.mean()[3])*self.dt
            self.proc_model[1,7] = np.sin(self.spatial_state.mean()[3])*self.dt
            self.proc_model[0,8] = 0.5*np.cos(self.spatial_state.mean()[3])*self.dt**2
            self.proc_model[1,8] = 0.5*np.sin(self.spatial_state.mean()[3])*self.dt**2
            self.proc_model[3,9] = self.dt
            self.proc_model[7,8] = self.dt

        elif trkr.obj_props[self.obj_class_str]['model_type'] in ['ack']:
            self.proc_model[0,7] = np.cos(self.spatial_state.mean()[3])*self.dt
            self.proc_model[1,7] = np.sin(self.spatial_state.mean()[3])*self.dt
            self.proc_model[3,7] = self.spatial_state.mean()[8]*self.dt
        else:
            raise AttributeError('Invalid process model type.')
    
    def predict(self, trkr, stamp):
        self.dt = (Time.from_msg(stamp) - self.timestamp).nanoseconds/10**9
        self.timestamp = Time.from_msg(stamp)
        # TODO - update process noise with dt
        if trkr.obj_props[self.obj_class_str]['model_type'] in ['cp']:
            self.spatial_state = self.kf.predict(self.spatial_state,self.proc_model,np.zeros((7,7)),np.zeros((7,1)),self.proc_noise)
        elif trkr.obj_props[self.obj_class_str]['model_type'] in ['ack']:
            self.compute_proc_model(trkr)
            self.spatial_state = self.kf.predict(self.spatial_state,self.proc_model,np.zeros((9,9)),np.zeros((9,1)),self.proc_noise)            
        elif trkr.obj_props[self.obj_class_str]['model_type'] in ['cvcy','cvcy_obj','ctra']:
            self.compute_proc_model(trkr)
            self.spatial_state = self.kf.predict(self.spatial_state,self.proc_model,np.zeros((10,10)),np.zeros((10,1)),self.proc_noise)

    def update(self, det, trkr):
        # Admin
        self.metadata = det.metadata
        self.time_updated = det.timestamp

        # Track management
        if trkr.detectors[det.detector_name]['detection_params'][det.det_class_str]['score_update_function']=='multiply':
            self.track_conf = 1 - ((1 - det.class_conf)*(1 - self.track_conf))
        elif trkr.detectors[det.detector_name]['detection_params'][det.det_class_str]['score_update_function']=='parallel_add':
            self.track_conf = 1 - ((1 - det.class_conf)*(1 - self.track_conf))/((1 - det.class_conf)+(1 - self.track_conf))
        else:
            raise AttributeError('Invalid score update function.')

        # Update spatial state
        rot = gtsam.Rot3(det.pose.orientation.w,det.pose.orientation.x, det.pose.orientation.y, det.pose.orientation.z)
        det_yaw = rot.rpy()[2]

        # Correct yaw per https://github.com/xinshuoweng/AB3DMOT/blob/61f3bd72574093e367916c757b4747ca445f978c/AB3DMOT_libs/model.py
        if trkr.yaw_corr:
            # Convert detection and track yaw to range [-pi, pi]
            if abs(det_yaw)>=np.pi: det_yaw -= 2*np.pi*np.sign(det_yaw)
            if abs(self.spatial_state.mean()[3])>=np.pi: self.spatial_state.mean()[3] -= 2*np.pi*np.sign(self.spatial_state.mean()[3])

            # Ensure delta_yaw is acute angle
            if abs(det_yaw - self.spatial_state.mean()[3])>np.pi/2 and abs(det_yaw - self.spatial_state.mean()[3]) < np.pi*3/2:
                det_yaw += np.pi
                if abs(det_yaw)>=np.pi: det_yaw -= 2*np.pi*np.sign(det_yaw)
            if abs(det_yaw - self.spatial_state.mean()[3]) > np.pi*3/2:
                self.spatial_state.mean()[3] += np.pi*2*np.sign(det_yaw)

        self.spatial_state = self.kf.update(self.spatial_state, # current state
                                            trkr.detectors[det.detector_name]['obs_model'][trkr.obj_props[self.obj_class_str]['model_type']], # observation model for this object type
                                            np.vstack((det.pos, det_yaw, det.size)), # stacked detection vector
                                            trkr.detectors[det.detector_name]['detection_params'][det.det_class_str]['obs_var']) # detector variance for this detection type

        # Semantic state
        self.class_conf = det.class_conf

        # Visual
        self.image_available = det.image_available
        self.image = det.image