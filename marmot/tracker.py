#!/usr/bin/env python3

import copy 

import gtsam
import numpy as np

import rclpy
from rclpy.node import Node

from foxglove_msgs.msg import SceneUpdate, SceneEntity
from tracking_msgs.msg import Tracks3D, Track3D, Detections3D, Detection3D, TracksWithImg3D, TrackWithImg3D, DetectionsWithImg3D, DetectionWithImg3D
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Empty

from marmot.management import create_tracks, delete_tracks
from marmot.datatypes import Detection
from marmot.assignment import compute_assignment
from marmot.output import publish_tracks, publish_scene, publish_markers

class TBDTracker(Node):
    def __init__(self):

        super().__init__('tbd_tracker_node')
        self.get_logger().info("Creating tracking-by-detection tracker node")

        # Declare known process/observation models
        self.supported_proc_models = ['cp','cvcy','cvcy_obj','ctra','ack']
        self.supported_obs_models = ['pos_2d_xy','pos_3d','pos_bbox_3d']

        # Declare and set params
        self.declare_tracker_params()
        self.set_tracker_params()
        self.declare_obj_params()
        self.set_obj_properties()

        # Create publishers
        self.declare_parameter('tracker.publishers.names', rclpy.Parameter.Type.STRING_ARRAY)
        self.pub_names = self.get_parameter('tracker.publishers.names').get_parameter_value().string_array_value
        self.pubs = dict()
        for pub in self.pub_names:
            pub_dict = dict()
            self.declare_parameter('tracker.publishers.' + pub + '.pub_topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('tracker.publishers.' + pub + '.msg_type', rclpy.Parameter.Type.STRING)
            self.declare_parameter('tracker.publishers.' + pub + '.function', rclpy.Parameter.Type.STRING)
            self.declare_parameter('tracker.publishers.' + pub + '.queue_size', rclpy.Parameter.Type.INTEGER)
            pub_dict['topic'] = self.get_parameter('tracker.publishers.' + pub + '.pub_topic').get_parameter_value().string_value
            pub_dict['msg_type'] = self.get_parameter('tracker.publishers.' + pub + '.msg_type').get_parameter_value().string_value
            pub_dict['function'] = self.get_parameter('tracker.publishers.' + pub + '.function').get_parameter_value().string_value
            pub_dict['queue_size'] = self.get_parameter('tracker.publishers.' + pub + '.queue_size').get_parameter_value().integer_value
        
            self.pubs[pub] = (pub_dict)
            exec('self.%s  = self.create_publisher(%s,\'~/%s\',%s)' % (pub, pub_dict['msg_type'], pub_dict['topic'], pub_dict['queue_size']))

        # Generate detector models from .yaml
        self.declare_parameter('detectors.detector_names', rclpy.Parameter.Type.STRING_ARRAY)
        self.detector_names = self.get_parameter('detectors.detector_names').get_parameter_value().string_array_value
        self.detectors = dict()
        self.subs = []

        for detector in self.detector_names:

            # Declare parameters for detector
            self.declare_parameter('detectors.' + detector + '.topic', rclpy.Parameter.Type.STRING)
            self.declare_parameter('detectors.' + detector + '.msg_type', rclpy.Parameter.Type.STRING)
            self.declare_parameter('detectors.' + detector + '.detector_type',rclpy.Parameter.Type.STRING)
            self.declare_parameter('detectors.' + detector + '.detection_classes', rclpy.Parameter.Type.STRING_ARRAY)
            self.declare_parameter('detectors.' + detector + '.detection_classes_ignore', rclpy.Parameter.Type.STRING_ARRAY)

            # Form parameter dictionary for detector
            self.detectors[detector] = dict()
            self.detectors[detector]['topic'] = self.get_parameter('detectors.' + detector + '.topic').get_parameter_value().string_value
            self.detectors[detector]['msg_type'] = self.get_parameter('detectors.' + detector + '.msg_type').get_parameter_value().string_value
            self.detectors[detector]['detector_type'] = self.get_parameter('detectors.' + detector + '.detector_type').get_parameter_value().string_value
            self.detectors[detector]['detection_classes'] = self.get_parameter('detectors.' + detector + '.detection_classes').get_parameter_value().string_array_value
            self.detectors[detector]['detection_classes_ignore'] = self.get_parameter('detectors.' + detector + '.detection_classes_ignore').get_parameter_value().string_array_value

            self.detectors[detector]['detection_params'] = dict()

            for det_cls in self.detectors[detector]['detection_classes']:

                if det_cls in self.detectors[detector]['detection_classes_ignore']:
                    continue

                self.detectors[detector]['detection_params'][det_cls] = dict()

                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.sim_metric', rclpy.Parameter.Type.STRING)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.match_thresh', rclpy.Parameter.Type.DOUBLE)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.create_method', rclpy.Parameter.Type.STRING)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.active_thresh', rclpy.Parameter.Type.DOUBLE)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.detect_thresh', rclpy.Parameter.Type.DOUBLE)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.score_decay', rclpy.Parameter.Type.DOUBLE)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.score_update_function', rclpy.Parameter.Type.STRING)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.delete_method', rclpy.Parameter.Type.STRING)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.delete_thresh', rclpy.Parameter.Type.DOUBLE)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.pos_obs_var',rclpy.Parameter.Type.DOUBLE_ARRAY)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.yaw_obs_var',rclpy.Parameter.Type.DOUBLE_ARRAY)
                self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.object_class',rclpy.Parameter.Type.STRING)
                
                self.detectors[detector]['detection_params'][det_cls]['pos_obs_var'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.pos_obs_var').get_parameter_value().double_array_value
                self.detectors[detector]['detection_params'][det_cls]['yaw_obs_var'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.yaw_obs_var').get_parameter_value().double_array_value
                self.detectors[detector]['detection_params'][det_cls]['obj_class'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.object_class').get_parameter_value().string_value

                # Create management 
                self.detectors[detector]['detection_params'][det_cls]['sim_metric'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.sim_metric').get_parameter_value().string_value
                self.detectors[detector]['detection_params'][det_cls]['match_thresh'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.match_thresh').get_parameter_value().double_value

                # Set management-specific parameters
                self.detectors[detector]['detection_params'][det_cls]['create_method'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.create_method').get_parameter_value().string_value
                if self.detectors[detector]['detection_params'][det_cls]['create_method'] == 'conf':
                    self.detectors[detector]['detection_params'][det_cls]['active_thresh'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.active_thresh').get_parameter_value().double_value
                    self.detectors[detector]['detection_params'][det_cls]['detect_thresh'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.detect_thresh').get_parameter_value().double_value
                    self.detectors[detector]['detection_params'][det_cls]['score_decay'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.score_decay').get_parameter_value().double_value
                    self.detectors[detector]['detection_params'][det_cls]['score_update_function'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.score_update_function').get_parameter_value().string_value
                else:
                    raise TypeError('No track creation method: %s' % self.detectors[detector]['detection_params'][det_cls]['create_method'])

                self.detectors[detector]['detection_params'][det_cls]['delete_method'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.delete_method').get_parameter_value().string_value
                if self.detectors[detector]['detection_params'][det_cls]['delete_method'] == 'conf':
                    self.detectors[detector]['detection_params'][det_cls]['delete_thresh'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.delete_thresh').get_parameter_value().double_value
                else:
                    raise TypeError('No track deletion method: %s' % self.detectors[detector]['detection_params'][det_cls]['delete_method'] )

                # Create observation variance model based on detector type
                if self.detectors[detector]['detector_type'] == 'pos_bbox_3d':
                    self.declare_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.size_obs_var',rclpy.Parameter.Type.DOUBLE_ARRAY)
                    self.detectors[detector]['detection_params'][det_cls]['size_obs_var'] = self.get_parameter('detectors.' + detector + '.detection_properties.' + det_cls + '.size_obs_var').get_parameter_value().double_array_value
                    self.detectors[detector]['detection_params'][det_cls]['obs_var'] = gtsam.noiseModel.Diagonal.Variances(np.concatenate((self.detectors[detector]['detection_params'][det_cls]['pos_obs_var'], 
                                                                                                                                  self.detectors[detector]['detection_params'][det_cls]['yaw_obs_var'], 
                   
                                                                                                                                  self.detectors[detector]['detection_params'][det_cls]['size_obs_var'])))
                elif self.detectors[detector]['detector_type'] == 'pos_3d':
                    self.detectors[detector]['detection_params'][det_cls]['size_obs_var'] = np.array([1., 1., 1.])
                    self.detectors[detector]['detection_params'][det_cls]['obs_var'] = gtsam.noiseModel.Diagonal.Variances(np.concatenate((self.detectors[detector]['detection_params'][det_cls]['pos_obs_var'], 
                                                                                                                                  self.detectors[detector]['detection_params'][det_cls]['yaw_obs_var'], 
                                                                                                                                  self.detectors[detector]['detection_params'][det_cls]['size_obs_var'])))
                elif self.detectors[detector]['detector_type'] == 'pos_2d_xy':
                    self.detectors[detector]['detection_params'][det_cls]['size_obs_var'] = np.array([1., 1., 1.])
                    self.detectors[detector]['detection_params'][det_cls]['obs_var'] = gtsam.noiseModel.Diagonal.Variances(np.concatenate((self.detectors[detector]['detection_params'][det_cls]['pos_obs_var'], 
                                                                                                                                  self.detectors[detector]['detection_params'][det_cls]['yaw_obs_var'], 
                                                                                                                                  self.detectors[detector]['detection_params'][det_cls]['size_obs_var'])))
                else:
                    raise TypeError('No observation model for detector type: %s' % self.detectors[detector]['detector_type'])
            
            # Add observation models for each process/sensor model combination
            self.detectors[detector]['obs_model'] = dict()
            for proc_model in self.supported_proc_models:

                if proc_model in ['cp']:
                    dim_states = 7
                if proc_model in ['ack']:
                    dim_states = 9
                elif proc_model in ['cvcy', 'cvcy_obj','ctra']:
                    dim_states = 10

                if self.detectors[detector]['detector_type'] == 'pos_3d':
                    dim_obs = 7
                    self.detectors[detector]['obs_model'][proc_model] = np.zeros((dim_obs, dim_states))
                    self.detectors[detector]['obs_model'][proc_model][0,0], self.detectors[detector]['obs_model'][proc_model][1,1], self.detectors[detector]['obs_model'][proc_model][2,2] = 1,1,1

                elif self.detectors[detector]['detector_type'] == 'pos_2d_xy':
                    dim_obs = 7
                    self.detectors[detector]['obs_model'][proc_model] = np.zeros((dim_obs, dim_states))
                    self.detectors[detector]['obs_model'][proc_model][0,0], self.detectors[detector]['obs_model'][proc_model][1,1] = 1,1

                elif self.detectors[detector]['detector_type'] == 'pos_bbox_3d':
                    dim_obs = 7
                    self.detectors[detector]['obs_model'][proc_model] = np.zeros((dim_obs, dim_states))
                    self.detectors[detector]['obs_model'][proc_model][0,0] = 1
                    self.detectors[detector]['obs_model'][proc_model][1,1] = 1
                    self.detectors[detector]['obs_model'][proc_model][2,2] = 1
                    self.detectors[detector]['obs_model'][proc_model][3,3] = 1
                    self.detectors[detector]['obs_model'][proc_model][4,4] = 1
                    self.detectors[detector]['obs_model'][proc_model][5,5] = 1
                    self.detectors[detector]['obs_model'][proc_model][6,6] = 1

            # Create ROS2 subscription for this detector
            self.subs.append(self.create_subscription(eval(self.detectors[detector]['msg_type']),self.detectors[detector]['topic'], eval("lambda msg: self.det_callback(msg, \"" + detector + "\")",locals()), 1))

        # Initialize tracker Track and Detection set variables
        self.dets_msg = Detections3D()
        self.trk_id_count = 0
        self.dets = []
        self.trks = []

        # Assignment variables
        self.cost_matrix = np.empty(0)

        # Create publisher objects and empty messages
        self.trks_msg = Tracks3D()
        self.scene_msg = SceneUpdate()
        self.marker_msg = MarkerArray()

        # Declare services
        self.reset_srv = self.create_service(Empty, '~/reset_tracker', self.reset_tracker)
        self.reconfigure_srv = self.create_service(Empty, '~/reconfigure_tracker', self.reconfigure_tracker)

    def declare_tracker_params(self):
        # Configure tracker params from .yaml
        self.declare_parameter('tracker.frame_id', rclpy.Parameter.Type.STRING)
        self.declare_parameter('tracker.mismatch_penalty', rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter('tracker.assignment_algo', rclpy.Parameter.Type.STRING)
        self.declare_parameter('tracker.yaw_corr', rclpy.Parameter.Type.BOOL)
        self.declare_parameter('object_properties.object_classes', rclpy.Parameter.Type.STRING_ARRAY)

    def set_tracker_params(self):
        self.frame_id = self.get_parameter('tracker.frame_id').get_parameter_value().string_value
        self.mismatch_penalty = self.get_parameter('tracker.mismatch_penalty').get_parameter_value().double_value 
        self.assignment_algo = self.get_parameter('tracker.assignment_algo').get_parameter_value().string_value
        self.yaw_corr = self.get_parameter('tracker.yaw_corr').get_parameter_value().bool_value

    def declare_obj_params(self):

        self.obj_classes = self.get_parameter('object_properties.object_classes').get_parameter_value().string_array_value

        for obj_name in self.obj_classes:

            self.declare_parameter('object_properties.' + obj_name + '.length', rclpy.Parameter.Type.DOUBLE)
            self.declare_parameter('object_properties.' + obj_name + '.width', rclpy.Parameter.Type.DOUBLE)
            self.declare_parameter('object_properties.' + obj_name + '.height', rclpy.Parameter.Type.DOUBLE)       

            self.declare_parameter('object_properties.' + obj_name + '.model_type', rclpy.Parameter.Type.STRING)
            self.declare_parameter('object_properties.' + obj_name + '.yaw_proc_var', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter('object_properties.' + obj_name + '.size_proc_var', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter('object_properties.' + obj_name + '.pos_proc_var', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter('object_properties.' + obj_name + '.vel_proc_var', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter('object_properties.' + obj_name + '.acc_proc_var', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter('object_properties.' + obj_name + '.omega_proc_var', rclpy.Parameter.Type.DOUBLE_ARRAY)
            self.declare_parameter('object_properties.' + obj_name + '.curv_proc_var', rclpy.Parameter.Type.DOUBLE_ARRAY)

    def set_obj_properties(self):

        self.obj_classes = self.get_parameter('object_properties.object_classes').get_parameter_value().string_array_value

        self.obj_props = dict()
        
        for obj_name in self.obj_classes:

            temp_dict = dict()
            temp_dict['model_type'] = self.get_parameter('object_properties.' + obj_name + '.model_type').get_parameter_value().string_value
            temp_dict['length'] = self.get_parameter('object_properties.' + obj_name + '.length').get_parameter_value().double_value
            temp_dict['width'] = self.get_parameter('object_properties.' + obj_name + '.width').get_parameter_value().double_value
            temp_dict['height'] = self.get_parameter('object_properties.' + obj_name + '.height').get_parameter_value().double_value

            # Add process model-specific parameters
            if temp_dict['model_type'] in ['cp']:
                temp_dict['yaw_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.yaw_proc_var').get_parameter_value().double_array_value
                temp_dict['size_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.size_proc_var').get_parameter_value().double_array_value
                temp_dict['pos_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.pos_proc_var').get_parameter_value().double_array_value
            elif temp_dict['model_type'] in ['cvcy', 'cvcy_obj']:
                temp_dict['yaw_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.yaw_proc_var').get_parameter_value().double_array_value
                temp_dict['size_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.size_proc_var').get_parameter_value().double_array_value
                temp_dict['vel_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.vel_proc_var').get_parameter_value().double_array_value
            elif temp_dict['model_type'] in ['ctra']:
                temp_dict['yaw_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.yaw_proc_var').get_parameter_value().double_array_value
                temp_dict['size_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.size_proc_var').get_parameter_value().double_array_value
                temp_dict['acc_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.acc_proc_var').get_parameter_value().double_array_value   
                temp_dict['omega_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.omega_proc_var').get_parameter_value().double_array_value                
            elif temp_dict['model_type'] in ['ack']:
                temp_dict['yaw_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.yaw_proc_var').get_parameter_value().double_array_value
                temp_dict['size_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.size_proc_var').get_parameter_value().double_array_value
                temp_dict['vel_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.vel_proc_var').get_parameter_value().double_array_value
                temp_dict['curv_proc_var'] = self.get_parameter('object_properties.' + obj_name + '.curv_proc_var').get_parameter_value().double_array_value 
            else:
                raise TypeError('No process model for type: %s' % temp_dict['model_type'])

            self.obj_props[obj_name] = temp_dict

    def reset_tracker(self, _, resp):

        self.get_logger().info("Resetting tracker")

        # Clear track, detection, and assignment variables
        self.dets_msg = Detections3D()
        self.trk_id_count = 0
        self.dets = []
        self.trks = []

        self.cost_matrix = np.empty(0)
        self.matches = np.empty(0)

        return resp

    def reconfigure_tracker(self, _, resp):

        self.get_logger().info("Reconfiguring tracker")

        # Configure tracker from .yaml
        self.obj_classes = self.get_parameter('object_properties.object_classes').get_parameter_value().string_array_value
        self.frame_id = self.get_parameter('tracker.frame_id').get_parameter_value().string_value

        # Reconfigure tracker and object properties
        self.set_tracker_params()
        self.set_obj_properties()

        # Generate detector models from .yaml
        # TODO 

        return resp

    def predict_tracks(self):
        for trk in self.trks:
            trk.predict(self, self.dets_msg.header.stamp)

    def update_tracks(self, detector_name):
        for match in self.matches:
            # self.get_logger().info(f"{detector_name} callback: Track {self.trks[match[1]].trk_id} ({self.trks[match[1]].track_conf})% at ({self.trks[match[1]].pos[0]},{self.trks[match[1]].pos[1]},{self.trks[match[1]].pos[2]}) matched to det at ({self.dets[match[0]].pos[0]},{self.dets[match[0]].pos[1]},{self.dets[match[0]].pos[2]})")
            self.trks[match[1]].update(self.dets[match[0]],self)
            # self.get_logger().info(f"Post-update conf: {self.trks[match[1]].track_conf}")

    def det_callback(self, det_array_msg, detector_name):
       
        # POPULATE detections list from detections message
        self.dets_msg = det_array_msg
        self.dets = []
        for det in self.dets_msg.detections:
            if det.class_string in self.detectors[detector_name]['detection_classes_ignore']:
                continue
            self.dets.append(Detection(self, self.dets_msg, det, detector_name))

        # PREDICT existing tracks
        self.predict_tracks()

        # ASSIGN detections to tracks
        compute_assignment(self, detector_name)

        # UPDATE tracks with assigned detections
        self.update_tracks(detector_name)

        # UPDATE unmatched tracks (missed detections)
        for i, trk in enumerate(self.trks):

            # Disregard if the object is not capable of being detected by the detector
            if trk.det_class_str not in self.detectors[detector_name]['detection_classes']:
                continue

            if i not in self.matches[:,1]: # If track is unmatched, handle it as a missed detection
                # self.get_logger().info(f"Track {trk.trk_id} at ({trk.pos[0]},{trk.pos[1]},{trk.pos[2]})- no detections. Confidence {trk.track_conf}")

                trk.metadata = det_array_msg.metadata

                if self.detectors[detector_name]['detection_params'][trk.det_class_str]['delete_method'] == 'conf':
                    trk.track_conf *= (1 - self.detectors[detector_name]['detection_params'][trk.det_class_str]['score_decay'])
                else:
                    raise TypeError('Invalid track creation method: %s' % self.detectors[detector_name]['detection_params'][trk.det_class_str]['create_method'])                    

                # self.get_logger().info(f"Track {trk.trk_id} - Posterior confidence {trk.track_conf}")

        # Manage unmatched tracks and detections
        delete_tracks(self)
        create_tracks(self, detector_name)

        # OUTPUT tracker results
        for pub_name in self.pub_names:
            exec('%s(self,\'%s\')' % (self.pubs[pub_name]['function'],pub_name))