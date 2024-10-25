import numpy as np

from diagnostic_msgs.msg import KeyValue
from tracking_msgs.msg import Track3D, Tracks3D
from visualization_msgs.msg import Marker, MarkerArray
from foxglove_msgs.msg import SceneEntity, SceneUpdate, ArrowPrimitive, CubePrimitive, TextPrimitive, KeyValuePair

def is_track_active(tracker, track):

    for detector in tracker.detectors.keys():

        if track.obj_class_str not in tracker.detectors[detector]['detection_classes']:
            continue

        if (tracker.detectors[detector]['detection_params'][track.obj_class_str]['create_method']=='conf' and 
              track.track_conf >= tracker.detectors[detector]['detection_params'][track.obj_class_str]['active_thresh']):
            track_is_active = True
            break
        else:
            track_is_active = False

    return track_is_active

def publish_tracks(tracker, pub_name):
    tracker.trks_msg = Tracks3D()
    tracker.trks_msg.header.frame_id = tracker.frame_id
    tracker.trks_msg.header.stamp = tracker.dets_msg.header.stamp
    tracker.trks_msg.metadata = tracker.dets_msg.metadata

    for trk in tracker.trks:

        # Check if tracklet meets track creation criteria
        if is_track_active(tracker, trk)==False:
            continue
        
        # Create track message
        trk_msg = Track3D()

        # Add track information to message
        trk_msg.track_id = trk.trk_id
        trk_msg.track_confidence = trk.track_conf
        trk_msg.time_created = trk.time_created.to_msg()
        trk_msg.time_updated = trk.time_updated.to_msg()

        # Populate track message for publication
        trk_msg.pose.pose.position.x = trk.spatial_state.mean()[0]
        trk_msg.pose.pose.position.y = trk.spatial_state.mean()[1]
        trk_msg.pose.pose.position.z = trk.spatial_state.mean()[2]

        # compute orientation from rpy
        cr = 1.
        sr = 0.
        cp = 1.
        sp = 0.
        cy = np.cos(trk.spatial_state.mean()[3])
        sy = np.sin(trk.spatial_state.mean()[3])
        trk_msg.pose.pose.orientation.w = cr*cp*cy + sr*sp*sy
        trk_msg.pose.pose.orientation.x = sr*cp*cy - cr*sp*sy
        trk_msg.pose.pose.orientation.y = cr*sp*cy + sr*cp*sy
        trk_msg.pose.pose.orientation.z = cr*cp*sy - sr*sp*cy
    
        trk_msg.bbox.size.x = trk.spatial_state.mean()[4]
        trk_msg.bbox.size.y = trk.spatial_state.mean()[5]
        trk_msg.bbox.size.z =  trk.spatial_state.mean()[6]

        # CP state variables: pos_x, pos_y, pos_z, yaw, length, width, height
        if tracker.obj_props[trk.obj_class_str]['model_type'] in ['cp']:
            # Add spatial information to message
            trk_msg.twist.twist.linear.x = 0.
            trk_msg.twist.twist.linear.y = 0.
            trk_msg.twist.twist.linear.z = 0.

        # pos_x, pos_y, pos_z, yaw, length, width, height, vel_x, vel_y, vel_z
        elif tracker.obj_props[trk.obj_class_str]['model_type'] in ['cvcy']:
            # Add spatial information to message
            trk_msg.twist.twist.linear.x = trk.spatial_state.mean()[7]
            trk_msg.twist.twist.linear.y = trk.spatial_state.mean()[8]
            trk_msg.twist.twist.linear.z = trk.spatial_state.mean()[9]

        elif tracker.obj_props[trk.obj_class_str]['model_type'] in ['cvcy_obj']:
            # Add spatial information to message
            trk_msg.twist.twist.linear.z = trk.spatial_state.mean()[9]
            trk_msg.twist.twist.linear.x = trk.spatial_state.mean()[7]*cy - trk.spatial_state.mean()[8]*sy
            trk_msg.twist.twist.linear.y = trk.spatial_state.mean()[7]*sy + trk.spatial_state.mean()[8]*cy 
        elif tracker.obj_props[trk.obj_class_str]['model_type'] in ['ctra']:
            # Add spatial information to message
            trk_msg.twist.twist.linear.z = 0.
            trk_msg.twist.twist.angular.z = trk.spatial_state.mean()[9]
            trk_msg.twist.twist.linear.x = trk.spatial_state.mean()[7]*cy
            trk_msg.twist.twist.linear.y = trk.spatial_state.mean()[7]*sy
        elif tracker.obj_props[trk.obj_class_str]['model_type'] in ['ack']:
            # Add spatial information to message
            trk_msg.twist.twist.linear.z = 0.
            trk_msg.twist.twist.angular.z = trk.spatial_state.mean()[7]*trk.spatial_state.mean()[8]
            trk_msg.twist.twist.linear.x = trk.spatial_state.mean()[7]*cy
            trk_msg.twist.twist.linear.y = trk.spatial_state.mean()[7]*sy
        else:
            raise AttributeError('Invalid process model type.')
    
        # Add semantic information to message
        trk_msg.track_confidence = float(trk.track_conf)
        trk_msg.class_confidence = float(trk.class_conf)
        trk_msg.class_string = trk.obj_class_str

        # Add visual information to message if available
        if trk.image_available:
            trk_msg.image_available = True
            trk_msg.image = trk.image

        tracker.trks_msg.tracks.append(trk_msg)

    tracker.trks_msg.metadata.append(KeyValue(key='time_tracks_published', value=str(tracker.get_clock().now().nanoseconds)))
    tracker.trks_msg.metadata.append(KeyValue(key='num_tracks_published', value=str(len(tracker.trks))))
    
    # Publish populated message
    exec('tracker.%s.publish(tracker.trks_msg)' % pub_name)

def publish_scene(tracker, pub_name):
    # Create scene message
    tracker.scene_msg = SceneUpdate()

    for trk in tracker.trks:

        # Check if tracklet meets track creation criteria
        if is_track_active(tracker, trk)==False:
            continue

        # Create track message
        entity_msg = SceneEntity()

        # Add track information to message
        entity_msg.frame_id = tracker.frame_id
        entity_msg.timestamp = tracker.dets_msg.header.stamp
        entity_msg.id = str(trk.trk_id)
        entity_msg.frame_locked = True
        entity_msg.lifetime.nanosec = 500000000

        cube = CubePrimitive()
        cube.pose.position.x = trk.spatial_state.mean()[0]
        cube.pose.position.y = trk.spatial_state.mean()[1]
        cube.pose.position.z = trk.spatial_state.mean()[2]

        cr = 1.
        sr = 0.
        cp = 1.
        sp = 0.
        cy = np.cos(trk.spatial_state.mean()[3])
        sy = np.sin(trk.spatial_state.mean()[3])

        cube.pose.orientation.w = cr*cp*cy + sr*sp*sy
        cube.pose.orientation.x = sr*cp*cy - cr*sp*sy
        cube.pose.orientation.y = cr*sp*cy + sr*cp*sy
        cube.pose.orientation.z = cr*cp*sy - sr*sp*cy
        cube.size.x = trk.spatial_state.mean()[4]
        cube.size.y = trk.spatial_state.mean()[5]
        cube.size.z = trk.spatial_state.mean()[6]
        cube.color.a = 0.1
        entity_msg.cubes.append(cube)

        # Add semantic information to message
        text = TextPrimitive()
        text.billboard = True
        text.font_size = 12.
        text.scale_invariant = True
        text.color.a = 1.0
        text.pose.position.x = trk.spatial_state.mean()[0]
        text.pose.position.y = trk.spatial_state.mean()[1]
        text.pose.position.z = trk.spatial_state.mean()[2]
        text.text = "%s-%.0f: %.0f %%" % (trk.obj_class_str, trk.trk_id, trk.class_conf*100)
        entity_msg.texts.append(text)

        # Add metadata
        name_md = KeyValuePair()
        name_md.key = 'class_name'
        name_md.value = trk.obj_class_str
        entity_msg.metadata.append(name_md)

        score_md = KeyValuePair()
        score_md.key = 'class_score'
        score_md.value = str(trk.class_conf)
        entity_msg.metadata.append(score_md)

        att_md = KeyValuePair()
        att_md.key = 'attribute'
        att_md.value = '' 
        entity_msg.metadata.append(att_md)

        trk_md = KeyValuePair()
        trk_md.key = 'track_score'
        trk_md.value = str(trk.track_conf)
        entity_msg.metadata.append(trk_md)

        tracker.scene_msg.entities.append(entity_msg)
    
    # Publish scene message
    exec('tracker.%s.publish(tracker.scene_msg)' % pub_name)

def publish_markers(tracker, pub_name):

    # Create scene message
    tracker.marker_array_msg = MarkerArray()

    for trk in tracker.trks:

        # Check if tracklet meets track creation criteria
        if is_track_active(tracker, trk)==False:
            continue

        # Create track message
        marker_msg = Marker()
        txt_marker_msg = Marker()

        # Add track information to message
        marker_msg.header = tracker.dets_msg.header
        marker_msg.id = trk.trk_id
        marker_msg.action = 0 # add/modify
        marker_msg.frame_locked = True
        marker_msg.lifetime.nanosec = 500000000
        marker_msg.type = 1 # cube

        txt_marker_msg.header = tracker.dets_msg.header
        txt_marker_msg.id = trk.trk_id*100 + 101 # arbitrarily deconflict marker id and track id
        txt_marker_msg.action = 0 # add/modify
        txt_marker_msg.frame_locked = True
        txt_marker_msg.lifetime.nanosec = 500000000
        txt_marker_msg.type = 9 # text, view-facing

        marker_msg.pose.position.x = trk.spatial_state.mean()[0]
        marker_msg.pose.position.y = trk.spatial_state.mean()[1]
        marker_msg.pose.position.z = trk.spatial_state.mean()[2]

        cr = 1.
        sr = 0.
        cp = 1.
        sp = 0.
        cy = np.cos(trk.spatial_state.mean()[3])
        sy = np.sin(trk.spatial_state.mean()[3])
        marker_msg.pose.orientation.w = cr*cp*cy + sr*sp*sy
        marker_msg.pose.orientation.x = sr*cp*cy - cr*sp*sy
        marker_msg.pose.orientation.y = cr*sp*cy + sr*cp*sy
        marker_msg.pose.orientation.z = cr*cp*sy - sr*sp*cy
        marker_msg.scale.x = trk.spatial_state.mean()[4]
        marker_msg.scale.y = trk.spatial_state.mean()[5]
        marker_msg.scale.z = trk.spatial_state.mean()[6]
        marker_msg.color.g = 1.0
        marker_msg.color.a = .25

        # Add semantic information to message
        txt_marker_msg.scale.z = .06
        txt_marker_msg.color.a = 1.0
        txt_marker_msg.pose.position.x = trk.spatial_state.mean()[0]
        txt_marker_msg.pose.position.y = trk.spatial_state.mean()[1]
        txt_marker_msg.pose.position.z = trk.spatial_state.mean()[2]
        txt_marker_msg.text = "%s-%.0f: %.0f %%" % (trk.obj_class_str, trk.trk_id, trk.class_conf*100)

        tracker.marker_array_msg.markers.append(marker_msg)
        tracker.marker_array_msg.markers.append(txt_marker_msg)
    
    # Publish scene message
    exec('tracker.%s.publish(tracker.marker_array_msg)' % pub_name)