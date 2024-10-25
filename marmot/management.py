#!/usr/bin/env python3

from marmot.datatypes import Track

def keep_track(trk, trkr):

    keep_this_track = True

    for detector in trkr.detectors.keys():

        if trk.obj_class_str not in trkr.detectors[detector]['detection_classes']:
            continue
        
        if (trkr.detectors[detector]['detection_params'][trk.obj_class_str]['delete_method']=='conf' and trk.track_conf <= trkr.detectors[detector]['detection_params'][trk.obj_class_str]['delete_thresh']):
            keep_this_track = False

    return keep_this_track

def create_tracks(trkr, det_name):

    while trkr.dets:

        det = trkr.dets.pop()
        
        # if (len(trkr.dets)) in trkr.det_asgn_idx: # If detection at end of list is matched, move on
        if (len(trkr.dets)) in trkr.matches[:,0]: # If detection at end of list is matched, move on
            continue

        else:

            # Ignore tracks as appropriate
            if trkr.detectors[det_name]['detection_params'][det.det_class_str] in trkr.detectors[det_name]['detection_classes_ignore']:
                continue
            
            # If using confidence based method, do not create tracks with confidence below threshold
            elif trkr.detectors[det_name]['detection_params'][det.det_class_str]['create_method']=='conf' and det.class_conf < trkr.detectors[det_name]['detection_params'][det.det_class_str]['detect_thresh']:
                continue
            
            else:
                # Create new tracklets from all unmatched detections
                trkr.trks.append(Track(trkr, det))
                trkr.trk_id_count += 1

def delete_tracks(trkr):
    trkr.trks = [track for track in trkr.trks if keep_track(track, trkr)]