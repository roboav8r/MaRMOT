import os

from ament_index_python import get_package_share_directory

from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch import LaunchDescription
from launch_ros.actions import Node, LoadComposableNodes, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    ld = LaunchDescription()

    # Config files
    tracker_config = os.path.join(
        get_package_share_directory('ahg_smart_space'),
        'config',
        'smart_space_tracker_params.yaml'
    )

    # Detector preprocessing node
    left_preproc_node = Node(
        package='marmot',
        executable='depthai_preproc',
        name='left_depthai_preproc_node',
        remappings=[('/depthai_detections','/left_oak/left_oak/nn/spatial_detections')],
        output='screen')
    ld.add_action(left_preproc_node)

    right_preproc_node = Node(
        package='marmot',
        executable='depthai_preproc',
        name='right_depthai_preproc_node',
        remappings=[('/depthai_detections','/right_oak/right_oak/nn/spatial_detections')],
        output='screen')
    ld.add_action(right_preproc_node)

    # Tracker node
    trk_node = Node(
        package='marmot',
        executable='tbd_node.py',
        name='tbd_tracker_node',
        output='screen',
        remappings=[('/detections','/converted_detections')],
        parameters=[tracker_config]
    )
    ld.add_action(trk_node)

    # Foxglove bridge for visualization
    viz_node = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('marmot'),
                'launch/foxglove_bridge_launch.xml'))
    )
    ld.add_action(viz_node)

    return ld