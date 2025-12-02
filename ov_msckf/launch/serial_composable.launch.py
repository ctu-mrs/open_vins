"""
Launch file for ROS2 OpenVINS serial MSCKF rosbag processor.

This launch file starts the serial MSCKF node that reads a rosbag file sequentially
without any real-time constraints, processing IMU and camera data as fast as possible.

Example usage:
  ros2 launch ov_msckf serial_msckf.launch.py path_bag:=/path/to/bag config:=euroc_mav
  ros2 launch ov_msckf serial_msckf.launch.py path_bag:=/path/to/bag config_path:=/path/to/estimator_config.yaml
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


launch_args = [
    # DeclareLaunchArgument(
    #     name="config",
    #     default_value="euroc_mav",
    #     description="Config preset: euroc_mav, tum_vi, rpng_aruco, etc.",
    # ),
    DeclareLaunchArgument(
        name="config_path",
        default_value="/home/hermiodth/git/open_vins_ws/src/mrs_open_vins_core/ros_packages/mrs_open_vins_core/tmux/handheld_lowlight/config/estimator_config.yaml",
        description="Path to estimator_config.yaml. If not set, determined from 'config' parameter.",
    ),
    DeclareLaunchArgument(
        name="verbosity",
        default_value="INFO",
        description="Log level: ALL, DEBUG, INFO, WARNING, ERROR, SILENT",
    ),
    DeclareLaunchArgument(
        name="path_bag",
        #default_value="/home/hermiodth/git/open_vins_ws/src/mrs_open_vins_core/ros_packages/mrs_open_vins_core/tmux/handheld_lowlight/rosbag2_2025_11_28-14_08_10/bag/bag_0.mcap",
        default_value="/home/hermiodth/git/open_vins_ws/src/mrs_open_vins_core/ros_packages/mrs_open_vins_core/tmux/handheld_lowlight/rosbag2_2025_12_02-21_02_13/rosbag2_2025_12_02-21_02_13_0.mcap",
        description="Path to the rosbag2 directory (without extension)",
    ),
    DeclareLaunchArgument(
        name="bag_start",
        default_value="0.0",
        description="Start time offset in seconds",
    ),
    # DeclareLaunchArgument(
    #     name="bag_durr",
    #     default_value="-1.0",
    #     description="Duration to process in seconds. Use -1 to process entire bag.",
    # ),
    DeclareLaunchArgument(
        name="topic_imu",
        default_value="/imu0",
        description="IMU topic name in rosbag",
    ),
    DeclareLaunchArgument(
        name="topic_camera0",
        default_value="/cam0/image_raw",
        description="Camera 0 topic name in rosbag",
    ),
    DeclareLaunchArgument(
        name="topic_camera1",
        default_value="/cam1/image_raw",
        description="Camera 1 topic name in rosbag (for stereo)",
    ),
    DeclareLaunchArgument(
        name="path_gt",
        default_value="",
        description="Path to groundtruth file (ASL CSV format). Leave empty to skip.",
    ),
    DeclareLaunchArgument(
        name="rviz_enable",
        default_value="true",
        description="Enable RViz visualization",
    ),
    DeclareLaunchArgument(
        name="rviz_config",
        default_value="display_ros2.rviz",
        description="RViz config file name",
    ),
]


def launch_setup(context):
    """Setup launch configuration with parameter resolution."""
    
    # Resolve config path
    config_path = LaunchConfiguration("config_path").perform(context)
    # if config_path == '':
    #     config = LaunchConfiguration("config").perform(context)
    #     config_path = os.path.join(
    #         get_package_share_directory("ov_msckf"),
    #         "config",
    #         config,
    #         "estimator_config.yaml"
    #     )
    
    # Validate config file exists
    # if not os.path.isfile(config_path):
    #     return [
    #         LogInfo(
    #             msg=f"[ERROR] Config file not found: {config_path}. "
    #                 "Please check 'config' or 'config_path' parameter."
    #         )
    #     ]
    
    print(f"[SERIAL_MSCKF] Using config: {config_path}")
    
    # Get RViz config path
    rviz_config_name = LaunchConfiguration("rviz_config").perform(context)
    rviz_config_path = os.path.join(
        get_package_share_directory("ov_msckf"),
        "launch",
        rviz_config_name
    )
    
    # Build parameters dictionary
    params = {
        "verbosity": LaunchConfiguration("verbosity"),
        "config_path": config_path,
        "path_bag": LaunchConfiguration("path_bag"),
        "bag_start": LaunchConfiguration("bag_start"),
        #"bag_durr": LaunchConfiguration("bag_durr"),
        "topic_imu": LaunchConfiguration("topic_imu"),
        "topic_camera0": LaunchConfiguration("topic_camera0"),
        "topic_camera1": LaunchConfiguration("topic_camera1"),
    }
    
    # Add optional groundtruth path if provided
    path_gt = LaunchConfiguration("path_gt").perform(context)
    if path_gt:
        params["path_gt"] = path_gt
    
    # Serial MSCKF node
    serial_msckf_node = Node(
        package="ov_msckf",
        executable="serial_msckf",
        name="serial_msckf",
        output="screen",
        parameters=[params],
        arguments=[
            "--ros-args",
            "--log-level", LaunchConfiguration("verbosity"),
        ],
        #prefix=['xterm -e gdb -ex run --args'],
    )
    
    # RViz node
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        condition=IfCondition(LaunchConfiguration("rviz_enable")),
        arguments=[
            "-d", rviz_config_path,
            "--ros-args",
            "--log-level", "warn",
        ]
    )
    
    return [serial_msckf_node, rviz_node]


def generate_launch_description():
    """Generate the launch description."""
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld