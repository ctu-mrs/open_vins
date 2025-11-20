from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import ComposableNodeContainer, Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os
from mrs_lib.remappings_custom_config_parser import RemappingsCustomConfigParser

launch_args = [
    DeclareLaunchArgument(
        name='uav_name',
        default_value=os.environ["UAV_NAME"]
    ),
    DeclareLaunchArgument(
        name="rviz_enable", default_value="false", description="enable rviz node"
    ),
    DeclareLaunchArgument(
        name="config",
        default_value="euroc_mav",
        description="euroc_mav, tum_vi, rpng_aruco...",
    ),
    DeclareLaunchArgument(
        name="config_path",
        default_value="",
        description="path to estimator_config.yaml. If not given, determined based on provided 'config' above",
    ),
    DeclareLaunchArgument(
        name="verbosity",
        default_value="INFO",
        description="ALL, DEBUG, INFO, WARNING, ERROR, SILENT",
    ),
    DeclareLaunchArgument(
        name="use_stereo",
        default_value="true",
        description="if we have more than 1 camera, if we should try to track stereo constraints between pairs",
    ),
    DeclareLaunchArgument(
        name="max_cameras",
        default_value="2",
        description="how many cameras we have 1 = mono, 2 = stereo, >2 = binocular (all mono tracking)",
    ),
    DeclareLaunchArgument(
        name="save_total_state",
        default_value="false",
        description="record the total state with calibration and features to a txt file",
    ),
    DeclareLaunchArgument(
        name="use_sim_time",
        default_value="false",
        description="if set to true, simulation time from the '/clock' topic  will be used",
    ),
    DeclareLaunchArgument(
        name="use_intra_process",
        default_value="false",
        description="enable ROS 2 intra-process communication for zero-copy message passing",
    ),
    DeclareLaunchArgument(
        name="container_name",
        default_value="openvins_container",
        description="name of the component container",
    ),
    DeclareLaunchArgument(
        name="standalone",
        default_value="true",
        description="determines whether this node will be running in it's own component container or in the one who's name is provided"
    ),
    DeclareLaunchArgument(
        name="custom_config",
        default_value="",
        description=""
    ),
    DeclareLaunchArgument(
        name="enable_filter",
        default_value="false",
        description="determinmes whether imu filter will be put between imu node and OpenVINS node",
        choices=['true', 'false']
    ),
    DeclareLaunchArgument(
        name="topic_namespace",
        default_value="vio_imu",
        description="entire topic name is constructed as '/<uav_name>/<topic_namespace>/<topic_name>', e.g. /uav1/vio_imu/imu_raw"
    )
]

def launch_setup(context):
    config_path = LaunchConfiguration("config_path").perform(context)
    if config_path == '':
        #configs_dir = os.path.join(get_package_share_directory("ov_msckf"), "config")
        #available_configs = os.listdir(configs_dir)
        config = LaunchConfiguration("config").perform(context)
        # if config in available_configs:
        #     config_path = os.path.join(
        #                     get_package_share_directory("ov_msckf"),
        #                     "config",config,"estimator_config.yaml"
        #                 )
        config_path = os.path.join(
            get_package_share_directory("ov_msckf"),
            "config",config,"estimator_config.yaml"
        )
    
        # else:
        #     return [
        #         LogInfo(
        #             msg="ERROR: unknown config: '{}' - Available configs are: {} - not starting OpenVINS".format(
        #                 config, ", ".join(available_configs)
        #             )
        #         )
        #     ]
    else:
        if not os.path.isfile(config_path):
            return [
                LogInfo(
                    msg="ERROR: config_path file: '{}' - does not exist. - not starting OpenVINS".format(
                        config_path)
                    )
            ]
            
    print("config_path: ", config_path)
    
    # Create the composable node description
    msckf_node = ComposableNode(
        package="ov_msckf",
        plugin="msckf_component::SubscribeMSCKF",  # Update this with the actual component class name
        name="open_vins",
        namespace = LaunchConfiguration('uav_name'),
        parameters=[
            {"verbosity": LaunchConfiguration("verbosity")},
            {"use_stereo": LaunchConfiguration("use_stereo")},
            {"max_cameras": LaunchConfiguration("max_cameras")},
            {"save_total_state": LaunchConfiguration("save_total_state")},
            {"use_sim_time": LaunchConfiguration('use_sim_time')},
            {"frames_prefix": LaunchConfiguration('uav_name')},
            {"global_frame_name": "global"},
            {"imu_frame_name": "imu"},
            {"cam_frame_name": "cam0"},
            {"config_path": config_path},
            {"topic_imu": PythonExpression(["'", PathJoinSubstitution(["/", LaunchConfiguration('uav_name'), LaunchConfiguration('topic_namespace')]), "/imu_filtered' if '", LaunchConfiguration("enable_filter"), "' == 'true' else ''"])}
        ],
        remappings=[
            ("~/poseimu_out", "~/poseimu"),
            ("~/odomimu_out", "~/odomimu"),
            ("~/pathimu_out", "~/pathimu"),
            ("~/points_msckf_out", "~/points_msckf"),
            ("~/points_slam_out", "~/points_slam"),
            ("~/points_aruco_out", "~/points_aruco"),
            ("~/points_sim_out", "~/points_sim"),
            ("~/posegt_out", "~/posegt"),
            ("~/pathgt_out", "~/pathgt"),
            ("~/loop_pose_out", "~/loop_pose"),
            ("~/loop_feats_out", "~/loop_feats"),
            ("~/loop_extrinsic_out", "~/loop_extrinsic"),
            ("~/loop_intrinsics_out", "~/loop_intrinsics")
        ],
        extra_arguments=[{"use_intra_process_comms": LaunchConfiguration("use_intra_process")}]
    )
    
    parser = RemappingsCustomConfigParser(msckf_node, LaunchConfiguration('custom_config'))

    filter = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('mrs_vins_imu_filter'),
            'launch',
            'filter_icm_42688.py'
        ]),
        launch_arguments={
            'standalone': 'True',
            'container_name': LaunchConfiguration("container_name"),
            'topic_namespace': LaunchConfiguration('topic_namespace')
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_filter'))
    )

    loader = LoadComposableNodes(
        condition=UnlessCondition(LaunchConfiguration('standalone')),
        composable_node_descriptions=[msckf_node],
        target_container=LaunchConfiguration('container_id'),
    )
    
    # Create the component container
    container = ComposableNodeContainer(
        name=LaunchConfiguration("container_name"),
        namespace=LaunchConfiguration('uav_name'),
        package="rclcpp_components",
        executable="component_container",
        condition=IfCondition(LaunchConfiguration("standalone")),
        composable_node_descriptions=[msckf_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
        #prefix=['xterm -e gdb -ex run --args']
    )

    # RViz node (remains as regular node)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        condition=IfCondition(LaunchConfiguration("rviz_enable")),
        arguments=[
            "-d"
            + os.path.join(
                get_package_share_directory("ov_msckf"), "launch", "display_ros2.rviz"
            ),
            "--ros-args",
            "--log-level",
            "warn",
        ],
    )

    return [parser, loader, container, rviz_node, filter]


def generate_launch_description():
    opfunc = OpaqueFunction(function=launch_setup)
    ld = LaunchDescription(launch_args)
    ld.add_action(opfunc)
    return ld
