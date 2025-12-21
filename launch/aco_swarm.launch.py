from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    pheromone_map = Node(
        package='swarm_aco',
        executable='pheromone_map_node',
        name='pheromone_map_node',
        output='screen',
        parameters=[{
            'width': 100,
            'height': 100,
            'resolution': 1.0,
            'origin_x': -50.0,
            'origin_y': -50.0,
            'evaporation': 0.05,
            'publish_rate': 2.0,
            'max_val': 100.0,
            'deposit_amount': 5.0
        }]
    )

    pheromone_map_neg = Node(
        package='swarm_aco',
        executable='pheromone_map_neg_node',
        name='pheromone_map_neg_node',
        output='screen',
        parameters=[{
            'width': 100, 'height': 100, 'resolution': 1.0,
            'origin_x': -50.0, 'origin_y': -50.0,
            'evaporation_neg': 0.02, 'publish_rate': 2.0,
            'max_val_neg': 100.0, 'deposit_amount_neg': 2.0
        }]
    )

    hotspot = Node(
        package='swarm_aco',
        executable='pheromone_hotspot_node',
        name='pheromone_hotspot',
        output='screen',
        parameters=[{
            'width': 100, 'height': 100, 'resolution': 1.0,
            'origin_x': -50.0, 'origin_y': -50.0,

            # hotspot behavior
            'randomize': True,
            'respawn_secs': 50.0,
            'publish_rate': 2.0,
            'amount': 70.0,
            'spread_cells': 2,
            'falloff_sigma': 1.2,
        }]
    )

    policy_arg = DeclareLaunchArgument(
        'policy',
        default_value='stigmergic_aco',
        description='Decision policy: stigmergic_aco | random_walk | nearest_waypoint | pure_aco_no_negative'
    )

    seed_arg = DeclareLaunchArgument(
        'seed',
        default_value='1',
        description='Random seed for repeatable runs'
    )

    out_csv_arg = DeclareLaunchArgument(
        'out_csv',
        default_value='/tmp/metrics.csv',
        description='Output CSV path for metrics logger'
    )

    metrics = Node(
        package='swarm_aco',
        executable='aco_metrics_logger',
        name='aco_metrics_logger',
        output='screen',
        parameters=[{
            'resolution': 1.0,
            'origin_x': -50.0,
            'origin_y': -50.0,
            'width': 100,
            'height': 100,

            'footprint_m': 1.0,
            'sample_dt': 1.0,
            't_end_sec': 300.0,
            'x_thresholds': [0.5, 0.8, 0.9],

            'pose_topics': [
                '/drone1/local_position/pose',
                '/drone2/local_position/pose',
                '/drone3/local_position/pose',
            ],
            'mode_topic': '/aco_mode',

            # forward experiment identifiers so CSV rows are self-describing
            'policy': LaunchConfiguration('policy'),
            'seed': LaunchConfiguration('seed'),

            # file per run is easiest
            'out_csv': LaunchConfiguration('out_csv'),
        }]
    )

    
    # ========== DRONE 1 ==========
    drone1_deposit = Node(
        package='swarm_aco',
        executable='pheromone_deposit_node',
        name='drone1_deposit',
        output='screen',
        parameters=[{
            'drone_id': 'drone1',
            'deposit_rate': 2.0,
            'base_deposit': 5.0,
            'success_deposit': 20.0
        }],
        remappings=[
            ('/odom', '/drone1/local_position/odom'),
            ('/task_status', '/drone1/task_status'),
        ]
    )
    
    drone1_aco = Node(
        package='swarm_aco',
        executable='aco_decision_node',
        name='drone1_aco',
        output='screen',
        parameters=[{
            'alpha': 2.0,
            'beta': 0.2,
            'hotspot_on_threshold': 50.0,
            'hotspot_off_threshold': 10.0, # Wide hysteresis
            'visit_neg_amount': 100.0, # Strong cooldown
            'mode_cooldown_secs': 10.0, # 10s refractory period
            'dwell_cycles': 3,
            'resolution': 1.0,
            'origin_x': -50.0,
            'origin_y': -50.0,
            'width': 100,
            'height': 100,
            'decision_rate': 0.5,
            'waypoints': [
                -15.0, -15.0,
                -15.0,  15.0,
                 15.0,  15.0,
                 15.0, -15.0,
                  0.0,   0.0,
]
        }],
        remappings=[
            ('/aco_next_waypoint', '/drone1/aco_next_waypoint'),
            ('/local_position/pose', '/drone1/local_position/pose')
        ]
    )
    
    drone1_bridge = Node(
        package='swarm_aco',
        executable='aco_navigation_bridge',
        name='drone1_bridge',
        output='screen',
        parameters=[{
            'drone_id': 'drone1',
            'waypoint_tolerance': 2.0
        }],
        remappings=[
            ('/odom', '/drone1/odometry/out'),
            ('/local_position/pose', '/drone1/local_position/pose'),
            ('/setpoint_position/local', '/drone1/setpoint_position/local'),
            ('/state', '/drone1/state'),
            ('/cmd/arming', '/drone1/cmd/arming'),
            ('/set_mode', '/drone1/set_mode'),
            ('/aco_next_waypoint', '/drone1/aco_next_waypoint'),
        ]
    )
    
    # ========== DRONE 2 ==========
    drone2_deposit = Node(
        package='swarm_aco',
        executable='pheromone_deposit_node',
        name='drone2_deposit',
        output='screen',
        parameters=[{
            'drone_id': 'drone2',
            'deposit_rate': 2.0,
            'base_deposit': 5.0,
            'success_deposit': 20.0
        }],
        remappings=[
            ('/odom', '/drone2/local_position/odom'),
            ('/task_status', '/drone2/task_status'),
        ]
    )
    
    drone2_aco = Node(
        package='swarm_aco',
        executable='aco_decision_node',
        name='drone2_aco',
        output='screen',
        parameters=[{
            'alpha': 2.0,
            'beta': 0.2,
            'hotspot_on_threshold': 50.0,
            'hotspot_off_threshold': 10.0, # Wide hysteresis
            'visit_neg_amount': 100.0, # Strong cooldown
            'mode_cooldown_secs': 10.0, # 10s refractory period
            'dwell_cycles': 3,
            'resolution': 1.0,
            'origin_x': -50.0,
            'origin_y': -50.0,
            'width': 100,
            'height': 100,
            'decision_rate': 0.5,
            'waypoints': [
                -20.0, 0.0,
                0.0, 20.0,
                20.0, 0.0,
                0.0, -20.0,
                0.0, 0.0,
            ]
        }],
        remappings=[
            ('/aco_next_waypoint', '/drone2/aco_next_waypoint'),
            ('/local_position/pose', '/drone2/local_position/pose')
        ]
    )
    
    drone2_bridge = Node(
        package='swarm_aco',
        executable='aco_navigation_bridge',
        name='drone2_bridge',
        output='screen',
        parameters=[{
            'drone_id': 'drone2',
            'waypoint_tolerance': 2.0
        }],
        remappings=[
            ('/odom', '/drone2/odometry/out'),
            ('/local_position/pose', '/drone2/local_position/pose'),
            ('/setpoint_position/local', '/drone2/setpoint_position/local'),
            ('/state', '/drone2/state'),
            ('/cmd/arming', '/drone2/cmd/arming'),
            ('/set_mode', '/drone2/set_mode'),
            ('/aco_next_waypoint', '/drone2/aco_next_waypoint'),
        ]
    )
    
    # ========== DRONE 3 ==========
    drone3_deposit = Node(
        package='swarm_aco',
        executable='pheromone_deposit_node',
        name='drone3_deposit',
        output='screen',
        parameters=[{
            'drone_id': 'drone3',
            'deposit_rate': 2.0,
            'base_deposit': 5.0,
            'success_deposit': 20.0
        }],
        remappings=[
            ('/odom', '/drone3/local_position/odom'),
            ('/task_status', '/drone3/task_status'),
        ]
    )
    
    drone3_aco = Node(
        package='swarm_aco',
        executable='aco_decision_node',
        name='drone3_aco',
        output='screen',
        parameters=[{
            'alpha': 2.0,
            'beta': 0.2,
            'hotspot_on_threshold': 50.0,
            'hotspot_off_threshold': 10.0, # Wide hysteresis
            'visit_neg_amount': 100.0, # Strong cooldown
            'mode_cooldown_secs': 10.0, # 10s refractory period
            'dwell_cycles': 3,
            'resolution': 1.0,
            'origin_x': -50.0,
            'origin_y': -50.0,
            'width': 100,
            'height': 100,
            'decision_rate': 0.5,
            'waypoints': [
                -10.0, -10.0,
                10.0, -10.0,
                10.0, 10.0,
                -10.0, 10.0,
                0.0, 0.0,
            ]
        }],
        remappings=[
            ('/aco_next_waypoint', '/drone3/aco_next_waypoint'),
            ('/local_position/pose', '/drone3/local_position/pose')
        ]
    )
    
    drone3_bridge = Node(
        package='swarm_aco',
        executable='aco_navigation_bridge',
        name='drone3_bridge',
        output='screen',
        parameters=[{
            'drone_id': 'drone3',
            'waypoint_tolerance': 2.0
        }],
        remappings=[
            ('/odom', '/drone3/odometry/out'),
            ('/local_position/pose', '/drone3/local_position/pose'),
            ('/setpoint_position/local', '/drone3/setpoint_position/local'),
            ('/state', '/drone3/state'),
            ('/cmd/arming', '/drone3/cmd/arming'),
            ('/set_mode', '/drone3/set_mode'),
            ('/aco_next_waypoint', '/drone3/aco_next_waypoint'),
        ]
    )
    
    return LaunchDescription([
        pheromone_map,
        pheromone_map_neg,
        hotspot,
        drone1_deposit,
        drone1_aco,
        drone1_bridge,
        drone2_deposit,
        drone2_aco,
        drone2_bridge,
        drone3_deposit,
        drone3_aco,
        drone3_bridge,
    ])