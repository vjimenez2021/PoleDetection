from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='controller',
            executable='pole_detector_main',
            name='pole_detector',
            output='screen',
            parameters=[
                # ----- Filtros de altura -----
                {'min_height': 0.2},
                {'max_height': 4.0},

                # ----- Clustering -----
                {'cluster_tolerance': 5.5},
                {'min_cluster_size': 10},
                {'max_cluster_size': 100},

                # ----- Criterios cilíndricos -----
                {'min_cylindrical_aspect_ratio': 0.0},
                {'max_cylindrical_width': 5.4},
                {'cylinder_distance_threshold': 0.15},
                {'min_cylinder_radius': 0.05},
                {'max_cylinder_radius': 0.5},

                # ----- RANSAC -----
                {'ransac_inlier_ratio': 0.7},           # ratio mínimo de inliers
                {'ransac_min_cluster_points': 15},      # puntos mínimos para RANSAC
                {'ransac_normal_distance_weight': 0.1}, # peso de distancia de normales
                {'ransac_max_iterations': 1500},        # máximo número de iteraciones
            ]
        )
    ])
