#!/usr/bin/env python3
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'swarm_aco'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='artem',
    maintainer_email='artem@todo.todo',
    description='Swarm ACO pheromone-based navigation package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pheromone_map_node = swarm_aco.pheromone_map_node:main',
            'pheromone_deposit_node = swarm_aco.pheromone_deposit_node:main',
            'aco_decision_node = swarm_aco.aco_decision_node:main',
            'aco_navigation_bridge = swarm_aco.aco_navigation_bridge:main',
            'aco_visualizer_node = swarm_aco.aco_visualizer_node:main',
        ],
    },
)