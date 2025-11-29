import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/artem/maga/gz_ws/src/swarm_aco/install/swarm_aco'
