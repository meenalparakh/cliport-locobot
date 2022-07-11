"""
    Action space:
        - 0: Turn left 15deg
        - 1: Turn right 15 deg
        - 2: Move forward 20cm
        - 3: Move backward 20cm
        - 4: Grasp

"""
from math import pi

NUM_DISCRETE_ACTIONS = 11
# Actions discrete environment
TURN_LEFT_LONG_ACTION = 0
TURN_LEFT_ACTION = 1
TURN_RIGHT_LONG_ACTION = 2
TURN_RIGHT_ACTION = 3

MOVE_FORWARD_ACTION = 4
MOVE_FORWARD_SHORT_ACTION = 5

MOVE_BACKWARD_ACTION = 6
MOVE_BACKWARD_SHORT_ACTION = 7

GRASP_ACTION = 8
DROP_ACTION = 9
TERMINATE_EPISODE = 10

MULTIGOAL_DESCRIPTION = """
- Pick blue (0)
- Drop blue (1)
- Pick black (2)
- Drop black (3)
- pick orange (4)
- drop orange (5)
- pick green (6)
- drop green (7)
"""

# Discrete environment constants
ARM_RESET_POS = [1.96, 0.52, -0.51, 1.67, 0.01]
ARM_OBJ_VIS_REST_POS = [0.75, 0.52, 0.0, -1, 0.01]
MAX_TURNS_ALLOWED = 15
WALL_HEIGHT = 0.12
FLOOR_HEIGHT = 0.01
DELTA_TIME = 1 # seconds
ROTATION_SPEED = pi/6
FORWARD_SPEED = 0.2
NAVIGATION_CAM_TILT = 0.6
GRASPING_CAM_TILT = 0.8
WALL_CLOSE_DISTANCE = 0.45

actions_dict = {
    0: 'TURN_LEFT_ACTION',
    1: 'TURN_RIGHT_ACTION',
    2: 'MOVE_FORWARD_ACTION' ,
    3: 'MOVE_BACKWARD_ACTION' ,
    4: 'GRASP_ACTION',
    5: 'MOVE_FORWARD_SHORT_ACTION',
    6: 'MOVE_BACKWARD_SHORT_ACTION' ,
    7: 'TURN_LEFT_LONG_ACTION',
    8: 'TURN_RIGHT_LONG_ACTION' ,
    9: 'DROP_ACTION',
    10: 'TERMINATE_EPISODE'
}