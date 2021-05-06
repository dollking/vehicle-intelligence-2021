import numpy as np
import itertools

# Given map
grid = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1]
])

# List of possible actions defined in terms of changes in
# the coordinates (y, x)
forward = [
    (-1, 0),  # Up
    (0, -1),  # Left
    (1, 0),  # Down
    (0, 1),  # Right
]

# Three actions are defined:
# - right turn & move forward
# - straight forward
# - left turn & move forward
# Note that each action transforms the orientation along the
# forward array defined above.
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

init = (4, 3, 0)  # Representing (y, x, o), where
# o denotes the orientation as follows:
# 0: up
# 1: left
# 2: down
# 3: right
# Note that this order corresponds to forward above.
goal = (2, 0)
cost = (2, 2, 2)  # Cost for each action (right, straight, left)


# EXAMPLE OUTPUT:
# calling optimum_policy_2D with the given parameters should return
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]

def optimum_policy_2D(grid, init, goal, cost):
    # Initialize the value function with (infeasibly) high costs.
    value = np.full((4,) + grid.shape, 999, dtype=np.int32)
    # Initialize the policy function with negative (unused) values.
    policy = np.full((4,) + grid.shape, -1, dtype=np.int32)
    # Final path policy will be in 2D, instead of 3D.
    policy2D = np.full(grid.shape, ' ')

    # Apply dynamic programming with the flag change.
    change = True
    while change:
        change = False
        # This will provide a useful iterator for the state space.
        p = itertools.product(
            range(grid.shape[0]),
            range(grid.shape[1]),
            range(len(forward))
        )
        # Compute the value function for each state and
        # update policy function accordingly.
        for y, x, o in p:
            # Mark the final state with a special value that we will
            # use in generating the final path policy.
            if (y, x) == goal and value[(o, y, x)] > 0:
                change = True
                value[(o, y, x)] = 0
                policy[(o, y, x)] = 9999

            # Try to use simple arithmetic to capture state transitions.
            elif grid[(y, x)] == 0:
                for i in range(len(action)):
                    prev_o = (o + action[i]) % 4
                    prev_x = x + forward[prev_o][1]
                    prev_y = y + forward[prev_o][0]

                    if (0 <= prev_x < grid.shape[1]) and (0 <= prev_y < grid.shape[0]) and grid[(prev_y, prev_x)] == 0:
                        _cost = value[(prev_o, prev_y, prev_x)] + cost[i]

                        if _cost < value[(o, y, x)]:
                            change = True
                            value[(o, y, x)] = _cost
                            policy[(o, y, x)] = action[i]

    # Now navigate through the policy table to generate a
    # sequence of actions to take to follow the optimal path.
    y, x, o = init
    policy2D[(y, x)] = '#'
    while policy[(o, y, x)] != 9999:
        if policy[(o, y, x)] == action[0]:
            _o = (o - 1) % 4
        elif policy[(o, y, x)] == action[1]:
            _o = o
        else:
            _o = (o + 1) % 4

        x += forward[_o][1]
        y += forward[_o][0]
        o = _o

        policy2D[(y, x)] = '*' if policy[(o, y, x)] == 9999 else action_name[policy[(o, y, x)] + 1]

        # Return the optimum policy generated above.
    return policy2D


print(optimum_policy_2D(grid, init, goal, cost))
