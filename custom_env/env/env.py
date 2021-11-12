# -*- coding: utf-8 -*-
from gym import spaces

from custom_env import *
from operator import add

from custom_env.mini_grid import MiniGridEnv, Grid, Goal, Ball, Exit


class FourRoomsDynamicObstaclesEnv21x21(MiniGridEnv):
    """
    4 rooms gridworld environment with dynamic obstacles.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, size=21, n_obstacles=16, agent_pos=None, goal_pos=None):

        if n_obstacles <= size:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size)
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=size, max_steps=450, see_through_walls=True)

        # Allow only 3 actions permitted: left, right, forward 0,1,2
        # gym.spaces.Discrete->{0,1,..,n-1}
        self.action_space = spaces.Discrete(self.actions.forward + 1)
        # self.reward_range = (-1, 1)

    def _gen_wall(self, width, height):
        """
        create the walls of along the boundary and each room, the exit place for each room.
        :param width: the width of the grid env
        :param height: the height of the grid env
        :return: the grid world with wall setting up
        """
        self.grid.horizontal_wall(0, 0)
        self.grid.horizontal_wall(0, height - 1)
        self.grid.vertical_wall(0, 0)
        self.grid.vertical_wall(width - 1, 0)

        # set a fixed goal in the environment
        # self.grid.set(width - 2, height - 2, Goal())
        room_w = width // 2  # width for each room
        room_h = height // 2  # height of each room
        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                x_left = i * room_w
                y_top = j * room_h
                x_right = x_left + room_w
                y_bottom = y_top + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vertical_wall(x_right, y_top, room_h)
                    # random initialize the exit place along the wall of each room
                    pos = (x_right, self._rand_int(y_top + 2, y_bottom - 1))
                    # set None means that this grid has no object, which represents the exit
                    self.grid.set(*pos, Exit())

                # Construction of horizontal walls bottom wall and exit(represented by None)
                if j + 1 < 2:
                    self.grid.horizontal_wall(x_left, y_bottom, room_w)
                    # self.grid.horizontal_wall(0, 20, 20)
                    # pos = (self._rand_int(x_left+5, x_right), y_bottom)
                    # self.grid.set(*pos, Exit())
        # for i in range(0, 2):
        #     x_left = room_w
        #     y_top = room_h
        #     # print('y_top', y_top)
        #     x_right = x_left + room_w
        #     y_bottom = y_top + room_h
        #     # print('y_bottom', y_bottom)
        #     # Bottom wall and door
        #     if i + 1 < 2:
        #         # random initialize the exit place along the wall of each room
        #         pos = (x_left, self._rand_int(1, y_top))
        #         # set None means that this grid has no object, which represents the exit
        #         self.grid.set(*pos, Exit())
        for j in range(0, 2):
            x_left = room_w
            y_top = room_h
            x_right = x_left + room_w
            # Construction of horizontal walls bottom wall and exit(represented by None)
            if j + 1 < 2:
                pos1 = (self._rand_int(room_w + 2, x_right), y_top)
                self.grid.set(*pos1, Exit())

        return self.grid

    def _gen_grid(self, width, height):
        global goal_cur_pos_x, goal_cur_pos_y, goal_cur_pos
        goal_cur_pos_x = 0
        goal_cur_pos_y = 0
        goal_cur_pos = 0
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self._gen_wall(width, height)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)

            # assuming random start direction of the agent(0,1,2,3)
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.current_pos = self._goal_default_pos
        else:
            # self.place_obj(Goal())
            goal_current_pos = self.place_goal_obj_left_conner(Goal())
            # print('goal_current_pos', goal_current_pos)
            goal_cur_pos_x = goal_current_pos[0]
            goal_cur_pos_y = goal_current_pos[1]

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = 'Reach the goal'

    def step(self, action):
        self.step_count += 1
        reward = 0

        done = False
        reward_max = 100
        reward_min = -150
        # Get the position in front of the agent
        forward_pos = self.front_pos  # self.front_pos = self.agent_pos + self.dir_vec
        # Get the contents of the cell in front of the agent
        forward_cell = self.grid.get(*forward_pos)

        if action >= self.action_space.n:
            action = 0

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            #  如果forward_cell object 为空或object可以重叠（一个cell包含多个objects)
            if forward_cell is None or forward_cell.can_overlap():
                self.agent_pos = forward_pos
        else:
            assert False, "unknown action"
        # Check if there is an obstacle in front of the agent
        # forward_cell = self.grid.get(*self.front_pos)
        not_clear = forward_cell and (forward_cell.type != 'goal' and forward_cell.type != 'exit')
        clear = forward_cell and forward_cell.type == 'empty'
        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].current_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.place_obj(self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100)
                self.grid.set(*old_pos, None)
            except:
                pass

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # Update the agent's position/direction
        # self.agent_pos, obs, reward, done, info = MiniGridEnv.step(self, action)

        if forward_cell is not None and forward_cell.type == 'exit':
            self.reach_exit_count += 1
            # print('self.reach_exit_count', self.reach_exit_count)
            if 1 <= self.reach_exit_count <= 3:
                reward = 100
            else:
                reward = -10
        elif action == self.actions.forward and clear:
            reward = 5
        elif forward_cell is not None and forward_cell.type == 'goal':
            done = True
            reward = 400
        elif action == self.actions.forward and not_clear:
            done = True
            reward = -300
        else:
            agent_cur_pos_x = self.agent_pos[0]
            agent_cur_pos_y = self.agent_pos[1]
            reward = -0.5 * (abs(goal_cur_pos_x - agent_cur_pos_x) + abs(goal_cur_pos_y - agent_cur_pos_y))
        # If the agent tried to walk over an obstacle or wall
        # return reward
        if not done:
            reward += -1
        # if reward > reward_max:
        #     reward = reward_max
        # if reward < reward_min:
        #     reward = reward_min
        # return obs, reward, done, info
        # print(reward)
        return obs, reward, done, {}


class ThreeRoomsDynamicObstaclesEnv21x21(MiniGridEnv):
    """
    3 rooms gridworld environment with dynamic obstacles and exits.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, size=21, n_obstacles=16, agent_pos=None, goal_pos=None):

        if n_obstacles <= size:
            self.n_obstacles = int(n_obstacles)
        else:
            self.n_obstacles = int(size)
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=size, max_steps=450, see_through_walls=True)

        # Allow only 3 actions permitted: left, right, forward 0,1,2
        # gym.spaces.Discrete->{0,1,..,n-1}
        self.action_space = spaces.Discrete(self.actions.forward + 1)
        # self.reward_range = (-1, 1)

    def _gen_wall(self, width, height):
        """
        create the walls of along the boundary and each room, the exit place for each room.
        :param width: the width of the grid env
        :param height: the height of the grid env
        :return: the grid world with wall setting up
        """
        self.grid.horizontal_wall(0, 0)
        self.grid.horizontal_wall(0, height - 1)
        self.grid.vertical_wall(0, 0)
        self.grid.vertical_wall(width - 1, 0)

        # set a fixed goal in the environment
        # self.grid.set(width - 2, height - 2, Goal())
        room_w = width // 2  # width for each room
        room_h = height // 2  # height of each room
        # For each row of rooms
        for j in range(0, 2):
            # For each column
            for i in range(0, 2):
                x_left = i * room_w
                y_top = j * room_h
                x_right = x_left + room_w
                y_bottom = y_top + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vertical_wall(x_right, y_top, room_h)

                # Construction of horizontal walls bottom wall and exit(represented by None)
                if j + 1 < 2:
                    self.grid.horizontal_wall(room_w, y_bottom, room_w)

        for i in range(0, 2):
            x_left = room_w
            y_top = room_h
            # Bottom wall and door
            if i + 1 < 2:
                # random initialize the exit place along the wall of each room
                pos = (x_left, self._rand_int(1, y_top))
                # set None means that this grid has no object, which represents the exit
                self.grid.set(*pos, Exit())
        for j in range(0, 2):
            x_left = room_w
            y_top = room_h
            x_right = x_left + room_w
            # Construction of horizontal walls bottom wall and exit(represented by None)
            if j + 1 < 2:
                pos1 = (self._rand_int(room_w + 2, x_right), y_top)
                self.grid.set(*pos1, Exit())

        return self.grid

    def _gen_grid(self, width, height):
        global goal_cur_pos_x, goal_cur_pos_y, goal_cur_pos
        goal_cur_pos_x = 0
        goal_cur_pos_y = 0
        goal_cur_pos = 0
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self._gen_wall(width, height)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)

            # assuming random start direction of the agent(0,1,2,3)
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.current_pos = self._goal_default_pos
        else:
            # self.place_obj(Goal())
            goal_current_pos = self.place_goal_obj_right_conner(Goal())
            # print('goal_current_pos', goal_current_pos)
            goal_cur_pos_x = goal_current_pos[0]
            goal_cur_pos_y = goal_current_pos[1]

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = 'Reach the goal'

    def step(self, action):
        self.step_count += 1
        reward = 0

        done = False
        reward_max = 100
        reward_min = -150
        # Get the position in front of the agent
        forward_pos = self.front_pos  # self.front_pos = self.agent_pos + self.dir_vec
        # Get the contents of the cell in front of the agent
        forward_cell = self.grid.get(*forward_pos)

        if action >= self.action_space.n:
            action = 0

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            #  如果forward_cell object 为空或object可以重叠（一个cell包含多个objects)
            if forward_cell is None or forward_cell.can_overlap():
                self.agent_pos = forward_pos
        else:
            assert False, "unknown action"
        # Check if there is an obstacle in front of the agent
        # forward_cell = self.grid.get(*self.front_pos)
        not_clear = forward_cell and (forward_cell.type != 'goal' and forward_cell.type != 'exit')
        clear = forward_cell and forward_cell.type == 'empty'
        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].current_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                self.place_obj(self.obstacles[i_obst], top=top, size=(3, 3), max_tries=100)
                self.grid.set(*old_pos, None)
            except:
                pass

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # Update the agent's position/direction
        # self.agent_pos, obs, reward, done, info = MiniGridEnv.step(self, action)

        if forward_cell is not None and forward_cell.type == 'exit':
            self.reach_exit_count += 1
            # print('self.reach_exit_count', self.reach_exit_count)
            if 1 <= self.reach_exit_count <= 2:
                reward = 100
            else:
                reward = -10
        elif action == self.actions.forward and clear:
            reward = 5
        elif forward_cell is not None and forward_cell.type == 'goal':
            done = True
            reward = 400
        elif action == self.actions.forward and not_clear:
            done = True
            reward = -300
        else:
            agent_cur_pos_x = self.agent_pos[0]
            agent_cur_pos_y = self.agent_pos[1]
            reward = -0.5 * (abs(goal_cur_pos_x - agent_cur_pos_x) + abs(goal_cur_pos_y - agent_cur_pos_y))
        # If the agent tried to walk over an obstacle or wall
        # return reward
        if not done:
            reward += -1
        # if reward > reward_max:
        #     reward = reward_max
        # if reward < reward_min:
        #     reward = reward_min
        # return obs, reward, done, info
        # print(reward)
        return obs, reward, done, {}
