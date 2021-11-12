import numpy as np
import math
import gym
import hashlib
from enum import IntEnum
from gym import error, spaces, utils
from gym.utils import seeding
from custom_env.rendering import *
from custom_env.rendering import fill_coords, point_in_rect, point_in_circle
# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

COLORS = {'red': np.array([255, 0, 0]), 'green': np.array([0, 255, 0]), 'blue': np.array([0, 0, 255]),
          'cream': np.array([255, 255, 210]), 'grey': np.array([100, 100, 100]), 'black': np.array([0, 0, 0])
          , 'white': np.array([255, 255, 255]), 'azure': np.array([240, 255, 255])}
COLOR_TO_IDX = {'red': 0, 'green': 1, 'blue': 2, 'cream': 3, 'grey': 4, 'black': 5, 'white': 6, 'azure': 7}
OBJECT_TO_IDX = {'unseen': 0, 'empty': 1, 'wall': 2, 'floor': 3, 'ball': 4, 'box': 5, 'goal': 6, 'agent': 7, 'exit': 8}
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.initial_pos = None

        # Current position of the object
        self.current_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        if obj_type == 'wall':
            v = Wall()
        elif obj_type == 'floor':
            v = Floor()
        elif obj_type == 'ball':
            v = Ball()
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'exit':
            v = Exit()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Exit(WorldObj):
    def __init__(self):
        super().__init__('exit', 'cream')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS['cream'])


class Wall(WorldObj):
    def __init__(self):
        super().__init__('wall', 'grey')

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS['grey'])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self):
        super().__init__('floor', 'cream')

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        # color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), COLORS['cream'])


class Ball(WorldObj):
    def __init__(self):
        super(Ball, self).__init__('ball', 'blue')

    # def can_pickup(self):
    #     return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS['blue'])


class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS['green'])


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        # initialize the matrix to store objects for each grid
        self.grid = [None] * width * height

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        """

        :param i:
        :param j:
        :param v:
        :return: assign the specific attributes for each grid
        """
        assert 0 <= i < self.width
        assert 0 <= j < self.height

        #  索引坐标系的转换，如将（1，0）转化为1，并赋予属性
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        """

        :param i:
        :param j:
        :return: the specific attributes of each grid
        """
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        return self.grid[j * self.width + i]

    def horizontal_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vertical_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horizontal_wall(x, y, w)
        self.horizontal_wall(x, y + h - 1, w)
        self.vertical_wall(x, y, h)
        self.vertical_wall(x + w - 1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if 0 <= x < self.width and 0 <= y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()
                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
            cls,
            obj,
            agent_dir=None,
            highlight=False,
            tile_size=TILE_PIXELS,
            subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Down sample the image to perform super sampling/anti-aliasing
        img = down_sample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
            self,
            tile_size,
            agent_pos=None,
            agent_dir=None,
            highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param agent_pos:
        :param agent_dir:
        :param highlight_mask:
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()
        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=np.bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width - 1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                # wall 不能 see_behind ,return False
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        """
        The actions space of the environment, which in details: 0:left 1:right 2:forward 3:done
        """
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Done completing task, not used by default
        # done = 3

    def __init__(
            self,
            grid_size=None,
            width=None,
            height=None,
            max_steps=1000,
            see_through_walls=False,
            seed=1337,
            agent_view_size=7
    ):
        # Can't set both grid_size and width/height
        self.mission = None
        self.grid = None
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        # 需要保证 view_size 为 奇数且大于等于3
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0
        self.reach_exit_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high]
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high]
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return self.np_random.randint(0, 2) == 0

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)
        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        COLOR_NAMES = sorted(list(COLORS.keys()))
        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, x_Low, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(x_Low, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid
        :param obj:
        :param max_tries:
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)
        num_tries = 0
        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop 这是为了处理拒绝采样陷入无限循环的罕见情况
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1
            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))
            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue
            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue
            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue
            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.current_pos = pos
        return pos

    def place_goal_obj_left_conner(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid
        :param obj:
        :param max_tries:
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)
        num_tries = 0
        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop 这是为了处理拒绝采样陷入无限循环的罕见情况
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1
            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width//2)),
                self._rand_int(top[1]+self.grid.height//2, min(top[1] + size[1], self.grid.height))
            ))
            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue
            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue
            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue
            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.current_pos = pos
        return pos

    def place_goal_obj_right_conner(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid
        :param obj:
        :param max_tries:
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)
        num_tries = 0
        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop 这是为了处理拒绝采样陷入无限循环的罕见情况
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1
            pos = np.array((
                self._rand_int(top[0]+self.grid.height//2, min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1]+self.grid.height//2, min(top[1] + size[1], self.grid.height))
            ))
            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue
            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue
            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue
            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.current_pos = pos
        return pos

    def place_agent_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid
        :param obj:
        :param max_tries:
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)
        num_tries = 0
        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop 这是为了处理拒绝采样陷入无限循环的罕见情况
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1
            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width//2)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height//2))
            ))
            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue
            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue
            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue
            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.current_pos = pos
        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.current_pos = (i, j)
        return obj.current_pos

    def place_agent(
            self,
            top=None,
            size=None,
            rand_dir=True,
            max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None
        pos = self.place_agent_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        0: right 1: down 2:left 3:up
        """
        DIR_TO_VEC = [
            # Pointing right (positive X)
            np.array((1, 0)),

            # Down (positive Y)
            np.array((0, 1)),

            # Pointing left (negative X)
            np.array((-1, 0)),

            # Up (negative Y)
            np.array((0, -1)),
        ]
        assert 0 <= self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent. 顺时针旋转90°
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        :return: the coordinates of the cells that right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        # 当前坐标系原点为矩形视野的右上角顶点，得到相对此原点的坐标
        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exits(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        获取代理可见的方形瓷砖集的范围
        注意：底部范围索引不包括在集合中
        // 整除(向小取整) 相当于floor函数
        """

        # Facing right
        if self.agent_dir == 0:
            topX = self.agent_pos[0]
            topY = self.agent_pos[1] - self.agent_view_size // 2
        # Facing down
        elif self.agent_dir == 1:
            topX = self.agent_pos[0] - self.agent_view_size // 2
            topY = self.agent_pos[1]
        # Facing left
        elif self.agent_dir == 2:
            topX = self.agent_pos[0] - self.agent_view_size + 1
            topY = self.agent_pos[1] - self.agent_view_size // 2
        # Facing up
        elif self.agent_dir == 3:
            topX = self.agent_pos[0] - self.agent_view_size // 2
            topY = self.agent_pos[1] - self.agent_view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.agent_view_size
        botY = topY + self.agent_view_size

        return topX, topY, botX, botY

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view(FOV), and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)
        # if vx >= 0 and vy >= 0
        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False

        # Get the position in front of the agent
        forward_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*forward_pos)

        # DIRECTIONS: right: 0(1,0) down:1(0,1) left:2(-1,0) up:3(0,-1)
        # ACTIONS: 0:left 1:right 2:forward

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
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = forward_pos
            # if fwd_cell is not None and fwd_cell.type == 'goal':
            #     done = True
            #     # reward = self._reward()
            #     reward = 100

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return self.agent_pos, obs, reward, done, {}

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exits()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            from custom_env import window
            self.window = window.Window('custom_env')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size - 1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def close(self):
        if self.window:
            self.window.close()
        return
