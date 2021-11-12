import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import MultipleLocator

START = []  # initialize a list to store the start point
ICE_HOLE = []
GOAL = []
EXIT = []
WALL = []


def transfer_matrix_to_coordinate(file):
    """Given the txt file, which store the map information
    (0 for ice surface, 1 for ice holes, 2 for start point, 3 for GOAL, 4 for exit, 5 for wall).

    Args:
        file (Any): two options: 'map/map_10x10.txt' and 'map/map_4x4.txt'

    Returns:
        grid_size (int): The gide size of the env.
        MAP_MATRIX (ndarray): The matrix gained from the .txt file.
        START (list): List that stores the start position.
        ICE_HOLE (list): List that stores the ice holes' position.
        GOAL (list): List that stores the GOAL position.
        mask (ndarray): The matrix represent the ice holes with boolean(True for ice holes), used in heatmap plot.
    """
    file = open(file).readlines()  # Read all data files into a list
    grid_size = len(file)
    MAP_MATRIX = np.zeros((grid_size, grid_size), dtype=float)  # First create a matrix(all zeros)
    map_matrix_row = 0  # Represents the rows of the matrix, starting from row 0
    for line in file:  # Read the data in lines row by row

        # Process line-by-line data: strip means to remove the'\n' at the beginning and end,
        # split means to split the line data with spaces, and then return the processed row data to the list list
        map_list = line.strip('\n').split(' ')

        # Put the processed data into square matrix A.
        # list[0:4] means that the 0,1,2,3 columns of the list are placed in the row of matrix A
        MAP_MATRIX[map_matrix_row:] = map_list[0:grid_size]

        map_matrix_row += 1  # Then read the next row of square matrix A

    ice_holes = list(np.argwhere(MAP_MATRIX == 1))  # 1 in txt file represent the ice holes
    start = list(np.argwhere(MAP_MATRIX == 2))  # 2 in txt file represent the start point
    goal = list(np.argwhere(MAP_MATRIX == 3))  # 3 in txt file represent the GOAL
    exits = list(np.argwhere(MAP_MATRIX == 4))
    walls = list(np.argwhere(MAP_MATRIX == 5))
    start_position = (start[0][1], start[0][0])  # gain the start position in tuple
    START.append(start_position)  # append the tuple of start point into the list
    goal_position = (goal[0][1], goal[0][0])
    GOAL.append(goal_position)
    exit_position = (exits[0][1], exits[0][0])
    EXIT.append(exit_position)
    wall_position = (walls[0][1], walls[0][0])
    WALL.append(wall_position)
    # generate the mask used in heatmap, in the form a matrix with boolean(True is the ice hole)
    mask = np.full((grid_size, grid_size), False, dtype=bool)
    for i in range(len(ice_holes)):
        ice_hole = ice_holes[i]
        ice_hole = (ice_hole[1], ice_hole[0])
        ICE_HOLE.append(ice_hole)
        mask[ice_hole[1]][ice_hole[0]] = True
    return grid_size, MAP_MATRIX, START, ICE_HOLE, GOAL, EXIT, WALL


def plot_matrix():
    grid_size = map_matrix.shape[0]
    color_map = colors.ListedColormap(
        ['Azure', '#FF4500', 'Wheat', 'ForestGreen', '#ffff99', 'Gray'])  # represent ice, hole, start, GOAL
    plt.figure(figsize=(grid_size, grid_size), dpi=300)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.pcolor(map_matrix, cmap=color_map, edgecolors='k', linewidths=3)

    # The file name must be changed into 'map_4x4.png' if you use 'map_4x4.txt'
    plt.savefig('map_21x21.png', bbox_inches='tight', pad_inches=0)  # List each element in reverse order output.
    plt.show()


if __name__ == "__main__":
    GRID_SIZE, map_matrix, START, ICE_HOLE, GOAL, EXIT, WALL = transfer_matrix_to_coordinate('map_21x21.txt')
    plot_matrix()
    for i in range(len(ICE_HOLE)):
        ice_hole_coordinates = ICE_HOLE[i]
        x = ice_hole_coordinates[0]
        y = ice_hole_coordinates[1]
    print(ICE_HOLE)
