# Name: CA-based Forest fire model
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# ---

from capyle.ca import Grid2D, Neighbourhood, randomise2d
import capyle.utils as utils
import numpy as np

NAME = "CA-based Forest fire model"
STATES = (0, 1, 2, 3, 4, 5, 6)
GRID = (50, 50)

# Colors
COLORS = {
    'BURNT': (0/255, 0/255, 0/255),           # "#000000"
    'FIRE': (255/255, 0/255, 0/255),          # "#FF0000"
    'CHAPARRAL': (190/255, 190/255, 61/255),  # "#BEBE3D"
    'DENSE_FOREST': (84/255, 97/255, 48/255), # "#546130"
    'LAKE': (77/255, 170/255, 243/255),       # "#4DAAF3"
    'CANYON': (253/255, 254/255, 84/255),     # "#FDFE54"
    'TOWN': (105/255, 105/255, 105/255)       # "#696969"
}

def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = NAME
    config.dimensions = 2
    config.states = STATES

    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----

    # config.state_colors = [(0,0,0),(1,1,1)]
    # config.grid_dims = (200,200)

    # ----------------------------------------------------------------------
    
    config.state_colors = [color for color in COLORS.values()]
    config.grid_dims = (50, 50)
    config.wrap = False

    # Chaparral
    config.initial_grid = np.zeros((50, 50))
    config.initial_grid.fill(STATES[2])

    # Dense forest
    config.initial_grid[8:11, 0:10] = STATES[3]
    config.initial_grid[8:40, 10:20] = STATES[3]
    config.initial_grid[15:20, 30:42] = STATES[3]

    # Lake
    config.initial_grid[40:42, 15:30] = STATES[4]
    config.initial_grid[27:40, 43:45] = STATES[4]

    # Canyon
    config.initial_grid[23:25, 26:42] = STATES[5]

    # Town
    config.initial_grid[34:36, 24:26] = STATES[6]
    
    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config


def transition_function(grid, neighbourstates, neighbourcounts):
    """Function to apply the transition rules
    and return the new grid"""
    # YOUR CODE HERE
    return grid


def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])

    # Create grid object using parameters from config + transition function
    grid = Grid2D(config, transition_function)

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)

if __name__ == "__main__":
    main()
