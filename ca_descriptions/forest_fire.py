# file: forest_fire.py

# Name: CA-based Forest fire model
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index("ca_descriptions")]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + "capyle")
sys.path.append(main_dir_loc + "capyle/ca")
sys.path.append(main_dir_loc + "capyle/guicomponents")
# ---

from capyle.ca import Grid2D, Neighbourhood, randomise2d
import capyle.utils as utils
import numpy as np

NAME = "CA-based forest fire model"
GRID_SHAPE = (50, 50)

COLORS = {
    "burnt": (0/255, 0/255, 0/255),           # "#000000"
    "fire": (255/255, 0/255, 0/255),          # "#FF0000"
    "chaparral": (190/255, 190/255, 61/255),  # "#BEBE3D"
    "dense_forest": (84/255, 97/255, 48/255), # "#546130"
    "lake": (77/255, 170/255, 243/255),       # "#4DAAF3"
    "canyon": (253/255, 254/255, 84/255),     # "#FDFE54"
    "town": (105/255, 105/255, 105/255)       # "#696969"
}

STATES = {
    "burnt": 0,
    "fire": 1,
    "chaparral": 2,
    "dense_forest": 3,
    "lake": 4,
    "canyon": 5,
    "town": 6
}

INIT_FIRE = {
    "powe_plant": True,
    "proposed_incinerator": False
}

def calculate_spread_rate(temp=20, wind_speed=5, humidity=50):
    """
    Calculate initial fire spread rate based on equation (1) from:

    Jiang, W., Wang, F., Fang, L., Zheng, X., Qiao, X., Li, Z. and Meng, Q., 2021. 
    Modelling of wildland-urban interface fire spread with the heterogeneous cellular automata model. 
    Environmental Modelling & Software, 135, p.104895.
    """
    R0 = 0.03 * temp + 0.05 * wind_speed + 0.01 * (100 - humidity) - 0.3
    
    Kw = 1.0  # Wind coefficient
    Ks = 1.0  # Fuel coefficient
    Kf = 1.0  # Slope coefficient
    
    return R0 * Kw * Ks * Kf

# def calculate_spread_rate(temp=20, wind_speed=5, humidity=50, wind_direction=0, spread_direction=0):
#     """
#     Calculate fire spread rate considering wind direction, based on equation (1) from Jiang et al.
    
#     Args:
#         temp (float): Temperature in Celsius
#         wind_speed (float): Wind speed in m/s
#         humidity (float): Relative humidity percentage
#         wind_direction (float): Wind direction in degrees (0-360, 0=North)
#         spread_direction (float): Direction of potential fire spread in degrees
        
#     Returns:
#         float: Adjusted spread rate
#     """
#     # Base spread rate from equation (1) in paper
#     R0 = 0.03 * temp + 0.05 * wind_speed + 0.01 * (100 - humidity) - 0.3
    
#     # Calculate angle between wind and spread direction
#     angle_diff = abs(wind_direction - spread_direction)
#     if angle_diff > 180:
#         angle_diff = 360 - angle_diff
    
#     # Wind coefficient (Kw) varies with angle difference
#     # Maximum effect when wind aligns with spread direction
#     # Minimum effect when wind opposes spread direction
#     Kw = np.cos(np.radians(angle_diff))
#     Kw = (Kw + 1) / 2  # Normalize to 0-1 range
    
#     # Apply coefficients (based on paper's equation)
#     Ks = 1.0  # Fuel coefficient 
#     Kf = 1.0  # Slope coefficient
    
#     return R0 * max(0.1, Kw) * Ks * Kf  # Ensure minimum spread rate

SPREAD_RATE = calculate_spread_rate()
TIME_STEP_IN_HOURS = 2


class TerrainProperties:
    def __init__(self, grid):
        self.ignition_probs = {
            "chaparral": 0.6,     # Catches fire easily
            "dense_forest": 0.2,  # Harder to ignite
            "canyon": 0.8         # Very easily ignited
        }
        
        self.burn_duration = {
            "chaparral": 72,      # Several days (72 hours)
            "dense_forest": 720,  # Up to one month (30 days * 24 hours)
            "canyon": 12          # Several hours
        }
        
        # Init burning_time as a 3D array where each cell contains [current_time, max_duration]
        self.burning_time = np.zeros((*GRID_SHAPE, 2))

        self.prob_map = np.zeros(GRID_SHAPE)

        # Set both burning_time and prob_map
        for terrain_type, state_val in STATES.items():
            terrain_mask = (grid == state_val)
            
            if terrain_type in self.burn_duration:
                self.burning_time[terrain_mask, 1] = self.burn_duration[terrain_type]
                self.prob_map[terrain_mask] = self.ignition_probs[terrain_type]


def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = NAME
    config.dimensions = 2

    config.states = tuple(STATES.values())

    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----

    # config.state_colors = [(0,0,0),(1,1,1)]
    # config.grid_dims = (200,200)

    # ----------------------------------------------------------------------
    
    config.state_colors = [color for color in COLORS.values()]
    config.grid_dims = GRID_SHAPE
    config.wrap = False

    # Chaparral
    config.initial_grid = np.zeros(GRID_SHAPE)
    config.initial_grid.fill(STATES["chaparral"])

    # Dense forest
    config.initial_grid[8:11, 0:10] = STATES["dense_forest"]
    config.initial_grid[8:40, 10:20] = STATES["dense_forest"]
    config.initial_grid[15:20, 30:42] = STATES["dense_forest"]

    # Lake
    config.initial_grid[40:42, 15:30] = STATES["lake"]
    config.initial_grid[27:40, 43:45] = STATES["lake"]

    # Canyon
    config.initial_grid[23:25, 26:42] = STATES["canyon"]

    # Town
    config.initial_grid[34:36, 24:26] = STATES["town"]
    
    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()

    return config


def initate_fire(grid, terrain_props):
    if INIT_FIRE["powe_plant"]:
        grid[15, 5] = STATES["fire"]
        terrain_props.burning_time[15, 5, 0] = 0
    
    if INIT_FIRE["proposed_incinerator"]:
        grid[0, 49] = STATES["fire"]
        terrain_props.burning_time[0, 49, 0] = 0
        
    return grid


def transition_function(grid, neighbourstates, neighbourcounts, terrain_props):
    """
    Efficient transition function using numpy masks for forest fire simulation
    
    Args:
        grid: Current state of the grid
        neighbourstates: States of neighboring cells
        neighbourcounts: Count of neighbors in each state
        terrain_props: TerrainProperties instance tracking burn times
    """
    # Create masks for different states
    fire_mask = (grid == STATES["fire"])
    burnt_mask = (grid == STATES["burnt"])
    fire_resistant_mask = (grid == STATES["lake"]) | (grid == STATES["town"])

    # Init fire
    if not (fire_mask.any() or burnt_mask.any()):
        return initate_fire(grid, terrain_props)

    _, burning_neighbors, _, _, _, _, _ = neighbourcounts
    
    # Update burning times for cells that are currently on fire
    terrain_props.burning_time[fire_mask, 0] += TIME_STEP_IN_HOURS
    
    # Handle burnt out cells
    burnout_mask = fire_mask & (terrain_props.burning_time[..., 0] >= terrain_props.burning_time[..., 1])
    grid[burnout_mask] = STATES["burnt"]

    terrain_props.burning_time[burnout_mask, 0] = 0
    
    # Handle fire spread to neighboring cells
    # Cells that can catch fire: not burning, not burnt, not fire resistant
    can_ignite = (
        ~fire_mask &
        ~burnt_mask &
        ~fire_resistant_mask &
        (burning_neighbors > 0)
    )

    # Calculate ignition probability based on number of burning neighbors, base probability and spread rate
    ignition_probs = (
        terrain_props.prob_map[can_ignite] * 
        burning_neighbors[can_ignite] / 8 * 
        SPREAD_RATE
    )

    # Create mask for cells that will ignite
    random_vals = np.random.random(np.sum(can_ignite))

    will_ignite = np.zeros_like(can_ignite)
    will_ignite[can_ignite] = random_vals < ignition_probs

    # Set new fires
    grid[will_ignite] = STATES["fire"]
    terrain_props.burning_time[will_ignite, 0] = 0

    return grid


def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])
    
    terrain_props = TerrainProperties(config.initial_grid)
    transition_func = (transition_function, terrain_props)

    # Create grid object using parameters from config + transition function
    grid = Grid2D(config, transition_func)

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()