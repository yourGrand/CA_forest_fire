# file: forest_fire.py

# Name: CA-based forest fire model
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

import os
import json
from capyle.ca import Grid2D, Neighbourhood, randomise2d
import capyle.utils as utils
import numpy as np

# Load configuration using absolute path
script_dir = os.path.dirname(os.path.abspath(this_file_loc))
config_path = os.path.join(script_dir, 'forest_fire_config.json')
with open(config_path, 'r') as f:
    CONFIG = json.load(f)

# Extract configuration values
NAME = CONFIG['simulation']['name']
GRID_SHAPE = tuple(CONFIG['simulation']['grid_shape'])
TIME_STEP_IN_HOURS = CONFIG['simulation']['time_step_hours']

# Convert colors from [0-255] to [0-1] range
COLORS = {
    name: tuple(val/255 for val in rgb)
    for name, rgb in CONFIG['colors'].items()
}

STATES = CONFIG['states']
INIT_FIRE = CONFIG['initial_fire']
WEATHER = CONFIG['weather']


class TerrainProperties:
    def __init__(self, grid):
        config = CONFIG['terrain_properties']
        self.ignition_probs = config['ignition_probabilities']
        self.burn_duration = config['burn_duration_hours']
        
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
    config.state_colors = [color for color in COLORS.values()]
    config.grid_dims = GRID_SHAPE
    config.wrap = False

    # Initialise grid with chaparral
    config.initial_grid = np.zeros(GRID_SHAPE)
    config.initial_grid.fill(STATES["chaparral"])

    # Set up terrain based on configuration
    for terrain_type, regions in CONFIG['initial_terrain'].items():
        for region in regions:
            coords = region['coords']
            x_range = slice(coords[0][0], coords[0][1])
            y_range = slice(coords[1][0], coords[1][1])
            config.initial_grid[x_range, y_range] = STATES[terrain_type]
    
    if len(args) == 2:
        config.save()
        sys.exit()

    return config


def initate_fire(grid, terrain_props):
    if INIT_FIRE["power_plant"]:
        grid[14, 4] = STATES["fire"]
        terrain_props.burning_time[14, 4, 0] = 0
    
    if INIT_FIRE["proposed_incinerator"]:
        grid[0, 49] = STATES["fire"]
        terrain_props.burning_time[0, 49, 0] = 0
        
    return grid


def calculate_spread_rate(spread_direction, temp, wind_speed, humidity, wind_direction):
    """
    Vectorised calculation of fire spread rate considering wind direction.
    
    Args:
        spread_direction (ndarray): Array of directions of potential fire spread in degrees
            - 0° = Wind FROM East
            - 90° = Wind FROM North
            - 180° = Wind FROM West
            - 270° = Wind FROM South
        temp (float): Temperature in Celsius
        wind_speed (float): Wind speed in m/s
        humidity (float): Relative humidity percentage
        wind_direction (float): Wind direction in degrees:
            (same convention as spread_direction)
        
    Returns:
        ndarray: Array of adjusted spread rates
    """
    # print(f"\nSpread rate calculation:")
    # print(f"Wind direction: {wind_direction}°")
    # print(f"Sample spread directions: {spread_direction[:3]}")

    # Base spread rate from equation (1) in paper
    R0 = 0.03 * temp + 0.05 * wind_speed + 0.01 * (100 - humidity) - 0.3
    
    # Calculate angle between wind and spread direction
    angle_diff = np.abs(wind_direction - spread_direction)
    angle_diff = np.where(angle_diff > 180, 360 - angle_diff, angle_diff)
    
    # Wind coefficient (Kw) varies with angle difference
    # Maximum effect when wind aligns with spread direction (angle_diff = 0)
    # Minimum effect when wind opposes spread direction (angle_diff = 180)
    Kw = np.cos(np.radians(angle_diff))
    Kw = (Kw + 1) / 2

    # Debug sample wind coefficients
    # print(f"Sample angle differences: {angle_diff[:3]}")
    # print(f"Sample wind coefficients: {Kw[:3]}")
    
    # Slope and Fuel coefficients' placeholder
    Ks = 1.0
    Kf = 1.0
    
    rates = R0 * Kw * Ks * Kf
    # rates = R0 * np.maximum(0.1, Kw) * Ks * Kf     # Maintain non-zero spread-rate (would makes sense from a physics perspective)
    # print(f"Sample spread rates: {rates[:3]}")
    
    return rates


def calculate_spread_direction(fire_mask, can_ignite):
    """
    Calculate spread directions from burning cells to ignitable cells.
    Returns angles in degrees where:
        - 0° = Spreading FROM East
        - 90° = Spreading FROM North
        - 180° = Spreading FROM West
        - 270° = Spreading FROM South
    """
    ignitable_rows, ignitable_cols = np.where(can_ignite)
    burning_rows, burning_cols = np.where(fire_mask)
    
    if len(burning_rows) == 0 or len(ignitable_rows) == 0:
        return np.zeros(np.sum(can_ignite))
    
    # Create coordinate matrices
    ign_rows = ignitable_rows.reshape(-1, 1)
    ign_cols = ignitable_cols.reshape(-1, 1)
    burn_rows = burning_rows.reshape(1, -1)
    burn_cols = burning_cols.reshape(1, -1)
    
    # Calculate coordinate differences
    # Note: Grid coordinates have y-axis pointing down, so we negate dy
    dy = -(burn_rows - ign_rows)
    dx = burn_cols - ign_cols
    
    # Calculate angles for all combinations
    angles = np.degrees(np.arctan2(dy, dx))
    
    # Find nearest burning cell for each ignitable cell
    distances = np.sqrt(dy**2 + dx**2)
    nearest_burning_idx = np.argmin(distances, axis=1)

    # Debug prints
    # print("\nSample coordinate pairs (burning -> ignitable):")
    # for i in range(min(3, len(ignitable_rows))):
    #     b_idx = nearest_burning_idx[i]
    #     print(f"Burning cell ({burning_rows[b_idx]}, {burning_cols[b_idx]}) -> Ignitable cell ({ignitable_rows[i]}, {ignitable_cols[i]})")
    #     print(f"dx={dx[i,b_idx]}, dy={dy[i,b_idx]}")
    #     raw_angle = angles[i,b_idx]
    #     print(f"Raw angle={raw_angle:.1f}°")

    final_angles = angles[np.arange(len(ignitable_rows)), nearest_burning_idx]
    
    # Debug prints
    # print("\nFinal angles (meteorological convention):")
    # print(f"Sample final angles: {final_angles[:3]}")

    return final_angles


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

    _, burning_neighbors, _, _, _, _, _, _, _, _ = neighbourcounts
    
    # Update burning times for cells that are currently on fire
    terrain_props.burning_time[fire_mask, 0] += TIME_STEP_IN_HOURS
    
    # Handle burnt out cells
    burnout_mask = fire_mask & (terrain_props.burning_time[..., 0] >= terrain_props.burning_time[..., 1])
    grid[burnout_mask] = STATES["burnt"]
    terrain_props.burning_time[burnout_mask, 0] = 0
    
    # Handle fire spread
    can_ignite = (
        ~fire_mask &
        ~burnt_mask &
        ~fire_resistant_mask &
        (burning_neighbors > 0)
    )
    
    if np.any(can_ignite):
        spread_directions = calculate_spread_direction(fire_mask, can_ignite)
        
        spread_rate = calculate_spread_rate(
            spread_directions,
            WEATHER['temperature'],
            WEATHER['wind_speed'],
            WEATHER['humidity'],
            WEATHER['wind_direction']
        )
        
        ignition_probs = (
            terrain_props.prob_map[can_ignite] * 
            burning_neighbors[can_ignite] / 8 * 
            spread_rate
        )

        # Create mask for cells that will ignite
        random_vals = np.random.random(len(ignition_probs))
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