# Name: Bioinspired
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

def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "BioinspiredV2"
    config.dimensions = 2
    config.states = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----

    config.state_colors = [(0,0,1), #lake
                           (0.5,0.5,0.5), #town
                           (1,0,0), #burning_forest
                           (0,0.5,0), #forest
                           (1,0.5,0), #burning_chap
                           (0.5,1,0.5), #chap
                           (1,1,0), #burning_canyon
                           (0.25,0.5,0), #canyon
                           (0,0,0), #burnt_out

                           (0,0.5,0), #forest_ex 9 
                           (0,0.5,0), #forest_ex_prot 10

                           (0.5,1,0.5), #chap_ex 11
                           (0.5,1,0.5), #chap_ex_prot 12

                           (0.25,0.5,0), #canyon_ex 13
                           (0.25,0.5,0)] #canyon_ex_prot 14
    
    # config.grid_dims = (200,200)

    # Unpack neighbor counts based on terrain and states

    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config

def transition_function(grid, neighbourstates, neighbourcounts):
    """
    Transition function to simulate terrain types and burning states for a forest fire model.
    """

    # Unpack neighbor counts based on terrain and states
    (lake_neighbours, town_neighbours, burning_forest_neighbours,
     forest_neighbours, burn_chap_neighbours, chap_neighbours,
     canyon_burn_neighbours, canyon_neighbours, burnt_neighbours, 
     forest_ex, forest_ex_protect,
     chap_ex, chap_ex_protect,
     canyon_ex, canyon_ex_protect) = neighbourcounts

    # Copy the grid for modifications
    new_grid = grid.copy()
    
    # Get grid dimensions
    rows, cols = grid.shape

    # Define transition probabilities
    prob_canyon_burn = 0.25
    prob_chaparral_burn = 0.05
    prob_forest_burn = 0.005
    prob_town_burn = 0.20
    burn_out_chance_forest = 1 / 360
    burn_out_chance_chaparral = 1 / 60
    burn_out_chance_canyon = 0.2
    burn_protection_chance = 0.1

    # Iterate through each cell in the grid
    for i in range(rows):
        for j in range(cols):
            current_state = grid[i, j]
            
            # Ensure we’re referencing a single neighbor cell state
            north_burning = False
            south_burning = False
            if i > 0:  # Check if there's a cell to the north
                north_burning = grid[i-1, j] in {2, 4, 6}
            if i < rows - 1:  # Check if there's a cell to the south
                south_burning = grid[i+1, j] in {2, 4, 6}
            
            # Adjust burning neighbor counts based on fire direction probabilities
            burning_neighbors = (burning_forest_neighbours[i, j] +
                                 burn_chap_neighbours[i, j] +
                                 canyon_burn_neighbours[i, j])

            if north_burning:
                burning_neighbors += np.random.choice([1, 2], p=[0.5, 0.5])
            if south_burning:
                burning_neighbors += np.random.choice([0, 1], p=[0.5, 0.5])

            # Apply transition rules based on current state and adjusted burning_neighbors

            # Canyon + burning neighbors -> chance to turn Canyon Burning
            if current_state == 7 and burning_neighbors > 0:
                if np.random.rand() < burning_neighbors * prob_canyon_burn:
                    new_grid[i, j] = 6

            # Chaparral + burning neighbors -> chance to turn Chaparral Burning
            elif current_state == 5 and burning_neighbors > 0:
                if np.random.rand() < burning_neighbors * prob_chaparral_burn:
                    new_grid[i, j] = 4

            # Dense Forest + burning neighbors -> chance to turn Dense Forest Burning
            elif current_state == 3 and burning_neighbors > 0:
                if np.random.rand() < burning_neighbors * prob_forest_burn:
                    new_grid[i, j] = 2

            # Town + burning neighbors -> chance to turn Burnt Out
            elif current_state == 1 and burning_neighbors > 0:
                if np.random.rand() < burning_neighbors * prob_town_burn:
                    new_grid[i, j] = 8

            # Dense Forest Burning -> chance to turn Burnt Out
            elif current_state == 2:
                if np.random.rand() < burn_out_chance_forest:
                    new_grid[i, j] = 8

            # Chaparral Burning -> chance to turn Burnt Out
            elif current_state == 4:
                if np.random.rand() < burn_out_chance_chaparral:
                    new_grid[i, j] = 8

            # Canyon Burning -> chance to turn Burnt Out
            elif current_state == 6:
                if np.random.rand() < burn_out_chance_canyon:
                    new_grid[i, j] = 8

            # protected forest tile + burning neightbours -> chance to turn to inactive forest
            elif current_state == 9 and burning_neighbors > 0:
                if np.random.rand() < burning_neighbors * prob_forest_burn:
                    new_grid[i, j] = 10

            # inactive forest -> chance to become active in the fire again
            elif current_state == 10:
                if np.random.rand() < burn_protection_chance:
                    new_grid[i,j] = 3

            # protected chap tile + burning neightbours -> chance to turn to inactive chap
            elif current_state == 11 and burning_neighbors > 0:
                if np.random.rand() < burning_neighbors * prob_forest_burn:
                    new_grid[i, j] = 12

            # inactive chap -> chance to become active in the fire again
            elif current_state == 12:
                if np.random.rand() < burn_protection_chance:
                    new_grid[i,j] = 5

            # protected canyon tile + burning neightbours -> chance to turn to inactive canyon
            elif current_state == 13 and burning_neighbors > 0:
                if np.random.rand() < burning_neighbors * prob_forest_burn:
                    new_grid[i, j] = 14

            # inactive canyon -> chance to become active in the fire again
            elif current_state == 14:
                if np.random.rand() < burn_protection_chance:
                    new_grid[i,j] = 7

            # Burnt Out (state 8) and Lake (state 0) remain the same

    return new_grid

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
