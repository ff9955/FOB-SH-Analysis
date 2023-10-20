import numpy as np
import matplotlib.pyplot as plt
import itertools
import string

def data_splitter(file_path, list_length):
    
    '''
    function that filters out irrelevant lines in text files, and reads the relevant ones 
    into a numpy array

    input: path to the text file, list_length = the number of elements in the relevant lines 
    of the file, which is used to filter out the ones you don't want

    output: list of lists, where each list contains the chosen lines of data
    '''

    def internal_split_function(word):
        string_list = word.split() 
        return string_list

    # defining a function that splits strings into lists, such that it can be passed into the
    # 'map' function
    
    with open(file_path) as full_file:
        data_list = full_file.readlines() #data_list contains all lines in the file as a list of strings
        
        split_data_list = map(internal_split_function, data_list)
        split_data_list = list(split_data_list)
        #splitting every line of the file into a list of strings separated by spaces

        filtered_data_list = itertools.filterfalse(lambda element: len(element) != list_length, split_data_list)
        filtered_data_list = list(filtered_data_list)
        #filterfalse discards the lines longer than {list_length} elements, as this just leaves the raw data
    
    return filtered_data_list


def read_diabatic_populations(data_directory, number_trajectories):

    '''
    function for reading the diabatic state populations from all specified FOB-SH trajectories' 
    population files
    input: path to file's directory
    output: array with populations of all diabatic states of all trajectories at each timestep, the time axis,
    and the number of diabats in the simulation
    '''

    with open(data_directory + '/run-pop-1.xyz') as full_file:
        data_list = full_file.readlines()

        for string in data_list[0].split():
            if string[:-1].isnumeric():
                chosen_string = string[:-1]
                number_diabats = int(chosen_string)
                break
        #iterating over the first line of string to get the number of diabats in the run

        for string in data_list[2 + number_diabats].split():
            if string[-1].isnumeric():
                timestep = int(string)
                break

        #interating over the second string preceding the SECOND timestep, to get the timestep length

    first_split_data_list = data_splitter(data_directory + '/run-pop-1.xyz', list_length = 3)
    #filtering and splitting the lines into lists of strings for the first pop file

    first_data_array = np.array(first_split_data_list)
    #converting the resulting list of lists into a 2D array

    total_data_array = np.zeros((len(first_split_data_list), number_trajectories))
    #initialising a zero array; rows = number of timesteps; columns = number of trajectories

    #total_data_array will only contain the diabatic populations of the trajectories, NOT the 
    #time axis

    total_data_array[:, 0] = first_data_array[:,2]
    print('1')
    #first column filled with diabatic populations of first trajectory, which are located in the third
    #column of the pop file

    for number in range(1, number_trajectories):

        current_directory = data_directory[:-1] + f'{number + 1}'
        #re-naming the current directory so we enter the directory of the next consecutive trajectory

        print(f'{number + 1}')

        #basically doing the same thing as above but for the other trajectory

        split_data_list = data_splitter(current_directory + '/run-pop-1.xyz', list_length=3)
        split_data_array = np.array(split_data_list)

        total_data_array[:, number] = split_data_array[:, 2]


    number_timesteps = len(split_data_list)/number_diabats
    #each timestep has n diabats, so the total number of rows we're looking at is n*number_timesteps
        
    time_array = np.arange(0,number_timesteps,timestep)
    time_array = time_array[:, np.newaxis]
    #making the time axis using the timestep 

    print('oh yes') #oh yes

    return (total_data_array, time_array, number_diabats)


def read_supercell(data_directory):

    '''
    this function reads the file that has the xyz coordinates of each electronically 
    active molecule and returns the coordinates in a 2D numpy array format

    input: path to file

    output: 2D array of floats where rows contain xyz coordinates of every electronically active molecule
    '''

    file_path = data_directory + '/prepare_init/supercell/active_supercell_com.xyz'
        
    split_data_list = data_splitter(file_path, list_length = 4)
    #filtering out empty lines

    number_diabats = len(split_data_list)//2
    #number of molecules = number of diabats

    split_data_list = split_data_list[:number_diabats]
    #removing additional lines after the block of diabats
    CoM_array = np.array(split_data_list)

    CoM_array = CoM_array[:, 1:]
    #removing the first column of elements, as they just say 'C' - not important

    vector_int = np.vectorize(float)
    CoM_array = vector_int(CoM_array)
    #converting every string in the array into a float; we need to do this because
    #we're not passing our array of strings into an array of zeros

    return CoM_array


def MSD_x(population_info, supercell):
    
    '''
    function that uses cell coordinates and diabatic populations to calculate
    mean-squared x-displacement for each trajectory and as an average over all trajectories

    input: array of diabatic populations vs time step of every trajectory, array of cell coordinates
    '''

    diabatic_populations = population_info[0]

    time_array = population_info[1]
    number_diabats = population_info[2]

    #reading in information from the output of the read_diabatic_populations function

    displacements = supercell - supercell[0]
    #subtracting the first row from all rows to get an origin of zero

    x_displacements = displacements[:,0]
    #slicing the coordinate array to obtain only the x-coordinates of every diabat

    squared_x_displacements = x_displacements**2
    squared_x_displacements = squared_x_displacements[:, np.newaxis]

    squared_x_displacements = np.tile(squared_x_displacements, (len(time_array),1))
    #the aim here is to multiply the diabatic populations by their corresponding x-coordinates,
    #but each timestep will have n diabatic populations

    #the squared_x_displacements array has a length of n, but to multiply the whole array,
    #it needs a length of n*number_timesteps

    #the tile function extends the coordinate column by repeating it number_timesteps times, 
    #so every diabatic population can be multiplied by the corresponding x-coordinate

    weighted_squared_displacements = squared_x_displacements*diabatic_populations
    #multiplying the x-coordinates and the populations of the diabats

    WSD_shape = weighted_squared_displacements.shape
    average_squared_displacements = np.zeros((WSD_shape[0]//number_diabats, WSD_shape[1]))
    #initialising an array where rows = number_timesteps, since we're about to calculate the MSD
    #for each timestep by summing over each diabat's population-weighted displacement

    for index in range(0, len(diabatic_populations), number_diabats):

        #this loops down the rows of the array of the population-weighted displacements, and
        #sums all the displacements per timestep to get the MSD per timestep

        d_index = index//number_diabats
        #d_index is the index at which we pass the MSD at the given timestep into the
        #average_squared_displacements array

        average_squared_displacements[d_index, :] = np.sum(weighted_squared_displacements[index: index + number_diabats], axis=0)

    MSD_all_trajectories = np.hstack((time_array, average_squared_displacements))
    #adding the time axis to the array of MSD per time of every trajectory
    
    MSD_average = np.mean(MSD_all_trajectories[:,1:], axis = 1)
    MSD_average = MSD_average[:, np.newaxis]
    #MSD_average is the true MSD, as we have now averaged the population-weighted squared
    #displacements over every trajectory

    MSD_final = np.hstack((time_array, MSD_average))
    
    return (MSD_final, MSD_all_trajectories)


def MSD_y(population_info, supercell):

    '''
    function that does what MSD_X does, but along the y-axis
    '''

    diabatic_populations = population_info[0]

    time_array = population_info[1]
    number_diabats = population_info[2]

    displacements = supercell - supercell[0]

    y_displacements = displacements[:,1]
    # this is the only difference: slice through the second column of the file
    # to get the molecules' y-coordinates

    squared_y_displacements = y_displacements**2
    squared_y_displacements = squared_y_displacements[:, np.newaxis]

    squared_y_displacements = np.tile(squared_y_displacements, (len(time_array),1))
    weighted_squared_displacements = squared_y_displacements*diabatic_populations

    WSD_shape = weighted_squared_displacements.shape
    average_displacements = np.zeros((WSD_shape[0]//number_diabats, WSD_shape[1]))

    for index in range(0, len(diabatic_populations), number_diabats):

        d_index = index//number_diabats
        average_displacements[d_index, :] = np.sum(weighted_squared_displacements[index: index + number_diabats], axis=0)

    MSD_all_trajectories = np.hstack((time_array, average_displacements))
    
    MSD_average = np.mean(MSD_all_trajectories[:,1:], axis = 1)
    MSD_average = MSD_average[:, np.newaxis]

    MSD_final = np.hstack((time_array, MSD_average))
    
    return (MSD_final, MSD_all_trajectories)


def IPR(population_info):

    '''
    function that calculates the average IPR of all trajectories at each timestep

    input: the output of the read_diabatic_populations function

    output: 2D array of average IPR vs time, and 2D array of every individual trajectory's IPR vs time
    '''

    diabatic_populations = population_info[0]
    #rows = number of timesteps; columns = number of trajectories

    time_array = population_info[1]
    number_diabats = population_info[2]

    squared_diabatic_populations = diabatic_populations**2
    # IPR = reciprocal of sum of all squared diabatic populations

    population_shape = diabatic_populations.shape
    IPR_block = np.zeros((population_shape[0]//number_diabats, population_shape[1]))
    #summing over all diabatic populations of each timestep again, so number of rows
    #in the IPR array reduces by a factor of number_diabats

    for index in range(0, len(diabatic_populations), number_diabats):

        d_index = index//number_diabats
        summed_populations = np.sum(squared_diabatic_populations[index: index + number_diabats] ,axis=0)
        #sums over diabatic population of EVERY trajectory

        IPR_block[d_index, :] = 1/summed_populations

        #same idea as that for MSD, loop over the rows in steps of number_diabats, calculate the IPR
        #of every trajectory at each timestep by summing the squared diabatic populations

    full_IPR = np.hstack((time_array, IPR_block))

    average_IPR = np.mean(full_IPR[:, 1:] , axis=1)
    average_IPR = average_IPR[:, np.newaxis]

    average_IPR = np.hstack((time_array, average_IPR))

    return (average_IPR, full_IPR)


def statistical_sd(full_data_array, number_blocks):
    
    '''
    this function calculates the standard deviation associated with IPR's and MSD's
    calculated by averaging over trajectories, it does this by calculating the average 
    quantities for smaller sections of trajectories, and summing the differences they have
    with the average obtained from all the trajectories

    input: 2D array of IPR/MSD of every trajectory vs time, number of section you want to use

    output: row vector containing standard deviations of average IPR/MSD values
    at every timestep
    '''

    partial_data_array = full_data_array[:, 1:]
    full_average = np.mean(partial_data_array, axis=1)
    #re-calculating the IPR/MSD averaged over all trajectories

    number_trajectories = len(partial_data_array[0])

    assert (number_trajectories%number_blocks) == 0 ,'Number of trajectories must be divisible by number of blocks'

    block_size = number_trajectories//number_blocks
    #calculating the number of trajectories that should be in each block

    differences = np.zeros((len(partial_data_array)))
    #row vector of zeros with length equal to number of timesteps

    for index in range(0, number_trajectories, block_size):

        #looping over columns of block_size trajectories at a time, and calculating the mean

        single_block_average = np.mean(partial_data_array[:, index: index + block_size], axis=1)
        differences = differences + (full_average - single_block_average)**2

        #incrementally adding the squared differences to array of zeros, to get the total sum of
        #differences as a single row vector


    standard_deviations = np.sqrt(differences/number_blocks)
    #square-rooting and dividing by the number of blocks to convert the row vector of
    #sums (of differences) to a row vector of standard deviations for each time step
    
    return standard_deviations


def coc_x(population_info, supercell):
    
    '''
    function that calculates the centre of charge in the x-direction as a function of time

    input: ouput of read_diabatic_populations function, cell_coordinates

    output: 2D array, rows = time axis, columns = different trajectories
    '''

    diabatic_populations = population_info[0]

    time_array = population_info[1]
    number_diabats = population_info[2]

    displacements = supercell - supercell[0]

    #extracting diabatic populations of trajectories and the x-coordinate of each diabat

    x_displacements = displacements[:,0]
    x_displacements = x_displacements[:, np.newaxis]

    #the only difference between this function and MSD_x is that we don't square the
    #x-coordinates here

    x_displacements = np.tile(x_displacements, (len(time_array),1))
    weighted_displacements = x_displacements*diabatic_populations

    #tiling the x_displacement column so each diabatic population is multiplied by 
    #its corresponding x-coordinate

    WD_shape = weighted_displacements.shape
    average_displacements = np.zeros((WD_shape[0]//number_diabats, WD_shape[1]))

    for index in range(0, len(diabatic_populations), number_diabats):

        d_index = index//number_diabats
        average_displacements[d_index, :] = np.sum(weighted_displacements[index: index + number_diabats], axis=0)

        #looping over rows in intervals equal to number_diabats, summing over all population-weighted
        #x-displacements for a specific timestep before moving on

        #each time step has number_diabats diabatic populations

    COC_all_trajectories = np.hstack((time_array, average_displacements))
    
    return COC_all_trajectories


def coc_y(population_info, supercell):
    
    '''
    same function as coc_x, but calculating centre-of-charge along the y-axis
    '''

    diabatic_populations = population_info[0]

    time_array = population_info[1]
    number_diabats = population_info[2]

    displacements = supercell - supercell[0]

    y_displacements = displacements[:,1]
    y_displacements = y_displacements[:, np.newaxis]

    y_displacements = np.tile(y_displacements, (len(time_array),1))
    weighted_displacements = y_displacements*diabatic_populations

    WD_shape = weighted_displacements.shape
    average_displacements = np.zeros((WD_shape[0]//number_diabats, WD_shape[1]))

    for index in range(0, len(diabatic_populations), number_diabats):

        d_index = index//number_diabats
        average_displacements[d_index, :] = np.sum(weighted_displacements[index: index + number_diabats], axis=0)

    COC_all_trajectories = np.hstack((time_array, average_displacements))
    
    return COC_all_trajectories


def plot_coc(array, indexes):
    
    '''
    function that plots how the centre-of-charge of any (range of) trajectories
    varies with time

    input: ouput of coc_x or coc_y, a tuple of indeces or index of the trajectory 
    to be plotted

    output: matplotlib figure of COC vs time plot
    '''

    fig = plt.figure(figsize = (6,4))
    
    if type(indexes) == int:
        plt.plot(array[:,0], array[:,indexes], color = 'b')

        #int means you only specified one trajectory
    
    elif type(indexes) == tuple:

        #tuple means you specified a range of (consecutive) trajectories, so we will
        # have to iterate over them 

        loop_counter = 0
        N_traj = indexes[1] - indexes[0]
        
        for number in range(indexes[0], indexes[1]):

            shading_number = 1 - loop_counter/N_traj
            #shading gets more transparent as index increases

            plt.plot(array[:,0], array[:, number], color = 'b', alpha = shading_number, label = f'Trajectory {number}')
            #plotting slices of the full COC array vs time, using the indeces specified in the argument

            loop_counter += 1
            #loop counter is only used to change the shading of the graph, N_traj used to ensure
            #no graph has exactly the same shading


    #plt.legend()
    plt.title('Singe Traj. Centre of Charge')
    plt.ylabel(r'$\AA$')
    plt.xlabel('Time /fs')
    
    return fig


def variance(MSD_all_trajectories, COC_all_trajectories):
    '''
    function that calculates the spread of the electronic wavefunction at a given
    time using the COC and MSD quantities

    input: 2D array containing all trajectories' MSD and COC values vs time

    output: 2D array with a column of times and a column of corresponding variances
    '''

    MSD_shape = MSD_all_trajectories.shape

    variance_block = np.zeros((MSD_shape[0], MSD_shape[1]))
    #initialising array for time and variance of each trajectory

    variance_block[:,0] = MSD_all_trajectories[:,0]
    #setting first column to the time axis

    variance_block[:, 1:] = MSD_all_trajectories[:,1:] - (COC_all_trajectories[:,1:])**2
    #subtracting the squared COC's from each trajectory from the MSD of each individual trajectory

    #this was done for all trajectories by subtracting all trajectories' columns at once,
    #which allows us to immediately set all columns in variance_block to the individual 
    #trajectories' variances vs time

    total_variance = np.mean(variance_block[:, 1:], axis = 1)
    #calculating a row vector, whose elements are the variances averaged over all trajectories for
    #a given time
    
    variance_column = np.zeros((len(MSD_all_trajectories), 2))
    #initialising an array with 2 columns: one for the time axis, and the other for the average variances

    variance_column[:,0] = MSD_all_trajectories[:,0]
    #setting first column to time axis

    variance_column[:,1] = total_variance
    #setting second column to average variances

    return variance_column



def hits(coc_all_trajectories, variance, supercell):
    '''
    This function takes the COC vs time of individual trajectories and checks if they've
    reached the edge of the simulation box.

    Input:
        coc_all_trajectories (numpy array): Full array of trajectories' COC vs time.
        variance (numpy array): The average variance vs time.
        supercell (list of tuples): The coordinates of the simulation cell.

    Output:
        Prints out the trajectory and time at which a hit occurs.
    '''

    cell_edge = supercell[-1][0]  # Extracting the x-coordinate of the simulation cell's edge.

    # Calculating standard deviations from the variance array.
    variance_shape = variance.shape
    standard_deviations = np.zeros(variance.shape)

    standard_deviations[:, 0] = variance[:, 0]
    standard_deviations[:, 1] = np.sqrt(variance[:, 1])

    number_trajectories = len(coc_all_trajectories[0][1:])  # Getting the number of trajectories.
    number_timesteps = len(coc_all_trajectories)  # Getting the number of timesteps in the trajectories.

    # Loop through each trajectory.
    for index in range(1, number_trajectories):

        specific_trajectory = coc_all_trajectories[:, index]  # Extracting the specific trajectory.

        # Loop through each timestep of the trajectory.
        for index2 in range(number_timesteps):

            # Calculate overall displacement at each timestep.
            overall_displacement = specific_trajectory[index2] + standard_deviations[:, 1][index2]

            # Check if the overall displacement exceeds the cell's edge.
            if overall_displacement > cell_edge:

                # Print the hit occurrence information and exit the inner loop.
                print(f'Hit! Time: {index2} fs ; Trajectory: {index}')

                break  # Break the inner loop when a hit is found.