import numpy as np
from slice import SliceEnv



def RandomBraid(braid_length_stdev=2, num_crossings_stdev=5, max_braid_index=12, max_braid_length=20, inaction_penalty=0.005, starting_strand_weighting=4):
	# Select the length of the braid word, by taking the smaller of the following two quantities:  the max_braid_length,
	# and a random normal variable, with mean 0 and standard deviation braid_length_stdev, which we take the absolute
	# value of, followed by the ceiling of.
    braid_word_length=np.minimum(np.ceil(np.abs(np.random.normal(0,braid_length_stdev))).astype(int),max_braid_length)
    # Select the braid index (not the actual braid index, but the range of crossings we can select), by taking the
    # minimum of the following two quantities:  max_braid_index-1, and a random normal variable, with mean 0 and
    # standard deviation braid_index_stdev, which we take the absolute value of, then the ceiling of. The quantity
    # braid_index is then chosen to be the resulting maximum, or 2, whichever is smaller.
    #
    # Notice this is less than max_braid_index, because this is meant to be used to pick the possible crossings; so it
    # for example max_braid_index is 12, then this will not be larger than 11, so that no crossings higher than 11 will
    # show up.  It shouldn't be called braid_index I guess, so I'll change it at some point.
    num_crossings=np.maximum(np.minimum(np.ceil(np.abs(np.random.normal(0,num_crossings_stdev))).astype(int),max_braid_index-1),2)
    # Decide whether to have the main component be the first strand, or another strand chosen uniformly from the set of all possible strands.
    if np.random.binomial(1, 0.9):
        strand_component = 1
    else:
        strand_component = np.random.choice(np.arange(1, num_crossings+2, 1), 1)[0]
    # Create a list of all of the possible crossings given our choice of braid index (goes from -num_crossings to +num_crossings and deletes the zero in the middle)
    possible_crossings = np.delete(np.arange(-num_crossings, num_crossings+1, 1), num_crossings)
    # Select the braid word using uniform probabilities.
    word = np.random.choice(possible_crossings, braid_word_length-1)
    # Insert a crossing from strand_component to the list.
    word = np.insert(word, np.random.choice(np.arange(0, len(word)+1)), np.random.choice([-1, 1])*np.min([np.max([strand_component-np.random.choice([0, 1]), 1]), num_crossings]))
    # Create the braid using the parameters computed above.
    braid = SliceEnv(word, max_braid_index, max_braid_length, inaction_penalty, starting_knot_strand=strand_component)
    # With 50% probability, redefine the Euler characterisitic list randomly (from the standard list that is created in SliceEnv())
    if np.random.binomial(1, 0.5):
        for key in list(braid.eulerchar.keys()):
            braid.eulerchar[key]-=np.floor(np.abs(np.random.normal(0,2))).astype(int)
    # Randomly select the cursor position.
    vert_position=np.random.randint(0, len(word)+1)
    hor_position=np.minimum(np.maximum(1, np.ceil(np.abs(np.random.normal(0, 0.5*num_crossings_stdev)))), max_braid_index-1).astype(int)
    braid.cursor=[vert_position,hor_position]
    return braid