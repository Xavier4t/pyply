import random
import numpy as np
import argparse
from itertools import groupby
"""
Reference: 
Michael Chun-Yu Niu, Michael Niu, "Composite Airframe Structures", Hong Kong Conmilit Press Ltd. (2005)

Summary of Stacking Sequence Design Considerations:
1. Use balanced and symmetric layups
2. Intersperse ply orientations

"If possible, avoid grouping 90° plies; separate them by 0° or ±45° 
  (0° is direction of critical load) to  minimize interlaminar shear and normal stresses.
  and whenever possible maintain a homogeneous stacking sequence and avoid 
  grouping or similar plies. If plies must be grouped, avoid grouping more 
  than 5 plies of the same oriantation together to minimaze edge splitting."

3. Minimize groupings of same orientation
4. Alternate +θ° and -θ° plies through the layup
5. Separate groups of same orientation by 45° plies

6. Provide at least 10% of each of the four ply orientation (10% design rule)
7. Careful designs needed at locations prone to delamination (e.g. avoiding tape plies with fibers perpendicular to an edge)
8. Exterior surface plies should be continuous and ±45° to the primary load direction (e.g. locating 0° plies at least 3 plies from the surface).

"Design to reduce Poisson's ratio: 
    - consider the use of 90° plies in a laminate
    - reduce the % of 0° plies
    - reduce of Poisson's ration is critical in bonded parts"
"""
parser= argparse.ArgumentParser()

# Number of plies in each stacking sequence
NUM_PLIES = parser.add_argument('--num_plies', type=int, required=True, help="int: number of plies in the stacking sequence")


# Population size and the number of iterations
POP_SIZE = parser.add_argument('--pop_size', type=int, required=False, help="int: maximum number of stackings generated")
NUM_ITERATIONS = parser.add_argument('--num_iter', type=int, required=False, help="int: maximum number of iterations")

# Parse the argument
args=parser.parse_args()
if not args.pop_size:
    POP_SIZE = 100
if not args.num_iter:
    NUM_ITERATIONS=1000

# Fitness function
def fitness(seq):
    # 1. Use balanced and symmetric layups
    num_45deg_upper = sum(ply[1] == 1 for ply in seq[:NUM_PLIES//2])
    num_45deg_lower = sum(ply[1] == 2 for ply in seq[NUM_PLIES//2:])
    if num_45deg_upper != num_45deg_lower:
        return float('-inf')
    
    # 2. Intersperse ply orientations
    num_0deg = sum(ply[1] == 0 for ply in seq)
    num_45deg = sum(ply[1] == 1 or ply[1] == 2 for ply in seq)
    num_90deg = sum(ply[1] == 3 for ply in seq)
    if num_0deg < NUM_PLIES//10 or num_45deg < NUM_PLIES//10 or num_90deg < NUM_PLIES//10:
        return float('-inf')

    # 3. Minimize groupings of plies with the same orientation
    group_lengths = [len(list(group)) for key, group in groupby(seq, key=lambda ply: ply[1])]
    if max(group_lengths) > 1:
        return float('-inf')


    # 4. Alternate +θ° and -θ° plies except for the closest ply on either side of the mid-plane
    if seq[NUM_PLIES//2-1][1] == seq[NUM_PLIES//2][1] or seq[NUM_PLIES//2][1] == seq[NUM_PLIES//2+1][1]:
        return float('-inf')
    for i in range(NUM_PLIES-3):
        if seq[i][1] == seq[i+1][1] and seq[i+1][1] == seq[i+2][1]:
            return float('-inf')

    # 5. Separate groups of same orientation by 45° plies (e.g, tape plies of 90° apart by at least one ply of 45° apart)
    for i in range(NUM_PLIES-2):
        if seq[i][1] == seq[i+1][1] == seq[i+2][1]:
            if seq[i][1] == 3:
                if i == NUM_PLIES//2-2:
                    if seq[i+3][1] == 1 or seq[i+3][1] == 2:
                        return float('-inf')
                elif i == NUM_PLIES//2-1:
                    if seq[i-1][1] == 1 or seq[i-1][1] == 2 or seq[i+2][1] == 1 or seq[i+2][1] == 2:
                        return float('-inf')
                elif i == NUM_PLIES//2:
                    if seq[i-2][1] == 1 or seq[i-2][1] == 2 or seq[i+1][1] == 1 or seq[i+1][1] == 2:
                        return float('-inf')
                    
    # 6. 10% design rule
    orientations = [0, 0, 0, 0] # count of 0°, 45°, -45°, 90°
    for i in range(NUM_PLIES):
        orientations[seq[i][1]] += 1
        if min(orientations) < NUM_PLIES * 0.1:
            return float('-inf')
               
    # 7. At a free edge, do not locate tape plies with fibers orientated perpendicular to the edge
    if seq[0][1] == 3 or seq[NUM_PLIES-1][1] == 3:
        if seq[NUM_PLIES//2][1] == 1 or seq[NUM_PLIES//2][1] == 2:
            return float('-inf')
        
    # 8. Locate tape plies with fibers oriented in the primary load direction such that there is a least three plies between them and the laminate surface.
    for i in range(NUM_PLIES):
        if seq[i][1] == 0:
            if i < 3 or i > NUM_PLIES - 4:
                return float('-inf')
            if seq[i-1][1] != 45 or seq[i-2][1] != -45 or seq[i-3][1] != 45:
                return float('-inf')
            if seq[i+1][1] != 45 or seq[i+2][1] != -45 or seq[i+3][1] != 45:
                return float('-inf')