import copy

# Define the standard orientations
orientations = [0, 45, -45, 90]

def generate_sequences(stack, num_plies):
    """
    Recursively generate all valid stacking sequences for the given number of plies.
    """
    if not stack:
        return []
    # Check if the desired number of plies has been reached
    if len(stack) == num_plies:
        return [stack]

    # Determine the allowed orientations for the next ply
    allowed_orientations = orientations.copy()

    # Rule 2: Use at least 10% of each standard orientation
    for orientation in orientations:
        if stack.count(orientation) >= 0.1*num_plies:
            allowed_orientations.remove(orientation)

    # Rule 3: Minimize groupings of plies with the same orientation
    if stack.count(stack[-1]) >= 2:
        if stack[-1] in allowed_orientations:
            allowed_orientations.remove(stack[-1])

    # Rule 4: Do not locate tape plies with fibers perpendicular to a free edge
    if len(stack) >= num_plies-2:
        if stack[-1] == 90:
            allowed_orientations.remove(-90)
        elif stack[-1] == -90:
            allowed_orientations.remove(90)

    # Rule 6: Alternate +45° and -45° plies except for the closest to the mid-plane
    if (len(stack)-1) % 4 == 0 and len(stack) != num_plies-1:
        if -stack[-1] in allowed_orientations:
            allowed_orientations.remove(-stack[-1])

    # Rule 7: Separate groups of same-oriented tape plies from 90° plies by 45° plies
    prev_tape = len(stack) % 2 == 0 or (len(stack) - 1) % 2 == 0
    if prev_tape and stack.count(stack[-1]) > 1:
        for orientation in [45, -45]:
            if orientation in allowed_orientations:
                allowed_orientations.remove(orientation)

    # Rule 8: Locate 0° plies at least 3 plies from the outer surface
    if len(stack) > 3 and stack[0] == stack[-1] == 0:
        if stack[-1] in allowed_orientations:
            allowed_orientations.remove(0)

    # Generate all possible next plies and recursively call the function with each one
    sequences = []
    for orientation in allowed_orientations:
        next_stack = copy.copy(stack)
        next_stack.append(orientation)
        sequences.extend(generate_sequences(next_stack, num_plies))

    return sequences

# Example usage
if __name__=="__main__":
    stacks = generate_sequences([0], 16)
    for stack in stacks:
        print(stack)