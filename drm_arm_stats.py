import numpy as np

def unpack(line):
    values = line.split(",")
    assert(len(values) == 6)
    drmc_str, drme_str, armc_str, arme_str, depth, sol = values
    sol = sol.strip(" \n,")
    return (int(drmc_str), int(drme_str)), (int(armc_str), int(arme_str)), int(depth), sol

drm_to_index = {}
k = 0
for bc in [0, 2, 3, 4, 5, 6, 7, 8]:
    for be in range(0, 10, 2):
        drm_to_index[(bc, be)]=k
        k+=1
n_drm = len(drm_to_index)

arm_to_index = {}
k = 0
for bc in [0, 1, 2, 3, 4, 5, 6, 8]:
    for be in range(0, 5):
        arm_to_index[(bc, be)]=k
        k+=1
n_arm = len(arm_to_index)

class Table2D:
    # Custom object to store the 2D data of worst and best solutions
    def __init__(self, value):
        self.contents = [[value for i in arm_to_index] for j in drm_to_index]

    def set(self, drm, arm, value):
        self.contents[drm_to_index[drm]][arm_to_index[arm]] = value
    
    def get(self, drm, arm):
        return self.contents[drm_to_index[drm]][arm_to_index[arm]]

    def __iter__(self):
        return self.contents.__iter__()

    def __str__(self):
        return str(self.contents)
    
# Using numpy to store data on average optimal and probabilities
# because I want to use numpy's savetxt functions
 
data_2d = np.zeros((n_drm, n_arm), 
    dtype=[("n_cases", int),
           ("avg_length", np.float64),
           ("sub6_probability", np.float64),
           ("sub7_probability", np.float64)])

best_solutions_2d = Table2D("UNSOLVABLE")
worst_solutions_2d = Table2D("UNSOLVABLE")
max_2d = Table2D(-1)
min_2d = Table2D(20)

with open("raw_data.csv", "r", encoding="utf-8") as file:
    # Parsing the file to get drm arm depth and alg for each case
    for line in file:
        drm, arm, depth, alg = unpack(line)
        assert(depth < 11)

        # Add up the values for avg length and sub-x probability
        drm_i, arm_i = drm_to_index[tuple(drm)], arm_to_index[tuple(arm)]
        data_2d[drm_i, arm_i]["n_cases"] += 1
        if depth < 6:
            data_2d[drm_i, arm_i]["sub6_probability"] += 1
        if depth < 7:
            data_2d[drm_i, arm_i]["sub7_probability"] += 1
        data_2d[drm_i, arm_i]["avg_length"] += depth

        # Update best and worst alg as we meet them
        if depth > max_2d.get(drm, arm):
            worst_solutions_2d.set(drm, arm, alg)
            max_2d.set(drm, arm, depth)
        if depth < min_2d.get(drm, arm):
            best_solutions_2d.set(drm, arm, alg)
            min_2d.set(drm, arm, depth)

# This produces nan values, but numpy can handle it safely
data_2d["avg_length"] /= data_2d["n_cases"]
data_2d["sub6_probability"] /= data_2d["n_cases"]
data_2d["sub7_probability"] /= data_2d["n_cases"]


# Writing to files
np.savetxt('n_cases.csv', data_2d["n_cases"], fmt='%d', delimiter=',')
np.savetxt('avg_length.csv', data_2d["avg_length"], fmt='%2.4f', delimiter=',')
np.savetxt('sub6_probability.csv', data_2d["sub6_probability"], fmt='%2.4f', delimiter=',')
np.savetxt('sub7_probability.csv', data_2d["sub7_probability"], fmt='%2.4f', delimiter=',')

with open("worst_solutions.csv", "w") as f:
    for row in worst_solutions_2d:
        for alg in row:
            f.write(alg)
            f.write(", ")
        f.write('\n')

with open("best_solutions.csv", "w") as f:
    for row in best_solutions_2d:
        for alg in row:
            f.write(alg)
            f.write(", ")
        f.write('\n')