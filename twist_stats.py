import numpy as np
import collections as cl

def unpack(line):
    values = line.split(",")
    assert(len(values) == 6)
    drmc, drme, cw, ccw, depth, sol = values
    sol = sol.strip(" \n,")
    return (int(drmc), int(drme)), int(cw), int(ccw), int(depth), sol

twist_index = {(3, 3):0, (6, 0):1, (0, 6):1, (4, 4):2, (7, 1):3, (1, 7):3}
edges_index = {0:0, 2:1, 4:2, 6:3, 8:4}

data_2d = np.zeros((4, len(edges_index)), 
    dtype=[("n_cases", int),
           ("avg_length", np.float64),
           ("sub6_chances", np.float64),
           ("sub7_chances", np.float64),
           ("best_alg", 'U30'),
           ("worst_alg", 'U30')])

depths = np.zeros(
    (4, len(edges_index)), 
    dtype = [("best", int), ("worst", int)])
depths.fill((100, 0))


with open("twist_raw_data.csv", "r", encoding="utf-8") as file:
    # Parsing the file to get drm cw and ccw depth and alg for each case
    for line in file:
        drm, cw, ccw, depth, alg = unpack(line)
        assert(depth < 11)
        if drm[0] == 6 or drm[0] == 8:
            i, j = twist_index[cw, ccw], edges_index[drm[1]]
            data_2d[i, j]["n_cases"] += 1
            data_2d[i, j]["avg_length"] += depth
            if depth <6:
                data_2d[i, j]["sub6_chances"] += 1
            if depth <7:
                data_2d[i, j]["sub7_chances"] += 1
            if depth < depths[i, j]["best"]:
                depths[i, j]["best"] = depth
                data_2d[i, j]["best_alg"] = alg
            if depth > depths[i, j]["worst"]:
                depths[i, j]["worst"] = depth
                data_2d[i, j]["worst_alg"] = alg

# Compute averages
data_2d["avg_length"] /= data_2d["n_cases"]
data_2d["sub6_chances"] /= data_2d["n_cases"]
data_2d["sub7_chances"] /= data_2d["n_cases"]

print(data_2d["avg_length"])
print(data_2d["sub6_chances"])
print(data_2d["sub7_chances"])
