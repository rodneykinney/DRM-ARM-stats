import numpy as np
import collections as cl

def unpack(line):
    values = line.split(",")
    assert(len(values) == 6)
    drmc, drme, cw, ccw, depth, sol = values
    sol = sol.strip(" \n,")
    return (int(drmc), int(drme)), int(cw), int(ccw), int(depth), sol

class CaseEntry:
    n_cases = 0 
    avg_length=0.0
    sub6_chances=0.0
    sub7_chances=0.0
    best_alg=""
    worst_alg=""

    def __repr__(self):
        return f"Average optimal: {self.avg_length:.5f}\nChances of sub 6: {self.sub6_chances:.5f}\nChances of sub 7: {self.sub7_chances:.5f}\nBest case: {self.best_alg}\nWorst case: {self.worst_alg}"

squashed = {
    (twist, edges):CaseEntry() 
        for twist in ((3, 3), (6, 0), (0, 6), (4, 4), (7, 1), (1, 7)) 
        for edges in (0, 2, 4, 6, 8)
    }

worst_cases = {
    (twist, edges):0
        for twist in ((3, 3), (6, 0), (0, 6), (4, 4), (7, 1), (1, 7)) 
        for edges in (0, 2, 4, 6, 8)
    }

best_cases = {
    (twist, edges):100
        for twist in ((3, 3), (6, 0), (0, 6), (4, 4), (7, 1), (1, 7)) 
        for edges in (0, 2, 4, 6, 8)
    }

with open("twist_raw_data.csv", "r", encoding="utf-8") as file:
    # Parsing the file to get drm cw and ccw depth and alg for each case
    for line in file:
        drm, cw, ccw, depth, alg = unpack(line)
        assert(depth < 11)
        if drm[0] == 6 or drm[0] == 8:
            squashed[((cw, ccw), drm[1])].n_cases += 1
            squashed[((cw, ccw), drm[1])].avg_length += depth
            if depth <6:
                squashed[((cw, ccw), drm[1])].sub6_chances += 1
            if depth <7:
                squashed[((cw, ccw), drm[1])].sub7_chances += 1
            if depth < best_cases[((cw, ccw), drm[1])]:
                best_cases[((cw, ccw), drm[1])] = depth
                squashed[((cw, ccw), drm[1])].best_alg = alg
            if depth > worst_cases[((cw, ccw), drm[1])]:
                worst_cases[((cw, ccw), drm[1])] = depth
                squashed[((cw, ccw), drm[1])].worst_alg = alg

# Compute averages
for key in squashed:
    squashed[key].avg_length /= squashed[key].n_cases
    squashed[key].sub6_chances /= squashed[key].n_cases
    squashed[key].sub7_chances /= squashed[key].n_cases

# # Clear duplicates cause (7, 1) is just a (7, 1) symetry
# squashed[(7, 1)].n_cases += squashed[(1, 7)].n_cases
# squashed[(6, 0)].n_cases += squashed[(0, 6)].n_cases
# squashed = {key: squashed[key] for key in squashed if key not in ((1, 7), (0, 6))}

for key in squashed:
    print(key)
    print(squashed[key])
