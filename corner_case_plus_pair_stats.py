from orbit_stats import*

# For each UD edge position, store the index
# of the adjacent corners and the orientation
# that they need to have to make a pair
corner_neighbours = {
    0  : ((0, 1), (2, 1)),
    1  : ((1, 2), (2, 1)),
    2  : ((2, 3), (2, 1)),
    3  : ((3, 0), (2, 1)),
    8  : ((4, 5), (1, 2)),
    9  : ((5, 6), (1, 2)),
    10 : ((6, 7), (1, 2)),
    11 : ((7, 4), (1, 2))
}

def get_number_of_pairs(co, esl):
    np = 0
    for e in (0, 1, 2, 3, 8, 9, 10, 11):
        if esl[e] == '1':
            cp, cop = corner_neighbours[e]
            if int(co[cp[0]]) == cop[0]: np += 1
            if int(co[cp[1]]) == cop[1]: np += 1
    return np

if __name__ == "__main__":
    pairs_data = np.zeros((n_corner_cases, 5), 
        dtype=[("n_cases", int),
            ("avg_length", np.float64),
            ("sub6_probability", np.float64),
            ("sub7_probability", np.float64)])

    with open("full_data.csv", "r", encoding="utf-8") as file:

        for line in file:
            index, co, esl, d, sol = unpack(line)
            drm = get_drm(co, esl)
            cc = get_corner_case(co)
            if drm == (4, 4):
                ci = cc_to_index[cc]
                pi = get_number_of_pairs(co, esl)
                pairs_data[ci, pi]['n_cases'] += 1
                pairs_data[ci, pi]['avg_length'] += d
                if d < 6:
                    pairs_data[ci, pi]['sub6_probability'] += 1
                if d < 7:
                    pairs_data[ci, pi]['sub7_probability'] += 1

    for ci in range(n_corner_cases):
        for pi in range(5):
            if pairs_data[ci, pi]['n_cases'] != 0:
                pairs_data[ci, pi]['avg_length'] /= pairs_data[ci, pi]['n_cases']
                pairs_data[ci, pi]['sub6_probability'] /= pairs_data[ci, pi]['n_cases']
                pairs_data[ci, pi]['sub7_probability'] /= pairs_data[ci, pi]['n_cases']

    np.set_printoptions(precision = 2, suppress = True)
    print(pairs_data['n_cases'])
    print(pairs_data['sub6_probability'])
    print(pairs_data['sub7_probability'])
    print(pairs_data['avg_length'])