import re
import sys
from collections import Counter, namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from itertools import chain, product
import random

import numpy as np

import vfmc_core

Bucket = namedtuple(
    "Bucket",
    [
        "n_bad_corners",
        "corner_orbit_split",
        "corner_orbit_parity",
        "corner_arm",
#        "corner_arm_split",
        "edge_arm",
#        "edge_arm_split",
        "n_bad_edges",
        "n_pairs",
        # "n_fake_pairs",
        # "n_side_pairs",
        "move_count",
    ]
)

Selection = namedtuple(
    "Selection",
    [
        "n_bad_corners",
        "corner_orbit_split",
        "corner_orbit_parity",
        "corner_arm",
#        "corner_arm_split",
        "edge_arm",
#        "edge_arm_split",
        "n_bad_edges",
        "n_pairs",
        # "n_fake_pairs",
        # "n_side_pairs",
        "move_count",
    ]
)


def unpack(line):
    values = line.split(",")
    assert (len(values) == 5)
    index_str, co_str, eo_str, depth, sol = values
    sol = sol.strip(" \n,")
    return int(index_str), co_str, eo_str, int(depth), sol


def get_drm(co, eo):
    drm_c = sum(int(o != '0') for o in co)
    drm_e = 8 - 2 * sum(int(eo[i]) for i in range(4, 8))
    return drm_c, drm_e


def get_arm(co, eo):
    arm_c_l = sum(int(co[i] == '1') for i in (3, 4)) + \
            sum(int(co[i] == '2') for i in (0, 7))
    arm_c_r = sum(int(co[i] == '1') for i in (1, 6)) + \
              sum(int(co[i] == '2') for i in (2, 5))
    arm_e_u = sum(int(eo[i] == '1') for i in (0, 2))
    arm_e_d = sum(int(eo[i] == '1') for i in (8, 10))
    return (arm_c_l, arm_c_r), (arm_e_u, arm_e_d)


# Orbit 1 : ULF, URB, DRF, DLB
# Orbit 2 : ULB, URF, DLF, DRB
orbit_1 = (0, 2, 5, 7)
orbit_2 = (1, 3, 4, 6)


def get_orbit_splits(co):
    orbit_1_bc = sum(int(co[i] != '0') for i in orbit_1)
    orbit_2_bc = sum(int(co[i] != '0') for i in orbit_2)
    return orbit_1_bc, orbit_2_bc


U_layer = (0, 1, 2, 3)
D_layer = (4, 5, 6, 7)


def get_UD_splits(co):
    U_bc = sum(int(co[i] != '0') for i in U_layer)
    D_bc = sum(int(co[i] != '0') for i in D_layer)
    return U_bc, D_bc


L_layer = (0, 3, 4, 7)
R_layer = (1, 2, 5, 6)


def get_LR_splits(co):
    L_bc = sum(int(co[i] != '0') for i in L_layer)
    R_bc = sum(int(co[i] != '0') for i in R_layer)
    return L_bc, R_bc


# For each UD edge position, store the index
# of the adjacent corners and the orientation
# that they need to have to make a pair
corner_neighbours = {
    0: ((0, 1), (2, 1)),
    1: ((1, 2), (2, 1)),
    2: ((2, 3), (2, 1)),
    3: ((3, 0), (2, 1)),
    4: ((3, 4), (2, 1)),
    5: ((2, 5), (2, 1)),
    6: ((1, 6), (1, 2)),
    7: ((0, 7), (1, 2)),
    8: ((4, 5), (1, 2)),
    9: ((5, 6), (1, 2)),
    10: ((6, 7), (1, 2)),
    11: ((7, 4), (1, 2))
}


def get_pair_counts(co, eo):
    """Return number of top pairs, top pseudo-pairs, side pairs"""
    np = 0 # Top pairs
    npp = 0 # Top pseudo-pairs
    nsp = 0 # Side pairs
    for e in (0, 1, 2, 3, 8, 9, 10, 11):
        if eo[e] == '1':
            cp, cop = corner_neighbours[e]
            if int(co[cp[0]]) == cop[0]:
                np += 1
            elif int(co[cp[0]]) != 0:
                npp += 1
            if int(co[cp[1]]) == cop[1]:
                np += 1
            elif int(co[cp[1]]) != 0:
                npp += 1
    for e in (4, 5, 6, 7):
        if eo[e] == '0':
            cp, cop = corner_neighbours[e]
            if int(co[cp[0]]) == cop[0]:
                nsp += 1
            elif int(co[cp[1]]) == cop[1]:
                nsp += 1
    return np, npp, nsp


RECOMMENDATIONS = {
    '4c2e': ['4a-2,2e,1', '4a-2,2e,2', '4b-0,2e,1', '4b-0,2e,2', '4b-2,2e,2', '4c-1,2e,1',
             '4c-1,2e,1', '4d-2,2e,1'],
    '4c4e': ['4a-0,4e,2', '4a-0,4e,4', '4a-2,4e,2', '4a-2,4e,3', '4b-0,4e,3', '4b-2,4e,2',
             '4b-2,4e,3', '4b-2,4e,4', '4c-1,4e,2', '4c-1,4e,3', '4d-2,4e,1', '4d-2,4e,2'],
    '4c4e,3+': ['4a-0,4e,4', '4a-2,4e,3', '4b-0,4e,3', '4b-2,4e,3', '4b-2,4e,4', '4c-1,4e,3'],
    '4a-04e': ['4a-0,4e,2', '4a-0,4e,4'],
    '4a-24e': ['4a-2,4e,2', '4a-2,4e,3'],
    '4b-04e': ['4b-0,4e,3'],
    '4b-24e': ['4b-2,4e,2', '4b-2,4e,3', '4b-2,4e,4'],
    '4c-14e': ['4c-1,4e,2', '4c-1,4e,3'],
    '4d-24e': ['4d-2,4e,1', '4d-2,4e,2'],
}


def pattern(sol: str) -> str:
    sol = sol.replace("'", "")
    sol = re.sub("L", "R", sol)
    sol = re.sub("D", "U", sol)
    p = "".join(reversed(sol.split(' ')))
    p = re.sub("[UDRLFB]2", ".", p)
    p = re.sub(r"^[\.U]*", "", p)
    p = re.sub(r"(R[\.R]+)R", "R+", p)
    p = re.sub(r"([\.U]*U([\.U]+))|([\.U]+)U[\.U]*", "U+", p)
    p = re.sub(r"^\.*", "", p)
    return p


def select(corner_case: str,
           n_bad_edges: int,
           corner_arm: Optional[int] = None,
           corner_arm_split: Optional[int] = None,
           edge_arm: Optional[int] = None,
           edge_arm_split: Optional[int] = None,
           n_pairs: Optional[int] = None,
           n_fake_pairs: Optional[int] = None,
           n_side_pairs: Optional[int] = None,
           max_move_count: Optional[int] = None,
           ):
    m = re.search(r"^([02345678])([abcd])(?:-([01234]))?$", corner_case)
    n_bad_corners = int(m.group(1))
    orbit_case = m.group(2)
    corner_orbit_split = slice(None)
    corner_orbit_parity = slice(None)
    if n_bad_corners != 4:
        if orbit_case == "a":
            corner_orbit_split = 1
        elif orbit_case == "b":
            corner_orbit_split = 0
    elif orbit_case == "a":
        corner_orbit_split = 2
        corner_orbit_parity = 0
    elif orbit_case == "b":
        corner_orbit_split = 2
        corner_orbit_parity = 1
    elif orbit_case == "c":
        corner_orbit_split = 1
    elif orbit_case == "d":
        corner_orbit_split = 0
    if corner_arm is None:
        corner_arm = slice(None)
        if m.group(3):
            corner_arm = int(m.group(3))
            if n_bad_corners - corner_arm != corner_arm:
                corner_arm = [corner_arm, n_bad_corners-corner_arm]
    if corner_arm_split is None:
        corner_arm_split = slice(None)
    if edge_arm is None:
        edge_arm = slice(None)
    if edge_arm_split is None:
        edge_arm_split = slice(None)
    if n_pairs is None:
        n_pairs = slice(None)
    if n_fake_pairs is None:
        n_fake_pairs = slice(None)
    if n_side_pairs is None:
        n_side_pairs = slice(None)
    move_count = slice(None)
    if max_move_count is not None:
        move_count = slice(0, max_move_count + 1)
    return Selection(
        n_bad_corners,
        corner_orbit_split,
        corner_orbit_parity,
        corner_arm,
        # corner_arm_split,
        edge_arm,
        # edge_arm_split,
        n_bad_edges,
        n_pairs,
        # n_fake_pairs,
        # n_side_pairs,
        move_count,
    )


def trigger(generator: str, default_drm) -> Tuple[str, bool, int]:
    sol = generator.replace("'", "")
    sol = re.sub("D", "U", sol)
    p = "".join(reversed(sol.split(' ')))
    p = p.replace("RL2UR","L2RUR")
    p = p.replace("LR2UL","R2LUL")
    p = p.replace("LR2UR","R2LUR")
    p = p.replace("RL2UL","L2RUL")
    p = re.sub("[UDRLFB]2", ".", p)
    trigger_drm, trigger_moves = "4c4e", p[-1:]
    m = re.search(r"^.*(?:([RL]\.\.+[RL])|(R\.R|L\.L)|(R\.L|L\.R)|(RUR|LUL)|(RUL|LUR)|(RL|LR))$", p)
    if m:
        triggers = ["AR", "4c2e", "4c6e", "3c2e", "7c8e", "8c8e"]
        trigger_moves = next((t for t in m.groups() if t))
        trigger_drm = next((s for i, s in enumerate(triggers) if m.group(i + 1)))

    setup = re.sub(f"{trigger_moves}$", "", p)
    off_axis_count = sum(1 for m in setup if m in ("L", "R"))
    dr_breaking = (trigger_drm in (default_drm, "AR") and off_axis_count > 0) or off_axis_count > 1
    return trigger_drm, dr_breaking, len(trigger_moves)

def stages(generator: str, default_drm) -> List[str]:
    trigger_drm, dr_breaking, trigger_move_count = trigger(generator, default_drm)
    if not dr_breaking:
        return [] if trigger_drm == default_drm else [trigger_drm]
    else:
        stages = [trigger_drm]
        cube = vfmc_core.Cube("")
        step = vfmc_core.StepInfo("dr","ud")
        for i, move in enumerate(generator.split(" ")):
            cube.apply(vfmc_core.Algorithm(move))
            if i >= trigger_move_count and move in {"R","R'","L","L'"}:
                stages.append(step.case_name(cube))
        stages = stages[:-1]
        stages.reverse()
        return stages


bad_corner_cases = 9
orbit_split_cases = 5
orbit_parity_cases = 2
corner_arm_cases = 9
corner_arm_split_cases=5
edge_arm_cases = 5
#edge_arm_split_cases=3
bad_edge_cases = 9
max_pairs = 9
# max_fake_pairs = 9
# max_side_pairs = 9
max_move_count = 11


class Stats:
    def __init__(self, n_bad_corners, n_bad_edges):
        self.counts = np.zeros((
            bad_corner_cases,
            orbit_split_cases,
            orbit_parity_cases,
            corner_arm_cases,
            # corner_arm_split_cases,
            edge_arm_cases,
            # edge_arm_split_cases,
            bad_edge_cases,
            max_pairs,
            # max_fake_pairs,
            # max_side_pairs,
            max_move_count,
        ),
            dtype = [("n", int),("solutions", object)]
        )

    def pattern_counts(self, selection):
        patterns = [pattern(sol) for sol in self.solutions(selection)]
        counts = Counter(patterns)
        total = sum(n for _, n in counts.items())
        return sorted([(p, n / total) for p, n in counts.items()], key=lambda x: -x[1])

    def trigger_counts(self, drm, selection):
        triggers = [(trig, dr_breaking) for s in self.solutions(selection) for trig, dr_breaking, _ in [trigger(s, drm)] ]
        counts = dict(Counter(triggers).items())
        total = sum(n for _, n in counts.items())
        header = ["trigger","DR-preserving","example","DR-breaking","example"]
        table = []
        for trig in set((t for t,_  in triggers)):
            l = [s for s in self.solutions(selection) for t, drb, _ in [trigger(s, drm)] if (t, drb) == (trig, False)]
            example = f'"{random.choice(l)}"' if l else ""
            l = [s for s in self.solutions(selection) for t, drb, _ in [trigger(s, drm)] if (t, drb) == (trig, True)]
            example_drb = f'"{random.choice(l)}"' if l else ""
            table.append((trig, counts.get((trig, False), 0) / total, example,
                          counts.get((trig, True), 0) / total, example_drb))
        table.sort(key=lambda x: -x[1])
        return [header] + table

    def stage_counts(self, drm, selection):
        stage = [" ".join(stages(sol, drm)) for sol in self.solutions(selection)]
        counts = dict(Counter(stage).items())
        total = sum(n for _, n in counts.items())
        pass
        header = ["Switch", "Frequency", "Generator"]
        table = []
        for st in set(stage):
            l = [s for s in self.solutions(selection) if " ".join(stages(s, drm)) == st]
            examples = ""
            if l:
                random.shuffle(l)
                examples = "\t".join(l[:5])
            table.append((st, counts.get(st) / total, examples))
        table.sort(key=lambda x: -x[1])
        return [header] + table



    def solutions_with_pattern(self, selection, p):
            return [s for s in self.solutions(selection) if pattern(s) == p]

    def p_sub(self, n: int):
        def f(selection: Selection):
            total = np.sum(self.counts[selection]["n"])
            if total == 0:
                return "-"
            selection = selection[:-1] + (slice(0,n),)
            return np.sum(self.counts[selection]["n"]) / total
        return f

    def solutions(self, selection: Selection):
        return list(chain.from_iterable(t for t in self.counts[selection]["solutions"].ravel() if t != 0))

    def n_cases(self, selection: Selection):
        return np.sum(self.counts[selection]["n"])

    def print(self, rows: List[Tuple[str, List]], columns: Tuple[str, List[int]], quantity: Callable):
        row_names = [r[0] for r in rows]
        header = "{}\t{}".format('\t'.join(row_names), columns[0])
        print(header)
        header = "{}\t{}".format('\t'.join(' ' *len(rows)), '\t'.join(str(v) for v in columns[1]))
        print(header)
        for row_values in product(*[v[1] for v in rows]):
            print("\t".join(str(v) for v in row_values),end="\t")
            row_features = dict(zip(row_names, row_values))
            col_features = [{columns[0]:v} for v in columns[1]]
            selections = [select(**(row_features | col_f)) for col_f in col_features]
            quantities = [str(quantity(s)) for s in selections]
            print("\t".join(quantities))

    @staticmethod
    def parse_line(line) -> Tuple[Bucket, str]:
        index, co, eo, move_count, sol = unpack(line)
        n_bad_corners, n_bad_edges = get_drm(co, eo)
        a, b = get_orbit_splits(co)
        corner_orbit_split = min(a, b, 4 - a, 4 - b)
        corner_orbit_parity = 0
        if n_bad_corners == 4 and corner_orbit_split == 2:
            if min(*(get_LR_splits(co) + get_UD_splits(co))) == 1:
                corner_orbit_parity = 1
        (arm_c_l, arm_c_r), (arm_e_u, arm_e_d) = get_arm(co, eo)
        n_pairs, n_fake_pairs, n_side_pairs = get_pair_counts(co, eo)
        drm = f"{n_bad_corners}c{n_bad_edges}e"

        # rzp, dr_breaking = trigger(sol, drm)
        bucket = Bucket(n_bad_corners=n_bad_corners,
                     corner_orbit_split=corner_orbit_split,
                     corner_orbit_parity=corner_orbit_parity,
                     corner_arm=arm_c_l+arm_c_r,
                     # corner_arm_split = min(arm_c_l, arm_c_r),
                     edge_arm=arm_e_u+arm_e_d,
#                     edge_arm_split = min(arm_e_u, arm_e_d),
                     n_bad_edges=n_bad_edges,
                     n_pairs=n_pairs,
                     # n_fake_pairs=n_fake_pairs,
                     # n_side_pairs=n_side_pairs,
                     move_count=move_count,
                     )
        return bucket, sol

    @staticmethod
    def load(n_bad_corners, n_bad_edges, filename="full_data.csv") -> "Stats":
        stats = Stats(n_bad_corners, n_bad_edges)
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    bucket, sol = Stats.parse_line(line)
                    stats.counts[bucket]["n"] += 1
                    sols = stats.counts[bucket]["solutions"]
                    if sols == 0:
                        sols = []
                        stats.counts[bucket]["solutions"] = sols
                    sols.append(sol)
                except:
                    print(line)
                    raise

        return stats

def corner_case_name(n_bad_corners, corner_orbit_split, corner_orbit_parity):
    if n_bad_corners == 4:
        if corner_orbit_split == 2:
            if corner_orbit_parity == 0:
                return "4a"
            else:
                return "4b"
        elif corner_orbit_split == 1:
            return "4c"
        elif corner_orbit_split == 0:
            return "4d"
    elif n_bad_corners in [2,3,5,6]:
        if corner_orbit_split == 1:
            return f"{n_bad_corners}a"
        else:
            return f"{n_bad_corners}b"
    else:
        return f"{n_bad_corners}c"

ALL_CORNER_CASES = [["0c-0"],[],["2c-0","2c-1"],["3c-0","3c-1"],["4a-0","4a-2","4b-0","4b-2","4c-1","4d-2"],["5a-1","5a-2","5b-0","5b-2"],["6a-0","6a-2","6a-3","6b-1","6b-2","6b-3"],["7c-1","7c-2","7c-3"],["8c-0","8c-2","8c-3","8c-4"]]
REDUCED_CORNER_CASES = [["0c"],[],["2c"],["3c"],["4a-0","4a-2","4b-0","4b-2","4c-1","4d-2"],["5c"],["6c"],["7c"],["8c"]]

def columns(field, values, **kwargs):
    def f(case: str, edges: int):
        return [select(case, edges, **(kwargs | {field: v})) for v in values]
    return f

if __name__ == "__main__":
    # sol = "R D' R' L' D L F2"
    # sol_stages = stages(sol)
    # print(sol_stages)
    trigger("L U' R2 L U", "3c2e")

    nc = int(sys.argv[1])
    ne = int(sys.argv[2])
    stats = Stats.load(nc, ne, f"{nc}c{ne}e.csv")
    def print_stages(case_name, selection: Selection):
        table = stats.stage_counts(f"{nc}c{ne}e", selection)
        print(case_name)
        for row in table:
            print("\t".join((str(s) or "-" for s in row)))
        print("")
    combined = Selection(
        n_bad_corners=nc,
        n_bad_edges=ne,
        corner_orbit_split=slice(None),
        corner_orbit_parity=slice(None),
        corner_arm=slice(None),
        edge_arm=slice(None),
        n_pairs=slice(None),
        move_count=slice(0, 7),
    )
    print_stages(f"{nc}c", combined)

    for case_name in ALL_CORNER_CASES[nc]:
        selection = select(corner_case=case_name, n_bad_edges = ne, corner_arm=None, max_move_count=6)
        print_stages(case_name, selection)


    # table = stats.trigger_counts("4c4e", selection)
    # for row in table:
    #     print("\t".join((str(s) for s in row)))


    # rows = [("corner_case", ["4a","4b","4c","4d"]), ("n_bad_edges", [ne]), ("corner_arm", [0,1,2,3,4])]
    # columns = ("edge_arm", [0,1,2,3,4])
    # stats.print(rows, columns, stats.p_sub(7))
    #stats.print([("corner_case", ["3c"]), ("n_bad_edges", [4]), ("n_fake_pairs", [0,1,2])], ("n_pairs", [0,1,2]), stats.p_sub(7, None))
