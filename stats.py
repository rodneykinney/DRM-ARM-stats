import re
import sys
from collections import defaultdict, Counter, namedtuple
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from itertools import chain
import random

import numpy as np

Bucket = namedtuple(
    "Bucket",
    [
        "n_bad_corners",
        "corner_orbit_split",
        "corner_orbit_parity",
        "corner_arm",
        "edge_arm",
        "n_bad_edges",
        "n_pairs",
        "n_ppairs",
        "n_spairs",
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
        "edge_arm",
        "n_bad_edges",
        "n_pairs",
        "n_ppairs",
        "n_spairs",
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
    drm_c, drm_e = get_drm(co, eo)
    c_arm = sum(int(co[i] == '1') for i in (1, 3, 4, 6)) + \
            sum(int(co[i] == '2') for i in (0, 2, 5, 7))
    e_arm = sum(int(eo[i] == '1') for i in (0, 2, 8, 10))
    if drm_c - c_arm < c_arm:
        c_arm = drm_c - c_arm
        e_arm = drm_e - e_arm
    elif drm_c - c_arm == c_arm and drm_e - e_arm < e_arm:
        e_arm = drm_e - e_arm
    return c_arm, e_arm


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
           edge_arm: Optional[int] = None,
           n_pairs: Optional[int] = None,
           n_ppairs: Optional[int] = None,
           n_spairs: Optional[int] = None,
           max_move_count: Optional[int] = None
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
    corner_arm = slice(None)
    if m.group(3):
        corner_arm = int(m.group(3))
    if edge_arm is None:
        edge_arm = slice(None)
    if n_pairs is None:
        n_pairs = slice(None)
    if n_ppairs is None:
        n_ppairs = slice(None)
    if n_spairs is None:
        n_spairs = slice(None)
    move_count = slice(None)
    if max_move_count is not None:
        move_count = slice(0, max_move_count + 1)
    return Selection(
        n_bad_corners,
        corner_orbit_split,
        corner_orbit_parity,
        corner_arm,
        edge_arm,
        n_bad_edges,
        n_pairs,
        n_ppairs,
        n_spairs,
        move_count
    )


def trigger(sol: str, default_drm) -> Tuple[str, bool]:
    sol = sol.replace("'", "")
    sol = re.sub("D", "U", sol)
    p = "".join(reversed(sol.split(' ')))
    p = re.sub("[UDRLFB]2", ".", p)
    drm, moves = "4c4e", p[-1:]
    m = re.search(r"^.*(?:([RL]\.\.+[RL])|(R\.R|L\.L)|(R\.L|L\.R)|(RUR|LUL)|(RUL|LUR)|(RL|LR))$", p)
    if m:
        triggers = ["AR", "4c2e", "4c6e", "3c2e", "7c8e", "8c8e"]
        moves = next((t for t in m.groups() if t))
        drm = next((s for i, s in enumerate(triggers) if m.group(i + 1)))
    setup = re.sub(f"{moves}$", "", p)
    off_axis_count = sum(1 for m in setup if m in ("L", "R"))
    dr_breaking = (drm in (default_drm, "AR") and off_axis_count > 0) or off_axis_count > 1
    return drm, dr_breaking


bad_corner_cases = 9
orbit_split_cases = 5
orbit_parity_cases = 2
corner_arm_cases = 5
edge_arm_cases = 5
bad_edge_cases = 9
max_pairs = 9
max_pseudo_pairs = 9
max_side_pairs = 9
max_move_count = 11


@dataclass
class Stats:
    counts = np.zeros((
        bad_corner_cases,
        orbit_split_cases,
        orbit_parity_cases,
        corner_arm_cases,
        edge_arm_cases,
        bad_edge_cases,
        max_pairs,
        max_pseudo_pairs,
        max_side_pairs,
        max_move_count),
        dtype = [("n", int),("solutions", object)]
    )

    def pattern_counts(self, selection):
        patterns = [pattern(sol) for sol in self.solutions(selection)]
        counts = Counter(patterns)
        total = sum(n for _, n in counts.items())
        return sorted([(p, n / total) for p, n in counts.items()], key=lambda x: -x[1])

    def trigger_counts(self, drm, selection):
        triggers = [trigger(s, drm) for s in self.solutions(selection)]
        counts = dict(Counter(triggers).items())
        total = sum(n for _, n in counts.items())
        table = []
        for setup in set((s for s, _ in triggers)):
            l = [s for s in self.solutions(selection) if trigger(s, drm) == (setup, False)]
            example = f'"{random.choice(l)}"' if l else ""
            l = [s for s in self.solutions(selection) if trigger(s, drm) == (setup, True)]
            example_drb = f'"{random.choice(l)}"' if l else ""
            table.append((setup, counts.get((setup, False), 0) / total, example,
                          counts.get((setup, True), 0) / total, example_drb))
        table.sort(key=lambda x: -x[1])
        return table

    def solutions_with_pattern(self, selection, p):
        return [s for s in self.solutions[selection] if pattern(s) == p]

    def p_sub(self, n: int):
        def f(selection: Selection):
            total = np.sum(self.counts[selection]["n"])
            if total == 0:
                return "-"
            with_max_move_count = selection[:-1] + (slice(0, n),)
            return np.sum(self.counts[with_max_move_count]["n"]) / total
        return f

    def solutions(self, selection: Selection):
        return list(chain.from_iterable(t for t in self.counts[selection]["solutions"].ravel() if t != 0))

    def n_cases(self, selection: Selection):
        return np.sum(self.counts[selection]["n"])

    def print(self, corner_cases: List[List[str]], edges: List, columns: Callable, quantity: Callable):
        header = "DRM\tCorner Variant\t0\t1\t2\t3\t4\t5\t6\t7\t8"
        print(header)
        for drm_cases in corner_cases:
            for edges in edges:
                for case in drm_cases:
                    drm = f"{drm_cases[0][:1]}c{edges}e"
                    selections = columns(case, edges)
                    quantities = [str(quantity(s)) for s in selections]
                    q = '\t'.join(quantities)
                    print(f"{drm}\t{case}\t{q}")

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
        corner_arm, edge_arm = get_arm(co, eo)
        n_pairs, n_ppairs, n_spairs = get_pair_counts(co, eo)
        key = Bucket(n_bad_corners=n_bad_corners,
                     corner_orbit_split=corner_orbit_split,
                     corner_orbit_parity=corner_orbit_parity,
                     corner_arm=corner_arm,
                     edge_arm=edge_arm,
                     n_bad_edges=n_bad_edges,
                     n_pairs=n_pairs,
                     n_ppairs=n_ppairs,
                     n_spairs=n_spairs,
                     move_count=move_count)
        return key, sol

    @staticmethod
    def load(filename="full_data.csv") -> "Stats":
        stats = Stats()
        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    bucket, sol = Stats.parse_line(line)
                    stats.counts[*bucket]["n"] += 1
                    sols = stats.counts[*bucket]["solutions"]
                    if sols == 0:
                        sols = []
                        stats.counts[*bucket]["solutions"] = sols
                    sols.append(sol)
                except:
                    print(line)
                    raise

        return stats

ALL_CORNER_CASES = [["0c-0"],["2c-0","2c-1"],["3c-0","3c-1"],["4a-0","4a-2","4b-0","4b-2","4c-1","4d-2"],["5a-1","5a-2","5b-0","5b-2"],["6a-0","6a-2","6a-3","6b-1","6b-2","6b-3"],["7c-1","7c-2","7c-3"],["8c-0","8c-2","8c-3","8c-4"]]
REDUCED_CORNER_CASES = [["0c"],["2c"],["3c"],["4a-0","4a-2","4b-0","4b-2","4c-1","4d-2"],["5c"],["6c"],["7c"],["8c"]]

def by_pair_count(case: str, edges:int):
    return [select(case, edges, n_pairs=np) for np in range(0,9)]

def columns(field, values, **kwargs):
    def f(case: str, edges: int):
        return [select(case, edges, **(kwargs | {field: v})) for v in values]
    return f

def by_pairs(**kwargs):
    def f(case: str, edges:int):
        return [select(case, edges, n_pairs=np, **kwargs) for np in range(0,9)]
    return f

def by_edge_arm(case: str, edges:int):
    return [select(case, edges, edge_arm=ne) for ne in range(0,int(edges/2)+1)]

if __name__ == "__main__":
    # bucket, sol = Stats.parse_line("141136,02110101,100010110000,6,R U' R' L U' L'")

    stats = Stats.load()
    stats.print(ALL_CORNER_CASES, [0,2,4,6,8], by_pair_count, stats.p_sub(7))
