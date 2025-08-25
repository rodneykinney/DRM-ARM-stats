import dataclasses
import re
from collections import Counter, namedtuple
from functools import cached_property
from typing import List, Tuple, Optional, Callable
from itertools import chain, product, takewhile, dropwhile
import random
import sys

import numpy as np

# Corner indices are reflected along E layer
#   Counter-clockwise when looking at U
#   Starting from UFL/DFL

# Orbit 1 : ULF, URB, DRF, DLB
# Orbit 2 : ULB, URF, DLF, DRB
orbit_1 = (0, 2, 5, 7)
orbit_2 = (1, 3, 4, 6)

U_layer = (0, 1, 2, 3)
D_layer = (4, 5, 6, 7)
L_layer = (0, 3, 4, 7)
R_layer = (1, 2, 5, 6)

# Orientation is number of clockwise rotation clicks
# Orientation needed to be out-of-AR
corner_arm_orientations = "21211212"

# U/D Edges reflected along E layer
#   Counter-clockwise when looking at U
#   Starting from UF/DF

# E-slice edges follow corners
#    Counter-clockwise from FL

edge_arm_orientations = "1.1.....1.1."

# For each edge position,
#   the index of the adjacent corners
#   and the orientation needed to form a top/side pair
corner_neighbours = [
    # U
    ((0, 1), (2, 1)),
    ((1, 2), (2, 1)),
    ((2, 3), (2, 1)),
    ((3, 0), (2, 1)),
    # E
    ((0, 4), (1, 2)),
    ((1, 5), (2, 1)),
    ((2, 6), (1, 2)),
    ((3, 7), (2, 1)),
    # D
    ((4, 5), (1, 2)),
    ((5, 6), (1, 2)),
    ((6, 7), (1, 2)),
    ((7, 4), (1, 2))
]

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
        "n_fake_pairs",
        "n_side_pairs",
        "trigger_type",
        "difficulty",
        "move_count",
    ]
)


@dataclasses.dataclass
class SolutionCategory:
    trigger_type: int
    difficulty: int

    def __repr__(self):
        if self.trigger_type == 0:
            return "DR + Trigger"
        if self.trigger_type == 1:
            return f"AR ({self.difficulty_str})"
        if self.trigger_type == 2:
            return f"4c4e ({self.difficulty_str})"
        if self.trigger_type == 3:
            return f"3c2e ({self.difficulty_str})"
        raise f"Unknown trigger type {self.trigger_type}"

    @property
    def difficulty_str(self):
        if self.difficulty == 0:
            return "easy"
        if self.difficulty == 1:
            return "findable"
        if self.difficulty == 2:
            return "hard"
        raise f"Unknown findability {self.difficulty}"

    @staticmethod
    def all_categories() -> List["SolutionCategory"]:
        return ([SolutionCategory(trigger_type=0,difficulty=0)] +
                [SolutionCategory(trigger_type=t, difficulty=d) for t in range(1,4) for d in range(3)])


@dataclasses.dataclass
class TriggerSetup:
    generator: str
    starting_drm: str
    trigger_drm: str
    dr_breaking: bool
    trigger_move_count: int
    off_axis_count: int

    @cached_property
    def moves(self):
        return self.generator.strip().replace("L", "R").replace("D", "U").replace("'", "").split(" ")

    @property
    def is_dr_plus_trigger(self):
        return self.trigger_drm == self.starting_drm and not self.dr_breaking

    @property
    def moves_to_4c4e(self):
        if self.trigger_drm == "4c4e" and self.starting_drm == "4c4e" and not self.dr_breaking:
            return 0
        moves_in_4c4e = sum(1 for _ in takewhile(lambda m: m != "R", self.moves[1:])) + 1
        return len(self.moves) - moves_in_4c4e

    @property
    def moves_to_3c2e(self):
        if self.trigger_drm == "3c2e":
            if self.starting_drm == "3c2e" and not self.dr_breaking:
                return 0
            moves_in_3c2e = sum(1 for _ in takewhile(lambda m: m != "R", self.moves[3:])) + 3
            return len(self.moves) - moves_in_3c2e
        return None

    @property
    def ar_setup(self) -> List[str]:
        if self.trigger_drm == self.starting_drm and not self.dr_breaking:
            return ["-"]
        u_count = sum(1 for s in self.generator.split(" ") if s in {"U", "U'", "D", "D'"})
        return "U " * u_count

    @property
    def u_qt_to_ar(self):
        setup = dropwhile(lambda m: m == "R" or m.endswith("2"), self.moves)
        return sum(1 for m in setup if m == "U")

    @property
    def r_qt_to_4c4e(self):
        setup = dropwhile(lambda m: m == "U" or m.endswith("2"), self.moves[1:])
        return sum(1 for m in setup if m == "R")

    @property
    def r_qt_to_3c2e(self):
        setup = dropwhile(lambda m: m == "U" or m.endswith("2"), self.moves[3:])
        return sum(1 for m in setup if m == "R")

    @property
    def moves_to_ar(self):
        sol = self.generator.replace("'", "")
        sol = re.sub("[UDRLFB]2", ".", sol)
        sol = sol.replace(" ", "")
        setup = dropwhile(lambda c: c in "RL.", sol)
        setup_length = sum(1 for c in setup)
        return setup_length

    @property
    def category(self) -> SolutionCategory:
        if self.is_dr_plus_trigger:
            return SolutionCategory(trigger_type=0, difficulty=0)
        if self.moves_to_3c2e is not None:
            difficulty = 0
            if self.moves_to_3c2e > 1:
                if self.r_qt_to_3c2e > 1:
                    difficulty = 2
                else:
                    difficulty = 1
            return SolutionCategory(trigger_type=3, difficulty=difficulty)
        if self.moves_to_4c4e >= self.moves_to_ar:
            difficulty = 0
            if self.moves_to_ar > 1:
                if self.u_qt_to_ar > 1:
                    difficulty = 2
                else:
                    difficulty = 1
            return SolutionCategory(trigger_type=1, difficulty=difficulty)
        else:
            difficulty = 0
            if self.moves_to_4c4e > 1:
                if self.r_qt_to_4c4e > 1:
                    difficulty = 2
                else:
                    difficulty = 1
            return SolutionCategory(trigger_type=2, difficulty=difficulty)

    @staticmethod
    def from_generator(generator: str, default_drm) -> "TriggerSetup":
        sol = generator.replace("'", "")
        sol = re.sub("D", "U", sol)
        p = "".join(reversed(sol.split(' ')))
        p = p.replace("RL2UR", "L2RUR")
        p = p.replace("LR2UL", "R2LUL")
        p = p.replace("LR2UR", "R2LUR")
        p = p.replace("RL2UL", "L2RUL")
        p = re.sub("[UDRLFB]2", ".", p)
        trigger_drm, trigger_moves = "4c4e", p[-1:]
        m = re.search(
            r"^.*(?:([RL]\.\.+[RL])|(R\.R|L\.L)|(R\.L|L\.R)|(RUR|LUL)|(RUL|LUR)|(RL|LR))$", p)
        if m:
            triggers = ["AR", "4c2e", "4c6e", "3c2e", "7c8e", "8c8e"]
            trigger_moves = next((t for t in m.groups() if t))
            trigger_drm = next((s for i, s in enumerate(triggers) if m.group(i + 1)))

        setup = re.sub(f"{trigger_moves}$", "", p)
        off_axis_count = sum(1 for m in setup if m in ("L", "R"))
        dr_breaking = (trigger_drm in (
        default_drm, "AR") and off_axis_count > 0) or off_axis_count > 1
        return TriggerSetup(
            generator=generator,
            starting_drm=default_drm,
            trigger_drm=trigger_drm,
            dr_breaking=dr_breaking,
            trigger_move_count=len(trigger_moves),
            off_axis_count=off_axis_count,
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
    arm_c = sum(1 for a, b in zip(co, corner_arm_orientations) if a == b)
    arm_e = sum(1 for a, b in zip(eo, edge_arm_orientations) if a == b)
    return arm_c, arm_e


def get_orbit_splits(co):
    orbit_1_bc = sum(int(co[i] != '0') for i in orbit_1)
    orbit_2_bc = sum(int(co[i] != '0') for i in orbit_2)
    return orbit_1_bc, orbit_2_bc


def get_UD_splits(co):
    U_bc = sum(int(co[i] != '0') for i in U_layer)
    D_bc = sum(int(co[i] != '0') for i in D_layer)
    return U_bc, D_bc


def get_LR_splits(co):
    L_bc = sum(int(co[i] != '0') for i in L_layer)
    R_bc = sum(int(co[i] != '0') for i in R_layer)
    return L_bc, R_bc


Pairs = namedtuple(
    "Pairs",
    [
        "n_pairs",
        "n_pseudo",
        "n_side",
        "n_ar"
    ]
)


def get_pair_counts(co, eo) -> Pairs:
    """Return number of top pairs, top pseudo-pairs, side pairs"""
    n_pairs = 0  # Top pairs
    n_pseudo = 0  # Top pseudo-pairs
    n_side = 0  # Side pairs
    n_ar = 0  # AR pairs
    for e in (0, 1, 2, 3, 8, 9, 10, 11):
        neighbor_ids, target_orientations = corner_neighbours[e]
        # Normal/pseudo pairs
        if eo[e] == '1':
            for id, target in zip(neighbor_ids, target_orientations):
                if int(co[id]) == target:
                    n_pairs += 1
                elif int(co[id]) != 0:
                    n_pseudo += 1
        # AR pairs
        elif e not in (0, 2, 8, 10):  # Ignore M slice
            if eo[e] == '0':
                ar = 0
                for id, target in zip(neighbor_ids, target_orientations):
                    if co[id] == '0':
                        ar += 1  # in-DR pair
                    elif co[id] == corner_arm_orientations[id]:
                        ar += 1  # out-of-AR pair
                if ar == 2:
                    n_ar += 1
    for e in (4, 5, 6, 7):
        if eo[e] == '0':
            neighbor_ids, target_orientations = corner_neighbours[e]
            ar = 0
            for id, target in zip(neighbor_ids, target_orientations):
                if int(co[id]) == target:
                    n_side += 1
                    ar += 1
                elif co[id] == corner_arm_orientations[id]:
                    ar += 1  # out-of-AR pair
            if ar == 2:
                n_ar += 1
    return Pairs(n_pairs=n_pairs, n_pseudo=n_pseudo, n_side=n_side, n_ar=n_ar)


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
           corner_arm: Optional[int] = slice(None),
           edge_arm: Optional[int] = slice(None),
           n_pairs: Optional[int] = slice(None),
           n_fake_pairs: Optional[int] = slice(None),
           n_side_pairs: Optional[int] = slice(None),
           n_ar_pairs: Optional[int] = slice(None),
           max_move_count: Optional[int] = None,
           category: Optional[SolutionCategory] = None,
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
    else:
        if orbit_case == "a":
            corner_orbit_split = 2
            corner_orbit_parity = 0
        elif orbit_case == "b":
            corner_orbit_split = 2
            corner_orbit_parity = 1
        elif orbit_case == "c" and m.group(3):
            corner_orbit_split = 1
        elif orbit_case == "d":
            corner_orbit_split = 0
    if corner_arm == slice(None):
        if m.group(3):
            corner_arm = int(m.group(3))
            if n_bad_corners - corner_arm != corner_arm:
                corner_arm = [corner_arm, n_bad_corners - corner_arm]
    if max_move_count is None:
        move_count = slice(None)
    elif type(max_move_count) is int:
        move_count = slice(0, max_move_count + 1)
    else:
        move_count = max_move_count
    return Selection(
        n_bad_corners,
        corner_orbit_split,
        corner_orbit_parity,
        corner_arm,
        edge_arm,
        n_bad_edges,
        n_pairs,
        n_fake_pairs,
        n_side_pairs,
        category.trigger_type if category else slice(None),
        category.difficulty if category else slice(None),
        move_count,
    )


def selection(**kwargs) -> Selection:
    features = dict((n, slice(None)) for n in Selection._fields)
    features.update(kwargs)
    return Selection(**features)


class Stats:
    def __init__(self, n_bad_corners, n_bad_edges):
        self.counts = np.zeros((
            n_bad_corners + 1,  # n_bad_corners
            5,  # corner_orbit_split
            2,  # corner_orbit_parity
            min(5, n_bad_corners + 1),  # corner_arm
            5,  # edge_arm
            n_bad_edges + 1,  # n_bad_edges
            n_bad_edges + 1,  # n_pairs,
            min(5, n_bad_edges + 1),  # n_fake_pairs,
            min(5, n_bad_edges + 1),  # n_side_pairs,
            4,  # trigger_type
            3,  # difficulty
            11,  # max_move_count
        ),
            dtype=[("n", int), ("solutions", object)]
        )

    def pattern_counts(self, selection):
        patterns = [pattern(sol) for sol in self.solutions(selection)]
        counts = Counter(patterns)
        total = sum(n for _, n in counts.items())
        return sorted([(p, n / total) for p, n in counts.items()], key=lambda x: -x[1])

    def trigger_counts(self, drm, selection):
        triggers = [(ts.trigger_drm, ts.dr_breaking) for s in self.solutions(selection) for
                    ts in [TriggerSetup.from_generator(s, drm)]]
        counts = dict(Counter(triggers).items())
        total = sum(n for _, n in counts.items())
        header = ["trigger", "DR-preserving", "example", "DR-breaking", "example"]
        table = []
        for trig in set((t for t, _ in triggers)):
            l = [s for s in self.solutions(selection) for ts in
                 [TriggerSetup.from_generator(s, drm)] if
                 (ts.trigger_drm, ts.dr_breaking) == (trig, False)]
            example = f'"{random.choice(l)}"' if l else ""
            l = [s for s in self.solutions(selection) for ts in
                 [TriggerSetup.from_generator(s, drm)] if
                 (ts.trigger_drm, ts.dr_breaking) == (trig, True)]
            example_drb = f'"{random.choice(l)}"' if l else ""
            table.append((trig, counts.get((trig, False), 0) / total, example,
                          counts.get((trig, True), 0) / total, example_drb))
        table.sort(key=lambda x: -x[1])
        return [header] + table

    def solutions_with_pattern(self, selection, p):
        return [s for s in self.solutions(selection) if pattern(s) == p]

    def p_sub(self, n: int):
        def f(selection: Selection):
            total = np.sum(self.counts[selection]["n"])
            if total == 0:
                return "-"
            selection = selection[:-1] + (slice(0, n),)
            return np.sum(self.counts[selection]["n"]) / total

        return f

    def solutions(self, selection: Selection, max: int = None):
        l = list(
            chain.from_iterable(t for t in self.counts[selection]["solutions"].ravel() if t != 0))
        if max is not None:
            random.shuffle(l)
            l = l[:max]
        return l

    def n_cases(self, selection: Selection):
        return np.sum(self.counts[selection]["n"])

    def print(self, rows: List[Tuple[str, List]], columns: Tuple[str, List[int]],
              quantity: Callable, select_row_col: Callable = lambda r, c: select(**(r | c))):
        row_names = [r[0] for r in rows]
        header = "{}\t{}".format('\t'.join(row_names), columns[0])
        print(header)
        header = "{}\t{}".format('\t'.join(' ' * len(rows)), '\t'.join(str(v) for v in columns[1]))
        print(header)
        for row_values in product(*[v[1] for v in rows]):
            print("\t".join(str(v) for v in row_values), end="\t")
            row_features = dict(zip(row_names, row_values))
            col_features = [{columns[0]: v} for v in columns[1]]
            selections = [select_row_col(row_features, col_f) for col_f in col_features]
            quantities = [str(quantity(s)) for s in selections]
            print("\t".join(quantities))

    @staticmethod
    def parse_line(line) -> Tuple[Selection, str]:
        index, co, eo, move_count, sol = unpack(line)
        n_bad_corners, n_bad_edges = get_drm(co, eo)
        a, b = get_orbit_splits(co)
        corner_orbit_split = min(a, b, 4 - a, 4 - b)
        corner_orbit_parity = 0
        if n_bad_corners == 4 and corner_orbit_split == 2:
            if min(*(get_LR_splits(co) + get_UD_splits(co))) == 1:
                corner_orbit_parity = 1
        arm_c, arm_e = get_arm(co, eo)
        n_pairs, n_fake_pairs, n_side_pairs, n_ar_pairs = get_pair_counts(co, eo)
        drm = f"{n_bad_corners}c{n_bad_edges}e"

        trigger_setup = TriggerSetup.from_generator(sol, drm)
        category = trigger_setup.category
        bucket = Selection(n_bad_corners=n_bad_corners,
                           corner_orbit_split=corner_orbit_split,
                           corner_orbit_parity=corner_orbit_parity,
                           corner_arm=min(4, arm_c),
                           edge_arm=arm_e,
                           n_bad_edges=n_bad_edges,
                           n_pairs=n_pairs,
                           n_fake_pairs=min(4, n_fake_pairs),
                           n_side_pairs=min(4, n_side_pairs),
                           trigger_type=category.trigger_type,
                           difficulty=category.difficulty,
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


def corner_case_name(bucket):
    case = ""
    if bucket.n_bad_corners == 4:
        if bucket.corner_orbit_split == 2:
            if bucket.corner_orbit_parity == 0:
                case = "4a"
            else:
                case = "4b"
        elif bucket.corner_orbit_split == 1:
            case = "4c"
        elif bucket.corner_orbit_split == 0:
            case = "4d"
    elif bucket.n_bad_corners in [2, 3, 5, 6]:
        if bucket.corner_orbit_split == 1:
            case = f"{bucket.n_bad_corners}a"
        else:
            case = f"{bucket.n_bad_corners}b"
    else:
        case = f"{bucket.n_bad_corners}c"
    arm = bucket.corner_arm
    if case != "4a":
        arm = min(arm,4-arm)
    return f"{case}-{arm},"


ALL_CORNER_CASES = [["0c-0"], [], ["2c-0", "2c-1"], ["3c-0", "3c-1"],
                    ["4a-0", "4a-2", "4b-0", "4b-2", "4c-1", "4d-2"],
                    ["5a-1", "5a-2", "5b-0", "5b-2"],
                    ["6a-0", "6a-2", "6a-3", "6b-1", "6b-2", "6b-3"], ["7c-1", "7c-2", "7c-3"],
                    ["8c-0", "8c-2", "8c-3", "8c-4"]]
REDUCED_CORNER_CASES = [["0c"], [], ["2c"], ["3c"],
                        ["4a-0", "4a-2", "4b-0", "4b-2", "4c-1", "4d-2"], ["5c"], ["6c"], ["7c"],
                        ["8c"]]


def columns(field, values, **kwargs):
    def f(case: str, edges: int):
        return [select(case, edges, **(kwargs | {field: v})) for v in values]

    return f


def print_psubn(nc, ne, nmoves):
    stats = Stats.load(nc, ne, f"{nc}c{ne}e.csv")

    rows = [("corner_case", [f"{nc}c"] + ALL_CORNER_CASES[nc]), ("n_bad_edges", [ne])]
    columns = ("n_pairs", [0, 1, 2, 3, 4])
    stats.print(rows, columns, stats.p_sub(nmoves))


def print_ncases(nc, ne):
    stats = Stats.load(nc, ne, f"{nc}c{ne}e.csv")

    def select_row_col(rows, col):
        return selection(**(rows | col))

    rows = [("n_bad_corners", [nc]), ("n_bad_edges", [ne]), ("n_pairs", list(range(0, ne + 1)))]
    columns = ("move_count", [slice(0, 7), 7, slice(8, None)])
    stats.print(rows, columns, stats.n_cases, select_row_col)


def findability_families(**kwargs):
    families = []
    for c in SolutionCategory.all_categories():
        if "difficulty" in kwargs and c.difficulty not in list(range(*kwargs["difficulty"].indices(100))):
            continue
        args = {} | kwargs | {"trigger_type": c.trigger_type, "difficulty": c.difficulty}
        families.append((str(c), selection(**args)))
    return families


def print_findability(stats, drm, **kwargs):
    total = np.sum(stats.counts[selection(**kwargs)]["n"])
    families = findability_families(**kwargs)
    family_counts = [(name, np.sum(stats.counts[sel]["n"]), stats.solutions(sel, 5)) for
                     name, sel in families]
    family_counts.sort(key=lambda t: -t[1])
    for family_name, count, sols in family_counts:
        if count == 0:
            continue
        print("{}\t{}\t{}\t{}".format(drm, family_name, count / total if total else "-",
                                      "\t".join(sols)))


def print_mutual_info(max_move_count = 6):
    stats = Stats.load(8, 8)
    columns = []
    select_target = [
        {"move_count": slice(0, max_move_count + 1), "difficulty": slice(0, 2)},
    ]
    total_target = sum(np.sum(stats.counts[selection(**s)]["n"]) for s in select_target)
    for nc in [0, 2, 3, 4, 5, 6, 7, 8]:
        for ne in [0, 2, 4, 6, 8]:
            drm = f"{nc}c{ne}e"
            drm_args = {"n_bad_corners": nc, "n_bad_edges": ne}
            total_count = np.sum(stats.counts[selection(**drm_args)]["n"])
            target_count = 0
            for s in select_target:
                select = s | drm_args
                target_count += np.sum(stats.counts[selection(**select)]["n"])
            non_target_count = total_count-target_count
            findability = float(target_count) / (target_count + non_target_count)
            frequency = target_count / total_target
            mi = findability * frequency
            columns.append((drm, mi, findability, frequency, target_count, non_target_count))
    columns.sort(key=lambda r: -r[1])
    print(f"DRM\tMutual info\tp(sub-{max_move_count+1} | drm)\tp(drm | sub-{max_move_count+1})\tsub-{max_move_count+1}\t{max_move_count+1}+")
    for col in columns:
        print("\t".join(str(t) for t in col))

def print_all_findability(nmoves: int = 6):
    print("DRM\tSetup\tFrequency\tGenerators")
    stats = Stats.load(8, 8)
    for nc in [0,2,3,4,5,6,7,8]:
        for ne in [0,2,4,6,8]:
            print_findability(stats, f"{nc}c{ne}e", move_count=slice(0,nmoves+1), n_bad_corners=nc, n_bad_edges=ne)
            print("")


def print_special_findability():
    stats = Stats.load(8,8)
    subsets = {
        (4,4): {"n_pairs": slice(2, None)},
        (3,2): {},
        (7,8): {"n_pairs": slice(5, None)},
        (4,2): {"n_pairs": slice(1, None), "n_side_pairs": slice(1, None)},
        (5,4): {"n_pairs": slice(2, None), "n_side_pairs": slice(2, None)},
        (5,6): {"n_pairs": slice(3, None)},
        (4,6): {"n_pairs": slice(3, None)},
        (6,6): {"n_pairs": slice(3, None)},
        (3,4): {"n_pairs": slice(2, None)},
        (2,2): {"n_pairs": slice(1, None)},
    }
    for (nc,ne),args  in subsets.items():
        print_findability(stats,f"{nc}c{ne}e",n_bad_corners=nc,n_bad_edges=ne, move_count=slice(0,7), difficulty=slice(0,2), **args)
        print("")

def print_4c4e_findability():
    stats = Stats.load(4,4, "4c4e.csv")
    subsets = {
        "4a-0": {"n_pairs": slice(2, None)},
        "4a-2": {"n_pairs": slice(1, None)},
        "4a-4": {"edge_arm": 2},
        "4b-0": {"n_pairs": slice(3, None)},
        "4b-2": {"n_pairs": slice(2, None), "edge_arm": 1},
        "4c-1": {"n_pairs": slice(2, None)},
        "4d-2": {"n_pairs": slice(1, None), "edge_arm": 1},
    }
    for case,args  in subsets.items():
        selection = select(corner_case=case, n_bad_edges=4)
        print_findability(stats,case,
                          n_bad_corners=4,n_bad_edges=4,
                          corner_orbit_split=selection.corner_orbit_split, corner_orbit_parity=selection.corner_orbit_parity,corner_arm=selection.corner_arm,
                          move_count=slice(0,7), difficulty=slice(0,2), **args)
        print("")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stats.py [mutual-info|findability-all|findability-top10|findability|4c4e")
        sys.exit(1)
    if sys.argv[1] == "mutual-info":
        print_mutual_info(6)
    elif sys.argv[1] == "findability-all":
        print_all_findability()
    elif sys.argv[1] == "findability-top10":
        print_special_findability()
    elif sys.argv[1] == "findability-4c4e":
        print_4c4e_findability()
    else:
        print(f"Unknown command '{sys.argv[1]}'")
        sys.exit(1)

