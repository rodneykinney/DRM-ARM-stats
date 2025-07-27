import sys
import numpy as np
import re
import itertools
import subprocess

import stats
from stats import trigger, get_drm
from collections import namedtuple


def moves_to_ar(generator):
    sol = generator.replace("'", "")
    sol = re.sub("[UDRLFB]2", ".", sol)
    sol = sol.replace(" ", "")
    setup = itertools.dropwhile(lambda c: c in "RL.", sol)
    setup_length = sum(1 for c in setup)
    return setup_length


def optimal_ar(generator):
    output = subprocess.check_output(
        ["/Users/rodneykinney/workspace/cubelib/cli/target/release/cubelib-cli",
         "solve",
         "--format", "compact",
         "--steps", "EO[fb;niss=never] > AR[arud-eofb;niss=never]",
         generator]
    )
    return int(re.split(r"[()]", output.decode("utf-8"))[1])


def print_actual_minus_optimal(nc, ne, max_move_count):
    dr_stats = stats.Stats.load(nc, ne, f"{nc}c{ne}e.csv")
    counts = np.zeros((8, 7), dtype=[("n", int)])
    for n_moves in range(1, max_move_count + 1):
        selection = stats.Selection(
            nc,
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            ne,
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            n_moves
        )
        solutions = dr_stats.solutions(selection)
        for sol in solutions:
            actual = moves_to_ar(sol)
            optimal = optimal_ar(sol)
            counts[n_moves, actual - optimal]["n"] += 1

    print("N Moves\t{}".format("\t".join(str(i) for i in range(8))))
    for nm in range(1, max_move_count + 1):
        c = [np.sum(counts[(nm, d)]["n"]) for d in range(0, nm)]
        print("{}\t{}".format(nm, "\t".join(str(i) for i in c)))


Selection = namedtuple(
    "Selection",
    [
        "move_count",
        "moves_to_ar",
        "drm_corners",
        "is_dr_plus_trigger",
        "u_qt_count",
    ]
)

def read_counts(nc, ne):
    filename = f"{nc}c{ne}e.csv" if nc or ne else "full_data.csv"
    counts = np.zeros((
        12, # move_count
        11, # moves_to_ar
        9,  # drm_corners
        2,  # is_dr_plus_trigger
        7,  # U moves
    ), dtype=[("n", int)])
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            index, co, eo, move_count, sol = stats.unpack(line)
            ar_move_count = moves_to_ar(sol)
            drm_c, drm_e = get_drm(co, eo)
            drm = f"{drm_c}c{drm_e}e"
            trigger_drm, dr_breaking, _, _ = trigger(sol, drm)
            is_dr_plus_trigger = 1 if trigger_drm == drm and not dr_breaking else 0
            u_count = sum(1 for s in sol.split(" ") if s in {"U", "U'", "D", "D'"})
            counts[move_count, ar_move_count, drm_c, is_dr_plus_trigger, u_count]["n"] += 1
    return counts

def select(
        max_move_count,
        drm_corners=slice(None),
        moves_to_ar=slice(None),
        is_dr_plus_trigger=slice(None),
        u_qt_count=slice(None)
):
    return Selection(
        move_count=slice(0, max_move_count + 1),
        drm_corners=drm_corners,
        moves_to_ar=moves_to_ar,
        is_dr_plus_trigger=is_dr_plus_trigger,
        u_qt_count=u_qt_count,
    )






def print_total_vs_ar(nc, ne, max_move_count):
    counts = read_counts(nc, ne)

    print("DRM\tMoves to AR\t{}".format("\t".join(f"+{i}" for i in range(1, max_move_count + 1))))
    for n_ar_m in range(0, max_move_count + 1):
        c = [
            np.sum(counts[(n_ar_m + delta, n_ar_m)]["n"]) if n_ar_m + delta <= max_move_count else 0
            for delta in range(1, max_move_count + 1)]
        print("{}\t{}\t{}".format(f"{nc}c{ne}e", n_ar_m, "\t".join(str(i) for i in c)))


def print_psub(nc, ne, max_move_count):
    counts = read_counts(nc, ne)
    print("DRM\tMoves to AR\tp(sub-7)\tp(sub-8)")
    for n_ar_m in range(0, max_move_count + 1):
        total = np.sum(counts[slice(None), n_ar_m]["n"])
        if not total:
            continue
        sub7 = np.sum(counts[select(max_move_count=6, moves_to_ar=n_ar_m)]["n"])
        sub8 = np.sum(
            counts[select(max_move_count=7, moves_to_ar=n_ar_m)]["n"])
        print(f"{nc}c{ne}e\t{n_ar_m}\t{sub7 / total}\t{sub8 / total}")


def print_findability(max_move_count):
    counts = read_counts(0, 0)
    print(f"DRM\tDR+Trigger\tAR in <=2\tAR + U\tAR + Ux2\tAR + Ux3+")
    for nc in [0, 2, 3, 4, 5, 6, 7, 8]:
        drm = f"{nc}c"
        dr_plus_trigger = np.sum(counts[select(max_move_count=max_move_count,drm_corners=nc, is_dr_plus_trigger=1)]["n"])
        ar_sub_3 = np.sum(counts[select(max_move_count=max_move_count,drm_corners=nc, is_dr_plus_trigger=0,moves_to_ar=slice(0,3))]["n"])
        u_qt_1 = np.sum(counts[select(max_move_count=max_move_count,drm_corners=nc, is_dr_plus_trigger=0, moves_to_ar=slice(3,None), u_qt_count=1)]["n"])
        u_qt_2 = np.sum(counts[select(max_move_count=max_move_count,drm_corners=nc, is_dr_plus_trigger=0, moves_to_ar=slice(3,None), u_qt_count=2)]["n"])
        u_qt_3 = np.sum(counts[select(max_move_count=max_move_count,drm_corners=nc, is_dr_plus_trigger=0, moves_to_ar=slice(3,None), u_qt_count=slice(3,None))]["n"])
        print(f"{drm}\t{dr_plus_trigger}\t{ar_sub_3}\t{u_qt_1}\t{u_qt_2}\t{u_qt_3}")

if __name__ == "__main__":
    # nc = int(sys.argv[1])
    # ne = int(sys.argv[2])
    # n_moves = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    # print_total_vs_ar(nc, ne, n_moves)
    # print_psub(0,0,10)
    # for nc,ne in [(3,2), (3,4), (4,2), (4,4), (4,6), (5,2), (5,4), (5,6), (7,4), (7,6), (7,8)]:
    #     print_psub(nc, ne, 10)
    # for ne in [2, 4, 6, 8]:
    #     print_psub(6, ne, 10)

    print_findability(int(sys.argv[1]))
