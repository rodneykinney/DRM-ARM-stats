import sys
import numpy as np
import re
import itertools
import subprocess
import random

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
    counts = np.zeros((8, 7), dtype=[("n", int), ("solutions", object)])
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
        "drm_corners",
        "drm_edges",
        "moves_to_ar",
        "is_dr_plus_trigger",
        "moves_to_4c4e",
        "moves_to_3c2e",
        "u_moves_to_ar",
    ]
)

def solutions(counts, selection: Selection, max: int = None):
    l = list(
        itertools.chain.from_iterable(t for t in counts[selection]["solutions"].ravel() if t != 0))
    if max is not None:
        random.shuffle(l)
        l = l[:max]
    return l

def read_counts(nc, ne):
    filename = f"{nc}c{ne}e.csv" if nc or ne else "full_data.csv"
    counts = np.zeros((
        9, # move_count
        9, # drm_corners
        9, # drm_edges
        4, # moves_to_ar
        2, # is_dr_plus_trigger
        4, # moves_to_4c4e
        4, # moves_to_3c2e
        4, # u_moves_to_ar
    ), dtype=[("n", int), ("solutions", object)])
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            index, co, eo, move_count, sol = stats.unpack(line)
            ar_move_count = moves_to_ar(sol)
            drm_c, drm_e = get_drm(co, eo)
            drm = f"{drm_c}c{drm_e}e"
            trigger_setup = trigger(sol, drm)
            is_dr_plus_trigger = 1 if trigger_setup.is_dr_plus_trigger else 0
            u_count = sum(1 for s in sol.split(" ") if s in {"U", "U'", "D", "D'"})
            selection = Selection(move_count=min(move_count,8),moves_to_ar=min(ar_move_count, 3),drm_corners=drm_c,drm_edges=drm_e,is_dr_plus_trigger=is_dr_plus_trigger,moves_to_4c4e=min(trigger_setup.moves_to_4c4e(3),3), moves_to_3c2e=min(trigger_setup.moves_to_3c2e(3),3), u_moves_to_ar=min(u_count,3))
            counts[selection]["n"] += 1
            sols = counts[selection]["solutions"]
            if sols == 0:
                sols = []
                counts[selection]["solutions"] = sols
            sols.append(sol)

    return counts

def select(
        max_move_count=slice(None),
        drm_corners=slice(None),
        drm_edges=slice(None),
        moves_to_ar=slice(None),
        is_dr_plus_trigger=slice(None),
        moves_to_4c4e=slice(None),
        moves_to_3c2e=slice(None),
        u_moves_to_ar=slice(None)
):
    return Selection(
        move_count=slice(0, max_move_count + 1),
        drm_corners=drm_corners,
        drm_edges=drm_edges,
        moves_to_ar=moves_to_ar,
        is_dr_plus_trigger=is_dr_plus_trigger,
        moves_to_4c4e=moves_to_4c4e,
        moves_to_3c2e=moves_to_3c2e,
        u_moves_to_ar=u_moves_to_ar,
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

def findability_families(**kwargs):
    s_args = {}
    s_args.update(kwargs)
    families = []
    s_args.update({"is_dr_plus_trigger": 1})
    families.append(("DR + Trigger", select(**s_args)))
    s_args.update({"is_dr_plus_trigger":0,"moves_to_ar":slice(0,3)})
    families.append(("AR in 0-2", select(**s_args)))
    s_args.update({"moves_to_ar":slice(3,None), "moves_to_4c4e": slice(0,2)})
    families.append(("4c4e in 1", select(**s_args)))
    s_args.update({"moves_to_4c4e":slice(2,None), "moves_to_3c2e": slice(0,2)})
    families.append(("3c2e in 1", select(**s_args)))
    s_args.update({"moves_to_3c2e": slice(2,None), "u_moves_to_ar": slice(0, 2)})
    families.append(("DR-RL U DR-RL", select(**s_args)))
    s_args.update({"u_moves_to_ar": slice(2, None)})
    families.append(("(DR-RL U)x2+ DR-RL", select(**s_args)))
    return families



def print_findability(counts_data, name, **kwargs):
    print(name)
    print("Family\tFrequency\tGenerators")
    total = np.sum(counts_data[select(**kwargs)]["n"])
    families = findability_families(**kwargs)
    family_counts = [(name, np.sum(counts_data[sel]["n"]), solutions(counts_data, sel, 5)) for
                     name, sel in families]
    family_counts.sort(key=lambda t: -t[1])
    for name, c, sols in family_counts:
        print("{}\t{}\t{}".format(name, c / total if total else "-", "\t".join(sols)))

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

    n_moves = int(sys.argv[1])-1 if len(sys.argv) > 1 else 6
    counts = read_counts(0,0)
    for nc in [0,2,3,4,5,6,7,8]:
        print_findability(counts, f"{nc}c", max_move_count=n_moves, drm_corners=nc)
        print("")
        for ne in [0,2,4,6,8]:
            print_findability(counts, f"{nc}c{ne}e", max_move_count=n_moves, drm_corners=nc,drm_edges=ne)
            print("")
