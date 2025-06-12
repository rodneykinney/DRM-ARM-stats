from collections import defaultdict

if __name__ == "__main__":
    lines = open("full_data.csv").readlines()

    cases = defaultdict(list)
    for line in lines:
        fields = line.split(",")
        co = fields[1]
        drm_c = sum(1 for c in co if c != "0")
        eo = fields[2]
        drm_e = 2*sum(1 for c in eo[4:8] if c == "0")
        cases[(drm_c, drm_e)].append(line)

    for ((drm_c, drm_e), lines) in cases.items():
        with open(f"{drm_c}c{drm_e}e.csv", "w") as f:
            f.writelines(lines)