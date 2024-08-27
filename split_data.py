

def unpack(line):
    values = line.split(",")
    assert(len(values) == 6)
    drmc_str, drme_str, armc_str, arme_str, depth, sol = values
    sol = sol.strip(" \n,")
    return (int(drmc_str), int(drme_str)), (int(armc_str), int(arme_str)), int(depth), sol

def line_to_str(drm, arm, depth, alg):
    return f"{drm[0]}, {drm[1]}, {arm[0]}, {arm[1]}, {depth}, {alg}\n"


data32 = open("DRM-32.csv", "a", encoding="utf-8")
data44 = open("DRM-44.csv", "a", encoding="utf-8")
data42 = open("DRM-42.csv", "a", encoding="utf-8")
data78 = open("DRM-78.csv", "a", encoding="utf-8")
data88 = open("DRM-88.csv", "a", encoding="utf-8")

with open("raw_data.csv", "r", encoding="utf-8") as file:
    # Parsing the file to get drm arm depth and alg for each case
    for line in file:
        drm, arm, depth, alg = unpack(line)
        assert(depth < 11)
        l = line_to_str(drm, arm, depth, alg)

        match drm:
            case (3, 2):
                data32.write(l)
            case (4, 4):
                data44.write(l)
            case (4, 2):
                data42.write(l)
            case (7, 8):
                data78.write(l)
            case (8, 8):
                data88.write(l)

