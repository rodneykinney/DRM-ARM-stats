#pragma once
#include "algorithm.hpp"
#include "coordinate.hpp"

unsigned dr_coord(DominoCube &cube) {
  unsigned coc = co_coord(cube.co.data(), NC - 1);
  unsigned eslc = layout_coord(cube.esl.data(), NE);

  return (DominoCube::N_ESL * coc + eslc);
}

struct StateMinus {
  unsigned bad_corners = 0;
  unsigned bad_edges = 0;

  // Defaulted operator== can only be used from C++20
  // bool operator==(const StateMinus &other) const = default;
  bool operator==(const StateMinus &other) { return !(*this != other); }

  bool operator!=(const StateMinus &other) {
    return bad_corners != other.bad_corners || bad_edges != other.bad_edges;
  }
};

auto drm(const DominoCube cube) {
  constexpr unsigned esldr[NE] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};

  unsigned bad_corners{0}, bad_edges{0};

  for (Cubie c = ULF; c <= DLB; ++c) {
    if (cube.co[c] != 0) {
      bad_corners += 1;
    }
  }

  for (Cubie e = UF; e <= DL; ++e) {
    if (cube.esl[e] == 1 && esldr[e] != 1) {
      bad_edges += 1;
    }
  }
  return StateMinus{bad_corners, 2 * bad_edges};
}

auto arm(const DominoCube cube) {
  constexpr unsigned car[NC] = {1, 2, 1, 2, 2, 1, 2, 1};
  constexpr unsigned eslar[NE] = {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1};

  unsigned bad_corners{0}, bad_edges{0};

  for (Cubie c = ULF; c <= DLB; ++c) {
    if (cube.co[c] != 0 && cube.co[c] != car[c]) {
      bad_corners += 1;
    }
  }
  for (Cubie e = UF; e <= DL; ++e) {
    if (cube.esl[e] == 1 && eslar[e] != 1) {
      bad_edges += 1;
    }
  }

  return StateMinus{bad_corners, bad_edges};
}
