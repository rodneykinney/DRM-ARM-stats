#pragma once
#include "algorithm.hpp"
#include "cubie_cube.hpp"

struct DominoCube {
  std::array<Orientation, NC> co{0, 0, 0, 0, 0, 0, 0, 0};
  std::array<Cubie, NE> esl{0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};

  static constexpr unsigned N_CO = ipow(3, 7);
  static constexpr unsigned N_ESL = 495;

  void set_solved() {
    co.fill(0);
    esl = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
  }

  void corner_apply(const CubieCube &cc) {
    Orientation new_co[NC];
    for (Cubie c = ULF; c <= DLB; ++c) {
      new_co[c] = (co[cc.cp[c]] + cc.co[c]);
    };
    for (Cubie c = ULF; c <= DLB; ++c) {
      co[c] = new_co[c] % 3;
    }
  };

  void edge_apply(const CubieCube &cc) {
    Orientation new_esl[NE];
    for (Cubie e = UF; e <= DL; ++e) {
      new_esl[e] = esl[cc.ep[e]];
    };
    for (Cubie e = UF; e <= DL; ++e) {
      esl[e] = new_esl[e];
    }
  };

  void apply(const CubieCube &cc) {
    edge_apply(cc);
    corner_apply(cc);
  }

  void apply(const Move &m) { apply(elementary_transformations[m]); }

  void apply(const Algorithm &alg) {
    for (const Move &m : alg.sequence) {
      apply(m);
    }
  }

  void apply_inverse(const unsigned &move) {
    apply(inverse_of_HTM_Moves_and_rotations[move]);
  }

  void apply_inverse(const Algorithm &alg) {
    for (auto move = alg.sequence.rbegin(); move != alg.sequence.rend();
         ++move) {
      apply_inverse(*move);
    }
  }

  bool is_solved() const {
    unsigned esl_check[NE] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
    for (auto o : co) {
      if (o != 0)
        return false;
    }
    for (unsigned k = 0; k < NE; ++k) {
      if (esl[k] != esl_check[k])
        return false;
    }
    return true;
  }

  void show() const {
    // Display the 4 arrays defining a cube at the cubie level
    std::cout << "Domino object:"
              << "\nCO:\n";
    for (Cubie c = ULF; c <= DLB; ++c) {
      std::cout << co[c] << " ";
    }
    std::cout << "\nE-Slice layout:\n";
    for (Cubie e = UF; e <= DL; ++e) {
      std::cout << esl[e] << " ";
    }
    std::cout << std::endl;
  };
};

struct DomiNode {
  DominoCube state = DominoCube();
  unsigned depth = 0;
  Algorithm path;

  auto make_child(const Move &m) {
    auto cube = state;
    cube.apply(m);
    auto child = DomiNode{cube, depth + 1, path};
    child.path.append(m);
    return child;
  }
};