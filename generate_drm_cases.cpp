#include <algorithm>
#include <cassert>
#include <deque>
#include <fstream>
#include <iostream>

#include "domino_cube.hpp"
#include "state_count.hpp"

struct CaseData {
  StateMinus drm, arm;
  unsigned depth;
  Algorithm solution;

  void show() const {
    std::cout << "DRM: " << drm.bad_corners << "c" << drm.bad_edges << "e\n";
    std::cout << "ARM: " << arm.bad_corners << "c" << arm.bad_edges << "e\n";
    std::cout << "Depth: " << depth << std::endl;
  }
};

constexpr unsigned N_EOFB_HTM_MOVES = 14;
std::array<Move, N_EOFB_HTM_MOVES> EOFB_HTM_Moves{U,  U2, U3, D,  D2, D3, R,
                                                  R2, R3, L,  L2, L3, F2, B2};

constexpr unsigned table_size = DominoCube::N_ESL * DominoCube::N_CO;
std::array<CaseData, table_size> table;

struct DomiNode {
  DominoCube state = DominoCube();
  unsigned depth = 0;
  Algorithm path;
};

constexpr unsigned unassigned = 1000;

int main() {

  std::deque<DomiNode> queue{DomiNode()};
  table.fill(CaseData{StateMinus(), StateMinus(), unassigned});
  unsigned current_depth = 0;

  while (queue.size() > 0) {
    // This loop fills the table with the CaseData struct for each EO case
    auto node = queue.back();
    unsigned coord = dr_coord(node.state);
    assert(coord < table_size);
    if (table[coord].depth == unassigned) {
      auto sol = node.path.get_inverse();
      table[coord] =
          CaseData{drm(node.state), arm(node.state), node.depth, sol};
      for (Move m : EOFB_HTM_Moves) {
        DominoCube cube = node.state;
        cube.apply(m);
        auto child = DomiNode{cube, node.depth + 1, node.path};
        child.path.append(m);
        queue.push_front(child);
      }
      if (node.depth > current_depth) {
        std::cout << "Searching at depth: " << node.depth << std::endl;
        current_depth = node.depth;
      }
    }
    queue.pop_back();
  }

  DominoCube cube;
  for (auto dr_case : table) {
    // Making sure everything looks normal
    cube.set_solved();
    cube.apply(dr_case.solution.get_inverse());
    assert(drm(cube).bad_corners == dr_case.drm.bad_corners);
    assert(drm(cube).bad_edges == dr_case.drm.bad_edges);
    assert(arm(cube).bad_corners == dr_case.arm.bad_corners);
    assert(arm(cube).bad_edges == dr_case.arm.bad_edges);
  }

  std::array<unsigned, 11> eo_dr_distribution;
  eo_dr_distribution.fill(0);
  {
    // Write data to file
    std::ofstream solution_file("raw_data.csv");
    for (auto dr_case : table) {
      assert(dr_case.depth != unassigned);
      solution_file << dr_case.drm.bad_corners << ", " << dr_case.drm.bad_edges
                    << ", " << dr_case.arm.bad_corners << ", "
                    << dr_case.arm.bad_edges << ", " << dr_case.depth << ", "
                    << dr_case.solution << std::endl;
      eo_dr_distribution[dr_case.depth] += 1;
    }
  }
}