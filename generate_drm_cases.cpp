#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

#include "domino_cube.hpp"
#include "generator.hpp"
#include "state_count.hpp"

struct CaseData {
  StateMinus drm, arm;
  unsigned depth;
  Algorithm solution;

  bool operator==(const CaseData &other) { return !(*this != other); }

  bool operator!=(const CaseData &other) {
    return depth != other.depth || arm != other.arm || drm != other.drm;
  }

  void show() const {
    std::cout << "DRM: " << drm.bad_corners << "c" << drm.bad_edges << "e\n";
    std::cout << "ARM: " << arm.bad_corners << "c" << arm.bad_edges << "e\n";
    std::cout << "Depth: " << depth << std::endl;
  }
};

constexpr unsigned table_size = DominoCube::N_ESL * DominoCube::N_CO;
std::array<CaseData, table_size> table;

auto unassigned = CaseData{StateMinus(), StateMinus(), 1000};
auto filler = std::function([](DomiNode &node) -> CaseData {
  auto sol = node.path.get_inverse();
  return CaseData{drm(node.state), arm(node.state), node.depth, sol};
});

int main() {

  table.fill(unassigned);

  // This function fills the table with the CaseData struct for each EO case
  generator<DomiNode>(table, filler, unassigned, EOFB_HTM_Moves);

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
      assert(dr_case != unassigned);
      solution_file << dr_case.drm.bad_corners << ", " << dr_case.drm.bad_edges
                    << ", " << dr_case.arm.bad_corners << ", "
                    << dr_case.arm.bad_edges << ", " << dr_case.depth << ", "
                    << dr_case.solution << std::endl;
      eo_dr_distribution[dr_case.depth] += 1;
    }
  }
}