#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>

#include "domino_cube.hpp"
#include "generator.hpp"
#include "state_count.hpp"

struct TwistData {
  StateMinus drm;
  unsigned cw_corners, ccw_corners;
  unsigned depth;
  Algorithm solution;

  bool operator==(const TwistData &other) { return !(*this != other); }

  bool operator!=(const TwistData &other) {
    return depth != other.depth || cw_corners != other.cw_corners ||
           ccw_corners != other.ccw_corners || drm != other.drm;
  }

  void show() const {
    std::cout << "DRM: " << drm.bad_corners << "c" << drm.bad_edges << "e\n";
    std::cout << "CW: " << cw_corners << " CCW: " << ccw_corners << "\n";
    std::cout << "Depth: " << depth << std::endl;
  }
};

unsigned cw(const DominoCube &cube) {
  unsigned ret = 0;
  for (Orientation o : cube.co) {
    if (o == 1) {
      ret += 1;
    }
  }
  return ret;
}

unsigned ccw(const DominoCube &cube) {
  unsigned ret = 0;
  for (Orientation o : cube.co) {
    if (o == 2) {
      ret += 1;
    }
  }
  return ret;
}

constexpr unsigned table_size = DominoCube::N_ESL * DominoCube::N_CO;
std::array<TwistData, table_size> table;

auto unassigned = TwistData{StateMinus(), 0, 0, 1000};
auto filler = std::function([](DomiNode &node) -> TwistData {
  auto sol = node.path.get_inverse();
  return TwistData{drm(node.state), cw(node.state), ccw(node.state), node.depth,
                   sol};
});

int main() {

  table.fill(unassigned);

  // This function fills the table with the TwistData struct for each EO case
  generator<DomiNode>(table, filler, unassigned, EOFB_HTM_Moves);

  DominoCube cube;
  for (auto dr_case : table) {
    // Making sure everything is alright
    cube.set_solved();
    cube.apply(dr_case.solution.get_inverse());
    assert(drm(cube).bad_corners == dr_case.drm.bad_corners);
    assert(drm(cube).bad_edges == dr_case.drm.bad_edges);
    assert(cw(cube) == dr_case.cw_corners);
    assert(ccw(cube) == dr_case.ccw_corners);
    assert(dr_case.cw_corners + dr_case.ccw_corners == dr_case.drm.bad_corners);
  }

  {
    // Write data to file
    std::ofstream solution_file("twist_raw_data.csv");
    for (auto dr_case : table) {
      assert(dr_case != unassigned);
      solution_file << dr_case.drm.bad_corners << ", " << dr_case.drm.bad_edges
                    << ", " << dr_case.cw_corners << ", " << dr_case.ccw_corners
                    << ", " << dr_case.depth << ", " << dr_case.solution
                    << std::endl;
    }
  }
}