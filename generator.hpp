#pragma once
#include <array>
#include <deque>

template <typename Node, typename Entry, long unsigned table_size,
          typename Moves>
void generator(std::array<Entry, table_size> &table,
               std::function<Entry(Node &)> &filler, const Entry &unassigned,
               const Moves &moves) {

  std::deque<Node> queue{Node()};
  unsigned current_depth = 0;

  while (queue.size() > 0) {
    auto node = queue.back();
    unsigned coord = dr_coord(node.state);
    assert(coord < table.size());
    if (table[coord] == unassigned) {
      table[coord] = filler(node);
      for (Move m : moves) {
        auto child = node.make_child(m);
        queue.push_front(child);
      }
      if (node.depth > current_depth) {
        std::cout << "Searching at depth: " << node.depth << std::endl;
        current_depth = node.depth;
      }
    }
    queue.pop_back();
  }
}