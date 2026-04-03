// =============================================================================
// A* path planner ROS 2 node — simple overview
// -----------------------------------------------------------------------------
// This node listens for an occupancy grid map (like a floor plan: free vs wall).
// When a map arrives, it runs the A* algorithm from one corner to the opposite
// corner, then publishes the resulting path for other nodes (e.g. navigation)
// to use. A* finds a short path on a grid while avoiding obstacles.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

// -----------------------------------------------------------------------------
// ROS 2 node class: wires topics + holds the planning logic.
// -----------------------------------------------------------------------------
class AStarPlannerNode : public rclcpp::Node
{
public:
  AStarPlannerNode()
  : Node("astar_planner_node")
  {
    // QoS: "transient_local" + reliable = late subscribers can still get the last
    // map/path (useful for maps that don’t change every millisecond).
    const auto map_qos = rclcpp::QoS(1).reliable().transient_local();
    const auto path_qos = rclcpp::QoS(1).reliable().transient_local();

    // Subscribe: whenever /map updates, mapCallback runs and replans.
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", map_qos,
      std::bind(&AStarPlannerNode::mapCallback, this, std::placeholders::_1));

    // Publish: the planned polyline as a Path message (sequence of poses).
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/path", path_qos);
  }

private:
  // --- Grid cell in map coordinates (column x, row y). ---
  struct Cell
  {
    int x;
    int y;
  };

  // --- One item in the A* "open set" priority queue. ---
  // index: which cell in the 1D flattened map array.
  // f_cost: g (cost so far) + h (estimate to goal) — lower is explored first.
  struct QueueEntry
  {
    int index;
    int f_cost;
  };

  // --- Makes std::priority_queue a *min-heap* on f_cost (smallest on top). ---
  // Default is max-heap; this comparator flips it so we pop the best cell first.
  struct CompareQueueEntry
  {
    bool operator()(const QueueEntry & a, const QueueEntry & b) const
    {
      return a.f_cost > b.f_cost;
    }
  };

  // -----------------------------------------------------------------------------
  // mapCallback — "main reaction" when a new map arrives
  // Simple terms: read map size, pick start (top-left) and goal (bottom-right),
  // skip if invalid/blocked, run A*, then publish the path.
  // -----------------------------------------------------------------------------
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    const int width = static_cast<int>(msg->info.width);
    const int height = static_cast<int>(msg->info.height);

    if (width <= 0 || height <= 0) {
      RCLCPP_WARN(get_logger(), "Received map with invalid dimensions.");
      return;
    }

    if (static_cast<int>(msg->data.size()) != width * height) {
      RCLCPP_WARN(get_logger(), "Received map with inconsistent data size.");
      return;
    }

    // Fixed start/goal: corners of the map (you could change these to a goal topic).
    const Cell start{0, 0};
    const Cell goal{width - 1, height - 1};

    if (isBlocked(*msg, start.x, start.y)) {
      RCLCPP_WARN(get_logger(), "Start cell is blocked. No path published.");
      return;
    }

    if (isBlocked(*msg, goal.x, goal.y)) {
      RCLCPP_WARN(get_logger(), "Goal cell is blocked. No path published.");
      return;
    }

    std::vector<int> path_indices;
    if (!runAStar(*msg, start, goal, path_indices)) {
      RCLCPP_WARN(get_logger(), "No valid path found from start to goal.");
      return;
    }

    publishPath(*msg, path_indices);
    RCLCPP_INFO(get_logger(), "Published path with %zu poses.", path_indices.size());
  }

  // -----------------------------------------------------------------------------
  // runAStar — classic A* on a 4-connected grid (up/down/left/right, cost 1/step)
  // Simple terms:
  //   g_cost[i] = cheapest known steps from start to cell i.
  //   parent[i] = which cell we came from (to rebuild the path at the end).
  //   closed[i] = already fully processed (don’t expand again).
  //   open_set  = "frontier" — cells to explore, ordered by f = g + h.
  // -----------------------------------------------------------------------------
  bool runAStar(
    const nav_msgs::msg::OccupancyGrid & map, const Cell & start, const Cell & goal,
    std::vector<int> & path_indices)
  {
    const int width = static_cast<int>(map.info.width);
    const int total_cells = static_cast<int>(map.info.width * map.info.height);
    const int start_index = toIndex(start.x, start.y, width);
    const int goal_index = toIndex(goal.x, goal.y, width);

    std::priority_queue<QueueEntry, std::vector<QueueEntry>, CompareQueueEntry> open_set;
    std::vector<int> g_cost(total_cells, std::numeric_limits<int>::max());
    std::vector<int> parent(total_cells, -1);
    std::vector<bool> closed(total_cells, false);

    g_cost[start_index] = 0;
    open_set.push({start_index, heuristic(start.x, start.y, goal.x, goal.y)});

    // Neighbors: only 4 directions (no diagonals in this version).
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};

    while (!open_set.empty()) {
      const QueueEntry current = open_set.top();
      open_set.pop();

      // Same cell might be pushed multiple times with different f; skip stale entries.
      if (closed[current.index]) {
        continue;
      }

      closed[current.index] = true;

      if (current.index == goal_index) {
        reconstructPath(parent, start_index, goal_index, path_indices);
        return true;
      }

      const Cell current_cell = fromIndex(current.index, width);

      for (int i = 0; i < 4; ++i) {
        const int next_x = current_cell.x + dx[i];
        const int next_y = current_cell.y + dy[i];

        if (next_x < 0 || next_x >= static_cast<int>(map.info.width) || next_y < 0 ||
          next_y >= static_cast<int>(map.info.height))
        {
          continue;
        }

        if (isBlocked(map, next_x, next_y)) {
          continue;
        }

        const int next_index = toIndex(next_x, next_y, width);
        if (closed[next_index]) {
          continue;
        }

        const int tentative_g = g_cost[current.index] + 1;
        if (tentative_g >= g_cost[next_index]) {
          continue;
        }

        g_cost[next_index] = tentative_g;
        parent[next_index] = current.index;

        const int f_cost = tentative_g + heuristic(next_x, next_y, goal.x, goal.y);
        open_set.push({next_index, f_cost});
      }
    }

    return false;
  }

  // -----------------------------------------------------------------------------
  // reconstructPath — walk parent links from goal back to start, then reverse
  // Simple terms: like following breadcrumbs backward, then flipping the list
  // so the path goes start → goal.
  // -----------------------------------------------------------------------------
  void reconstructPath(
    const std::vector<int> & parent, int start_index, int goal_index,
    std::vector<int> & path_indices) const
  {
    path_indices.clear();

    int current = goal_index;
    while (current != -1) {
      path_indices.push_back(current);
      if (current == start_index) {
        break;
      }
      current = parent[current];
    }

    if (path_indices.empty() || path_indices.back() != start_index) {
      path_indices.clear();
      return;
    }

    std::reverse(path_indices.begin(), path_indices.end());
  }

  // -----------------------------------------------------------------------------
  // publishPath — turn grid cell indices into real-world x,y poses
  // Simple terms: map says "cell (3,5)"; this converts to meters using resolution
  // and map origin, and fills orientation as identity (no rotation).
  // -----------------------------------------------------------------------------
  void publishPath(
    const nav_msgs::msg::OccupancyGrid & map,
    const std::vector<int> & path_indices)
  {
    nav_msgs::msg::Path path_msg;
    path_msg.header = map.header;

    const double resolution = map.info.resolution;
    const double origin_x = map.info.origin.position.x;
    const double origin_y = map.info.origin.position.y;
    const int width = static_cast<int>(map.info.width);

    for (const int index : path_indices) {
      const Cell cell = fromIndex(index, width);

      geometry_msgs::msg::PoseStamped pose;
      pose.header = map.header;
      // +0.5 = center of the cell, not the corner.
      pose.pose.position.x = origin_x + (static_cast<double>(cell.x) + 0.5) * resolution;
      pose.pose.position.y = origin_y + (static_cast<double>(cell.y) + 0.5) * resolution;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.w = 1.0;

      path_msg.poses.push_back(pose);
    }

    path_pub_->publish(path_msg);
  }

  // -----------------------------------------------------------------------------
  // isBlocked — occupancy grid convention (ROS):
  //   100 = occupied, -1 = unknown → both treated as "can’t walk here"
  //   0 = free
  // -----------------------------------------------------------------------------
  bool isBlocked(const nav_msgs::msg::OccupancyGrid & map, int x, int y) const
  {
    const int8_t value = map.data[toIndex(x, y, static_cast<int>(map.info.width))];
    return value == 100 || value == -1;
  }

  // Manhattan distance heuristic — admissible for 4-neighbor grid with cost 1.
  // Simple terms: "straight-line-ish" estimate of steps left if you could only
  // move along the grid axes; never overestimates, so A* stays optimal.
  int heuristic(int x1, int y1, int x2, int y2) const
  {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
  }

  // Flatten 2D (x,y) → 1D index in row-major order (row * width + col).
  int toIndex(int x, int y, int width) const
  {
    return y * width + x;
  }

  // Inverse of toIndex: 1D index → Cell.
  Cell fromIndex(int index, int width) const
  {
    return Cell{index % width, index / width};
  }

  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
};

// -----------------------------------------------------------------------------
// main — standard ROS 2 entry: init, spin forever processing callbacks, shutdown.
// -----------------------------------------------------------------------------
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<AStarPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
