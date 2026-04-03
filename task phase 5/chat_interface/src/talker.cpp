#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class TalkerNode : public rclcpp::Node
{
public:
  TalkerNode()
  : Node("talker"), running_(true)
  {
    publisher_ = create_publisher<std_msgs::msg::String>("/chat", 10);

    input_thread_ = std::thread([this]() {
      std::string line;
      while (running_.load() && std::getline(std::cin, line)) {
        std_msgs::msg::String msg;
        msg.data = line;
        publisher_->publish(msg);
      }
      running_.store(false);
      rclcpp::shutdown();
    });
  }

  ~TalkerNode() override
  {
    running_.store(false);
    if (input_thread_.joinable()) {
      input_thread_.join();
    }
  }

private:
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  std::thread input_thread_;
  std::atomic<bool> running_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TalkerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
