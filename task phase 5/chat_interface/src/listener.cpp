#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class ListenerNode : public rclcpp::Node
{
public:
  ListenerNode()
  : Node("listener")
  {
    subscription_ = create_subscription<std_msgs::msg::String>(
      "/chat", 10,
      [this](const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(get_logger(), "Received: %s", msg->data.c_str());
      });
  }

private:
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ListenerNode>());
  rclcpp::shutdown();
  return 0;
}
