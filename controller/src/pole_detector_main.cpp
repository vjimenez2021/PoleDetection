#include "controller/pole_detector.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoleDetector>());
    rclcpp::shutdown();
    return 0;
}
