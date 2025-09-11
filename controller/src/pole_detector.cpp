#include "controller/pole_detector.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <limits>

PoleDetector::PoleDetector() : Node("pole_detector") {
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/front_laser/points", 10,
        std::bind(&PoleDetector::cloudCallback, this, std::placeholders::_1));
}

void PoleDetector::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Convertir mensaje de ROS a nube de PCL
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    // Me quedo con los puntos entre 1.5 y 5 metros
    pcl::PassThrough<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud.makeShared());
    filter.setFilterFieldName("z");
    filter.setFilterLimits(1.5, 5.0);

    pcl::PointCloud<pcl::PointXYZ> filtered;
    filter.filter(filtered);

    // Si no hay puntos se acaba
    if (filtered.empty()) {
        RCLCPP_INFO(this->get_logger(), "No objects taller than 1.5m detected.");
        return;
    }

    // Tamaños de los objetos
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(filtered, min_pt, max_pt);

    float height = max_pt.z - min_pt.z;
    float width_x = max_pt.x - min_pt.x;
    float width_y = max_pt.y - min_pt.y;

    // Decidir si es un poste o no
    if (height > 1.5 && width_x < 0.5 && width_y < 0.5) {
        RCLCPP_INFO(this->get_logger(), "Possible pole detected! Height: %.2f m", height);
    } else {
        RCLCPP_INFO(this->get_logger(), "Object detected but not a pole (h=%.2f, w=%.2f,%.2f)",
                    height, width_x, width_y);
    }
}
