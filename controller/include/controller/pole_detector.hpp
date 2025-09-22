#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>

class PoleDetector : public rclcpp::Node {
public:
    PoleDetector();

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
    
    // Parámetros para filtrado por altura y clustering
    double min_height_;
    double max_height_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
};