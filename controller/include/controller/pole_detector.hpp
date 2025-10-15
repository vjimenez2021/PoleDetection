#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/pca.h>

class PoleDetector : public rclcpp::Node {
public:
    PoleDetector();

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std_msgs::msg::Header& header);
    void analyzeClusterShape(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster, size_t cluster_id);
    bool isCylindrical(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster);
    bool fitCylinderRANSAC(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster);
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr poles_cloud_pub_;
    
    // Parámetros para filtrado por altura y clustering
    double min_height_;
    double max_height_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    
    // Parámetros para detección de formas
    double min_cylindrical_aspect_ratio_;
    double max_cylindrical_width_;
    double cylinder_distance_threshold_;
    double min_cylinder_radius_;
    double max_cylinder_radius_;
};