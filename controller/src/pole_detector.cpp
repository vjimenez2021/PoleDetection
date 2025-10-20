#include "controller/pole_detector.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <vector>
#include <Eigen/Dense>

PoleDetector::PoleDetector() : Node("pole_detector") {
    // Height filtering parameters
    this->declare_parameter("min_height", 0.2);
    this->declare_parameter("max_height", 8.0);
    this->declare_parameter("cluster_tolerance", 5.5);
    this->declare_parameter("min_cluster_size", 1);
    this->declare_parameter("max_cluster_size", 1000);
    
    // Cylindrical shape detection parameters
    this->declare_parameter("min_cylindrical_aspect_ratio", 0.0);
    this->declare_parameter("max_cylindrical_width", 0.4);
    this->declare_parameter("cylinder_distance_threshold", 0.05);
    this->declare_parameter("min_cylinder_radius", 0.05);
    this->declare_parameter("max_cylinder_radius", 3.3);
    
    this->get_parameter("min_height", min_height_);
    this->get_parameter("max_height", max_height_);
    this->get_parameter("cluster_tolerance", cluster_tolerance_);
    this->get_parameter("min_cluster_size", min_cluster_size_);
    this->get_parameter("max_cluster_size", max_cluster_size_);
    this->get_parameter("min_cylindrical_aspect_ratio", min_cylindrical_aspect_ratio_);
    this->get_parameter("max_cylindrical_width", max_cylindrical_width_);
    this->get_parameter("cylinder_distance_threshold", cylinder_distance_threshold_);
    this->get_parameter("min_cylinder_radius", min_cylinder_radius_);
    this->get_parameter("max_cylinder_radius", max_cylinder_radius_);

    // LIDAR point cloud subscriber
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/front_laser/points", 10,
        std::bind(&PoleDetector::cloudCallback, this, std::placeholders::_1));
    
    // Filtered point cloud publisher
    filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/filtered_cloud", 10);

    // Detected poles publisher (colored)
    poles_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/detected_poles", 10);

    RCLCPP_INFO(this->get_logger(), "Pole Detector initialized");
}

void PoleDetector::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Convert ROS to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) {
        return;
    }

    // Filter by height (z) - remove ground points
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_height_, max_height_);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pass.filter(*filtered_cloud);

    if (filtered_cloud->empty()) {
        return;
    }

    // Perform clustering on filtered cloud
    performClustering(filtered_cloud, msg->header);

    // Publish filtered cloud
    sensor_msgs::msg::PointCloud2 filtered_msg;
    pcl::toROSMsg(*filtered_cloud, filtered_msg);
    filtered_msg.header = msg->header;
    filtered_cloud_pub_->publish(filtered_msg);
}

// Detect far poles without RANSAC
bool PoleDetector::isPoleLikeSimple(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    if (cluster->size() < 3) {
        return false;
    }

    // Calculate dimensions
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    double width = max_pt.x - min_pt.x;
    double depth = max_pt.y - min_pt.y;
    double height = max_pt.z - min_pt.z;
    
    // Flexible criteria for far poles
    double max_horizontal = std::max(width, depth);
    double aspect_ratio = height / max_horizontal;
    
    // Tall and thin criteria (relaxed)
    bool good_aspect_ratio = (aspect_ratio > 1.0);
    bool narrow_width = (max_horizontal < 1.6);
    bool sufficient_height = (height > 0.4);

    return good_aspect_ratio && narrow_width && sufficient_height;
}

void PoleDetector::performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std_msgs::msg::Header& header) {
    // Create KdTree for neighbor search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    // Vector for cluster indices
    std::vector<pcl::PointIndices> cluster_indices;
    
    // Configure Euclidean cluster extraction
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    int cylindrical_count = 0;
    size_t total_clusters = cluster_indices.size();

    // RGB point cloud for detected poles (green color)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr poles_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    // Vector to store distances
    std::vector<double> pole_distances;

    // Analyze each cluster
    for (size_t i = 0; i < total_clusters; ++i) {
        const pcl::PointIndices& indices = cluster_indices[i];
        
        // Create cluster point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& idx : indices.indices) {
            cluster_cloud->points.push_back(cloud->points[idx]);
        }
        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;
        
        // Calculate average cluster distance
        double avg_distance = 0.0;
        for (const auto& point : cluster_cloud->points) {
            avg_distance += sqrt(point.x * point.x + point.y * point.y);
        }
        avg_distance /= cluster_cloud->size();
        
        bool is_pole = false;
        
        // Dual strategy based on distance
        if (avg_distance <= 4.3) {
            // Near clusters: original method with RANSAC
            if (isCylindrical(cluster_cloud) && fitCylinderRANSAC(cluster_cloud)) {
                is_pole = true;
            }
        } else {
            // Far clusters (>4.3m): simple method without RANSAC
            if (isPoleLikeSimple(cluster_cloud)) {
                is_pole = true;
            }
        }
        
        if (is_pole) {
            cylindrical_count++;
            pole_distances.push_back(avg_distance);
            
            // Add cluster to poles cloud (GREEN color)
            for (const auto& idx : indices.indices) {
                pcl::PointXYZRGB colored_point;
                colored_point.x = cloud->points[idx].x;
                colored_point.y = cloud->points[idx].y;
                colored_point.z = cloud->points[idx].z;
                colored_point.r = 0;
                colored_point.g = 255;
                colored_point.b = 0;
                poles_cloud->points.push_back(colored_point);
            }
        }
    }

    // Publish detected poles if any
    if (!poles_cloud->empty()) {
        poles_cloud->width = poles_cloud->points.size();
        poles_cloud->height = 1;
        poles_cloud->is_dense = true;

        sensor_msgs::msg::PointCloud2 poles_msg;
        pcl::toROSMsg(*poles_cloud, poles_msg);
        poles_msg.header = header;
        poles_cloud_pub_->publish(poles_msg);
    }

    // Show results
    RCLCPP_INFO(this->get_logger(), "Número de farolas detectadas: %d", cylindrical_count);
    for (size_t i = 0; i < pole_distances.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "Farola %zu: %.2f metros", i + 1, pole_distances[i]);
    }
}

bool PoleDetector::isCylindrical(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    if (cluster->size() < 30) {
        return false;
    }

    // Calculate dimensions
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    double width = max_pt.x - min_pt.x;
    double depth = max_pt.y - min_pt.y;
    double height = max_pt.z - min_pt.z;
    
    // Cylindrical shape criteria
    double max_horizontal = std::max(width, depth);
    double aspect_ratio = height / max_horizontal;
    
    // Tall and thin
    bool good_aspect_ratio = (aspect_ratio > min_cylindrical_aspect_ratio_);
    bool narrow_width = (max_horizontal < max_cylindrical_width_);
    
    // PCA for linear shape verification
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cluster);
    Eigen::Vector3f eigenvalues = pca.getEigenValues();
    double linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
    
    bool good_linearity = (linearity > 0.6);

    return good_aspect_ratio && narrow_width && good_linearity;
}

bool PoleDetector::fitCylinderRANSAC(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    if (cluster->size() < 30) {
        return false;
    }

    // Estimate normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setInputCloud(cluster);
    ne.setKSearch(20);
    
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*normals);

    // RANSAC cylinder segmentation
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(cylinder_distance_threshold_);
    seg.setRadiusLimits(min_cylinder_radius_, max_cylinder_radius_);
    seg.setInputCloud(cluster);
    seg.setInputNormals(normals);
    
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients coefficients;
    seg.segment(*inliers, coefficients);

    if (inliers->indices.empty()) {
        return false;
    }

    double inlier_ratio = static_cast<double>(inliers->indices.size()) / cluster->size();
    
    return (inlier_ratio > 0.3);
}