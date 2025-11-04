#include "controller/pole_detector.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <vector>
#include <Eigen/Dense>
#include <chrono>

PoleDetector::PoleDetector() : Node("pole_detector") {
    // Height filtering parameters
    this->declare_parameter("min_height", 0.2);
    this->declare_parameter("max_height", 4.0);
    this->declare_parameter("cluster_tolerance", 5.5);
    this->declare_parameter("min_cluster_size", 10);
    this->declare_parameter("max_cluster_size", 100);
    
    // Cylindrical shape detection parameters
    this->declare_parameter("min_cylindrical_aspect_ratio", 0.0);
    this->declare_parameter("max_cylindrical_width", 0.4);
    this->declare_parameter("cylinder_distance_threshold", 0.05);
    this->declare_parameter("min_cylinder_radius", 0.05);
    this->declare_parameter("max_cylinder_radius", 0.25);

    // ----- RANSAC parameters -----
    this->declare_parameter("ransac_inlier_ratio", 0.5);
    this->declare_parameter("ransac_min_cluster_points", 15);
    this->declare_parameter("ransac_normal_distance_weight", 0.1);
    this->declare_parameter("ransac_max_iterations", 1000);

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

    // ----- Read RANSAC parameters -----
    this->get_parameter("ransac_inlier_ratio", ransac_inlier_ratio_);
    this->get_parameter("ransac_min_cluster_points", ransac_min_cluster_points_);
    this->get_parameter("ransac_normal_distance_weight", ransac_normal_distance_weight_);
    this->get_parameter("ransac_max_iterations", ransac_max_iterations_);

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

    // Publisher for cylinder centers
    cylinder_centers_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/pole/cylinders_center", 10);




    RCLCPP_INFO(this->get_logger(), "Pole Detector initialized");
}

void PoleDetector::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {

    // Initial time
    auto start_time = std::chrono::steady_clock::now();

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

    // Final time
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    RCLCPP_INFO(this->get_logger(), "Callback ejecutado en %ld ms", duration_ms);
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
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    int cylindrical_count = 0;
    size_t total_clusters = cluster_indices.size();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr poles_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    std::vector<double> pole_distances;
    std::vector<std::pair<double, double>> pole_centers; // almacenar centros detectados

    for (size_t i = 0; i < total_clusters; ++i) {
        const pcl::PointIndices& indices = cluster_indices[i];

        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& idx : indices.indices)
            cluster_cloud->points.push_back(cloud->points[idx]);

        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;

        double avg_distance = 0.0;
        for (const auto& p : cluster_cloud->points)
            avg_distance += sqrt(p.x * p.x + p.y * p.y);
        avg_distance /= cluster_cloud->size();

        bool is_pole = false;
        std::optional<std::pair<double, double>> cylinder_center;

        if (avg_distance <= 4.3) {
            if (isCylindrical(cluster_cloud)) {
                cylinder_center = fitCylinderRANSAC(cluster_cloud);
                if (cylinder_center.has_value()) {
                    is_pole = true;
                    pole_centers.push_back(cylinder_center.value());
                }
            }
        } else {
            if (isPoleLikeSimple(cluster_cloud)) {
                is_pole = true;
                // Aproximar el centro como el promedio de puntos del cluster
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*cluster_cloud, centroid);
                pole_centers.push_back({centroid[0], centroid[1]});
            }
        }

        if (is_pole) {
            cylindrical_count++;
            pole_distances.push_back(avg_distance);

            // Añadir al cloud coloreado
            for (const auto& idx : indices.indices) {
                pcl::PointXYZRGB p;
                p.x = cloud->points[idx].x;
                p.y = cloud->points[idx].y;
                p.z = cloud->points[idx].z;
                p.r = 0; p.g = 255; p.b = 0;
                poles_cloud->points.push_back(p);
            }
        }
    }

    // Publicar nube coloreada
    if (!poles_cloud->empty()) {
        poles_cloud->width = poles_cloud->points.size();
        poles_cloud->height = 1;
        poles_cloud->is_dense = true;

        sensor_msgs::msg::PointCloud2 poles_msg;
        pcl::toROSMsg(*poles_cloud, poles_msg);
        poles_msg.header = header;
        poles_cloud_pub_->publish(poles_msg);
    }

    // Publicar centros (x, y) en /pole/cylinders_center
    if (!pole_centers.empty()) {
        std_msgs::msg::String msg;
        std::stringstream ss;
        current_pole_id_ = 0;

        for (size_t i = 0; i < pole_centers.size(); ++i) {
            current_pole_id_++;
            double distance = pole_distances[i];
            ss << "pole_" << current_pole_id_
               << ": x=" << std::fixed << std::setprecision(2) << pole_centers[i].first
               << ", y=" << pole_centers[i].second
               << " | distancia=" << std::setprecision(2) << distance << " m\n";
        }

        msg.data = ss.str();
        cylinder_centers_pub_->publish(msg);
    }

    // Log tradicional
    RCLCPP_INFO(this->get_logger(), "Número de farolas detectadas: %d", cylindrical_count);
    for (size_t i = 0; i < pole_distances.size(); ++i)
        RCLCPP_INFO(this->get_logger(), "Farola %zu: %.2f metros", i + 1, pole_distances[i]);
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


std::optional<std::pair<double, double>> PoleDetector::fitCylinderRANSAC(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    if (cluster->size() < ransac_min_cluster_points_) {
        return std::nullopt;
    }

    // --- Optional: filter out low points (base or ground noise) ---
    pcl::PointCloud<pcl::PointXYZ>::Ptr upper_cluster(new pcl::PointCloud<pcl::PointXYZ>());
    for (const auto& p : cluster->points) {
        if (p.z > 0.3) {  // keep points above 30 cm
            upper_cluster->points.push_back(p);
        }
    }
    if (upper_cluster->points.size() < ransac_min_cluster_points_) {
        return std::nullopt;
    }

    // Estimate normals
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setInputCloud(upper_cluster);
    ne.setKSearch(10);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*normals);

    // Orient normals toward the sensor
    Eigen::Vector3f viewpoint(0.0f, 0.0f, 0.0f);
    for (auto& n : normals->points) {
        Eigen::Vector3f normal(n.normal_x, n.normal_y, n.normal_z);
        if (normal.dot(viewpoint) < 0) {
            n.normal_x *= -1.0f;
            n.normal_y *= -1.0f;
            n.normal_z *= -1.0f;
        }
    }

    // RANSAC cylinder segmentation
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(ransac_normal_distance_weight_);
    seg.setMaxIterations(ransac_max_iterations_);
    seg.setDistanceThreshold(cylinder_distance_threshold_);
    seg.setRadiusLimits(min_cylinder_radius_, max_cylinder_radius_);
    seg.setInputCloud(upper_cluster);
    seg.setInputNormals(normals);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients coefficients;
    seg.segment(*inliers, coefficients);

    if (inliers->indices.empty()) {
        return std::nullopt;
    }

    double inlier_ratio = static_cast<double>(inliers->indices.size()) / upper_cluster->size();
    if (inlier_ratio > ransac_inlier_ratio_ && coefficients.values.size() >= 7) {
        double cx = coefficients.values[0];
        double cy = coefficients.values[1];
        RCLCPP_INFO(this->get_logger(),
                    "Cluster OK | Inliers: %zu | Ratio: %.2f | Center (%.2f, %.2f)",
                    inliers->indices.size(), inlier_ratio, cx, cy);
        return std::make_pair(cx, cy);
    }

    return std::nullopt;
}
