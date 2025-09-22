#include "controller/pole_detector.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <vector>

PoleDetector::PoleDetector() : Node("pole_detector") {
    // Parámetros para filtrado por altura
    this->declare_parameter("min_height", 0.3);       // Altura mínima (metros)
    this->declare_parameter("max_height", 5.0);       // Altura máxima (metros)
    this->declare_parameter("cluster_tolerance", 0.2); // Distancia para clustering (metros)
    this->declare_parameter("min_cluster_size", 5);   // Mínimo puntos por cluster
    this->declare_parameter("max_cluster_size", 1000); // Máximo puntos por cluster
    
    this->get_parameter("min_height", min_height_);
    this->get_parameter("max_height", max_height_);
    this->get_parameter("cluster_tolerance", cluster_tolerance_);
    this->get_parameter("min_cluster_size", min_cluster_size_);
    this->get_parameter("max_cluster_size", max_cluster_size_);

    // Suscriptor a la nube de puntos del LIDAR
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/front_laser/points", 10,
        std::bind(&PoleDetector::cloudCallback, this, std::placeholders::_1));
    
    // Publicador para la nube de puntos filtrada
    filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/filtered_cloud", 10);

    RCLCPP_INFO(this->get_logger(), "Pole Detector inicializado");
    RCLCPP_INFO(this->get_logger(), "Filtrando puntos entre %.2f y %.2f metros", min_height_, max_height_);
    RCLCPP_INFO(this->get_logger(), "Clustering: tolerance=%.2fm, min_size=%d, max_size=%d", 
                cluster_tolerance_, min_cluster_size_, max_cluster_size_);
}

void PoleDetector::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Convertir ROS -> PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);

    RCLCPP_INFO(this->get_logger(), "════════════════════════════════════════");
    RCLCPP_INFO(this->get_logger(), "Nube original: %zu puntos", cloud->size());

    if (cloud->empty()) {
        RCLCPP_WARN(this->get_logger(), "¡La nube está vacía!");
        return;
    }

    // Filtrar por altura (z) - eliminar puntos pegados al suelo
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_height_, max_height_);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pass.filter(*filtered_cloud);

    RCLCPP_INFO(this->get_logger(), "Nube filtrada: %zu puntos (%.1f%%)", 
                filtered_cloud->size(),
                (filtered_cloud->size() * 100.0) / cloud->size());

    if (filtered_cloud->empty()) {
        RCLCPP_WARN(this->get_logger(), "No hay puntos después del filtro de altura");
        return;
    }

    // Realizar clustering en la nube filtrada
    performClustering(filtered_cloud);

    // Publicar la nube filtrada
    sensor_msgs::msg::PointCloud2 filtered_msg;
    pcl::toROSMsg(*filtered_cloud, filtered_msg);
    filtered_msg.header = msg->header;
    filtered_cloud_pub_->publish(filtered_msg);
}

void PoleDetector::performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Crear el árbol KdTree para la búsqueda de vecinos
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    // Vector para almacenar los índices de los clusters
    std::vector<pcl::PointIndices> cluster_indices;
    
    // Configurar el extractor de clusters euclidianos
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    RCLCPP_INFO(this->get_logger(), "Se encontraron %zu clusters", cluster_indices.size());

    // Mostrar información de cada cluster
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        const pcl::PointIndices& indices = cluster_indices[i];
        RCLCPP_INFO(this->get_logger(), "Cluster %zu: %zu puntos", i, indices.indices.size());

        // Calcular el centroide del cluster
        pcl::PointXYZ centroid;
        centroid.x = 0; centroid.y = 0; centroid.z = 0;
        
        for (const auto& idx : indices.indices) {
            centroid.x += cloud->points[idx].x;
            centroid.y += cloud->points[idx].y;
            centroid.z += cloud->points[idx].z;
        }
        
        centroid.x /= indices.indices.size();
        centroid.y /= indices.indices.size();
        centroid.z /= indices.indices.size();

        RCLCPP_INFO(this->get_logger(), "  Centroide: (%.2f, %.2f, %.2f) m", 
                   centroid.x, centroid.y, centroid.z);

        // Calcular dimensiones del cluster
        pcl::PointXYZ min_pt, max_pt;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        
        for (const auto& idx : indices.indices) {
            cluster_cloud->points.push_back(cloud->points[idx]);
        }
        
        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;

        pcl::getMinMax3D(*cluster_cloud, min_pt, max_pt);
        
        double width = max_pt.x - min_pt.x;
        double depth = max_pt.y - min_pt.y;
        double height = max_pt.z - min_pt.z;

        RCLCPP_INFO(this->get_logger(), "  Dimensiones: %.2f x %.2f x %.2f m (WxDxH)", 
                   width, depth, height);
        RCLCPP_INFO(this->get_logger(), "  Rango X: [%.2f, %.2f] m", min_pt.x, max_pt.x);
        RCLCPP_INFO(this->get_logger(), "  Rango Y: [%.2f, %.2f] m", min_pt.y, max_pt.y);
        RCLCPP_INFO(this->get_logger(), "  Rango Z: [%.2f, %.2f] m", min_pt.z, max_pt.z);
    }

    // Información adicional sobre clusters muy pequeños o muy grandes
    int small_clusters = 0;
    int large_clusters = 0;
    
    for (const auto& cluster : cluster_indices) {
        if (cluster.indices.size() < 10) {
            small_clusters++;
        }
        if (cluster.indices.size() > 30) {
            large_clusters++;
        }
    }

    if (small_clusters > 0) {
        RCLCPP_INFO(this->get_logger(), "%d clusters con menos de 10 puntos", small_clusters);
    }
    if (large_clusters > 0) {
        RCLCPP_INFO(this->get_logger(), "%d clusters con más de 30 puntos", large_clusters);
    }
}