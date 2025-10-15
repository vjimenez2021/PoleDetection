#include "controller/pole_detector.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <vector>
#include <Eigen/Dense>

PoleDetector::PoleDetector() : Node("pole_detector") {
    // Parámetros para filtrado por altura
    this->declare_parameter("min_height", 0.3);       // Altura mínima (metros)
    this->declare_parameter("max_height", 5.0);       // Altura máxima (metros)
    this->declare_parameter("cluster_tolerance", 0.2); // Distancia para clustering (metros)
    this->declare_parameter("min_cluster_size", 5);   // Mínimo puntos por cluster
    this->declare_parameter("max_cluster_size", 1000); // Máximo puntos por cluster
    
    // Parámetros para detección de formas cilíndricas
    this->declare_parameter("min_cylindrical_aspect_ratio", 3.0); // Relación altura/ancho mínima
    this->declare_parameter("max_cylindrical_width", 0.4);       // Ancho máximo para cilindro (metros)
    this->declare_parameter("cylinder_distance_threshold", 0.05); // Umbral para RANSAC
    this->declare_parameter("min_cylinder_radius", 0.05);        // Radio mínimo (metros)
    this->declare_parameter("max_cylinder_radius", 0.3);         // Radio máximo (metros)
    
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

    // Suscriptor a la nube de puntos del LIDAR
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/front_laser/points", 10,
        std::bind(&PoleDetector::cloudCallback, this, std::placeholders::_1));
    
    // Publicador para la nube de puntos filtrada (normal)
    filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/filtered_cloud", 10);

    // Nuevo publicador para los postes detectados (con color)
    poles_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/detected_poles", 10);

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
    performClustering(filtered_cloud, msg->header);

    // Publicar la nube filtrada normal
    sensor_msgs::msg::PointCloud2 filtered_msg;
    pcl::toROSMsg(*filtered_cloud, filtered_msg);
    filtered_msg.header = msg->header;
    filtered_cloud_pub_->publish(filtered_msg);
}

void PoleDetector::performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std_msgs::msg::Header& header) {
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

    int cylindrical_count = 0;
    int total_clusters = cluster_indices.size();

    // Crear una nube de puntos RGB para los postes detectados (color verde)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr poles_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    // Analizar cada cluster
    for (size_t i = 0; i < total_clusters; ++i) {
        const pcl::PointIndices& indices = cluster_indices[i];
        
        // Crear nube de puntos del cluster
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& idx : indices.indices) {
            cluster_cloud->points.push_back(cloud->points[idx]);
        }
        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;

        RCLCPP_INFO(this->get_logger(), "--- Cluster %zu: %zu puntos ---", i, cluster_cloud->size());
        
        // Analizar la forma del cluster
        analyzeClusterShape(cluster_cloud, i);

        // Verificar si es cilíndrico
        if (isCylindrical(cluster_cloud)) {
            RCLCPP_INFO(this->get_logger(), "  ✅ FORMA CILÍNDRICA DETECTADA");
            cylindrical_count++;
            
            // Confirmar con RANSAC
            if (fitCylinderRANSAC(cluster_cloud)) {
                RCLCPP_INFO(this->get_logger(), " CILINDRO CONFIRMADO POR RANSAC");
                
                // Añadir este cluster a la nube de postes con color VERDE
                for (const auto& idx : indices.indices) {
                    pcl::PointXYZRGB colored_point;
                    colored_point.x = cloud->points[idx].x;
                    colored_point.y = cloud->points[idx].y;
                    colored_point.z = cloud->points[idx].z;
                    colored_point.r = 0;    // Rojo
                    colored_point.g = 255;  // Verde (máximo)
                    colored_point.b = 0;    // Azul
                    poles_cloud->points.push_back(colored_point);
                }
            } else {
                RCLCPP_INFO(this->get_logger(), " Forma cilíndrica pero RANSAC no confirmó");
            }
        } else {
            RCLCPP_INFO(this->get_logger(), " No es forma cilíndrica");
        }
    }

    // Publicar la nube de postes detectados si hay alguno
    if (!poles_cloud->empty()) {
        poles_cloud->width = poles_cloud->points.size();
        poles_cloud->height = 1;
        poles_cloud->is_dense = true;

        sensor_msgs::msg::PointCloud2 poles_msg;
        pcl::toROSMsg(*poles_cloud, poles_msg);
        poles_msg.header = header;
        poles_cloud_pub_->publish(poles_msg);
        
        RCLCPP_INFO(this->get_logger(), "Publicados %zu puntos de postes (color VERDE)", poles_cloud->size());
    }

    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "RESUMEN: %d de %zu clusters son cilíndricos", 
                cylindrical_count, total_clusters);
}

void PoleDetector::analyzeClusterShape(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster, size_t cluster_id) {
    if (cluster->size() < 10) {
        RCLCPP_INFO(this->get_logger(), "  Cluster demasiado pequeño para análisis de forma");
        return;
    }

    // Calcular dimensiones
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    double width = max_pt.x - min_pt.x;
    double depth = max_pt.y - min_pt.y;
    double height = max_pt.z - min_pt.z;
    
    RCLCPP_INFO(this->get_logger(), "  Dimensiones: %.2f x %.2f x %.2f m (WxDxH)", width, depth, height);

    // Calcular relación de aspecto (altura/ancho)
    double max_horizontal = std::max(width, depth);
    double aspect_ratio = height / max_horizontal;
    
    RCLCPP_INFO(this->get_logger(), "  Relación altura/ancho: %.2f:1", aspect_ratio);

    // Análisis PCA para entender la forma
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cluster);
    
    Eigen::Vector3f eigenvalues = pca.getEigenValues();
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
    
    // Calcular medidas de forma
    double linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
    double planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0];
    double scattering = eigenvalues[2] / eigenvalues[0];
    
    RCLCPP_INFO(this->get_logger(), "  Linealidad: %.3f", linearity);
    RCLCPP_INFO(this->get_logger(), "  Planaridad: %.3f", planarity);
    RCLCPP_INFO(this->get_logger(), "  Dispersión: %.3f", scattering);

    // Dirección del eje principal
    Eigen::Vector3f main_axis = eigenvectors.col(0);
    RCLCPP_INFO(this->get_logger(), "  Eje principal: (%.2f, %.2f, %.2f)", 
               main_axis.x(), main_axis.y(), main_axis.z());
}

bool PoleDetector::isCylindrical(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    if (cluster->size() < 30) {
        return false; // Muy pequeño para ser un poste
    }

    // Calcular dimensiones
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    double width = max_pt.x - min_pt.x;
    double depth = max_pt.y - min_pt.y;
    double height = max_pt.z - min_pt.z;
    
    // Criterios para forma cilíndrica
    double max_horizontal = std::max(width, depth);
    double aspect_ratio = height / max_horizontal;
    
    // 1. Debe ser alto y delgado
    bool good_aspect_ratio = (aspect_ratio > min_cylindrical_aspect_ratio_);
    bool narrow_width = (max_horizontal < max_cylindrical_width_);
    
    // 2. Análisis PCA para verificar forma lineal
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cluster);
    Eigen::Vector3f eigenvalues = pca.getEigenValues();
    double linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
    
    bool good_linearity = (linearity > 0.6); // Muy lineal
    
    RCLCPP_INFO(this->get_logger(), "  Criterios cilíndricos:");
    RCLCPP_INFO(this->get_logger(), "    Aspect ratio: %.2f > %.2f: %s", 
               aspect_ratio, min_cylindrical_aspect_ratio_, good_aspect_ratio ? "✅" : "❌");
    RCLCPP_INFO(this->get_logger(), "    Ancho: %.2f < %.2f: %s", 
               max_horizontal, max_cylindrical_width_, narrow_width ? "✅" : "❌");
    RCLCPP_INFO(this->get_logger(), "    Linealidad: %.2f > 0.6: %s", 
               linearity, good_linearity ? "✅" : "❌");

    return good_aspect_ratio && narrow_width && good_linearity;
}

bool PoleDetector::fitCylinderRANSAC(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    if (cluster->size() < 30) {
        return false;
    }

    // Estimar normales
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setInputCloud(cluster);
    ne.setKSearch(20);
    
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*normals);

    // Segmentación RANSAC para cilindro
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
        RCLCPP_INFO(this->get_logger(), "  RANSAC: No se encontró cilindro");
        return false;
    }

    double inlier_ratio = static_cast<double>(inliers->indices.size()) / cluster->size();
    double radius = coefficients.values[6];
    
    RCLCPP_INFO(this->get_logger(), "  RANSAC: Inliers: %zu (%.1f%%), Radio: %.3f m", 
               inliers->indices.size(), inlier_ratio * 100.0, radius);

    return (inlier_ratio > 0.3); // Al menos 30% de inliers
}