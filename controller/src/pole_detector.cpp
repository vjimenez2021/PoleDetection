#include "controller/pole_detector.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <vector>
#include <Eigen/Dense>

PoleDetector::PoleDetector() : Node("pole_detector") {
    // Parámetros para filtrado por altura
    this->declare_parameter("min_height", 0.2);       // Altura mínima (metros)
    this->declare_parameter("max_height", 8.0);       // Altura máxima (metros)
    this->declare_parameter("cluster_tolerance", 5.5); // Distancia para clustering (metros)
    this->declare_parameter("min_cluster_size", 1);   // Mínimo puntos por cluster
    this->declare_parameter("max_cluster_size", 1000); // Máximo puntos por cluster
    
    // Parámetros para detección de formas cilíndricas
    this->declare_parameter("min_cylindrical_aspect_ratio", 0.0); // Relación altura/ancho mínima
    this->declare_parameter("max_cylindrical_width", 0.4);       // Ancho máximo para cilindro (metros)
    this->declare_parameter("cylinder_distance_threshold", 0.05); // Umbral para RANSAC
    this->declare_parameter("min_cylinder_radius", 0.05);        // Radio mínimo (metros)
    this->declare_parameter("max_cylinder_radius", 3.3);         // Radio máximo (metros)
    
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
}

void PoleDetector::cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Convertir ROS -> PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) {
        return;
    }

    // Filtrar por altura (z) - eliminar puntos pegados al suelo
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_height_, max_height_);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pass.filter(*filtered_cloud);

    if (filtered_cloud->empty()) {
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

// Nueva función para detectar postes lejanos (sin RANSAC)
bool PoleDetector::isPoleLikeSimple(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    if (cluster->size() < 3) {  // Menos puntos requeridos para lejanos
        return false;
    }

    // Calcular dimensiones
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    double width = max_pt.x - min_pt.x;
    double depth = max_pt.y - min_pt.y;
    double height = max_pt.z - min_pt.z;
    
    // Criterios más flexibles para postes lejanos
    double max_horizontal = std::max(width, depth);
    double aspect_ratio = height / max_horizontal;
    
    // 1. Debe ser alto y delgado (criterios más relajados)
    bool good_aspect_ratio = (aspect_ratio > 1.0);  // Más bajo que para cercanos
    bool narrow_width = (max_horizontal < 1.6);     // Más ancho que para cercanos
    
    // 2. Debe tener una altura mínima razonable
    bool sufficient_height = (height > 0.4);        // Más bajo que para cercanos

    return good_aspect_ratio && narrow_width && sufficient_height;
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

    int cylindrical_count = 0;
    size_t total_clusters = cluster_indices.size();

    // Crear una nube de puntos RGB para los postes detectados (color verde)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr poles_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    // Vector para almacenar distancias
    std::vector<double> pole_distances;

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
        
        // Calcular distancia promedio del cluster
        double avg_distance = 0.0;
        for (const auto& point : cluster_cloud->points) {
            avg_distance += sqrt(point.x * point.x + point.y * point.y);
        }
        avg_distance /= cluster_cloud->size();
        
        bool is_pole = false;
        
        // Estrategia dual según distancia
        if (avg_distance <= 4.3) {
            // Para clusters cercanos: usar tu método original con RANSAC
            if (isCylindrical(cluster_cloud) && fitCylinderRANSAC(cluster_cloud)) {
                is_pole = true;
            }
        } else {
            // Para clusters lejanos (>4.3m): usar método simple sin RANSAC
            if (isPoleLikeSimple(cluster_cloud)) {
                is_pole = true;
            }
        }
        
        if (is_pole) {
            cylindrical_count++;
            pole_distances.push_back(avg_distance);
            
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
    }

    // Mostrar resultados
    RCLCPP_INFO(this->get_logger(), "Número de farolas detectadas: %d", cylindrical_count);
    for (size_t i = 0; i < pole_distances.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "Farola %zu: %.2f metros", i + 1, pole_distances[i]);
    }
}

void PoleDetector::analyzeClusterShape(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster, size_t cluster_id) {
    if (cluster->size() < 10) {
        return;
    }

    // Calcular dimensiones
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cluster, min_pt, max_pt);
    
    double width = max_pt.x - min_pt.x;
    double depth = max_pt.y - min_pt.y;
    double height = max_pt.z - min_pt.z;
    
    // Calcular relación de aspecto (altura/ancho)
    double max_horizontal = std::max(width, depth);
    double aspect_ratio = height / max_horizontal;

    // Análisis PCA para entender la forma
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cluster);
    
    Eigen::Vector3f eigenvalues = pca.getEigenValues();
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
    
    // Calcular medidas de forma
    double linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0];
    double planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0];
    double scattering = eigenvalues[2] / eigenvalues[0];
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
        return false;
    }

    double inlier_ratio = static_cast<double>(inliers->indices.size()) / cluster->size();
    
    return (inlier_ratio > 0.3); // Al menos 30% de inliers
}