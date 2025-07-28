#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>
#include <vector>
#include <string>

// 定义点云类型别名
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

/**
 * @brief 从任意二次曲面方程生成点云
 * @param coeffs 10个二次曲面系数 [c0, c1, ..., c9] 对应 x², y², z², xy, yz, xz, x, y, z, 1
 * @param num_points 要生成的点的数量
 * @param box_size 采样边界框的边长（以原点为中心）
 * @param tolerance 判断点是否在曲面上的容差 (epsilon)
 * @return 指向生成的点云的智能指针
 */
PointCloudT::Ptr generateArbitraryQuadric(const std::vector<double>& coeffs, int num_points, float box_size, float tolerance) {
    if (coeffs.size() != 10) {
        ROS_ERROR("Coefficients vector must have 10 elements.");
        return nullptr;
    }

    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->reserve(num_points); // 预分配内存提高效率

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> rand_coord(-box_size / 2.0f, box_size / 2.0f);

    while (cloud->points.size() < num_points) {
        // 1. 在边界框内随机生成一个点
        float x = rand_coord(generator);
        float y = rand_coord(generator);
        float z = rand_coord(generator);

        // 2. 代入二次曲面方程计算值
        // f = c₀x² + c₁y² + c₂z² + c₃xy + c₄yz + c₅xz + c₆x + c₇y + c₈z + c₉
        double value = coeffs[0] * x * x + coeffs[1] * y * y + coeffs[2] * z * z +
                       coeffs[3] * x * y + coeffs[4] * y * z + coeffs[5] * x * z +
                       coeffs[6] * x + coeffs[7] * y + coeffs[8] * z +
                       coeffs[9];

        // 3. 判断是否接受该点
        if (std::abs(value) < tolerance) {
            cloud->points.emplace_back(x, y, z);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

// 高斯噪声和离群点函数 
// --- 辅助函数：噪声部分  ---
void addGaussianNoise(PointCloudT::Ptr cloud, double std_dev) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, std_dev);
    for (auto& point : cloud->points) {
        point.x += distribution(generator);
        point.y += distribution(generator);
        point.z += distribution(generator);
    }
}

void addOutliers(PointCloudT::Ptr cloud, int num_outliers, float bounding_box_size) {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-bounding_box_size / 2.0f, bounding_box_size / 2.0f);
    for (int i = 0; i < num_outliers; ++i) {
        PointT outlier;
        outlier.x = distribution(generator);
        outlier.y = distribution(generator);
        outlier.z = distribution(generator);
        cloud->points.push_back(outlier);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_generator_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/generated_cloud", 1, true);


    std::vector<double> coeffs;
    int num_points;
    double noise_std_dev;
    float bounding_box_size;
    float tolerance;
    int num_outliers;


    std::string coeffs_str;
    

    pnh.param<std::string>("coefficients", coeffs_str, "1.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.0");
    ROS_INFO("Read 'coefficients' as a string: [%s]", coeffs_str.c_str());

    std::stringstream ss(coeffs_str);
    double value;
    while (ss >> value) {
        coeffs.push_back(value);
    }

    // 检查解析后的向量大小是否正确
    if (coeffs.size() != 10) {
        ROS_FATAL("Failed to parse coefficients string correctly. Parsed %zu elements, expected 10.", coeffs.size());
        return -1; // 致命错误，直接退出
    }

    // 加载其他参数
    pnh.param("num_points", num_points, 2000);
    pnh.param("num_outliers", num_outliers, 200);
    pnh.param("noise_std_dev", noise_std_dev, 0.01);
    pnh.param("bounding_box_size", bounding_box_size, 3.0f);
    pnh.param("tolerance", tolerance, 0.05f);

    // --- 打印最终生效的配置 ---
    ROS_INFO("--- FINAL Point Cloud Generator Settings (Parsed) ---");
    ROS_INFO("Num points: %d, Num outliers: %d", num_points, num_outliers);
    ROS_INFO("Noise std_dev: %.3f, Tolerance: %.3f", noise_std_dev, tolerance);
    std::stringstream final_coeffs_ss;
    for(const auto& c : coeffs) final_coeffs_ss << c << " ";
    ROS_INFO("Final Coefficients in use: [ %s]", final_coeffs_ss.str().c_str());
    ROS_INFO("-----------------------------------------------------");

    // --- 生成点云 ---
    PointCloudT::Ptr cloud_base = generateArbitraryQuadric(coeffs, num_points, bounding_box_size, tolerance);
    if (!cloud_base) {
        ROS_ERROR("Failed to generate base cloud. Shutting down.");
        return -1;
    }
    
    addGaussianNoise(cloud_base, noise_std_dev);// 高斯噪声开关
    // addOutliers(cloud_base, num_outliers, bounding_box_size);  //离群点开关

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_base, cloud_msg);
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = "map";

    cloud_pub.publish(cloud_msg);
    ROS_INFO("Published generated point cloud to /generated_cloud. Node will now sleep.");

    ros::spin();
    return 0;
}
