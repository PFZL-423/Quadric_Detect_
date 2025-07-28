#include "point_cloud_generator/Point_cloud_preprocessor.h"
#include <iostream>

PointCloudPreprocessor::PointCloudPreprocessor(const PreprocessingParams &params):params_(params),kdtree_(new pcl::search::KdTree<PointT>)
{
    voxel_filter_.setLeafSize(params_.leaf_size_x, params_.leaf_size_y, params_.leaf_size_z);
    if(params_.use_outlier_removal)
    {
        sor_filter_.setMeanK(params_.sor_mean_k);
        sor_filter_.setStddevMulThresh(params_.sor_stddev_mul_thresh);
    }
    // 3. 初始化法线估计器
    // 将KdTree与法线估计器关联，用于加速搜索
    normal_estimator_.setSearchMethod(kdtree_);
    // 设置用于法线估计的邻居数量
    normal_estimator_.setKSearch(params_.ne_k_search);
    // NormalEstimationOMP会自动使用所有可用的CPU核心
}

bool PointCloudPreprocessor::process(const PointCloudConstPtr &raw_cloud,
                                     PointCloudPtr &processed_cloud,
                                     NormalCloudPtr &normals_out)
{
    if (!raw_cloud || raw_cloud->empty())
    {
        std::cerr << "Error: Input raw cloud is null or empty." << std::endl;
        return false;
    }
    PointCloudPtr downsampled_cloud(new PointCloud());
    voxel_filter_.setInputCloud(raw_cloud);
    voxel_filter_.filter(*downsampled_cloud);
    if (downsampled_cloud->empty())
    {
        std::cerr << "Warning: Cloud is empty after downsampling." << std::endl;
        // 即使为空，也应该清理输出，并返回true表示流程正常结束
        processed_cloud->clear();
        normals_out->clear();
        return true;
    }
    PointCloudPtr current_cloud = downsampled_cloud;
    if(params_.use_outlier_removal)
    {
        PointCloudPtr filtered_cloud(new PointCloud());
        sor_filter_.setInputCloud(current_cloud);
        sor_filter_.filter(*filtered_cloud);
        if (filtered_cloud->empty())
        {
            std::cerr << "Warning: Cloud is empty after outlier removal." << std::endl;
            processed_cloud->clear();
            normals_out->clear();
            return true;
        }
        current_cloud = filtered_cloud;
    }
    pcl::copyPointCloud(*current_cloud, *processed_cloud);
    if (processed_cloud->empty())
    {
        // 如果经过处理后点云为空，则不需要计算法线
        normals_out->clear();
        return true;
    }
    normal_estimator_.setInputCloud(processed_cloud);
    normal_estimator_.compute(*normals_out);
    if (normals_out->empty() || normals_out->size() != processed_cloud->size())
    {
        std::cerr << "Error: Normal estimation failed or produced incorrect number of normals." << std::endl;
        return false;
    }
    std::cout << "Preprocessing finished. Raw: " << raw_cloud->size()
              << ", Processed: " << processed_cloud->size()
              << ", Normals: " << normals_out->size() << std::endl;

    return true;
}