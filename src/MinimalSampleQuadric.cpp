#include "point_cloud_generator/MinimalSampleQuadric.h"
#include <iostream>
#include <pcl/sample_consensus/method_types.h> // for SAC_RANSAC
#include <pcl/sample_consensus/model_types.h>  // for SACMODEL_PLANE
#include <pcl/segmentation/sac_segmentation.h> // for SACSegmentation
#include <pcl/filters/extract_indices.h>       // for ExtractIndices
#include <iomanip>
#include <algorithm> 
#include <cmath>     // 包含此头文件
#include <chrono>
using MSQ=MinimalSampleQuadric;
MSQ::MinimalSampleQuadric(const DetectorParams &params)
    // 使用成员初始化列表（Member Initializer List）来初始化成员变量
    : params_(params),                                                    // 将传入的参数直接赋值给成员变量params_
      preprocessor_(params.preprocessing), // 使用params中的特定值初始化preprocessor_对象
      final_remaining_cloud_(new PointCloud()),                           // 初始化指向点云的智能指针
      initial_point_count_(0)                                             // 初始化计数器为0
{
    // 这里可以放一些额外的逻辑，比如打印日志等。
    std::cout << "MinimalSampleQuadric detector initialized." << std::endl;
}

bool MSQ::processCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud)
{
    detected_primitives_.clear();
    if (final_remaining_cloud_)
    {
        final_remaining_cloud_->clear();
    }
    else
    {
        final_remaining_cloud_.reset(new PointCloud());
    }
    if (!input_cloud || input_cloud->empty())
    {
        std::cerr << "Error: Input cloud for processCloud is null or empty." << std::endl;
        return false;
    }
    // 记录最原始的点云数量（预处理前）
    initial_point_count_ = input_cloud->size();
    std::cout << "Starting detection process with " << initial_point_count_ << " initial points." << std::endl;

    PointCloudPtr working_cloud(new PointCloud());
    if (!preProcess(input_cloud, working_cloud) || working_cloud->empty())
    {
        std::cout << "Preprocessing resulted in an empty point cloud. Stopping." << std::endl;
        // 注意：即使预处理后为空，也应将空的 working_cloud 赋给 final_remaining_cloud_
        pcl::copyPointCloud(*working_cloud, *final_remaining_cloud_);
        return true; // 处理流程正常结束，只是没有点可检测
    }
    std::cout << "Preprocessing finished. Working cloud has " << working_cloud->size() << " points with normals." << std::endl;

    detectPlanes(working_cloud);
    detectQuadric(working_cloud);

    // pcl::copyPointCloud(*working_cloud, *final_remaining_cloud_);
    final_remaining_cloud_ = working_cloud;

    std::cout << "Detection process finished. " << detected_primitives_.size() << " primitives detected." << std::endl;
    std::cout << "Final remaining cloud has " << final_remaining_cloud_->size() << " points." << std::endl;

    return true;
}
bool MSQ::preProcess(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud, PointCloudPtr &preprocessed_cloud_out)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed_xyz(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

    if(!preprocessor_.process(input_cloud,processed_xyz,normals))
    {
        // 如果预处理失败（例如，输入点云为空），则直接返回失败
        std::cerr << "Error during preprocessing step." << std::endl;
        return false;
    }
    // 检查处理完是否为空
    if (processed_xyz->empty() || normals->empty())
    {
        std::cout << "Warning: Point cloud is empty after preprocessing." << std::endl;
        preprocessed_cloud_out->clear(); // 确保输出为空
        return true;                     
    }
    pcl::concatenateFields(*processed_xyz, *normals, *preprocessed_cloud_out);

    return true;
}
void MSQ::detectPlanes(PointCloudPtr &remain_cloud)
{
    // 如果初始点云为空或点数不足，则直接返回
    if (!remain_cloud || remain_cloud->size() < 3)
    {
        return;
    }
    std::cout << "---Starting Plane Detection---" << std::endl;

    // 创建一个临时点云指针，用于在循环中操作，避免直接修改传入的引用
    PointCloudPtr current_cloud(new PointCloud(*remain_cloud));

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);  //设置拟合类型为平面
    seg.setMethodType(pcl::SAC_RANSAC);     //方法为RANSAC

    // 使用我们在DetectorParams中定义的参数
    seg.setMaxIterations(params_.plane_max_iterations);
    seg.setDistanceThreshold(params_.plane_distance_threshold);

    pcl::PointIndices::Ptr inlier_indices(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    pcl::ExtractIndices<PointT> extract;//过滤器

    while (true)
    {
        size_t current_point_count=current_cloud->size();
        
        // 1. 终止条件检查：如果剩余点数太少，无法形成平面，则退出循环
        if (current_point_count < 3)
        {
            std::cout << "Remaining points (" << current_point_count << ") are too few to form a plane. Stopping." << std::endl;
            break;
        }
        // 2. 在当前点云上执行RANSAC分割
        seg.setInputCloud(current_cloud);
        seg.segment(*inlier_indices, *coefficients);

        // 3. 检查是否找到了内点
        if (inlier_indices->indices.empty())
        {
            std::cout << "RANSAC could not find any planar model in the remaining " << current_point_count << " points. Stopping." << std::endl;
            break;
        }
        // 4. 决策：检查找到的平面是否足够“显著”
        double inlier_percentage = static_cast<double>(inlier_indices->indices.size()) / current_point_count;
        if (inlier_percentage < params_.min_plane_inlier_percentage)
        {
            std::cout << "Found a plane with " << inlier_indices->indices.size() << " inliers ("
                      << "which is below the threshold of " << params_.min_plane_inlier_percentage * 100
                      << "%. Stopping plane detection." << std::endl;
            break;
        }

        std::cout << "Plane detected with " << inlier_indices->indices.size() << " inliers." << std::endl;

        DetectedPrimitive plane_primitive;
        plane_primitive.type = "plane";

        // 将平面系数(a,b,c,d)存入4x4矩阵。对于平面，很多项是0。
        // 对于平面 Ax+By+Cz+D=0，可以将其表示为 q^T*x = 0，其中 x=[x,y,z,1]^T
        // q = [A, B, C, D]^T。我们将其存储在矩阵的最后一列。
        plane_primitive.model_coefficients.setZero();
        plane_primitive.model_coefficients(0, 3) = coefficients->values[0]; // A
        plane_primitive.model_coefficients(1, 3) = coefficients->values[1]; // B
        plane_primitive.model_coefficients(2, 3) = coefficients->values[2]; // C
        plane_primitive.model_coefficients(3, 3) = coefficients->values[3]; // D

        extract.setInputCloud(current_cloud);
        extract.setIndices(inlier_indices);
        extract.setNegative(false); // false = 提取内点
        extract.filter(*(plane_primitive.inliers));

        detected_primitives_.push_back(plane_primitive);

        // 6. 更新点云：移除已找到的内点，为下一次迭代做准备
        extract.setNegative(true); // true = 移除内点，保留剩余部分
        PointCloudPtr remaining_points(new PointCloud());
        extract.filter(*remaining_points);
        current_cloud.swap(remaining_points); // 用剩余点云替换当前点云
    }

    remain_cloud.swap(current_cloud);

    std::cout << "--- Plane Detection Finished. " << remain_cloud->size() << " points remain. ---" << std::endl;
}

// 添加了动态迭代和模型修正的版本
bool MSQ::findQuadric(const PointCloudPtr &cloud,
                      Eigen::Matrix4f &best_model_coefficients,
                      pcl::PointIndices::Ptr &best_inlier_indices)
{
    const int total_points = cloud->size();
    // 确保有足够的点进行采样和验证
    if (total_points < 9)
    {
        return false;
    }

    // 初始化RANSAC的最佳结果
    size_t best_inlier_count = 0;
    best_inlier_indices.reset(new pcl::PointIndices());

    // 使用现代C++的随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, total_points - 1);

    // 【优化点1】动态迭代次数
    int max_iterations = params_.quadric_max_iterations; // 从参数中获取初始最大迭代次数

    // RANSAC 主循环开始
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // --- 步骤 A: 生成一个候选模型 (Hypothesis Generation) ---
        auto start_time = std::chrono::high_resolution_clock::now();
        // A.1. 从点云中不重复地随机采样3个点的索引
        int idx[3];
        idx[0] = distrib(gen);
        do
        {
            idx[1] = distrib(gen);
        } while (idx[1] == idx[0]);
        do
        {
            idx[2] = distrib(gen);
        } while (idx[2] == idx[0] || idx[2] == idx[1]);

        // A.2. 构建 9x10 约束矩阵 A_basis
        Eigen::Matrix<double, 9, 10> A_basis;
        bool sample_valid = true;
        for (int i = 0; i < 3; ++i)
        {
            const auto &p = cloud->points[idx[i]];
            if (!pcl::isFinite(p) || std::isnan(p.normal_x) || std::isnan(p.normal_y) || std::isnan(p.normal_z))
            {
                sample_valid = false;
                break;
            }
            A_basis.row(i * 3) << p.x * p.x, p.y * p.y, p.z * p.z, 2 * p.x * p.y, 2 * p.x * p.z, 2 * p.y * p.z, 2 * p.x, 2 * p.y, 2 * p.z, 1.0;
            A_basis.row(i * 3 + 1) << p.x * p.normal_y, -p.y * p.normal_x, 0, p.y * p.normal_y - p.x * p.normal_x, p.z * p.normal_y, -p.z * p.normal_x, p.normal_y, -p.normal_x, 0, 0;
            A_basis.row(i * 3 + 2) << p.x * p.normal_z, 0, -p.z * p.normal_x, p.y * p.normal_z, p.z * p.normal_z - p.x * p.normal_x, -p.y * p.normal_x, p.normal_z, 0, -p.normal_x, 0;
        }
        if (!sample_valid)
            continue;

        Eigen::JacobiSVD<Eigen::Matrix<double, 9, 10>> svd(A_basis, Eigen::ComputeFullV);
        Eigen::VectorXd m = svd.matrixV().col(9);

        Eigen::Matrix<double, 9, 9> A_sub = A_basis.leftCols(9);
        Eigen::VectorXd b_sub = -A_basis.col(9);
        Eigen::VectorXd q_sub = A_sub.colPivHouseholderQr().solve(b_sub);
        if (q_sub.hasNaN())
            continue;
        Eigen::VectorXd p(10);
        p.head(9) = q_sub;
        p(9) = 1.0;

        // A.3. 投票
        Eigen::Matrix4f P = vectorToQMatrix(p);
        Eigen::Matrix4f M = vectorToQMatrix(m);
        const int num_bins = static_cast<int>(M_PI / params_.voting_bin_size);
        std::vector<int> voting_bins(num_bins, 0);
        for (int j = 0; j < total_points; ++j)
        {
            Eigen::Vector4f pt_h(cloud->points[j].x, cloud->points[j].y, cloud->points[j].z, 1.0f);
            float num = -(pt_h.transpose() * P * pt_h).coeff(0, 0);
            float den = (pt_h.transpose() * M * pt_h).coeff(0, 0);
            if (std::abs(den) < 1e-9)
                continue;
            double mu = num / den;
            double angle = std::atan(mu);
            int bin_idx = static_cast<int>((angle + M_PI / 2.0) / params_.voting_bin_size);
            if (bin_idx >= 0 && bin_idx < num_bins)
            {
                voting_bins[bin_idx]++;
            }
        }

        auto max_it = std::max_element(voting_bins.begin(), voting_bins.end());
        int peak_bin_idx = std::distance(voting_bins.begin(), max_it);
        double best_angle = (peak_bin_idx + 0.5) * params_.voting_bin_size - M_PI / 2.0;
        double best_mu = std::tan(best_angle);

        // A.4. 合成候选模型
        Eigen::VectorXd q_candidate_vec = p + best_mu * m;
        Eigen::Matrix4f candidate_model = vectorToQMatrix(q_candidate_vec);
        double norm = candidate_model.norm();
        if (std::abs(norm) < 1e-9)
            continue;
        candidate_model /= norm;
        auto end_hypothesis_time = std::chrono::high_resolution_clock::now();
        // --- 步骤 B: 验证候选模型 (Verification) ---
        pcl::PointIndices::Ptr current_inliers(new pcl::PointIndices);
        current_inliers->indices.reserve(total_points); // 预分配内存
        for (int i = 0; i < total_points; ++i)
        {
            Eigen::Vector4f pt_h(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z, 1.0f);
            float dist = std::abs((pt_h.transpose() * candidate_model * pt_h).coeff(0, 0));
            if (dist < params_.quadric_distance_threshold)
            {
                current_inliers->indices.push_back(i);
            }
        }
        auto end_verification_time = std::chrono::high_resolution_clock::now();
        // --- 步骤 C: 更新最佳结果 (Update) ---
        if (current_inliers->indices.size() > best_inlier_count)
        {
            best_inlier_count = current_inliers->indices.size();
            best_inlier_indices = current_inliers; 

            // 【优化点1 实现】: 动态更新迭代次数
            double inlier_ratio = static_cast<double>(best_inlier_count) / total_points;
            double desired_confidence = 0.99; 
            int sample_size = 3;

            if (inlier_ratio > 1e-6 && inlier_ratio < 1.0 - 1e-9)
            {
                double p_no_outliers = std::pow(inlier_ratio, sample_size);
                // 再次检查分母，防止log(0)
                if (p_no_outliers < 1.0 - 1e-9)
                {
                    int required_iterations = static_cast<int>(std::log(1.0 - desired_confidence) / std::log(1.0 - p_no_outliers));
                    // 更新迭代次数，但不超过初始设定的最大值
                    max_iterations = std::min(params_.quadric_max_iterations, std::max(1, required_iterations));
                }
            }
        }
        auto end_update_time = std::chrono::high_resolution_clock::now();
        if (iter % 500 == 0)
        { // 每500次迭代打印一次，避免刷屏
            auto duration_hypothesis = std::chrono::duration_cast<std::chrono::microseconds>(end_hypothesis_time - start_time);
            auto duration_verification = std::chrono::duration_cast<std::chrono::microseconds>(end_verification_time - end_hypothesis_time);
            auto duration_update = std::chrono::duration_cast<std::chrono::microseconds>(end_update_time - end_verification_time);

            std::cout << "Iter " << iter << ": "
                      << "Hypothesis: " << duration_hypothesis.count() << " us, "
                      << "Verification: " << duration_verification.count() << " us, "
                      << "Update: " << duration_update.count() << " us" << std::endl;
        }
    }
    // RANSAC 主循环结束

    // --- 步骤 D: 模型修正与返回 ---
    const double min_inlier_abs = total_points * params_.min_quadric_inlier_percentage;
    if (best_inlier_count > std::max(9.0, min_inlier_abs))
    {
        std::cout << "RANSAC found a potential model with " << best_inlier_count << " inliers. Refining with Ceres..." << std::endl;

        // --- 尝试使用Ceres进行非线性优化 ---

        // 1. 将RANSAC找到的最佳模型作为Ceres的初始值
        // 首先需要一个Eigen::Matrix4f到double[10]的转换
        Eigen::VectorXd q_initial_vec = QMatrixToVector(best_model_coefficients); // 您需要实现这个辅助函数
        double q_initial[10];
        for (int i = 0; i < 10; ++i)
            q_initial[i] = q_initial_vec(i);

        // 2. 创建Ceres问题
        ceres::Problem problem;

        // 3. 为每一个内点添加一个残差块
        for (const auto &index : best_inlier_indices->indices)
        {
            const auto &p = cloud->points[index];
            ceres::CostFunction *cost_function =
                new ceres::AutoDiffCostFunction<QuadricGeometricError, 1, 10>(
                    new QuadricGeometricError(p.x, p.y, p.z));
            problem.AddResidualBlock(cost_function, nullptr, q_initial);
        }

        // 4. 配置并运行求解器
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false; 
        options.max_num_iterations = 50;
        options.function_tolerance = 1e-6;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        // 5. 从Ceres结果中恢复优化后的模型
        Eigen::VectorXd q_refined_vec(10);
        for (int i = 0; i < 10; ++i)
            q_refined_vec(i) = q_initial[i];

        best_model_coefficients = vectorToQMatrix(q_refined_vec); 
        best_model_coefficients.normalize();                      // 归一化最终结果

        std::cout << "Ceres refinement finished. Final loss: " << summary.final_cost << std::endl;

        return true; 
    }

    return false; 
}

/**
 * @brief 在点云中循环检测二次曲面。
 * @param remain_cloud [输入/输出] 待处理的点云，函数会从中移除已识别的基元。
 */
void MSQ::detectQuadric(PointCloudPtr &remain_cloud)
{
    const size_t initial_point_count = remain_cloud->size();
    if (initial_point_count == 0)
        return;
    // --- 循环检测，直到剩余点太少 ---
    while (true)
    {
        // 检查剩余点数是否满足继续检测的条件
        if (remain_cloud->size() < initial_point_count * params_.min_remaining_points_percentage)
        {
            std::cout << "Remaining points too few. Stopping quadric detection." << std::endl;
            break;
        }

        Eigen::Matrix4f model_coefficients;
        pcl::PointIndices::Ptr inlier_indices(new pcl::PointIndices);

        // 调用内部核心函数，尝试在当前剩余点云中找到一个最佳二次曲面
        bool found = findQuadric(remain_cloud, model_coefficients, inlier_indices);

        if (found)
        {
            // 找到了一个有效的二次曲面
            std::cout << "Found a quadric with " << inlier_indices->indices.size() << " inliers." << std::endl;

            // 1. 保存检测结果
            DetectedPrimitive primitive;
            primitive.type = "quadric";
            primitive.model_coefficients = model_coefficients;

            // 从 remain_cloud 中提取内点，保存到基元信息中
            pcl::ExtractIndices<PointT> extract_inliers;
            extract_inliers.setInputCloud(remain_cloud);
            extract_inliers.setIndices(inlier_indices);
            extract_inliers.setNegative(false); // false = 提取内点
            extract_inliers.filter(*(primitive.inliers));

            detected_primitives_.push_back(primitive);

            // 2. 从 remain_cloud 中移除内点，为下一次循环做准备
            pcl::ExtractIndices<PointT> extract_outliers;
            extract_outliers.setInputCloud(remain_cloud);
            extract_outliers.setIndices(inlier_indices);
            extract_outliers.setNegative(true); // true = 移除内点, 保留外点
            PointCloudPtr next_remain_cloud(new PointCloud());
            extract_outliers.filter(*next_remain_cloud);

            remain_cloud = next_remain_cloud; // 更新剩余点云
        }
        else
        {
            // 在当前剩余点云中再也找不到满足条件的二次曲面了
            std::cout << "No more quadrics could be found." << std::endl;
            break; // 结束循环
        }
    }
}
/**
 * @brief 将4x4的二次曲面矩阵Q转换为10维参数向量q。
 * @param Q 输入的4x4对称矩阵。
 * @return 包含10个二次曲面系数的向量 [A,B,C,D,E,F,G,H,I,J]。
 */
Eigen::VectorXd MinimalSampleQuadric::QMatrixToVector(const Eigen::Matrix4f &Q) const
{
    Eigen::VectorXd q_vec(10);

    // 根据映射关系从矩阵中提取系数
    // Q(row, col)
    q_vec(0) = Q(0, 0); // A: x^2 coeff
    q_vec(1) = Q(1, 1); // B: y^2 coeff
    q_vec(2) = Q(2, 2); // C: z^2 coeff

    // 注意：非对角线元素对应的是 2*D*xy, 2*E*xz 等，所以我们直接取上三角部分
    q_vec(3) = Q(0, 1); // D: xy coeff
    q_vec(4) = Q(0, 2); // E: xz coeff
    q_vec(5) = Q(1, 2); // F: yz coeff

    q_vec(6) = Q(0, 3); // G: x coeff
    q_vec(7) = Q(1, 3); // H: y coeff
    q_vec(8) = Q(2, 3); // I: z coeff

    q_vec(9) = Q(3, 3); // J: constant term

    return q_vec;
}
