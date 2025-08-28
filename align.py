import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd, voxel_size):
    print(f"Downsampling с voxel_size = {voxel_size} мм")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(f"Вычисление нормалей с radius = {radius_normal}")
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down


def execute_global_registration_fast(source_down, target_down, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(f"Глобальная регистрация (FAST) с порогом {distance_threshold} мм...")

    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        False,  # mutual_filter обязательно нужно указать
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000))
    return result


def refine_registration(source, target, initial_transformation, voxel_size, method='point_to_plane'):
    distance_threshold = voxel_size * 1.0
    print(f"Уточнение трансформации ICP методом {method}, порог: {distance_threshold} мм")

    if method == 'point_to_plane':
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transformation, estimation)
    return result


def evaluate_registration(source, target):
    distances = source.compute_point_cloud_distance(target)
    distances = np.asarray(distances)
    print(f"Среднее расстояние: {np.mean(distances):.4f} мм")
    print(f"Максимальное расстояние: {np.max(distances):.4f} мм")
    print(f"Медиана расстояния: {np.median(distances):.4f} мм")


def main():
    voxel_sizes = [3.0, 1.5, 0.5]  # Перебор разных масштабов для оптимизации
    best_fitness = 0
    best_result = None
    best_voxel = None
    best_icp_method = None

    # Загрузка моделей
    pcd1 = o3d.io.read_point_cloud("/home/kirill/projects/3d_reconstr/models/0705_01_pc.ply")
    pcd2 = o3d.io.read_point_cloud("/home/kirill/projects/3d_reconstr/models/0705_02_pc.ply")

    for voxel_size in voxel_sizes:
        print(f"\n=== Пробуем voxel_size = {voxel_size} ===")

        source_down = preprocess_point_cloud(pcd2, voxel_size)
        target_down = preprocess_point_cloud(pcd1, voxel_size)

        # Глобальная регистрация с использованием FPFH и RANSAC
        result_ransac = execute_global_registration_fast(source_down, target_down, voxel_size)
        print("Грубая трансформация из глобальной регистрации:")
        print(result_ransac.transformation)

        # Уточнение ICP с Point-to-Plane
        result_icp_plane = refine_registration(source_down, target_down, result_ransac.transformation, voxel_size, method='point_to_plane')
        print(f"Уточнение трансформации ICP методом Point-to-Plane")
        print(f"Fitness: {result_icp_plane.fitness:.4f}, RMSE: {result_icp_plane.inlier_rmse:.4f}")

        # Уточнение ICP с Point-to-Point
        result_icp_point = refine_registration(source_down, target_down, result_icp_plane.transformation, voxel_size, method='point_to_point')
        print(f"Уточнение трансформации ICP методом Point-to-Point")
        print(f"Fitness: {result_icp_point.fitness:.4f}, RMSE: {result_icp_point.inlier_rmse:.4f}")

        # Сохраняем лучший результат по fitness
        if result_icp_point.fitness > best_fitness:
            best_fitness = result_icp_point.fitness
            best_result = result_icp_point
            best_voxel = voxel_size
            best_icp_method = 'point_to_point'

    print("\n=== Лучшие параметры ===")
    print(f"Voxel size: {best_voxel}, ICP method: {best_icp_method}")
    print("Трансформация:")
    print(best_result.transformation)

    # Применяем лучшую трансформацию к исходному облаку (для анализа)
    pcd2.transform(best_result.transformation)
    evaluate_registration(pcd2, pcd1)

    # Визуализация
    pcd1.paint_uniform_color([1, 0.706, 0])  # Оранжевый
    pcd2.paint_uniform_color([0, 0.651, 0.929])  # Синий
    o3d.visualization.draw_geometries([pcd1, pcd2])


if __name__ == "__main__":
    main()
