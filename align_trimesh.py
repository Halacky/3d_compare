import open3d as o3d
import os
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
def visualize_with_plotly(file1_path, file2_path, output_file="plotly_visualization.html"):
    """
    Интерактивная 3D визуализация с использованием Plotly
    """
    try:
        # Загрузка и преобразование в облака точек
        mesh1 = o3d.io.read_triangle_mesh(file1_path)
        pcd1 = mesh1.sample_points_poisson_disk(number_of_points=800)
        points1 = np.asarray(pcd1.points)
        
        mesh2 = o3d.io.read_triangle_mesh(file2_path)
        pcd2 = mesh2.sample_points_poisson_disk(number_of_points=800)
        points2 = np.asarray(pcd2.points)
        
        # Создание trace для облаков точек
        trace1 = go.Scatter3d(
            x=points1[:, 0], y=points1[:, 1], z=points1[:, 2],
            mode='markers',
            marker=dict(size=3, color='red', opacity=0.7),
            name='Облако 1'
        )
        
        trace2 = go.Scatter3d(
            x=points2[:, 0], y=points2[:, 1], z=points2[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.7),
            name='Облако 2'
        )
        
        # Создание фигуры
        fig = go.Figure(data=[trace1, trace2])
        
        # Настройка layout
        fig.update_layout(
            title='Интерактивное сравнение облаков точек из STL файлов',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000,
            height=800
        )
        
        # Сохранение в HTML файл
        plot(fig, filename=output_file, auto_open=False)
        print(f"Интерактивная визуализация сохранена в: {output_file}")
        
    except Exception as e:
        print(f"Ошибка: {e}")

def global_feature_based_alignment(source_path, target_path):
    """
    Глобальная регистрация с использованием признаков FPFH
    """
    # Загрузка мешей
    source_mesh = o3d.io.read_triangle_mesh(source_path)
    target_mesh = o3d.io.read_triangle_mesh(target_path)
    
    # Создание облаков точек
    source = source_mesh.sample_points_poisson_disk(number_of_points=3000)
    target = target_mesh.sample_points_poisson_disk(number_of_points=3000)
    
    # Downsampling
    voxel_size = 2.0
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # Оценка нормалей
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # Вычисление FPFH признаков
    radius_feature = voxel_size * 5
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, 
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    # Глобальная регистрация RANSAC
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # RANSAC n points
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    
    return result.transformation, result.fitness

def refine_with_icp(source_path, target_path, initial_transformation):
    """
    Точная доводка ICP после грубого выравнивания
    """
    source_mesh = o3d.io.read_triangle_mesh(source_path)
    target_mesh = o3d.io.read_triangle_mesh(target_path)
    
    source = source_mesh.sample_points_poisson_disk(number_of_points=5000)
    target = target_mesh.sample_points_poisson_disk(number_of_points=5000)
    
    # Применяем начальное преобразование
    source.transform(initial_transformation)
    
    # ICP refinement
    threshold = 10.0  # Меньший порог для точной настройки
    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=1000,
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
    )
    
    # Комбинируем преобразования
    final_transformation = result.transformation @ initial_transformation
    return final_transformation, result.inlier_rmse

def manual_initial_alignment(source_path, target_path):
    """
    Ручное грубое выравнивание на основе bounding box
    """
    source_mesh = o3d.io.read_triangle_mesh(source_path)
    target_mesh = o3d.io.read_triangle_mesh(target_path)
    
    # Получаем bounding boxes
    source_bbox = source_mesh.get_axis_aligned_bounding_box()
    target_bbox = target_mesh.get_axis_aligned_bounding_box()
    
    # Центры bounding boxes
    source_center = source_bbox.get_center()
    target_center = target_bbox.get_center()
    
    # Смещение
    translation = target_center - source_center
    
    # Масштабирование (если нужно)
    source_size = source_bbox.get_extent()
    target_size = target_bbox.get_extent()
    scale = target_size / source_size
    
    # Создаем матрицу преобразования
    T = np.eye(4)
    T[:3, 3] = translation
    # T[0, 0] = scale[0]  # Раскомментировать если нужно масштабирование
    # T[1, 1] = scale[1]
    # T[2, 2] = scale[2]
    
    return T

def apply_transformation_to_mesh(mesh_path, transformation, output_path):
    """Применение преобразования к mesh"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.transform(transformation)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh(output_path, mesh)
    return mesh

def main():
    file1_path = "/home/kirill/projects/3d_reconstr/data/test.stl"
    file2_path = "/home/kirill/projects/3d_reconstr/aligned_source_trimesh2.stl"
    
    if not all(os.path.exists(f) for f in [file1_path, file2_path]):
        print("Файлы не найдены")
        return
    
    print("1. Ручное грубое выравнивание...")
    manual_transform = manual_initial_alignment(file1_path, file2_path)
    apply_transformation_to_mesh(file1_path, manual_transform, "manual_aligned.stl")
    
    print("2. Глобальная регистрация с признаками...")
    try:
        feature_transform, fitness = global_feature_based_alignment("manual_aligned.stl", file2_path)
        print(f"Fitness: {fitness}")
        
        # Комбинируем преобразования
        combined_transform = feature_transform @ manual_transform
        apply_transformation_to_mesh(file1_path, combined_transform, "feature_aligned.stl")
        
        print("3. Точная доводка ICP...")
        final_transform, error = refine_with_icp("feature_aligned.stl", file2_path, np.eye(4))
        final_transform = final_transform @ combined_transform
        
        print(f"Финальная ошибка: {error}")
        print("Финальная матрица преобразования:")
        print(final_transform)
        
        # Сохраняем результат
        apply_transformation_to_mesh(file1_path, final_transform, "final_aligned.stl")
        print("Финальный файл сохранен как: final_aligned.stl")
        
    except Exception as e:
        print(f"Ошибка при feature-based регистрации: {e}")
        print("Используем только ручное выравнивание")
        
    # Визуализация
    visualize_with_plotly("final_aligned.stl", file2_path, "final_visualization.html")

if __name__ == "__main__":
    main()
