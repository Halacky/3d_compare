import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def load_mesh_as_pcd(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    pcd = mesh.sample_points_uniformly(number_of_points=500000)
    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = source.transform(transformation)
    o3d.visualization.draw_geometries([source_temp.paint_uniform_color([1, 0, 0]),
                                       target.paint_uniform_color([0, 1, 0])])

def align_with_pca(pcd):
    points = np.asarray(pcd.points)
    
    # Центрируем точки
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    
    # PCA: находим собственные векторы ковариационной матрицы
    cov_matrix = np.cov(centered_points.T)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    
    # Создаем матрицу 4x4 для Open3D
    R = eigvecs.T  # Транспонируем, чтобы использовать как ортогональную базу
    T = np.eye(4)  # Создаем единичную матрицу 4x4
    T[:3, :3] = R  # Вставляем матрицу вращения
    T[:3, 3] = -R @ mean  # Корректируем с учетом среднего положения

    # Применяем трансформацию
    pcd.transform(T)
    return pcd


def compute_icp(source_pcd, target_pcd):
    threshold = 10.0  # Допустимый порог для ICP
    trans_init = np.eye(4)  # Начальная трансформация

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2p.transformation


def visualize_difference(pcd1, pcd2):
    # Вычисляем расстояния от точек pcd1 до ближайших точек в pcd2
    dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    print(dists.mean())

    # Сортировка расстояний и индексов
    sorted_indices = np.argsort(dists)[::-1]  # Индексы отсортированных расстояний (по убыванию)

    # Устанавливаем максимальное количество точек, которые будем выделять красным
    top_n = 2000
    red_indices = sorted_indices[:top_n]  # Топ-50 индексов максимальных расстояний
    green_indices = sorted_indices[top_n:]  # Остальные индексы

    # Инициализация цветов: сначала все зеленые
    colors = np.ones((len(dists), 3)) * np.array([0, 1, 0])  # Зеленый цвет для всех точек

    # Красным выделяем топ-50 точек
    colors[red_indices] = np.array([1, 0, 0])  # Красный для топ-50 точек

    # Применяем цвета к облаку точек
    pcd1.colors = o3d.utility.Vector3dVector(colors)

    # Второе облако делаем однотонным (например, зеленым для контраста)
    pcd2.paint_uniform_color([0, 1, 0])  # Зеленый цвет для второго облака

    # Визуализируем
    o3d.visualization.draw_geometries([pcd1, pcd2])

if __name__ == "__main__":
    min_dists = 1
    while min_dists>0.1:
        source_pcd = load_mesh_as_pcd("/home/handbook/work/3d/some_drist_def.obj")
        target_pcd = load_mesh_as_pcd("/home/handbook/work/3d/some_drist.obj")

        # target_pcd = o3d.io.read_point_cloud("/home/handbook/work/3d/final_target_pcd.pcd")
        # source_pcd = o3d.io.read_point_cloud("/home/handbook/work/3d/final_source_pcd.pcd")

        source_pcd = align_with_pca(source_pcd)
        target_pcd = align_with_pca(target_pcd)

        dists = np.asarray(source_pcd.compute_point_cloud_distance(target_pcd)).mean()
        if dists < min_dists:
            min_dists = dists
            print(dists)
            o3d.io.write_point_cloud("/home/handbook/work/3d/final_source_pcd.pcd", source_pcd)
            o3d.io.write_point_cloud("/home/handbook/work/3d/final_target_pcd.pcd", target_pcd)


            visualize_difference(source_pcd, target_pcd)
    
