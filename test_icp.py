import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def create_cube_point_cloud(size=1.0, resolution=50):
    """Создает облако точек для идеального куба."""
    x = np.linspace(-size / 2, size / 2, resolution)
    y = np.linspace(-size / 2, size / 2, resolution)
    z = np.linspace(-size / 2, size / 2, resolution)
    
    points = []
    for i in x:
        for j in y:
            points.append([i, j, -size/2])  # Нижняя грань
            points.append([i, j, size/2])   # Верхняя грань
    for i in x:
        for k in z:
            points.append([i, -size/2, k])  # Передняя грань
            points.append([i, size/2, k])   # Задняя грань
    for j in y:
        for k in z:
            points.append([-size/2, j, k])  # Левая грань
            points.append([size/2, j, k])   # Правая грань
    
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points)))

def create_defective_cube(size=1.0, resolution=50, defect_size=0.2):
    """Создает облако точек для куба с различными дефектами: скос, вмятины, трещины, наросты."""
    pcd = create_cube_point_cloud(size, resolution)
    
    points = np.asarray(pcd.points)
        
    # Вмятина на одной грани
    dent_mask = (points[:, 0] > -0.2) & (points[:, 0] < 0.2) & (points[:, 1] > -0.2) & (points[:, 1] < 0.2) & (points[:, 2] > 0.4)
    
    # Вводим параболическое смещение по оси Z
    distance_from_center = np.sqrt(points[dent_mask, 0]**2 + points[dent_mask, 1]**2)
    k = 1.0  # Можно настроить для более сильного/слабого смещения
    b = 0.1  # Смещение в центре (max смещение)
    
    # Параболическое смещение по оси Z
    z_shift = -k * (distance_from_center ** 2) + b
    points[dent_mask, 2] -= z_shift  # Применяем смещение к координате Z
    
    # Трещина - удаляем тонкую полосу точек
    # crack_mask = (points[:, 0] > -0.1) & (points[:, 0] < 0.1) & (points[:, 2] < -0.3)
    # points = points[~crack_mask]
    
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


def align_and_compare(pcd1, pcd2):
    """Выполняет выравнивание ICP, отображает оба куба и выделяет разницу на одном холсте."""
    # Выравнивание ICP
    threshold = 0.02  # Максимальное расстояние для соответствующих точек
    transformation = o3d.pipelines.registration.registration_icp(
        pcd2, pcd1, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation
    
    pcd2.transform(transformation)  # Применяем трансформацию к дефектному кубу
    
    # Вычисляем разницу
    dists = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    colors = plt.cm.jet(dists / max(dists))[:, :3]  # Преобразование в цвета
    pcd1.colors = o3d.utility.Vector3dVector(colors)
    
    # Окрашиваем дефектный куб в красный
    pcd2.paint_uniform_color([1, 0, 0])
    
    # Создаем копии для индивидуальной визуализации
    pcd1_original = o3d.geometry.PointCloud(pcd1)
    pcd2_original = o3d.geometry.PointCloud(pcd2)
    
    # Окрашиваем оригинальные кубы
    pcd1_original.paint_uniform_color([0, 1, 0])
    pcd2_original.paint_uniform_color([1, 0, 0])
    
    # Размещаем кубы в разных позициях для визуализации на одном холсте
    pcd1_original.translate([-1.5, 0, 0])  # Сдвигаем эталонный куб влево
    pcd2_original.translate([1.5, 0, 0])   # Сдвигаем дефектный куб вправо
    
    # Визуализация всех объектов на одном холсте
    o3d.visualization.draw_geometries([pcd1_original, pcd2_original, pcd1, pcd2], 
                                      window_name="Сравнение кубов с дефектами на одном холсте")

if __name__ == "__main__":
    cube = create_cube_point_cloud()
    defective_cube = create_defective_cube()
    align_and_compare(cube, defective_cube)
