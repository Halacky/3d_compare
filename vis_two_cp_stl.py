import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os

def visualize_with_matplotlib(file1_path, file2_path):
    """
    Визуализация облаков точек с использованием matplotlib
    """
    try:
        # Загрузка и преобразование в облака точек
        mesh1 = o3d.io.read_triangle_mesh(file1_path)
        pcd1 = mesh1.sample_points_poisson_disk(number_of_points=1000)
        points1 = np.asarray(pcd1.points)
        
        mesh2 = o3d.io.read_triangle_mesh(file2_path)
        pcd2 = mesh2.sample_points_poisson_disk(number_of_points=1000)
        points2 = np.asarray(pcd2.points)
        
        # Создание 3D графика
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Отображение точек
        ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], 
                  c='red', marker='o', s=20, label='Облако 1', alpha=0.6)
        ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], 
                  c='blue', marker='^', s=20, label='Облако 2', alpha=0.6)
        
        # Настройка графика
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Визуализация облаков точек из STL файлов')
        ax.legend()
        
        # Включение вращения
        ax.view_init(elev=20, azim=45)
        
        plt.show()
        
    except Exception as e:
        print(f"Ошибка: {e}")

def main():
    # Установка переменной окружения
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-user'
    
    file1_path = "/home/src/data/test.stl"
    file2_path = "/home/src/data/test_rotated.stl"
    
    # Проверка файлов
    if not all(os.path.exists(f) for f in [file1_path, file2_path]):
        print("Создание тестовых файлов...")
        create_sample_stl_files()
        file1_path, file2_path = "sample1.stl", "sample2.stl"
    
    # Попробовать Open3D визуализацию
    try:
        pcd1, pcd2 = load_and_visualize_stl_files(file1_path, file2_path)
    except:
        print("Open3D визуализация не сработала, используем matplotlib...")
        visualize_with_matplotlib(file1_path, file2_path)

if __name__ == "__main__":import numpy as np
import open3d as o3d
import os
import plotly.graph_objects as go
from plotly.offline import plot

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

def main():
    file1_path = "/home/src/data/test.stl"
    file2_path = "/home/src/data/test_rotated.stl"
    
    # Проверка файлов
    if not all(os.path.exists(f) for f in [file1_path, file2_path]):
        print("Файлы не найдены")
        return
    
    print("Создание интерактивной визуализации с Plotly...")
    visualize_with_plotly(file1_path, file2_path)

if __name__ == "__main__":
    main()
    main()
