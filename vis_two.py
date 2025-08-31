
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

def main():
    file1_path = "/home/kirill/projects/3d_reconstr/data/test.stl"
    file2_path = "/home/kirill/projects/3d_reconstr/aligned_source_trimesh2.stl"
    
    # Проверка файлов
    if not all(os.path.exists(f) for f in [file1_path, file2_path]):
        print("Файлы не найдены")
        return
    
    print("Создание интерактивной визуализации с Plotly...")
    visualize_with_plotly(file1_path, file2_path)

if __name__ == "__main__":
    main()
