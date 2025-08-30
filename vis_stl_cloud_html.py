"""
visualize_stl_pointcloud.py

Скрипт для интерактивной визуализации STL-файлов как облака точек.
Выбирает точки с поверхности меша (равномерная выборка по площадям треугольников)
и отображает их в интерактивном виде в браузере с помощью Plotly.

Требования:
    pip install trimesh plotly numpy

Использование (из командной строки):
    python visualize_stl_pointcloud.py path/to/model.stl --points 50000 --marker-size 2 --export output.html

Если вы запускаете в Jupyter / Colab, можно просто импортировать функцию `visualize`.

Автор: автогенерация (пример)
"""

import argparse
import os
import sys

import numpy as np

try:
    import trimesh
except Exception as e:
    raise ImportError("Требуется библиотека 'trimesh'. Установите: pip install trimesh") from e

try:
    import plotly.graph_objects as go
    from plotly.offline import plot as plot_offline
except Exception as e:
    raise ImportError("Требуется библиотека 'plotly'. Установите: pip install plotly") from e


def sample_points_from_mesh(mesh: trimesh.Trimesh, n_points: int = 20000, return_normals: bool = True):
    """
    Равномерно выбирает n_points точек по поверхности меша (с учётом площади треугольников).
    Возвращает точки (N,3) и опционально нормали (N,3).
    """
    # trimesh имеет встроенную функцию sample
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    if return_normals:
        # получим нормали треугольников и назначим их точкам
        face_normals = mesh.face_normals
        normals = face_normals[face_idx]
        return pts, normals
    return pts, None


def visualize(stl_path: str, n_points: int = 20000, marker_size: float = 2.0, color_by: str = "z", output_html: str = None):
    """
    Загружает STL, выбирает точки и отображает интерактивно в браузере.

    color_by: 'z' | 'normals' | 'rgb' | 'height'
        'z' - цвет по координате Z
        'normals' - цвет по нормалям (если доступны)
        'height' - то же, что 'z'
        'rgb' - пытается взять цвета из mesh.visual (если присутствуют)
    """
    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"Файл не найден: {stl_path}")

    mesh = trimesh.load_mesh(stl_path, force='mesh')
    if mesh.is_empty:
        raise RuntimeError("Загруженный меш пуст.")

    pts, normals = sample_points_from_mesh(mesh, n_points=n_points, return_normals=True)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # формируем цвета
    if color_by == 'normals' and normals is not None:
        # нормализуем в диапазон 0..255
        c = ((normals - normals.min()) / (normals.ptp() + 1e-9) * 255).astype(int)
        colors = ['rgb({},{},{})'.format(r, g, b) for r, g, b in c]
    elif color_by in ('z', 'height'):
        colorscale = 'Viridis'
        colors = z  # Plotly автоматически применит colorscale
    elif color_by == 'rgb' and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        # попробуем взять цвета вершин; если их мало, используем средний цвет
        vc = np.array(mesh.visual.vertex_colors)
        if vc.shape[0] >= pts.shape[0]:
            # некоторая привязка: используем ближайшую вершину для каждой точки
            # быстрый (но приближённый) способ — найти ближайшую вершину через KDTree
            try:
                from scipy.spatial import cKDTree as KDTree
                tree = KDTree(mesh.vertices)
                _, idx = tree.query(pts, k=1)
                selected = vc[idx]
                colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b, *_ in selected]
            except Exception:
                # fallback — просто используем усреднённый цвет
                avg = vc.mean(axis=0)
                colors = ['rgb({},{},{})'.format(int(avg[0]), int(avg[1]), int(avg[2]))] * pts.shape[0]
        else:
            avg = vc.mean(axis=0)
            colors = ['rgb({},{},{})'.format(int(avg[0]), int(avg[1]), int(avg[2]))] * pts.shape[0]
    else:
        # дефолт: цвет по Z
        colors = z

    scatter = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=colors,
            colorscale='Viridis' if isinstance(colors, (list, np.ndarray)) and not isinstance(colors[0], str) else None,
            opacity=0.9,
        ),
        hoverinfo='none'
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    fig = go.Figure(data=[scatter], layout=layout)

    if output_html:
        # Запишем интерактивный HTML-файл
        plot_offline(fig, filename=output_html, auto_open=True)
        print(f"Экспортировано в {output_html}")
    else:
        # Откроем в браузере как временную HTML-страницу
        plot_offline(fig, filename='tmp_stl_pointcloud.html', auto_open=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Визуализатор STL как облака точек (интерактивный plotly)')
    parser.add_argument('stl', help='Путь к STL файлу')
    parser.add_argument('--points', '-n', type=int, default=20000, help='Количество точек для выборки (по умолчанию 20000)')
    parser.add_argument('--marker-size', type=float, default=2.0, help='Размер маркера (по умолчанию 2.0)')
    parser.add_argument('--color-by', type=str, choices=['z', 'normals', 'rgb', 'height'], default='z', help='Как окрашивать точки (z, normals, rgb)')
    parser.add_argument('--export', type=str, default=None, help='Путь для экспорта HTML (если задан — файл будет записан)')

    args = parser.parse_args()

    visualize(args.stl, n_points=args.points, marker_size=args.marker_size, color_by=args.color_by, output_html=args.export)
