import json
import numpy as np
from collections import defaultdict, deque
import folium
from flask import Flask, jsonify, send_from_directory, request
import os

# Tạo ứng dụng Flask
app = Flask(__name__)

# Định nghĩa thư mục để lưu trữ tệp map
OUTPUT_FOLDER = 'output_maps'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load data từ các file JSON
def load_data(points_file, info_file):
    with open(points_file, 'r', encoding='utf-8') as f:
        points_data = json.load(f)
    with open(info_file, 'r', encoding='utf-8') as f:
        info_data = {point['id']: point for point in json.load(f)}
    return points_data, info_data

# Tạo ma trận kề cho các điểm
def create_adjacency_matrix(points_data, info_data, alpha=0.5):
    matrix_size = 209
    adjacency_matrix = np.zeros((matrix_size, matrix_size))
    id_to_index = {point_id: idx for idx, point_id in enumerate(info_data.keys())}

    for point in points_data:
        from_id = point.get('id')
        u = id_to_index.get(from_id)
        if u is None:
            continue
        for distance in point['distances']:
            to_id = distance.get('id')
            v = id_to_index.get(to_id)
            if v is None:
                continue
            
            # Lấy giá trị length, duration và baseDuration
            length = distance.get('length')
            duration = distance.get('duration', 1)
            base_duration = distance.get('baseDuration', duration)
            
            # Tính toán công suất
            capacity = alpha * length + (1 - alpha) * (length * base_duration / duration)
            adjacency_matrix[u][v] = capacity
    return adjacency_matrix, id_to_index

# Dinic's Algorithm for Maximum Flow
class Dinic:
    def __init__(self, n, scaling_factor=1):
        self.n = n
        self.adj = defaultdict(list)
        self.capacity = defaultdict(lambda: defaultdict(int))
        self.flow_passed = defaultdict(lambda: defaultdict(int))
        self.scaling_factor = scaling_factor

    def add_edge(self, u, v, cap):
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.capacity[u][v] += cap

    def bfs(self, source, sink, level):
        queue = deque([source])
        level[source] = 0
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if level[v] < 0 and self.capacity[u][v] > 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        return level[sink] != -1

    def dfs(self, u, sink, flow, level, start):
        if u == sink:
            return flow
        while start[u] < len(self.adj[u]):
            v = self.adj[u][start[u]]
            if level[v] == level[u] + 1 and self.capacity[u][v] > 0:
                min_flow = min(flow, self.capacity[u][v])
                pushed_flow = self.dfs(v, sink, min_flow, level, start)
                if pushed_flow > 0:
                    self.capacity[u][v] -= pushed_flow
                    self.capacity[v][u] += pushed_flow
                    self.flow_passed[u][v] += pushed_flow
                    return pushed_flow
            start[u] += 1
        return 0

    def max_flow(self, source, sink):
        total_flow = 0
        while True:
            level = [-1] * self.n
            if not self.bfs(source, sink, level):
                break
            start = [0] * self.n
            while (flow := self.dfs(source, sink, float('inf'), level, start)):
                total_flow += flow
        return total_flow

    def save_flow_to_txt(self, filename="flow_values.txt"):
        with open(filename, 'w') as f:
            for u in self.flow_passed:
                for v in self.flow_passed[u]:
                    flow_value = self.flow_passed[u][v] / self.scaling_factor
                    if flow_value > 0:
                        f.write(f"Flow from {u} to {v}: {flow_value:.2f}\n")
        print(f"Flow values saved to {filename}")

# Hàm vẽ bản đồ với flow values
def plot_graph_on_map(info_data, adjacency_matrix, dinic_graph, id_to_index, source, sink):
    map_center = [10.789990, 106.678101]  # Coordinates near Ho Chi Minh City
    map_ = folium.Map(location=map_center, zoom_start=13)

    # Set để lưu trữ các điểm có dòng chảy
    points_with_flow = set()

    # Kiểm tra các cạnh và lưu trữ các điểm có dòng chảy qua
    for u in range(len(adjacency_matrix)):
        for v in range(len(adjacency_matrix)):
            if adjacency_matrix[u][v] > 0:  # Kiểm tra nếu có cạnh giữa u và v
                flow_value = dinic_graph.flow_passed[u][v] / dinic_graph.scaling_factor
                
                if flow_value > 0:  # Nếu có dòng chảy
                    points_with_flow.add(u)  # Thêm điểm u vào set
                    points_with_flow.add(v)  # Thêm điểm v vào set

    for point_id, point_info in info_data.items():
        idx = id_to_index[point_id]
        lat, lng = point_info['position']['lat'], point_info['position']['lng']

        if idx in points_with_flow:  # Chỉ vẽ điểm nếu có dòng chảy qua
            if idx == source:
                color = 'yellow'  # Màu đỏ cho nguồn
            elif idx == sink:
                color = 'black'  # Màu tím cho đích
            else:
                color = 'red'  # Màu đỏ cho các điểm có dòng chảy qua

            folium.CircleMarker(location=[lat, lng], radius=8, color=color, fill=True, fill_color=color, fill_opacity=1).add_to(map_)

    # Vẽ các cạnh chỉ khi có dòng chảy
    for u in range(len(adjacency_matrix)):
        for v in range(len(adjacency_matrix)):
            if adjacency_matrix[u][v] > 0:  # Kiểm tra nếu có cạnh giữa u và v
                flow_value = dinic_graph.flow_passed[u][v] / dinic_graph.scaling_factor
                
                if flow_value > 0:  # Vẽ đường đi chỉ khi có dòng chảy
                    from_point = info_data[list(info_data.keys())[u]]
                    to_point = info_data[list(info_data.keys())[v]]
                    from_lat, from_lng = from_point['position']['lat'], from_point['position']['lng']
                    to_lat, to_lng = to_point['position']['lat'], to_point['position']['lng']

                    # Vẽ PolyLine với màu và độ dày phụ thuộc vào dòng chảy
                    folium.PolyLine(locations=[[from_lat, from_lng], [to_lat, to_lng]], color='blue', weight=4, opacity=0.6).add_to(map_)

    # Lưu bản đồ dưới dạng file HTML
    map_path = os.path.join(OUTPUT_FOLDER, "flow_network_map.html")
    map_.save(map_path)
    return map_path



@app.route('/run_algorithm', methods=['GET'])
def run_algorithm():
    points_file = './data/208Points.json'
    info_file = './data/info_point.json'
    
    points_data, info_data = load_data(points_file, info_file)
    alpha = 0.5  # Set alpha based on the importance of length vs. congestion
    adjacency_matrix, id_to_index = create_adjacency_matrix(points_data, info_data, alpha)

    # Initialize Dinic's graph
    scaling_factor = 1
    dinic_graph = Dinic(len(adjacency_matrix), scaling_factor)
    for u in range(len(adjacency_matrix)):
        for v in range(len(adjacency_matrix)):
            if adjacency_matrix[u][v] > 0:
                dinic_graph.add_edge(u, v, adjacency_matrix[u][v])

    source = 103
    sink = 188
    max_flow_value = dinic_graph.max_flow(source, sink) / scaling_factor
    dinic_graph.save_flow_to_txt("flow_values.txt")
    
    # Vẽ bản đồ
    map_path = plot_graph_on_map(info_data, adjacency_matrix, dinic_graph, id_to_index, source, sink)
    
    return jsonify({
        'max_flow': max_flow_value,
        'map_url': map_path
    })

# Chạy ứng dụng Flask
if __name__ == "__main__":
    app.run(debug=True)
