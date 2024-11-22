import json
import numpy as np
import folium
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt
import webbrowser
import os
import time

# Load data from JSON files
def load_data(points_file, info_file):
    with open(points_file, 'r', encoding='utf-8') as f:
        points_data = json.load(f)
    with open(info_file, 'r', encoding='utf-8') as f:
        info_data = {point['id']: point for point in json.load(f)}
    return points_data, info_data

# Create adjacency matrix for the 209 points
def create_adjacency_matrix(points_data, info_data, alpha=0.5):
    import numpy as np
    
    matrix_size = 208  # Số lượng điểm
    adjacency_matrix = np.zeros((matrix_size, matrix_size))
    base_duration_matrix = np.zeros((matrix_size, matrix_size))  # Ma trận lưu `baseDuration`
    pedestrians_matrix = np.zeros((matrix_size, matrix_size))   # Ma trận lưu `pedestrians`
    motorcycles_matrix = np.zeros((matrix_size, matrix_size))   # Ma trận lưu `motorcycles`
    cars_matrix = np.zeros((matrix_size, matrix_size))          # Ma trận lưu `cars`

    # Map id của điểm tới chỉ số trong ma trận
    id_to_index = {point_id: idx for idx, point_id in enumerate(info_data.keys())}

    for point in points_data:
        from_id = point.get('id')  # ID của điểm xuất phát
        u = id_to_index.get(from_id)
        if u is None:  # Nếu không tìm thấy điểm trong `id_to_index`
            continue

        for distance in point['distances']:
            to_id = distance.get('id')  # ID của điểm đến
            v = id_to_index.get(to_id)
            if v is None:  # Nếu không tìm thấy điểm đến trong `id_to_index`
                continue

            # Lấy các giá trị cần thiết từ dữ liệu
            length = distance.get('length')
            duration = distance.get('duration', 1)  # Tránh chia cho 0
            base_duration = distance.get('baseDuration', duration)  # Mặc định là duration nếu không có baseDuration
            pedestrians = distance.get('pedestrians', 0)  # Mặc định 0 nếu không có giá trị
            motorcycles = distance.get('motorcycles', 0)  # Mặc định 0 nếu không có giá trị
            cars = distance.get('cars', 0)  # Mặc định 0 nếu không có giá trị

            # Công thức tính `capacity` dựa trên `alpha`
            capacity = alpha * length + (1 - alpha) * (length * base_duration / duration)

            # Gán giá trị vào các ma trận tương ứng
            adjacency_matrix[u][v] = capacity
            base_duration_matrix[u][v] = base_duration
            pedestrians_matrix[u][v] = pedestrians
            motorcycles_matrix[u][v] = motorcycles
            cars_matrix[u][v] = cars

    # Trả về tất cả các ma trận và mapping id_to_index
    return adjacency_matrix, base_duration_matrix, pedestrians_matrix, motorcycles_matrix, cars_matrix, id_to_index

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

    # Save scaled flow values to a txt file
    def save_flow_to_txt(self, filename="flow_values.txt"):
        with open(filename, 'w') as f:
            for u in self.flow_passed:
                for v in self.flow_passed[u]:
                    flow_value = self.flow_passed[u][v] / self.scaling_factor
                    if flow_value > 0:
                        f.write(f"Flow from {u} to {v}: {flow_value:.2f}\n")
        print(f"Flow values saved to {filename}")

# Initialize graph for Dinic's algorithm
def initialize_dinic_graph(adjacency_matrix, scaling_factor=1):
    n = len(adjacency_matrix)
    dinic_graph = Dinic(n, scaling_factor)
    for u in range(n):
        for v in range(n):
            if adjacency_matrix[u][v] > 0:
                dinic_graph.add_edge(u, v, adjacency_matrix[u][v])
    return dinic_graph

# Push-Relabel Algorithm for Maximum Flow
class PushRelabel:
    """
    Implements the Push-Relabel algorithm to compute maximum flow in a graph.
    """
    def __init__(self, n):
        self.n = n
        self.capacity = defaultdict(lambda: defaultdict(int))  # Stores capacities
        self.flow = defaultdict(lambda: defaultdict(int))  # Tracks flow between nodes
        self.height = [0] * n  # Node height (used for relabeling)
        self.excess = [0] * n  # Excess flow at each node
        self.adj = defaultdict(list)  # Adjacency list

    def add_edge(self, u, v, cap):
        """
        Adds an edge from node u to v with a specified capacity.
        """
        self.capacity[u][v] += cap
        self.adj[u].append(v)
        self.adj[v].append(u)

    def push(self, u, v):
        """
        Push flow from node u to v.
        Only pushes if there is excess at u and residual capacity to v.
        """
        send = min(self.excess[u], self.capacity[u][v] - self.flow[u][v])
        self.flow[u][v] += send
        self.flow[v][u] -= send
        self.excess[u] -= send
        self.excess[v] += send

    def relabel(self, u):
        """
        Update the height of node u to enable pushing flow.
        """
        min_height = float('inf')
        for v in self.adj[u]:
            if self.capacity[u][v] - self.flow[u][v] > 0:    # Residual capacity exists
                min_height = min(min_height, self.height[v])
        self.height[u] = min_height + 1 # Set height to minimum neighbor height + 1

    def discharge(self, u):
        """
        Pushes flow out of node u until its excess is zero or no valid pushes remain.
        """
        while self.excess[u] > 0:
            for v in self.adj[u]:
                if self.capacity[u][v] - self.flow[u][v] > 0 and self.height[u] > self.height[v]:
                    self.push(u, v)
                    if self.excess[u] == 0:  # Stop if no more excess
                        break   
            else:
                self.relabel(u)

    def max_flow(self, source, sink):
        """
        Compute the maximum flow from source to sink.
        """
        self.height[source] = self.n        # Source height set to number of nodes
        self.excess[source] = float('inf')  # Infinite flow pushed into the source
        for v in self.adj[source]:
            self.push(source, v)

        active_nodes = [i for i in range(self.n) if i != source and i != sink and self.excess[i] > 0]
        while active_nodes:
            u = active_nodes.pop(0)
            old_height = self.height[u]
            self.discharge(u)
            if self.height[u] > old_height:
                active_nodes.insert(0, u)

         # Total flow is the sum of flow leaving the source
        return sum(self.flow[source][v] for v in self.adj[source])

# Bellman-Ford Algorithm for Shortest Paths
def bellman_ford(adjacency_matrix, source):
    """
    Compute shortest paths using the Bellman-Ford algorithm.
    The edge weights are reciprocals of capacities.
    """
    n = len(adjacency_matrix)
    distances = [float('inf')] * n
    predecessors = [-1] * n

    # Initialize distance to source as 0
    distances[source] = 0

    # Relax edges repeatedly
    for _ in range(n - 1):
        for u in range(n):
            for v in range(n):
                if adjacency_matrix[u][v] > 0:  # Edge exists
                    weight = 1 / adjacency_matrix[u][v]  # Use reciprocal of capacity as weight
                    if distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        predecessors[v] = u

    # Check for negative-weight cycles
    for u in range(n):
        for v in range(n):
            if adjacency_matrix[u][v] > 0:
                weight = 1 / adjacency_matrix[u][v]
                if distances[u] + weight < distances[v]:
                    raise ValueError("Graph contains a negative-weight cycle")

    return distances, predecessors

# Plot the points, edges, and flow values on a map using folium
def plot_graph_on_map_Dinic(info_data, 
                            adjacency_matrix, 
                            dinic_graph, 
                            id_to_index, 
                            source, 
                            sink, 
                            base_duration_matrix,
                            pedestrians_matrix, 
                            motorcycles_matrix,
                            cars_matrix):
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
        lat, lng, title , address= point_info['position']['lat'], point_info['position']['lng'], point_info['title'], point_info['address']['label']

        if idx in points_with_flow:  # Chỉ vẽ điểm nếu có dòng chảy qua
            if idx == source:
                color = 'green'  # Màu đỏ cho nguồn
            elif idx == sink:
                color = 'black'  # Màu tím cho đích
            else:
                color = 'blue'  # Màu đỏ cho các điểm có dòng chảy qua

            folium.Marker(
            location=[lat, lng],
            popup=f"<b>{title}</b><br>{address}",  # Hiển thị thông tin khi click vào marker
            tooltip=title,  # Hiển thị thông tin khi di chuột qua
            icon=folium.Icon(color=color, icon="info-sign")  # Biểu tượng marker
        ).add_to(map_)

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

                    # Xác định baseDuration từ ma trận base_duration_matrix
                    base_duration = base_duration_matrix[u][v]
                    pedestrians = pedestrians_matrix[u][v]
                    motorcycles = motorcycles_matrix[u][v]
                    cars = cars_matrix[u][v]
                    # Tính trọng số tổng hợp
                    total_weight = pedestrians * 1 + motorcycles * 3 + cars * 5
                    # Xác định độ dày của đường đi dựa trên baseDuration
                    if total_weight < 300:
                        weight = 3
                        color = "blue"
                    elif 300 <= total_weight <= 500:
                        weight = 6
                        color = "green"
                    elif 500 < total_weight <= 700:
                        weight = 9
                        color = "orange"
                    else:
                        weight = 12
                        color = "red"

                    # Vẽ PolyLine với màu và độ dày phụ thuộc vào dòng chảy
                    folium.PolyLine(
                        locations=[[from_lat, from_lng], [to_lat, to_lng]], 
                        color=color, 
                        weight=weight, 
                        opacity=0.6
                    ).add_to(map_)
                    
    # Save or show the map
    map_.save("output/flow_network_map_by_Dinic_Algorithm.html")
    print("Map saved as output/flow_network_map_by_Dinic_Algorithm.html")

def plot_graph_on_map_PushRelabel(info_data, 
                                  adjacency_matrix, 
                                  pr_graph, 
                                  id_to_index, 
                                  source, sink, 
                                  distances, 
                                  pedestrians_matrix, 
                                  motorcycles_matrix,
                                  cars_matrix):
    import os
    import folium

    map_center = [10.789990, 106.678101]  # Center map around a specific location
    map_ = folium.Map(location=map_center, zoom_start=13)

    # Set để lưu trữ các điểm có dòng chảy
    points_with_flow = set()

    # Kiểm tra các cạnh và lưu trữ các điểm có dòng chảy qua
    for u in range(len(adjacency_matrix)):
        for v in range(len(adjacency_matrix)):
            if adjacency_matrix[u][v] > 0:  # Kiểm tra nếu có cạnh giữa u và v
                flow_value = pr_graph.flow[u][v]

                if flow_value > 0:  # Nếu có dòng chảy
                    points_with_flow.add(u)  # Thêm điểm u vào set
                    points_with_flow.add(v)  # Thêm điểm v vào set

    # Vẽ các marker (chỉ với các điểm có dòng chảy)
    for point_id, point_info in info_data.items():
        idx = id_to_index[point_id]
        lat, lng, title, address = point_info['position']['lat'], point_info['position']['lng'], point_info['title'], point_info['address']['label']

        if idx in points_with_flow:  # Chỉ vẽ điểm nếu có dòng chảy qua
            if idx == source:
                color = 'green'  # Màu xanh lá cho nguồn
            elif idx == sink:
                color = 'black'  # Màu đen cho đích
            else:
                color = 'blue'  # Màu xanh dương cho các điểm khác

            folium.Marker(
                location=[lat, lng],
                popup=f"<b>{title}</b><br>{address}",  # Hiển thị thông tin khi click vào marker
                tooltip=title,  # Hiển thị thông tin khi di chuột qua
                icon=folium.Icon(color=color, icon="info-sign")  # Biểu tượng marker
            ).add_to(map_)

    # Vẽ các cạnh (chỉ khi có dòng chảy)
    for u in range(len(adjacency_matrix)):
        for v in range(len(adjacency_matrix)):
            if adjacency_matrix[u][v] > 0:  # Kiểm tra nếu có cạnh giữa u và v
                flow_value = pr_graph.flow[u][v]

                if flow_value > 0:  # Vẽ đường đi chỉ khi có dòng chảy
                    from_point = info_data[list(info_data.keys())[u]]
                    to_point = info_data[list(info_data.keys())[v]]
                    from_lat, from_lng = from_point['position']['lat'], from_point['position']['lng']
                    to_lat, to_lng = to_point['position']['lat'], to_point['position']['lng']
                    pedestrians = pedestrians_matrix[u][v]
                    motorcycles = motorcycles_matrix[u][v]
                    cars = cars_matrix[u][v]
                    # Tính trọng số tổng hợp
                    total_weight = pedestrians * 1 + motorcycles * 3 + cars * 5
                    # Độ dày của đường đi phụ thuộc vào giá trị dòng chảy
                    if total_weight < 300:
                        weight = 3
                        color = "blue"
                    elif 300 <= total_weight <= 500:
                        weight = 6
                        color = "green"
                    elif 500 < total_weight <= 700:
                        weight = 9
                        color = "orange"
                    else:
                        weight = 12
                        color = "red"
                        
                    folium.PolyLine(
                        locations=[[from_lat, from_lng], [to_lat, to_lng]], 
                        color=color, 
                        weight=weight, 
                        opacity=0.6
                    ).add_to(map_)

    # Lưu hoặc hiển thị bản đồ
    os.makedirs("output", exist_ok=True)
    map_.save("output/flow_network_map_by_Push_relabel.html")
    print("Map saved as output/flow_network_map_by_Push_relabel.html")

# Main execution
# Main logic
if __name__ == "__main__":
    points_file = './data/208Points.json'
    info_file = './data/info_point.json'

    # Load data and create adjacency matrix with estimated capacities
    points_data, info_data = load_data(points_file, info_file)
    alpha = 0.5  # Set alpha based on the importance of length vs. congestion
    adjacency_matrix, base_duration_matrix, pedestrians_matrix, motorcycles_matrix, cars_matrix, id_to_index = create_adjacency_matrix(points_data, info_data, alpha)

    # Set scaling factor (e.g., 10 to reduce flow values by a factor of 10)
    scaling_factor = 1

    # Input source and sink points from the user
    while True:
        source = int(input("Enter the source point ID (1-208, e.g., 102): ").strip())
        if 1 <= source <= 208:
            break
        print("Invalid input. Please enter a valid source ID between 1 and 208.")

    while True:
        sink = int(input("Enter the sink point ID (1-208, e.g., 188): ").strip())
        if 1 <= sink <= 208:
            break
        print("Invalid input. Please enter a valid sink ID between 1 and 208.")

    # Allow the user to select the algorithm
    print("\nChoose the algorithm to compute the maximum flow:")
    print("1. Dinic's Algorithm")
    print("2. Push-Relabel Algorithm")
    algorithm_choice = int(input("Enter your choice (1 or 2): ").strip())

    if algorithm_choice == 1:
        # Run Dinic's algorithm
        start_time = time.time()
        dinic_graph = initialize_dinic_graph(adjacency_matrix, scaling_factor)
        max_flow_value = dinic_graph.max_flow(source, sink) / scaling_factor
        end_time = time.time()

        print(f"Maximum flow from {source} to {sink} (Dinic, scaled): {max_flow_value:.2f}")
        print(f"Execution time (Dinic): {end_time - start_time:.4f} seconds")

        # Plot the graph
        map_path = plot_graph_on_map_Dinic(info_data, 
                                           adjacency_matrix, 
                                           dinic_graph, id_to_index, 
                                           source, sink, 
                                           base_duration_matrix, 
                                           pedestrians_matrix, 
                                           motorcycles_matrix,
                                           cars_matrix)
        dinic_graph.save_flow_to_txt("flow_values.txt")

    elif algorithm_choice == 2:
        # Run Push-Relabel algorithm
        pushRelabel_graph = PushRelabel(len(adjacency_matrix))
        for u in range(len(adjacency_matrix)):
            for v in range(len(adjacency_matrix)):
                if adjacency_matrix[u][v] > 0:
                    pushRelabel_graph.add_edge(u, v, adjacency_matrix[u][v])

        start_time = time.time()
        max_flow_value = pushRelabel_graph.max_flow(source, sink)
        end_time = time.time()

        print(f"Maximum flow from {source} to {sink} (Push-Relabel): {max_flow_value:.2f}")
        print(f"Execution time (Push-Relabel): {end_time - start_time:.4f} seconds")

        # Compute distances and plot the graph
        distances, predecessors = bellman_ford(adjacency_matrix, source)
        plot_graph_on_map_PushRelabel(info_data, 
                                      adjacency_matrix, 
                                      pushRelabel_graph, 
                                      id_to_index, source, 
                                      sink, distances, 
                                      pedestrians_matrix, 
                                      motorcycles_matrix,
                                        cars_matrix)

    else:
        print("Invalid choice. Please select 1 or 2.")
