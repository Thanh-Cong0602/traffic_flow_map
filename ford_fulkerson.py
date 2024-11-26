import numpy as np
import json
import pandas as pd
import folium

def ford_fulkerson_with_bottlenecks(graph, source, sink):
    max_flow = 0
    augmenting_path_count = 0  # Counter for augmenting paths
    bottlenecks = []  # List to store bottleneck edges
    edge_usage = {}
    n = len(graph)

    
    # Initialize flow matrix
    flow_matrix = [[0] * n for _ in range(n)]

    while True:
        # Initialize visited list and parent tracking
        visited = [False] * len(graph)
        parent = [-1] * len(graph)
        queue = []

        # Begin BFS from the source node
        queue.append(source)
        visited[source] = True

        # Perform BFS to find an augmenting path
        while queue:
            u = queue.pop(0)
            for v in range(len(graph[u])):
                # If there is a path from u to v and v is not yet visited
                if not visited[v] and graph[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    queue.append(v)
                    # Stop BFS if we reach the sink
                    if v == sink:
                        break

        # If sink is not reached, there is no more augmenting path
        if not visited[sink]:
            break

        # Find the minimum capacity in the path filled by BFS
        flow = float("inf")
        v = sink
        bottleneck_edge = None  # To track the bottleneck edge
        while v != source:
            u = parent[v]
            if graph[u][v] < flow:
                flow = graph[u][v]
                bottleneck_edge = (u, v)  # Update bottleneck edge

            if (u, v) in edge_usage:
                edge_usage[(u, v)] += 1
            else:
                edge_usage[(u, v)] = 1
            v = u

        # Save the bottleneck edge for this augmenting path
        if bottleneck_edge:
            bottlenecks.append(bottleneck_edge)

        # Update residual capacities of the edges and reverse edges along the path
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= flow
            graph[v][u] += flow
            flow_matrix[u][v] += flow  # Record flow along the edge
            flow_matrix[v][u] -= flow  # Reverse flow for the residual graph
            v = u

        # Add flow to the overall flow
        max_flow += flow
        augmenting_path_count += 1  # Increment the count for each augmenting path
    frequently_used_edges = {edge: count for edge, count in edge_usage.items() if count >= 5}

    return max_flow, augmenting_path_count, bottlenecks, frequently_used_edges, flow_matrix


def init_graph():  
    src_lst = []
    des_lst = []
    avg_v_lst = []
    id_lst = []
    
    with open('./data/208Points.json', encoding='utf-8') as file:
      data = json.load(file)
    for point in data:
        src_idx = point['fromIndex']
        id = point['id']
        for edge in point['distances']:
            des_idx = edge['toIndex']
            avg_v = edge['length'] / edge['baseDuration'] * 83368
    
            src_lst.append(src_idx)
            des_lst.append(des_idx)
            avg_v_lst.append(avg_v)
            id_lst.append(id)
    
    data = pd.DataFrame.from_dict({
        'src' : src_lst,
        'des' : des_lst,
        'avg_v' : avg_v_lst,
        'id' : id_lst
    })

    np_graph = np.zeros((209, 209))
    for _, row in data.iterrows():
        np_graph[int(row['src'] - 1)][int(row['des'] - 1)] = row['avg_v']
    initial_graph = np_graph.copy()
    
    return np_graph, initial_graph

def export_json_result(flow_matrix, initial_graph):
    with open('./data/208Points.json', 'r', encoding='utf-8') as f:
        result_dict = json.load(f)
    points_with_flow = set()
    map_center = [10.789990, 106.678101]
    map_ = folium.Map(location=map_center, zoom_start=13)
    for idx_point1 in range(len(result_dict)):
        try:
            point1 = result_dict[idx_point1]
            u_index = point1['fromIndex']
            for idx_point2 in range(len(point1['distances'])):
                point2 = point1['distances'][idx_point2]
                v_index = point2['toIndex']
                result_dict[idx_point1]['distances'][idx_point2]['flow'] = flow_matrix[u_index - 1][v_index - 1]
                result_dict[idx_point1]['distances'][idx_point2]['capacity'] = initial_graph[u_index - 1][v_index -1]
    
            result_dict[idx_point1]['is_source'] = True if u_index == source else False
            result_dict[idx_point1]['is_sink'] = True if u_index == sink else False
        except:
            print(u_index, v_index)
    with open(f"result_{source}_{sink}.json", "w") as outfile: 
        json.dump(result_dict, outfile)

if __name__ == "__main__":
    np_graph, initial_graph = init_graph()
    
    source = 50
    sink = 150
    max_flow, augmenting_path_count, bottlenecks, frequently_used_edges, flow_matrix = ford_fulkerson_with_bottlenecks(np_graph, source , sink)
    
    export_json_result(flow_matrix, initial_graph)
    
