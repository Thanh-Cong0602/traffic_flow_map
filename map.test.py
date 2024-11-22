import json
import folium

def plot_random_points_with_connections(random_points, output_file="random_points_map_with_connections.html"):
    """
    Hàm vẽ bản đồ với các điểm từ random_points và hiển thị các đường nối giữa chúng với độ đậm nhạt.

    :param random_points: Danh sách các điểm với thông tin như title, address, position.
    :param output_file: Tên file HTML để lưu bản đồ.
    """
    
    # Xác định tâm bản đồ (sử dụng tọa độ trung bình của tất cả các điểm)
    avg_lat = sum(point['position']['lat'] for point in random_points) / len(random_points)
    avg_lng = sum(point['position']['lng'] for point in random_points) / len(random_points)
    map_center = [avg_lat, avg_lng]

    # Khởi tạo bản đồ Folium
    map_ = folium.Map(location=map_center, zoom_start=13)

    # Thêm các điểm vào bản đồ
    for point in random_points:
        title = point['title']
        address = point['address']['label']
        lat = point['position']['lat']
        lng = point['position']['lng']

        # Thêm marker vào bản đồ với thông tin tiêu đề và địa chỉ
        folium.Marker(
            location=[lat, lng],
            popup=f"<b>{title}</b><br>{address}",  # Hiển thị thông tin khi click vào marker
            tooltip=title,  # Hiển thị thông tin khi di chuột qua
            icon=folium.Icon(color="blue", icon="info-sign")  # Biểu tượng marker
        ).add_to(map_)

    # Vẽ các đường nối giữa các điểm với độ đậm nhạt dựa trên khoảng cách
    for i in range(len(random_points) - 1):
        from_point = random_points[i]
        to_point = random_points[i + 1]

        # Tọa độ
        from_lat, from_lng = from_point['position']['lat'], from_point['position']['lng']
        to_lat, to_lng = to_point['position']['lat'], to_point['position']['lng']

        # Khoảng cách giữa hai điểm (giả sử trường `distance` có sẵn)
        distance = from_point['distance']  # Sử dụng khoảng cách từ `random_points`

        # Tùy chỉnh độ dày và màu sắc dựa trên khoảng cách
        if distance < 500:
            weight = 3
            color = "green"
        elif 500 <= distance <= 750:
            weight = 6
            color = "red"
        elif 750 <= distance <= 1000:
            weight = 9
            color = "blue"
        else:
            weight = 12
            color = "orange"

        # Vẽ đường nối
        folium.PolyLine(
            locations=[[from_lat, from_lng], [to_lat, to_lng]],
            color=color,
            weight=weight,
            opacity=0.6
        ).add_to(map_)

    # Lưu bản đồ vào file HTML
    map_.save(output_file)
    print(f"Bản đồ đã được lưu vào {output_file}")


def read_json(input_file):
    """
    Hàm đọc dữ liệu từ tệp JSON.

    :param input_file: Đường dẫn đến tệp JSON.
    :return: Danh sách các điểm đã được đọc từ tệp JSON.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    # Đọc dữ liệu từ tệp JSON
    input_file = './data/point_flow.json'  # Đường dẫn đến tệp JSON đầu vào
    output_file = 'random_points_map_with_connections.html'  # Đường dẫn đến tệp HTML đầu ra

    # Đọc danh sách điểm từ file JSON
    random_points = read_json(input_file)

    # Vẽ bản đồ và lưu vào tệp HTML
    plot_random_points_with_connections(random_points, output_file)
