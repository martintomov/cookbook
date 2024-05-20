from flask import Flask, render_template, request, jsonify
import osmnx as ox
import networkx as nx
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('map.html')

@app.route('/update_routes', methods=['GET'])
def update_routes():
    lat = float(request.args.get('lat'))
    lng = float(request.args.get('lng'))
    start_location = (lat, lng)
    
    logging.debug(f"Received request to update routes from lat: {lat}, lng: {lng}")
    
    # Define travel speeds in km/h
    travel_speed = {'walk': 5, 'bike': 20, 'drive': 40}
    
    # Define colors for each transport mode
    transport_colors = {'walk': 'blue', 'bike': 'orange', 'drive': 'purple'}
    
    routes = []
    
    for mode in ['walk', 'bike', 'drive']:
        G = ox.graph_from_point(start_location, network_type=mode, dist=10000, simplify=True)
        center_node = ox.distance.nearest_nodes(G, lng, lat)
        max_distance = travel_speed[mode] * 1000 / 60 * 15  # distance in meters for 15 minutes
        subgraph = nx.ego_graph(G, center_node, radius=max_distance, distance='length')
        
        for node1, node2, data in subgraph.edges(data=True):
            routes.append({
                'coordinates': [(G.nodes[node1]['y'], G.nodes[node1]['x']), (G.nodes[node2]['y'], G.nodes[node2]['x'])],
                'color': transport_colors[mode]
            })
    
    logging.debug(f"Generated {len(routes)} routes.")
    
    return jsonify({'routes': routes})

if __name__ == '__main__':
    app.run(debug=True)