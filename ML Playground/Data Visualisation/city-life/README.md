# 15 Minute City Map

This project is a Flask web application that visualizes routes for walking, biking, and driving within a 15-minute travel distance in the Eindhoven area. The map is restricted to a 30km square around Eindhoven to ensure all calculations and visualizations are localized.

## Features

- Interactive map centered on Eindhoven, Netherlands
- Visualize routes for walking, biking, and driving
- Routes are color-coded for easy differentiation
- Map panning and zooming are restricted to a specific area around Eindhoven
- "Walk" routes are always displayed on top of "bike" and "drive" routes

## Technologies Used

- Flask
- OSMnx
- NetworkX
- Leaflet.js

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/15-minute-city-map.git
    cd 15-minute-city-map
    ```

2. **Create a virtual environment and activate it:**

    ```sh
    mkdir <name>
    cd <name>
    conda create --prefix ./env python=3.11    # choose python version
    conda activate ./env
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Flask application:**

    ```sh
    python app.py
    ```

5. **Open your web browser and go to:**

    ```
    http://127.0.0.1:5000/
    ```

## Usage

- Double-click on the map to set a starting location. The routes for walking, biking, and driving will be calculated and displayed.
- Use the control buttons on the top-right to filter the routes:
  - "Clear Map": Clear all routes from the map
  - "All": Show all routes
  - "Walk": Show only walking routes
  - "Bike": Show only biking routes
  - "Drive": Show only driving routes

## Project Structure

- `app.py`: The main Flask application file.
- `templates/map.html`: The HTML file for the map visualization.
- `requirements.txt`: The Python dependencies required to run the application.