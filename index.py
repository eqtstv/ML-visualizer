import sys

import ml_visualizer.callbacks
from ml_visualizer.app import app, config, server
from ml_visualizer.layout import app_layout

app.layout = app_layout


if __name__ == "__main__":
    server.run(host=f"{config['ip']}", port=f"{config['port']}", debug=True)
