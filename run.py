import ml_visualizer.dash_app.callbacks
from ml_visualizer.app import config, server
from ml_visualizer.dash_app.dash_app import dash_app

if __name__ == "__main__":
    server.run(host=f"{config['ip']}", port=f"{config['port']}", debug=True)
