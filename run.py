import os
import sys

import waitress

import ml_visualizer.dash_app.callbacks
from ml_visualizer.app import config, server
from ml_visualizer.dash_app.dash_app import dash_app

if __name__ == "__main__":
    if sys.argv[1] == "debug":
        server.run(host=config["ip"], port=config["port"], debug=True)

    if sys.argv[1] == "production":
        server.debug = False
        port = int(os.environ.get("PORT", config["port"]))
        waitress.serve(server, port=port)
