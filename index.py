import callbacks
from app import app
from layout import app_layout

app.layout = app_layout


if __name__ == "__main__":
    app.run_server(debug=True)
