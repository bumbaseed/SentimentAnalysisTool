from flask import Flask
from routes.main_routes import main
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # secret key for session management

# Register the blueprint
app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)
