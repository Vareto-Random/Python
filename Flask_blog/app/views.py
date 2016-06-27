from app import app
"""
views are the handlers that respond to requests from web browsers or other clients
"""

@app.route('/')
@app.route('/index')
@app.route('/vareto')
def index():
    return "Hello, World!"