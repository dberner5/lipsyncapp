from flask import Flask, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s : %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@app.route('/health')
def health():
    logger.debug('Health check requested')
    return jsonify({"status": "healthy"})

@app.route('/api/hello')
def hello():
    logger.debug('Hello endpoint requested')
    return jsonify({"message": "Hello World!"})

if __name__ == '__main__':
    logger.info('Starting Flask development server...')
    # Disable threading to see if it helps with request handling
    app.run(host='0.0.0.0', port=5000, threaded=False, debug=True) 