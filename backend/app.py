from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
import logging
import os
from werkzeug.utils import secure_filename
from pyannote.audio import Pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
import json
from datetime import datetime
import requests
import torchaudio
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
import subprocess
from pathlib import Path
import time


# Configure logging first
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s : %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Verify .env loading
if os.path.exists('.env'):
    logger.info("Found .env file")
else:
    logger.warning("No .env file found in current directory")

logger.info(f"Environment variables loaded. HF token present: {bool(os.getenv('HUGGING_FACE_TOKEN'))}")

app = Flask(__name__)
CORS(app)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'splits'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'results'), exist_ok=True)

# Configure rhubarb path
RHUBARB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'executables', 'Rhubarb-Lip-Sync-1.13.0-Linux', 'rhubarb')
logger.info(f"Rhubarb path: {RHUBARB_PATH}")

# Ensure rhubarb has execute permissions
try:
    os.chmod(RHUBARB_PATH, 0o755)
    logger.info("Set execute permissions for rhubarb")
except Exception as e:
    logger.error(f"Failed to set execute permissions for rhubarb: {e}")

# Initialize pyannote pipeline
pipeline = None
hf_token = os.getenv('HUGGING_FACE_TOKEN')

# Add at the top with other global variables
PROGRESS_DATA = {}

def init_pipeline():
    global pipeline
    try:
        logger.info("Initializing pyannote pipeline...")
        if not hf_token:
            logger.error("No Hugging Face token found in environment variables")
            return "No Hugging Face token found. Please check your .env file"
            
        logger.debug(f"Using token: {hf_token[:4]}...{hf_token[-4:]}")
        
        # First verify the token using the model API
        headers = {"Authorization": f"Bearer {hf_token}"}
        api_url = "https://huggingface.co/api/models/pyannote/speaker-diarization-3.1"
        
        logger.debug(f"Verifying token with URL: {api_url}")
        response = requests.get(api_url, headers=headers)
        logger.debug(f"Token verification response status: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            logger.debug(f"Response body: {response_json}")
        except:
            logger.debug("No JSON response body")
        
        if response.status_code == 401:
            logger.error("Invalid Hugging Face token")
            return "Invalid Hugging Face token. Please check your credentials."
        elif response.status_code == 403:
            logger.error("Need to accept the model's license agreement")
            return "Please accept the model's license agreement at https://huggingface.co/pyannote/speaker-diarization-3.1"
        elif response.status_code != 200:
            logger.error(f"Unexpected status code: {response.status_code}")
            return f"Unexpected error when verifying token. Status code: {response.status_code}"
        
        # If we get here, token is valid, try to load the pipeline
        logger.info("Token verified successfully, loading pipeline...")
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            logger.info("Successfully initialized pyannote pipeline")
            return None
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            logger.exception("Pipeline loading error:")
            return f"Failed to load pipeline: {str(e)}"
            
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.exception("Full stack trace:")
        return str(e)

# Try to initialize the pipeline
pipeline_error = init_pipeline()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def split_audio(audio_path, diarization):
    """Split audio file based on diarization results, maintaining original length with silence"""
    audio = AudioSegment.from_file(audio_path)
    splits = []
    
    # Group segments by speaker
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append({
            'start': int(turn.start * 1000),  # convert to milliseconds
            'end': int(turn.end * 1000)
        })
    
    # Create a separate audio file for each speaker
    splits_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'splits')
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Get the full duration of the original audio
    full_duration = len(audio)
    
    for speaker, segments in speaker_segments.items():
        # Create a silent audio segment of the same length as the original
        silent_audio = AudioSegment.silent(duration=full_duration)
        
        # Overlay each segment from this speaker at its original position
        for segment in segments:
            segment_audio = audio[segment['start']:segment['end']]
            silent_audio = silent_audio.overlay(segment_audio, position=segment['start'])
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{base_name}_{speaker}_{timestamp}.wav"
        output_path = os.path.join(splits_dir, output_filename)
        
        # Export audio file
        silent_audio.export(output_path, format='wav')
        splits.append({
            'speaker': speaker,
            'filename': output_filename,
            'segments': segments  # Include timing information
        })
    
    return splits

def convert_to_wav(input_path):
    """Convert audio file to WAV format"""
    logger.info(f"Converting {input_path} to WAV format")
    try:
        # Load audio file using pydub
        audio = AudioSegment.from_file(input_path)
        
        # Generate WAV filename
        wav_path = os.path.splitext(input_path)[0] + '.wav'
        
        # Export as WAV
        audio.export(wav_path, format='wav')
        logger.info(f"Successfully converted to {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        raise

def process_audio_file(file_path):
    """Process audio file using torchaudio and return diarization"""
    logger.info(f"Processing audio file: {file_path}")
    
    # Convert to WAV if not already
    if not file_path.lower().endswith('.wav'):
        file_path = convert_to_wav(file_path)
    
    # Load audio using torchaudio
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    diarization = None

    with ProgressHook() as hook:
        # Process with pipeline
        diarization = pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate,
            "num_speakers": 2,
            "hook": hook
        })
    
    return diarization

@app.route('/health')
def health():
    logger.debug('Health check requested')
    return jsonify({"status": "healthy"})

@app.route('/api/hello')
def hello():
    logger.debug('Hello endpoint requested')
    return jsonify({"message": "Hello World!"})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({
            "message": "File uploaded successfully",
            "filename": filename
        })
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/diarize/<filename>', methods=['POST'])
def diarize_audio(filename):
    if pipeline_error:
        return jsonify({
            "error": pipeline_error,
            "type": "initialization_error"
        }), 500
    
    if not pipeline:
        return jsonify({
            "error": "Diarization pipeline not initialized. Please check server logs.",
            "type": "initialization_error"
        }), 500
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
    
    try:
        # Process audio file
        diarization = process_audio_file(file_path)
        
        # Split audio into separate files
        splits = split_audio(file_path, diarization)
        
        return jsonify({
            "message": "Diarization completed successfully",
            "splits": splits
        })
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        logger.exception("Full stack trace:")
        return jsonify({
            "error": str(e),
            "type": "processing_error"
        }), 500

@app.route('/api/audio/<filename>')
def get_audio(filename):
    # Check if file is in splits directory first
    splits_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'splits')
    if os.path.exists(os.path.join(splits_dir, filename)):
        return send_from_directory(splits_dir, filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/files')
def list_files():
    files = []
    # List main uploads
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)) and allowed_file(filename):
            files.append({
                "filename": filename,
                "type": "original"
            })
    
    # List split files
    splits_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'splits')
    if os.path.exists(splits_dir):
        for filename in os.listdir(splits_dir):
            if allowed_file(filename):
                files.append({
                    "filename": filename,
                    "type": "split"
                })
    
    return jsonify({"files": files})

@app.route('/api/process-rhubarb/<filename>', methods=['POST'])
def process_rhubarb(filename):
    """Process a split audio file with rhubarb to generate lip sync data"""
    logger.info(f"Processing {filename} with rhubarb")
    
    def generate():
        # Initialize progress tracking
        progress_data = {
            'progress': 0,
            'status': 'processing'
        }
        
        # Send initial progress
        yield f"data: {json.dumps(progress_data)}\n\n"
        
        # Verify the file exists in splits directory
        splits_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'splits')
        input_path = os.path.join(splits_dir, filename)
        
        if not os.path.exists(input_path):
            error_data = {'status': 'error', 'error': 'File not found'}
            yield f"data: {json.dumps(error_data)}\n\n"
            return
        
        # Prepare output path
        results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
        output_filename = f"{Path(filename).stem}_rhubarb.json"
        output_path = os.path.join(results_dir, output_filename)
        
        try:
            # Run rhubarb command
            command = [
                RHUBARB_PATH,
                "-f", "json",
                input_path,
                "-o", output_path
            ]
            
            logger.debug(f"Running command: {' '.join(command)}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read stderr in real-time to track progress
            for line in process.stderr:
                logger.debug(f"Rhubarb output: {line.strip()}")
                if "Generating output" in line:
                    progress_data['progress'] = 90
                elif "Analyzing audio" in line:
                    progress_data['progress'] = 30
                elif "Detecting speech" in line:
                    progress_data['progress'] = 60
                yield f"data: {json.dumps(progress_data)}\n\n"
            
            process.wait()
            
            if process.returncode != 0:
                error_data = {'status': 'error', 'error': 'Rhubarb processing failed'}
                yield f"data: {json.dumps(error_data)}\n\n"
                return
            
            # Read and return the generated JSON
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    rhubarb_data = json.load(f)
                
                success_data = {
                    'status': 'done',
                    'progress': 100,
                    'data': rhubarb_data
                }
                yield f"data: {json.dumps(success_data)}\n\n"
            else:
                error_data = {'status': 'error', 'error': 'Rhubarb output file not generated'}
                yield f"data: {json.dumps(error_data)}\n\n"
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing file with rhubarb: {error_msg}")
            logger.exception("Full stack trace:")
            error_data = {'status': 'error', 'error': error_msg}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/results/<filename>')
def get_result(filename):
    """Retrieve a rhubarb result file"""
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'results'), filename)

@app.route('/api/results')
def list_results():
    """List all rhubarb result files"""
    results = []
    results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                results.append(filename)
    return jsonify({"results": results})

@app.route('/api/generate-video/<filename>', methods=['POST'])
def generate_video(filename):
    """Return lip sync timing data from rhubarb results"""
    logger.info(f"Getting lip sync data for {filename}")
    
    # Get the rhubarb results
    results_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
    rhubarb_file = os.path.join(results_dir, filename)
    
    if not os.path.exists(rhubarb_file):
        logger.error(f"Rhubarb file not found: {rhubarb_file}")
        return jsonify({"error": "Rhubarb results not found"}), 404
    
    try:
        # Read and return the rhubarb data
        with open(rhubarb_file, 'r') as f:
            rhubarb_data = json.load(f)
            logger.info("Successfully loaded rhubarb data")
        
        return jsonify({
            "message": "Lip sync data retrieved successfully",
            "mouthCues": rhubarb_data['mouthCues']
        })
        
    except Exception as e:
        logger.error(f"Error retrieving lip sync data: {e}")
        logger.exception("Full stack trace:")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info('Starting Flask development server...')
    app.run(host='0.0.0.0', port=5000, threaded=False, debug=True) 