from flask import Flask, jsonify, request, send_from_directory, Response, send_file
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
import tempfile
import base64
from urllib.parse import unquote
import fnmatch
import fcntl
import errno

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
os.makedirs(os.path.join(UPLOAD_FOLDER, 'lipsync'), exist_ok=True)

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

def process_audio_file(file_path, num_speakers):
    """Process audio file using torchaudio and return diarization"""
    logger.info(f"Processing audio file: {file_path} with {num_speakers} speakers")
    
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
            "num_speakers": num_speakers,
            "hook": hook
        })
    
    return diarization

def clean_directory(directory, pattern=None, keep_latest=None):
    """Clean a directory by removing all files except the latest ones matching the pattern.
    
    Args:
        directory (str): Directory to clean
        pattern (str, optional): Glob pattern to match files. If None, matches all files.
        keep_latest (str, optional): Base filename to keep latest version of. If None, keeps latest of all files.
    """
    if not os.path.exists(directory):
        return

    # Get all files in directory
    files = []
    for f in os.listdir(directory):
        if pattern and not fnmatch.fnmatch(f, pattern):
            continue
        
        full_path = os.path.join(directory, f)
        if os.path.isfile(full_path):
            files.append((full_path, os.path.getmtime(full_path)))
    
    if not files:
        return

    # Sort by modification time, newest first
    files.sort(key=lambda x: x[1], reverse=True)
    
    # If keep_latest is specified, keep only the latest file with that base name
    # Otherwise, keep only the latest file overall
    if keep_latest:
        kept_files = set()
        for file_path, _ in files:
            base_name = os.path.basename(file_path)
            if base_name.startswith(keep_latest):
                kept_files.add(file_path)
                break
        
        # Delete all other files
        for file_path, _ in files:
            if file_path not in kept_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove file {file_path}: {e}")
    else:
        # Keep only the latest file
        latest_file = files[0][0]
        for file_path, _ in files[1:]:
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up old file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove file {file_path}: {e}")

def clean_all_uploads():
    """Clean all files in uploads directory and its subdirectories"""
    # Clean main uploads directory
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove file {file_path}: {e}")

    # Clean splits directory
    splits_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'splits')
    if os.path.exists(splits_dir):
        for filename in os.listdir(splits_dir):
            file_path = os.path.join(splits_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up split file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove split file {file_path}: {e}")

    # Clean lipsync directory
    lipsync_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'lipsync')
    if os.path.exists(lipsync_dir):
        for filename in os.listdir(lipsync_dir):
            file_path = os.path.join(lipsync_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up lipsync file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove lipsync file {file_path}: {e}")

@app.route('/health')
def health():
    logger.debug('Health check requested')
    return jsonify({"status": "healthy"})

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
        
        # Clean all files in uploads directory and its subdirectories
        clean_all_uploads()
        
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
        # Get number of speakers from request
        data = request.get_json()
        num_speakers = data.get('numSpeakers', 2)  # Default to 2 if not specified
        
        # Clean up old split files before generating new ones
        base_name = os.path.splitext(filename)[0]
        splits_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'splits')
        clean_directory(splits_dir, pattern=f"{base_name}_*.wav")
        
        # Process audio file
        diarization = process_audio_file(file_path, num_speakers)
        
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
        
        try:
            # Get the input file path
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(input_path):
                # Check if file exists in splits directory
                splits_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'splits')
                input_path = os.path.join(splits_dir, filename)
                if not os.path.exists(input_path):
                    error_data = {'status': 'error', 'error': 'File not found'}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return

            # Convert to WAV if not already
            if not input_path.lower().endswith('.wav'):
                try:
                    input_path = convert_to_wav(input_path)
                    logger.info(f"Converted to WAV: {input_path}")
                except Exception as e:
                    error_data = {'status': 'error', 'error': f'Failed to convert to WAV: {str(e)}'}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    return
            
            # Clean up old results before generating new ones
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            lipsync_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'lipsync')
            clean_directory(lipsync_dir, pattern=f"{base_name}_*.json")
            
            # Prepare output path
            output_filename = f"{base_name}_rhubarb.json"
            output_path = os.path.join(lipsync_dir, output_filename)
            
            # Run rhubarb command
            command = [
                RHUBARB_PATH,
                "-f", "json",
                "--machineReadable",
                input_path,
                "-o", output_path
            ]
            
            logger.debug(f"Running command: {' '.join(command)}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffering
                universal_newlines=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Force Python to disable buffering
            )
            
            # Read stderr in real-time to track progress
            logger.info("Starting to read Rhubarb output...")
            
            # Set non-blocking mode for stderr
            stderr_fd = process.stderr.fileno()
            flags = fcntl.fcntl(stderr_fd, fcntl.F_GETFL)
            fcntl.fcntl(stderr_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            # Read both stdout and stderr for debugging
            while True:
                try:
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        logger.info(f"Rhubarb stderr: {stderr_line.strip()}")
                        
                        # Try to parse the JSON event from stderr
                        try:
                            event = json.loads(stderr_line)
                            if event['type'] == 'progress':
                                # Convert from 0-1 to 0-100 percentage
                                progress_data['progress'] = int(event['value'] * 100)
                                logger.info(f"Progress update: {progress_data['progress']}%")
                                yield f"data: {json.dumps(progress_data)}\n\n"
                            elif event['type'] == 'failure':
                                error_data = {'status': 'error', 'error': event['reason']}
                                yield f"data: {json.dumps(error_data)}\n\n"
                                return
                        except json.JSONDecodeError:
                            # Not a JSON line, ignore it
                            pass
                except IOError as e:
                    if e.errno != errno.EAGAIN:  # If error is not "no data available"
                        raise  # Re-raise the error if it's not the "no data" error
                    time.sleep(0.1)  # Short sleep to prevent CPU spinning
                
                # Check if process has finished
                if process.poll() is not None:
                    # Read any remaining output
                    remaining_stderr = process.stderr.read()
                    if remaining_stderr:
                        for line in remaining_stderr.splitlines():
                            logger.info(f"Rhubarb stderr: {line.strip()}")
                            try:
                                event = json.loads(line)
                                if event['type'] == 'progress':
                                    progress_data['progress'] = int(event['value'] * 100)
                                    logger.info(f"Progress update: {progress_data['progress']}%")
                                    yield f"data: {json.dumps(progress_data)}\n\n"
                            except json.JSONDecodeError:
                                pass
                    break
            
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
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'lipsync'), filename)

@app.route('/api/results')
def list_results():
    """List all rhubarb result files"""
    results = []
    lipsync_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'lipsync')
    if os.path.exists(lipsync_dir):
        for filename in os.listdir(lipsync_dir):
            if filename.endswith('.json'):
                results.append(filename)
    return jsonify({"results": results})

@app.route('/api/generate-video/<filename>', methods=['POST'])
def generate_video(filename):
    """Return lip sync timing data from rhubarb results"""
    logger.info(f"Getting lip sync data for {filename}")
    
    # Get the rhubarb results
    lipsync_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'lipsync')
    rhubarb_file = os.path.join(lipsync_dir, filename)
    
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

@app.route('/api/generate-mp4', methods=['POST'])
def generate_mp4():
    """Generate MP4 video using ffmpeg"""
    try:
        data = request.json
        audio_file = data['audioFile']
        background = data['background']
        mouths = data['mouths']
        
        # Prepare paths
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file)
        if background == 'default':
            bg_path = os.path.join('assets', 'backgrounds', 'background.jpg')
        else:
            # Handle base64 background image
            bg_data = background.split(',')[1]
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                temp.write(base64.b64decode(bg_data))
                bg_path = temp.name

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'output.mp4')
            
            # Initialize lists for command building
            inputs = []
            filter_complex = []
            
            # Start with background image
            filter_complex.append(f"[0:v]scale=1280:720,setsar=1[bg]")
            
            # Add each mouth as an overlay
            for i, mouth in enumerate(mouths):
                pos = mouth['position']
                reflected = mouth['reflected']
                mouth_cues = mouth['mouthCues']
                
                # Create overlay switches for each mouth shape
                shapes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X']
                for shape in shapes:
                    shape_path = os.path.join(
                        'assets',
                        'mouth_shapes_reflected' if reflected else 'mouth_shapes',
                        f'lisa-{shape}.png'
                    )
                    
                    # Add input for this shape
                    inputs.append('-i')
                    inputs.append(shape_path)
                    
                    # Calculate enable times for this shape
                    enable_expr = []
                    for cue in mouth_cues:
                        if cue['value'] == shape:
                            enable_expr.append(
                                f"between(t,{cue['start']},{cue['end']})"
                            )
                    
                    if enable_expr:
                        enable = '+'.join(enable_expr)
                        
                        # Scale and rotate the mouth shape
                        filter_complex.append(
                            f"[{len(inputs)//2}:v]scale=iw*{pos['scale']}:-1," +
                            f"rotate={pos['rotation']}*PI/180:c=none:ow=rotw({pos['rotation']}*PI/180):oh=roth({pos['rotation']}*PI/180)" +
                            f"[mouth{i}_{shape}]"
                        )
                        
                        # Overlay with enable condition
                        filter_complex.append(
                            f"[bg][mouth{i}_{shape}]overlay=" +
                            f"x={pos['x']}*W/100-w/2:y={pos['y']}*H/100-h/2:" +
                            f"enable='{enable}'[bg]"
                        )
            
            # Build the ffmpeg command
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-loop', '1',  # Loop the background image
                '-i', bg_path,  # Background image
                *inputs,  # All mouth shape inputs
                '-i', audio_path,  # Audio input
                '-filter_complex', ';'.join(filter_complex),
                '-map', '[bg]',  # Use the final background with overlays
                '-map', f'{len(inputs)//2 + 1}:a',  # Map the audio
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-tune', 'stillimage',
                '-crf', '23',
                '-c:a', 'aac',
                '-shortest',  # End when audio ends
                '-pix_fmt', 'yuv420p',  # Ensure compatibility
                '-r', '30',  # 30fps
                output_path
            ]
            
            logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"ffmpeg error: {stderr.decode()}")
                return jsonify({"error": "Failed to generate video"}), 500
            
            # Return the video file
            return send_file(
                output_path,
                mimetype='video/mp4',
                as_attachment=True,
                download_name=f'animation_{int(time.time())}.mp4'
            )
            
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        logger.exception("Full stack trace:")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info('Starting Flask development server...')
    app.run(host='0.0.0.0', port=5000, threaded=False, debug=True) 