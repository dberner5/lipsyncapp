import { useState, useEffect, useRef } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Button, 
  List, 
  ListItem, 
  ListItemText,
  IconButton,
  Paper,
  Alert,
  Divider,
  CircularProgress,
  Collapse,
  LinearProgress,
  Grid
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import GroupIcon from '@mui/icons-material/Group';
import SpeakerNotesIcon from '@mui/icons-material/SpeakerNotes';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import MovieIcon from '@mui/icons-material/Movie';

function App() {
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [files, setFiles] = useState([]);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [rhubarbResults, setRhubarbResults] = useState({});
  const [expandedResults, setExpandedResults] = useState({});
  const [processingRhubarb, setProcessingRhubarb] = useState({});
  const [rhubarbProgress, setRhubarbProgress] = useState({});
  const [generatingVideo, setGeneratingVideo] = useState({});
  const [mouthCues, setMouthCues] = useState({});
  const [currentMouthShape, setCurrentMouthShape] = useState({});

  useEffect(() => {
    fetch('http://127.0.0.1:5000/api/hello')
      .then(response => response.json())
      .then(data => setMessage(data.message))
      .catch(error => {
        console.error('Error:', error);
        setError(error.message);
      });

    fetchFiles();
  }, []);

  const fetchFiles = () => {
    fetch('http://127.0.0.1:5000/api/files')
      .then(response => response.json())
      .then(data => setFiles(data.files))
      .catch(error => console.error('Error fetching files:', error));
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
      if (response.ok) {
        setSelectedFile(data.filename);
        fetchFiles();
      } else {
        setError(data.error || 'Upload failed');
      }
    } catch (error) {
      console.error('Error:', error);
      setError('Upload failed');
    }
  };

  const handleDiarize = async () => {
    if (!selectedFile) return;

    setProcessing(true);
    setError('');

    try {
      const response = await fetch(`http://127.0.0.1:5000/api/diarize/${selectedFile}`, {
        method: 'POST'
      });
      const data = await response.json();

      if (response.ok) {
        fetchFiles();
      } else {
        setError(data.error || 'Diarization failed');
      }
    } catch (error) {
      console.error('Error:', error);
      setError('Diarization failed');
    } finally {
      setProcessing(false);
    }
  };

  const handlePlayPause = (filename) => {
    if (currentAudio && currentAudio.src.includes(filename)) {
      if (isPlaying) {
        currentAudio.pause();
        setIsPlaying(false);
      } else {
        currentAudio.play();
        setIsPlaying(true);
      }
    } else {
      if (currentAudio) {
        currentAudio.pause();
      }
      const audio = new Audio(`http://127.0.0.1:5000/api/audio/${filename}`);
      audio.addEventListener('timeupdate', () => updateMouthShape(filename, audio.currentTime));
      audio.play();
      setCurrentAudio(audio);
      setIsPlaying(true);
    }
  };

  const handleRhubarbProcess = async (filename) => {
    setProcessingRhubarb(prev => ({ ...prev, [filename]: true }));
    setRhubarbProgress(prev => ({
      ...prev,
      [filename]: { status: 'processing', progress: 0 }
    }));
    setError('');

    try {
      const response = await fetch(`http://127.0.0.1:5000/api/process-rhubarb/${filename}`, {
        method: 'POST'
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.trim() === '') continue;
          
          try {
            const data = JSON.parse(line.replace('data: ', ''));
            console.log('Progress update:', filename, data);
            
            setRhubarbProgress(prev => ({
              ...prev,
              [filename]: data
            }));

            if (data.status === 'done') {
              setRhubarbResults(prev => ({
                ...prev,
                [filename]: data.data
              }));
              break;
            } else if (data.status === 'error') {
              setError(data.error || 'Rhubarb processing failed');
              break;
            }
          } catch (e) {
            console.error('Error parsing progress data:', e);
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setError('Rhubarb processing failed');
    } finally {
      setProcessingRhubarb(prev => ({ ...prev, [filename]: false }));
    }
  };

  const toggleResultExpansion = (filename) => {
    setExpandedResults(prev => ({
      ...prev,
      [filename]: !prev[filename]
    }));
  };

  const getLatestSplitFiles = (files) => {
    // Group files by base name (everything before the timestamp)
    const groupedFiles = {};
    files.forEach(file => {
      // Extract base name (everything before the timestamp)
      const match = file.filename.match(/(.+)_\d{8}_\d{6}\.wav$/);
      if (match) {
        const baseName = match[1];
        if (!groupedFiles[baseName] || 
            file.filename > groupedFiles[baseName].filename) {
          groupedFiles[baseName] = file;
        }
      }
    });
    
    // Return only the latest files
    return Object.values(groupedFiles);
  };

  const originalFiles = files.filter(file => 
    file.type === 'original' && 
    file.filename.toLowerCase().endsWith('.wav')
  );
  const splitFiles = getLatestSplitFiles(
    files.filter(file => file.type === 'split')
  );

  const renderRhubarbResults = (filename) => {
    const results = rhubarbResults[filename];
    if (!results) return null;

    return (
      <Box sx={{ pl: 2, pr: 2, pb: 1 }}>
        <Typography variant="subtitle2" color="text.secondary">
          Mouth Shapes Timeline:
        </Typography>
        <Box sx={{ 
          maxHeight: '150px', 
          overflowY: 'auto',
          bgcolor: 'background.paper',
          borderRadius: 1,
          p: 1,
          mt: 1
        }}>
          {results.mouthCues.map((cue, index) => (
            <Typography key={index} variant="body2" sx={{ fontFamily: 'monospace' }}>
              {cue.start.toFixed(3)}s - {cue.end.toFixed(3)}s: {cue.value}
            </Typography>
          ))}
        </Box>
      </Box>
    );
  };

  const handleGenerateVideo = async (filename) => {
    // Get the rhubarb results filename - remove .wav extension first
    const baseFilename = filename.replace('.wav', '');
    const rhubarbFilename = `${baseFilename}_rhubarb.json`;
    console.log('Getting lip sync data for:', rhubarbFilename);
    
    setGeneratingVideo(prev => ({ ...prev, [filename]: true }));
    setError('');

    try {
        const response = await fetch(`http://127.0.0.1:5000/api/generate-video/${rhubarbFilename}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (response.ok) {
            console.log('Lip sync data received:', data);
            setMouthCues(prev => ({
                ...prev,
                [filename]: data.mouthCues
            }));
        } else {
            console.error('Failed to get lip sync data:', data.error);
            setError(data.error || 'Failed to get lip sync data');
        }
    } catch (error) {
        console.error('Error:', error);
        setError('Failed to get lip sync data');
    } finally {
        setGeneratingVideo(prev => ({ ...prev, [filename]: false }));
    }
  };

  const updateMouthShape = (filename, currentTime) => {
    const cues = mouthCues[filename];
    if (!cues) return;
    
    // Find the current mouth shape based on time
    const currentCue = cues.find(cue => 
        currentTime >= cue.start && currentTime <= cue.end
    );
    
    const shape = currentCue ? currentCue.value : 'X';
    setCurrentMouthShape(prev => ({
      ...prev,
      [filename]: shape
    }));
  };

  return (
    <Container maxWidth="sm">
      <Box sx={{ 
        marginTop: 8,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 3
      }}>
        <Typography variant="h4" component="h1" gutterBottom>
          {message || 'Loading...'}
        </Typography>

        {error && (
          <Alert severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        )}

        <Paper sx={{ p: 2, width: '100%' }}>
          <Typography variant="h6" gutterBottom>
            Step 1: Upload Audio
          </Typography>
          <Button
            variant="contained"
            component="label"
            fullWidth
          >
            Upload Audio File
            <input
              type="file"
              hidden
              accept=".mp3,.wav,.ogg,.m4a"
              onChange={handleFileUpload}
            />
          </Button>

          {selectedFile && (
            <>
              <Box sx={{ mt: 3, mb: 2 }}>
                <Divider />
              </Box>
              <Typography variant="h6" gutterBottom>
                Step 2: Split by Speakers
              </Typography>
              <Button
                variant="contained"
                color="secondary"
                fullWidth
                onClick={handleDiarize}
                disabled={processing}
                startIcon={processing ? <CircularProgress size={20} /> : <GroupIcon />}
              >
                {processing ? 'Processing...' : 'Split by Speakers'}
              </Button>
            </>
          )}

          {originalFiles.length > 0 && (
            <>
              <Box sx={{ mt: 3, mb: 1 }}>
                <Typography variant="h6">Original Files</Typography>
              </Box>
              <List>
                {originalFiles.map(({filename}) => (
                  <ListItem
                    key={filename}
                    secondaryAction={
                      <IconButton 
                        edge="end" 
                        onClick={() => handlePlayPause(filename)}
                      >
                        {currentAudio && 
                         currentAudio.src.includes(filename) && 
                         isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                      </IconButton>
                    }
                  >
                    <ListItemText primary={filename} />
                  </ListItem>
                ))}
              </List>
            </>
          )}

          {splitFiles.length > 0 && (
            <>
              <Box sx={{ mt: 3, mb: 1 }}>
                <Typography variant="h6">Speaker Splits</Typography>
              </Box>
              <Grid container spacing={2}>
                {splitFiles.map(({filename}) => (
                  <Grid item xs={6} key={filename}>
                    <Paper elevation={2} sx={{ p: 2 }}>
                      <Box sx={{ 
                        display: 'flex', 
                        flexDirection: 'column',
                        gap: 1
                      }}>
                        <Typography variant="subtitle1" noWrap>
                          {filename}
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center' }}>
                          <IconButton 
                            onClick={() => handlePlayPause(filename)}
                            size="small"
                          >
                            {currentAudio && 
                             currentAudio.src.includes(filename) && 
                             isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                          </IconButton>
                          <Button
                            variant="contained"
                            size="small"
                            onClick={() => handleRhubarbProcess(filename)}
                            disabled={processingRhubarb[filename]}
                            startIcon={processingRhubarb[filename] ? 
                              <CircularProgress size={16} /> : 
                              <SpeakerNotesIcon />}
                          >
                            {processingRhubarb[filename] ? 'Processing...' : 'Generate Lip Sync'}
                          </Button>
                          {rhubarbResults[filename] && (
                            <IconButton
                              onClick={() => toggleResultExpansion(filename)}
                              size="small"
                            >
                              {expandedResults[filename] ? 
                                <ExpandLessIcon /> : 
                                <ExpandMoreIcon />}
                            </IconButton>
                          )}
                        </Box>
                        {(processingRhubarb[filename] || rhubarbProgress[filename]?.status === 'processing') && (
                          <Box sx={{ width: '100%', mt: 1 }}>
                            <Typography variant="body2" color="text.secondary" align="center" gutterBottom>
                              {rhubarbProgress[filename]?.progress || 0}%
                            </Typography>
                            <LinearProgress 
                              variant="determinate" 
                              value={rhubarbProgress[filename]?.progress || 0}
                              sx={{ height: 8, borderRadius: 4 }}
                            />
                          </Box>
                        )}
                        {rhubarbResults[filename] && (
                          <Collapse in={expandedResults[filename]}>
                            {renderRhubarbResults(filename)}
                          </Collapse>
                        )}
                        {rhubarbResults[filename] && (
                          <>
                            <Button
                              variant="contained"
                              size="small"
                              onClick={() => handleGenerateVideo(filename)}
                              disabled={generatingVideo[filename]}
                              startIcon={generatingVideo[filename] ? 
                                <CircularProgress size={16} /> : 
                                <MovieIcon />}
                            >
                              {generatingVideo[filename] ? 'Generating...' : 'Show Animation'}
                            </Button>
                            {mouthCues[filename] && (
                              <Box sx={{ 
                                width: '100%', 
                                mt: 1,
                                position: 'relative',
                                aspectRatio: '16/9',
                                backgroundColor: '#000'
                              }}>
                                {['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X'].map(shape => (
                                  <img
                                    key={shape}
                                    src={`/assets/mouth_shapes/lisa-${shape}.png`}
                                    alt={`Mouth shape ${shape}`}
                                    style={{ 
                                      position: 'absolute',
                                      top: '50%',
                                      left: '50%',
                                      transform: 'translate(-50%, -50%)',
                                      maxWidth: '33%',
                                      display: currentMouthShape[filename] === shape ? 'block' : 'none'
                                    }}
                                  />
                                ))}
                              </Box>
                            )}
                          </>
                        )}
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </>
          )}
        </Paper>
      </Box>
    </Container>
  );
}

export default App; 