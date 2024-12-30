import { useState, useEffect } from 'react';
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
  CircularProgress
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import GroupIcon from '@mui/icons-material/Group';

function App() {
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [files, setFiles] = useState([]);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

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
      audio.play();
      setCurrentAudio(audio);
      setIsPlaying(true);
    }
  };

  const originalFiles = files.filter(file => file.type === 'original');
  const splitFiles = files.filter(file => file.type === 'split');

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
              <List>
                {splitFiles.map(({filename}) => (
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
                    <ListItemText 
                      primary={filename}
                      secondary="Split Audio"
                    />
                  </ListItem>
                ))}
              </List>
            </>
          )}
        </Paper>
      </Box>
    </Container>
  );
}

export default App; 