import { useState, useEffect, useRef, useMemo } from 'react';
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
  Grid,
  Card,
  CardMedia,
  CardActionArea
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import GroupIcon from '@mui/icons-material/Group';
import SpeakerNotesIcon from '@mui/icons-material/SpeakerNotes';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import MovieIcon from '@mui/icons-material/Movie';
import AddPhotoAlternateIcon from '@mui/icons-material/AddPhotoAlternate';
import DownloadIcon from '@mui/icons-material/Download';

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
  const [selectedBackground, setSelectedBackground] = useState(null);
  const [customBackground, setCustomBackground] = useState(null);
  const [mouthPositions, setMouthPositions] = useState({});
  const [selectedMouth, setSelectedMouth] = useState(null);
  const [preloadedImages, setPreloadedImages] = useState({});
  const [imageLoadingStatus, setImageLoadingStatus] = useState('loading');
  const [reflectedMouths, setReflectedMouths] = useState({});
  const [masterAudio, setMasterAudio] = useState(null);
  const [isMasterPlaying, setIsMasterPlaying] = useState(false);
  const [generatingMp4, setGeneratingMp4] = useState(false);
  const [mp4Progress, setMp4Progress] = useState(0);
  const [numSpeakers, setNumSpeakers] = useState(2);


  useEffect(() => {
    const shapes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X'];
    const maxRetries = 3;
    const retryDelay = 1000;

    const loadImageWithRetry = (shape, isReflected, retryCount = 0) => {
      return new Promise((resolve, reject) => {
        const img = new Image();
        const folder = isReflected ? 'mouth_shapes_reflected' : 'mouth_shapes';
        
        img.onload = () => {
          console.log(`Successfully loaded ${isReflected ? 'reflected' : 'regular'} image: lisa-${shape}.png`);
          resolve({ shape, isReflected, img });
        };
        
        img.onerror = () => {
          console.error(`Failed to load ${isReflected ? 'reflected' : 'regular'} image: lisa-${shape}.png (attempt ${retryCount + 1})`);
          if (retryCount < maxRetries) {
            setTimeout(() => {
              loadImageWithRetry(shape, isReflected, retryCount + 1)
                .then(resolve)
                .catch(reject);
            }, retryDelay);
          } else {
            reject(new Error(`Failed to load ${isReflected ? 'reflected' : 'regular'} image: lisa-${shape}.png after ${maxRetries} attempts`));
          }
        };

        img.src = `/assets/${folder}/lisa-${shape}.png`;
      });
    };

    setImageLoadingStatus('loading');
    
    const loadAllImages = async () => {
      const imageMap = { regular: {}, reflected: {} };
      let errors = [];

      for (const shape of shapes) {
        try {
          // Load both regular and reflected versions
          const [regular, reflected] = await Promise.all([
            loadImageWithRetry(shape, false),
            loadImageWithRetry(shape, true)
          ]);
          
          imageMap.regular[shape] = regular.img.src;
          imageMap.reflected[shape] = reflected.img.src;
        } catch (error) {
          console.error(error);
          errors.push(shape);
        }
      }

      if (errors.length > 0) {
        console.error('Failed to load some images:', errors);
        setImageLoadingStatus('error');
        setError(`Failed to load some mouth shapes: ${errors.join(', ')}`);
      } else {
        console.log('All images loaded successfully');
        setImageLoadingStatus('success');
        setPreloadedImages(imageMap);
      }
    };

    loadAllImages();
  }, []);

  const fetchFiles = () => {
    fetch('http://127.0.0.1:5000/api/files')
      .then(response => response.json())
      .then(data => {
        setFiles(data.files);
        
        // Check for existing rhubarb results for each split file
        const splitFiles = getLatestSplitFiles(
          data.files.filter(file => file.type === 'split')
        );
        
        splitFiles.forEach(({filename}) => {
          const baseFilename = filename.replace('.wav', '');
          const rhubarbFilename = `${baseFilename}_rhubarb.json`;
          
          // Try to fetch existing rhubarb results
          fetch(`http://127.0.0.1:5000/api/results/${rhubarbFilename}`)
            .then(response => {
              if (!response.ok) throw new Error('No cached results');
              return response.json();
            })
            .then(data => {
              console.log('Found cached rhubarb results for:', filename);
              setRhubarbResults(prev => ({
                ...prev,
                [filename]: data
              }));
              setMouthCues(prev => ({
                ...prev,
                [filename]: data.mouthCues
              }));
            })
            .catch(error => {
              console.log('No cached results for:', filename);
            });
        });
      })
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
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ numSpeakers })
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
        
        const line = decoder.decode(value);
        const lines = line.split('\n');
        
        for (const line of lines) {
          if (line.trim() === '') continue;
          
          try {
            const data = JSON.parse(line.replace('data: ', ''));
            console.log('Progress update:', filename, data);
            
            if (data.status === 'error') {
              setError(data.error || 'Rhubarb processing failed');
              break;
            }
            
            if (data.status === 'done') {
              setRhubarbResults(prev => ({
                ...prev,
                [filename]: data.data
              }));
              setMouthCues(prev => ({
                ...prev,
                [filename]: data.data.mouthCues
              }));
              setCurrentMouthShape(prev => ({
                ...prev,
                [filename]: 'X'
              }));
              // Set progress to 100% when done
              setRhubarbProgress(prev => ({
                ...prev,
                [filename]: { status: 'done', progress: 100 }
              }));
              break;
            } else {
              // Update progress for any other status
              setRhubarbProgress(prev => ({
                ...prev,
                [filename]: { 
                  status: 'processing',
                  progress: data.progress || prev[filename]?.progress || 0
                }
              }));
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

  const handleGenerateAllLipSync = async () => {
    setError('');
    
    // Process each speaker's audio file sequentially
    for (const {filename} of splitFiles) {
      await handleRhubarbProcess(filename);
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

  const getSpeakerDisplayName = (filename, index) => {
    return `Speaker ${index + 1}`;
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

  const handleBackgroundUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const imageUrl = URL.createObjectURL(file);
    setCustomBackground(imageUrl);
    setSelectedBackground('custom');
  };

  const handleAddMouth = (filename) => {
    setMouthPositions(prev => ({
      ...prev,
      [filename]: {
        x: 50,
        y: 50,
        scale: 1,
        rotation: 0
      }
    }));
    setReflectedMouths(prev => ({
      ...prev,
      [filename]: false
    }));
    setCurrentMouthShape(prev => ({
      ...prev,
      [filename]: 'X'
    }));
  };

  const handleMouthChange = (filename, property, value) => {
    setMouthPositions(prev => ({
      ...prev,
      [filename]: {
        ...prev[filename],
        [property]: value
      }
    }));
  };

  const handleMouthReflect = (filename) => {
    setReflectedMouths(prev => ({
      ...prev,
      [filename]: !prev[filename]
    }));
  };

  const renderMouthShape = useMemo(() => (shape, style, filename) => {
    if (imageLoadingStatus === 'loading') {
      return (
        <Box
          sx={{
            ...style,
            width: '100px',
            height: '100px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(0, 0, 0, 0.1)'
          }}
        >
          <CircularProgress size={20} />
        </Box>
      );
    }

    const isReflected = reflectedMouths[filename];
    const images = preloadedImages[isReflected ? 'reflected' : 'regular'];

    if (!images || !images[shape]) {
      return (
        <Box
          sx={{
            ...style,
            width: '100px',
            height: '100px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: 'rgba(255, 0, 0, 0.1)'
          }}
        >
          <Typography variant="caption" color="error">
            Failed to load
          </Typography>
        </Box>
      );
    }

    return (
      <img
        key={shape}
        src={images[shape]}
        alt={`Mouth shape ${shape}`}
        style={style}
      />
    );
  }, [preloadedImages, imageLoadingStatus, reflectedMouths]);

  const handleMasterPlayPause = () => {
    if (!originalFiles[0]) return;

    if (isMasterPlaying) {
      masterAudio?.pause();
      setIsMasterPlaying(false);
    } else {
      if (masterAudio) {
        masterAudio.play();
      } else {
        const audio = new Audio(`http://127.0.0.1:5000/api/audio/${originalFiles[0].filename}`);
        audio.addEventListener('timeupdate', () => {
          // Update all mouth shapes
          Object.keys(mouthPositions).forEach(filename => {
            updateMouthShape(filename, audio.currentTime);
          });
        });
        audio.addEventListener('ended', () => {
          setIsMasterPlaying(false);
        });
        audio.play();
        setMasterAudio(audio);
      }
      setIsMasterPlaying(true);
    }
  };

  const handleDownloadVideo = async () => {
    if (!originalFiles[0] || !Object.keys(mouthPositions).length) return;
    
    setGeneratingMp4(true);
    setMp4Progress(0);
    setError('');

    try {
      const videoData = {
        audioFile: originalFiles[0].filename,
        background: selectedBackground === 'default' ? 'default' : customBackground,
        mouths: Object.entries(mouthPositions).map(([filename, position]) => ({
          filename,
          position,
          reflected: reflectedMouths[filename],
          mouthCues: mouthCues[filename]
        }))
      };

      const response = await fetch('http://127.0.0.1:5000/api/generate-mp4', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(videoData)
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Failed to generate video');
      }

      // Get the video as a blob
      const blob = await response.blob();
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `animation_${new Date().getTime()}.mp4`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error generating video:', error);
      setError(error.message || 'Failed to generate video');
    } finally {
      setGeneratingMp4(false);
      setMp4Progress(0);
    }
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

          {originalFiles.length > 0 && (
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
          )}

          {selectedFile && (
            <>
              <Box sx={{ mt: 3, mb: 2 }}>
                <Divider />
              </Box>
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 2,
                mb: 2 
              }}>
                <Typography variant="h6" gutterBottom sx={{ mb: 0 }}>
                  Step 2: Split by Speakers
                </Typography>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  gap: 1
                }}>
                  <Typography variant="body1">
                    Number of speakers:
                  </Typography>
                  <select
                    value={numSpeakers}
                    onChange={(e) => setNumSpeakers(Number(e.target.value))}
                    style={{
                      padding: '4px 8px',
                      borderRadius: '4px',
                      border: '1px solid #ccc'
                    }}
                  >
                    <option value={2}>2</option>
                    <option value={3}>3</option>
                  </select>
                </Box>
              </Box>
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

          {splitFiles.length > 0 && (
            <>
              <Box sx={{ mt: 3, mb: 1 }}>
                <Typography variant="h6">Speaker Splits</Typography>
              </Box>
              <Grid container spacing={2}>
                {splitFiles.map(({filename}, index) => (
                  <Grid item xs={6} key={filename}>
                    <Paper elevation={2} sx={{ p: 2 }}>
                      <Box sx={{ 
                        display: 'flex', 
                        flexDirection: 'column',
                        gap: 1
                      }}>
                        <Box sx={{ 
                          display: 'flex', 
                          alignItems: 'center',
                          justifyContent: 'space-between'
                        }}>
                          <Typography variant="subtitle1" noWrap>
                            {getSpeakerDisplayName(filename, index)}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1 }}>
                            <IconButton 
                              onClick={() => handlePlayPause(filename)}
                              size="small"
                            >
                              {currentAudio && 
                               currentAudio.src.includes(filename) && 
                               isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                            </IconButton>
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
                        </Box>
                        {(processingRhubarb[filename] || rhubarbProgress[filename]?.status === 'processing') && (
                          <Box sx={{ width: '100%' }}>
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
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
              <Box sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  fullWidth
                  onClick={handleGenerateAllLipSync}
                  disabled={Object.values(processingRhubarb).some(Boolean)}
                  startIcon={Object.values(processingRhubarb).some(Boolean) ? 
                    <CircularProgress size={20} /> : 
                    <SpeakerNotesIcon />}
                >
                  {Object.values(processingRhubarb).some(Boolean) ? 
                    'Generating Lip Sync...' : 
                    'Generate Lip Sync for All Speakers'}
                </Button>
              </Box>
            </>
          )}
        </Paper>

        {splitFiles.length > 0 && (
          <Paper sx={{ p: 2, width: '100%', mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Create Video
            </Typography>
            <Typography variant="subtitle1" gutterBottom>
              Select Background Image
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Card>
                  <CardActionArea 
                    onClick={() => setSelectedBackground('default')}
                    sx={{ 
                      border: selectedBackground === 'default' ? 2 : 0,
                      borderColor: 'primary.main'
                    }}
                  >
                    <CardMedia
                      component="img"
                      height="140"
                      image="/assets/backgrounds/background.jpg"
                      alt="Default background"
                    />
                    <Box sx={{ p: 1 }}>
                      <Typography variant="body2">
                        Default Background
                      </Typography>
                    </Box>
                  </CardActionArea>
                </Card>
              </Grid>
              <Grid item xs={6}>
                <Card>
                  <CardActionArea 
                    component="label"
                    sx={{ 
                      height: '100%',
                      border: selectedBackground === 'custom' ? 2 : 0,
                      borderColor: 'primary.main'
                    }}
                  >
                    {customBackground ? (
                      <CardMedia
                        component="img"
                        height="140"
                        image={customBackground}
                        alt="Custom background"
                      />
                    ) : (
                      <Box sx={{ 
                        height: 140, 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        backgroundColor: 'grey.100'
                      }}>
                        <AddPhotoAlternateIcon sx={{ fontSize: 40, color: 'grey.500' }} />
                      </Box>
                    )}
                    <input
                      type="file"
                      hidden
                      accept="image/*"
                      onChange={handleBackgroundUpload}
                    />
                    <Box sx={{ p: 1 }}>
                      <Typography variant="body2">
                        {customBackground ? 'Custom Background' : 'Upload Background'}
                      </Typography>
                    </Box>
                  </CardActionArea>
                </Card>
              </Grid>
            </Grid>

            {selectedBackground && (
              <Box sx={{ mt: 3 }}>
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between', 
                  alignItems: 'center',
                  mb: 2
                }}>
                  <Typography variant="subtitle1">
                    Selected Background
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="contained"
                      startIcon={isMasterPlaying ? <PauseIcon /> : <PlayArrowIcon />}
                      onClick={handleMasterPlayPause}
                      disabled={!originalFiles.length || !Object.keys(mouthPositions).length}
                    >
                      {isMasterPlaying ? 'Pause' : 'Play Animation'}
                    </Button>
                    <Button
                      variant="contained"
                      color="secondary"
                      startIcon={<DownloadIcon />}
                      onClick={handleDownloadVideo}
                      disabled={!originalFiles.length || !Object.keys(mouthPositions).length || generatingMp4}
                    >
                      {generatingMp4 ? 'Generating...' : 'Download MP4'}
                    </Button>
                  </Box>
                </Box>
                {generatingMp4 && (
                  <Box sx={{ width: '100%', mt: 1 }}>
                    <LinearProgress />
                  </Box>
                )}
                <Grid container spacing={2}>
                  <Grid item xs={9}>
                    <Card>
                      <Box sx={{ position: 'relative' }}>
                        <CardMedia
                          component="img"
                          sx={{
                            width: '100%',
                            height: 'auto',
                            objectFit: 'contain',
                            maxHeight: '70vh'
                          }}
                          image={selectedBackground === 'default' ? 
                            '/assets/backgrounds/background.jpg' : 
                            customBackground}
                          alt="Selected background"
                        />
                        
                        {Object.entries(mouthPositions).map(([filename, position]) => (
                          <Box
                            key={filename}
                            sx={{
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              width: '100%',
                              height: '100%',
                              cursor: selectedMouth === filename ? 'move' : 'pointer',
                              border: selectedMouth === filename ? '2px solid blue' : 'none'
                            }}
                            onClick={() => setSelectedMouth(filename)}
                          >
                            {['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'X'].map(shape => 
                              renderMouthShape(shape, { 
                                position: 'absolute',
                                top: `${position.y}%`,
                                left: `${position.x}%`,
                                transform: `translate(-50%, -50%) scale(${position.scale}) rotate(${position.rotation}deg)`,
                                maxWidth: '33%',
                                display: currentMouthShape[filename] === shape ? 'block' : 'none'
                              }, filename)
                            )}
                          </Box>
                        ))}
                      </Box>
                    </Card>
                  </Grid>
                  <Grid item xs={3} sx={{ maxWidth: '300px' }}>
                    <Box sx={{ 
                      display: 'flex', 
                      flexDirection: 'column',
                      gap: 2
                    }}>
                      <Box>
                        <Typography variant="caption" gutterBottom>
                          Mouth Animation
                        </Typography>
                        <Grid container spacing={0.5}>
                          {splitFiles.map(({filename}, index) => (
                            <Grid item xs={12} key={filename}>
                              <Button
                                variant={selectedMouth === filename ? "contained" : "outlined"}
                                fullWidth
                                onClick={() => {
                                  if (!mouthPositions[filename]) {
                                    handleAddMouth(filename);
                                  }
                                  setSelectedMouth(selectedMouth === filename ? null : filename);
                                }}
                                size="small"
                                sx={{ 
                                  py: 0.5,
                                  fontSize: '0.75rem',
                                  minHeight: 0
                                }}
                              >
                                {mouthPositions[filename] ? 
                                  getSpeakerDisplayName(filename, index) :
                                  `Add ${getSpeakerDisplayName(filename, index)}`
                                }
                              </Button>
                            </Grid>
                          ))}
                        </Grid>
                      </Box>

                      {selectedMouth && mouthPositions[selectedMouth] && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Adjust Selected Mouth
                          </Typography>
                          <Grid container spacing={2}>
                            <Grid item xs={12}>
                              <Typography variant="caption">X Position: {mouthPositions[selectedMouth].x}%</Typography>
                              <input
                                type="range"
                                min="0"
                                max="100"
                                value={mouthPositions[selectedMouth].x}
                                onChange={(e) => handleMouthChange(selectedMouth, 'x', Number(e.target.value))}
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <Typography variant="caption">Y Position: {mouthPositions[selectedMouth].y}%</Typography>
                              <input
                                type="range"
                                min="0"
                                max="100"
                                value={mouthPositions[selectedMouth].y}
                                onChange={(e) => handleMouthChange(selectedMouth, 'y', Number(e.target.value))}
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <Typography variant="caption">Scale: {mouthPositions[selectedMouth].scale.toFixed(2)}x</Typography>
                              <input
                                type="range"
                                min="0.1"
                                max="3"
                                step="0.01"
                                value={mouthPositions[selectedMouth].scale}
                                onChange={(e) => handleMouthChange(selectedMouth, 'scale', Number(e.target.value))}
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <Typography variant="caption">Rotation: {mouthPositions[selectedMouth].rotation}°</Typography>
                              <input
                                type="range"
                                min="-180"
                                max="180"
                                value={mouthPositions[selectedMouth].rotation}
                                onChange={(e) => handleMouthChange(selectedMouth, 'rotation', Number(e.target.value))}
                              />
                            </Grid>
                            <Grid item xs={12}>
                              <Typography variant="caption">Reflect</Typography>
                              <Button
                                variant={reflectedMouths[selectedMouth] ? "contained" : "outlined"}
                                onClick={() => handleMouthReflect(selectedMouth)}
                                fullWidth
                                size="small"
                              >
                                {reflectedMouths[selectedMouth] ? "Reflected" : "Normal"}
                              </Button>
                            </Grid>
                          </Grid>
                        </Box>
                      )}
                    </Box>
                  </Grid>
                </Grid>
              </Box>
            )}
          </Paper>
        )}
      </Box>
    </Container>
  );
}

export default App; 