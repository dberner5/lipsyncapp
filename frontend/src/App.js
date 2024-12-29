import { useState, useEffect } from 'react';
import { Container, Typography, Box } from '@mui/material';

function App() {
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    console.log('Attempting to fetch from backend...');
    fetch('http://127.0.0.1:5000/api/hello', {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      }
    })
      .then(response => {
        console.log('Response received:', response);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('Data received:', data);
        setMessage(data.message);
      })
      .catch(error => {
        console.error('Detailed error:', error);
        setError(error.message);
        setMessage('Error connecting to backend');
      });
  }, []);

  return (
    <Container maxWidth="sm">
      <Box sx={{ 
        marginTop: 8,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
      }}>
        <Typography variant="h4" component="h1" gutterBottom>
          {message || 'Loading...'}
        </Typography>
        {error && (
          <Typography color="error" variant="body1">
            Error details: {error}
          </Typography>
        )}
      </Box>
    </Container>
  );
}

export default App; 