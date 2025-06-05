import { useState } from 'react';
import axios from 'axios';
import '../styles/DetectionPages.css';

const CrowdDetection = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [error, setError] = useState(null);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    try {
      setLoading(true);
      setError(null);
      const response = await axios.post('http://localhost:5000/api/crowd-detection/upload', formData);
      setResults(response.data);
      setResultImage(`data:image/jpeg;base64,${response.data.image}`);
    } catch (err) {
      setError('Error processing image');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLiveCapture = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await axios.post('http://localhost:5000/api/crowd-detection/live');
      setResults(response.data);
      setResultImage(`data:image/jpeg;base64,${response.data.image}`);
    } catch (err) {
      setError('Error capturing image');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: 'center', padding: '2rem' }}>
      <h1>Crowd Detection</h1>
      
      <div style={{ marginTop: '2rem', display:'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
        <button 
          onClick={handleLiveCapture}
          disabled={loading}
          style={{ margin: '0.5rem', padding: '0.5rem 1rem' }}
        >
          {loading ? 'Processing...' : 'Live Capture'}
        </button>

        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          disabled={loading}
          id = 'file-upload'
        />
        <div className='file-upload-button'>
          <label htmlFor="file-upload" className={`action-button ${loading ? 'disabled' : ''}`}>
            {loading ? 'Processing...' : 'Upload Image'}
          </label>
        </div>
      </div>

      {loading && <p>Processing...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}

      {results && (
        <div style={{ marginTop: '2rem' }}>
          <h3>Results:</h3>
          <p>People detected: {results.people_count}</p>
          <p>Number of clusters: {results.cluster_count}</p>
          <p>Largest cluster size: {results.largest_cluster}</p>
        </div>
      )}

      {resultImage && (
        <div style={{ marginTop: '2rem' }}>
          <h3>Processed Image:</h3>
          <img 
            src={resultImage} 
            alt="Processed" 
            style={{ maxWidth: '100%', height: 'auto' }}
          />
        </div>
      )}
    </div>
  );
};

export default CrowdDetection;