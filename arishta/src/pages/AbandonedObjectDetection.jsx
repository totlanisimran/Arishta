import { useEffect, useState } from 'react';
import axios from 'axios';

const AbandonedObjectDetection = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [resultImage, setResultImage] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('Hi');
        const res = await axios.get('http://localhost:5000/');
        // const data = await res.json();
        // console.log(data);
        console.log(res);
      } catch (err) {
        console.error(err);
      }
    };
  
    fetchData();
  }, []);

  const handleLiveCapture = async () => {
    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5000/api/live-capture');
      setResult(response.data.results);
      setResultImage(`data:image/jpeg;base64,${response.data.image}`);
    } catch (error) {
      console.error('Error during live capture:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = async (event) => {
    const files = event.target.files;
    if (files.length === 0) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('images', files[i]);
    }

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:5000/api/process-images', formData);
      setResult(response.data.results);
      setResultImage(`data:image/jpeg;base64,${response.data.image}`);
    } catch (error) {
      console.error('Error processing images:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: 'center', padding: '2rem' }}>
      <h1>Abandoned Object Detection</h1>
      
      <div style={{ marginTop: '2rem', display:'flex', flexDirection: 'row', justifyContent: 'center', alignItems: 'center' }}>
        <button 
          onClick={handleLiveCapture}
          disabled={loading}
          style={{ margin: '0.5rem', padding: '0.5rem 1rem' }}
        >
          {loading ? 'Processing...' : 'Start Live Capture'}
        </button>

        <input
          type="file"
          multiple
          accept="image/*"
          onChange={handleImageUpload}
          disabled={loading}
          id = 'file-upload'
        />
        <div className='file-upload-button'>
          <label htmlFor="file-upload" className={`action-button ${loading ? 'disabled' : ''}`}>
            {loading ? 'Processing...' : 'Upload Images'}
          </label>
        </div>
      </div>

      {loading && <p>Processing...</p>}

      {result && (
        <div style={{ marginTop: '2rem' }}>
          <h3>Results:</h3>
          <pre>{result}</pre>
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

export default AbandonedObjectDetection;