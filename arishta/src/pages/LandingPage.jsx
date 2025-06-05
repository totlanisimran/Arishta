import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/LandingPage.css';
import drone from '../assets/drone.svg'

const LandingPage = () => {
  const navigate = useNavigate();
  const [selected, setSelected] = useState('');

  const handleSelect = (module, path) => {
    setSelected(module);
    navigate(path);
  };

  return (
    <div className="landing-container">
      <h1 className="landing-title">Arishta</h1>
      <p className="landing-subtitle">
        Object and Threat detection in Military using AI-ML
      </p>

      <div className="module-bar">
        <div className="module-tabs">
          <button
            className={selected === 'object' ? 'tab active' : 'tab'}
            onClick={() => handleSelect('object', '/object-detection')}
          >
            Object Detection
          </button>
          <button
            className={selected === 'crowd' ? 'tab active' : 'tab'}
            onClick={() => handleSelect('crowd', '/crowd-detection')}
          >
            Crowd Detection
          </button>
          <button
            className={selected === 'abandoned' ? 'tab active' : 'tab'}
            onClick={() => handleSelect('abandoned', '/abandoned-object')}
          >
            Abandoned Object
          </button>
        </div>
      </div>

      <img src={drone} alt="Preview" className="landing-image" />
    </div>
  );
};

export default LandingPage;
