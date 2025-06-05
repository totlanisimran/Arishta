import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import ObjectDetection from './pages/ObjectDetection';
import CrowdDetection from './pages/CrowdDetection';
import AbandonedObject from './pages/AbandonedObjectDetection';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/object-detection" element={<ObjectDetection />} />
        <Route path="/crowd-detection" element={<CrowdDetection />} />
        <Route path="/abandoned-object" element={<AbandonedObject />} />
      </Routes>
    </Router>
  );
}

export default App;