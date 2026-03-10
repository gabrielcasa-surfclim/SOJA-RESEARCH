import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import DiseasePage from './pages/DiseasePage';
import Research from './pages/Research';
import Gallery from './pages/Gallery';
import Upload from './pages/Upload';
import Training from './pages/Training';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/doenca/:diseaseId" element={<DiseasePage />} />
          <Route path="/pesquisa" element={<Research />} />
          <Route path="/galeria" element={<Gallery />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/treinamento" element={<Training />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
