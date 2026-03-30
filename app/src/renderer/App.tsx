import { Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import PipelinePage from "./pages/PipelinePage";
import TourOnlyPage from "./pages/TourOnlyPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/pipeline/:projectId" element={<PipelinePage />} />
      <Route path="/tour-only" element={<TourOnlyPage />} />
    </Routes>
  );
}
