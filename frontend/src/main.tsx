import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Shell } from './Shell';
import { Overview } from './pages/Overview';
import { Data } from './pages/Data';
import { Tasks } from './pages/Tasks';
import { Architecture } from './pages/Architecture';
import { Agents } from './pages/Agents';
import { Runs } from './pages/Runs';
import { Run } from './pages/Run';
import { Trace } from './pages/Trace';
import { Compare } from './pages/Compare';
import { Loop } from './pages/Loop';
import { Evolution } from './pages/Evolution';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/" element={<Overview />} />
          <Route path="/data" element={<Data />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="/tasks/:domain" element={<Tasks />} />
          <Route path="/tasks/:domain/:taskId" element={<Tasks />} />
          <Route path="/architecture" element={<Architecture />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/agents/:domain/:name" element={<Agents />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:runId" element={<Run />} />
          <Route path="/runs/:runId/traces/:name" element={<Trace />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/loop" element={<Loop />} />
          <Route path="/evolution" element={<Evolution />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);
