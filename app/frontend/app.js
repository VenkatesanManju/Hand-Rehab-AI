/**
 * HandRehab AI — Frontend Application Logic (Phase 5 & 6)
 * Real-time MediaPipe Hand Tracking, NLP Diary Analysis & Recovery Trend Tracking
 */

import { FilesetResolver, HandLandmarker } from './vendor/vision_bundle.mjs';

// --- State Variables ---
let handLandmarker = null;
let webcamRunning = false;
let videoElement = null;
let canvasElement = null;
let canvasCtx = null;
let lastVideoTime = -1;
let animationFrameId = null;

// Simulation Mode State
let simulationRunning = false;
let simAngle = 175.0;
let simDirection = -1;
let simInterval = null;

// Vision Metrics & Rep Counter State
let repCount = 0;
let repState = 'EXTENDED'; // 'EXTENDED' or 'FLEXED'
let hasCurled = false;
let currentROM = 5.0;
let currentAngle = 175.0;
let currentFormScore = 95;
const EXTEND_THRESH = 140.0;
const FLEX_THRESH = 85.0;

// Longitudinal Patient History (Default Initial 14-day history)
let patientHistory = [];
let trendChartInstance = null;

// --- Mathematical Helper Functions ---

/**
 * Calculates 3D angle between points A, B, and C with vertex at B in degrees.
 */
function computeAngle3D(a, b, c) {
  const ba = [a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0)];
  const bc = [c.x - b.x, c.y - b.y, (c.z || 0) - (b.z || 0)];

  const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
  const normBA = Math.sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]);
  const normBC = Math.sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);

  if (normBA === 0 || normBC === 0) return 0.0;
  const cosine = Math.max(-1.0, Math.min(1.0, dot / (normBA * normBC)));
  return (Math.acos(cosine) * 180.0) / Math.PI;
}

/**
 * Computes active flexion Range of Motion (0 to 100 degrees) from extension angle.
 */
function computeROM(meanPipAngle) {
  return Math.max(0.0, Math.min(100.0, 180.0 - meanPipAngle));
}

/**
 * Updates repetition counter state machine.
 */
function updateRepCounter(angle) {
  if (angle <= FLEX_THRESH) {
    repState = 'FLEXED';
    hasCurled = true;
  } else if (angle >= EXTEND_THRESH) {
    if (repState === 'FLEXED' && hasCurled) {
      repCount += 1;
      hasCurled = false;
      // Form quality bonus on complete range
      currentFormScore = Math.min(100, Math.max(80, Math.round(90 + Math.random() * 10)));
    }
    repState = 'EXTENDED';
  }
}

// --- MediaPipe Hand Tracking Setup ---

async function initHandLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks('./vendor/wasm');
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: './vendor/hand_landmarker.task',
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 1
      });
      console.log('HandLandmarker initialized with GPU delegate.');
    } catch (gpuErr) {
      console.warn('GPU initialization failed, falling back to CPU:', gpuErr);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: './vendor/hand_landmarker.task',
          delegate: 'CPU'
        },
        runningMode: 'VIDEO',
        numHands: 1
      });
      console.log('HandLandmarker initialized with CPU delegate.');
    }
  } catch (err) {
    console.error('Error loading MediaPipe Tasks Vision:', err);
  }
}

// Landmark connection indices for hand skeleton drawing
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [0, 17]                               // Palm base
];

function drawLandmarks(ctx, landmarks, width, height) {
  ctx.save();
  ctx.strokeStyle = '#0284c7';
  ctx.lineWidth = 3;
  ctx.fillStyle = '#38bdf8';

  // Draw connections
  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = landmarks[startIdx];
    const p2 = landmarks[endIdx];
    ctx.beginPath();
    ctx.moveTo(p1.x * width, p1.y * height);
    ctx.lineTo(p2.x * width, p2.y * height);
    ctx.stroke();
  }

  // Draw points
  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    const x = pt.x * width;
    const y = pt.y * height;
    ctx.beginPath();
    ctx.arc(x, y, (i === 6 || i === 10) ? 6 : 4, 0, 2 * Math.PI);
    ctx.fillStyle = (i === 6 || i === 10) ? '#f59e0b' : '#38bdf8';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.restore();
}

async function renderVideoLoop() {
  if (!webcamRunning) return;

  if (videoElement.currentTime !== lastVideoTime) {
    lastVideoTime = videoElement.currentTime;

    canvasElement.width = videoElement.videoWidth || 640;
    canvasElement.height = videoElement.videoHeight || 480;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    if (handLandmarker) {
      const results = handLandmarker.detectForVideo(videoElement, performance.now());
      if (results && results.landmarks && results.landmarks.length > 0) {
        const lm = results.landmarks[0];
        drawLandmarks(canvasCtx, lm, canvasElement.width, canvasElement.height);

        // Compute PIP angles:
        // Index: 5 (MCP), 6 (PIP), 7 (DIP)
        const indexAngle = computeAngle3D(lm[5], lm[6], lm[7]);
        // Middle: 9 (MCP), 10 (PIP), 11 (DIP)
        const middleAngle = computeAngle3D(lm[9], lm[10], lm[11]);

        currentAngle = Math.round((indexAngle + middleAngle) / 2.0);
        currentROM = Math.round(computeROM(currentAngle));
        updateRepCounter(currentAngle);

        updateVisionUI();
      }
    }
  }

  animationFrameId = requestAnimationFrame(renderVideoLoop);
}

function updateVisionUI() {
  document.getElementById('live-angle').textContent = `${currentAngle}°`;
  document.getElementById('live-rom').textContent = `${currentROM}°`;
  document.getElementById('live-reps').textContent = `${repCount}`;
  document.getElementById('live-form').textContent = `${currentFormScore}%`;
}

// --- Webcam Handlers ---

async function startCamera() {
  if (simulationRunning) stopSimulation();

  // Browsers require a secure context (https or localhost/127.0.0.1) for navigator.mediaDevices
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      const targetUrl = `http://localhost:${window.location.port || 5000}`;
      if (confirm(`Webcam access in web browsers is blocked on non-localhost IP addresses (like ${window.location.hostname}).\n\nClick OK to open ${targetUrl} (where webcam access is allowed).`)) {
        window.location.href = targetUrl;
        return;
      }
    }
    alert('Webcam is not available in this browser window.\nPlease open http://localhost:5000 or http://127.0.0.1:5000 in your browser, or click "Simulated Demo" to test movement tracking.');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    });
    videoElement.srcObject = stream;
    videoElement.onloadedmetadata = () => {
      videoElement.play();
      webcamRunning = true;
      document.getElementById('vision-placeholder').classList.add('hidden');
      document.getElementById('btn-start-camera').disabled = true;
      document.getElementById('btn-stop-camera').disabled = false;
      renderVideoLoop();
    };
  } catch (err) {
    alert('Webcam access failed or denied: ' + err.message + '\nYou can test tracking using the "Simulated Demo" mode.');
  }
}


function stopCamera() {
  webcamRunning = false;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  if (videoElement.srcObject) {
    videoElement.srcObject.getTracks().forEach(track => track.stop());
    videoElement.srcObject = null;
  }
  document.getElementById('vision-placeholder').classList.remove('hidden');
  document.getElementById('btn-start-camera').disabled = false;
  document.getElementById('btn-stop-camera').disabled = true;
}

// --- Simulated Exercise Demo Mode ---

function startSimulation() {
  if (webcamRunning) stopCamera();
  simulationRunning = true;
  document.getElementById('vision-placeholder').classList.add('hidden');
  document.getElementById('btn-start-camera').disabled = false;
  document.getElementById('btn-stop-camera').disabled = false;

  canvasElement.width = 640;
  canvasElement.height = 480;

  if (simInterval) clearInterval(simInterval);

  simInterval = setInterval(() => {
    // Oscillate joint angle between 175 and 65
    simAngle += simDirection * 4;
    if (simAngle <= 65) {
      simAngle = 65;
      simDirection = 1;
    } else if (simAngle >= 175) {
      simAngle = 175;
      simDirection = -1;
    }

    currentAngle = Math.round(simAngle);
    currentROM = Math.round(computeROM(currentAngle));
    updateRepCounter(currentAngle);
    updateVisionUI();

    // Render simulated 2D hand skeleton on canvas
    drawSimulatedHand(canvasCtx, canvasElement.width, canvasElement.height, simAngle);
  }, 50);
}

function stopSimulation() {
  simulationRunning = false;
  if (simInterval) {
    clearInterval(simInterval);
    simInterval = null;
  }
  document.getElementById('vision-placeholder').classList.remove('hidden');
  document.getElementById('btn-stop-camera').disabled = true;
}

function drawSimulatedHand(ctx, width, height, angle) {
  ctx.clearRect(0, 0, width, height);

  ctx.save();
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(0, 0, width, height);

  // Draw simulated finger joint curl
  const rad = (angle * Math.PI) / 180.0;
  const startX = width * 0.5;
  const startY = height * 0.75;
  const mcpX = startX;
  const mcpY = startY - 80;
  const pipX = mcpX + Math.cos(rad - Math.PI / 2) * 60;
  const pipY = mcpY - Math.sin(rad - Math.PI / 2) * 60;
  const dipX = pipX + Math.cos(rad - Math.PI / 2 + 0.3) * 45;
  const dipY = pipY - Math.sin(rad - Math.PI / 2 + 0.3) * 45;

  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 6;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(mcpX, mcpY);
  ctx.lineTo(pipX, pipY);
  ctx.lineTo(dipX, dipY);
  ctx.stroke();

  // Draw joints
  const joints = [
    { x: startX, y: startY, r: 6, color: '#38bdf8' },
    { x: mcpX, y: mcpY, r: 7, color: '#38bdf8' },
    { x: pipX, y: pipY, r: 9, color: '#f59e0b' },
    { x: dipX, y: dipY, r: 5, color: '#38bdf8' }
  ];

  joints.forEach(j => {
    ctx.beginPath();
    ctx.arc(j.x, j.y, j.r, 0, 2 * Math.PI);
    ctx.fillStyle = j.color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Overlay text
  ctx.fillStyle = '#94a3b8';
  ctx.font = '14px sans-serif';
  ctx.fillText(`Simulated Finger Curl: ${Math.round(angle)}° (Vertex PIP)`, 20, 30);
  ctx.restore();
}

// --- Diary NLP Handlers ---

async function analyzeDiaryText() {
  const text = document.getElementById('diary-input').value.trim();
  if (!text) {
    alert('Please enter diary notes or click one of the quick samples above.');
    return;
  }

  const spinner = document.getElementById('diary-loading');
  spinner.classList.remove('hidden');

  try {
    const res = await fetch('/api/diary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error(`Server returned error ${res.status}`);
    }

    const data = await res.json();
    renderDiaryResults(data);
  } catch (err) {
    console.error('Diary analysis error:', err);
    alert('Failed to analyze diary: ' + err.message);
  } finally {
    spinner.classList.add('hidden');
  }
}

function renderDiaryResults(data) {
  const symptoms = ['pain', 'stiffness', 'mobility', 'grip'];
  symptoms.forEach(sym => {
    const val = data[sym];
    const cardVal = document.getElementById(`val-${sym}`);
    const badge = document.getElementById(`badge-${sym}`);

    cardVal.textContent = val;
    badge.textContent = data[`${sym}_label`];
    badge.className = `sev-badge badge-sev-${val}`;
  });

  // Render Causal Attribution Chips
  const chipContainer = document.getElementById('causal-chips');
  chipContainer.innerHTML = '';

  if (data.highlighted_words && data.highlighted_words.length > 0) {
    data.highlighted_words.forEach(item => {
      const chip = document.createElement('span');
      chip.className = `causal-chip ${item.is_causal ? 'causal-active' : ''}`;
      chip.innerHTML = `${item.word} <span class="chip-w">${(item.weight * 100).toFixed(0)}%</span>`;
      chipContainer.appendChild(chip);
    });
  } else {
    chipContainer.innerHTML = '<span class="placeholder-text">No tokens detected.</span>';
  }
}

// --- Trend & Chart Handlers ---

function initTrendChart() {
  const ctx = document.getElementById('trendChart').getContext('2d');
  trendChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Pain Severity (0–3)',
          data: [],
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          yAxisID: 'yPain',
          tension: 0.25,
          borderWidth: 2,
          pointRadius: 3
        },
        {
          label: 'Active ROM (°)',
          data: [],
          borderColor: '#0284c7',
          backgroundColor: 'rgba(2, 132, 199, 0.1)',
          yAxisID: 'yRom',
          tension: 0.25,
          borderWidth: 2,
          pointRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Rehabilitation Day', font: { size: 11 } },
          grid: { display: false }
        },
        yPain: {
          type: 'linear',
          position: 'left',
          min: 0,
          max: 3,
          title: { display: true, text: 'Pain', font: { size: 11 } }
        },
        yRom: {
          type: 'linear',
          position: 'right',
          min: 0,
          max: 100,
          title: { display: true, text: 'ROM (°)', font: { size: 11 } },
          grid: { drawOnChartArea: false }
        }
      },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } }
      }
    }
  });
}

function updateTrendChart(history) {
  if (!trendChartInstance) return;

  const labels = history.map(d => `Day ${d.day}`);
  const painData = history.map(d => d.pain);
  const romData = history.map(d => d.rom);

  trendChartInstance.data.labels = labels;
  trendChartInstance.data.datasets[0].data = painData;
  trendChartInstance.data.datasets[1].data = romData;
  trendChartInstance.update();
}

async function requestTrendAnalysis(history) {
  if (history.length < 14) return;

  try {
    const res = await fetch('/api/trend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history })
    });

    if (!res.ok) throw new Error(`Trend server error ${res.status}`);
    const data = await res.json();
    renderTrendResults(data);
  } catch (err) {
    console.error('Trend analysis error:', err);
  }
}

function renderTrendResults(data) {
  const banner = document.getElementById('alert-banner');
  const title = document.getElementById('alert-title');
  const msg = document.getElementById('alert-message');
  const icon = document.getElementById('alert-icon');

  banner.className = `status-banner banner-${data.alert_level}`;
  msg.textContent = data.alert;

  if (data.alert_level === 'danger') {
    icon.textContent = '⚠️';
    title.textContent = 'Acute Flare-Up / Relapse Alert';
  } else if (data.alert_level === 'warning') {
    icon.textContent = '⚠️';
    title.textContent = 'Recovery Plateau Warning';
  } else {
    icon.textContent = '✅';
    title.textContent = 'Progression On Track';
  }

  // Next-day forecast readouts
  if (data.forecast) {
    document.getElementById('f-pain').textContent = data.forecast.pain;
    document.getElementById('f-stiff').textContent = data.forecast.stiffness;
    document.getElementById('f-mob').textContent = data.forecast.mobility;
    document.getElementById('f-grip').textContent = data.forecast.grip;
    document.getElementById('f-rom').textContent = `${data.forecast.rom}°`;
  }
}

// --- Sample Patient Trajectory & Session Storage ---

async function loadSamplePatient() {
  try {
    const res = await fetch('/api/sample_patient');
    if (!res.ok) throw new Error('Sample patient fetch failed');
    const data = await res.json();

    patientHistory = data.trajectory;
    localStorage.setItem('handrehab_patient_history', JSON.stringify(patientHistory));
    updateTrendChart(patientHistory);
    requestTrendAnalysis(patientHistory);
  } catch (err) {
    console.error('Error loading sample patient:', err);
    // Fallback baseline trajectory
    generateFallbackTrajectory();
  }
}

function generateFallbackTrajectory() {
  patientHistory = [];
  for (let d = 0; d < 20; d++) {
    patientHistory.push({
      day: d,
      pain: Math.max(0.5, (2.6 - d * 0.08 + (Math.random() - 0.5) * 0.3)).toFixed(1),
      stiffness: Math.max(0.5, (2.5 - d * 0.07 + (Math.random() - 0.5) * 0.3)).toFixed(1),
      mobility: Math.max(0.4, (2.4 - d * 0.08 + (Math.random() - 0.5) * 0.3)).toFixed(1),
      grip: Math.max(0.5, (2.5 - d * 0.07 + (Math.random() - 0.5) * 0.3)).toFixed(1),
      rom: Math.min(85.0, (35.0 + d * 2.2 + (Math.random() - 0.5) * 3.0)).toFixed(1)
    });
  }
  updateTrendChart(patientHistory);
  requestTrendAnalysis(patientHistory);
}

function saveTodaySession() {
  const currentDay = patientHistory.length > 0 ? patientHistory[patientHistory.length - 1].day + 1 : 1;

  const painVal = parseFloat(document.getElementById('val-pain').textContent) || 1.0;
  const stiffVal = parseFloat(document.getElementById('val-stiffness').textContent) || 1.0;
  const mobVal = parseFloat(document.getElementById('val-mobility').textContent) || 1.0;
  const gripVal = parseFloat(document.getElementById('val-grip').textContent) || 1.0;
  const romVal = parseFloat(currentROM) || 50.0;

  patientHistory.push({
    day: currentDay,
    pain: painVal,
    stiffness: stiffVal,
    mobility: mobVal,
    grip: gripVal,
    rom: romVal
  });

  localStorage.setItem('handrehab_patient_history', JSON.stringify(patientHistory));
  updateTrendChart(patientHistory);
  requestTrendAnalysis(patientHistory);
  alert(`Recorded Day ${currentDay} session to patient history.`);
}

function resetSession() {
  if (confirm('Reset patient session history to default sample patient?')) {
    localStorage.removeItem('handrehab_patient_history');
    loadSamplePatient();
  }
}

// --- Initialization ---

window.addEventListener('DOMContentLoaded', async () => {
  videoElement = document.getElementById('webcam-video');
  canvasElement = document.getElementById('vision-canvas');
  canvasCtx = canvasElement.getContext('2d');

  initTrendChart();

  // Load stored history or sample patient
  const stored = localStorage.getItem('handrehab_patient_history');
  if (stored) {
    try {
      patientHistory = JSON.parse(stored);
      updateTrendChart(patientHistory);
      requestTrendAnalysis(patientHistory);
    } catch {
      loadSamplePatient();
    }
  } else {
    loadSamplePatient();
  }

  // Bind Diary Events
  document.getElementById('btn-analyze-diary').addEventListener('click', analyzeDiaryText);
  document.querySelectorAll('.chip-sample').forEach(chip => {
    chip.addEventListener('click', () => {
      document.getElementById('diary-input').value = chip.getAttribute('data-text');
      analyzeDiaryText();
    });
  });

  // Bind Vision Events
  document.getElementById('btn-start-camera').addEventListener('click', startCamera);
  document.getElementById('btn-stop-camera').addEventListener('click', () => {
    stopCamera();
    stopSimulation();
  });
  document.getElementById('btn-simulate-exercise').addEventListener('click', startSimulation);

  // Bind Patient History Events
  document.getElementById('btn-load-sample').addEventListener('click', loadSamplePatient);
  document.getElementById('btn-reset-data').addEventListener('click', resetSession);
  document.getElementById('btn-save-session').addEventListener('click', saveTodaySession);

  // Initialize MediaPipe Tasks in background
  initHandLandmarker();
});
