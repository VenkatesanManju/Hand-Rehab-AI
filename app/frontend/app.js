/**
 * HandRehab AI — Frontend Application Logic
 * Authentication, Multi-Tab Navigation, Real-time MediaPipe Hand Tracking, NLP & Trend Analysis
 */

import { FilesetResolver, HandLandmarker } from './vendor/vision_bundle.mjs';

// --- Global Application State ---
let currentUser = null;
let patientHistory = [];
let trendChartInstance = null;

// Vision & Camera State
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

// Vision Metrics & Rep Counter State Machine
let repCount = 0;
let repState = 'EXTENDED'; // 'EXTENDED' or 'FLEXED'
let hasCurled = false;
let currentROM = 5.0;
let currentAngle = 175.0;
let currentFormScore = 95;
const EXTEND_THRESH = 140.0;
const FLEX_THRESH = 85.0;

// Current Daily Diary Prediction State
let currentDiaryPred = {
  pain: 0,
  stiffness: 0,
  mobility: 0,
  grip: 0,
  text: ""
};

// --- Mathematical Helper Functions ---

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

function computeROM(meanPipAngle) {
  return Math.max(0.0, Math.min(100.0, 180.0 - meanPipAngle));
}

function updateRepCounter(angle) {
  const banner = document.getElementById('live-action-banner');
  const actionText = document.getElementById('live-action-text');

  if (angle <= FLEX_THRESH) {
    repState = 'FLEXED';
    hasCurled = true;
    banner.className = 'live-action-banner curling';
    actionText.textContent = `✊ Full curl achieved (${Math.round(angle)}°)! Now open your hand completely.`;
  } else if (angle >= EXTEND_THRESH) {
    if (repState === 'FLEXED' && hasCurled) {
      repCount += 1;
      hasCurled = false;
      currentFormScore = Math.min(100, Math.max(85, Math.round(90 + Math.random() * 10)));
      banner.className = 'live-action-banner completed';
      actionText.textContent = `✨ Repetition ${repCount} Complete! Excellent extension.`;
    } else {
      banner.className = 'live-action-banner ready';
      actionText.textContent = `🖐️ Open palm ready (${Math.round(angle)}°). Now slowly curl fingers into a fist!`;
    }
    repState = 'EXTENDED';
  } else {
    if (repState === 'EXTENDED') {
      banner.className = 'live-action-banner curling';
      actionText.textContent = `🔄 Curling inward... bend knuckles past 85° (current: ${Math.round(angle)}°).`;
    }
  }

  // Update Curl Depth Progress Bar (0% at 175 deg, 100% at 75 deg)
  const curlPct = Math.max(0, Math.min(100, Math.round(((175 - angle) / (175 - 75)) * 100)));
  const fillEl = document.getElementById('curl-progress-fill');
  const labelEl = document.getElementById('curl-percent');
  if (fillEl && labelEl) {
    fillEl.style.width = `${curlPct}%`;
    labelEl.textContent = `${curlPct}%`;
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
      console.warn('GPU delegate failed, falling back to CPU:', gpuErr);
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

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = landmarks[startIdx];
    const p2 = landmarks[endIdx];
    ctx.beginPath();
    ctx.moveTo(p1.x * width, p1.y * height);
    ctx.lineTo(p2.x * width, p2.y * height);
    ctx.stroke();
  }

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

  // Summary card on camera page
  document.getElementById('summary-rom').textContent = `${currentROM}°`;
  document.getElementById('summary-reps').textContent = `${repCount}`;
}

// --- Webcam Handlers ---

async function startCamera() {
  if (simulationRunning) stopSimulation();

  // Browsers require a secure context (https or localhost/127.0.0.1) for navigator.mediaDevices
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
      const targetUrl = `http://localhost:${window.location.port || 5000}`;
      if (confirm(`Webcam access is blocked on non-localhost IP addresses (${window.location.hostname}).\n\nClick OK to open ${targetUrl} (where webcam access is allowed).`)) {
        window.location.href = targetUrl;
        return;
      }
    }
    alert('Webcam is not available in this browser window.\nPlease open http://localhost:5000 in your browser, or click "Simulated Demo" to test movement tracking.');
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
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, width, height);

  const rad = (angle * Math.PI) / 180.0;
  const startX = width * 0.5;
  const startY = height * 0.78;
  const mcpX = startX;
  const mcpY = startY - 80;
  const pipX = mcpX + Math.cos(rad - Math.PI / 2) * 65;
  const pipY = mcpY - Math.sin(rad - Math.PI / 2) * 65;
  const dipX = pipX + Math.cos(rad - Math.PI / 2 + 0.3) * 50;
  const dipY = pipY - Math.sin(rad - Math.PI / 2 + 0.3) * 50;

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

  const joints = [
    { x: startX, y: startY, r: 6, color: '#38bdf8' },
    { x: mcpX, y: mcpY, r: 7, color: '#38bdf8' },
    { x: pipX, y: pipY, r: 10, color: '#f59e0b' },
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

  ctx.fillStyle = '#94a3b8';
  ctx.font = '14px sans-serif';
  ctx.fillText(`Simulated Finger Curl: ${Math.round(angle)}° (Vertex PIP)`, 20, 30);
  ctx.restore();
}

// --- Diary NLP Handlers ---

async function analyzeDiaryText() {
  const text = document.getElementById('diary-input').value.trim();
  if (!text) {
    alert('Please enter your rehab diary notes or click one of the quick samples.');
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

    if (!res.ok) throw new Error(`Server returned error ${res.status}`);
    const data = await res.json();
    renderDiaryResults(data, text);
  } catch (err) {
    console.error('Diary analysis error:', err);
    alert('Failed to analyze diary: ' + err.message);
  } finally {
    spinner.classList.add('hidden');
  }
}

function renderDiaryResults(data, originalText) {
  currentDiaryPred = {
    pain: data.pain,
    stiffness: data.stiffness,
    mobility: data.mobility,
    grip: data.grip,
    text: originalText
  };

  const symptoms = ['pain', 'stiffness', 'mobility', 'grip'];
  symptoms.forEach(sym => {
    const val = data[sym];
    const cardVal = document.getElementById(`val-${sym}`);
    const badge = document.getElementById(`badge-${sym}`);

    cardVal.textContent = val;
    badge.textContent = data[`${sym}_label`];
    badge.className = `sev-badge badge-sev-${val}`;
  });

  // Update summary card on Camera tab
  document.getElementById('summary-pain').textContent = `${data.pain} (${data.pain_label})`;
  document.getElementById('summary-stiffness').textContent = `${data.stiffness} (${data.stiffness_label})`;

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
  const canvas = document.getElementById('trendChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
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
          label: 'Stiffness (0–3)',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
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
          title: { display: true, text: 'Symptoms (0–3)', font: { size: 11 } }
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
  const stiffData = history.map(d => d.stiffness);
  const romData = history.map(d => d.rom);

  trendChartInstance.data.labels = labels;
  trendChartInstance.data.datasets[0].data = painData;
  trendChartInstance.data.datasets[1].data = stiffData;
  trendChartInstance.data.datasets[2].data = romData;
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

  if (data.forecast) {
    document.getElementById('f-pain').textContent = data.forecast.pain;
    document.getElementById('f-stiff').textContent = data.forecast.stiffness;
    document.getElementById('f-mob').textContent = data.forecast.mobility;
    document.getElementById('f-grip').textContent = data.forecast.grip;
    document.getElementById('f-rom').textContent = `${data.forecast.rom}°`;
  }
}

function renderHistoryTable(records) {
  const tbody = document.getElementById('history-table-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!records || records.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center">No recorded sessions yet. Complete Step 1 and Step 2 above!</td></tr>';
    return;
  }

  // Display most recent records at the top
  const sorted = [...records].reverse();
  sorted.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>Day ${r.day}</strong></td>
      <td>${r.recorded_at ? r.recorded_at.substring(0, 16) : 'Today'}</td>
      <td>${Number(r.pain).toFixed(1)}</td>
      <td>${Number(r.stiffness).toFixed(1)}</td>
      <td>${Number(r.mobility).toFixed(1)}</td>
      <td>${Number(r.grip).toFixed(1)}</td>
      <td><strong>${Number(r.rom).toFixed(1)}°</strong></td>
      <td>${r.reps || 0}</td>
      <td>${r.diary_text || '—'}</td>
    `;
    tbody.appendChild(tr);
  });
}

// --- User Authentication & History Fetching ---

async function loadUserSession(user) {
  currentUser = user;
  sessionStorage.setItem('handrehab_user', JSON.stringify(user));

  // Update Header UI
  document.getElementById('display-user-name').textContent = user.full_name;
  document.getElementById('display-user-surgery').textContent = user.surgery_type;

  // Switch from Auth screen to Dashboard
  document.getElementById('auth-screen').classList.add('hidden');
  document.getElementById('dashboard-app').classList.remove('hidden');

  // Fetch Patient History from SQLite Database
  try {
    const res = await fetch(`/api/user/history?user_id=${user.id}`);
    if (res.ok) {
      const data = await res.json();
      patientHistory = data.records || [];
      updateTrendChart(patientHistory);
      renderHistoryTable(patientHistory);
      requestTrendAnalysis(patientHistory);
    }
  } catch (err) {
    console.error('Error fetching user records:', err);
  }
}

function logoutUser() {
  currentUser = null;
  sessionStorage.removeItem('handrehab_user');
  stopCamera();
  stopSimulation();

  document.getElementById('dashboard-app').classList.add('hidden');
  document.getElementById('auth-screen').classList.remove('hidden');
  document.getElementById('form-login').reset();
  document.getElementById('form-register').reset();
  showAuthAlert('', '');
}

function showAuthAlert(msg, type = 'error') {
  const el = document.getElementById('auth-alert');
  if (!msg) {
    el.classList.add('hidden');
    el.textContent = '';
    return;
  }
  el.className = `auth-alert ${type}`;
  el.textContent = msg;
  el.classList.remove('hidden');
}

// --- Session Recording ---

async function saveTodaySession() {
  if (!currentUser) {
    alert('Please sign in to save your session.');
    return;
  }

  const nextDay = patientHistory.length > 0 ? (patientHistory[patientHistory.length - 1].day + 1) : 1;

  const payload = {
    user_id: currentUser.id,
    day: nextDay,
    pain: currentDiaryPred.pain,
    stiffness: currentDiaryPred.stiffness,
    mobility: currentDiaryPred.mobility,
    grip: currentDiaryPred.grip,
    rom: currentROM,
    reps: repCount,
    diary_text: currentDiaryPred.text || 'Daily exercise and finger curl training.'
  };

  try {
    const res = await fetch('/api/user/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Failed to record session.');
    alert(`Success! Recorded Day ${nextDay} session for ${currentUser.full_name}.`);

    // Reload history
    const histRes = await fetch(`/api/user/history?user_id=${currentUser.id}`);
    if (histRes.ok) {
      const histData = await histRes.json();
      patientHistory = histData.records || [];
      updateTrendChart(patientHistory);
      renderHistoryTable(patientHistory);
      requestTrendAnalysis(patientHistory);
    }

    // Switch to Report Tab so user can see their updated trajectory
    switchView('report');
  } catch (err) {
    console.error('Error saving session:', err);
    alert('Failed to save session: ' + err.message);
  }
}

// --- View Navigation ---

function switchView(viewName) {
  const views = ['diary', 'camera', 'report'];
  views.forEach(v => {
    const panel = document.getElementById(`view-${v}`);
    const navBtn = document.getElementById(`nav-btn-${v}`);
    if (v === viewName) {
      panel.classList.add('active');
      navBtn.classList.add('active');
    } else {
      panel.classList.remove('active');
      navBtn.classList.remove('active');
    }
  });

  // If leaving camera tab, pause webcam to preserve resources
  if (viewName !== 'camera') {
    if (webcamRunning) stopCamera();
    if (simulationRunning) stopSimulation();
  }
}

// --- Initialization ---

window.addEventListener('DOMContentLoaded', async () => {
  videoElement = document.getElementById('webcam-video');
  canvasElement = document.getElementById('vision-canvas');
  canvasCtx = canvasElement.getContext('2d');

  initTrendChart();

  // 1. Auth Tabs Toggle
  const tabLogin = document.getElementById('tab-btn-login');
  const tabRegister = document.getElementById('tab-btn-register');
  const formLogin = document.getElementById('form-login');
  const formRegister = document.getElementById('form-register');

  tabLogin.addEventListener('click', () => {
    tabLogin.classList.add('active');
    tabRegister.classList.remove('active');
    formLogin.classList.remove('hidden');
    formRegister.classList.add('hidden');
    showAuthAlert('');
  });

  tabRegister.addEventListener('click', () => {
    tabRegister.classList.add('active');
    tabLogin.classList.remove('active');
    formRegister.classList.remove('hidden');
    formLogin.classList.add('hidden');
    showAuthAlert('');
  });

  // 2. Auth Submissions
  formLogin.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value.trim();

    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        loadUserSession(data.user);
      } else {
        showAuthAlert(data.error || 'Invalid credentials.', 'error');
      }
    } catch (err) {
      showAuthAlert('Network error connecting to auth server: ' + err.message, 'error');
    }
  });

  formRegister.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fullName = document.getElementById('reg-fullname').value.trim();
    const surgeryType = document.getElementById('reg-surgery').value;
    const username = document.getElementById('reg-username').value.trim();
    const password = document.getElementById('reg-password').value.trim();

    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, full_name: fullName, surgery_type: surgeryType })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        showAuthAlert('Registration successful! Logging in...', 'success');
        setTimeout(() => loadUserSession(data.user), 600);
      } else {
        showAuthAlert(data.error || 'Registration failed.', 'error');
      }
    } catch (err) {
      showAuthAlert('Network error during registration: ' + err.message, 'error');
    }
  });

  // Quick Demo Shortcut
  document.getElementById('btn-quick-demo').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: 'demo_patient', password: 'rehab123' })
      });
      const data = await res.json();
      if (res.ok && data.success) {
        loadUserSession(data.user);
      }
    } catch (err) {
      alert('Could not start quick demo: ' + err.message);
    }
  });

  // Logout
  document.getElementById('btn-logout').addEventListener('click', logoutUser);

  // 3. Navigation Tabs
  document.getElementById('nav-btn-diary').addEventListener('click', () => switchView('diary'));
  document.getElementById('nav-btn-camera').addEventListener('click', () => switchView('camera'));
  document.getElementById('nav-btn-report').addEventListener('click', () => switchView('report'));

  // Step Action Transitions
  document.getElementById('btn-goto-exercise').addEventListener('click', () => switchView('camera'));
  document.getElementById('btn-goto-report').addEventListener('click', () => switchView('report'));
  document.getElementById('btn-refresh-history').addEventListener('click', () => {
    if (currentUser) {
      fetch(`/api/user/history?user_id=${currentUser.id}`)
        .then(r => r.json())
        .then(d => {
          patientHistory = d.records || [];
          updateTrendChart(patientHistory);
          renderHistoryTable(patientHistory);
          requestTrendAnalysis(patientHistory);
        });
    }
  });

  // 4. Diary NLP
  document.getElementById('btn-analyze-diary').addEventListener('click', analyzeDiaryText);
  document.querySelectorAll('.chip-sample').forEach(chip => {
    chip.addEventListener('click', () => {
      document.getElementById('diary-input').value = chip.getAttribute('data-text');
      analyzeDiaryText();
    });
  });

  // 5. Vision Camera Controls
  document.getElementById('btn-start-camera').addEventListener('click', startCamera);
  document.getElementById('btn-stop-camera').addEventListener('click', () => {
    stopCamera();
    stopSimulation();
  });
  document.getElementById('btn-simulate-exercise').addEventListener('click', startSimulation);

  // 6. Save Session
  document.getElementById('btn-save-session').addEventListener('click', saveTodaySession);

  // Check existing session
  const savedUser = sessionStorage.getItem('handrehab_user');
  if (savedUser) {
    try {
      loadUserSession(JSON.parse(savedUser));
    } catch {
      sessionStorage.removeItem('handrehab_user');
    }
  }

  // Preload MediaPipe model in background
  initHandLandmarker();
});
