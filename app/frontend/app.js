/**
 * HandRehab AI — Frontend Application Logic
 * Authentication, Multi-Tab Navigation, Real-time MediaPipe Hand Tracking, NLP & Trend Analysis
 */

import { FilesetResolver, HandLandmarker } from './vendor/vision_bundle.mjs';

// --- Global Application State ---
let currentUser = null;
let patientHistory = [];
let painStiffnessChartInstance = null;
let romChartInstance = null;
let mobilityGripChartInstance = null;
let trendChartInstance = null; // Legacy alias
let currentHistoryFilter = 'all'; // 'all', '14', '7'
let historySortAscending = true; // true = Day 1 -> Day N (chronological), false = Day N -> Day 1

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

// --- Rehab Condition Guides & Adaptive Instructions ---

const CONDITION_GUIDES = {
  'flexor': {
    name: 'Palm / Finger Tendon Repair (Flexor Tendon Zone II)',
    shortName: 'Palm / Finger Tendon Repair',
    exerciseTitle: 'Controlled Finger Flexion & Tendon Gliding (Tendon Glide)',
    goal: 'Smooth, non-jerky finger curls to safely glide deep flexor tendons without excessive load, preventing scar adhesions inside the tendon sheath.',
    hintDesc: 'Repair of tendons on the palm side that pull your fingers inward to make a fist. Rehabilitation focuses on gentle active gliding to prevent internal scar adhesions.',
    hintExercise: 'Controlled Finger Flexion & Tendon Gliding (Open Palm ➡️ Gentle Fist ➡️ Open)',
    step1: {
      icon: '🖐️',
      title: '1. Open Palm Facing Camera',
      desc: 'Hold your hand upright facing the camera with fingers relaxed and straight. Finger angle should exceed <strong>140°</strong>.'
    },
    step2: {
      icon: '✊',
      title: '2. Smooth Composite Fist Curl',
      desc: 'Slowly curl all four fingers inward toward your palm until knuckles bend past <strong>85°</strong>. Watch the curl depth bar fill up.'
    },
    step3: {
      icon: '✨',
      title: '3. Controlled Extension Return',
      desc: 'Smoothly straighten fingers back to full open palm. Crossing <strong>140°</strong> records <strong>+1 Rep</strong> without snapping.'
    },
    liveTip: 'Focus on smooth, non-jerky finger curls to safely glide repaired flexor tendons.'
  },
  'extensor': {
    name: 'Back of Hand / Finger Straightening (Extensor Tendon)',
    shortName: 'Back of Hand Tendon Repair',
    exerciseTitle: 'Active Finger Extension & Tabletop/Hook Curl',
    goal: 'Rebuild finger straightening strength, eliminate extensor tendon lag, and restore balanced opening coordination.',
    hintDesc: 'Repair of tendons on the back of your hand that lift and straighten fingers. Rehabilitation focuses on active extension and eliminating tendon lag.',
    hintExercise: 'Active Finger Extension & Tabletop Curl (Tabletop ➡️ Gentle Fist ➡️ Full Open)',
    step1: {
      icon: '🖐️',
      title: '1. Aligned Open Hand (Start)',
      desc: 'Position palm facing camera with fingers actively held straight and upright (angle <strong>&gt;140°</strong>).'
    },
    step2: {
      icon: '✊',
      title: '2. Gentle Knuckle Curl',
      desc: 'Bend fingers smoothly inward past <strong>85°</strong> into a comfortable curl. Avoid strain on the back of the hand.'
    },
    step3: {
      icon: '✨',
      title: '3. Active Extension & Rep Count',
      desc: 'Actively focus on straightening every knuckle completely back to <strong>&gt;140°</strong> to register <strong>+1 Rep</strong>.'
    },
    liveTip: 'Actively straighten fingers on each rep to prevent extensor tendon lag.'
  },
  'radius': {
    name: 'Broken Wrist Recovery (Distal Radius Fracture)',
    shortName: 'Broken Wrist Recovery',
    exerciseTitle: 'Finger-to-Wrist Mobility & Composite Curl (Post-Cast)',
    goal: 'Overcome post-cast finger stiffness, promote lymphatic drainage, and rebuild coordinated active range of motion.',
    hintDesc: 'Recovery after a wrist fracture (broken distal radius) treated with a cast or surgery. Rehab restores finger mobility and prevents forearm stiffness.',
    hintExercise: 'Finger-to-Wrist Mobility Curls (Forearm Steady ➡️ Finger Curl ➡️ Wide Spread)',
    step1: {
      icon: '🖐️',
      title: '1. Neutral Forearm & Open Palm',
      desc: 'Support your forearm comfortably on a table with palm facing camera and fingers spread (angle <strong>&gt;140°</strong>).'
    },
    step2: {
      icon: '✊',
      title: '2. Full Composite Curl',
      desc: 'Slowly curl all fingers into a full fist past <strong>85°</strong>, keeping your wrist neutral and relaxed.'
    },
    step3: {
      icon: '✨',
      title: '3. Wide Finger Spread & Count',
      desc: 'Extend fingers back open and spread them comfortably wide (<strong>&gt;140°</strong>) to count <strong>+1 Rep</strong> and reduce swelling.'
    },
    liveTip: 'Keep your wrist neutral and relaxed while curling fingers.'
  },
  'carpal': {
    name: 'Carpal Tunnel Release (Nerve Decompression & Recovery)',
    shortName: 'Carpal Tunnel Release',
    exerciseTitle: 'Median Nerve Mobilization & Tendon Gliding Curls',
    goal: 'Encourage healthy median nerve gliding through the carpal canal, minimize pillar pain, and regain grip agility.',
    hintDesc: 'Post-operative recovery following surgical release of the carpal ligament to decompress the median nerve and relieve numbness/tingling.',
    hintExercise: 'Median Nerve Gliding Curls (Neutral Wrist ➡️ Gradual Curl ➡️ Gentle Extension)',
    step1: {
      icon: '🖐️',
      title: '1. Neutral Wrist & Open Fingers',
      desc: 'Keep wrist straight and relaxed with palm facing camera. Fingers straight upright (angle <strong>&gt;140°</strong>).'
    },
    step2: {
      icon: '✊',
      title: '2. Progressive Fist Squeeze',
      desc: 'Roll fingertips into the palm past <strong>85°</strong>, feeling a gentle, comfortable gliding motion through your wrist.'
    },
    step3: {
      icon: '✨',
      title: '3. Full Release & Rep Count',
      desc: 'Smoothly release and open the palm back out (<strong>&gt;140°</strong>) to decompress the carpal tunnel and log <strong>+1 Rep</strong>.'
    },
    liveTip: 'Keep wrist straight to allow the median nerve to glide freely.'
  },
  'metacarpal': {
    name: 'Broken Hand Bone Recovery (Metacarpal Fracture)',
    shortName: 'Broken Hand Bone Recovery',
    exerciseTitle: 'Knuckle (MCP) Articulation & Finger Flexion',
    goal: 'Restore knuckle joint (MCP) range of motion, prevent collateral ligament shortening, and restore grip closure.',
    hintDesc: 'Rehabilitation after a broken hand bone (palm/knuckle fracture) to restore knuckle movement and prevent palm stiffness.',
    hintExercise: 'Knuckle Articulation Curls (Open Hand ➡️ Knuckle Bending ➡️ Full Return)',
    step1: {
      icon: '🖐️',
      title: '1. Extended Palm Position',
      desc: 'Position your hand upright facing the camera with knuckles gently extended and fingers open (angle <strong>&gt;140°</strong>).'
    },
    step2: {
      icon: '✊',
      title: '2. Knuckle Flexion (Fist)',
      desc: 'Bend from the main knuckles (MCP joints) curling fingers past <strong>85°</strong> into a compact fist without forcing pain.'
    },
    step3: {
      icon: '✨',
      title: '3. Controlled Knuckle Return',
      desc: 'Straighten your knuckles smoothly back to the open start position (<strong>&gt;140°</strong>) to record <strong>+1 Rep</strong>.'
    },
    liveTip: 'Focus on bending from the main knuckles without twisting the hand.'
  },
  'general': {
    name: 'General Hand Mobility & Stiffness (Everyday Recovery)',
    shortName: 'General Hand Mobility',
    exerciseTitle: 'Full Range Active Finger Flexion & Joint Lubrication',
    goal: 'Increase joint fluid circulation, overcome morning stiffness, and maintain healthy active range of motion.',
    hintDesc: 'Designed for everyday finger stiffness, arthritis management, or general post-immobilization hand recovery.',
    hintExercise: 'Full Range Finger Curls (Open Hand Stretch ➡️ Full Fist ➡️ Release)',
    step1: {
      icon: '🖐️',
      title: '1. Open Hand Stretch',
      desc: 'Hold fingers comfortably open and facing the camera with a relaxed upright posture (angle <strong>&gt;140°</strong>).'
    },
    step2: {
      icon: '✊',
      title: '2. Full Fist Flexion',
      desc: 'Curl fingers inward toward your palm past <strong>85°</strong> at a steady, comfortable pace. Watch the curl progress track.'
    },
    step3: {
      icon: '✨',
      title: '3. Full Return & Rep Count',
      desc: 'Open your hand back out fully (angle <strong>&gt;140°</strong>) to complete <strong>+1 Rep</strong> toward your target.'
    },
    liveTip: 'Move at a steady, relaxed rhythm through your full comfortable range.'
  }
};

function getConditionKey(surgeryType) {
  if (!surgeryType) return 'general';
  const s = String(surgeryType).toLowerCase();
  if (s.includes('extensor')) return 'extensor';
  if (s.includes('flexor')) return 'flexor';
  if (s.includes('radius') || s.includes('wrist')) return 'radius';
  if (s.includes('carpal')) return 'carpal';
  if (s.includes('metacarpal') || s.includes('bone')) return 'metacarpal';
  if (CONDITION_GUIDES[s]) return s;
  return 'general';
}

function updateRegistrationHint(surgeryType) {
  const key = getConditionKey(surgeryType);
  const guide = CONDITION_GUIDES[key] || CONDITION_GUIDES['general'];

  const titleEl = document.getElementById('reg-card-title');
  const descEl = document.getElementById('reg-card-desc');
  const exEl = document.getElementById('reg-card-exercise');

  if (titleEl) titleEl.textContent = guide.shortName;
  if (descEl) descEl.textContent = guide.hintDesc;
  if (exEl) exEl.textContent = guide.hintExercise;
}

function updateExerciseGuide(surgeryTypeOrKey) {
  const key = getConditionKey(surgeryTypeOrKey);
  const guide = CONDITION_GUIDES[key] || CONDITION_GUIDES['general'];

  // Update badge and title
  const badgeEl = document.getElementById('guide-condition-badge');
  if (badgeEl) badgeEl.textContent = `🎯 Prescribed for: ${guide.shortName}`;

  const titleEl = document.getElementById('guide-exercise-title');
  if (titleEl) titleEl.textContent = `Prescribed Exercise: ${guide.exerciseTitle}`;

  const goalEl = document.getElementById('guide-condition-goal');
  if (goalEl) goalEl.innerHTML = `<strong>Rehabilitation Goal:</strong> ${guide.goal}`;

  // Update Step 1
  const s1Icon = document.getElementById('guide-step-1-icon');
  if (s1Icon) s1Icon.textContent = guide.step1.icon;
  const s1Title = document.getElementById('guide-step-1-title');
  if (s1Title) s1Title.textContent = guide.step1.title;
  const s1Desc = document.getElementById('guide-step-1-desc');
  if (s1Desc) s1Desc.innerHTML = guide.step1.desc;

  // Update Step 2
  const s2Icon = document.getElementById('guide-step-2-icon');
  if (s2Icon) s2Icon.textContent = guide.step2.icon;
  const s2Title = document.getElementById('guide-step-2-title');
  if (s2Title) s2Title.textContent = guide.step2.title;
  const s2Desc = document.getElementById('guide-step-2-desc');
  if (s2Desc) s2Desc.innerHTML = guide.step2.desc;

  // Update Step 3
  const s3Icon = document.getElementById('guide-step-3-icon');
  if (s3Icon) s3Icon.textContent = guide.step3.icon;
  const s3Title = document.getElementById('guide-step-3-title');
  if (s3Title) s3Title.textContent = guide.step3.title;
  const s3Desc = document.getElementById('guide-step-3-desc');
  if (s3Desc) s3Desc.innerHTML = guide.step3.desc;

  // Sync selector dropdown
  const selectEl = document.getElementById('guide-condition-select');
  if (selectEl && selectEl.value !== key) {
    selectEl.value = key;
  }

  // Update idle camera banner tip if camera is not active
  const actionText = document.getElementById('live-action-text');
  if (actionText && (!webcamStream && !simulationRunning)) {
    actionText.textContent = `Camera ready for ${guide.shortName}. ${guide.liveTip}`;
  }
}

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
  const liveAngle = document.getElementById('live-angle');
  if (liveAngle) liveAngle.textContent = `${currentAngle}°`;
  const liveRom = document.getElementById('live-rom');
  if (liveRom) liveRom.textContent = `${currentROM}°`;
  const liveReps = document.getElementById('live-reps');
  if (liveReps) liveReps.textContent = `${repCount}`;
  const liveForm = document.getElementById('live-form');
  if (liveForm) liveForm.textContent = `${currentFormScore}%`;

  // Summary card on camera page
  const sumRom = document.getElementById('summary-rom');
  if (sumRom) sumRom.textContent = `${currentROM}°`;
  const sumReps = document.getElementById('summary-reps');
  if (sumReps) sumReps.textContent = `${repCount}`;

  // Summary card on diary page
  const diaryRom = document.getElementById('diary-summary-rom');
  if (diaryRom) diaryRom.textContent = `${currentROM}°`;
  const diaryReps = document.getElementById('diary-summary-reps');
  if (diaryReps) diaryReps.textContent = `${repCount}`;
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
  const sumPain = document.getElementById('summary-pain');
  if (sumPain) sumPain.textContent = `${data.pain} (${data.pain_label})`;
  const sumStiff = document.getElementById('summary-stiffness');
  if (sumStiff) sumStiff.textContent = `${data.stiffness} (${data.stiffness_label})`;

  // Update summary card on Diary tab
  const diaryPain = document.getElementById('diary-summary-pain');
  if (diaryPain) diaryPain.textContent = `${data.pain} (${data.pain_label})`;
  const diaryStiff = document.getElementById('diary-summary-stiffness');
  if (diaryStiff) diaryStiff.textContent = `${data.stiffness} (${data.stiffness_label})`;
  const diaryMob = document.getElementById('diary-summary-mobility');
  if (diaryMob) diaryMob.textContent = `${data.mobility} (${data.mobility_label})`;
  const diaryGrip = document.getElementById('diary-summary-grip');
  if (diaryGrip) diaryGrip.textContent = `${data.grip} (${data.grip_label})`;

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

// --- Recovery Progress & Hand Movement (ROM) Trend Chart Handlers ---

function parseMetric(val) {
  if (val === null || val === undefined || val === '' || isNaN(Number(val))) {
    return null; // Return null so Chart.js does not treat missing data as zero
  }
  return Number(val);
}

function formatDayLabel(d) {
  let datePart = '';
  if (d && d.recorded_at) {
    try {
      // Handles both ISO strings and SQLite TIMESTAMP strings
      const dt = new Date(d.recorded_at.replace(' ', 'T'));
      if (!isNaN(dt.getTime())) {
        datePart = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      }
    } catch (e) {}
  }
  return datePart ? `Day ${d.day} (${datePart})` : `Day ${d.day}`;
}

function getSeverityWord(val) {
  if (val === null || val === undefined || val === '' || isNaN(Number(val))) return 'Not recorded';
  const rounded = Math.round(Number(val));
  const words = ['None', 'Mild', 'Moderate', 'Severe'];
  return words[rounded] !== undefined ? words[rounded] : '';
}

function getMobilityWord(val) {
  if (val === null || val === undefined || val === '' || isNaN(Number(val))) return 'Not recorded';
  const rounded = Math.round(Number(val));
  const words = ['Normal mobility', 'Mild limitation', 'Moderate limitation', 'Severe limitation'];
  return words[rounded] !== undefined ? words[rounded] : '';
}

function getGripWord(val) {
  if (val === null || val === undefined || val === '' || isNaN(Number(val))) return 'Not recorded';
  const rounded = Math.round(Number(val));
  const words = ['Normal strength', 'Mild weakness', 'Moderate weakness', 'Severe weakness'];
  return words[rounded] !== undefined ? words[rounded] : '';
}

function escapeHtml(str) {
  if (!str) return '—';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// --- Recovery Summary Overview (6 Metric Cards) ---

function updateRecoverySummaryCards(history) {
  const records = history || [];
  const latest = records.length > 0 ? records[records.length - 1] : null;

  const dayBadge = document.getElementById('summary-last-recorded-day');
  const valPain = document.getElementById('rep-val-pain');
  const descPain = document.getElementById('rep-desc-pain');
  const valStiff = document.getElementById('rep-val-stiffness');
  const descStiff = document.getElementById('rep-desc-stiffness');
  const valMob = document.getElementById('rep-val-mobility');
  const descMob = document.getElementById('rep-desc-mobility');
  const valGrip = document.getElementById('rep-val-grip');
  const descGrip = document.getElementById('rep-desc-grip');
  const valRom = document.getElementById('rep-val-rom');
  const descRom = document.getElementById('rep-desc-rom');
  const valReps = document.getElementById('rep-val-reps');
  const descReps = document.getElementById('rep-desc-reps');
  const nextStepEl = document.getElementById('summary-next-step');

  if (!latest) {
    if (dayBadge) dayBadge.textContent = 'Day —';
    if (valPain) valPain.textContent = '—';
    if (descPain) descPain.textContent = 'No records yet';
    if (valStiff) valStiff.textContent = '—';
    if (descStiff) descStiff.textContent = 'No records yet';
    if (valMob) valMob.textContent = '—';
    if (descMob) descMob.textContent = 'No records yet';
    if (valGrip) valGrip.textContent = '—';
    if (descGrip) descGrip.textContent = 'No records yet';
    if (valRom) valRom.textContent = '—';
    if (descRom) descRom.textContent = 'No records yet';
    if (valReps) valReps.textContent = '—';
    if (descReps) descReps.textContent = 'No records yet';
    if (nextStepEl) {
      nextStepEl.innerHTML = '<strong>Next Step:</strong> Record your first session in Step 1 and Step 2 above to view your initial recovery summary.';
    }
    return;
  }

  // Date and day label
  let dateText = '';
  if (latest.recorded_at) {
    try {
      const dt = new Date(latest.recorded_at.replace(' ', 'T'));
      if (!isNaN(dt.getTime())) {
        dateText = dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
      }
    } catch (e) {}
  }
  if (dayBadge) {
    dayBadge.textContent = dateText ? `Day ${latest.day} (${dateText})` : `Day ${latest.day}`;
  }

  // Card 1: Pain
  if (latest.pain !== null && latest.pain !== undefined && latest.pain !== '' && !isNaN(Number(latest.pain))) {
    if (valPain) valPain.textContent = `${Number(latest.pain).toFixed(1)} / 3`;
    if (descPain) descPain.textContent = getSeverityWord(latest.pain);
  } else {
    if (valPain) valPain.textContent = '—';
    if (descPain) descPain.textContent = 'Not recorded';
  }

  // Card 2: Stiffness
  if (latest.stiffness !== null && latest.stiffness !== undefined && latest.stiffness !== '' && !isNaN(Number(latest.stiffness))) {
    if (valStiff) valStiff.textContent = `${Number(latest.stiffness).toFixed(1)} / 3`;
    if (descStiff) descStiff.textContent = getSeverityWord(latest.stiffness);
  } else {
    if (valStiff) valStiff.textContent = '—';
    if (descStiff) descStiff.textContent = 'Not recorded';
  }

  // Card 3: Mobility
  if (latest.mobility !== null && latest.mobility !== undefined && latest.mobility !== '' && !isNaN(Number(latest.mobility))) {
    if (valMob) valMob.textContent = `${Number(latest.mobility).toFixed(1)} / 3`;
    if (descMob) descMob.textContent = getMobilityWord(latest.mobility);
  } else {
    if (valMob) valMob.textContent = '—';
    if (descMob) descMob.textContent = 'Not recorded';
  }

  // Card 4: Grip Strength
  if (latest.grip !== null && latest.grip !== undefined && latest.grip !== '' && !isNaN(Number(latest.grip))) {
    if (valGrip) valGrip.textContent = `${Number(latest.grip).toFixed(1)} / 3`;
    if (descGrip) descGrip.textContent = getGripWord(latest.grip);
  } else {
    if (valGrip) valGrip.textContent = '—';
    if (descGrip) descGrip.textContent = 'Not recorded';
  }

  // Card 5: Range of Motion (ROM)
  if (latest.rom !== null && latest.rom !== undefined && latest.rom !== '' && !isNaN(Number(latest.rom))) {
    const romVal = Number(latest.rom);
    if (valRom) valRom.textContent = `${romVal.toFixed(1)}°`;
    if (descRom) {
      if (romVal >= 80) descRom.textContent = 'Good active flexion';
      else if (romVal >= 50) descRom.textContent = 'Moderate flexion';
      else descRom.textContent = 'Early gentle movement';
    }
  } else {
    if (valRom) valRom.textContent = '—';
    if (descRom) descRom.textContent = 'Not measured';
  }

  // Card 6: Completed Repetitions
  const reps = latest.reps !== null && latest.reps !== undefined ? Number(latest.reps) : 0;
  if (valReps) valReps.textContent = `${reps} reps`;
  if (descReps) {
    descReps.textContent = reps >= 10 ? 'Session target reached' : `${reps} of 10 target reps`;
  }

  // Next-step guidance
  if (nextStepEl) {
    nextStepEl.innerHTML = '<strong>Next Step:</strong> Continue your prescribed exercises as advised by your physiotherapist. If you experience persistent pain, sudden stiffness, or sudden drop in movement, contact your physiotherapist.';
  }
}

function updateRecoveryCharts(history) {
  const records = history || [];

  const emptySymptoms = document.getElementById('empty-state-symptoms');
  const emptyRom = document.getElementById('empty-state-rom');
  const emptyMobGrip = document.getElementById('empty-state-mobility-grip');
  const noticeSymptoms = document.getElementById('notice-single-session-symptoms');
  const noticeRom = document.getElementById('notice-single-session-rom');
  const noticeMobGrip = document.getElementById('notice-single-session-mobility-grip');
  const wrapperSymptoms = document.getElementById('wrapper-pain-stiffness');
  const wrapperRom = document.getElementById('wrapper-rom');
  const wrapperMobGrip = document.getElementById('wrapper-mobility-grip');

  if (records.length === 0) {
    // Zero sessions: display patient-friendly message, hide charts and 1-session banners
    if (emptySymptoms) emptySymptoms.classList.remove('hidden');
    if (emptyRom) emptyRom.classList.remove('hidden');
    if (emptyMobGrip) emptyMobGrip.classList.remove('hidden');
    if (noticeSymptoms) noticeSymptoms.classList.add('hidden');
    if (noticeRom) noticeRom.classList.add('hidden');
    if (noticeMobGrip) noticeMobGrip.classList.add('hidden');
    if (wrapperSymptoms) wrapperSymptoms.classList.add('hidden');
    if (wrapperRom) wrapperRom.classList.add('hidden');
    if (wrapperMobGrip) wrapperMobGrip.classList.add('hidden');
    return;
  }

  // 1 or more sessions recorded: show canvas wrappers and hide empty states
  if (emptySymptoms) emptySymptoms.classList.add('hidden');
  if (emptyRom) emptyRom.classList.add('hidden');
  if (emptyMobGrip) emptyMobGrip.classList.add('hidden');
  if (wrapperSymptoms) wrapperSymptoms.classList.remove('hidden');
  if (wrapperRom) wrapperRom.classList.remove('hidden');
  if (wrapperMobGrip) wrapperMobGrip.classList.remove('hidden');

  if (records.length === 1) {
    // Exactly 1 session: show explanation that more sessions are needed for trend slope
    if (noticeSymptoms) noticeSymptoms.classList.remove('hidden');
    if (noticeRom) noticeRom.classList.remove('hidden');
    if (noticeMobGrip) noticeMobGrip.classList.remove('hidden');
  } else {
    // 2 or more sessions: hide single-session notices
    if (noticeSymptoms) noticeSymptoms.classList.add('hidden');
    if (noticeRom) noticeRom.classList.add('hidden');
    if (noticeMobGrip) noticeMobGrip.classList.add('hidden');
  }

  const labels = records.map(d => formatDayLabel(d));
  const painData = records.map(d => parseMetric(d.pain));
  const stiffData = records.map(d => parseMetric(d.stiffness));
  const romData = records.map(d => parseMetric(d.rom));
  const mobData = records.map(d => parseMetric(d.mobility));
  const gripData = records.map(d => parseMetric(d.grip));

  // Prominent point radius if only 1 session, clean standard points if multiple
  const pointSize = records.length === 1 ? 8 : 4.5;
  const pointHoverSize = records.length === 1 ? 10 : 7;

  // -------------------------------------------------------------
  // GRAPH 1: Recovery Progress Over Time (Daily Pain & Stiffness)
  // -------------------------------------------------------------
  const ctxSymptoms = document.getElementById('painStiffnessChart');
  if (ctxSymptoms) {
    if (painStiffnessChartInstance) {
      painStiffnessChartInstance.data.labels = labels;
      painStiffnessChartInstance.data.datasets[0].data = painData;
      painStiffnessChartInstance.data.datasets[0].pointRadius = pointSize;
      painStiffnessChartInstance.data.datasets[0].pointHoverRadius = pointHoverSize;
      painStiffnessChartInstance.data.datasets[1].data = stiffData;
      painStiffnessChartInstance.data.datasets[1].pointRadius = pointSize;
      painStiffnessChartInstance.data.datasets[1].pointHoverRadius = pointHoverSize;
      painStiffnessChartInstance.update();
    } else {
      painStiffnessChartInstance = new Chart(ctxSymptoms.getContext('2d'), {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Pain Score (0–3)',
              data: painData,
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              borderWidth: 2.5,
              pointBackgroundColor: '#ef4444',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 1.5,
              pointRadius: pointSize,
              pointHoverRadius: pointHoverSize,
              tension: 0.25,
              spanGaps: true
            },
            {
              label: 'Stiffness Score (0–3)',
              data: stiffData,
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245, 158, 11, 0.1)',
              borderWidth: 2.5,
              pointBackgroundColor: '#f59e0b',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 1.5,
              pointRadius: pointSize,
              pointHoverRadius: pointHoverSize,
              tension: 0.25,
              spanGaps: true
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: 'index',
            intersect: false
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Rehabilitation Day & Date',
                font: { size: 12, weight: '600' },
                color: '#475569'
              },
              grid: { color: 'rgba(226, 232, 240, 0.6)' }
            },
            y: {
              min: 0,
              max: 3,
              title: {
                display: true,
                text: 'Symptom Severity (0–3 Scale)',
                font: { size: 12, weight: '600' },
                color: '#475569'
              },
              ticks: {
                stepSize: 1,
                color: '#475569',
                callback: function(val) {
                  const map = { 0: '0 - None', 1: '1 - Mild', 2: '2 - Moderate', 3: '3 - Severe' };
                  return map[val] !== undefined ? map[val] : val;
                }
              },
              grid: { color: 'rgba(226, 232, 240, 0.6)' }
            }
          },
          plugins: {
            legend: {
              position: 'top',
              labels: { boxWidth: 14, font: { size: 12, weight: '600' }, color: '#334155' }
            },
            tooltip: {
              callbacks: {
                title: function(items) {
                  if (!items || items.length === 0) return '';
                  const idx = items[0].dataIndex;
                  const record = records[idx];
                  const dateStr = record && record.recorded_at ? record.recorded_at.substring(0, 16) : 'Recorded Session';
                  return `Day ${record ? record.day : items[0].label} • ${dateStr}`;
                },
                label: function(context) {
                  if (context.raw === null || context.raw === undefined) {
                    return `${context.dataset.label}: Not recorded`;
                  }
                  const val = Number(context.raw).toFixed(1);
                  const word = getSeverityWord(context.raw);
                  return `${context.dataset.label}: ${val} / 3 (${word})`;
                }
              }
            }
          }
        }
      });
    }
  }

  // -------------------------------------------------------------
  // GRAPH 2: Hand Movement (ROM) Trend (Actual Measured ROM in °)
  // -------------------------------------------------------------
  const ctxRom = document.getElementById('romChart');
  if (ctxRom) {
    if (romChartInstance) {
      romChartInstance.data.labels = labels;
      romChartInstance.data.datasets[0].data = romData;
      romChartInstance.data.datasets[0].pointRadius = pointSize;
      romChartInstance.data.datasets[0].pointHoverRadius = pointHoverSize;
      romChartInstance.update();
    } else {
      romChartInstance = new Chart(ctxRom.getContext('2d'), {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Actual Active Range of Motion (°)',
              data: romData,
              borderColor: '#0284c7',
              backgroundColor: 'rgba(2, 132, 199, 0.12)',
              fill: true,
              borderWidth: 2.5,
              pointBackgroundColor: '#0284c7',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 1.5,
              pointRadius: pointSize,
              pointHoverRadius: pointHoverSize,
              tension: 0.25,
              spanGaps: true
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: 'index',
            intersect: false
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Rehabilitation Day & Date',
                font: { size: 12, weight: '600' },
                color: '#475569'
              },
              grid: { color: 'rgba(226, 232, 240, 0.6)' }
            },
            y: {
              min: 0,
              max: 100,
              title: {
                display: true,
                text: 'Active Flexion ROM (°)',
                font: { size: 12, weight: '600' },
                color: '#475569'
              },
              ticks: {
                stepSize: 20,
                color: '#475569',
                callback: function(val) {
                  return val + '°';
                }
              },
              grid: { color: 'rgba(226, 232, 240, 0.6)' }
            }
          },
          plugins: {
            legend: {
              position: 'top',
              labels: { boxWidth: 14, font: { size: 12, weight: '600' }, color: '#334155' }
            },
            tooltip: {
              callbacks: {
                title: function(items) {
                  if (!items || items.length === 0) return '';
                  const idx = items[0].dataIndex;
                  const record = records[idx];
                  const dateStr = record && record.recorded_at ? record.recorded_at.substring(0, 16) : 'Recorded Session';
                  return `Day ${record ? record.day : items[0].label} • ${dateStr}`;
                },
                label: function(context) {
                  if (context.raw === null || context.raw === undefined) {
                    return 'Range of Motion (ROM): Not measured';
                  }
                  const val = Number(context.raw).toFixed(1);
                  const idx = context.dataIndex;
                  const record = records[idx];
                  const repsText = record && record.reps ? ` (${record.reps} reps completed)` : '';
                  return `Measured Range of Motion (ROM): ${val}° flexion${repsText}`;
                }
              }
            }
          }
        }
      });
    }
  }

  // -------------------------------------------------------------
  // GRAPH 3: Mobility and Grip Strength Trend (0–3 Functional Scale)
  // -------------------------------------------------------------
  const ctxMobGrip = document.getElementById('mobilityGripChart');
  if (ctxMobGrip) {
    if (mobilityGripChartInstance) {
      mobilityGripChartInstance.data.labels = labels;
      mobilityGripChartInstance.data.datasets[0].data = mobData;
      mobilityGripChartInstance.data.datasets[0].pointRadius = pointSize;
      mobilityGripChartInstance.data.datasets[0].pointHoverRadius = pointHoverSize;
      mobilityGripChartInstance.data.datasets[1].data = gripData;
      mobilityGripChartInstance.data.datasets[1].pointRadius = pointSize;
      mobilityGripChartInstance.data.datasets[1].pointHoverRadius = pointHoverSize;
      mobilityGripChartInstance.update();
    } else {
      mobilityGripChartInstance = new Chart(ctxMobGrip.getContext('2d'), {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Mobility Limitation (0–3)',
              data: mobData,
              borderColor: '#8b5cf6',
              backgroundColor: 'rgba(139, 92, 246, 0.1)',
              borderWidth: 2.5,
              pointBackgroundColor: '#8b5cf6',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 1.5,
              pointRadius: pointSize,
              pointHoverRadius: pointHoverSize,
              tension: 0.25,
              spanGaps: true
            },
            {
              label: 'Grip Weakness (0–3)',
              data: gripData,
              borderColor: '#06b6d4',
              backgroundColor: 'rgba(6, 182, 212, 0.1)',
              borderWidth: 2.5,
              pointBackgroundColor: '#06b6d4',
              pointBorderColor: '#ffffff',
              pointBorderWidth: 1.5,
              pointRadius: pointSize,
              pointHoverRadius: pointHoverSize,
              tension: 0.25,
              spanGaps: true
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: 'index',
            intersect: false
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Rehabilitation Day & Date',
                font: { size: 12, weight: '600' },
                color: '#475569'
              },
              grid: { color: 'rgba(226, 232, 240, 0.6)' }
            },
            y: {
              min: 0,
              max: 3,
              title: {
                display: true,
                text: 'Functional Score (0–3 Scale)',
                font: { size: 12, weight: '600' },
                color: '#475569'
              },
              ticks: {
                stepSize: 1,
                color: '#475569',
                callback: function(val) {
                  const map = { 0: '0 - Normal', 1: '1 - Mild', 2: '2 - Moderate', 3: '3 - Severe' };
                  return map[val] !== undefined ? map[val] : val;
                }
              },
              grid: { color: 'rgba(226, 232, 240, 0.6)' }
            }
          },
          plugins: {
            legend: {
              position: 'top',
              labels: { boxWidth: 14, font: { size: 12, weight: '600' }, color: '#334155' }
            },
            tooltip: {
              callbacks: {
                title: function(items) {
                  if (!items || items.length === 0) return '';
                  const idx = items[0].dataIndex;
                  const record = records[idx];
                  const dateStr = record && record.recorded_at ? record.recorded_at.substring(0, 16) : 'Recorded Session';
                  return `Day ${record ? record.day : items[0].label} • ${dateStr}`;
                },
                label: function(context) {
                  if (context.raw === null || context.raw === undefined) {
                    return `${context.dataset.label}: Not recorded`;
                  }
                  const val = Number(context.raw).toFixed(1);
                  const isMob = context.datasetIndex === 0;
                  const word = isMob ? getMobilityWord(context.raw) : getGripWord(context.raw);
                  return `${context.dataset.label}: ${val} / 3 (${word})`;
                }
              }
            }
          }
        }
      });
    }
  }
}

// Backwards-compatible alias
function updateTrendChart(history) {
  updateRecoveryCharts(history);
}

// --- Longitudinal AI Forecast & Alert Handlers ---

async function requestTrendAnalysis(history) {
  const pendingNotice = document.getElementById('forecast-pending-notice');
  const pendingText = document.getElementById('forecast-pending-text');
  const pillsRow = document.getElementById('forecast-pills-row');
  const banner = document.getElementById('alert-banner');
  const title = document.getElementById('alert-title');
  const msg = document.getElementById('alert-message');
  const icon = document.getElementById('alert-icon');

  const count = (history && history.length) || 0;

  if (count < 14) {
    // Show clear explanatory pending state instead of dummy values
    if (pendingNotice) pendingNotice.classList.remove('hidden');
    if (pendingText) {
      pendingText.innerHTML = `AI 1-day forecasting requires at least 14 days of recorded sessions to identify recovery trends accurately. Currently recorded: <strong>${count} / 14 days</strong>.`;
    }
    if (pillsRow) pillsRow.classList.add('hidden');

    if (banner) {
      banner.className = 'status-banner banner-success';
      if (icon) icon.textContent = 'ℹ️';
      if (title) title.textContent = count === 0 ? 'Welcome to Your Recovery Progress' : 'Baseline Recovery In Progress';
      if (msg) {
        msg.textContent = count === 0 
          ? 'Record your first diary entry and camera exercise to begin tracking your recovery progress.'
          : `You have recorded ${count} session${count > 1 ? 's' : ''}. AI recovery trends and estimated forecasts will activate after 14 days of rehabilitation history. Continue your prescribed exercises as advised by your physiotherapist.`;
      }
    }
    return;
  }

  // When 14 or more days exist: fetch real model predictions
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
  const pendingNotice = document.getElementById('forecast-pending-notice');
  const pillsRow = document.getElementById('forecast-pills-row');

  if (pendingNotice) pendingNotice.classList.add('hidden');
  if (pillsRow) pillsRow.classList.remove('hidden');

  if (banner) {
    banner.className = `status-banner banner-${data.alert_level}`;
    if (msg) msg.textContent = data.alert;

    if (data.alert_level === 'danger') {
      if (icon) icon.textContent = '⚠️';
      if (title) title.textContent = 'Temporary Symptom Increase / Flare-Up Notice';
    } else if (data.alert_level === 'warning') {
      if (icon) icon.textContent = '⚠️';
      if (title) title.textContent = 'Movement Plateau Observed';
    } else {
      if (icon) icon.textContent = '✅';
      if (title) title.textContent = 'Steady Recovery Progress Observed';
    }
  }

  if (data.forecast) {
    const fPain = document.getElementById('f-pain');
    const fStiff = document.getElementById('f-stiff');
    const fMob = document.getElementById('f-mob');
    const fGrip = document.getElementById('f-grip');
    const fRom = document.getElementById('f-rom');

    if (fPain) fPain.textContent = `${data.forecast.pain} / 3 (Estimated)`;
    if (fStiff) fStiff.textContent = `${data.forecast.stiffness} / 3 (Estimated)`;
    if (fMob) fMob.textContent = `${data.forecast.mobility} / 3 (Estimated)`;
    if (fGrip) fGrip.textContent = `${data.forecast.grip} / 3 (Estimated)`;
    if (fRom) fRom.textContent = `${data.forecast.rom}° flexion (Estimated)`;
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

  // Apply active duration filter (all, last 14, last 7)
  let filtered = [...records];
  if (currentHistoryFilter === '14') {
    filtered = filtered.slice(-14);
  } else if (currentHistoryFilter === '7') {
    filtered = filtered.slice(-7);
  }

  if (filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-center">No recorded sessions for this filter view.</td></tr>';
    return;
  }

  // Sort order: Chronological (Day 1 -> Day N) if historySortAscending, else newest first
  filtered.sort((a, b) => {
    return historySortAscending ? (Number(a.day) - Number(b.day)) : (Number(b.day) - Number(a.day));
  });

  filtered.forEach(r => {
    const tr = document.createElement('tr');

    let dateStr = 'Today';
    if (r.recorded_at) {
      dateStr = r.recorded_at.substring(0, 16).replace('T', ' ');
    }

    // Pain cell
    const painCell = (r.pain !== null && r.pain !== undefined && r.pain !== '' && !isNaN(Number(r.pain)))
      ? `${Number(r.pain).toFixed(1)} <small>(${getSeverityWord(r.pain)})</small>`
      : '<span class="val-missing" title="Measurement not recorded">—</span>';

    // Stiffness cell
    const stiffCell = (r.stiffness !== null && r.stiffness !== undefined && r.stiffness !== '' && !isNaN(Number(r.stiffness)))
      ? `${Number(r.stiffness).toFixed(1)} <small>(${getSeverityWord(r.stiffness)})</small>`
      : '<span class="val-missing" title="Measurement not recorded">—</span>';

    // Mobility cell
    const mobCell = (r.mobility !== null && r.mobility !== undefined && r.mobility !== '' && !isNaN(Number(r.mobility)))
      ? `${Number(r.mobility).toFixed(1)} <small>(${getMobilityWord(r.mobility)})</small>`
      : '<span class="val-missing" title="Measurement not recorded">—</span>';

    // Grip cell
    const gripCell = (r.grip !== null && r.grip !== undefined && r.grip !== '' && !isNaN(Number(r.grip)))
      ? `${Number(r.grip).toFixed(1)} <small>(${getGripWord(r.grip)})</small>`
      : '<span class="val-missing" title="Measurement not recorded">—</span>';

    // Range of Motion (ROM) cell
    const romCell = (r.rom !== null && r.rom !== undefined && r.rom !== '' && !isNaN(Number(r.rom)))
      ? `<strong>${Number(r.rom).toFixed(1)}°</strong>`
      : '<span class="val-missing" title="Measurement not recorded">—</span>';

    const repsVal = (r.reps !== null && r.reps !== undefined) ? r.reps : 0;
    const notesStr = escapeHtml(r.diary_text || '—');

    tr.innerHTML = `
      <td><strong>Day ${r.day}</strong></td>
      <td>${dateStr}</td>
      <td>${painCell}</td>
      <td>${stiffCell}</td>
      <td>${mobCell}</td>
      <td>${gripCell}</td>
      <td>${romCell}</td>
      <td>${repsVal}</td>
      <td>${notesStr}</td>
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

  // Tailor Camera Exercise Guide to Patient's Rehab Condition
  updateExerciseGuide(user.surgery_type);

  // Switch from Auth screen to Dashboard
  document.getElementById('auth-screen').classList.add('hidden');
  document.getElementById('dashboard-app').classList.remove('hidden');

  // Fetch Patient History from SQLite Database
  try {
    const res = await fetch(`/api/user/history?user_id=${user.id}`);
    if (res.ok) {
      const data = await res.json();
      patientHistory = data.records || [];
      updateRecoverySummaryCards(patientHistory);
      updateRecoveryCharts(patientHistory);
      renderHistoryTable(patientHistory);
      requestTrendAnalysis(patientHistory);
      updateDayLabels();
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

function updateDayLabels() {
  const saveLabel = document.getElementById('label-save-day');
  const advanceLabel = document.getElementById('label-advance-day');
  const diarySaveLabel = document.getElementById('label-diary-save-day');
  const diaryAdvanceLabel = document.getElementById('label-diary-advance-day');

  const todayStr = new Date().toISOString().substring(0, 10);
  let saveText = 'Day 1';
  let advanceText = 'Day 2';

  if (patientHistory && patientHistory.length > 0) {
    const lastRecord = patientHistory[patientHistory.length - 1];
    const lastDate = lastRecord.recorded_at ? lastRecord.recorded_at.substring(0, 10) : '';

    if (lastRecord.day === 0) {
      // Day 0 is registration baseline; first rehabilitation session is Day 1
      saveText = 'Day 1';
      advanceText = 'Day 2';
    } else if (lastDate === todayStr) {
      saveText = `Day ${lastRecord.day}`;
      advanceText = `Day ${lastRecord.day + 1}`;
    } else {
      const nextDay = lastRecord.day + 1;
      saveText = `Day ${nextDay}`;
      advanceText = `Day ${nextDay + 1}`;
    }
  }

  if (saveLabel) saveLabel.textContent = saveText;
  if (advanceLabel) advanceLabel.textContent = advanceText;
  if (diarySaveLabel) diarySaveLabel.textContent = saveText;
  if (diaryAdvanceLabel) diaryAdvanceLabel.textContent = advanceText;
}

// --- Session Recording ---

async function saveSession(advanceDay = false) {
  if (!currentUser) {
    alert('Please sign in to save your session.');
    return;
  }

  // If user entered text in diary-input but hasn't run analysis yet, analyze it now
  const diaryInputEl = document.getElementById('diary-input');
  const inputVal = diaryInputEl ? diaryInputEl.value.trim() : '';
  if (inputVal && (!currentDiaryPred.text || currentDiaryPred.text !== inputVal)) {
    try {
      await analyzeDiaryText();
    } catch (e) {
      console.warn('Auto-analysis before save failed:', e);
    }
  }

  const todayStr = new Date().toISOString().substring(0, 10);
  let targetDay = 1;
  let updateIfSame = false;

  if (patientHistory && patientHistory.length > 0) {
    const lastRecord = patientHistory[patientHistory.length - 1];
    const lastDate = lastRecord.recorded_at ? lastRecord.recorded_at.substring(0, 10) : '';

    if (lastRecord.day === 0) {
      // Day 0 is baseline registration; first exercise session is Day 1
      targetDay = advanceDay ? 2 : 1;
      updateIfSame = false;
    } else if (lastDate === todayStr) {
      if (advanceDay) {
        targetDay = lastRecord.day + 1;
        updateIfSame = false;
      } else {
        targetDay = lastRecord.day;
        updateIfSame = true;
      }
    } else {
      const nextDay = lastRecord.day + 1;
      targetDay = advanceDay ? nextDay + 1 : nextDay;
      updateIfSame = false;
    }
  }

  const diaryText = (currentDiaryPred.text && currentDiaryPred.text.trim())
    ? currentDiaryPred.text
    : (inputVal || 'Daily exercise and finger curl training.');

  const payload = {
    user_id: currentUser.id,
    day: targetDay,
    pain: currentDiaryPred.pain,
    stiffness: currentDiaryPred.stiffness,
    mobility: currentDiaryPred.mobility,
    grip: currentDiaryPred.grip,
    rom: currentROM,
    reps: repCount,
    diary_text: diaryText,
    update_if_same_day: updateIfSame
  };

  try {
    const res = await fetch('/api/user/record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error('Failed to record session.');
    const data = await res.json();

    if (data.updated) {
      alert(`Success! Updated today's Day ${targetDay} session for ${currentUser.full_name}.\nPeak ROM and completed reps were accumulated for today.`);
    } else {
      alert(`Success! Recorded Day ${targetDay} session for ${currentUser.full_name}.`);
    }

    // Reset rep counter for next session
    repCount = 0;
    updateVisionUI();

    // Reload history
    const histRes = await fetch(`/api/user/history?user_id=${currentUser.id}`);
    if (histRes.ok) {
      const histData = await histRes.json();
      patientHistory = histData.records || [];
      updateRecoverySummaryCards(patientHistory);
      updateRecoveryCharts(patientHistory);
      renderHistoryTable(patientHistory);
      requestTrendAnalysis(patientHistory);
      updateDayLabels();
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

  // Initialize summary cards and charts in zero-session state until patient session loads
  updateRecoverySummaryCards([]);
  updateRecoveryCharts([]);

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

  // Condition Hint & Dynamic Exercise Guide
  const regSurgerySelect = document.getElementById('reg-surgery');
  if (regSurgerySelect) {
    regSurgerySelect.addEventListener('change', (e) => updateRegistrationHint(e.target.value));
    updateRegistrationHint(regSurgerySelect.value);
  }

  const guideConditionSelect = document.getElementById('guide-condition-select');
  if (guideConditionSelect) {
    guideConditionSelect.addEventListener('change', (e) => updateExerciseGuide(e.target.value));
  }

  // Pre-seed default exercise guide
  updateExerciseGuide('flexor');

  // Step Action Transitions
  document.getElementById('btn-goto-exercise').addEventListener('click', () => switchView('camera'));
  document.getElementById('btn-goto-report').addEventListener('click', () => switchView('report'));
  const diaryGotoReport = document.getElementById('btn-diary-goto-report');
  if (diaryGotoReport) {
    diaryGotoReport.addEventListener('click', () => switchView('report'));
  }
  document.getElementById('btn-refresh-history').addEventListener('click', () => {
    if (currentUser) {
      fetch(`/api/user/history?user_id=${currentUser.id}`)
        .then(r => r.json())
        .then(d => {
          patientHistory = d.records || [];
          updateRecoverySummaryCards(patientHistory);
          updateRecoveryCharts(patientHistory);
          renderHistoryTable(patientHistory);
          requestTrendAnalysis(patientHistory);
          updateDayLabels();
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
  document.getElementById('btn-save-session').addEventListener('click', () => saveSession(false));
  const advanceBtn = document.getElementById('btn-advance-day');
  if (advanceBtn) {
    advanceBtn.addEventListener('click', () => saveSession(true));
  }

  const diarySaveBtn = document.getElementById('btn-diary-save-session');
  if (diarySaveBtn) {
    diarySaveBtn.addEventListener('click', () => saveSession(false));
  }
  const diaryAdvanceBtn = document.getElementById('btn-diary-advance-day');
  if (diaryAdvanceBtn) {
    diaryAdvanceBtn.addEventListener('click', () => saveSession(true));
  }

  // 7. History Filter & Sort Controls
  ['all', '14', '7'].forEach(filterKey => {
    const btn = document.getElementById(`filter-${filterKey}`);
    if (btn) {
      btn.addEventListener('click', () => {
        currentHistoryFilter = filterKey;
        document.querySelectorAll('.history-filter-group .btn-filter').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderHistoryTable(patientHistory);
      });
    }
  });

  const sortBtn = document.getElementById('btn-sort-history');
  const sortLabel = document.getElementById('sort-order-label');
  if (sortBtn) {
    sortBtn.addEventListener('click', () => {
      historySortAscending = !historySortAscending;
      if (sortLabel) {
        sortLabel.textContent = historySortAscending ? 'Oldest First' : 'Newest First';
      }
      renderHistoryTable(patientHistory);
    });
  }

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
