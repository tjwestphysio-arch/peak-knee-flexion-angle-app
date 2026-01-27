let detector;
let video;
let canvas, ctx;

let currentLeg = null; // 'left' or 'right'
let trialCount = 0;
const maxTrials = 3;

const leftTrials = [];
const rightTrials = [];

// Landing detection state
let lastAnkleY = null;
let lastVel = null;
let lastAngle = null;
let lastTime = null;
let landingArmed = false;
let peakWindow = null;

// One-Euro-like simple smoothing
let lastFilteredAngle = null;

let currentFacing = 'user'; // 'user' (front) or 'environment' (rear)

window.onload = initUI;

function initUI() {
  video = document.getElementById('video');
  canvas = document.getElementById('overlay');
  ctx = canvas.getContext('2d');

  // UI elements
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const flipBtn = document.getElementById('flipBtn');
  const leftBtn = document.getElementById('leftLegBtn');
  const rightBtn = document.getElementById('rightLegBtn');

  startBtn.onclick = async () => {
    startBtn.disabled = true;
    showLoading();
    await startCamera(currentFacing);
    await setupDetector();
    hideLoading();
    stopBtn.disabled = false;
    flipBtn.disabled = false;
    leftBtn.disabled = false;
    rightBtn.disabled = false;
    document.getElementById('status').textContent = 'Ready – perform jump landings';
    requestAnimationFrame(loop);
  };

  stopBtn.onclick = () => {
    stopCamera();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    flipBtn.disabled = true;
    leftBtn.disabled = true;
    rightBtn.disabled = true;
    document.getElementById('status').textContent = 'Camera stopped';
  };

  flipBtn.onclick = async () => {
    currentFacing = currentFacing === 'user' ? 'environment' : 'user';
    // restart camera with new facing
    showLoading();
    await startCamera(currentFacing);
    hideLoading();
    document.getElementById('status').textContent = `Camera flipped to ${currentFacing}`;
  };

  leftBtn.onclick = () => startLeg('left');
  rightBtn.onclick = () => startLeg('right');

  // Try auto-starting camera; if blocked, user can press "Start Camera"
  (async () => {
    try {
      showLoading();
      await startCamera(currentFacing);
      await setupDetector();
      hideLoading();
      stopBtn.disabled = false;
      flipBtn.disabled = false;
      leftBtn.disabled = false;
      rightBtn.disabled = false;
      document.getElementById('status').textContent = 'Ready – perform jump landings';
      requestAnimationFrame(loop);
    } catch (err) {
      // Autostart may fail in some browsers without user gesture; leave UI to start manually
      hideLoading();
      console.warn('Autostart failed (user gesture may be required):', err);
      document.getElementById('startBtn').disabled = false;
      document.getElementById('status').textContent = 'Press "Start Camera" to begin';
    }
  })();
}

function showLoading() {
  const el = document.getElementById('loading');
  if (el) el.style.display = 'flex';
}

function hideLoading() {
  const el = document.getElementById('loading');
  if (el) el.style.display = 'none';
}

async function startCamera(facing = 'user') {
  // Stop existing tracks if any
  stopCamera();

  const constraints = {
    audio: false,
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: { ideal: facing } // 'user' for front camera
    }
  };

  const stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;

  // Wait until the video has dimensions
  await video.play();

  // Ensure canvas matches actual video dimensions
  resizeCanvasToVideo();
}

function stopCamera() {
  if (video && video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(t => t.stop());
    video.srcObject = null;
  }
}

function resizeCanvasToVideo() {
  const w = video.videoWidth || video.clientWidth || 480;
  const h = video.videoHeight || video.clientHeight || 360;

  canvas.width = w;
  canvas.height = h;

  // Ensure CSS size covers the container (we use object-fit: cover)
  canvas.style.width = '100%';
  canvas.style.height = '100%';
}

async function setupDetector() {
  // Create the MoveNet detector (same as before)
  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER
    }
  );
}

// Existing startLeg and related logic preserved, but ensure UI updates remain
function startLeg(leg) {
  currentLeg = leg;
  trialCount = 0;
  document.getElementById('currentLeg').textContent = leg === 'left' ? 'Left' : 'Right';
  document.getElementById('trialCount').textContent = '0';
  document.getElementById('status').textContent = 'Ready – perform jump landings';
  landingArmed = true;
  peakWindow = null;
}

function loop(timestamp) {
  requestAnimationFrame(loop);
  if (!detector || video.readyState < 2) return;

  detector.estimatePoses(video).then(poses => {
    if (!poses.length) return;
    const pose = poses[0];

    drawSkeleton(pose);

    if (!currentLeg) return;

    const angle = computeKneeAngle(pose, currentLeg);
    if (!angle) return;

    const t = performance.now() / 1000;
    const smoothAngle = smooth(angle, t);
    document.getElementById('angleDisplay').textContent = smoothAngle.toFixed(1);

    const ankle = getKeypoint(pose, currentLeg === 'left' ? 'left_ankle' : 'right_ankle');
    if (!ankle) return;

    handleLandingAndTrials(ankle.y, smoothAngle, t);
  });
}

// --- Pose helpers ---

function getKeypoint(pose, name) {
  return pose.keypoints.find(kp => kp.name === name && kp.score > 0.3);
}

function computeKneeAngle(pose, leg) {
  const hip = getKeypoint(pose, leg === 'left' ? 'left_hip' : 'right_hip');
  const knee = getKeypoint(pose, leg === 'left' ? 'left_knee' : 'right_knee');
  const ankle = getKeypoint(pose, leg === 'left' ? 'left_ankle' : 'right_ankle');
  if (!hip || !knee || !ankle) return null;

  const thigh = { x: hip.x - knee.x, y: hip.y - knee.y };
  const shank = { x: ankle.x - knee.x, y: ankle.y - knee.y };

  const dot = thigh.x * shank.x + thigh.y * shank.y;
  const mag1 = Math.hypot(thigh.x, thigh.y);
  const mag2 = Math.hypot(shank.x, shank.y);
  if (!mag1 || !mag2) return null;

  let cos = dot / (mag1 * mag2);
  cos = Math.max(-1, Math.min(1, cos));
  const angleRad = Math.acos(cos);
  return angleRad * 180 / Math.PI;
}

// --- Simple smoothing ---

function smooth(value, t) {
  if (lastFilteredAngle == null) {
    lastFilteredAngle = value;
    lastTime = t;
    return value;
  }
  const alpha = 0.3; // simple EMA
  lastFilteredAngle = alpha * value + (1 - alpha) * lastFilteredAngle;
  return lastFilteredAngle;
}

// --- Landing detection + trials ---

function handleLandingAndTrials(ankleY, angle, t) {
  if (lastAnkleY == null) {
    lastAnkleY = ankleY;
    lastAngle = angle;
    lastTime = t;
    return;
  }

  const dt = t - lastTime;
  if (dt <= 0) return;

  const vel = (ankleY - lastAnkleY) / dt;
  const acc = (vel - (lastVel ?? vel)) / dt;
  const kneeRate = (angle - lastAngle) / dt;

  const wasDescending = (lastVel ?? 0) < -0.2;
  const nowNearZero = vel > -0.05;
  const impactSpike = acc > 0.3;
  const rapidFlexion = kneeRate > 80;

  if (landingArmed && wasDescending && nowNearZero && impactSpike && rapidFlexion) {
    // Start peak window
    peakWindow = {
      start: t,
      peak: angle
    };
    landingArmed = false;
    document.getElementById('status').textContent = 'Landing detected – capturing peak flexion';
  }

  if (peakWindow) {
    const elapsed = t - peakWindow.start;
    if (elapsed <= 0.75) {
      peakWindow.peak = Math.max(peakWindow.peak, angle);
    } else {
      // End of window – record trial
      recordTrial(peakWindow.peak);
      peakWindow = null;
      landingArmed = true;
    }
  }

  lastAnkleY = ankleY;
  lastVel = vel;
  lastAngle = angle;
  lastTime = t;
}

function recordTrial(peak) {
  trialCount++;
  document.getElementById('trialCount').textContent = trialCount.toString();

  if (currentLeg === 'left') {
    leftTrials.push(peak);
  } else if (currentLeg === 'right') {
    rightTrials.push(peak);
  }

  document.getElementById('status').textContent =
    `Recorded trial ${trialCount} for ${currentLeg} leg: ${peak.toFixed(1)}°`;

  if (trialCount >= maxTrials) {
    document.getElementById('status').textContent =
      `Completed ${maxTrials} trials for ${currentLeg} leg.`;
    trialCount = 0;
    currentLeg = null;
    document.getElementById('currentLeg').textContent = 'None';
    showResults();
  }
}

function showResults() {
  const resEl = document.getElementById('results');

  const leftMax = leftTrials.length ? Math.max(...leftTrials) : 0;
  const rightMax = rightTrials.length ? Math.max(...rightTrials) : 0;
  const leftAvg = leftTrials.length ? avg(leftTrials) : 0;
  const rightAvg = rightTrials.length ? avg(rightTrials) : 0;

  const text = `
Left trials:  ${leftTrials.map(v => v.toFixed(1)).join(', ')}
Right trials: ${rightTrials.map(v => v.toFixed(1)).join(', ')}

Left max:  ${leftMax.toFixed(1)}°
Right max: ${rightMax.toFixed(1)}°

Left avg:  ${leftAvg.toFixed(1)}°
Right avg: ${rightAvg.toFixed(1)}°
`;
  resEl.textContent = text;
}

function avg(arr) {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// --- Drawing ---

function drawSkeleton(pose) {
  if (!canvas.width || !canvas.height) resizeCanvasToVideo();

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const pairs = [
    ['left_shoulder','left_elbow'],
    ['left_elbow','left_wrist'],
    ['right_shoulder','right_elbow'],
    ['right_elbow','right_wrist'],
    ['left_hip','left_knee'],
    ['left_knee','left_ankle'],
    ['right_hip','right_knee'],
    ['right_knee','right_ankle'],
    ['left_shoulder','right_shoulder'],
    ['left_hip','right_hip']
  ];

  ctx.lineWidth = 3;
  ctx.strokeStyle = 'lime';
  ctx.fillStyle = 'yellow';

  // joints
  pose.keypoints.forEach(kp => {
    if (kp.score < 0.3) return;
    const x = kp.x;
    const y = kp.y;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  // bones
  pairs.forEach(([a, b]) => {
    const p1 = pose.keypoints.find(kp => kp.name === a && kp.score > 0.3);
    const p2 = pose.keypoints.find(kp => kp.name === b && kp.score > 0.3);
    if (!p1 || !p2) return;
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  });
}