// ============================================================
// PLAM SCRIPT - CLIENT ENGINE
// ============================================================

const video = document.getElementById('webcam');
const captureCanvas = document.getElementById('capture-canvas');
const captureCtx = captureCanvas.getContext('2d');
const ttsPlayer = document.getElementById('tts-player');

// UI Elements
const liveSignText = document.getElementById('live-sign-text');
const liveConfidence = document.getElementById('live-confidence');
const predBlock = document.getElementById('pred-block');
const centerCoords = document.getElementById('center-coords');
const liveCoords = document.getElementById('livecoords');
const statBuffer = document.getElementById('stat-buffer');
const statPrediction = document.getElementById('stat-prediction');
const handStatus = document.getElementById('hand-status');
const fpsDisplay = document.getElementById('fps-display');
const camBox = document.getElementById('cam-box');
const wsDot = document.getElementById('ws-dot');

const btnSound = document.getElementById('btn-sound');
const btnClear = document.getElementById('btn-clear');
const btnMode = document.getElementById('btn-mode');
const btnMic = document.getElementById('btn-mic');
const tokenRibbon = document.getElementById('token-ribbon');
const modeBadge = document.getElementById('mode-badge');
const person2Box = document.getElementById('person2-box');
const person2Text = document.getElementById('person2-text');

let ws = null;
let isSoundOn = true;
let isAudioUnlocked = false;
let isProcessingFrame = false;
let isSentenceMode = true; // True: Sentence Stream mode, False: Instant Word mode
let sentenceTokens = [];
let isRecordingMic = false;
let speechRecognizer = null;
let frameCount = 0;
let lastFpsTime = performance.now();

// ============================================================
// 1. AUTOMATIC AUDIO UNLOCK (NO MANUAL BUTTON REQUIRED)
// ============================================================
function unlockAudio() {
  if (isAudioUnlocked) return;

  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (AudioCtx) {
      const ctx = new AudioCtx();
      if (ctx.state === 'suspended') {
        ctx.resume();
      }
    }
  } catch (e) {}

  ttsPlayer.src = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";
  ttsPlayer.play().then(() => {
    isAudioUnlocked = true;
    console.log("🔊 Audio auto-unlocked successfully!");
    ['click', 'touchstart', 'touchend', 'pointerdown', 'keydown', 'wheel'].forEach(evt => {
      window.removeEventListener(evt, unlockAudio);
    });
  }).catch(() => {
    // Will retry seamlessly on next interaction
  });
}

// Automatically listen on any page interaction to activate audio
['click', 'touchstart', 'touchend', 'pointerdown', 'keydown', 'wheel'].forEach(evt => {
  window.addEventListener(evt, unlockAudio, { passive: true });
});

// Attempt immediate auto-unlock on launch
setTimeout(unlockAudio, 300);

// Single Word Speaker
function speakWord(word) {
  if (!isSoundOn) return;
  unlockAudio();

  const clean = word.toLowerCase().replace(/[^a-z0-9]/g, '_');
  const audioUrl = `/audio/${clean}.mp3`;
  
  ttsPlayer.src = audioUrl;
  const promise = ttsPlayer.play();
  
  if (promise !== undefined) {
    promise.catch(() => {
      // Fallback: Web Speech API
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(word);
        utter.rate = 1.0;
        window.speechSynthesis.speak(utter);
      }
    });
  }
}

// Full Sentence Speaker (Edge-TTS endpoint with Web Speech fallback)
function speakSentence(text) {
  if (!isSoundOn || !text) return;
  unlockAudio();

  const encoded = encodeURIComponent(text);
  const audioUrl = `/api/tts/${encoded}`;
  
  ttsPlayer.src = audioUrl;
  const promise = ttsPlayer.play();
  
  if (promise !== undefined) {
    promise.catch(() => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.0;
        window.speechSynthesis.speak(utter);
      }
    });
  }
}

// Render horizontal token ribbon chips
function updateRibbonUI() {
  if (!tokenRibbon) return;
  if (sentenceTokens.length === 0) {
    tokenRibbon.innerHTML = '<span class="token-placeholder">Perform signs to build sentence...</span>';
    return;
  }

  tokenRibbon.innerHTML = sentenceTokens.map((tok, idx) => {
    const isLast = (idx === sentenceTokens.length - 1);
    return `<div class="token-chip">${tok.toUpperCase()}${!isLast ? '<span class="arrow">➔</span>' : ''}</div>`;
  }).join('');
  
  tokenRibbon.scrollLeft = tokenRibbon.scrollWidth;
}

// ============================================================
// 2. WEBCAM SETUP
// ============================================================
async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      if (video.videoWidth && video.videoHeight) {
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        console.log(`📐 Capture resolution matched: ${video.videoWidth}x${video.videoHeight}`);
      }
    };
    console.log("📷 Camera connected.");
    unlockAudio();
  } catch (err) {
    console.error("Camera error:", err);
    liveSignText.textContent = "CAM ERROR";
    if (centerCoords) centerCoords.textContent = "PERMISSION DENIED";
  }
}

// ============================================================
// 3. WEBSOCKET WITH FLOW-CONTROL (NO QUEUE LAG)
// ============================================================
function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

  ws.onopen = () => {
    console.log("🔌 Connected to server.");
    wsDot.style.background = "#00ff66";
    isProcessingFrame = false;
    sendFrame();
  };

  ws.onmessage = (event) => {
    isProcessingFrame = false; // READY FOR NEXT FRAME

    try {
      const data = JSON.parse(event.data);
      handleServerResponse(data);
    } catch (e) {
      console.error(e);
    }

    // Immediately trigger next frame
    requestAnimationFrame(sendFrame);
  };

  ws.onclose = () => {
    wsDot.style.background = "#ff3344";
    isProcessingFrame = false;
    setTimeout(connectWebSocket, 2000);
  };
}

function sendFrame() {
  if (isProcessingFrame || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (video.readyState < video.HAVE_CURRENT_DATA) {
    requestAnimationFrame(sendFrame);
    return;
  }

  isProcessingFrame = true;
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  const jpeg = captureCanvas.toDataURL('image/jpeg', 0.75);
  ws.send(jpeg);

  // FPS calculation
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsDisplay.textContent = frameCount;
    frameCount = 0;
    lastFpsTime = now;
  }
}

function handleServerResponse(data) {
  // Hand tracking feedback
  if (data.hand_detected) {
    handStatus.textContent = "YES";
    handStatus.style.color = "#00ff66";
    camBox.style.borderColor = "#00ff66";
  } else {
    handStatus.textContent = "NO";
    handStatus.style.color = "#666666";
    camBox.style.borderColor = "#555555";
  }

  // Buffer progress in Left Box
  const buf = data.buffer_length || 0;
  statBuffer.textContent = `BUFFER: ${buf} / 30 FRAMES`;

  const isValidSign = data.sign && !['Signing...', 'Detecting...', 'READY', 'UNCERTAIN'].includes(data.sign);

  if (isValidSign) {
    const confPct = (data.confidence * 100).toFixed(1);
    if (liveConfidence) {
      liveConfidence.textContent = `${confPct}%`;
      liveConfidence.style.color = data.confidence >= 0.70 ? "#00ff66" : "#ffb700";
    }
    if (predBlock) predBlock.classList.add('detected');

    if (isSentenceMode) {
      // --- SENTENCE STREAM MODE ---
      if (data.is_new_sign) {
        const lastTok = sentenceTokens[sentenceTokens.length - 1];
        if (lastTok !== data.sign) {
          sentenceTokens.push(data.sign);
          updateRibbonUI();
          liveSignText.textContent = data.sign.toUpperCase();
          statPrediction.textContent = `TOKEN: ${data.sign.toUpperCase()}`;
        }
      } else {
        liveSignText.textContent = data.sign.toUpperCase();
      }
    } else {
      // --- INSTANT WORD MODE ---
      liveSignText.textContent = data.sign.toUpperCase();
      if (data.trigger_speech) {
        statPrediction.textContent = `SPOKEN: ${data.sign.toUpperCase()}`;
        speakWord(data.sign);
      } else {
        statPrediction.textContent = `LOCKED: ${data.sign.toUpperCase()}`;
      }
    }
  } else if (data.hand_detected) {
    if (data.sign) {
      liveSignText.textContent = data.sign.toUpperCase();
    }
    if (liveConfidence) {
      liveConfidence.textContent = data.confidence > 0 ? `${(data.confidence * 100).toFixed(1)}%` : "...";
      liveConfidence.style.color = "#aaaaaa";
    }
    if (predBlock) predBlock.classList.remove('detected');
    statPrediction.textContent = "TRACKING...";
  } else {
    // Hand not in view
    if (liveConfidence) {
      liveConfidence.textContent = "0.0%";
      liveConfidence.style.color = "#666666";
    }
    if (predBlock) predBlock.classList.remove('detected');

    // Sentence Completion on Pause / Hand Drop
    if (isSentenceMode && sentenceTokens.length > 0 && (data.no_hand_frames >= 20 || !data.hand_detected)) {
      const fullSentence = sentenceTokens.join(' ');
      liveSignText.textContent = `"${fullSentence.toUpperCase()}"`;
      statPrediction.textContent = `PHRASE SENT: ${fullSentence.toUpperCase()}`;
      speakSentence(fullSentence);
      sentenceTokens = [];
      setTimeout(() => {
        updateRibbonUI();
      }, 1500);
    } else if (!isSentenceMode || sentenceTokens.length === 0) {
      liveSignText.textContent = "READY";
      statPrediction.textContent = "EDGE-TTS READY";
    }
  }
}

// ============================================================
// TWO-WAY SPEECH-TO-TEXT FOR PERSON 2 (HEARING)
// ============================================================
function initSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    alert("Speech Recognition is supported natively in Google Chrome or Microsoft Edge. Please open this app in Chrome/Edge.");
    return null;
  }

  const rec = new SpeechRec();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = 'en-US';

  rec.onstart = () => {
    isRecordingMic = true;
    btnMic.classList.add('recording');
    btnMic.textContent = "🔴 LISTENING... (STOP)";
    if (person2Box) person2Box.style.display = "flex";
    if (person2Text) person2Text.textContent = "Listening to Person 2...";
  };

  rec.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript;
    }
    if (transcript.trim() && person2Text) {
      person2Text.textContent = `"${transcript.trim()}"`;
    }
  };

  rec.onerror = (e) => {
    console.warn("STT Error:", e.error);
    if (e.error !== 'no-speech') {
      stopMic();
    }
  };

  rec.onend = () => {
    if (isRecordingMic) {
      try { rec.start(); } catch(e) {}
    } else {
      btnMic.classList.remove('recording');
      btnMic.textContent = "🎤 TALK BACK (MIC)";
    }
  };

  return rec;
}

function startMic() {
  unlockAudio();
  if (!speechRecognizer) {
    speechRecognizer = initSpeechRecognition();
  }
  if (speechRecognizer) {
    try {
      isRecordingMic = true;
      speechRecognizer.start();
    } catch(e) {}
  }
}

function stopMic() {
  isRecordingMic = false;
  if (speechRecognizer) {
    try { speechRecognizer.stop(); } catch(e) {}
  }
  btnMic.classList.remove('recording');
  btnMic.textContent = "🎤 TALK BACK (MIC)";
}

// Controls
btnMode.addEventListener('click', () => {
  isSentenceMode = !isSentenceMode;
  if (isSentenceMode) {
    btnMode.textContent = "⚡ MODE: SENTENCE";
    btnMode.className = "btn active";
    modeBadge.textContent = "STREAM MODE";
    modeBadge.style.color = "#00ff66";
    modeBadge.style.borderColor = "#00ff66";
  } else {
    btnMode.textContent = "⚡ MODE: INSTANT WORD";
    btnMode.className = "btn";
    modeBadge.textContent = "WORD MODE";
    modeBadge.style.color = "#00ffff";
    modeBadge.style.borderColor = "#00ffff";
    sentenceTokens = [];
    updateRibbonUI();
  }
});

btnMic.addEventListener('click', () => {
  if (isRecordingMic) {
    stopMic();
  } else {
    startMic();
  }
});

btnSound.addEventListener('click', () => {
  isSoundOn = !isSoundOn;
  if (isSoundOn) {
    btnSound.textContent = "🔊 SOUND: ON";
    btnSound.className = "btn active";
  } else {
    btnSound.textContent = "🔇 SOUND: MUTED";
    btnSound.className = "btn";
  }
});

btnClear.addEventListener('click', () => {
  sentenceTokens = [];
  updateRibbonUI();
  liveSignText.textContent = "READY";
  if (liveConfidence) liveConfidence.textContent = "0.0%";
  if (predBlock) predBlock.classList.remove('detected');
  if (person2Box) person2Box.style.display = "none";
});

// ============================================================
// 4. EXACT PALMSCRIPT HUD CANVAS ANIMATION
// ============================================================
const canvas = document.getElementById('hud');
const ctx = canvas.getContext('2d');
const cx = 260, cy = 260;
let tick = 0;

function drawArc(x, y, r, startDeg, endDeg, width, color, dash=[]) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, r, startDeg * Math.PI/180, endDeg * Math.PI/180);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.stroke();
  ctx.restore();
}

function drawTick(x, y, r, angleDeg, len, width, color) {
  const a = angleDeg * Math.PI / 180;
  const x1 = x + r * Math.cos(a);
  const y1 = y + r * Math.sin(a);
  const x2 = x + (r + len) * Math.cos(a);
  const y2 = y + (r + len) * Math.sin(a);
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  ctx.restore();
}

function drawRotatingRing(angleDeg, r, segCount, segGap, width, color, reverse=false) {
  const total = 360;
  const segSize = (total - segCount * segGap) / segCount;
  for (let i = 0; i < segCount; i++) {
    const start = i * (segSize + segGap) + (reverse ? -angleDeg : angleDeg);
    const end = start + segSize;
    drawArc(cx, cy, r, start - 90, end - 90, width, color);
  }
}

function drawTickRing(angleDeg, r, count, len, width, color, reverse=false) {
  const step = 360 / count;
  for (let i = 0; i < count; i++) {
    const angle = i * step + (reverse ? -angleDeg : angleDeg) - 90;
    drawTick(cx, cy, r, angle, len, width, color);
  }
}

function drawPalmScriptText(angleDeg, r) {
  const text = 'PLAM  SCRIPT';
  const chars = text.split('');
  const total = chars.length;
  const arcLen = 200;
  const startAngle = -110 + angleDeg * 0.5;

  ctx.save();
  chars.forEach((ch, i) => {
    const angle = (startAngle + (i / total) * arcLen) * Math.PI / 180;
    const tx = cx + r * Math.cos(angle);
    const ty = cy + r * Math.sin(angle);
    ctx.save();
    ctx.translate(tx, ty);
    ctx.rotate(angle + Math.PI / 2);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 9px Courier New';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(ch, 0, 0);
    ctx.restore();
  });
  ctx.restore();
}

function drawScaleNumbers(angleDeg, r, offset=0) {
  const nums = ['20', '10', '5', '1'];
  const angles = [-75, -45, 0, 45];
  ctx.save();
  ctx.fillStyle = '#888';
  ctx.font = '8px Courier New';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  nums.forEach((n, i) => {
    const angle = (angles[i] + angleDeg * 0.3 + offset) * Math.PI / 180;
    const tx = cx + r * Math.cos(angle);
    const ty = cy + r * Math.sin(angle);
    ctx.save();
    ctx.translate(tx, ty);
    ctx.fillText(n, 0, 0);
    ctx.restore();
  });
  ctx.restore();
}

function drawSmallCircle(ox, oy, r, label1, label2) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(ox, oy, r, 0, Math.PI * 2);
  ctx.strokeStyle = '#555';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(ox, oy, r * 0.6, 0, Math.PI * 2);
  ctx.strokeStyle = '#444';
  ctx.lineWidth = 0.5;
  ctx.stroke();

  ctx.fillStyle = '#777';
  ctx.font = '7px Courier New';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label1, ox, oy - 4);
  ctx.fillText(label2, ox, oy + 5);
  ctx.restore();
}

function drawTriangle(ox, oy, size, color) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(ox, oy - size);
  ctx.lineTo(ox + size * 0.8, oy + size * 0.5);
  ctx.lineTo(ox - size * 0.8, oy + size * 0.5);
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.2;
  ctx.stroke();
  ctx.restore();
}

function draw() {
  ctx.clearRect(0, 0, 520, 520);
  tick += 0.3;

  const cw = tick * 0.8;
  const ccw = -tick * 0.5;

  // Outermost thin arc
  drawArc(cx, cy, 240, -160, 80, 0.5, '#333');

  // Outer scale ticks (clockwise slow)
  drawTickRing(cw * 0.2, 228, 60, 5, 0.5, '#444');
  drawTickRing(cw * 0.2, 228, 12, 10, 1, '#666');
  drawScaleNumbers(cw * 0.2, 210);

  // Outer white segment ring (clockwise)
  drawRotatingRing(cw, 200, 8, 8, 14, '#fff');

  // Second ring segments (anticlockwise)
  drawRotatingRing(ccw, 178, 12, 4, 6, '#aaa');

  // PLAM SCRIPT text arc (slow clockwise)
  drawPalmScriptText(cw * 0.15, 160);

  // Dashed ring
  drawArc(cx, cy, 148, -180, 180, 0.5, '#444', [4, 4]);

  // Inner segment ring (clockwise faster)
  drawRotatingRing(cw * 1.5, 130, 20, 3, 4, '#777');

  // Fine tick ring (anticlockwise)
  drawTickRing(ccw * 1.2, 118, 48, 6, 0.5, '#555');

  // Inner circles
  drawArc(cx, cy, 85, -180, 180, 1, '#555');
  drawArc(cx, cy, 60, -180, 180, 0.5, '#333');

  // Small decorative circles
  drawSmallCircle(cx - 90, cy, 22, 'LSPd', 'LUNNE');
  drawTriangle(cx - 90, cy, 8, '#fff');

  // Live coords update
  const angle = tick * 0.01;
  const lat = (18 + Math.sin(angle) * 0.5).toFixed(8);
  const lon = (22 + Math.cos(angle) * 0.3).toFixed(8);
  if (centerCoords) centerCoords.textContent = `${lat} / ${lon}`;
  if (liveCoords) liveCoords.textContent = `${(10 + Math.sin(angle*1.3)*0.2).toFixed(8)} / ${(19 + Math.cos(angle*1.1)*0.1).toFixed(8)}`;

  requestAnimationFrame(draw);
}

// START
initWebcam();
connectWebSocket();
draw();
