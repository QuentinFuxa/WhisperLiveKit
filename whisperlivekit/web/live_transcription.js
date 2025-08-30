/* Theme, WebSocket, recording, rendering logic extracted from inline script and adapted for segmented theme control and WS caption */

let isRecording = false;
let websocket = null;
let recorder = null;
let chunkDuration = 100;
let websocketUrl = "ws://localhost:8000/asr";
let userClosing = false;
let wakeLock = null;
let wakeLockVideo = null;
let selectedMicrophoneId = null;
let availableMicrophones = [];
let startTime = null;
let timerInterval = null;
let audioContext = null;
let analyser = null;
let microphone = null;
let waveCanvas = document.getElementById("waveCanvas");
let waveCtx = waveCanvas.getContext("2d");
let animationFrame = null;
let waitingForStop = false;
let lastReceivedData = null;
let lastSignature = null;

waveCanvas.width = 120 * (window.devicePixelRatio || 1);
waveCanvas.height = 120 * (window.devicePixelRatio || 1);
waveCtx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

const statusText = document.getElementById("status");
const recordButton = document.getElementById("recordButton");
const actionText = document.querySelector(".action-text");
const chunkSelector = document.getElementById("chunkSelector");
const websocketInput = document.getElementById("websocketInput");
const websocketDefaultSpan = document.getElementById("wsDefaultUrl");
const linesTranscriptDiv = document.getElementById("linesTranscript");
const timerElement = document.querySelector(".timer");
const themeRadios = document.querySelectorAll('input[name="theme"]');
const scrollToBottomBtn = document.getElementById("scrollToBottom");
const headerControl = document.getElementById("headerControl");
const headerRecordButton = document.getElementById("headerRecordButton");
const headerTimer = document.querySelector(".header-timer");
const topNav = document.querySelector(".top-nav");
const centralHub = document.querySelector(".central-hub");
const transcriptFlow = document.querySelector(".transcript-flow");
const microphoneSelect = document.getElementById("microphoneSelect");
const refreshMicsButton = document.getElementById("refreshMicsButton");

function getWaveStroke() {
  const styles = getComputedStyle(document.documentElement);
  const v = styles.getPropertyValue("--wave-stroke").trim();
  return v || "#000";
}

let waveStroke = getWaveStroke();
function updateWaveStroke() {
  waveStroke = getWaveStroke();
}

function applyTheme(pref) {
  if (pref === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else if (pref === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  updateWaveStroke();
}

// Persisted theme preference
const savedThemePref = localStorage.getItem("themePreference") || "system";
applyTheme(savedThemePref);
if (themeRadios.length) {
  themeRadios.forEach((r) => {
    r.checked = r.value === savedThemePref;
    r.addEventListener("change", () => {
      if (r.checked) {
        localStorage.setItem("themePreference", r.value);
        applyTheme(r.value);
      }
    });
  });
}

// React to OS theme changes when in "system" mode
const darkMq = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
const handleOsThemeChange = () => {
  const pref = localStorage.getItem("themePreference") || "system";
  if (pref === "system") updateWaveStroke();
};
if (darkMq && darkMq.addEventListener) {
  darkMq.addEventListener("change", handleOsThemeChange);
} else if (darkMq && darkMq.addListener) {
  // deprecated, but included for Safari compatibility
  darkMq.addListener(handleOsThemeChange);
}

// Helpers
function fmt1(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(1) : x;
}

// Microphone enumeration and selection
async function enumerateMicrophones() {
  try {
    // Request permission first to get device labels
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach(track => track.stop());
    
    const devices = await navigator.mediaDevices.enumerateDevices();
    availableMicrophones = devices.filter(device => device.kind === 'audioinput');
    
    populateMicrophoneSelect();
    console.log(`Found ${availableMicrophones.length} microphone(s)`);
  } catch (error) {
    console.error('Error enumerating microphones:', error);
    statusText.textContent = "Error accessing microphones. Please grant permission.";
  }
}

function populateMicrophoneSelect() {
  if (!microphoneSelect) return;
  
  // Clear existing options except the default
  microphoneSelect.innerHTML = '<option value="">Default Microphone</option>';
  
  availableMicrophones.forEach((device, index) => {
    const option = document.createElement('option');
    option.value = device.deviceId;
    option.textContent = device.label || `Microphone ${index + 1}`;
    microphoneSelect.appendChild(option);
  });
  
  // Restore previously selected microphone
  const savedMicId = localStorage.getItem('selectedMicrophone');
  if (savedMicId && availableMicrophones.some(mic => mic.deviceId === savedMicId)) {
    microphoneSelect.value = savedMicId;
    selectedMicrophoneId = savedMicId;
  }
}

function handleMicrophoneChange() {
  selectedMicrophoneId = microphoneSelect.value || null;
  localStorage.setItem('selectedMicrophone', selectedMicrophoneId || '');
  
  const selectedDevice = availableMicrophones.find(mic => mic.deviceId === selectedMicrophoneId);
  const deviceName = selectedDevice ? selectedDevice.label : 'Default Microphone';
  
  console.log(`Selected microphone: ${deviceName}`);
  statusText.textContent = `Microphone changed to: ${deviceName}`;
  
  // If currently recording, we need to restart with the new microphone
  if (isRecording) {
    statusText.textContent = "Switching microphone... Please wait.";
    stopRecording().then(() => {
      setTimeout(() => {
        toggleRecording();
      }, 1000);
    });
  }
}

// Default WebSocket URL computation
const host = window.location.hostname || "localhost";
const port = window.location.port;
const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const defaultWebSocketUrl = `${protocol}://${host}${port ? ":" + port : ""}/asr`;

// Populate default caption and input
if (websocketDefaultSpan) websocketDefaultSpan.textContent = defaultWebSocketUrl;
websocketInput.value = defaultWebSocketUrl;
websocketUrl = defaultWebSocketUrl;

// Optional chunk selector (guard for presence)
if (chunkSelector) {
  chunkSelector.addEventListener("change", () => {
    chunkDuration = parseInt(chunkSelector.value);
  });
}

// WebSocket input change handling
websocketInput.addEventListener("change", () => {
  const urlValue = websocketInput.value.trim();
  if (!urlValue.startsWith("ws://") && !urlValue.startsWith("wss://")) {
    statusText.textContent = "Invalid WebSocket URL (must start with ws:// or wss://)";
    return;
  }
  websocketUrl = urlValue;
  statusText.textContent = "WebSocket URL updated. Ready to connect.";
});

function setupWebSocket() {
  return new Promise((resolve, reject) => {
    try {
      websocket = new WebSocket(websocketUrl);
    } catch (error) {
      statusText.textContent = "Invalid WebSocket URL. Please check and try again.";
      reject(error);
      return;
    }

    websocket.onopen = () => {
      statusText.textContent = "Connected to server.";
      resolve();
    };

    websocket.onclose = () => {
      if (userClosing) {
        if (waitingForStop) {
          statusText.textContent = "Processing finalized or connection closed.";
          if (lastReceivedData) {
            renderLinesWithBuffer(
              lastReceivedData.lines || [],
              lastReceivedData.buffer_diarization || "",
              lastReceivedData.buffer_transcription || "",
              0,
              0,
              true
            );
          }
        }
      } else {
        statusText.textContent = "Disconnected from the WebSocket server. (Check logs if model is loading.)";
        if (isRecording) {
          stopRecording();
        }
      }
      isRecording = false;
      waitingForStop = false;
      userClosing = false;
      lastReceivedData = null;
      websocket = null;
      updateUI();
    };

    websocket.onerror = () => {
      statusText.textContent = "Error connecting to WebSocket.";
      reject(new Error("Error connecting to WebSocket"));
    };

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "ready_to_stop") {
        console.log("Ready to stop received, finalizing display and closing WebSocket.");
        waitingForStop = false;

        if (lastReceivedData) {
          renderLinesWithBuffer(
            lastReceivedData.lines || [],
            lastReceivedData.buffer_diarization || "",
            lastReceivedData.buffer_transcription || "",
            0,
            0,
            true
          );
        }
        statusText.textContent = "Finished processing audio! Ready to record again.";
        recordButton.disabled = false;

        if (websocket) {
          websocket.close();
        }
        return;
      }

      lastReceivedData = data;

      const {
        lines = [],
        buffer_transcription = "",
        buffer_diarization = "",
        remaining_time_transcription = 0,
        remaining_time_diarization = 0,
        status = "active_transcription",
      } = data;

      renderLinesWithBuffer(
        lines,
        buffer_diarization,
        buffer_transcription,
        remaining_time_diarization,
        remaining_time_transcription,
        false,
        status
      );
    };
  });
}

function renderLinesWithBuffer(
  lines,
  buffer_diarization,
  buffer_transcription,
  remaining_time_diarization,
  remaining_time_transcription,
  isFinalizing = false,
  current_status = "active_transcription"
) {
  if (current_status === "no_audio_detected") {
    linesTranscriptDiv.innerHTML =
      "<p style='text-align: center; color: var(--muted); margin-top: 20px;'><em>No audio detected...</em></p>";
    return;
  }

  const showLoading = !isFinalizing && (lines || []).some((it) => it.speaker == 0);
  const showTransLag = !isFinalizing && remaining_time_transcription > 0;
  const showDiaLag = !isFinalizing && !!buffer_diarization && remaining_time_diarization > 0;
  const signature = JSON.stringify({
    lines: (lines || []).map((it) => ({ speaker: it.speaker, text: it.text, beg: it.beg, end: it.end })),
    buffer_transcription: buffer_transcription || "",
    buffer_diarization: buffer_diarization || "",
    status: current_status,
    showLoading,
    showTransLag,
    showDiaLag,
    isFinalizing: !!isFinalizing,
  });
  if (lastSignature === signature) {
    const t = document.querySelector(".lag-transcription-value");
    if (t) t.textContent = fmt1(remaining_time_transcription);
    const d = document.querySelector(".lag-diarization-value");
    if (d) d.textContent = fmt1(remaining_time_diarization);
    const ld = document.querySelector(".loading-diarization-value");
    if (ld) ld.textContent = fmt1(remaining_time_diarization);
    return;
  }
  lastSignature = signature;

  const linesHtml = (lines || [])
    .map((item, idx) => {
      let timeInfo = "";
      if (item.beg !== undefined && item.end !== undefined) {
        timeInfo = ` ${item.beg} - ${item.end}`;
      }

      let speakerLabel = "";
      if (item.speaker === -2) {
        speakerLabel = `<span class="silence">Silence<span id='timeInfo'>${timeInfo}</span></span>`;
      } else if (item.speaker == 0 && !isFinalizing) {
        speakerLabel = `<span class='loading'><span class="spinner"></span><span id='timeInfo'><span class="loading-diarization-value">${fmt1(
          remaining_time_diarization
        )}</span> second(s) of audio are undergoing diarization</span></span>`;
      } else if (item.speaker !== 0) {
        speakerLabel = `<span id="speaker">Speaker ${item.speaker}<span id='timeInfo'>${timeInfo}</span></span>`;
      }

      let currentLineText = item.text || "";

      if (idx === lines.length - 1) {
        if (!isFinalizing && item.speaker !== -2) {
          if (remaining_time_transcription > 0) {
            speakerLabel += `<span class="label_transcription"><span class="spinner"></span>Transcription lag <span id='timeInfo'><span class="lag-transcription-value">${fmt1(
              remaining_time_transcription
            )}</span>s</span></span>`;
          }
          if (buffer_diarization && remaining_time_diarization > 0) {
            speakerLabel += `<span class="label_diarization"><span class="spinner"></span>Diarization lag<span id='timeInfo'><span class="lag-diarization-value">${fmt1(
              remaining_time_diarization
            )}</span>s</span></span>`;
          }
        }

        if (buffer_diarization) {
          if (isFinalizing) {
            currentLineText +=
              (currentLineText.length > 0 && buffer_diarization.trim().length > 0 ? " " : "") + buffer_diarization.trim();
          } else {
            currentLineText += `<span class="buffer_diarization">${buffer_diarization}</span>`;
          }
        }
        if (buffer_transcription) {
          if (isFinalizing) {
            currentLineText +=
              (currentLineText.length > 0 && buffer_transcription.trim().length > 0 ? " " : "") +
              buffer_transcription.trim();
          } else {
            currentLineText += `<span class="buffer_transcription">${buffer_transcription}</span>`;
          }
        }
      }

      return currentLineText.trim().length > 0 || speakerLabel.length > 0
        ? `<p>${speakerLabel}<br/><div class='textcontent'>${currentLineText}</div></p>`
        : `<p>${speakerLabel}<br/></p>`;
    })
    .join("");

  // Use requestAnimationFrame for smoother DOM updates
  requestAnimationFrame(() => {
    linesTranscriptDiv.innerHTML = linesHtml;
    
    // Add live indicator when actively transcribing
    if (!isFinalizing && isRecording) {
      linesTranscriptDiv.classList.add('live');
    } else {
      linesTranscriptDiv.classList.remove('live');
    }
    
    // Auto-scroll behavior
    const transcriptContainer = linesTranscriptDiv;
    if (transcriptContainer) {
      // Check if user is at the bottom (within 50px tolerance)
      const isAtBottom = transcriptContainer.scrollHeight - transcriptContainer.scrollTop - transcriptContainer.clientHeight < 50;
      
      // Show/hide scroll to bottom button
      if (scrollToBottomBtn) {
        const shouldShow = !isAtBottom && transcriptContainer.scrollHeight > transcriptContainer.clientHeight && !isFinalizing;
        scrollToBottomBtn.style.display = shouldShow ? 'flex' : 'none';
      }
      
      // Always auto-scroll during live recording, or if user is at bottom
      if (!isFinalizing && isRecording) {
        // Force scroll to bottom during live transcription
        transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
      } else if (isAtBottom || transcriptContainer.scrollHeight <= transcriptContainer.clientHeight) {
        transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
      }
    }
  });
}

function updateTimer() {
  if (!startTime) return;

  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const minutes = Math.floor(elapsed / 60).toString().padStart(2, "0");
  const seconds = (elapsed % 60).toString().padStart(2, "0");
  const timeString = `${minutes}:${seconds}`;
  
  if (timerElement) timerElement.textContent = timeString;
  if (headerTimer) headerTimer.textContent = timeString;
}

function drawWaveform() {
  if (!analyser) return;

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  const canvasWidth = waveCanvas.width / (window.devicePixelRatio || 1);
  const canvasHeight = waveCanvas.height / (window.devicePixelRatio || 1);
  const centerX = canvasWidth / 2;
  const centerY = canvasHeight / 2;
  const radius = Math.min(centerX, centerY) - 10;

  waveCtx.clearRect(0, 0, canvasWidth, canvasHeight);
  
  // Draw circular waveform
  waveCtx.lineWidth = 2;
  waveCtx.strokeStyle = '#00d4ff';
  waveCtx.shadowColor = '#00d4ff';
  waveCtx.shadowBlur = 10;
  waveCtx.beginPath();

  const angleStep = (Math.PI * 2) / bufferLength;
  
  for (let i = 0; i < bufferLength; i++) {
    const angle = i * angleStep;
    const amplitude = (dataArray[i] / 255) * 30;
    const x = centerX + Math.cos(angle) * (radius + amplitude);
    const y = centerY + Math.sin(angle) * (radius + amplitude);
    
    if (i === 0) {
      waveCtx.moveTo(x, y);
    } else {
      waveCtx.lineTo(x, y);
    }
  }
  
  waveCtx.closePath();
  waveCtx.stroke();

  animationFrame = requestAnimationFrame(drawWaveform);
}

async function startRecording() {
  try {
    // Request wake lock to keep screen awake during recording
    try {
      if ('wakeLock' in navigator) {
        wakeLock = await navigator.wakeLock.request("screen");
        console.log("Screen wake lock acquired - screen will stay awake");
        
        // Listen for wake lock release (e.g., when tab becomes hidden)
        wakeLock.addEventListener('release', () => {
          console.log('Screen wake lock was released');
        });
      } else {
        console.log("Wake Lock API not supported in this browser");
        // Fallback: use a hidden video element to prevent sleep (older method)
        requestWakeLockFallback();
      }
    } catch (err) {
      console.log("Error acquiring wake lock:", err);
      // Try fallback method
      requestWakeLockFallback();
    }

    // Use selected microphone if available
    const audioConstraints = { audio: true };
    if (selectedMicrophoneId) {
      audioConstraints.audio = { deviceId: { exact: selectedMicrophoneId } };
      console.log(`Using selected microphone: ${selectedMicrophoneId}`);
    }
    
    const stream = await navigator.mediaDevices.getUserMedia(audioConstraints);

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    microphone = audioContext.createMediaStreamSource(stream);
    microphone.connect(analyser);

    recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    recorder.ondataavailable = (e) => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(e.data);
      }
    };
    recorder.start(chunkDuration);

    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);
    drawWaveform();

    isRecording = true;
    updateUI();
  } catch (err) {
    if (window.location.hostname === "0.0.0.0") {
      statusText.textContent =
        "Error accessing microphone. Browsers may block microphone access on 0.0.0.0. Try using localhost:8000 instead.";
    } else {
      statusText.textContent = "Error accessing microphone. Please allow microphone access.";
    }
    console.error(err);
  }
}

async function stopRecording() {
  if (wakeLock) {
    try {
      await wakeLock.release();
    } catch (e) {
      // ignore
    }
    wakeLock = null;
  }

  userClosing = true;
  waitingForStop = true;

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    const emptyBlob = new Blob([], { type: "audio/webm" });
    websocket.send(emptyBlob);
    statusText.textContent = "Recording stopped. Processing final audio...";
  }

  if (recorder) {
    recorder.stop();
    recorder = null;
  }

  if (microphone) {
    microphone.disconnect();
    microphone = null;
  }

  if (analyser) {
    analyser = null;
  }

  if (audioContext && audioContext.state !== "closed") {
    try {
      await audioContext.close();
    } catch (e) {
      console.warn("Could not close audio context:", e);
    }
    audioContext = null;
  }

  if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }

  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerElement.textContent = "00:00";
  startTime = null;

  isRecording = false;
  updateUI();
}

async function toggleRecording() {
  if (!isRecording) {
    if (waitingForStop) {
      console.log("Waiting for stop, early return");
      return;
    }
    console.log("Connecting to WebSocket");
    try {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        await startRecording();
      } else {
        await setupWebSocket();
        await startRecording();
      }
    } catch (err) {
      statusText.textContent = "Could not connect to WebSocket or access mic. Aborted.";
      console.error(err);
    }
  } else {
    console.log("Stopping recording");
    stopRecording();
  }
}

function updateUI() {
  recordButton.classList.toggle("recording", isRecording);
  recordButton.disabled = waitingForStop;
  recordButton.setAttribute("aria-pressed", String(isRecording));

  // Toggle layout transformation
  if (topNav) topNav.classList.toggle("recording", isRecording);
  if (centralHub) centralHub.classList.toggle("recording", isRecording);
  if (transcriptFlow) transcriptFlow.classList.toggle("recording", isRecording);
  
  // Show/hide header control
  if (headerControl) {
    headerControl.style.display = isRecording ? 'flex' : 'none';
  }

  if (waitingForStop) {
    if (statusText.textContent !== "Recording stopped. Processing final audio...") {
      statusText.textContent = "Please wait for processing to complete...";
    }
  } else if (isRecording) {
    statusText.textContent = "Neural transcription active";
  } else {
    if (
      statusText.textContent !== "Finished processing audio! Ready to record again." &&
      statusText.textContent !== "Processing finalized or connection closed."
    ) {
      statusText.textContent = "Neural transcription ready";
    }
  }
  if (!waitingForStop) {
    recordButton.disabled = false;
  }

  if (actionText) {
    actionText.textContent = isRecording ? "Stop Transcribe" : "Start Transcribe";
  }
}

recordButton.addEventListener("click", toggleRecording);
recordButton.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    toggleRecording();
  }
});

// Scroll to bottom button functionality
if (scrollToBottomBtn) {
  scrollToBottomBtn.addEventListener("click", () => {
    if (linesTranscriptDiv) {
      linesTranscriptDiv.scrollTo({
        top: linesTranscriptDiv.scrollHeight,
        behavior: "smooth"
      });
    }
  });
}

// Header record button functionality
if (headerRecordButton) {
  headerRecordButton.addEventListener("click", toggleRecording);
}

// Microphone selection functionality
if (microphoneSelect) {
  microphoneSelect.addEventListener("change", handleMicrophoneChange);
}

if (refreshMicsButton) {
  refreshMicsButton.addEventListener("click", async () => {
    refreshMicsButton.disabled = true;
    refreshMicsButton.innerHTML = '<span class="refresh-icon">⏳</span><span>Refreshing...</span>';
    
    try {
      await enumerateMicrophones();
      statusText.textContent = "Microphone list refreshed";
    } catch (error) {
      statusText.textContent = "Error refreshing microphones";
      console.error("Error refreshing microphones:", error);
    } finally {
      refreshMicsButton.disabled = false;
      refreshMicsButton.innerHTML = '<span class="refresh-icon">🔄</span><span>Refresh</span>';
    }
  });
}

// Initialize microphones on page load
document.addEventListener('DOMContentLoaded', async () => {
  try {
    await enumerateMicrophones();
  } catch (error) {
    console.log("Could not enumerate microphones on load:", error);
  }
});

// Re-enumerate devices when they change
navigator.mediaDevices.addEventListener('devicechange', async () => {
  console.log('Device change detected, re-enumerating microphones');
  try {
    await enumerateMicrophones();
  } catch (error) {
    console.log("Error re-enumerating microphones:", error);
  }
});
