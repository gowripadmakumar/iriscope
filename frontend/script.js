/*
 * IRISCOPE frontend -- Module 7.
 *
 * Three independent input widgets share one factory (createCaptureWidget):
 * the single Iris Analysis widget (prefix "") and the two Iris Match
 * widgets ("match-a-", "match-b-"). Each widget handles its own
 * upload/camera toggle and hands back one File or Blob -- neither flow
 * talks to the backend directly. There is exactly one place that calls
 * /api/analyze and one place that calls /api/match, so every input path
 * always exercises the same server-side pipeline (carried over from
 * Module 6's design).
 */

const el = (id) => document.getElementById(id);
const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

// ---------------------------------------------------------------------------
// Theme toggle
// ---------------------------------------------------------------------------

function initTheme() {
  const stored = localStorage.getItem("iriscope-theme");
  const preferred = stored || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  setTheme(preferred);

  el("theme-toggle").addEventListener("click", () => {
    const next = document.body.dataset.theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("iriscope-theme", next);
  });
}

function setTheme(theme) {
  document.body.dataset.theme = theme;
  el("theme-toggle-label").textContent = theme === "dark" ? "Dark" : "Light";
  el("theme-toggle").setAttribute(
    "aria-label",
    theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
  );
}

// ---------------------------------------------------------------------------
// Top-level nav (Iris Analysis / Iris Match)
// ---------------------------------------------------------------------------

function initNav() {
  const analyzeBtn = el("nav-analyze-btn");
  const matchBtn = el("nav-match-btn");
  const analyzeSection = el("analyze-section");
  const matchSection = el("match-section");

  function showAnalyze() {
    analyzeBtn.classList.add("active");
    matchBtn.classList.remove("active");
    analyzeBtn.setAttribute("aria-selected", "true");
    matchBtn.setAttribute("aria-selected", "false");
    analyzeSection.hidden = false;
    matchSection.hidden = true;
  }

  function showMatch() {
    matchBtn.classList.add("active");
    analyzeBtn.classList.remove("active");
    matchBtn.setAttribute("aria-selected", "true");
    analyzeBtn.setAttribute("aria-selected", "false");
    matchSection.hidden = false;
    analyzeSection.hidden = true;
  }

  analyzeBtn.addEventListener("click", showAnalyze);
  matchBtn.addEventListener("click", showMatch);
}

// ---------------------------------------------------------------------------
// Capture widget factory -- one upload/camera input, reused three times
// ---------------------------------------------------------------------------

function createCaptureWidget(prefix, onChange) {
  const ids = (name) => (prefix ? `${prefix}-${name}` : name);

  const uploadModeBtn = el(ids("mode-upload-btn"));
  const cameraModeBtn = el(ids("mode-camera-btn"));
  const uploadPanel = el(ids("upload-mode"));
  const cameraPanel = el(ids("camera-mode"));
  const fileInput = el(ids("file-input"));
  const uploadPreview = el(ids("upload-preview"));
  const video = el(ids("camera-video"));
  const canvas = el(ids("camera-canvas"));
  const cameraPreview = el(ids("camera-preview"));
  const startCameraBtn = el(ids("start-camera-btn"));
  const captureBtn = el(ids("capture-btn"));
  const retakeBtn = el(ids("retake-btn"));
  const errorBox = el(ids("error"));
  const guidance = el(ids("camera-guidance"));
  const cameraGuide = el(ids("camera-guide"));
  const cameraStatus = el(ids("camera-status"));

  const state = { file: null, filename: null, stream: null };

  function showError(message) {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function hideError() {
    if (errorBox) errorBox.hidden = true;
  }

  function setFile(fileOrBlob, filename) {
    state.file = fileOrBlob;
    state.filename = filename;
    onChange();
  }

  function switchMode(mode) {
    const isUpload = mode === "upload";
    uploadModeBtn.classList.toggle("active", isUpload);
    cameraModeBtn.classList.toggle("active", !isUpload);
    uploadPanel.hidden = !isUpload;
    cameraPanel.hidden = isUpload;
    hideError();
    if (isUpload) {
      stopCamera();
      if (guidance) guidance.hidden = true;
      if (cameraGuide) cameraGuide.hidden = true;
      if (cameraStatus) cameraStatus.hidden = true;
    }
  }

  uploadModeBtn.addEventListener("click", () => switchMode("upload"));
  cameraModeBtn.addEventListener("click", () => switchMode("camera"));

  fileInput.addEventListener("change", () => {
    hideError();
    const file = fileInput.files[0];
    if (!file) return;

    if (!ALLOWED_TYPES.includes(file.type)) {
      showError("Please choose a JPG or PNG image.");
      setFile(null, null);
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      uploadPreview.src = event.target.result;
      uploadPreview.hidden = false;
    };
    reader.readAsDataURL(file);

    setFile(file, file.name || "upload.jpg");
  });

  async function startCamera() {
    hideError();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("This browser doesn't support camera access. Try uploading a photo instead.");
      return;
    }

    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
    } catch (err) {
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        showError("Camera access was denied. Allow camera access, or upload a photo instead.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        showError("No camera was found on this device. Try uploading a photo instead.");
      } else {
        showError("Couldn't access the camera. Try uploading a photo instead.");
      }
      return;
    }

    video.srcObject = state.stream;
    video.hidden = false;
    cameraPreview.hidden = true;

    startCameraBtn.hidden = true;
    captureBtn.hidden = false;
    retakeBtn.hidden = true;
    if (guidance) guidance.hidden = false;
    if (cameraGuide) cameraGuide.hidden = false;
    if (cameraStatus) {
      cameraStatus.textContent = "Camera live — align eye";
      cameraStatus.hidden = false;
    }
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
  }

  function captureFrame() {
    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!width || !height) {
      showError("The camera hasn't produced a frame yet. Try again in a moment.");
      return;
    }

    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(video, 0, 0, width, height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          showError("Couldn't capture that frame. Try again.");
          return;
        }

        cameraPreview.src = URL.createObjectURL(blob);
        cameraPreview.hidden = false;
        video.hidden = true;
        stopCamera();

        captureBtn.hidden = true;
        retakeBtn.hidden = false;
        if (guidance) guidance.hidden = true;
        if (cameraGuide) cameraGuide.hidden = true;
        if (cameraStatus) {
          cameraStatus.textContent = "Frame captured";
        }

        setFile(blob, "capture.jpg");
      },
      "image/jpeg",
      0.92
    );
  }

  function retakePhoto() {
    setFile(null, null);
    cameraPreview.hidden = true;
    retakeBtn.hidden = true;
    startCamera();
  }

  startCameraBtn.addEventListener("click", startCamera);
  captureBtn.addEventListener("click", captureFrame);
  retakeBtn.addEventListener("click", retakePhoto);

  return {
    hasFile: () => Boolean(state.file),
    getFile: () => state.file,
    getFilename: () => state.filename,
  };
}

// ---------------------------------------------------------------------------
// Iris Analysis flow
// ---------------------------------------------------------------------------

function initAnalyze() {
  const analyzeUploadBtn = el("analyze-upload-btn");
  const analyzeCameraBtn = el("analyze-camera-btn");

  const widget = createCaptureWidget("", () => {
    const ready = widget.hasFile();
    analyzeUploadBtn.disabled = !ready;
    analyzeCameraBtn.disabled = !ready;
    analyzeCameraBtn.hidden = !ready;
  });

  analyzeUploadBtn.addEventListener("click", () => runAnalyze(widget));
  analyzeCameraBtn.addEventListener("click", () => runAnalyze(widget));
}

// The real /api/analyze call is one request that returns one finished
// result -- the backend does not stream per-stage progress. This cycles
// through the pipeline's actual, documented stage names while that one
// request is in flight, as an honest "here's the procedure" indicator,
// not a claim that each line appears exactly when that backend step runs.
const ANALYSIS_STAGE_MESSAGES = [
  "01 Locating eye",
  "02 Estimating pupil",
  "03 Mapping iris",
  "04 Masking specimen",
  "05 Normalizing texture",
  "06 Detecting structures",
  "07 Finalizing count",
];

function startStageCycler(stageEl) {
  let i = 0;
  stageEl.textContent = ANALYSIS_STAGE_MESSAGES[0];
  const timer = setInterval(() => {
    i = (i + 1) % ANALYSIS_STAGE_MESSAGES.length;
    stageEl.textContent = ANALYSIS_STAGE_MESSAGES[i];
  }, 550);
  return () => clearInterval(timer);
}

async function runAnalyze(widget) {
  if (!widget.hasFile()) return;

  const inputError = el("input-error");
  const statusBlock = el("status-block");
  const statusStage = el("status-stage");
  const resultSection = el("result-section");
  const resultContent = el("result-content");
  const analyzeUploadBtn = el("analyze-upload-btn");
  const analyzeCameraBtn = el("analyze-camera-btn");

  inputError.hidden = true;
  resultSection.hidden = true;
  statusBlock.hidden = false;
  const stopCycler = startStageCycler(statusStage);
  analyzeUploadBtn.disabled = true;
  analyzeCameraBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", widget.getFile(), widget.getFilename());

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData });
    const data = await response.json();
    renderAnalyzeResult(data, resultSection, resultContent);
  } catch (err) {
    inputError.textContent = "Something went wrong. Please try again.";
    inputError.hidden = false;
  } finally {
    stopCycler();
    statusBlock.hidden = true;
    const ready = widget.hasFile();
    analyzeUploadBtn.disabled = !ready;
    analyzeCameraBtn.disabled = !ready;
  }
}

function renderAnalyzeResult(data, section, content) {
  content.innerHTML = "";
  section.hidden = false;
  replayRevealAnimation(section);

  if (!data.eyes || data.eyes.length === 0) {
    content.appendChild(buildFailureBlock(data.error, data.stages));
    return;
  }

  data.eyes.forEach((eye) => content.appendChild(buildEyeBlock(eye)));
}

function buildFailureAside() {
  const p = document.createElement("p");
  p.className = "error-aside";
  p.textContent = "The iris has declined to participate.";
  return p;
}

function buildFailureBlock(message, stages) {
  const div = document.createElement("div");
  div.className = "eye-result failed";

  const p = document.createElement("p");
  p.className = "eye-result-summary";
  p.textContent = message || "Analysis did not complete.";
  div.appendChild(p);
  div.appendChild(buildFailureAside());

  const viewer = buildStageViewer(stages);
  if (viewer) div.appendChild(viewer);

  return div;
}

let congratsShown = false;

function buildEyeBlock(eye) {
  const div = document.createElement("div");
  div.className = "eye-result" + (eye.success ? "" : " failed");

  const heading = document.createElement("div");
  heading.className = "eye-result-heading";
  const eyebrow = document.createElement("span");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = "Eye";
  const h3 = document.createElement("h3");
  h3.textContent = capitalize(eye.position);
  heading.appendChild(eyebrow);
  heading.appendChild(h3);
  div.appendChild(heading);

  if (eye.success) {
    div.appendChild(buildResultHero(eye.metrics));
    div.appendChild(buildResultMetrics(eye.metrics));
  } else {
    const p = document.createElement("p");
    p.className = "eye-result-summary";
    p.textContent = eye.error || "This eye could not be analyzed.";
    div.appendChild(p);
    div.appendChild(buildFailureAside());
  }

  const viewer = buildStageViewer(eye.stages);
  if (viewer) div.appendChild(viewer);

  return div;
}

// The large instrument-readout number. All values shown are real values
// already returned by the backend (eye.metrics) -- the count-up is purely
// a reveal animation ending on that real number, never an invented one.
function buildResultHero(metrics) {
  const wrap = document.createElement("div");
  wrap.className = "result-hero";
  wrap.innerHTML = `
    <span class="eyebrow">Iris structure count</span>
    <div class="result-hero-number">0</div>
    <p class="result-hero-caption">Detectable radial structures</p>
  `;
  animateCountUp(wrap.querySelector(".result-hero-number"), metrics.structure_count);

  if (!congratsShown) {
    congratsShown = true;
    const aside = document.createElement("p");
    aside.className = "result-hero-aside";
    aside.textContent = "Congratulations. You now know something completely unnecessary.";
    wrap.appendChild(aside);
  }

  return wrap;
}

function animateCountUp(numberEl, target, duration = 700) {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !target) {
    numberEl.textContent = target;
    return;
  }
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    numberEl.textContent = Math.round(progress * target);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function buildResultMetrics(metrics) {
  const wrap = document.createElement("div");
  wrap.className = "result-metrics";
  const quality = Math.round(metrics.detection_quality * 100);
  const usable = Math.round(metrics.usable_area_ratio * 100);
  wrap.appendChild(buildMetricRow("Analysis quality", `${quality}%`));
  wrap.appendChild(buildMetricRow("Usable iris area", `${usable}%`));
  wrap.appendChild(buildMetricRow(
    "Structure density",
    metrics.structure_density != null ? `${metrics.structure_density} / 100\u00B0` : "\u2014"
  ));
  return wrap;
}

function buildMetricRow(label, value) {
  const row = document.createElement("div");
  row.className = "result-metric-row";
  const labelSpan = document.createElement("span");
  labelSpan.className = "result-metric-label";
  labelSpan.textContent = label;
  const valueSpan = document.createElement("span");
  valueSpan.className = "result-metric-value";
  valueSpan.textContent = value;
  row.appendChild(labelSpan);
  row.appendChild(valueSpan);
  return row;
}

// ---------------------------------------------------------------------------
// Interactive stage viewer -- tabs through the real per-stage images the
// backend now returns individually (Module 10). Every image shown here is
// actual pipeline output; stages that don't exist (pipeline stopped early)
// simply don't get a tab.
// ---------------------------------------------------------------------------

function stageShortLabel(label) {
  return label.replace(/^\d+\.\s*/, "");
}

function buildStageViewer(stages) {
  if (!stages || stages.length === 0) return null;

  const container = document.createElement("div");
  container.className = "stage-viewer";

  const tabs = document.createElement("div");
  tabs.className = "stage-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Analysis stages");

  const display = document.createElement("div");
  display.className = "stage-display";

  const img = document.createElement("img");
  img.alt = "";

  const zoomBtn = document.createElement("button");
  zoomBtn.type = "button";
  zoomBtn.className = "stage-zoom-btn";
  zoomBtn.textContent = "Enlarge";

  const caption = document.createElement("p");
  caption.className = "stage-caption";

  display.appendChild(img);
  display.appendChild(zoomBtn);
  container.appendChild(tabs);
  container.appendChild(display);
  container.appendChild(caption);

  const tabButtons = stages.map((stage, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "stage-tab";
    btn.setAttribute("role", "tab");
    btn.textContent = stageShortLabel(stage.label);
    tabs.appendChild(btn);
    return btn;
  });

  function captionFor(index) {
    return `Fig. ${String(index + 1).padStart(2, "0")} \u2014 ${stageShortLabel(stages[index].label)}`;
  }

  function setActiveTab(index) {
    tabButtons.forEach((btn, i) => {
      const isActive = i === index;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-selected", isActive ? "true" : "false");
    });
  }

  // Paints a stage immediately, no fade -- used for the viewer's first render.
  function renderStage(index) {
    setActiveTab(index);
    const text = captionFor(index);
    img.src = stages[index].image;
    img.alt = text;
    caption.textContent = text;
  }

  // Switches to a stage the user picked -- same result, with a brief
  // cross-fade so the change reads as deliberate rather than a jump cut.
  function showStage(index) {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) {
      renderStage(index);
      return;
    }
    setActiveTab(index);
    img.classList.add("stage-image-swap");
    setTimeout(() => {
      const text = captionFor(index);
      img.src = stages[index].image;
      img.alt = text;
      caption.textContent = text;
      requestAnimationFrame(() => img.classList.remove("stage-image-swap"));
    }, 120);
  }

  tabButtons.forEach((btn, i) => {
    btn.addEventListener("click", () => showStage(i));
  });

  zoomBtn.addEventListener("click", () => openStageModal(img.src, caption.textContent, zoomBtn));
  img.addEventListener("click", () => openStageModal(img.src, caption.textContent, zoomBtn));

  const defaultIndex = stages.findIndex((s) => s.key === "final");
  renderStage(defaultIndex >= 0 ? defaultIndex : stages.length - 1);

  return container;
}

// ---------------------------------------------------------------------------
// Stage zoom modal (lightbox), shared by every stage viewer on the page.
// ---------------------------------------------------------------------------

let stageModalTrigger = null;

function openStageModal(src, captionText, triggerEl) {
  const modal = el("stage-modal");
  const modalImg = el("stage-modal-image");
  const modalCaption = el("stage-modal-caption");
  if (!modal || !src) return;

  modalImg.src = src;
  modalImg.alt = captionText || "";
  modalCaption.textContent = captionText || "";
  modal.hidden = false;
  stageModalTrigger = triggerEl || null;
  el("stage-modal-close").focus();
  document.addEventListener("keydown", onStageModalKeydown);
}

function closeStageModal() {
  const modal = el("stage-modal");
  modal.hidden = true;
  document.removeEventListener("keydown", onStageModalKeydown);
  if (stageModalTrigger) stageModalTrigger.focus();
}

function onStageModalKeydown(event) {
  if (event.key === "Escape") closeStageModal();
}

function initStageModal() {
  const closeBtn = el("stage-modal-close");
  if (!closeBtn) return;
  closeBtn.addEventListener("click", closeStageModal);
  document.querySelectorAll("[data-modal-close]").forEach((node) => {
    node.addEventListener("click", closeStageModal);
  });
}

function capitalize(text) {
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}

// One restrained fade-in, replayed by forcing a reflow so it re-triggers
// on every new result (not just the first). No motion beyond this.
function replayRevealAnimation(section) {
  section.classList.remove("reveal");
  // eslint-disable-next-line no-unused-expressions
  section.offsetWidth;
  section.classList.add("reveal");
}

// ---------------------------------------------------------------------------
// Iris Match flow
// ---------------------------------------------------------------------------

function initMatch() {
  const matchBtn = el("match-btn");

  function refreshMatchBtn() {
    matchBtn.disabled = !(widgetA.hasFile() && widgetB.hasFile());
  }

  const widgetA = createCaptureWidget("match-a", refreshMatchBtn);
  const widgetB = createCaptureWidget("match-b", refreshMatchBtn);

  matchBtn.addEventListener("click", () => runMatch(widgetA, widgetB));
}

async function runMatch(widgetA, widgetB) {
  if (!widgetA.hasFile() || !widgetB.hasFile()) return;

  const matchStatus = el("match-status");
  const matchResultSection = el("match-result-section");
  const matchResultContent = el("match-result-content");
  const matchBtn = el("match-btn");

  matchResultSection.hidden = true;
  matchStatus.hidden = false;
  matchBtn.disabled = true;
  el("match-a-error").hidden = true;
  el("match-b-error").hidden = true;

  const formData = new FormData();
  formData.append("image_a", widgetA.getFile(), widgetA.getFilename());
  formData.append("image_b", widgetB.getFile(), widgetB.getFilename());

  try {
    const response = await fetch("/api/match", { method: "POST", body: formData });
    const data = await response.json();
    renderMatchResult(data, matchResultSection, matchResultContent);
  } catch (err) {
    matchResultSection.hidden = false;
    matchResultContent.innerHTML = "";
    const p = document.createElement("p");
    p.className = "eye-result-summary";
    p.textContent = "Something went wrong. Please try again.";
    matchResultContent.appendChild(p);
  } finally {
    matchStatus.hidden = true;
    matchBtn.disabled = false;
  }
}

function renderMatchResult(data, section, content) {
  content.innerHTML = "";
  section.hidden = false;
  replayRevealAnimation(section);

  if (!data.success) {
    const p = document.createElement("p");
    p.className = "eye-result-summary";
    p.textContent = data.error || "This comparison could not be completed.";
    content.appendChild(p);
    return;
  }

  const verdictBlock = document.createElement("div");
  verdictBlock.className = "match-verdict-block";
  verdictBlock.innerHTML = `
    <span class="eyebrow">Iris Compatibility</span>
    <div class="match-compatibility-number">${data.scores.compatibility}%</div>
    <p class="match-verdict-text">${data.verdict}</p>
  `;
  content.appendChild(verdictBlock);

  const subscores = document.createElement("div");
  subscores.className = "match-subscores";
  subscores.appendChild(buildSubscoreRow("Texture Similarity", data.scores.texture_similarity));
  subscores.appendChild(buildSubscoreRow("Radial Harmony", data.scores.radial_harmony));
  subscores.appendChild(buildSubscoreRow("Density Harmony", data.scores.density_harmony));
  content.appendChild(subscores);

  const eyesGrid = document.createElement("div");
  eyesGrid.className = "match-eyes";
  eyesGrid.appendChild(buildMatchEyeCard("Specimen A", data.eye_a));
  eyesGrid.appendChild(buildMatchEyeCard("Specimen B", data.eye_b));
  content.appendChild(eyesGrid);

  const disclaimer = document.createElement("p");
  disclaimer.className = "match-disclaimer";
  disclaimer.textContent =
    "Fictional score based on visual texture only — not a real measure of compatibility, personality, or health.";
  content.appendChild(disclaimer);

  const scientificNote = document.createElement("p");
  scientificNote.className = "match-scientific-note";
  scientificNote.textContent =
    "Scientific significance: questionable. Entertainment value: considerable. " +
    "No peer-reviewed relationship conclusions were reached.";
  content.appendChild(scientificNote);
}

function buildSubscoreRow(label, value) {
  const row = document.createElement("div");
  row.className = "match-subscore-row";
  row.innerHTML = `
    <span class="match-subscore-label">${label}</span>
    <span class="match-subscore-bar"><span class="match-subscore-fill"></span></span>
    <span class="match-subscore-value">${value}%</span>
  `;
  // Bar starts at width 0 (see CSS) and transitions to its real value once
  // painted, so the fill reads as a measurement being taken rather than
  // appearing pre-filled. Two rAFs (not one) guarantee the 0% state has
  // actually painted before the transition to the target starts.
  const fill = row.querySelector(".match-subscore-fill");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      fill.style.width = `${value}%`;
    });
  });
  return row;
}

function buildMatchEyeCard(label, eye) {
  const div = document.createElement("div");
  div.className = "match-eye-card";

  const eyebrow = document.createElement("span");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = `${label} — ${capitalize(eye.position)} eye`;
  div.appendChild(eyebrow);

  if (eye.card_image) {
    const img = document.createElement("img");
    img.src = eye.card_image;
    img.alt = `${label} result card`;
    div.appendChild(img);
  }

  return div;
}

// ---------------------------------------------------------------------------
// A small easter egg -- clicking the wordmark five times reveals one line
// of dry commentary, then hides itself again a moment later. Rewards
// exploring the page without ever showing up unasked.
// ---------------------------------------------------------------------------

function initEasterEgg() {
  const title = el("wordmark-title");
  const line = el("easter-egg-line");
  if (!title || !line) return;

  let clicks = 0;
  let hideTimer = null;

  title.addEventListener("click", () => {
    clicks += 1;
    if (clicks < 5) return;
    clicks = 0;

    line.hidden = false;
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      line.hidden = true;
    }, 4000);
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

initTheme();
initNav();
initAnalyze();
initMatch();
initEasterEgg();
initStageModal();
