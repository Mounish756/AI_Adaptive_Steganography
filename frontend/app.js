/**
 * AI Adaptive Steganography Dashboard — Frontend Logic
 * Strictly handles:
 *  - Navigation & responsive sidebar
 *  - Backend API integration for embedding & extraction
 *  - Non-sensitive operational logging (no passwords/secrets recorded)
 *  - Client-side and server-side image fidelity calculations
 *  - Lossless PNG downloading and secure error handling
 */

"use strict";

// ============================================================
//  Configuration & API Endpoint
// ============================================================
let API_BASE = localStorage.getItem("stego_api_url") || "http://127.0.0.1:5050";

// ============================================================
//  Mobile Sidebar & Backdrop Navigation
// ============================================================
const sidebar = document.getElementById("sidebar");
const sidebarBackdrop = document.getElementById("sidebarBackdrop");
const menuBtn = document.getElementById("menuBtn");

function toggleSidebar(open) {
  if (!sidebar) return;
  const shouldOpen = open !== undefined ? open : !sidebar.classList.contains("open");
  sidebar.classList.toggle("open", shouldOpen);
  if (sidebarBackdrop) {
    sidebarBackdrop.classList.toggle("active", shouldOpen);
  }
}

menuBtn?.addEventListener("click", () => toggleSidebar());
sidebarBackdrop?.addEventListener("click", () => toggleSidebar(false));

function navigate(page) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));

  const pageEl = document.getElementById(`page-${page}`);
  if (pageEl) pageEl.classList.add("active");

  const navEl = document.querySelector(`[data-page="${page}"]`);
  if (navEl) navEl.classList.add("active");

  // Collapse sidebar on small screens
  toggleSidebar(false);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    navigate(item.dataset.page);
  });
});

// ============================================================
//  Server Status & Configuration Telemetry
// ============================================================
const statusDot        = document.getElementById("statusDot");
const statusText       = document.getElementById("statusText");
const dashStatusDot    = document.getElementById("dashboardStatusDot");
const dashStatusText   = document.getElementById("dashboardStatusText");
const dashApiStatus    = document.getElementById("dashboardApiStatus");
const headerNodeDevice = document.getElementById("headerNodeDevice");

async function checkServerStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      const data = await res.json().catch(() => ({}));
      const deviceStr = (data.device || "Local CPU").toUpperCase();

      if (statusDot) statusDot.className = "status-dot online";
      if (statusText) statusText.textContent = "API Service Online";

      if (dashStatusDot) dashStatusDot.className = "status-dot online";
      if (dashStatusText) dashStatusText.textContent = "API Connected • Active";
      if (dashApiStatus) dashApiStatus.className = "api-status-pill online";
      if (headerNodeDevice) headerNodeDevice.textContent = `Local Node (${deviceStr})`;

      return true;
    }
  } catch {}

  if (statusDot) statusDot.className = "status-dot offline";
  if (statusText) statusText.textContent = "API Service Offline";

  if (dashStatusDot) dashStatusDot.className = "status-dot offline";
  if (dashStatusText) dashStatusText.textContent = "API Service Offline";
  if (dashApiStatus) dashApiStatus.className = "api-status-pill offline";
  if (headerNodeDevice) headerNodeDevice.textContent = "Offline (start API)";

  return false;
}

async function loadCapacity() {
  try {
    const res = await fetch(`${API_BASE}/api/capacity`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const d = await res.json();
      const val = d.usable_payload_bytes;

      const statCap = document.getElementById("statCapacity");
      if (statCap) statCap.textContent = val ? `${val.toLocaleString()} B` : "—";

      const statCapCaption = document.getElementById("statCapacityCaption");
      if (statCapCaption && d.recommended_max_chars) {
        statCapCaption.textContent = `Max ~${d.recommended_max_chars.toLocaleString()} UTF-8 chars`;
      }

      const statRes = document.getElementById("statResolution");
      if (statRes && d.carrier_dim) {
        statRes.textContent = `${d.carrier_dim} × ${d.carrier_dim}`;
      }

      const statCrypto = document.getElementById("statCrypto");
      if (statCrypto && d.crypto_standard) {
        statCrypto.textContent = d.crypto_standard;
      }

      const statFEC = document.getElementById("statFEC");
      if (statFEC && d.fec_scheme) {
        statFEC.textContent = d.fec_scheme;
      }

      const settingsEl = document.getElementById("settingsCapacity");
      if (settingsEl) settingsEl.textContent = `${val} bytes (max ~${d.recommended_max_chars} UTF-8 chars recommended)`;
    }
  } catch {}
}

checkServerStatus().then(ok => { if (ok) loadCapacity(); });
setInterval(checkServerStatus, 25000);

// ============================================================
//  Recent Operations Log (Persisted Session Activity)
// ============================================================
const OPS_STORAGE_KEY = "stego_session_operations";
let operationsLog = [];

try {
  const saved = sessionStorage.getItem(OPS_STORAGE_KEY);
  if (saved) operationsLog = JSON.parse(saved);
} catch {
  operationsLog = [];
}

function saveOperationsLog() {
  try {
    sessionStorage.setItem(OPS_STORAGE_KEY, JSON.stringify(operationsLog));
  } catch {}
}

function recordOperation(entry) {
  operationsLog.unshift(entry);
  if (operationsLog.length > 25) operationsLog.pop();
  saveOperationsLog();
  renderOperationsLog();
}

function renderOperationsLog() {
  const tbody = document.getElementById("operationsTbody");
  const countBadge = document.getElementById("operationCountBadge");
  if (countBadge) {
    countBadge.textContent = `${operationsLog.length} operation${operationsLog.length === 1 ? '' : 's'}`;
  }
  if (!tbody) return;

  if (operationsLog.length === 0) {
    tbody.innerHTML = `
      <tr class="empty-row">
        <td colspan="6">
          <div class="table-empty-state">
            <div class="empty-icon-box">
              <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.8"/><path d="M12 7v5l3 3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
            </div>
            <span class="empty-title">No Cryptographic Operations Recorded</span>
            <span class="empty-sub">Embed or extract a secret message to log real-time fidelity measurements and verification events.</span>
          </div>
        </td>
      </tr>`;
    return;
  }

  tbody.innerHTML = operationsLog.map(item => `
    <tr>
      <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--text-muted);">${item.time}</td>
      <td><strong>${item.operation}</strong></td>
      <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">${item.size}</td>
      <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--luxury-gold); font-weight: 700;">${item.psnr}</td>
      <td style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">${item.ssim}</td>
      <td><span class="badge ${item.badgeClass}">${item.status}</span></td>
    </tr>
  `).join("");
}

// Initial render of existing session operations
renderOperationsLog();

document.getElementById("clearHistoryBtn")?.addEventListener("click", () => {
  operationsLog = [];
  saveOperationsLog();
  renderOperationsLog();
});

// ============================================================
//  UI Helpers
// ============================================================
function show(el) { if (el) el.hidden = false; }
function hide(el) { if (el) el.hidden = true; }

function showError(el, msg) {
  if (!el) return;
  el.textContent = msg;
  show(el);
}

function hideError(el) {
  if (!el) return;
  hide(el);
  el.textContent = "";
}

// Password toggle
document.querySelectorAll(".show-pw-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const input = document.getElementById(btn.dataset.target);
    if (!input) return;
    input.type = input.type === "password" ? "text" : "password";
  });
});

// Loading overlay
function showLoading(label = "Executing neural embedding…") {
  const el = document.getElementById("loadingOverlay");
  const lbl = document.getElementById("loadingLabel");
  if (lbl) lbl.textContent = label;
  if (el) {
    el.hidden = false;
    el.style.display = "flex";
  }
}

function hideLoading() {
  const el = document.getElementById("loadingOverlay");
  if (el) {
    el.hidden = true;
    el.style.display = "none";
  }
}


// File dropzone handler
function setupDropZone(zoneId, inputId, previewImgId, contentId, onFile) {
  const zone    = document.getElementById(zoneId);
  const input   = document.getElementById(inputId);
  const preview = document.getElementById(previewImgId);
  const content = document.getElementById(contentId);

  if (!zone) return;

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    onFile(file);
    const url = URL.createObjectURL(file);
    if (preview) {
      preview.src = url;
      show(preview);
    }
    if (content) hide(content);
  }

  zone.addEventListener("click", () => input?.click());
  zone.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input?.click();
    }
  });

  input?.addEventListener("change", () => {
    if (input.files[0]) handleFile(input.files[0]);
  });

  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) {
      if (input) {
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
      }
      handleFile(file);
    }
  });
}

// Char counter
const embedMessage = document.getElementById("embedMessage");
const charCount    = document.getElementById("embedCharCount");
embedMessage?.addEventListener("input", () => {
  const n = embedMessage.value.length;
  charCount.textContent = `${n} / 4000`;
  charCount.style.color = n > 3800 ? "#C0392B" : "var(--text-muted)";
});

// ============================================================
//  PAGE: EMBED MESSAGE
// ============================================================
let embedCoverFile   = null;
let embedStegoB64    = null;
let embedMessageText = "";
let embedPasswordVal = "";

setupDropZone("embedDropZone", "embedImageInput", "embedPreviewImg", "embedDropContent", f => {
  embedCoverFile = f;
});

const embedBtn = document.getElementById("embedBtn");
embedBtn?.addEventListener("click", async () => {
  const message  = document.getElementById("embedMessage").value.trim();
  const password = document.getElementById("embedPassword").value;
  const errEl    = document.getElementById("embedError");

  hideError(errEl);

  if (!embedCoverFile) {
    showError(errEl, "Please upload a cover carrier image.");
    return;
  }
  if (!message) {
    showError(errEl, "Please specify a secret message to embed.");
    return;
  }
  if (!password) {
    showError(errEl, "Please enter an encryption password.");
    return;
  }
  if (password.length < 4) {
    showError(errEl, "Password must be at least 4 characters long.");
    return;
  }

  embedMessageText = message;
  embedPasswordVal = password;

  const online = await checkServerStatus();
  if (!online) {
    showError(errEl, "Cannot connect to the backend server. Start it with: python start.py");
    return;
  }

  hide(document.getElementById("embedResultContent"));
  show(document.getElementById("embedResultPlaceholder"));

  showLoading("Encrypting payload & running adaptive neural embedding…");

  const origBtnHtml = embedBtn.innerHTML;
  embedBtn.disabled = true;
  embedBtn.innerHTML = `<span class="btn-spinner"></span> Encrypting &amp; Embedding…`;

  const form = new FormData();
  form.append("image",    embedCoverFile);
  form.append("message",  message);
  form.append("password", password);

  try {
    const res = await fetch(`${API_BASE}/api/embed`, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      hideLoading();
      showError(errEl, data.error || "Neural embedding failed.");
      return;
    }

    embedStegoB64 = data.stego_image_b64;
    const metrics = data.metrics;

    // Display previews & backend metrics
    document.getElementById("embedOriginalImg").src = data.original_b64;
    document.getElementById("embedStegoImg").src    = embedStegoB64;
    document.getElementById("mPSNR").textContent    = metrics.psnr_db.toFixed(2);
    document.getElementById("mSSIM").textContent    = metrics.ssim.toFixed(4);
    const dimsEl = document.getElementById("iDims");
    if (dimsEl) dimsEl.textContent = `${metrics.dimensions} (100% Preserved)`;
    document.getElementById("iPlain").textContent   = `${metrics.plaintext_bytes} bytes`;
    document.getElementById("iCipher").textContent  = `${metrics.ciphertext_bytes} bytes (AES-256-GCM)`;
    document.getElementById("iTime").textContent    = `${data.embedding_time_s.toFixed(3)} s`;

    hide(document.getElementById("embedResultPlaceholder"));
    show(document.getElementById("embedResultContent"));

    // Log operation without secret
    recordOperation({
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      operation: "Embed & Encrypt",
      size: `${metrics.plaintext_bytes} bytes`,
      psnr: `${metrics.psnr_db.toFixed(2)} dB`,
      ssim: metrics.ssim.toFixed(4),
      status: "AES-GCM Sealed",
      badgeClass: "badge-gold"
    });

  } catch (err) {
    showError(errEl, "Network connection error while contacting API server.");
    console.error(err);
  } finally {
    hideLoading();
    embedBtn.disabled = false;
    embedBtn.innerHTML = origBtnHtml;
  }
});

// Download stego PNG
document.getElementById("embedDownloadBtn")?.addEventListener("click", async () => {
  if (!embedCoverFile || !embedMessageText || !embedPasswordVal) return;
  showLoading("Packaging lossless PNG carrier…");

  const form = new FormData();
  form.append("image",    embedCoverFile);
  form.append("message",  embedMessageText);
  form.append("password", embedPasswordVal);

  try {
    const res = await fetch(`${API_BASE}/api/download_stego`, { method: "POST", body: form });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      alert(d.error || "Image download failed.");
      return;
    }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = "stego_image.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error(err);
    alert("Network error encountered during download.");
  } finally {
    hideLoading();
  }
});

// ============================================================
//  PAGE: EXTRACT MESSAGE
// ============================================================
let extractStegoFile = null;

setupDropZone("extractDropZone", "extractImageInput", "extractPreviewImg", "extractDropContent", f => {
  extractStegoFile = f;
});

const extractBtn = document.getElementById("extractBtn");
extractBtn?.addEventListener("click", async () => {
  const password = document.getElementById("extractPassword").value;
  const errEl    = document.getElementById("extractError");

  hideError(errEl);
  hide(document.getElementById("extractResultContent"));
  show(document.getElementById("extractResultPlaceholder"));

  if (!extractStegoFile) {
    showError(errEl, "Please upload the stego carrier image.");
    return;
  }
  if (!password) {
    showError(errEl, "Please enter the decryption password.");
    return;
  }

  const online = await checkServerStatus();
  if (!online) {
    showError(errEl, "Cannot connect to the backend server. Start it with: python start.py");
    return;
  }

  showLoading("Decoding carrier map & verifying cryptographic authentication tag…");

  const origExtractHtml = extractBtn.innerHTML;
  extractBtn.disabled = true;
  extractBtn.innerHTML = `<span class="btn-spinner"></span> Extracting &amp; Verifying…`;

  const form = new FormData();
  form.append("image",    extractStegoFile);
  form.append("password", password);

  try {
    const res  = await fetch(`${API_BASE}/api/extract`, { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      hideLoading();
      showError(errEl, data.error || "Authentication failed. Incorrect password or modified image.");
      recordOperation({
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        operation: "Extract & Decrypt",
        size: "—",
        psnr: "—",
        ssim: "—",
        status: "Auth Rejected",
        badgeClass: "badge-red"
      });
      return;
    }

    const msg = data.message || "";
    document.getElementById("recoveredMsg").textContent = msg;
    document.getElementById("eTime").textContent   = `${data.extraction_time_s.toFixed(3)} s`;
    document.getElementById("eLength").textContent = `${msg.length} characters`;

    hide(document.getElementById("extractResultPlaceholder"));
    show(document.getElementById("extractResultContent"));

    recordOperation({
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      operation: "Extract & Decrypt",
      size: `${msg.length} chars`,
      psnr: "—",
      ssim: "—",
      status: "GHASH Verified",
      badgeClass: "badge-green"
    });

  } catch (err) {
    showError(errEl, "Network error occurred while communicating with the extraction API.");
    console.error(err);
  } finally {
    hideLoading();
    extractBtn.disabled = false;
    extractBtn.innerHTML = origExtractHtml;
  }
});

// Copy message
document.getElementById("copyMsgBtn")?.addEventListener("click", () => {
  const msg = document.getElementById("recoveredMsg").textContent;
  navigator.clipboard.writeText(msg).then(() => {
    const btn = document.getElementById("copyMsgBtn");
    btn.textContent = "Copied!";
    setTimeout(() => {
      btn.innerHTML = `<svg viewBox="0 0 16 16" fill="none"><rect x="5" y="5" width="9" height="9" rx="1.5" stroke="currentColor" stroke-width="1.6"/><path d="M3 11V3h8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg> Copy to Clipboard`;
    }, 1500);
  });
});

// ============================================================
//  PAGE: IMAGE QUALITY ANALYSIS (Pixel-Level Verification)
// ============================================================
let qualOrigFile  = null;
let qualStegoFile = null;
let qualOrigImg   = null;
let qualStegoImg  = null;

setupDropZone("qualOrigDrop",  "qualOrigInput",  "qualOrigPreview",  "qualOrigContent",  f => { qualOrigFile  = f; qualOrigImg  = null; });
setupDropZone("qualStegoDrop", "qualStegoInput", "qualStegoPreview", "qualStegoContent", f => { qualStegoFile = f; qualStegoImg = null; });

function loadImageFile(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload  = () => resolve(img);
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function imageToPixels(img, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

function computeMetrics(d1, d2, w, h) {
  const n = w * h;
  let mse = 0;
  let maxDiff = 0;
  const a1 = d1.data;
  const a2 = d2.data;

  for (let i = 0; i < n; i++) {
    const base = i * 4;
    for (let c = 0; c < 3; c++) {
      const diff = a1[base + c] - a2[base + c];
      mse += diff * diff;
      if (Math.abs(diff) > maxDiff) maxDiff = Math.abs(diff);
    }
  }
  mse /= (n * 3);

  const psnr = mse > 0 ? 10 * Math.log10((255 * 255) / mse) : Infinity;

  // Global Structural Similarity index approximation
  let mu1 = 0, mu2 = 0;
  for (let i = 0; i < n; i++) {
    const b = i * 4;
    mu1 += (a1[b] * 0.299 + a1[b+1] * 0.587 + a1[b+2] * 0.114);
    mu2 += (a2[b] * 0.299 + a2[b+1] * 0.587 + a2[b+2] * 0.114);
  }
  mu1 /= n;
  mu2 /= n;

  let s1 = 0, s2 = 0, s12 = 0;
  for (let i = 0; i < n; i++) {
    const b  = i * 4;
    const l1 = (a1[b] * 0.299 + a1[b+1] * 0.587 + a1[b+2] * 0.114) - mu1;
    const l2 = (a2[b] * 0.299 + a2[b+1] * 0.587 + a2[b+2] * 0.114) - mu2;
    s1 += l1 * l1;
    s2 += l2 * l2;
    s12 += l1 * l2;
  }
  s1 /= n;
  s2 /= n;
  s12 /= n;

  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  const ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
               ((mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2));

  return { psnr, ssim, mse, maxDiff };
}

function renderDiff(d1, d2, canvas) {
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(w, h);
  const a1 = d1.data;
  const a2 = d2.data;

  for (let i = 0; i < w * h; i++) {
    const b = i * 4;
    for (let c = 0; c < 3; c++) {
      out.data[b + c] = Math.min(255, Math.abs(a1[b + c] - a2[b + c]) * 20);
    }
    out.data[b + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
}

document.getElementById("qualCompareBtn")?.addEventListener("click", async () => {
  const errEl = document.getElementById("qualError");
  hideError(errEl);
  hide(document.getElementById("qualityResults"));

  if (!qualOrigFile || !qualStegoFile) {
    showError(errEl, "Please upload both the original reference and stego images.");
    return;
  }

  showLoading("Computing comparative fidelity metrics and generating difference maps…");

  try {
    qualOrigImg  = await loadImageFile(qualOrigFile);
    qualStegoImg = await loadImageFile(qualStegoFile);

    const W = qualOrigImg.naturalWidth;
    const H = qualOrigImg.naturalHeight;

    const d1 = imageToPixels(qualOrigImg,  W, H);
    const d2 = imageToPixels(qualStegoImg, W, H);

    const { psnr, ssim, mse, maxDiff } = computeMetrics(d1, d2, W, H);

    document.getElementById("qPSNR").textContent = isFinite(psnr) ? psnr.toFixed(2) : "∞";
    document.getElementById("qSSIM").textContent = ssim.toFixed(4);
    document.getElementById("qMSE").textContent  = mse.toExponential(4);
    document.getElementById("qMaxDiff").textContent = maxDiff;

    document.getElementById("qPSNRNote").textContent = psnr >= 35 ? "High Fidelity" : psnr >= 25 ? "Standard" : "Degraded";
    document.getElementById("qSSIMNote").textContent = ssim >= 0.95 ? "Excellent Correlation" : ssim >= 0.8 ? "Good" : "Noticeable Shift";
    document.getElementById("qMSENote").textContent  = "";

    const origCanvas  = document.getElementById("qOrigCanvas");
    const stegoCanvas = document.getElementById("qStegoCanvas");
    const diffCanvas  = document.getElementById("qDiffCanvas");

    for (const c of [origCanvas, stegoCanvas, diffCanvas]) {
      c.width = W;
      c.height = H;
    }

    origCanvas.getContext("2d").drawImage(qualOrigImg,  0, 0, W, H);
    stegoCanvas.getContext("2d").drawImage(qualStegoImg, 0, 0, W, H);
    renderDiff(d1, d2, diffCanvas);

    show(document.getElementById("qualityResults"));
  } catch (err) {
    showError(errEl, "Image analysis error: " + err.message);
    console.error(err);
  } finally {
    hideLoading();
  }
});

// View mode toggles
document.getElementById("viewSideBySide")?.addEventListener("click", () => {
  document.getElementById("sideBySidePanel").hidden = false;
  document.getElementById("diffPanel").hidden = true;
  document.getElementById("viewSideBySide").classList.add("active");
  document.getElementById("viewDiff").classList.remove("active");
});

document.getElementById("viewDiff")?.addEventListener("click", () => {
  document.getElementById("sideBySidePanel").hidden = true;
  document.getElementById("diffPanel").hidden = false;
  document.getElementById("viewDiff").classList.add("active");
  document.getElementById("viewSideBySide").classList.remove("active");
});

// ============================================================
//  PAGE: SETTINGS
// ============================================================
document.getElementById("apiUrl")?.addEventListener("change", e => {
  API_BASE = e.target.value.replace(/\/$/, "");
  localStorage.setItem("stego_api_url", API_BASE);
});

document.getElementById("testConnBtn")?.addEventListener("click", async () => {
  const resultEl = document.getElementById("connTestResult");
  resultEl.className = "alert";
  resultEl.textContent = "Testing connection…";
  show(resultEl);

  try {
    const urlInput = document.getElementById("apiUrl").value.replace(/\/$/, "");
    const res = await fetch(`${urlInput}/api/health`, { signal: AbortSignal.timeout(4000) });
    const data = await res.json();
    if (res.ok) {
      resultEl.className = "alert alert-success";
      resultEl.textContent = `✓ Successfully connected to backend API running on device: [${data.device}]`;
      API_BASE = urlInput;
      localStorage.setItem("stego_api_url", API_BASE);
      checkServerStatus();
      loadCapacity();
    } else {
      resultEl.className = "alert alert-error";
      resultEl.textContent = "Server returned an unexpected response code.";
    }
  } catch {
    resultEl.className = "alert alert-error";
    resultEl.textContent = "Connection refused. Confirm server is running at " + document.getElementById("apiUrl").value;
  }
});

// Initialize dashboard view
navigate("dashboard");
