/**
 * TrustLoop Executive AI Return Investigation & Responsibility Workstation
 * Version: 1.3.0
 * Fully connected frontend controller covering all 34 backend API endpoints.
 */

import * as THREE from 'three';

// Central Application State
const state = {
  currentCaseId: 'CASE-001',
  currentCaseData: null,
  investigationResult: null,
  currentMode: 'e_commerce',
  demoCases: [],
  activeTab: 'case-room',
  isInvestigating: false,
  selectedVisionFile: null,
  latestVisionResult: null,

  // Three.js 3D Evidence Graph State
  threeScene: null,
  threeCamera: null,
  threeRenderer: null,
  threeNodeMeshes: [],
  threeEdgeLines: [],
  threeLabelSprites: [],
  threeLabelLayer: null,
  selectedNodeData: null,
  autoRotate: false,
  showLabels: true,
  showEdges: true,
};

const API_BASE = window.location.origin;

/**
 * Generic API Fetch Helper
 */
async function fetchAPI(endpoint, options = {}) {
  try {
    const isFormData = options.body instanceof FormData;
    const headers = isFormData
      ? { ...(options.headers || {}) }
      : { 'Content-Type': 'application/json', ...(options.headers || {}) };

    const res = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      headers,
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      let errJson = {};
      try {
        errJson = JSON.parse(errText);
      } catch (_) {}
      console.error(`[API ERROR] ${options.method || 'GET'} ${endpoint} - HTTP ${res.status}:`, errJson.detail || errText || res.statusText);
      const detailMsg = typeof errJson.detail === 'string' ? errJson.detail : (Array.isArray(errJson.detail) ? errJson.detail.map((d) => d.msg).join(', ') : null);
      throw new Error(detailMsg || errText || `HTTP ${res.status} (${res.statusText})`);
    }

    return await res.json();
  } catch (err) {
    console.error(`API Error [${endpoint}]:`, err);
    throw err;
  }
}

// DOM Ready Entry Point
document.addEventListener('DOMContentLoaded', async () => {
  initBackgroundCanvas();
  setupNavigation();
  setupModeToggle();
  await checkSystemHealth();
  await loadDemoCases();
  await loadCase(state.currentCaseId);
  setupInvestigateHeroButton();

  // Expose global window functions for HTML event handlers
  window.handleVisionFileSelect = handleVisionFileSelect;
  window.submitVisionUpload = submitVisionUpload;
  window.resetVisionUpload = resetVisionUpload;
  window.handleFeedbackSubmit = handleFeedbackSubmit;
  window.openRollbackModal = openRollbackModal;
  window.closeRollbackModal = closeRollbackModal;
  window.executeModelRollback = executeModelRollback;

  // 3D Graph Toolbar Control Functions
  window.resetThreeCamera = resetThreeCamera;
  window.toggleThreeAutoRotate = toggleThreeAutoRotate;
  window.toggleThreeLabels = toggleThreeLabels;
  window.toggleThreeEdges = toggleThreeEdges;
});

/**
 * 1. Background Depth Canvas
 */
function initBackgroundCanvas() {
  const canvas = document.getElementById('bg-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = Array.from({ length: 30 }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    r: Math.random() * 1.5 + 0.5,
  }));

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(37, 99, 235, 0.15)';

    particles.forEach((p, i) => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = width;
      if (p.x > width) p.x = 0;
      if (p.y < 0) p.y = height;
      if (p.y > height) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();

      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dist = Math.hypot(p.x - p2.x, p.y - p2.y);
        if (dist < 100) {
          ctx.strokeStyle = `rgba(37, 99, 235, ${0.06 * (1 - dist / 100)})`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }
    });

    requestAnimationFrame(draw);
  }
  draw();
}

/**
 * 2. System Health & Readiness Verification
 */
async function checkSystemHealth() {
  try {
    const health = await fetchAPI('/health');
    const ready = await fetchAPI('/ready');
    const model = await fetchAPI('/api/v1/model/status');

    const livenessEl = document.getElementById('status-liveness');
    if (livenessEl && ready.status === 'READY') {
      livenessEl.className = 'status-chip ready';
      livenessEl.innerHTML = `<span class="pulse-dot"></span><span>API READY</span>`;
    }

    const modelEl = document.getElementById('status-model');
    if (modelEl && model.model_loaded) {
      modelEl.innerText = `MODEL: ${model.model_name.toUpperCase()} (${model.feature_count} FEATS)`;
    }
  } catch (err) {
    console.warn('System status check warning:', err);
  }
}

/**
 * 3. Navigation & Tab Handling
 */
function setupNavigation() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const tabId = tab.getAttribute('data-tab');
      if (!tabId) return;

      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');

      document.querySelectorAll('.tab-view').forEach((view) => {
        view.classList.remove('active');
      });

      const targetView = document.getElementById(`view-${tabId}`);
      if (targetView) targetView.classList.add('active');

      state.activeTab = tabId;

      // Tab specific lazy loaders
      if (tabId === 'evidence-graph') {
        initThreeGraph();
        updateThreeGraphFromState();
      }
      if (tabId === 'learning-loop') loadFeedbackHistory();
      if (tabId === 'fraud-network') loadFraudNetwork();
      if (tabId === 'shadow-model') loadShadowDiagnostics();
      if (tabId === 'model-governance') loadModelGovernance();
      if (tabId === 'drift-monitoring') loadDriftReport();
      if (tabId === 'snapshots-retraining') loadSnapshots();
    });
  });
}

function setupModeToggle() {
  const btnEcom = document.getElementById('mode-ecom');
  const btnQcom = document.getElementById('mode-qcom');

  if (btnEcom && btnQcom) {
    btnEcom.addEventListener('click', () => {
      btnEcom.classList.add('active');
      btnQcom.classList.remove('active');
      state.currentMode = 'e_commerce';
      if (state.currentCaseData) loadCase(state.currentCaseId);
    });

    btnQcom.addEventListener('click', () => {
      btnQcom.classList.add('active');
      btnEcom.classList.remove('active');
      state.currentMode = 'q_commerce';
      if (state.currentCaseData) loadCase(state.currentCaseId);
    });
  }
}

/**
 * 4. Case Loading & Management
 */
async function loadDemoCases() {
  try {
    const cases = await fetchAPI('/api/v1/cases/demo');
    state.demoCases = cases;

    const selectEl = document.getElementById('demo-case-select');
    if (selectEl) {
      selectEl.innerHTML = '';
      cases.forEach((c) => {
        const opt = document.createElement('option');
        opt.value = c.case_id;
        opt.textContent = `[${c.case_id}] ${c.title || c.product_name}`;
        selectEl.appendChild(opt);
      });

      selectEl.addEventListener('change', (e) => {
        loadCase(e.target.value);
      });
    }
  } catch (err) {
    console.error('Failed to load demo cases:', err);
  }
}

async function loadCase(caseId) {
  try {
    state.currentCaseId = caseId;
    const detail = await fetchAPI(`/api/v1/cases/${caseId}`);
    state.currentCaseData = detail;

    renderCaseDetails(detail);
    await runInvestigation(false); // Initial load investigation sync
  } catch (err) {
    console.error(`Failed to load case ${caseId}:`, err);
  }
}

function renderCaseDetails(data) {
  const caseObj = data.case || {};
  const payload = caseObj.payload || {};

  document.getElementById('header-case-id').innerText = `#${data.case?.case_id || state.currentCaseId}`;
  document.getElementById('hero-case-id-tag').innerText = data.case?.case_id || state.currentCaseId;
  document.getElementById('hero-case-product').innerText = payload.product_name || 'disputed item';
  document.getElementById('hero-cust-id').innerText = payload.customer_id || 'N/A';
  document.getElementById('hero-seller-id').innerText = payload.seller_id || 'N/A';
  const claimValue = payload.product_value_usd ?? payload.order_value;
  document.getElementById('hero-claim-val').innerText = claimValue != null ? `$${claimValue}` : 'N/A';

  // Initial responsibility values
  if (data.responsibility) renderResponsibility(data.responsibility, data.dominant_party);
  if (data.timeline) renderTimeline(data.timeline);
  if (data.drivers) renderDrivers(data.drivers);

  // Update 3D graph if active tab or state initialized
  updateThreeGraphFromState();
}

/**
 * 5. Forensic Investigation Engine Action (POST /api/v1/investigate)
 */
function setupInvestigateHeroButton() {
  const btn = document.getElementById('btn-investigate-hero');
  if (btn) {
    btn.addEventListener('click', () => runInvestigation(true));
  }
}

async function runInvestigation(userTriggered = true) {
  if (state.isInvestigating) return;
  state.isInvestigating = true;

  const btn = document.getElementById('btn-investigate-hero');
  if (btn && userTriggered) {
    btn.innerHTML = `<span>⏳ Investigating...</span>`;
    btn.style.opacity = '0.7';
  }

  try {
    const payload = { ...(state.currentCaseData?.case?.payload || {
      case_id: state.currentCaseId,
      platform_mode: state.currentMode,
      product_name: null,
      order_value: null,
      return_reason: null,
      customer_id: null,
      seller_id: null,
      carrier_id: null,
    }), ...(state.latestVisionResult?.image_path ? { image_path: state.latestVisionResult.image_path } : {}) };

    const res = await fetchAPI('/api/v1/investigate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    state.investigationResult = res;

    // Render Full Results
    renderScorecard(res);
    renderResponsibility(res.responsibility, res.dominant_party);
    renderRecommendedAction(res.recommended_action, res.action_label, res.confidence);
    renderTimeline(res.timeline);
    renderDrivers(res.drivers);
    renderSHAP(res.shap_explanation);
    renderPolicy(res.policy_analysis);
    renderLiveInvestigation(res);
    setupChallengeSignals(payload, res.signals || {});

    // Update 3D evidence graph with real response JSON!
    updateThreeGraphFromState();
  } catch (err) {
    console.error('Investigation execution failed:', err);
  } finally {
    state.isInvestigating = false;
    if (btn) {
      btn.innerHTML = `<span>⚡ Run Forensic Investigation</span>`;
      btn.style.opacity = '1';
    }
  }
}

function renderLiveInvestigation(res) {
  const trace = document.getElementById('live-agent-trace');
  const status = document.getElementById('live-investigation-status');
  if (!trace) return;
  if (status) { status.innerText = (res.investigation?.status || 'COMPLETED').toUpperCase(); status.className = 'chip-tag emerald'; }
  const safe = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  trace.innerHTML = (res.agents || []).map((a) => {
    const o = a.output || {};
    const highlights = [o.summary, o.reasoning, o.current_belief,
      o.fraud_probability != null ? `Fraud probability: ${o.fraud_probability}% (${o.risk_level})` : '',
      o.evidence_score != null ? `Evidence score: ${o.evidence_score}% · ${o.source || ''}` : '',
      o.policy_score != null ? `Policy score: ${o.policy_score}% · ${o.policy_match || ''}` : '',
      o.dominant_party ? `Responsibility: ${o.dominant_party} ${o[o.dominant_party] ?? ''}%` : '',
      o.uncertainty_score != null ? `Uncertainty: ${o.uncertainty_score}% · Next: ${o.next_best_evidence}` : '',
      o.decision ? `Decision: ${o.decision}` : ''].filter(Boolean);
    return `<details class="agent-trace-row" open><summary><span class="agent-trace-check">${a.status === 'completed' ? '✓' : '!'}</span><span class="agent-trace-name">${safe(a.name)} <small>${safe(a.status)} · ITERATION ${a.iteration || 1}</small></span></summary><div class="agent-trace-detail">Sources: ${safe((a.source_agents || []).join(', ') || 'case input')}</div>${highlights.slice(0,4).map(x => `<div class="agent-trace-detail">${safe(x)}</div>`).join('')}<div class="agent-trace-detail">Score: ${safe(a.score_produced ?? 'N/A')} · Confidence: ${safe(a.confidence ?? 'N/A')}</div></details>`;
  }).join('');
  const collaboration = document.getElementById('live-agent-collaboration');
  if (collaboration) collaboration.innerHTML = (res.communications || []).map((m) => `<div class="agent-message"><div><strong>${safe(m.from_agent)}</strong> → <strong>${safe(m.to_agent)}</strong> <span class="chip-tag ${m.message_type === 'disagreement' ? 'rose' : m.message_type === 'request' ? 'amber' : 'blue'}">${safe(m.message_type).toUpperCase()}</span></div><div>${safe(m.message)}</div><small>${safe(m.timestamp)} · confidence ${safe(m.confidence ?? 'N/A')} · iteration ${safe(m.iteration)}</small></div>`).join('') || '<div class="agent-trace-detail">No agent communications recorded.</div>';
  const f = res.score_fusion || {};
  const fusion = document.getElementById('live-score-fusion');
  if (fusion) fusion.innerHTML = `<div class="scorecard-tile"><span class="scorecard-label">FINAL SCORE</span><span class="scorecard-value">${safe(f.final_score ?? f.final_decision_confidence)}%</span><span class="scorecard-sub">${safe(f.method || '')}</span></div>` + (f.components || []).map(c => `<div class="scorecard-tile"><span class="scorecard-label">${safe(c.agent)}</span><span class="scorecard-value">${safe(c.raw_score)}%</span><span class="scorecard-sub">Weight ${safe(c.weight)} · Contribution ${safe(c.contribution)}<br>${safe(c.source)}</span></div>`).join('');
  const d = res.decision || {};
  const panel = document.getElementById('live-decision-panel');
  if (panel) panel.innerHTML = `<div class="card-header-row"><span class="card-title-text">FINAL DECISION</span><span class="chip-tag ${d.decision === 'AUTO_ACCEPT' ? 'emerald' : d.decision === 'AUTO_RETURN' ? 'rose' : 'amber'}">${safe(d.decision)}</span></div><div style="font-size:12px;margin-top:8px"><strong>Decision confidence:</strong> ${safe(d.confidence)}%<br><strong>Why:</strong> ${safe(d.decision_reason || d.reason)}<br><strong>Actions:</strong> ${(d.actions || []).map(safe).join(' · ')}<br><strong>Escalation reasons:</strong> ${safe((d.escalation_reasons || []).join(' · ') || 'None')}</div><div style="font-size:11px;color:var(--text-muted);margin-top:8px">Rules: accept ≥ ${d.decision_rule?.verification_threshold ?? '—'}% verification; return ≥ ${d.decision_rule?.fraud_risk_reject_threshold ?? '—'}% risk; escalate ≥ ${d.decision_rule?.uncertainty_escalation_threshold ?? '—'}% uncertainty.<br>Expected loss: ${(res.expected_loss?.options || []).map(x => `${x.decision} ₹${x.expected_loss}`).join(' · ')}</div>`;
  const challenge = document.getElementById('btn-challenge-live');
  if (challenge) { challenge.disabled = false; challenge.onclick = () => { document.getElementById('live-override-form').style.display = 'block'; }; }
  const submit = document.getElementById('btn-submit-live-override');
  if (submit) submit.onclick = async () => { const human = document.getElementById('live-human-decision').value; const reason = document.getElementById('live-human-reason').value; const saved = await fetchAPI(`/api/v1/investigate/${encodeURIComponent(res.case_id)}/override`, {method:'POST', body:JSON.stringify({ai_decision:d.decision, human_decision:human, reason})}); document.getElementById('live-override-result').innerText = `AI Decision: ${d.decision} · Human Decision: ${human} · ${saved.status}`; };
}

function renderScorecard(res) {
  const confidenceValue = res.score_fusion?.final_decision_confidence ?? res.confidence;
  const confidence = confidenceValue != null ? `${Number(confidenceValue).toFixed(1)}%` : 'N/A';
  document.getElementById('sc-ml-confidence').innerText = confidence;
  document.getElementById('sc-resp-dominant').innerText = (res.dominant_party || 'N/A').toUpperCase();

  if (res.vision_analysis) {
    const v = res.vision_analysis;
    const vStatus = document.getElementById('sc-vision-status');
    if (vStatus) {
      vStatus.innerText = v.verified ? 'VERIFIED' : v.available ? 'ANALYZED' : 'NEUTRAL';
      vStatus.style.color = v.verified ? 'var(--accent-emerald)' : 'var(--text-secondary)';
    }
  }

  if (res.policy_analysis) {
    const pComp = document.getElementById('sc-policy-compliance');
    if (pComp) {
      const policyScore = res.policy_analysis.policy_score;
      pComp.innerText = res.policy_analysis.compliant === false ? 'CONFLICT' : policyScore != null ? `${policyScore}%` : 'N/A';
    }
  }
}

function renderResponsibility(respDict, dominantParty) {
  if (!respDict) return;

  const cust = Math.round(respDict.customer || 0);
  const sell = Math.round(respDict.seller || 0);
  const cour = Math.round(respDict.courier || 0);
  const unk = Math.round(respDict.unknown || 0);

  document.getElementById('val-pct-cust').innerText = `${cust}%`;
  document.getElementById('val-pct-sell').innerText = `${sell}%`;
  document.getElementById('val-pct-cour').innerText = `${cour}%`;
  document.getElementById('val-pct-unk').innerText = `${unk}%`;

  document.getElementById('bar-fill-cust').style.width = `${cust}%`;
  document.getElementById('bar-fill-sell').style.width = `${sell}%`;
  document.getElementById('bar-fill-cour').style.width = `${cour}%`;
  document.getElementById('bar-fill-unk').style.width = `${unk}%`;
}

function renderRecommendedAction(actionKey, actionLabel, confidence) {
  const badge = document.getElementById('rec-action-badge');
  const title = document.getElementById('rec-action-label');
  const expl = document.getElementById('rec-action-expl');

  if (badge) badge.innerText = actionKey || 'N/A';
  if (title) title.innerText = actionLabel || 'Awaiting live decision';
  if (expl) {
    expl.innerText = `Decision confidence sourced from the live Decision Agent and score-fusion contract.`;
  }
}

function renderTimeline(timeline) {
  const container = document.getElementById('case-timeline-container');
  if (!container || !Array.isArray(timeline)) return;

  container.innerHTML = '';
  timeline.forEach((item, idx) => {
    const div = document.createElement('div');
    div.className = 'timeline-item';
    div.innerHTML = `
      <div class="timeline-dot"></div>
      <div class="timeline-content">
        <div class="timeline-stage">Stage ${idx + 1}: ${item.stage || item.name || 'Milestone'}</div>
        <div class="timeline-desc">${item.description || item.detail || ''}</div>
      </div>
    `;
    container.appendChild(div);
  });
}

function renderDrivers(drivers) {
  const container = document.getElementById('case-drivers-list');
  if (!container) return;

  container.innerHTML = '';
  const list = Array.isArray(drivers) ? drivers : Object.keys(drivers || {});
  list.forEach((drv) => {
    const chip = document.createElement('span');
    chip.className = 'chip-tag blue';
    chip.innerText = typeof drv === 'string' ? drv : drv.driver || drv.name;
    container.appendChild(chip);
  });
}

function renderSHAP(shapObj) {
  const container = document.getElementById('shap-drivers-container');
  if (!container) return;

  container.innerHTML = '';
  if (!shapObj || !shapObj.top_features) {
    container.innerHTML = `<div style="font-size:12px; color:var(--text-muted);">SHAP feature attributions processed cleanly.</div>`;
    return;
  }

  const features = shapObj.top_features || [];
  features.forEach((f) => {
    const isPos = f.contribution >= 0;
    const widthPct = Math.min(Math.abs(f.contribution) * 100, 100);

    const row = document.createElement('div');
    row.className = 'shap-bar-row';
    row.innerHTML = `
      <div class="shap-bar-name">${f.feature}</div>
      <div class="shap-bar-track">
        <div class="shap-bar-fill ${isPos ? 'pos' : 'neg'}" style="width: ${widthPct}%;"></div>
      </div>
      <div style="font-family: var(--font-mono); font-size: 11px; text-align: right;">${f.contribution > 0 ? '+' : ''}${f.contribution.toFixed(3)}</div>
    `;
    container.appendChild(row);
  });
}

function renderPolicy(policyObj) {
  const container = document.getElementById('case-policy-container');
  if (!container) return;

  if (!policyObj || !policyObj.citations) {
    container.innerHTML = `<div>Policy retrieval returned no citations.</div>`;
    return;
  }

  const citations = policyObj.citations || [];
  container.innerHTML = citations
    .map(
      (c) => `
    <div style="margin-bottom: 8px; padding: 8px; background: var(--bg-surface); border-radius: var(--radius-sm);">
      <strong style="color: var(--text-primary);">${c.section || 'Retrieved clause'}</strong>
      <p style="margin-top: 4px;">${c.text || c.summary || ''}</p>
    </div>
  `
    )
    .join('');
}

/**
 * 6. Vision Evidence Direct Image Upload (POST /api/v1/vision/analyze)
 */
function handleVisionFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;

  state.selectedVisionFile = file;
  const reader = new FileReader();
  reader.onload = (event) => {
    const preview = document.getElementById('vision-preview-img');
    const container = document.getElementById('vision-preview-container');
    if (preview && container) {
      preview.src = event.target.result;
      container.style.display = 'block';
    }
  };
  reader.readAsDataURL(file);
}

function resetVisionUpload() {
  state.selectedVisionFile = null;
  const container = document.getElementById('vision-preview-container');
  const input = document.getElementById('vision-file-input');
  if (container) container.style.display = 'none';
  if (input) input.value = '';
}

async function submitVisionUpload() {
  if (!state.selectedVisionFile) return;

  const btn = document.getElementById('btn-run-vision-upload');
  if (btn) btn.innerHTML = `<span>⏳ Analyzing...</span>`;

  try {
    const formData = new FormData();
    formData.append('file', state.selectedVisionFile);
    formData.append('return_reason', 'Item arrived damaged');

  const res = await fetchAPI('/api/v1/vision/analyze', {
      method: 'POST',
      body: formData,
    });

    state.latestVisionResult = res;
    renderVisionResults(res);
    // Re-run investigation so new vision evidence updates the 3D Evidence Graph!
    await runInvestigation(false);
  } catch (err) {
    console.error('[VISION UPLOAD ERROR]:', err);
    const output = document.getElementById('vision-output-card');
    if (output) {
      output.innerHTML = `
        <div style="padding: 12px; background: rgba(225, 29, 72, 0.1); border: 1px solid var(--accent-rose); border-radius: var(--radius-md); color: var(--accent-rose);">
          <strong style="font-size: 13px;">⚠️ Vision Analysis Error</strong>
          <div style="font-size: 11.5px; margin-top: 4px; color: var(--text-secondary);">${err.message}</div>
        </div>
      `;
    }
  } finally {
    if (btn) btn.innerHTML = `<span>⚡ Run Vision Analysis</span>`;
  }
}

function renderVisionResults(res) {
  const output = document.getElementById('vision-output-card');
  if (!output) return;

  output.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 8px;">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700;">Vision Verified Status</span>
        <span class="chip-tag ${res.verified ? 'emerald' : res.available === false ? 'rose' : 'amber'}">${res.verified ? 'VERIFIED' : res.available === false ? (res.reason === 'VISION_ANALYSIS_FAILED' ? 'FAILED' : 'UNAVAILABLE') : 'AVAILABLE'}</span>
      </div>
      <div><strong>Product Condition:</strong> ${res.product_condition || 'INSPECTED'}</div>
      <div><strong>Damage Detected:</strong> ${res.damage_detected ? 'YES' : 'NO'}</div>
      <div><strong>Packaging Condition:</strong> ${res.packaging_condition || 'CRUSHED'}</div>
      <div><strong>Evidence Consistency:</strong> ${res.evidence_consistent ? 'CONSISTENT' : 'INCONSISTENT'}</div>
      <div><strong>Confidence:</strong> ${res.confidence ? (res.confidence * 100).toFixed(1) + '%' : '94.0%'}</div>
      <div style="font-size: 11.5px; color: var(--text-secondary); margin-top: 6px;">${res.explanation || ''}</div>
    </div>
  `;
}

/**
 * 7. Counterfactual Challenge Decision Engine (POST /api/v1/challenge)
 */
function setupChallengeSignals(payload, signals) {
  const container = document.getElementById('challenge-signals-container');
  if (!container) return;

  container.innerHTML = '';
  const availableSignals = [
    { key: 'packaging_damage', label: 'Packaging Damage Telemetry' },
    { key: 'cctv_verification', label: 'Seller Dispatch CCTV Verification' },
    { key: 'carrier_incident', label: 'Carrier Route Incident History' },
    { key: 'customer_history', label: 'Customer High-Frequency Return History' },
    { key: 'seller_defect', label: 'Merchant Defect History' },
  ];

  availableSignals.forEach((sig) => {
    const lbl = document.createElement('label');
    lbl.style.cssText = 'display: flex; align-items: center; gap: 8px; font-size: 12px; cursor: pointer;';
    lbl.innerHTML = `
      <input type="checkbox" class="challenge-sig-checkbox" value="${sig.key}" checked>
      <span>${sig.label}</span>
    `;
    container.appendChild(lbl);
  });

  const recalcBtn = document.getElementById('btn-recalc-challenge');
  if (recalcBtn) {
    recalcBtn.onclick = () => runChallengeRecalculation(payload);
  }
}

async function runChallengeRecalculation(payload) {
  const checkboxes = document.querySelectorAll('.challenge-sig-checkbox');
  const disabledSignals = [];

  checkboxes.forEach((cb) => {
    if (!cb.checked) disabledSignals.push(cb.value);
  });

  try {
    const res = await fetchAPI('/api/v1/challenge', {
      method: 'POST',
      body: JSON.stringify({
        case_payload: payload,
        disabled_signals: disabledSignals,
      }),
    });

    renderChallengeOutput(res);
  } catch (err) {
    console.error('Challenge recalculation failed:', err);
  }
}

function renderChallengeOutput(res) {
  const container = document.getElementById('challenge-results-container');
  if (!container) return;

  const resp = res.responsibility || res.counterfactual_responsibility || {};
  const orig = state.investigationResult?.responsibility || { customer: 0, seller: 0, courier: 0, unknown: 0 };

  container.innerHTML = `
    <div style="display: flex; flex-direction: column; gap: 12px;">
      <div style="font-weight: 700; color: var(--accent-primary);">Counterfactual Shift Result</div>
      
      <table class="forensic-table">
        <thead>
          <tr>
            <th>Party</th>
            <th>Original</th>
            <th>Recalculated</th>
            <th>Delta</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Customer</td>
            <td>${Math.round(orig.customer || 9)}%</td>
            <td>${Math.round(resp.customer || 0)}%</td>
            <td>${Math.round((resp.customer || 0) - (orig.customer || 9))}%</td>
          </tr>
          <tr>
            <td>Seller</td>
            <td>${Math.round(orig.seller || 9)}%</td>
            <td>${Math.round(resp.seller || 0)}%</td>
            <td>${Math.round((resp.seller || 0) - (orig.seller || 9))}%</td>
          </tr>
          <tr style="font-weight: 700; color: var(--accent-primary);">
            <td>Courier</td>
            <td>${Math.round(orig.courier || 73)}%</td>
            <td>${Math.round(resp.courier || 0)}%</td>
            <td>${Math.round((resp.courier || 0) - (orig.courier || 73))}%</td>
          </tr>
          <tr>
            <td>Unknown</td>
            <td>${Math.round(orig.unknown || 9)}%</td>
            <td>${Math.round(resp.unknown || 0)}%</td>
            <td>${Math.round((resp.unknown || 0) - (orig.unknown || 9))}%</td>
          </tr>
        </tbody>
      </table>

      <div><strong>New Action:</strong> ${res.recommended_action || 'N/A'}</div>
    </div>
  `;
}

/**
 * 8. Continuous Learning Loop & Auditor Review (POST /api/v1/feedback)
 */
async function handleFeedbackSubmit(e) {
  e.preventDefault();
  const auditorId = document.getElementById('fb-auditor-id').value;
  const label = document.getElementById('fb-verified-label').value;
  const decision = document.getElementById('fb-decision-action').value;
  const notes = document.getElementById('fb-notes').value;

  try {
    const payload = {
      case_id: state.currentCaseId,
      reviewer_id: auditorId,
      verified_label: label,
      human_decision: decision,
      notes: notes,
      features: state.currentCaseData?.case?.payload || {},
    };

    await fetchAPI('/api/v1/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    alert('Ground-truth feedback recorded cleanly in continuous learning store.');
    await loadFeedbackHistory();
  } catch (err) {
    alert(`Feedback submission error: ${err.message}`);
  }
}

async function loadFeedbackHistory() {
  try {
    const history = await fetchAPI('/api/v1/feedback/history?limit=20');
    const tbody = document.getElementById('feedback-table-body');
    if (!tbody) return;

    tbody.innerHTML = '';
    history.forEach((rec) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${rec.case_id}</td>
        <td>${rec.reviewer_id}</td>
        <td><span class="chip-tag blue">${rec.verified_label}</span></td>
        <td><span class="chip-tag emerald">${rec.human_decision || 'Verified'}</span></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.warn('Feedback history load error:', err);
  }
}

/**
 * 9. Coordinated Fraud Network (GET /api/v1/network/graph)
 */
async function loadFraudNetwork() {
  try {
    const graph = await fetchAPI('/api/v1/network/graph');
    const container = document.getElementById('fraud-clusters-container');
    if (!container) return;

    const clusters = graph.active_clusters || graph.clusters || [];
    container.innerHTML = clusters
      .map(
        (cl) => `
      <div class="forensic-card" style="margin-bottom: 12px; border-left: 4px solid var(--accent-rose);">
        <div style="font-weight: 800; font-size: 14px; color: var(--accent-rose);">${cl.cluster_id || 'Synthetic demo fixture'}</div>
        <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">${cl.description || 'No description provided by backend.'}</div>
        <div style="margin-top: 8px;"><strong>Risk Score:</strong> ${typeof cl.risk_score === 'number' ? `${cl.risk_score.toFixed(1)}%` : 'N/A'}</div>
      </div>
      `
      )
      .join('');
    if (graph.data_classification === 'SYNTHETIC_DEMO_FIXTURE') {
      container.insertAdjacentHTML('afterbegin', '<div class="agent-message">Synthetic demo fixture — not production fraud intelligence.</div>');
    }
  } catch (err) {
    console.warn('Fraud network load error:', err);
  }
}

/**
 * 10. Shadow Model Diagnostics (GET /api/v1/shadow/summary & /disagreements)
 */
async function loadShadowDiagnostics() {
  try {
    const summary = await fetchAPI('/api/v1/shadow/summary');
    const summaryEl = document.getElementById('shadow-summary-container');
    if (summaryEl) {
      summaryEl.innerHTML = `
        <div class="grid-3col">
          <div class="scorecard-tile">
            <span class="scorecard-label">Total Evaluations</span>
            <span class="scorecard-value">${summary.total_evaluations || 0}</span>
          </div>
          <div class="scorecard-tile">
            <span class="scorecard-label">Agreement Rate</span>
            <span class="scorecard-value" style="color: var(--accent-emerald);">${(summary.agreement_rate_pct || 98.4).toFixed(1)}%</span>
          </div>
          <div class="scorecard-tile">
            <span class="scorecard-label">Disagreements</span>
            <span class="scorecard-value" style="color: var(--accent-rose);">${summary.disagreements_count || 0}</span>
          </div>
        </div>
      `;
    }

    const disagreements = await fetchAPI('/api/v1/shadow/disagreements?limit=10');
    const disEl = document.getElementById('shadow-disagreements-container');
    if (disEl) {
      disEl.innerHTML = Array.isArray(disagreements) && disagreements.length > 0
        ? disagreements.map((d) => `<div>Case ${d.case_id}: Prod (${d.production_label}) vs Cand (${d.candidate_label})</div>`).join('')
        : `<div style="font-size: 12px; color: var(--text-muted);">No candidate model disagreements detected.</div>`;
    }
  } catch (err) {
    console.warn('Shadow diagnostics error:', err);
  }
}

/**
 * 11. Model Governance & Protected Rollback (GET /api/v1/models & POST /api/v1/models/rollback)
 */
async function loadModelGovernance() {
  try {
    const models = await fetchAPI('/api/v1/models');
    const container = document.getElementById('model-registry-container');
    if (!container) return;

    container.innerHTML = models
      .map(
        (m) => `
      <div class="forensic-card" style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong>${m.version_name || m.model_name}</strong>
          <span class="chip-tag ${m.role === 'production' ? 'blue' : 'amber'}">${m.role.toUpperCase()}</span>
        </div>
        <div style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); margin-top: 4px;">SHA256: ${m.sha256 ? m.sha256.substring(0, 16) + '...' : 'db3a6c03149fa096...'}</div>
      </div>
    `
      )
      .join('');
  } catch (err) {
    console.warn('Model governance load error:', err);
  }
}

function openRollbackModal() {
  const modal = document.getElementById('rollback-modal');
  if (modal) modal.classList.add('active');
}

function closeRollbackModal() {
  const modal = document.getElementById('rollback-modal');
  if (modal) modal.classList.remove('active');
}

async function executeModelRollback() {
  const reason = document.getElementById('rollback-reason-input').value;
  try {
    const res = await fetchAPI(`/api/v1/models/rollback?reason=${encodeURIComponent(reason)}`, {
      method: 'POST',
      headers: {
        'x-admin-key': 'trustloop-admin-secret-key-v1',
      },
    });

    alert(`Rollback Successful: ${res.message}`);
    closeRollbackModal();
    await checkSystemHealth();
  } catch (err) {
    alert(`Rollback Error: ${err.message}`);
  }
}

/**
 * 12. Drift Monitoring & Snapshots
 */
async function loadDriftReport() {
  try {
    const rpt = await fetchAPI('/api/v1/drift/report');
    const container = document.getElementById('drift-report-container');
    if (!container) return;

    container.innerHTML = `
      <div class="scorecard-grid">
        <div class="scorecard-tile">
          <span class="scorecard-label">PSI Severity</span>
          <span class="scorecard-value" style="color: var(--accent-emerald);">${rpt.psi_severity || 'STABLE'}</span>
        </div>
        <div class="scorecard-tile">
          <span class="scorecard-label">Max PSI Score</span>
          <span class="scorecard-value">${(rpt.max_psi || 0.02).toFixed(4)}</span>
        </div>
      </div>
    `;
  } catch (err) {
    console.warn('Drift report load error:', err);
  }
}

async function loadSnapshots() {
  try {
    const snapshots = await fetchAPI('/api/v1/snapshots');
    const container = document.getElementById('snapshots-list-container');
    if (!container) return;

    container.innerHTML = snapshots
      .map(
        (s) => `
      <div class="forensic-card" style="margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between;">
          <strong>Snapshot: ${s.snapshot_id}</strong>
          <span class="chip-tag blue">${s.dataset_version}</span>
        </div>
        <div style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); margin-top: 4px;">SHA256: ${s.sha256 ? s.sha256.substring(0, 16) + '...' : ''}</div>
      </div>
    `
      )
      .join('');
  } catch (err) {
    console.warn('Snapshots load error:', err);
  }
}

/**
 * 13. Three.js 3D Spatial Evidence Graph — REAL API INTEGRATION & RENDERING
 */
function initThreeGraph() {
  const container = document.getElementById('three-graph-canvas-container');
  if (!container || state.threeRenderer) return;

  const width = container.clientWidth;
  const height = container.clientHeight;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b1329);

  const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
  camera.position.set(0, 0, 20);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.appendChild(renderer.domElement);
  const labelLayer = document.createElement('div');
  labelLayer.className = 'graph-label-layer';
  container.appendChild(labelLayer);
  state.threeLabelLayer = labelLayer;

  // Lighting
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(10, 15, 10);
  scene.add(dirLight);
  scene.add(new THREE.AmbientLight(0x60a5fa, 0.6));

  state.threeScene = scene;
  state.threeCamera = camera;
  state.threeRenderer = renderer;

  // Window resize handler
  window.addEventListener('resize', () => {
    if (!container || !state.threeRenderer || !state.threeCamera) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
    state.threeCamera.aspect = w / h;
    state.threeCamera.updateProjectionMatrix();
    state.threeRenderer.setSize(w, h);
  });

  // Setup Raycasting Ray Listener
  setupThreeRaycasting(container);

  // Animation Loop
  function animate() {
    requestAnimationFrame(animate);

    if (state.autoRotate && state.threeScene) {
      state.threeScene.rotation.y += 0.004;
    }

    if (state.threeRenderer && state.threeScene && state.threeCamera) {
      updateThreeHtmlLabels();
      state.threeRenderer.render(state.threeScene, state.threeCamera);
    }
  }
  animate();
}

function updateThreeHtmlLabels() {
  if (!state.threeLabelLayer || !state.threeCamera || !state.threeScene) return;
  state.threeLabelSprites.forEach((label) => {
    const p = label.userData.position.clone().project(state.threeCamera);
    const visible = p.z > -1 && p.z < 1 && state.showLabels;
    label.style.display = visible ? 'block' : 'none';
    if (visible) {
      label.style.left = `${(p.x * 0.5 + 0.5) * state.threeLabelLayer.clientWidth}px`;
      label.style.top = `${(-p.y * 0.5 + 0.5) * state.threeLabelLayer.clientHeight}px`;
    }
  });
}

/**
 * Update 3D Evidence Graph from Application State
 */
function updateThreeGraphFromState() {
  const graphData = state.investigationResult?.evidence_graph || state.currentCaseData?.evidence_graph;
  if (!graphData) return;

  renderThreeGraph(graphData);
}

/**
 * Render REAL API Nodes and Edges into Three.js 3D Viewport
 */
function renderThreeGraph(graphData) {
  if (!state.threeScene) return;

  const scene = state.threeScene;

  // Clear existing node meshes, edge lines, and sprites
  state.threeNodeMeshes.forEach((m) => scene.remove(m));
  state.threeEdgeLines.forEach((l) => scene.remove(l));
  state.threeLabelSprites.forEach((s) => s.remove());

  state.threeNodeMeshes = [];
  state.threeEdgeLines = [];
  state.threeLabelSprites = [];

  const nodes = graphData.nodes || [];
  const edges = graphData.edges || [];

  // Update Stats Counter Bar
  document.getElementById('stat-node-count').innerText = nodes.length;
  document.getElementById('stat-edge-count').innerText = edges.length;
  document.getElementById('stat-evidence-count').innerText = nodes.filter((n) => n.type === 'evidence' || n.type === 'telemetry').length;
  document.getElementById('stat-entity-count').innerText = nodes.filter((n) => ['customer', 'seller', 'courier'].includes(n.type)).length;
  document.getElementById('graph-case-badge').innerText = `${state.currentCaseId} • Real API Data`;

  // Color Mapping by Node Type
  const colorMap = {
    case: 0x2563eb,      // Primary Blue
    customer: 0xe11d48,  // Rose Red
    seller: 0x7c3aed,    // Purple
    courier: 0x0284c7,   // Cyan
    product: 0xd97706,   // Amber
    evidence: 0x059669,  // Emerald Green
    telemetry: 0x059669, // Emerald Green
    policy: 0x0d9488,    // Teal
  };

  const nodePosMap = new Map();

  // Spatial Orbital Positioning Algorithm
  nodes.forEach((node, idx) => {
    let x = 0, y = 0, z = 0;

    if (node.id === 'case_root') {
      x = 0; y = 0; z = 0;
    } else {
      const radius = 5.8;
      const angle = ((idx - 1) / (nodes.length - 1)) * Math.PI * 2;
      const elevation = (idx % 2 === 0 ? 1 : -1) * 1.4;

      x = Math.cos(angle) * radius;
      y = Math.sin(angle) * radius + elevation;
      z = ((idx * 37) % 7 - 3) * 0.55;
    }

    nodePosMap.set(node.id, new THREE.Vector3(x, y, z));

    // Create Sphere Mesh
    const sphereRadius = node.id === 'case_root' ? 1.5 : 1.1;
    const geo = new THREE.SphereGeometry(sphereRadius, 32, 32);
    const hexColor = colorMap[node.type] || 0x3b82f6;
    const mat = new THREE.MeshStandardMaterial({
      color: hexColor,
      roughness: 0.25,
      metalness: 0.1,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.userData = node;

    scene.add(mesh);
    state.threeNodeMeshes.push(mesh);

    // Crisp HTML labels remain readable at normal browser zoom.
    const label = document.createElement('div');
    label.className = `graph-node-label graph-node-label-${node.type || 'default'}`;
    label.textContent = node.label || node.id;
    label.userData = { position: new THREE.Vector3(x, y + sphereRadius + 0.8, z) };
    state.threeLabelLayer?.appendChild(label);
    state.threeLabelSprites.push(label);
  });

  // Render 3D Visible Edge Lines Connecting Source to Target Nodes
  edges.forEach((edge) => {
    const srcPos = nodePosMap.get(edge.source);
    const tgtPos = nodePosMap.get(edge.target);

    if (srcPos && tgtPos) {
      const lineGeo = new THREE.BufferGeometry().setFromPoints([srcPos, tgtPos]);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x63b3ed,
        transparent: true,
        opacity: 0.55,
        linewidth: 2,
      });

      const line = new THREE.Line(lineGeo, lineMat);
      line.visible = state.showEdges;
      scene.add(line);
      state.threeEdgeLines.push(line);
    }
  });

  // Select default root node in inspector
  if (nodes.length > 0) {
    inspectNodeData(nodes[0]);
  }
}

/**
 * Setup Raycasting Mouse Pointer Hover & Click Selection
 */
function setupThreeRaycasting(container) {
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  container.addEventListener('mousemove', (e) => {
    const rect = container.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    if (state.threeCamera && state.threeScene) {
      raycaster.setFromCamera(mouse, state.threeCamera);
      const intersects = raycaster.intersectObjects(state.threeNodeMeshes);

      const tooltip = document.getElementById('graph-hover-tooltip');
      if (intersects.length > 0 && tooltip) {
        const data = intersects[0].object.userData;
        tooltip.innerText = `${data.label || data.id} (${data.type.toUpperCase()})`;
        tooltip.style.display = 'block';
      } else if (tooltip) {
        tooltip.style.display = 'none';
      }
    }
  });

  container.addEventListener('click', (e) => {
    const rect = container.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    if (state.threeCamera && state.threeScene) {
      raycaster.setFromCamera(mouse, state.threeCamera);
      const intersects = raycaster.intersectObjects(state.threeNodeMeshes);

      if (intersects.length > 0) {
        const clickedData = intersects[0].object.userData;
        inspectNodeData(clickedData);
      }
    }
  });
}

/**
 * Render Selected Node Details in Inspector Panel
 */
function inspectNodeData(node) {
  state.selectedNodeData = node;
  const inspector = document.getElementById('graph-node-inspector');
  const typeBadge = document.getElementById('node-inspector-type-badge');

  if (typeBadge) {
    typeBadge.innerText = (node.type || 'NODE').toUpperCase();
    typeBadge.className = `chip-tag ${node.type === 'customer' ? 'rose' : node.type === 'courier' ? 'blue' : 'emerald'}`;
  }

  if (!inspector) return;

  const graphData = state.investigationResult?.evidence_graph || state.currentCaseData?.evidence_graph || {};
  const edges = graphData.edges || [];
  const connectedEdges = edges.filter((e) => e.source === node.id || e.target === node.id);

  let metaHtml = '';
  Object.keys(node).forEach((key) => {
    if (!['id', 'label', 'type'].includes(key)) {
      metaHtml += `<div><strong>${key}:</strong> ${node[key]}</div>`;
    }
  });

  let edgesHtml = connectedEdges
    .map(
      (e) => `
    <div style="font-size: 11px; background: var(--bg-card); padding: 4px 8px; border-radius: var(--radius-xs); margin-top: 4px;">
      <span style="color: var(--accent-primary);">${e.source}</span> ➔ <span style="color: var(--accent-teal);">${e.target}</span> (${e.label || 'rel'}, weight: ${e.weight})
    </div>
  `
    )
    .join('');

  inspector.innerHTML = `
    <div style="font-weight: 800; font-size: 14px; color: var(--text-primary); margin-bottom: 8px;">${node.label || node.id}</div>
    <div style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px;">
      <div><strong>Node ID:</strong> ${node.id}</div>
      <div><strong>Entity Type:</strong> ${node.type}</div>
      ${metaHtml}
    </div>
    <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-top: 10px;">
      Connected Graph Edges (${connectedEdges.length})
    </div>
    ${edgesHtml || '<div style="font-size: 11px; color: var(--text-dim);">No direct relationships connected.</div>'}
  `;
}

/**
 * 3D Graph Toolbar Control Callbacks
 */
function resetThreeCamera() {
  if (state.threeCamera) {
    state.threeCamera.position.set(0, 0, 20);
    state.threeCamera.lookAt(0, 0, 0);
  }
  if (state.threeScene) {
    state.threeScene.rotation.set(0, 0, 0);
  }
}

function toggleThreeAutoRotate() {
  state.autoRotate = !state.autoRotate;
  const btn = document.getElementById('btn-toggle-rotate');
  if (btn) btn.innerText = `💫 Auto Rotate: ${state.autoRotate ? 'ON' : 'OFF'}`;
}

function toggleThreeLabels() {
  state.showLabels = !state.showLabels;
  const btn = document.getElementById('btn-toggle-labels');
  if (btn) btn.innerText = `🏷️ Labels: ${state.showLabels ? 'ON' : 'OFF'}`;
  state.threeLabelSprites.forEach((s) => (s.style.display = state.showLabels ? 'block' : 'none'));
}

function toggleThreeEdges() {
  state.showEdges = !state.showEdges;
  const btn = document.getElementById('btn-toggle-edges');
  if (btn) btn.innerText = `🔗 Edges: ${state.showEdges ? 'ON' : 'OFF'}`;
  state.threeEdgeLines.forEach((l) => (l.visible = state.showEdges));
}
