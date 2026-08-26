/**
 * TrustLoop Forensic Investigation Workstation - Controller (v1.3.0)
 * Features:
 * - Real Three.js 3D WebGL Spatial Evidence Graph with Raycasting
 * - Controlled 8-Stage Sequential Investigation Progression
 * - Live Counterfactual Challenge Recalculation (POST /api/v1/challenge)
 * - Continuous Learning Loop Persistence (POST /api/v1/feedback)
 * - Fraud Coordinated Ring Linkage (GET /api/v1/network/graph)
 */

import * as THREE from 'three';

// Application State
const state = {
  currentCaseId: 'CASE-001',
  currentCaseData: null,
  currentMode: 'e_commerce',
  demoCases: [],
  activeTab: 'case-room',
  isInvestigating: false,
  networkData: null,
  threeScene: null,
  threeCamera: null,
  threeRenderer: null,
  threeNodes: [],
  selectedNode: null,
};

// API Base URL (Relative to host/port)
const API_BASE = window.location.origin;

async function fetchAPI(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'API request failed');
    }
    return await res.json();
  } catch (err) {
    console.error(`API Error [${endpoint}]:`, err);
    throw err;
  }
}

// DOM Ready
document.addEventListener('DOMContentLoaded', async () => {
  initBackgroundCanvas();
  setupNavigation();
  setupModeToggle();
  await loadDemoCases();
  await loadCase(state.currentCaseId);
  setupInvestigateHeroButton();
  setupChallengeForm();
  setupFeedbackForm();
});

// ============================================================
// 1. SUBTLE BACKGROUND CANVAS DEPTH
// ============================================================
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

  const particles = Array.from({ length: 40 }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    vx: (Math.random() - 0.5) * 0.35,
    vy: (Math.random() - 0.5) * 0.35,
    r: Math.random() * 1.5 + 0.5,
  }));

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(99, 179, 237, 0.2)';

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
        if (dist < 110) {
          ctx.strokeStyle = `rgba(99, 179, 237, ${0.08 * (1 - dist / 110)})`;
          ctx.lineWidth = 0.6;
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

// ============================================================
// 2. NAVIGATION & TABS
// ============================================================
function setupNavigation() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const targetView = tab.dataset.tab;
      state.activeTab = targetView;

      document.querySelectorAll('.tab-view').forEach(view => {
        view.classList.remove('active');
      });
      const activeEl = document.getElementById(`view-${targetView}`);
      if (activeEl) activeEl.classList.add('active');

      if (targetView === 'evidence-graph') {
        initThree3DGraph();
      } else if (targetView === 'learning-loop') {
        loadFeedbackHistory();
      } else if (targetView === 'fraud-network') {
        loadFraudNetwork();
      }
    });
  });
}

// ============================================================
// 3. PLATFORM MODE TOGGLE (E-COMMERCE VS Q-COMMERCE)
// ============================================================
function setupModeToggle() {
  const ecomBtn = document.getElementById('mode-ecom');
  const qcomBtn = document.getElementById('mode-qcom');

  if (ecomBtn && qcomBtn) {
    ecomBtn.addEventListener('click', () => {
      ecomBtn.classList.add('active');
      qcomBtn.classList.remove('active');
      state.currentMode = 'e_commerce';
      document.getElementById('platform-mode-tag').textContent = 'E-COMMERCE';
      document.getElementById('demo-case-select').value = 'CASE-001';
      loadCase('CASE-001');
    });

    qcomBtn.addEventListener('click', () => {
      qcomBtn.classList.add('active');
      ecomBtn.classList.remove('active');
      state.currentMode = 'q_commerce';
      document.getElementById('platform-mode-tag').textContent = 'Q-COMMERCE (10-MIN SLA)';
      document.getElementById('demo-case-select').value = 'CASE-006';
      loadCase('CASE-006');
    });
  }
}

// ============================================================
// 4. LOAD DEMO CASES & CASE DETAIL
// ============================================================
async function loadDemoCases() {
  try {
    const cases = await fetchAPI('/api/v1/cases/demo');
    state.demoCases = cases;
    const select = document.getElementById('demo-case-select');
    if (select) {
      select.innerHTML = cases.map(c => 
        `<option value="${c.case_id}">[${c.case_id}] ${c.title}</option>`
      ).join('');

      select.addEventListener('change', (e) => {
        state.currentCaseId = e.target.value;
        loadCase(state.currentCaseId);
      });
    }
  } catch (e) {
    console.warn('Could not load demo cases from backend, using active baseline');
  }
}

async function loadCase(caseId) {
  try {
    const data = await fetchAPI(`/api/v1/cases/${caseId}`);
    state.currentCaseData = data;
    renderCaseProfile(data);
    renderResponsibilityHero(data);
    renderTimeline(data.timeline || []);
    renderDrivers(data.drivers || {});
    renderChallengeForm(data);
    if (state.activeTab === 'evidence-graph') {
      initThree3DGraph();
    }
  } catch (err) {
    console.error('Failed to load case:', err);
  }
}

// ============================================================
// 5. RENDER CASE PROFILE & LEFT COLUMN
// ============================================================
function renderCaseProfile(data) {
  const caseObj = data.case || {};
  const payload = caseObj.payload || {};
  const cust = caseObj.customer || {};
  const seller = caseObj.seller || {};
  const courier = caseObj.courier || {};

  // Header & Dossier Tags
  document.getElementById('header-case-id').textContent = `#${caseObj.case_id || state.currentCaseId}`;
  document.getElementById('profile-case-id').textContent = caseObj.case_id || state.currentCaseId;

  // Product & Claim
  document.getElementById('profile-product-name').textContent = caseObj.product_name || 'Item';
  document.getElementById('profile-category-name').textContent = caseObj.category || 'General';
  document.getElementById('profile-product-val').textContent = `$${(caseObj.product_value_usd || payload.order_value || 399.99).toFixed(2)}`;
  document.getElementById('profile-refund-val').textContent = `$${(caseObj.refund_amount_requested_usd || payload.refund_amount_requested_usd || 399.99).toFixed(2)}`;

  const claimText = payload.return_reason || caseObj.claim_type || 'Damaged in transit';
  document.getElementById('profile-claim-quote').textContent = `"${claimText}"`;

  // Customer
  document.getElementById('profile-cust-id').textContent = cust.customer_id || 'CUST-88219';
  document.getElementById('profile-cust-name').textContent = cust.name || 'Priya Sharma';
  document.getElementById('profile-cust-tier').textContent = cust.tier || 'Standard Customer';
  document.getElementById('profile-cust-rate').textContent = `${(cust.return_rate_pct || payload.return_rate_pct || 4.2).toFixed(1)}%`;
  document.getElementById('profile-cust-disputes').textContent = cust.previous_disputes || payload.previous_dispute_count || '0';

  // Seller
  document.getElementById('profile-seller-id').textContent = seller.seller_id || 'SELL-1049';
  document.getElementById('profile-seller-name').textContent = seller.name || 'Apex Electronics Ltd';
  document.getElementById('profile-seller-defect').textContent = `${(seller.defect_rate_pct || payload.seller_defect_rate || 1.2).toFixed(1)}%`;
  document.getElementById('profile-seller-cctv').textContent = seller.packaging_audit_passed ? 'Passed (Verified)' : 'Audit Flagged';
  document.getElementById('profile-seller-cctv').style.color = seller.packaging_audit_passed ? 'var(--accent-emerald)' : 'var(--accent-rose)';

  // Courier
  document.getElementById('profile-courier-name').textContent = courier.name || 'SwiftExpress Logistics';
  document.getElementById('profile-courier-hub').textContent = courier.hub_location || 'Central Distribution Hub 4B';
  document.getElementById('profile-courier-incident').textContent = `${(courier.incident_rate_pct || payload.courier_incident_rate || 8.4).toFixed(1)}%`;
  document.getElementById('profile-courier-conveyor').textContent = (courier.incident_rate_pct > 5.0) ? 'Compression Jam #4B' : 'Nominal Transit';
  document.getElementById('profile-courier-conveyor').style.color = (courier.incident_rate_pct > 5.0) ? 'var(--accent-rose)' : 'var(--accent-emerald)';

  // Status & Risk
  const statusEl = document.getElementById('profile-case-status');
  if (data.dominant_party === 'courier' || data.dominant_party === 'seller') {
    statusEl.className = 'case-status-indicator resolved';
    statusEl.textContent = 'RESOLVED • ATTRIBUTED';
  } else if (data.dominant_party === 'customer' && data.responsibility?.customer >= 60) {
    statusEl.className = 'case-status-indicator escalated';
    statusEl.textContent = 'REJECTED • ABUSE FLAGGED';
  } else {
    statusEl.className = 'case-status-indicator resolved';
    statusEl.textContent = 'UNDER ACTIVE REVIEW';
  }

  document.getElementById('meter-fraud-risk').textContent = `${(data.fraud_risk_score || 14.2).toFixed(1)}%`;
  document.getElementById('meter-linkage-risk').textContent = payload.has_multiple_accounts ? '85.0%' : '0.0%';
}

// ============================================================
// 6. RENDER RESPONSIBILITY ATTRIBUTION HERO
// ============================================================
function renderResponsibilityHero(data) {
  const resp = data.responsibility || { customer: 9, seller: 9, courier: 73, unknown: 9 };
  const domParty = data.dominant_party || 'courier';

  // Dominant Party Hero
  document.getElementById('dom-party-name').textContent = `${domParty.toUpperCase()} RESPONSIBLE`;
  document.getElementById('dom-party-pct').textContent = `${resp[domParty] || 0}%`;

  const domPartyEl = document.getElementById('dom-party-name');
  const domPctEl = document.getElementById('dom-party-pct');
  if (domParty === 'courier') {
    domPartyEl.style.color = 'var(--party-courier)';
    domPctEl.style.color = 'var(--party-courier)';
  } else if (domParty === 'seller') {
    domPartyEl.style.color = 'var(--party-seller)';
    domPctEl.style.color = 'var(--party-seller)';
  } else if (domParty === 'customer') {
    domPartyEl.style.color = 'var(--party-customer)';
    domPctEl.style.color = 'var(--party-customer)';
  } else {
    domPartyEl.style.color = 'var(--party-unknown)';
    domPctEl.style.color = 'var(--party-unknown)';
  }

  // Action Hero Banner
  const actionBanner = document.getElementById('action-banner-hero');
  const actionTitle = document.getElementById('action-title-display');
  const actionExpl = document.getElementById('action-expl-display');

  const action = data.recommended_action || (domParty === 'courier' ? 'REFUND_AND_COURIER_INVESTIGATION' : (domParty === 'seller' ? 'REFUND_AND_SELLER_INVESTIGATION' : (resp.customer >= 60 ? 'AUTO_REJECT' : 'AUTO_APPROVE')));
  const actionLabel = data.action_label || (action === 'REFUND_AND_COURIER_INVESTIGATION' ? 'Refund Customer & Open Carrier Liability Recovery Claim' : (action === 'AUTO_REJECT' ? 'Reject Return Claim (Coordinated Abuse Flagged)' : 'Auto-Approve Customer Refund'));

  actionBanner.className = `action-hero-banner ${action}`;
  actionTitle.textContent = action;
  actionExpl.textContent = actionLabel;

  // 4-Way Progress Bar & Cards
  document.getElementById('resp-bar-cust').style.width = `${resp.customer}%`;
  document.getElementById('resp-bar-sell').style.width = `${resp.seller}%`;
  document.getElementById('resp-bar-cour').style.width = `${resp.courier}%`;
  document.getElementById('resp-bar-unk').style.width = `${resp.unknown}%`;

  document.getElementById('quad-val-cust').textContent = `${resp.customer}%`;
  document.getElementById('quad-val-sell').textContent = `${resp.seller}%`;
  document.getElementById('quad-val-cour').textContent = `${resp.courier}%`;
  document.getElementById('quad-val-unk').textContent = `${resp.unknown}%`;

  ['customer', 'seller', 'courier', 'unknown'].forEach(p => {
    const card = document.getElementById(`quad-card-${p}`);
    if (card) {
      if (p === domParty) card.classList.add('dominant');
      else card.classList.remove('dominant');
    }
  });

  document.getElementById('hero-confidence-val').textContent = `${(data.confidence || 92.4).toFixed(1)}%`;
  document.getElementById('policy-sec-citation').textContent = 'Section 4.2 Courier Damage Policy';
}

// ============================================================
// 7. RENDER DRIVERS & TIMELINE
// ============================================================
function renderDrivers(drivers) {
  const container = document.getElementById('drivers-list-container');
  container.innerHTML = '';
  let count = 0;

  Object.keys(drivers).forEach(party => {
    (drivers[party] || []).forEach(text => {
      count++;
      const div = document.createElement('div');
      div.className = `driver-pill-row ${party}`;
      div.innerHTML = `
        <span class="driver-tag-lbl">[${party.toUpperCase()}]</span>
        <span>${text}</span>
      `;
      container.appendChild(div);
    });
  });

  if (count === 0) {
    container.innerHTML = '<div class="driver-pill-row">Standard verified return baseline within normal operational SLA.</div>';
  }
}

function renderTimeline(timeline) {
  const container = document.getElementById('timeline-rows-container');
  container.innerHTML = '';

  timeline.forEach(step => {
    const card = document.createElement('div');
    card.className = `timeline-stream-card ${step.status}`;
    card.innerHTML = `
      <div class="timeline-num-badge">${step.step}</div>
      <div class="timeline-card-content">
        <div class="timeline-header-line">
          <span class="timeline-stage-lbl">${step.stage}</span>
          <span class="timeline-time-lbl">${step.timestamp}</span>
        </div>
        <div class="timeline-actor-chip">${step.actor}</div>
        <div class="timeline-detail-desc">${step.detail}</div>
      </div>
    `;
    container.appendChild(card);
  });
}

// ============================================================
// 8. THE CONTROLLED INVESTIGATION EXPERIENCE (PHASE 5)
// ============================================================
function setupInvestigateHeroButton() {
  const btn = document.getElementById('btn-investigate-hero');
  const strip = document.getElementById('investigation-progress-strip');
  if (!btn || !strip) return;

  btn.addEventListener('click', async () => {
    if (state.isInvestigating) return;
    state.isInvestigating = true;
    btn.disabled = true;
    btn.innerHTML = '<span>⚡ Investigating...</span>';
    strip.classList.add('active');

    const stages = [
      { id: 'prog-1', name: 'INTAKE' },
      { id: 'prog-2', name: 'CONTEXT' },
      { id: 'prog-3', name: 'EVIDENCE' },
      { id: 'prog-4', name: 'POLICY' },
      { id: 'prog-5', name: 'RISK' },
      { id: 'prog-6', name: 'RESPONSIBILITY' },
      { id: 'prog-7', name: 'DECISION' },
      { id: 'prog-8', name: 'AUDIT' },
    ];

    stages.forEach(s => {
      const el = document.getElementById(s.id);
      if (el) {
        el.className = 'step-node';
        el.querySelector('.step-circle').textContent = s.id.split('-')[1];
      }
    });

    const payload = state.currentCaseData?.case?.payload || {};
    const apiPromise = fetchAPI('/api/v1/investigate', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    for (let i = 0; i < stages.length; i++) {
      const el = document.getElementById(stages[i].id);
      if (el) {
        el.classList.add('active');
        await new Promise(r => setTimeout(r, 220));
        el.classList.remove('active');
        el.classList.add('completed');
        el.querySelector('.step-circle').textContent = '✓';
      }
    }

    try {
      const res = await apiPromise;
      state.currentCaseData = {
        ...state.currentCaseData,
        ...res,
      };
      renderResponsibilityHero(res);
      renderDrivers(res.drivers || {});
      renderTimeline(res.timeline || []);
    } catch (e) {
      console.error('Investigation error:', e);
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<span>⚡ Re-Investigate Case</span>';
      state.isInvestigating = false;
    }
  });
}

// ============================================================
// 9. THE WOW INTERACTION: CHALLENGE DECISION (PHASE 6)
// ============================================================
function renderChallengeForm(data) {
  const caseObj = data.case || {};
  const payload = caseObj.payload || {};

  const signals = [
    { id: 'courier_incident_history', name: 'Carrier Hub Damage History', desc: 'Sortation hub conveyor damage and route incident rate >8.0%', checked: (payload.courier_incident_rate || 0) > 5.0 },
    { id: 'packaging_damage', name: 'Transit Box Crush / Puncture', desc: 'Outer corrugated box impact documented at doorstep delivery', checked: !!payload.package_damage_reported },
    { id: 'sku_mismatch', name: 'Merchant SKU Barcode Discrepancy', desc: 'Warehouse scan flagged barcode mismatch during packing', checked: !!payload.incorrect_sku_dispatched },
    { id: 'customer_history', name: 'Customer Return Frequency & Dispute Record', desc: 'Lifetime return rate >35% and prior dispute history', checked: (payload.return_rate_pct || 0) > 35.0 || (payload.previous_dispute_count || 0) > 0 },
    { id: 'identity_linkage', name: 'Identity Linkage / Multi-Account Flag', desc: 'Shared device canvas fingerprint or payment token collision', checked: !!payload.has_multiple_accounts },
    { id: 'transit_delay', name: 'Excessive Transit Delay Window', desc: 'Carrier logistics transit exceeded guaranteed 36-hour SLA', checked: (payload.transit_delay_hours || 0) > 24.0 },
  ];

  const container = document.getElementById('challenge-signals-container');
  if (container) {
    container.innerHTML = signals.map(s => `
      <label class="signal-checkbox-card">
        <input type="checkbox" name="challenge_signal" value="${s.id}" ${s.checked ? 'checked' : ''}>
        <div>
          <div class="signal-text-title">${s.name}</div>
          <div class="signal-text-desc">${s.desc}</div>
        </div>
      </label>
    `).join('');
  }
}

function setupChallengeForm() {
  const btn = document.getElementById('btn-recalc-challenge');
  if (!btn) return;

  btn.addEventListener('click', async () => {
    btn.textContent = 'Recalculating Evidence Weights...';
    btn.disabled = true;

    try {
      const checkboxes = document.querySelectorAll('input[name="challenge_signal"]');
      const disabledSignals = [];
      checkboxes.forEach(cb => {
        if (!cb.checked) disabledSignals.push(cb.value);
      });

      const payload = state.currentCaseData?.case?.payload || {};
      const res = await fetchAPI('/api/v1/challenge', {
        method: 'POST',
        body: JSON.stringify({
          case_payload: payload,
          disabled_signals: disabledSignals,
        }),
      });

      renderChallengeResults(res);
    } catch (err) {
      alert(`Challenge recalculation failed: ${err.message}`);
    } finally {
      btn.textContent = 'Recalculate Counterfactual Decision';
      btn.disabled = false;
    }
  });
}

function renderChallengeResults(res) {
  const container = document.getElementById('challenge-results-container');
  if (!container) return;

  const deltas = res.deltas || {};
  const baseAction = res.baseline.recommended_action;
  const cfAction = res.counterfactual.recommended_action;

  container.innerHTML = `
    <div class="delta-results-display">
      <div class="card-title-text" style="margin-bottom: 12px;">Counterfactual Responsibility Shifts</div>
      
      ${['customer', 'seller', 'courier', 'unknown'].map(p => {
        const d = deltas[p] || { before: 0, after: 0, delta: 0 };
        const badgeClass = d.delta < 0 ? 'negative' : (d.delta > 0 ? 'positive' : 'neutral');
        const sign = d.delta > 0 ? '+' : '';
        return `
          <div class="delta-stat-line">
            <span style="text-transform: capitalize; font-weight: 700;">${p} Responsibility</span>
            <div>
              <span style="color: var(--text-muted); font-family: var(--font-mono);">${d.before}% &rarr; </span>
              <strong style="color: var(--text-primary); font-family: var(--font-mono);">${d.after}%</strong>
              <span class="delta-badge-chip ${badgeClass}" style="margin-left: 8px;">${sign}${d.delta}%</span>
            </div>
          </div>
        `;
      }).join('')}

      <div class="delta-stat-line" style="margin-top: 6px;">
        <span style="font-weight: 700;">Recommended Platform Action</span>
        <div>
          <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">${baseAction}</span>
          <span> &rarr; </span>
          <span style="font-family: var(--font-mono); font-size: 12px; font-weight: 800; color: var(--accent-cyan);">${cfAction}</span>
        </div>
      </div>

      <div class="causal-mechanism-callout">
        <strong>Causal Mechanism:</strong> ${res.explanation}
      </div>
    </div>
  `;
}

// ============================================================
// 10. REAL THREE.JS 3D SPATIAL EVIDENCE GRAPH (PHASE 7 & 10)
// ============================================================
function initThree3DGraph() {
  const container = document.getElementById('three-graph-container');
  if (!container || !state.currentCaseData) return;

  container.innerHTML = '';
  const width = container.clientWidth || 850;
  const height = container.clientHeight || 520;

  // 1. Scene, Camera, Renderer
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050810);
  state.threeScene = scene;

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
  camera.position.set(0, 0, 180);
  state.threeCamera = camera;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);
  state.threeRenderer = renderer;

  // 2. Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambientLight);

  const pointLight = new THREE.PointLight(0x63b3ed, 1.5, 300);
  pointLight.position.set(50, 50, 80);
  scene.add(pointLight);

  // 3. Nodes Data from active case
  const graphData = state.currentCaseData.evidence_graph || { nodes: [], edges: [] };
  const rawNodes = graphData.nodes || [];
  const rawEdges = graphData.edges || [];

  const nodeMeshes = [];
  const nodeMap = {};

  // Center Root Node
  const rootGeom = new THREE.SphereGeometry(7, 32, 32);
  const rootMat = new THREE.MeshStandardMaterial({
    color: 0x63b3ed,
    emissive: 0x63b3ed,
    emissiveIntensity: 0.4,
    roughness: 0.3,
    metalness: 0.7,
  });
  const rootMesh = new THREE.Mesh(rootGeom, rootMat);
  rootMesh.position.set(0, 0, 0);
  rootMesh.userData = { id: 'case_root', label: 'CASE DOSSIER', type: 'root', data: state.currentCaseId };
  scene.add(rootMesh);
  nodeMeshes.push(rootMesh);
  nodeMap['case_root'] = rootMesh;

  // Satellite Nodes in 3D Orbital Coordinates
  const outerNodes = rawNodes.filter(n => n.id !== 'case_root');
  const count = outerNodes.length || 1;
  const radius = 65;

  outerNodes.forEach((n, i) => {
    const phi = Math.acos(-1 + (2 * i) / count);
    const theta = Math.sqrt(count * Math.PI) * phi;

    const x = radius * Math.cos(theta) * Math.sin(phi);
    const y = radius * Math.sin(theta) * Math.sin(phi) * 0.7;
    const z = radius * Math.cos(phi) * 0.7;

    const colorHex = (n.type === 'customer') ? 0xfc8181 : (n.type === 'seller' ? 0xb794f4 : (n.type === 'courier' ? 0x63b3ed : 0x48bb78));
    const geom = new THREE.SphereGeometry(5, 24, 24);
    const mat = new THREE.MeshStandardMaterial({
      color: colorHex,
      emissive: colorHex,
      emissiveIntensity: 0.35,
      roughness: 0.3,
      metalness: 0.7,
    });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.position.set(x, y, z);
    mesh.userData = { id: n.id, label: n.label, type: n.type, data: n.score || n.value || n.relevance || 'Verified' };
    scene.add(mesh);
    nodeMeshes.push(mesh);
    nodeMap[n.id] = mesh;

    // Connect edge line from root to satellite
    const lineMat = new THREE.LineBasicMaterial({ color: 0x182236, transparent: true, opacity: 0.6 });
    const points = [rootMesh.position, mesh.position];
    const lineGeom = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(lineGeom, lineMat);
    scene.add(line);
  });

  state.threeNodes = nodeMeshes;

  // 4. Mouse Raycasting & Interaction
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  let isDragging = false;
  let prevMousePos = { x: 0, y: 0 };

  const canvasEl = renderer.domElement;

  canvasEl.addEventListener('mousedown', (e) => {
    isDragging = true;
    prevMousePos = { x: e.clientX, y: e.clientY };
  });

  window.addEventListener('mouseup', () => { isDragging = false; });

  canvasEl.addEventListener('mousemove', (e) => {
    const rect = canvasEl.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    if (isDragging) {
      const deltaX = e.clientX - prevMousePos.x;
      const deltaY = e.clientY - prevMousePos.y;
      scene.rotation.y += deltaX * 0.008;
      scene.rotation.x += deltaY * 0.008;
      prevMousePos = { x: e.clientX, y: e.clientY };
    }
  });

  canvasEl.addEventListener('wheel', (e) => {
    camera.position.z = Math.max(80, Math.min(300, camera.position.z + e.deltaY * 0.15));
  });

  canvasEl.addEventListener('click', () => {
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(nodeMeshes);
    if (intersects.length > 0) {
      const clickedMesh = intersects[0].object;
      inspect3DNode(clickedMesh.userData);
    }
  });

  // Default Inspector for Case Root
  inspect3DNode(rootMesh.userData);

  // 5. Animation Loop
  let frameId;
  function animate() {
    frameId = requestAnimationFrame(animate);
    if (!isDragging) {
      scene.rotation.y += 0.0015;
    }
    renderer.render(scene, camera);
  }
  animate();
}

function inspect3DNode(data) {
  const inspector = document.getElementById('graph-node-inspector');
  if (!inspector) return;

  inspector.innerHTML = `
    <div class="graph-telemetry-drawer">
      <div class="card-title-text" style="margin-bottom: 8px;">Spatial Entity Telemetry Inspector</div>
      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 12px;">
        <div><span style="color: var(--text-muted); font-size: 10.5px; text-transform: uppercase;">Entity ID</span><br><strong style="font-family: var(--font-mono); color: var(--text-primary);">${data.id}</strong></div>
        <div><span style="color: var(--text-muted); font-size: 10.5px; text-transform: uppercase;">Entity Type</span><br><strong style="color: var(--accent-cyan); text-transform: uppercase;">${data.type}</strong></div>
        <div><span style="color: var(--text-muted); font-size: 10.5px; text-transform: uppercase;">Display Label</span><br><strong style="color: var(--text-primary);">${data.label}</strong></div>
        <div><span style="color: var(--text-muted); font-size: 10.5px; text-transform: uppercase;">Weight / Telemetry</span><br><strong style="color: var(--accent-blue); font-family: var(--font-mono);">${data.data}</strong></div>
      </div>
    </div>
  `;
}

// ============================================================
// 11. FRAUD NETWORK & LINKAGE (PHASE 8 & 9)
// ============================================================
async function loadFraudNetwork() {
  const container = document.getElementById('network-clusters-container');
  if (!container) return;

  try {
    const data = await fetchAPI('/api/v1/network/graph');
    state.networkData = data;
    const clusters = data.active_clusters || [];

    container.innerHTML = clusters.map(c => `
      <div class="forensic-card" style="border-left: 4px solid var(--accent-rose); margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
          <strong style="font-size: 13.5px; color: var(--text-primary);">${c.cluster_name} (${c.cluster_id})</strong>
          <span class="delta-badge-chip negative">${c.severity} RISK</span>
        </div>
        <div style="font-size: 11.5px; color: var(--text-secondary); margin-bottom: 6px;">
          <strong>Coordinated Link:</strong> ${c.primary_link} | <strong>Disputed Value:</strong> $${(c.total_disputed_value_usd || 0).toFixed(2)}
        </div>
        <div style="font-size: 12px; color: var(--text-primary); margin-bottom: 10px; line-height: 1.4;">${c.summary}</div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
          ${c.connected_customers.map(cust => `<span class="chip-tag cyan" style="font-size: 11px;">${cust}</span>`).join('')}
        </div>
      </div>
    `).join('');
  } catch (e) {
    console.warn('Could not load fraud network graph');
  }
}

// ============================================================
// 12. LEARNING LOOP & AUDITOR REVIEW (PHASE 9 & 10)
// ============================================================
function setupFeedbackForm() {
  const form = document.getElementById('feedback-submission-form');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const reviewerId = document.getElementById('fb-reviewer-id').value;
    const verifiedLabel = document.getElementById('fb-verified-label').value;
    const humanDecision = document.getElementById('fb-human-decision').value;
    const notes = document.getElementById('fb-notes').value;

    const payload = {
      case_id: state.currentCaseId,
      reviewer_id: reviewerId,
      human_verified_label: verifiedLabel,
      human_decision: humanDecision,
      notes: notes,
      traffic_type: 'production',
      raw_payload: state.currentCaseData?.case?.payload || {},
    };

    try {
      await fetchAPI('/api/v1/feedback', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
      alert('Human auditor review successfully persisted to verified feedback store!');
      form.reset();
      loadFeedbackHistory();
    } catch (err) {
      alert(`Feedback submission failed: ${err.message}`);
    }
  });
}

async function loadFeedbackHistory() {
  const container = document.getElementById('feedback-history-table-body');
  if (!container) return;

  try {
    const list = await fetchAPI('/api/v1/feedback/history?limit=15');
    if (!list || list.length === 0) {
      container.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--text-muted); padding: 20px;">No feedback records logged yet.</td></tr>';
      return;
    }

    container.innerHTML = list.map(r => `
      <tr>
        <td><strong style="font-family: var(--font-mono); color: var(--text-primary);">${r.case_id}</strong></td>
        <td>${r.reviewer_id}</td>
        <td><span style="color: var(--accent-cyan); font-weight: 700;">${r.human_verified_label}</span></td>
        <td><span class="delta-badge-chip positive">${r.quality_status || 'ELIGIBLE'}</span></td>
        <td>${r.notes || '-'}</td>
        <td style="font-family: var(--font-mono); font-size: 10px; color: var(--text-muted);">${(r.timestamp || '').slice(0, 19)}</td>
      </tr>
    `).join('');
  } catch (e) {
    console.warn('Could not load feedback history');
  }
}

// ============================================================
// 13. POLICY MODAL
// ============================================================
window.openPolicyModal = function() {
  const modal = document.getElementById('policy-modal');
  if (modal) modal.classList.add('active');
};

window.closePolicyModal = function() {
  const modal = document.getElementById('policy-modal');
  if (modal) modal.classList.remove('active');
};
