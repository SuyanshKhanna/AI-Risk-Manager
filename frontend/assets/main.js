/* ============================================================
   UPI Fraud Detector — Main JavaScript
   Vanilla ES6, no dependencies, performant animations
   ============================================================ */

// ============================================================
// Configuration
// ============================================================
const CONFIG = {
  API_BASE: 'http://localhost:8000',
  ENDPOINTS: {
    score: '/score',
    batch: '/batch_score',
    health: '/health',
    ready: '/ready',
    metrics: '/metrics',
    demo: '/demo'
  },
  THRESHOLDS: {
    challenge: 0.3,
    block: 0.7
  },
  ANIMATION: {
    counterDuration: 1500,
    staggerDelay: 80
  }
};

// ============================================================
// Utility Functions
// ============================================================
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const lerp = (a, b, t) => a + (b - a) * t;

const formatNumber = (n) => new Intl.NumberFormat().format(n);
const formatPercent = (n) => `${(n * 100).toFixed(1)}%`;

// Debounce
const debounce = (fn, ms) => {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
};

// Throttle
const throttle = (fn, ms) => {
  let last = 0;
  return (...args) => {
    const now = performance.now();
    if (now - last >= ms) {
      last = now;
      fn(...args);
    }
  };
}

// ============================================================
// Background Canvas Animation
// ============================================================
class BackgroundCanvas {
  constructor() {
    this.canvas = $('#bg-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.particles = [];
    this.mouse = { x: 0, y: 0 };
    this.animationId = null;
    this.init();
  }

  init() {
    this.resize();
    window.addEventListener('resize', debounce(() => this.resize(), 250));
    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
    this.spawnParticles();
    this.animate();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = window.innerWidth * dpr;
    this.canvas.height = window.innerHeight * dpr;
    this.canvas.style.width = `${window.innerWidth}px`;
    this.canvas.style.height = `${window.innerHeight}px`;
    this.ctx.scale(dpr, dpr);
    this.width = window.innerWidth;
    this.height = window.innerHeight;
  }

  spawnParticles() {
    const count = Math.min(250, Math.floor((this.width * this.height) / 8000));
    this.particles = [];
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: Math.random() * 1.5 + 0.5,
        opacity: Math.random() * 0.4 + 0.1,
        hue: 160 + Math.random() * 20
      });
    }
  }

  animate() {
    this.ctx.clearRect(0, 0, this.width, this.height);

    // Draw connections
    this.ctx.strokeStyle = 'rgba(0, 212, 170, 0.08)';
    this.ctx.lineWidth = 0.5;

    let snappedCount = 0;

    for (let i = 0; i < this.particles.length; i++) {
      const p1 = this.particles[i];
      
      // Mouse attraction
      const dx = this.mouse.x - p1.x;
      const dy = this.mouse.y - p1.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      if (dist < 60) {
        snappedCount++;
      }
      
      if (dist < 300) {
        const force = (300 - dist) / 300 * 0.05;
        p1.vx += dx / dist * force;
        p1.vy += dy / dist * force;
      }

      // Update position
      p1.x += p1.vx;
      p1.y += p1.vy;

      // Wrap around
      if (p1.x < 0) p1.x = this.width;
      if (p1.x > this.width) p1.x = 0;
      if (p1.y < 0) p1.y = this.height;
      if (p1.y > this.height) p1.y = 0;

      // Draw particle with glow
      this.ctx.beginPath();
      this.ctx.arc(p1.x, p1.y, p1.radius, 0, Math.PI * 2);
      this.ctx.shadowBlur = 10;
      this.ctx.shadowColor = `hsla(${p1.hue}, 100%, 50%, ${p1.opacity})`;
      this.ctx.fillStyle = `hsla(${p1.hue}, 100%, 50%, ${p1.opacity})`;
      this.ctx.fill();
      this.ctx.shadowBlur = 0; // Reset

      // Connections
      for (let j = i + 1; j < this.particles.length; j++) {
        const p2 = this.particles[j];
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const dist2 = Math.sqrt(dx * dx + dy * dy);
        if (dist2 < 120) {
          this.ctx.beginPath();
          this.ctx.moveTo(p1.x, p1.y);
          this.ctx.lineTo(p2.x, p2.y);
          this.ctx.strokeStyle = `rgba(0, 212, 170, ${0.08 * (1 - dist2 / 120)})`;
          this.ctx.stroke();
        }
      }
    }

    // Dampen velocity
    this.particles.forEach(p => {
      p.vx *= 0.98;
      p.vy *= 0.98;
    });

    // Despawn logic if too many snapped
    if (snappedCount > 80) {
      let despawned = 0;
      for (let i = 0; i < this.particles.length; i++) {
        const p = this.particles[i];
        const dist = Math.sqrt(Math.pow(this.mouse.x - p.x, 2) + Math.pow(this.mouse.y - p.y, 2));
        if (dist < 60) {
          // Respawn elsewhere
          p.x = Math.random() * this.width;
          p.y = Math.random() * this.height;
          p.vx = (Math.random() - 0.5) * 0.3;
          p.vy = (Math.random() - 0.5) * 0.3;
          despawned++;
          if (despawned >= 5) break; // Despawn up to 5 per frame
        }
      }
    }

    this.animationId = requestAnimationFrame(() => this.animate());
  }

  destroy() {
    cancelAnimationFrame(this.animationId);
  }
}

// ============================================================
// Counter Animation
// ============================================================
function animateCounter(el, target, duration = 1500) {
  const start = 0;
  const startTime = performance.now();
  const isPercent = target.toString().includes('%');
  const numTarget = parseFloat(target);

  function update(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
    const current = start + (numTarget - start) * eased;
    
    if (isPercent) {
      el.textContent = current.toFixed(1) + '%';
    } else {
      el.textContent = Math.floor(current).toLocaleString();
    }

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      el.textContent = target;
    }
  }

  requestAnimationFrame(update);
}

// ============================================================
// Intersection Observer for Scroll Animations
// ============================================================
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('counted');
        if (entry.target.dataset.count) {
          animateCounter(entry.target, entry.target.dataset.count, CONFIG.ANIMATION.counterDuration);
        }
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.3, rootMargin: '0px 0px -50px 0px' });

  $$('.metric-card__value[data-count]').forEach(el => observer.observe(el));
}

// ============================================================
// Pipeline Interaction
// ============================================================
function initPipelineInteraction() {
  const stages = $$('.pipeline__stage');
  const arrows = $$('.pipeline__arrow');

  stages.forEach((stage, idx) => {
    stage.addEventListener('mouseenter', () => {
      // Highlight connected arrows
      arrows.forEach((arrow, aIdx) => {
        if (aIdx === idx || aIdx === idx - 1) {
          arrow.style.opacity = '1';
          arrow.querySelector('svg path').style.stroke = 'var(--accent)';
        }
      });
    });

    stage.addEventListener('mouseleave', () => {
      arrows.forEach(arrow => {
        arrow.style.opacity = '';
        arrow.querySelector('svg path').style.stroke = '';
      });
    });
  });
}

// ============================================================
// Form Handling
// ============================================================
function initForm() {
  const form = $('#demo-form');
  const submitBtn = $('#submit-btn');
  const btnText = $('.btn__text', submitBtn);
  const btnLoader = $('.btn__loader', submitBtn);
  const resultEl = $('#result');
  const resultContent = $('.result__content', resultEl);
  const resultPlaceholder = $('.result__placeholder', resultEl);

  // Slider value displays
  $$('.form-group--slider input[type="range"]').forEach(input => {
    const valEl = $(`#${input.id}_val`);
    input.addEventListener('input', () => {
      if (valEl) valEl.textContent = parseFloat(input.value).toFixed(input.step === '0.01' ? 2 : 0);
    });
  });

  // Form submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'block';

    const formData = new FormData(form);
    const payload = {};
    
    for (const [key, value] of formData.entries()) {
      const el = form.elements[key];
      const isNumberField = el && (el.type === 'number' || el.type === 'range');

      if (value === 'on') {
        payload[key] = 1;
      } else if (value === '') {
        payload[key] = 0;
      } else if (isNumberField) {
        payload[key] = Number(value);
      } else {
        payload[key] = value;
      }
    }

    // Add defaults for missing fields
    const defaults = {
      velocity_5min: 0, velocity_1hr: 0, refund_count_1hr: 0,
      refund_rate: 0.0, same_device_refunds: 0, settlement_verified: 1,
      qr_mismatch: 0, remote_access_app: 0, device_emulator_score: 0.0,
      device_root_score: 0.0, device_fraud_reports_30d: 0, vpn_probability: 0.0,
      hour: 12, is_night: 0, new_upi_handle: 0, new_merchant: 0, new_device: 0,
      unusual_hour: 0, session_duration_sec: 0,
      merchant_id_txn_count_5min: 0, merchant_id_txn_count_1hr: 0,
      device_fingerprint_txn_count_5min: 0, device_fingerprint_txn_count_1hr: 0,
      upi_handle_txn_count_5min: 0, upi_handle_txn_count_1hr: 0
    };
    
    const finalPayload = Object.assign({}, defaults, payload);

    try {
      const start = performance.now();
      const response = await fetch(`${CONFIG.API_BASE}${CONFIG.ENDPOINTS.score}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(finalPayload)
      });
      const latency = performance.now() - start;

      if (!response.ok) {
        const err = await response.json();
        let errMsg = `HTTP ${response.status}`;
        if (err.detail) {
          if (Array.isArray(err.detail)) {
            errMsg = err.detail.map(d => {
              const loc = d.loc ? d.loc.slice(1).join('.') + ': ' : '';
              return loc + (d.msg || JSON.stringify(d));
            }).join('; ');
          } else {
            errMsg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
          }
        }
        throw new Error(errMsg);
      }

      const data = await response.json();
      renderResult(data, latency);

    } catch (err) {
      console.error('Score error:', err);
      showError(err.message);
    } finally {
      submitBtn.disabled = false;
      btnText.style.display = 'inline';
      btnLoader.style.display = 'none';
    }
  });

  function renderResult(data, latency) {
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'block';

    // Action badge
    $$('.result__action').forEach(el => el.style.display = 'none');
    const actionEl = $(`.result__action[data-action="${data.action}"]`);
    if (actionEl) actionEl.style.display = 'flex';

    // Score circle
    const scoreEl = $('#score-value');
    const scoreCircle = $('.score-circle');
    const scoreProgress = $('.score-progress');
    
    const scorePercent = Math.round(data.score * 100);
    animateCounter(scoreEl, scorePercent, 800);
    
    scoreCircle.dataset.action = data.action;
    const circumference = 2 * Math.PI * 50;
    const offset = circumference * (1 - data.score);
    scoreProgress.style.strokeDashoffset = offset;

    // Reasons
    const reasonsList = $('#reasons-list');
    reasonsList.innerHTML = '';
    data.reasons.forEach((reason, idx) => {
      const li = document.createElement('li');
      li.style.animationDelay = `${idx * 50}ms`;
      li.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>${reason}`;
      reasonsList.appendChild(li);
    });

    // SHAP bars
    const shapBars = $('#shap-bars');
    shapBars.innerHTML = '';
    
    if (data.shap_top3 && data.shap_top3.length) {
      data.shap_top3.forEach((item, idx) => {
        const bar = document.createElement('div');
        bar.className = `shap-bar ${item.contribution >= 0 ? 'shap-bar--positive' : 'shap-bar--negative'}`;
        bar.style.animationDelay = `${idx * 100}ms`;
        
        const pct = Math.min(Math.abs(item.contribution) / 15, 1) * 100;
        
        bar.innerHTML = `
          <span class="shap-bar__name">${item.feature}</span>
          <div class="shap-bar__track">
            <div class="shap-bar__fill" style="width: 0%"></div>
          </div>
          <span class="shap-bar__value">${item.contribution >= 0 ? '+' : ''}${item.contribution.toFixed(2)}</span>
        `;
        
        shapBars.appendChild(bar);
        
        // Animate fill
        requestAnimationFrame(() => {
          const fill = bar.querySelector('.shap-bar__fill');
          fill.style.width = `${pct}%`;
        });
      });
    }

    // Meta
    $('#latency').textContent = `Latency: ${latency.toFixed(1)}ms`;
    $('#model-version').textContent = `Model: ${data.model_version || '—'}`;
  }

  function showError(msg) {
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'block';
    resultContent.innerHTML = `
      <div class="result__action result__action--block" style="display:flex">
        <span class="result__action-label">ERROR</span>
        <span class="result__action-desc">${msg}</span>
      </div>
    `;
  }
}

// ============================================================
// Smooth Scroll for Anchor Links
// ============================================================
function initSmoothScroll() {
  $$('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const target = $(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, '', anchor.getAttribute('href'));
      }
    });
  });
}

// ============================================================
// Navigation Active State
// ============================================================
function initNavActive() {
  const sections = $$('section[id]');
  const navLinks = $$('.nav__link[href^="#"]');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navLinks.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
        });
      }
    });
  }, { threshold: 0.4, rootMargin: '-20% 0px -60% 0px' });

  sections.forEach(section => observer.observe(section));
}

// ============================================================
// Health Check on Load
// ============================================================
async function checkHealth() {
  try {
    const res = await fetch(`${CONFIG.API_BASE}${CONFIG.ENDPOINTS.health}`);
    if (res.ok) {
      console.log('✅ API healthy');
    }
  } catch (err) {
    console.warn('⚠️ API not reachable:', err.message);
  }
}

// ============================================================
// Initialize Everything
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  // Initialize background
  new BackgroundCanvas();

  // Initialize all modules
  initScrollAnimations();
  initPipelineInteraction();
  initForm();
  initSmoothScroll();
  initNavActive();
  checkHealth();

  // Add active nav styles dynamically
  const style = document.createElement('style');
  style.textContent = `
    .nav__link.active { color: var(--accent); }
    .nav__link.active::after { transform: scaleX(1); }
  `;
  document.head.appendChild(style);

  console.log('🚀 UPI Fraud Detector UI initialized');
});

// ============================================================
// Export for testing
// ============================================================
window.FraudDetectorUI = {
  animateCounter,
  BackgroundCanvas
};