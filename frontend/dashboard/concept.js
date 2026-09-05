/**
 * AI Risk Manager — Fraud Concept Template Controller
 * 
 * Handles URL query routing, sidebar navigation, and dynamic population
 * of the unified reusable fraud protection concept template.
 */

document.addEventListener('DOMContentLoaded', () => {
  if (typeof FRAUD_CONCEPTS === 'undefined') {
    console.error('FRAUD_CONCEPTS dictionary not loaded');
    return;
  }

  const navList = document.getElementById('concept-nav-list');
  const contentPane = document.getElementById('concept-content-pane');

  // Helper to extract ?type= param
  function getRequestedConceptId() {
    const params = new URLSearchParams(window.location.search);
    const requested = params.get('type');
    if (requested && FRAUD_CONCEPTS[requested]) {
      return requested;
    }
    return 'qr_overlay_image'; // Default to first researched type
  }

  let activeConceptId = getRequestedConceptId();

  // Populate sidebar navigation
  function renderSidebar() {
    navList.innerHTML = '';

    Object.values(FRAUD_CONCEPTS).forEach(concept => {
      const li = document.createElement('li');
      li.className = 'concept-nav-item';
      
      const isActive = concept.id === activeConceptId;
      li.innerHTML = `
        <button type="button" data-id="${concept.id}" class="${isActive ? 'active' : ''}" aria-current="${isActive ? 'true' : 'false'}">
          <span>${concept.short_name || concept.title}</span>
          <span class="nav-category">${concept.category}</span>
        </button>
      `;

      li.querySelector('button').addEventListener('click', () => {
        selectConcept(concept.id);
      });

      navList.appendChild(li);
    });
  }

  // Populate main template fields
  function renderConcept(id) {
    const data = FRAUD_CONCEPTS[id];
    if (!data) return;

    activeConceptId = id;
    document.title = `${data.title} — Fraud Protection Concept | AI Risk Manager`;

    // Header & Meta
    document.getElementById('c-category').textContent = data.category;
    document.getElementById('c-title').textContent = data.title;
    document.getElementById('c-metric').textContent = data.target_metric;
    document.getElementById('c-loss-owner').textContent = data.primary_loss_owner;
    document.getElementById('c-latency').textContent = data.detector_function.target_latency;
    document.getElementById('c-timeline').textContent = data.estimated_timeline;

    // 1. Fraud Pattern
    document.getElementById('c-pattern-summary').textContent = data.fraud_pattern.summary;
    document.getElementById('c-pattern-mechanics').textContent = data.fraud_pattern.mechanics;
    document.getElementById('c-pattern-impact').textContent = data.fraud_pattern.financial_impact;
    document.getElementById('c-pattern-why-fail').textContent = data.fraud_pattern.why_existing_rules_fail;

    // 2. Detector Function
    document.getElementById('c-detector-summary').textContent = data.detector_function.summary;
    
    const stepsContainer = document.getElementById('c-pipeline-steps');
    stepsContainer.innerHTML = '';
    data.detector_function.pipeline_steps.forEach(step => {
      const stepEl = document.createElement('div');
      stepEl.className = 'pipeline-step';
      stepEl.innerHTML = `
        <div class="step-badge">${step.step}</div>
        <div class="step-content">
          <div class="step-name">${step.name}</div>
          <div class="step-desc">${step.description}</div>
        </div>
      `;
      stepsContainer.appendChild(stepEl);
    });

    document.getElementById('c-decision-policy').textContent = data.detector_function.decision_policy;

    // 3. Data Requirements
    const streamsList = document.getElementById('c-data-streams');
    streamsList.innerHTML = '';
    data.data_requirements.streams.forEach(stream => {
      const li = document.createElement('li');
      li.className = 'data-item';
      li.innerHTML = `<span class="data-item-bullet" aria-hidden="true"></span><span>${stream}</span>`;
      streamsList.appendChild(li);
    });

    const entitiesList = document.getElementById('c-feature-entities');
    entitiesList.innerHTML = '';
    data.data_requirements.feature_store_entities.forEach(ent => {
      const li = document.createElement('li');
      li.className = 'data-item';
      // Format code tags
      const formatted = ent.replace(/^([a-z0-9_]+)(\s*\(.*?\))?$/, '<span class="code-tag">$1</span> $2');
      li.innerHTML = `<span class="data-item-bullet" aria-hidden="true"></span><span>${formatted}</span>`;
      entitiesList.appendChild(li);
    });

    const integrationsList = document.getElementById('c-integrations');
    integrationsList.innerHTML = '';
    data.data_requirements.external_integrations.forEach(integ => {
      const li = document.createElement('li');
      li.className = 'data-item';
      li.innerHTML = `<span class="data-item-bullet" aria-hidden="true"></span><span>${integ}</span>`;
      integrationsList.appendChild(li);
    });

    // 4. Why Not Implemented & Roadmap
    document.getElementById('c-why-unimplemented').textContent = data.why_not_implemented;

    const roadmapList = document.getElementById('c-roadmap-list');
    roadmapList.innerHTML = '';
    data.roadmap_to_production.forEach((item, idx) => {
      const li = document.createElement('li');
      li.className = 'roadmap-item';
      li.innerHTML = `
        <span class="roadmap-bullet" aria-hidden="true">M${idx + 1}</span>
        <span>${item}</span>
      `;
      roadmapList.appendChild(li);
    });

    // Update sidebar buttons active state
    const buttons = navList.querySelectorAll('button');
    buttons.forEach(btn => {
      const isCurrent = btn.dataset.id === id;
      btn.classList.toggle('active', isCurrent);
      btn.setAttribute('aria-current', isCurrent ? 'true' : 'false');
    });
  }

  // Handle user selection with history push
  function selectConcept(id) {
    if (id === activeConceptId) return;

    renderConcept(id);
    const newUrl = `${window.location.pathname}?type=${encodeURIComponent(id)}`;
    window.history.pushState({ type: id }, '', newUrl);

    // Scroll main pane into view smoothly on mobile/narrow screens
    if (window.innerWidth < 960) {
      contentPane.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // Handle browser back/forward buttons
  window.addEventListener('popstate', (e) => {
    const id = (e.state && e.state.type) ? e.state.type : getRequestedConceptId();
    renderConcept(id);
  });

  // Initial load
  renderSidebar();
  renderConcept(activeConceptId);
});
