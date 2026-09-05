document.addEventListener('DOMContentLoaded', async () => {
  const loadingState = document.getElementById('loading-state');
  const errorState = document.getElementById('error-state');
  const statsContainer = document.getElementById('stats-container');
  const errorMsg = document.getElementById('error-message');

  try {
    const response = await fetch('/fraud_type_stats');
    
    if (!response.ok) {
      throw new Error(`API Error (${response.status})`);
    }
    
    const data = await response.json();
    
    // Process and sort: implemented ones first, then by count descending
    const types = data.fraud_types || [];
    types.sort((a, b) => {
      if (a.implemented !== b.implemented) {
        return a.implemented ? -1 : 1;
      }
      return b.count_caught - a.count_caught;
    });
    
    // Clear loading state
    loadingState.style.display = 'none';
    statsContainer.style.display = 'flex';
    
    // Render cards
    types.forEach(type => {
      const card = document.createElement('div');
      card.className = `fraud-type-card ${type.implemented ? '' : 'not-implemented'}`;
      
      const badgeHtml = type.implemented 
        ? '' 
        : `<a href="concept.html?type=${encodeURIComponent(type.type_name)}" class="badge-unbuilt" title="View technical roadmap & architecture concept">Not yet built &mdash; see roadmap</a>`;
        
      const descHtml = type.implemented 
        ? `<p class="ft-desc">${type.detection_method}</p>`
        : ``; // If not implemented, don't show a description or fake reason
        
      const countHtml = type.implemented
        ? `<div class="ft-count">${type.count_caught}</div>
           <div class="ft-count-label">detections</div>`
        : `<div class="ft-count text-muted" style="opacity: 0.5;">0</div>
           <div class="ft-count-label">detections</div>`;

      card.innerHTML = `
        <div class="ft-info">
          <div class="ft-header">
            <h3 class="ft-title">${type.display_name}</h3>
            ${badgeHtml}
          </div>
          ${descHtml}
        </div>
        <div class="ft-count-box">
          ${countHtml}
        </div>
      `;
      
      statsContainer.appendChild(card);
    });

  } catch (err) {
    console.error(err);
    loadingState.style.display = 'none';
    errorState.style.display = 'flex';
    errorMsg.textContent = "Unable to fetch telemetry from the backend service. Ensure the API is running.";
  }
});
