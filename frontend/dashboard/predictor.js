document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictor-form');
  const submitBtn = document.getElementById('submit-btn');
  
  const resultEmpty = document.getElementById('result-empty');
  const resultError = document.getElementById('result-error');
  const resultSuccess = document.getElementById('result-success');
  const errorMsg = document.getElementById('error-message');
  
  // Range input value displays
  const rangeInputs = form.querySelectorAll('.range-input');
  rangeInputs.forEach(input => {
    const display = document.getElementById(input.id + '_val');
    if (display) {
      input.addEventListener('input', () => {
        // format based on step
        if (input.step === '0.01') {
          display.textContent = parseFloat(input.value).toFixed(2);
        } else {
          display.textContent = input.value;
        }
      });
    }
  });
  
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Set loading state
    submitBtn.disabled = true;
    submitBtn.classList.add('is-loading');
    
    // Build payload
    const formData = new FormData(form);
    const payload = {};
    
    // Initialize checkboxes to 0 (since unchecked ones don't appear in FormData)
    const checkboxes = form.querySelectorAll('.checkbox-input');
    checkboxes.forEach(cb => {
      payload[cb.name] = 0;
    });
    
    for (const [key, value] of formData.entries()) {
      const element = form.elements[key];
      const isNumberField = element && (element.type === 'number' || element.type === 'range');
      
      if (value === 'on' || value === '1') {
        payload[key] = 1;
      } else if (value === '') {
        payload[key] = 0;
      } else if (isNumberField) {
        payload[key] = Number(value);
      } else {
        payload[key] = value;
      }
    }
    
    // Fill in required API fields that might not be in the form
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
      const response = await fetch('http://localhost:8000/score', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(finalPayload)
      });
      
      if (!response.ok) {
        throw new Error(`API Error (${response.status}): Unable to score transaction. Ensure the backend is running.`);
      }
      
      const data = await response.json();
      
      // Render success
      renderResult(data);
      
    } catch (err) {
      console.error(err);
      renderError("Unable to connect to the prediction service or the backend returned an error. Please verify the API is running at http://localhost:8000.");
    } finally {
      submitBtn.disabled = false;
      submitBtn.classList.remove('is-loading');
    }
  });
  
  function renderError(msg) {
    resultEmpty.style.display = 'none';
    resultSuccess.style.display = 'none';
    resultError.style.display = 'flex'; // ErrorState uses flex column
    errorMsg.textContent = msg;
  }
  
  function renderResult(data) {
    resultEmpty.style.display = 'none';
    resultError.style.display = 'none';
    resultSuccess.style.display = 'block';
    
    // Set score
    const scorePercent = (data.score * 100).toFixed(1);
    document.getElementById('score-value').textContent = `${scorePercent}%`;
    
    // Set action badge
    const badge = document.getElementById('action-badge');
    const actionDesc = document.getElementById('action-desc');
    
    badge.className = 'status-badge'; // reset
    badge.textContent = data.action;
    
    if (data.action === 'ALLOW') {
      badge.classList.add('status-badge--allow');
      actionDesc.textContent = 'Low risk — transaction proceeds normally.';
    } else if (data.action === 'CHALLENGE') {
      badge.classList.add('status-badge--challenge');
      actionDesc.textContent = 'Elevated risk — require additional OTP or verification.';
    } else if (data.action === 'BLOCK') {
      badge.classList.add('status-badge--block');
      actionDesc.textContent = 'High risk — transaction rejected outright.';
    }
    
    // Render SHAP reasons
    const reasonsList = document.getElementById('reasons-list');
    reasonsList.innerHTML = '';
    
    if (data.reasons && data.reasons.length > 0) {
      data.reasons.forEach(reason => {
        const li = document.createElement('li');
        li.className = 'shap-item';
        li.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <path d="M20 6L9 17l-5-5"/>
          </svg>
          <span>${reason}</span>
        `;
        reasonsList.appendChild(li);
      });
    } else {
      const li = document.createElement('li');
      li.className = 'shap-item';
      li.innerHTML = `<span>No significant risk factors found.</span>`;
      reasonsList.appendChild(li);
    }
  }
});
