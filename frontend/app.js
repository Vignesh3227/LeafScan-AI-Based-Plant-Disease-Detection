const API_BASE = window.location.origin;
const CATEGORY_COLORS = {
  fungicide: 'cat-fungicide',
  insecticide: 'cat-insecticide',
  acaricide: 'cat-acaricide',
  fertilizer: 'cat-fertilizer',
  biopesticide: 'cat-biopesticide',
  equipment: 'cat-equipment',
};
const CATEGORY_LABELS = {
  fungicide: 'Fungicide',
  insecticide: 'Insecticide',
  acaricide: 'Acaricide',
  fertilizer: 'Fertilizer',
  biopesticide: 'Bio-Pesticide',
  equipment: 'Equipment',
};

let selectedFile = null;

// ── DOM refs ──
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const previewWrap = document.getElementById('previewWrap');
const previewImg = document.getElementById('previewImg');
const previewImgBox = document.getElementById('previewImgBox');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const gradcamCheck = document.getElementById('gradcamCheck');
const loadingOverlay = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('results');
const marketSection = document.getElementById('marketplace');
const errorToast = document.getElementById('errorToast');
const resetBtn = document.getElementById('resetBtn');

// ── Upload zone ──
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) setFile(file);
  else showToast('Please drop a valid image file.');
});
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

function setFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    previewWrap.classList.add('visible');
    analyzeBtn.disabled = false;
    previewImgBox.classList.add('full');
    hideResults();
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewWrap.classList.remove('visible');
  analyzeBtn.disabled = true;
  hideResults();
});

// ── Analyze ──
analyzeBtn.addEventListener('click', runAnalysis);
resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  previewWrap.classList.remove('visible');
  analyzeBtn.disabled = true;
  hideResults();
  document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
});

async function runAnalysis() {
  if (!selectedFile) return;

  showLoading();

  const formData = new FormData();
  formData.append('file', selectedFile);
  const useGradcam = gradcamCheck.checked;

  try {
    const res = await fetch(`${API_BASE}/api/predict?gradcam=${useGradcam}`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    hideLoading();
    renderResults(data);
  } catch (err) {
    hideLoading();
    showToast(`Analysis failed: ${err.message}`);
  }
}

// ── Loading ──
function showLoading() {
  loadingOverlay.classList.add('active');
  const steps = ['step1', 'step2', 'step3', 'step4'];
  steps.forEach((id, i) => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
    setTimeout(() => {
      el.classList.add('active');
      if (i > 0) {
        const prev = document.getElementById(steps[i - 1]);
        prev.classList.remove('active');
        prev.classList.add('done');
      }
    }, i * 700);
  });
}

function hideLoading() {
  loadingOverlay.classList.remove('active');
  ['step1','step2','step3','step4'].forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
  });
}

// ── Render results ──
function renderResults(data) {
  const { prediction, recommendation, products, images } = data;

  // Demo banner
  const demoBanner = document.getElementById('demoBanner');
  demoBanner.classList.toggle('visible', !!prediction.demo_mode);

  // Subtitle
  document.getElementById('resultSubtitle').textContent =
    `Analysis completed — ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

  // Plant & disease
  document.getElementById('plantName').textContent = prediction.plant;
  const diseaseEl = document.getElementById('diseaseName');
  diseaseEl.textContent = prediction.disease;
  diseaseEl.className = 'disease-name' + (prediction.is_healthy ? ' healthy' : '');

  // Severity
  const badge = document.getElementById('severityBadge');
  badge.textContent = recommendation.severity.charAt(0).toUpperCase() + recommendation.severity.slice(1);
  badge.className = 'severity-badge severity-' + recommendation.severity;

  // Confidence
  const pct = Math.round(prediction.confidence * 100);
  document.getElementById('confidenceText').textContent = `${pct}%`;
  setTimeout(() => {
    document.getElementById('confidenceFill').style.width = `${pct}%`;
  }, 100);

  // Top 5
  const top5El = document.getElementById('top5List');
  top5El.innerHTML = prediction.top5.map((item, i) => {
    const name = item.class.replace(/___/g, ' — ').replace(/_/g, ' ');
    const p = Math.round(item.confidence * 100);
    return `
      <div class="top5-item">
        <span class="top5-rank">${i + 1}</span>
        <span class="top5-name" title="${name}">${name}</span>
        <div class="top5-bar-wrap"><div class="top5-bar" style="width:${p}%"></div></div>
        <span class="top5-pct">${p}%</span>
      </div>`;
  }).join('');

  // Recommendation
  document.getElementById('recDescription').textContent = recommendation.description;
  renderList('recTreatment', recommendation.treatment);
  renderList('recPrevention', recommendation.prevention);

  const causesWrap = document.getElementById('causesSectionWrap');
  if (recommendation.causes && recommendation.causes.length > 0) {
    causesWrap.style.display = '';
    renderList('recCauses', recommendation.causes);
  } else {
    causesWrap.style.display = 'none';
  }

  // Grad-CAM
  const gradcamRow = document.getElementById('gradcamRow');
  if (images.gradcam) {
    document.getElementById('resultOriginal').src = images.original || previewImg.src;
    document.getElementById('resultGradcam').src = images.gradcam;
    gradcamRow.classList.add('visible');
  } else {
    gradcamRow.classList.remove('visible');
  }

  // Show sections
  resultsSection.classList.add('visible');
  marketSection.classList.add('visible');

  // Marketplace
  renderProducts(products, prediction.disease, prediction.plant);

  // Scroll
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);

  // Nav highlight
  document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
  document.querySelector('nav a[href="#results"]').classList.add('active');
}

function renderList(id, items) {
  document.getElementById(id).innerHTML = (items || []).map(
    item => `<li>${item}</li>`
  ).join('');
}

function renderProducts(products, disease, plant) {
  document.getElementById('marketSubtitle').textContent =
    `Recommended products for ${plant} — ${disease}`;

  const grid = document.getElementById('productsGrid');
  if (!products || products.length === 0) {
    grid.innerHTML = '<p style="color:var(--text-muted);font-size:13px;">No specific products found.</p>';
    return;
  }

  grid.innerHTML = products.map(p => {
    const catClass = CATEGORY_COLORS[p.category] || '';
    const catLabel = CATEGORY_LABELS[p.category] || p.category;
    const stars = renderStars(p.rating);
    const stockClass = p.in_stock ? 'in-stock' : 'out-stock';
    const stockLabel = p.in_stock ? 'In Stock' : 'Out of Stock';
    return `
      <div class="product-card">
        <span class="stock-badge ${stockClass}">${stockLabel}</span>
        <span class="product-category ${catClass}">${catLabel}</span>
        <div class="product-name">${p.name}</div>
        <div class="product-brand">${p.brand} &middot; ${p.unit}</div>
        <p class="product-desc">${p.description}</p>
        <div class="product-footer">
          <div class="product-price">$${p.price.toFixed(2)} <span>/ ${p.unit}</span></div>
          <div class="product-rating">${stars} ${p.rating}</div>
        </div>
        <button class="btn btn-primary buy-btn" onclick="handleBuy('${p.id}', '${p.name}')" ${p.in_stock ? '' : 'disabled'}>
          ${p.in_stock ? 'Add to Cart' : 'Out of Stock'}
        </button>
      </div>`;
  }).join('');
}

function renderStars(rating) {
  const full = Math.floor(rating);
  const half = rating % 1 >= 0.5;
  let stars = '';
  for (let i = 0; i < 5; i++) {
    if (i < full) stars += '&#9733;';
    else if (i === full && half) stars += '&#9734;';
    else stars += '&#9734;';
  }
  return stars;
}

function handleBuy(id, name) {
  showToast(`"${name}" added to cart (demo — no real purchase)`, 'success');
}

// ── Helpers ──
function hideResults() {
  resultsSection.classList.remove('visible');
  marketSection.classList.remove('visible');
  document.getElementById('confidenceFill').style.width = '0%';
  document.getElementById('gradcamRow').classList.remove('visible');
}

let toastTimer = null;
function showToast(msg, type = 'error') {
  errorToast.textContent = msg;
  errorToast.style.borderColor = type === 'error' ? 'rgba(248,113,113,0.3)' : 'rgba(74,222,128,0.3)';
  errorToast.style.color = type === 'error' ? 'var(--danger)' : 'var(--primary)';
  errorToast.style.background = type === 'error' ? '#1a0a0a' : 'rgba(8,13,11,0.95)';
  errorToast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => errorToast.classList.remove('show'), 3500);
}

// ── Smooth nav ──
document.querySelectorAll('nav a').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
    link.classList.add('active');
    const target = document.querySelector(link.getAttribute('href'));
    if (target) target.scrollIntoView({ behavior: 'smooth' });
  });
});