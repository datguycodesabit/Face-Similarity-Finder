const form = document.querySelector('#analyzeForm');
const inputs = [...document.querySelectorAll('.view-input')];
const submit = document.querySelector('#submitButton');
const statusBadge = document.querySelector('#statusBadge');
const errorBox = document.querySelector('#errorBox');
const loadingBox = document.querySelector('#loadingBox');
const resultsSection = document.querySelector('#resultsSection');
const resultsGrid = document.querySelector('#resultsGrid');
const overlayToggle = document.querySelector('#overlayToggle');
let ready = false;
let hasRenderedAnalysis = false;

function allViewsSelected() {
  return inputs.every(input => input.files.length === 1);
}

function updateSubmit() {
  submit.disabled = !(ready && allViewsSelected());
}

async function refreshStatus() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    ready = status.model_ready && status.index_ready;
    statusBadge.className = `status-badge ${ready ? 'ready' : 'blocked'}`;
    statusBadge.textContent = ready
      ? `${status.counts.identities.toLocaleString()} eligible identities`
      : (!status.model_ready ? 'Model setup needed'
        : status.indexing_status === 'running'
          ? `${status.counts.indexed.toLocaleString()} / ${status.counts.total.toLocaleString()} rebuilt`
          : 'Dense v2 reindex needed');
    updateSubmit();
  } catch (_) {
    statusBadge.className = 'status-badge blocked';
    statusBadge.textContent = 'Status unavailable';
  }
}

function showPreview(input) {
  const file = input.files[0];
  const drop = document.querySelector(`[data-view="${input.name}"]`);
  const canvas = drop.querySelector('canvas');
  drop.classList.toggle('has-image', Boolean(file));
  errorBox.hidden = true;
  if (hasRenderedAnalysis) {
    resultsSection.hidden = true;
    document.querySelector('#selectionNote').hidden = false;
    submit.firstChild.textContent = 'Analyze new photos ';
  }
  if (!file) { updateSubmit(); return; }
  const image = new Image();
  image.onload = () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.getContext('2d').drawImage(image, 0, 0);
    URL.revokeObjectURL(image.src);
  };
  image.src = URL.createObjectURL(file);
  updateSubmit();
}

inputs.forEach(input => {
  input.addEventListener('change', () => showPreview(input));
  const drop = document.querySelector(`[data-view="${input.name}"]`);
  ['dragenter', 'dragover'].forEach(type => drop.addEventListener(type, event => {
    event.preventDefault(); drop.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(type => drop.addEventListener(type, event => {
    event.preventDefault(); drop.classList.remove('dragging');
  }));
  drop.addEventListener('drop', event => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    showPreview(input);
  });
});

function drawOverlay(canvas, image, points) {
  const render = () => {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = 'rgba(200,242,91,.82)';
    points.forEach(([x, y]) => {
      context.beginPath();
      context.arc(x * canvas.width, y * canvas.height, Math.max(0.8, canvas.width / 300), 0, Math.PI * 2);
      context.fill();
    });
  };
  image.complete ? render() : image.addEventListener('load', render, { once: true });
}

function titleCase(value) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
}

function renderShape(shape) {
  document.querySelector('#shapeBlend').textContent = `Mostly ${shape.primary}, with ${shape.secondary} influence`;
  document.querySelector('#shapeCaveat').textContent = shape.caveat;
  document.querySelector('#agreementBadge').textContent = `${shape.three_view_agreement.toFixed(1)}% three-view agreement`;
  const labels = {
    length_width_ratio: 'Length ÷ width', forehead_cheek_ratio: 'Forehead ÷ cheeks',
    temple_cheek_ratio: 'Temples ÷ cheeks', jaw_cheek_ratio: 'Jaw ÷ cheeks',
    jaw_taper: 'Jaw taper', chin_cheek_ratio: 'Chin ÷ cheeks', jaw_angularity: 'Jaw angularity',
    upper_third: 'Upper third', middle_third: 'Middle third', lower_third: 'Lower third'
  };
  const grid = document.querySelector('#measurementsGrid');
  grid.replaceChildren();
  Object.entries(shape.measurements).forEach(([key, value]) => {
    const item = document.createElement('div');
    item.innerHTML = `<span>${labels[key] || titleCase(key)}</span><strong>${Number(value).toFixed(3)}</strong>`;
    grid.append(item);
  });
}

function renderHaircuts(recommendations) {
  const grid = document.querySelector('#haircutGrid');
  grid.replaceChildren();
  recommendations.forEach((recommendation, index) => {
    const card = document.createElement('article');
    card.className = 'haircut-card';
    card.innerHTML = `<span class="recommendation-rank">0${index + 1}</span><h3>${recommendation.name}</h3><p>${recommendation.rationale}</p><small><b>Watch for:</b> ${recommendation.caution}</small>`;
    grid.append(card);
  });
}

function renderDiagnostics(diagnostics) {
  const grid = document.querySelector('#diagnosticsGrid');
  grid.replaceChildren();
  diagnostics.forEach(item => {
    const row = document.createElement('div');
    row.innerHTML = `<strong>${titleCase(item.view)}</strong><span>Yaw ${Number(item.yaw).toFixed(1)}° · Pitch ${Number(item.pitch).toFixed(1)}° · Quality ${Math.round(Number(item.quality) * 100)}%</span>`;
    grid.append(row);
  });
}

function renderMatches(matches) {
  resultsGrid.replaceChildren();
  matches.forEach(match => {
    const card = document.createElement('article');
    card.className = 'result-card';
    card.innerHTML = `<div class="result-visual"><span class="result-rank">${match.rank}</span><img alt="LFW structural reference of ${match.identity}" src="${match.image_url}"><canvas aria-hidden="true"></canvas></div><div class="result-info"><h3>${match.identity}</h3><div class="metric"><span>Shape distance</span><strong>${match.distance.toFixed(4)}</strong></div><div class="metric"><span>Silhouette</span><strong>${match.breakdown.silhouette.toFixed(4)}</strong></div><div class="metric"><span>Proportions</span><strong>${match.breakdown.jaw_chin_and_proportions.toFixed(4)}</strong></div></div>`;
    drawOverlay(card.querySelector('canvas'), card.querySelector('img'), match.overlay);
    resultsGrid.append(card);
  });
}

overlayToggle.addEventListener('change', () => {
  resultsGrid.querySelectorAll('canvas').forEach(canvas => { canvas.hidden = !overlayToggle.checked; });
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!allViewsSelected()) return;
  errorBox.hidden = true;
  loadingBox.hidden = false;
  submit.disabled = true;
  const data = new FormData(form);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: data, cache: 'no-store' });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'The analysis could not be completed.');
    renderShape(payload.shape);
    renderDiagnostics(payload.diagnostics);
    renderHaircuts(payload.recommendations);
    renderMatches(payload.matches);
    hasRenderedAnalysis = true;
    document.querySelector('#selectionNote').hidden = true;
    submit.firstChild.textContent = 'Analyze these photos ';
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    errorBox.textContent = error.message;
    errorBox.hidden = false;
  } finally {
    loadingBox.hidden = true;
    updateSubmit();
  }
});

refreshStatus();
const statusTimer = window.setInterval(async () => {
  if (ready) { window.clearInterval(statusTimer); return; }
  await refreshStatus();
}, 3000);
