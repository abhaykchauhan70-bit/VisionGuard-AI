/* result.js - loads a single video's status, videos, predictions, detections */

function getVideoIdFromUrl() {
  return new URLSearchParams(window.location.search).get("id");
}

async function loadResult() {
  const id = getVideoIdFromUrl();
  if (!id) { window.location.href = "history.html"; return; }

  try {
    const video = await apiFetch(`/videos/${id}`);
    document.getElementById("video-title").textContent = video.original_filename;
    document.getElementById("video-status-line").innerHTML =
      `${statusBadge(video.status)} &nbsp; ${video.duration_seconds ? video.duration_seconds.toFixed(1) + "s" : ""} &nbsp; ${video.fps ? Math.round(video.fps) + " fps" : ""}`;

    document.getElementById("original-video").src = `${API_BASE}/uploads/${video.filename}`;

    if (video.status === "done") {
      document.getElementById("processed-video").src =
        `${API_BASE}/results/processed_${video.filename}`;
    } else {
      document.getElementById("processed-pending").style.display = "block";
    }

    const [predictions, objects] = await Promise.all([
      apiFetch(`/predictions/${id}`).catch(() => []),
      apiFetch(`/predictions/${id}/objects`).catch(() => []),
    ]);

    renderPrediction(predictions);
    renderObjectsSummary(objects);
    renderObjectsTable(objects);
  } catch (e) {
    console.error(e);
  }
}

function renderPrediction(predictions) {
  const box = document.getElementById("prediction-box");
  if (!predictions.length) {
    box.innerHTML = `<span class="text-muted">No activity prediction yet — train the CNN-LSTM model to enable this (see README).</span>`;
    return;
  }
  const p = predictions[0];
  box.innerHTML = `
    <div style="font-size:22px;font-weight:700;color:var(--accent);text-transform:capitalize;">${p.predicted_label}</div>
    <div class="text-muted" style="font-size:13px;margin-top:6px;">
      ${Math.round(p.confidence * 100)}% confidence · model: ${p.model_used}
    </div>
  `;
}

function renderObjectsSummary(objects) {
  const box = document.getElementById("objects-summary");
  if (!objects.length) { box.textContent = "No objects detected yet."; return; }
  const counts = {};
  objects.forEach(o => counts[o.object_name] = (counts[o.object_name] || 0) + 1);
  box.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => `<span class="badge badge-done" style="margin:0 6px 6px 0;">${name} × ${count}</span>`)
    .join("");
}

function renderObjectsTable(objects) {
  const tbody = document.querySelector("#objects-table tbody");
  if (!objects.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="text-muted">No detections yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = objects.slice(0, 100).map(o => `
    <tr>
      <td>${o.frame_number}</td>
      <td class="text-muted">${o.timestamp.toFixed(2)}s</td>
      <td>${o.object_name}</td>
      <td>${Math.round(o.confidence * 100)}%</td>
    </tr>
  `).join("");
}
