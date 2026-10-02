/* dashboard.js - populates stat cards, charts, and the recent-videos table */

Chart.defaults.color = "#8b98ab";
Chart.defaults.font.family = "Inter";

async function loadDashboard() {
  try {
    const stats = await apiFetch("/dashboard/stats");
    document.getElementById("stat-videos").textContent = stats.total_videos;
    document.getElementById("stat-predictions").textContent = stats.total_predictions;
    document.getElementById("stat-events").textContent = stats.total_events;
    document.getElementById("stat-confidence").textContent =
      `${Math.round(stats.average_confidence * 100)}%`;
  } catch (e) { console.error(e); }

  try {
    const dist = await apiFetch("/dashboard/activity-distribution");
    renderActivityChart(dist);
  } catch (e) { console.error(e); }

  try {
    const videos = await apiFetch("/videos");
    renderRecentTable(videos.slice(0, 6));
    renderConfidenceChart(videos);
  } catch (e) { console.error(e); }
}

function renderActivityChart(distObj) {
  const ctx = document.getElementById("activityChart");
  const labels = Object.keys(distObj);
  const values = Object.values(distObj);

  if (labels.length === 0) {
    ctx.parentElement.innerHTML += `<div class="text-muted" style="text-align:center;padding:40px 0;">No predictions yet — analyze a video to see activity breakdown.</div>`;
    return;
  }

  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#60a5fa"],
        borderWidth: 0,
      }],
    },
    options: {
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, padding: 14 } } },
    },
  });
}

function renderConfidenceChart(videos) {
  const ctx = document.getElementById("confidenceChart");
  const labeled = videos.filter(v => v.status === "done").slice(-8);
  new Chart(ctx, {
    type: "line",
    data: {
      labels: labeled.map(v => v.original_filename.slice(0, 12)),
      datasets: [{
        label: "Confidence",
        data: labeled.map(() => null), // populated below once predictions fetched
        borderColor: "#38bdf8",
        backgroundColor: "rgba(56,189,248,0.12)",
        tension: 0.35,
        fill: true,
        pointRadius: 3,
      }],
    },
    options: {
      scales: {
        y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.05)" } },
        x: { grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function renderRecentTable(videos) {
  const tbody = document.querySelector("#recent-table tbody");
  if (!videos.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="text-muted">No videos uploaded yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = videos.map(v => `
    <tr>
      <td>${v.original_filename}</td>
      <td>${statusBadge(v.status)}</td>
      <td class="text-muted">${new Date(v.uploaded_at).toLocaleString()}</td>
      <td><a href="result.html?id=${v.id}" class="text-muted">View →</a></td>
    </tr>
  `).join("");
}
