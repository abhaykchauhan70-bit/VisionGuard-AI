/* history.js - table with search, sort, delete */
let allVideos = [];

async function loadHistory() {
  try {
    allVideos = await apiFetch("/videos");
    renderHistoryTable(allVideos);
  } catch (e) {
    console.error(e);
  }

  document.getElementById("search-input").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    renderHistoryTable(allVideos.filter(v => v.original_filename.toLowerCase().includes(q)));
  });
}

function renderHistoryTable(videos) {
  const tbody = document.querySelector("#history-table tbody");
  if (!videos.length) {
    tbody.innerHTML = `<tr><td colspan="5">
      <div class="empty-state"><div class="icon">🎬</div>No videos found.</div>
    </td></tr>`;
    return;
  }
  tbody.innerHTML = videos.map(v => `
    <tr>
      <td>${v.original_filename}</td>
      <td>${statusBadge(v.status)}</td>
      <td class="text-muted">${v.duration_seconds ? v.duration_seconds.toFixed(1) + "s" : "–"}</td>
      <td class="text-muted">${new Date(v.uploaded_at).toLocaleString()}</td>
      <td style="white-space:nowrap;">
        <a href="result.html?id=${v.id}" class="text-muted" style="margin-right:14px;">View</a>
        <a href="#" class="text-muted" style="color:var(--danger);" onclick="deleteVideo(${v.id});return false;">Delete</a>
      </td>
    </tr>
  `).join("");
}

async function deleteVideo(id) {
  if (!confirm("Delete this video and all its results?")) return;
  try {
    await apiFetch(`/videos/${id}`, { method: "DELETE" });
    allVideos = allVideos.filter(v => v.id !== id);
    renderHistoryTable(allVideos);
  } catch (e) {
    alert("Failed to delete: " + e.message);
  }
}
