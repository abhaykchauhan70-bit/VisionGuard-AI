/* ==========================================================================
   app.js - shared API client, auth helpers, sidebar active-link logic.
   Loaded on every page.
   ========================================================================== */

const API_BASE = "http://127.0.0.1:8000";

const Auth = {
  getToken() { return localStorage.getItem("vg_token"); },
  setToken(t) { localStorage.setItem("vg_token", t); },
  clear() { localStorage.removeItem("vg_token"); localStorage.removeItem("vg_user"); },
  setUser(u) { localStorage.setItem("vg_user", JSON.stringify(u)); },
  getUser() { try { return JSON.parse(localStorage.getItem("vg_user")); } catch { return null; } },
  requireLogin() {
    if (!this.getToken()) window.location.href = "index.html";
  },
  logout() { this.clear(); window.location.href = "index.html"; },
};

async function apiFetch(path, options = {}) {
  const headers = options.headers || {};
  const token = Auth.getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (!(options.body instanceof FormData) && options.body) {
    headers["Content-Type"] = "application/json";
  }
  const resp = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (resp.status === 401) {
    Auth.clear();
    window.location.href = "index.html";
    throw new Error("Unauthorized");
  }
  if (!resp.ok) {
    let detail = "Request failed";
    try { detail = (await resp.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  if (resp.status === 204) return null;
  return resp.json();
}

function showAlert(el, message, type = "error") {
  el.textContent = message;
  el.className = `alert ${type}`;
  el.style.display = "block";
}

function statusBadge(status) {
  const map = { uploaded: "Uploaded", processing: "Processing", done: "Done", failed: "Failed" };
  return `<span class="badge badge-${status}">${map[status] || status}</span>`;
}

function fillUserChip() {
  const user = Auth.getUser();
  const nameEl = document.getElementById("chip-username");
  const avatarEl = document.getElementById("chip-avatar");
  if (user && nameEl) {
    nameEl.textContent = user.username;
    if (avatarEl) avatarEl.textContent = user.username.slice(0, 2).toUpperCase();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) logoutBtn.addEventListener("click", () => Auth.logout());
  fillUserChip();
});
