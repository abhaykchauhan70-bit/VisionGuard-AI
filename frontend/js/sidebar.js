/* Injects the sidebar nav into any page with <div id="sidebar-mount"></div>.
   `active` (global var, set before this script runs) highlights the current page. */
function renderSidebar(active) {
  const links = [
    { href: "dashboard.html", icon: "◧", label: "Dashboard", key: "dashboard" },
    { href: "upload.html", icon: "⬆", label: "Upload Video", key: "upload" },
    { href: "live.html", icon: "●", label: "Live Camera", key: "live" },
    { href: "history.html", icon: "☰", label: "History", key: "history" },
    { href: "performance.html", icon: "◇", label: "Model Performance", key: "performance" },
    { href: "settings.html", icon: "⚙", label: "Settings", key: "settings" },
  ];
  const user = Auth.getUser();
  const initials = user ? user.username.slice(0, 2).toUpperCase() : "??";

  const html = `
    <div class="brand">
      <div class="brand-mark">VG</div>
      <div>
        <div class="brand-name">VisionGuard AI</div>
        <div class="brand-sub">Video Intelligence</div>
      </div>
    </div>
    ${links.map(l => `
      <a href="${l.href}" class="nav-link ${l.key === active ? 'active' : ''}">
        <span class="nav-icon">${l.icon}</span><span>${l.label}</span>
      </a>`).join("")}
    <div class="nav-footer">
      <div class="user-chip">
        <div class="avatar" id="chip-avatar">${initials}</div>
        <div>
          <div style="font-size:13px;font-weight:600;" id="chip-username">${user ? user.username : ""}</div>
          <div style="font-size:11px;color:var(--text-muted);">Signed in</div>
        </div>
      </div>
      <button class="logout-btn" id="logout-btn">↩ Logout</button>
    </div>
  `;
  document.getElementById("sidebar-mount").innerHTML = html;
  document.getElementById("logout-btn").addEventListener("click", () => Auth.logout());
}
