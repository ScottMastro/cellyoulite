// Tiny DOM helpers shared across the app.
export const $ = (id) => document.getElementById(id);

export const setStatus = (cls, text) => {
  $("status").innerHTML = `<span class="dot ${cls}"></span>${text}`;
};

export const renderPills = (target, items) => {
  const el = $(target);
  el.innerHTML = items.map(([k, v]) => `<span class="pill">${k} <strong>${v}</strong></span>`).join("");
  el.hidden = items.length === 0;
};

export function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => (
    {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"}[c]
  ));
}
