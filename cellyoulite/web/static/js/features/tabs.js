// Top-level Curate / Results tabs. Each .app-tab carries data-panel; clicking it
// shows the matching .tab-panel and hides the rest.
export function initTabs() {
  const tabs = [...document.querySelectorAll(".app-tab")];
  const panels = [...document.querySelectorAll(".tab-panel")];
  tabs.forEach(btn => {
    btn.onclick = () => {
      tabs.forEach(b => b.classList.toggle("active", b === btn));
      panels.forEach(p => { p.hidden = p.id !== btn.dataset.panel; });
    };
  });
}
