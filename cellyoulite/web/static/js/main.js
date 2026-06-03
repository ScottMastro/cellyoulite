// Entry module for index.html. Wires every feature's event handlers, then
// boots: a returning visitor (cookie) goes straight in; otherwise the
// password-less profile gate is shown.
import { $ } from "./core/dom.js";
import { state } from "./core/state.js";
import { getCookie } from "./core/cookies.js";
import { initBranding } from "./features/branding.js";
import { initData } from "./features/data.js";
import { initBoxplot } from "./features/boxplot.js";
import { initWell } from "./features/well.js";
import { initUsers, initUserGate, showUserChip, USER_KEY } from "./features/users.js";
import { refreshAll, startPolling } from "./features/polling.js";

initBranding();
initData();
initBoxplot();
initWell();
initUsers();
startPolling();

// ---------------- boot ----------------
const saved = getCookie(USER_KEY);
if (saved) {
  // Returning visitor — gate is already pre-hidden server-side.
  state.user = saved;
  $("user-gate").classList.add("hidden");
  showUserChip();
  refreshAll();
} else {
  initUserGate();
}
