// Password-less profiles. The deployment sits behind an upstream basic-auth
// gate; here a user just picks (or creates) a username, persisted in a cookie
// so the server can read it and a returning visitor is taken straight in.
import { $, escapeHtml } from "../core/dom.js";
import { state } from "../core/state.js";
import { getCookie, setCookie, deleteCookie } from "../core/cookies.js";
import { refreshAll } from "./polling.js";

export const USER_KEY = "cyl_user";
const USER_TTL = 60 * 60 * 24 * 365;   // 1 year, in seconds

export function showUserChip() {
  const chip = $("userchip");
  if (!chip) return;
  if (state.user) {
    $("userchip-name").textContent = state.user;
    $("userchip-avatar").textContent = state.user.slice(0, 1).toUpperCase();
    chip.hidden = false;
  } else {
    chip.hidden = true;
  }
}

async function fetchUsers() {
  try {
    const r = await fetch("/api/users");
    if (!r.ok) return [];
    return (await r.json()).users || [];
  } catch (e) { return []; }
}

function renderUserList(users) {
  const list = $("user-list");
  const last = getCookie(USER_KEY);
  const cards = users.map(u =>
    `<button class="user-card${u === last ? " last" : ""}" data-user="${escapeHtml(u)}">
       <span class="user-avatar">${escapeHtml(u.slice(0, 1).toUpperCase())}</span>
       <span class="user-cardname">${escapeHtml(u)}</span>
     </button>`).join("");
  list.innerHTML = cards +
    `<button class="user-card user-card-new" id="user-card-new">
       <span class="user-avatar">+</span>
       <span class="user-cardname">New user</span>
     </button>`;
  list.querySelectorAll(".user-card[data-user]").forEach(b =>
    b.addEventListener("click", () => selectUser(b.dataset.user)));
  $("user-card-new").addEventListener("click", openNewUser);
}

function openNewUser() {
  $("user-new").hidden = false;
  const nc = $("user-card-new");
  if (nc) nc.style.display = "none";
  $("user-new-name").focus();
}

function closeNewUser() {
  $("user-new").hidden = true;
  $("user-new-name").value = "";
  $("user-new-err").textContent = "";
  const nc = $("user-card-new");
  if (nc) nc.style.display = "";
}

export function selectUser(name) {
  state.user = name;
  setCookie(USER_KEY, name, USER_TTL);
  $("user-gate").classList.add("hidden");
  showUserChip();
  refreshAll();
}

async function createUser() {
  const name = $("user-new-name").value.trim();
  const err = $("user-new-err");
  err.textContent = "";
  if (!name) { err.textContent = "enter a username"; return; }
  try {
    const r = await fetch("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: name }),
    });
    const d = await r.json();
    if (!r.ok) { err.textContent = d.detail || "could not create user"; return; }
    selectUser(d.username);
  } catch (e) { err.textContent = "network error"; }
}

export async function initUserGate() {
  $("user-gate").classList.remove("hidden");
  closeNewUser();
  renderUserList(await fetchUsers());
}

export function initUsers() {
  $("user-new-create").addEventListener("click", createUser);
  $("user-new-cancel").addEventListener("click", closeNewUser);
  $("user-new-name").addEventListener("keydown", e => { if (e.key === "Enter") createUser(); });
  $("user-switch").addEventListener("click", () => {
    state.user = null;
    deleteCookie(USER_KEY);
    showUserChip();
    initUserGate();
  });
}
