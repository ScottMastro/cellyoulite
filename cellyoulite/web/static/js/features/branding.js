// Easter egg: three quick clicks on the version pill renames the app and
// lights up the header with a neon glow. Click 3x again to revert.
export function initBranding() {
  const header = document.querySelector(".app-header");
  const name = document.getElementById("brand-name");
  if (!header || !name) return;
  const V = document.body.dataset.version || "";
  const NORMAL = `Cell<em>x</em>You Lite<span class="tag" id="brand-tag">organoid · v${V}</span>`;
  const NEON   = `Cell<em>x</em>Yue Lite<span class="tag" id="brand-tag">organoid · v${V}</span>`;
  let clicks = 0, timer = null;
  function arm() {
    const tag = document.getElementById("brand-tag");
    if (!tag) return;
    tag.style.cursor = "pointer";
    tag.addEventListener("click", () => {
      clicks += 1;
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => { clicks = 0; }, 600);
      if (clicks >= 3) {
        clicks = 0;
        const on = header.classList.toggle("neon");
        name.innerHTML = on ? NEON : NORMAL;
        arm();   // re-bind on the freshly-replaced tag span
      }
    });
  }
  arm();
}
