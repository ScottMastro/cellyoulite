// Cookie helpers (used for the password-less profile selection).
export function getCookie(name) {
  const esc = name.replace(/([.$?*|{}()\[\]\\\/+^])/g, "\\$1");
  const m = document.cookie.match(new RegExp("(?:^|; )" + esc + "=([^;]*)"));
  return m ? decodeURIComponent(m[1]) : null;
}

export function setCookie(name, value, maxAge) {
  const secure = location.protocol === "https:" ? "; Secure" : "";
  document.cookie =
    `${name}=${encodeURIComponent(value)}; path=/; max-age=${maxAge}; SameSite=Lax${secure}`;
}

export function deleteCookie(name) {
  document.cookie = `${name}=; path=/; max-age=0; SameSite=Lax`;
}
