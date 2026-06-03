// URL builders for the image / overlay endpoints.
import { $ } from "./dom.js";
import { state } from "./state.js";

export const alignFlag = () => ($("toggle-align") && $("toggle-align").checked) ? 1 : 0;

export const imgUrl = (key, thumb = 0, aligned = 0) => {
  let q = `key=${encodeURIComponent(key)}`;
  if (thumb)   q += `&thumb=${thumb}`;
  if (aligned) q += `&aligned=1`;
  return `/api/image?${q}`;
};

// `state.overlayVersion` is bumped whenever validation overrides change so the
// server-rendered edges PNG re-fetches instead of being served from cache.
export const edgesUrl = (key, aligned = 1, hideInvalid = 0) =>
  `/api/cellpose-edges?key=${encodeURIComponent(key)}&aligned=${aligned}`
  + `&hide_invalid=${hideInvalid ? 1 : 0}&v=${state.overlayVersion}`;
