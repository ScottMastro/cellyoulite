// URL builders for the image / overlay endpoints.
import { $ } from "./dom.js";
import { state } from "./state.js";

export const alignFlag = () => ($("toggle-align") && $("toggle-align").checked) ? 1 : 0;

// Identity of a well on the wire. A well is only unique within its batch, so
// every per-well endpoint takes all three.
export const wellQs = (mountId, batch, folderName) =>
  `mount_id=${encodeURIComponent(mountId)}`
  + `&batch=${encodeURIComponent(batch)}`
  + `&folder_name=${encodeURIComponent(folderName)}`;

// The currently open well, for the many call sites that just want "this one".
export const openWellQs = () =>
  wellQs(state.well.mount_id, state.well.batch, state.well.folder_name);

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
