/*! coi-service-worker v0.1.7 - Guido Zuidhof and contributors, licensed under MIT */
/*
 * Service worker that makes a page cross-origin isolated by adding
 * COOP/COEP headers to all responses. This enables SharedArrayBuffer
 * on hosts (like GitHub Pages) that don't allow custom HTTP headers.
 */
if (typeof window === "undefined") {
  // Service worker context
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) =>
    e.waitUntil(self.clients.claim())
  );
  self.addEventListener("fetch", (e) => {
    if (e.request.cache === "only-if-cached" && e.request.mode !== "same-origin") {
      return;
    }
    e.respondWith(
      fetch(e.request).then((r) => {
        if (r.status === 0) return r;
        const headers = new Headers(r.headers);
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        return new Response(r.body, {
          status: r.status,
          statusText: r.statusText,
          headers,
        });
      })
    );
  });
} else {
  // Window context — register the service worker then reload once active
  const register = async () => {
    if (window.crossOriginIsolated) return;
    const reg = await navigator.serviceWorker.register("coi-serviceworker.js", {
      scope: "./",
    });
    if (reg.active && !navigator.serviceWorker.controller) {
      window.location.reload();
    } else if (!reg.active) {
      const sw = reg.installing || reg.waiting;
      sw.addEventListener("statechange", () => {
        if (sw.state === "activated") window.location.reload();
      });
    }
  };
  register();
}
