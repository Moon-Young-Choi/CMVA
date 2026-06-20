(function () {
  const glossary = window.CMVA_GLOSSARY || {};
  const layer = document.getElementById("tooltip-layer");
  if (!layer) return;
  let activeAnchor = null;
  let hideTimer = null;
  let hoveringTooltip = false;
  let focusWithinTooltip = false;

  function contentFor(key) {
    const item = glossary[key];
    if (!item) return null;
    const formula = item.formula ? `<code>${escapeHtml(item.formula)}</code>` : "";
    const link = item.details_url ? `<a href="${item.details_url}">방법론</a>` : "";
    return `<strong>${escapeHtml(item.title || key)}</strong><span>${escapeHtml(item.short || "")}</span>${formula}${link}`;
  }

  function show(event) {
    const button = event.currentTarget;
    const html = contentFor(button.dataset.tooltipKey);
    if (!html) return;
    window.clearTimeout(hideTimer);
    if (activeAnchor && activeAnchor !== button) {
      hideNow();
    }
    activeAnchor = button;
    layer.innerHTML = html;
    layer.hidden = false;
    position(button);
  }

  function requestHide() {
    window.clearTimeout(hideTimer);
    hideTimer = window.setTimeout(() => {
      if (shouldStayOpen()) return;
      hideNow();
    }, 120);
  }

  function hideNow() {
    window.clearTimeout(hideTimer);
    layer.hidden = true;
    layer.innerHTML = "";
    activeAnchor = null;
  }

  function shouldStayOpen() {
    const anchorActive =
      activeAnchor &&
      (activeAnchor.matches(":hover") || activeAnchor === document.activeElement || activeAnchor.contains(document.activeElement));
    return Boolean(anchorActive || hoveringTooltip || focusWithinTooltip);
  }

  function position(anchor) {
    const rect = anchor.getBoundingClientRect();
    const spacing = 8;
    const width = Math.min(320, window.innerWidth - 24);
    let left = rect.left;
    if (left + width > window.innerWidth - 12) left = window.innerWidth - width - 12;
    left = Math.max(12, left);
    let top = rect.bottom + spacing;
    if (top + layer.offsetHeight > window.innerHeight - 12) {
      top = Math.max(12, rect.top - layer.offsetHeight - spacing);
    }
    layer.style.left = `${left}px`;
    layer.style.top = `${top}px`;
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  document.querySelectorAll("[data-tooltip-key]").forEach((button) => {
    button.addEventListener("mouseenter", show);
    button.addEventListener("focus", show);
    button.addEventListener("mouseleave", requestHide);
    button.addEventListener("blur", requestHide);
    button.addEventListener("keydown", (event) => {
      if (event.key === "Escape") hideNow();
    });
  });
  layer.addEventListener("mouseenter", () => {
    hoveringTooltip = true;
    window.clearTimeout(hideTimer);
  });
  layer.addEventListener("mouseleave", () => {
    hoveringTooltip = false;
    requestHide();
  });
  layer.addEventListener("focusin", () => {
    focusWithinTooltip = true;
    window.clearTimeout(hideTimer);
  });
  layer.addEventListener("focusout", () => {
    focusWithinTooltip = false;
    requestHide();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideNow();
  });
})();
