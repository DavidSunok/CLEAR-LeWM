const header = document.querySelector("[data-header]");

function updateHeader() {
  header?.classList.toggle("is-scrolled", window.scrollY > 24);
}

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
const protocolSection = document.querySelector(".protocol");

function selectTab(nextTab) {
  tabs.forEach((tab) => {
    const selected = tab === nextTab;
    const panel = document.getElementById(tab.getAttribute("aria-controls"));
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (panel) panel.hidden = !selected;
  });
  if (protocolSection) {
    protocolSection.dataset.mode = nextTab.id === "tab-strict" ? "strict" : "moderate";
  }
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const nextIndex = (index + direction + tabs.length) % tabs.length;
    selectTab(tabs[nextIndex]);
    tabs[nextIndex].focus();
  });
});

const copyButton = document.querySelector("[data-copy-code]");
const installCode = document.querySelector("[data-install-code]");
const overviewVideo = document.querySelector("[data-overview-video]");

overviewVideo?.addEventListener("ended", () => {
  overviewVideo.currentTime = 0;
  overviewVideo.pause();
});

copyButton?.addEventListener("click", async () => {
  if (!installCode) return;
  try {
    await navigator.clipboard.writeText(installCode.textContent.trim());
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy";
    }, 1600);
  } catch {
    copyButton.textContent = "Select text";
  }
});
