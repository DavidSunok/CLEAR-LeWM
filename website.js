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
const communityResults = document.querySelector("[data-community-results]");

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

const taskLabels = {
  pusht: "PushT",
  cube: "Cube",
  reacher: "Reacher",
  tworoom: "TwoRoom",
};

function formatRate(value) {
  return Number.isInteger(value) ? `${value}%` : `${value.toFixed(2)}%`;
}

function resultCell(result) {
  const cell = document.createElement("td");
  cell.className = "community-score";
  if (!result) {
    cell.textContent = "Not submitted";
    return cell;
  }
  cell.append(
    `${formatRate(result.success_rate_percent)} / ${formatRate(result.random_success_rate_percent)}`,
  );
  const excess = document.createElement("small");
  const value = result.excess_over_random_pp;
  excess.textContent = `${value > 0 ? "+" : ""}${value} pp`;
  excess.classList.toggle("is-negative", value < 0);
  cell.append(excess);
  return cell;
}

function communityCard(submission) {
  const card = document.createElement("article");
  card.className = "community-card";
  card.dataset.submission = submission.bundle;

  const head = document.createElement("div");
  head.className = "community-card-head";
  const identity = document.createElement("div");
  const title = document.createElement("h3");
  title.textContent = submission.method;
  const meta = document.createElement("p");
  meta.className = "community-card-meta";
  const coveredTasks = new Set(
    submission.results.map((result) => result.task),
  ).size;
  meta.textContent = `${submission.training_data_track} · ${coveredTasks}/4 tasks`;
  identity.append(title, meta);

  const links = document.createElement("div");
  links.className = "community-card-links";
  const cardLink = document.createElement("a");
  cardLink.href = `https://github.com/DavidSunok/CLEAR-LeWM/blob/main/${submission.method_card}`;
  cardLink.textContent = "Method card";
  const sourceLink = document.createElement("a");
  sourceLink.href = submission.method_repository;
  sourceLink.textContent = "Source";
  links.append(cardLink, sourceLink);
  head.append(identity, links);

  const status = document.createElement("p");
  status.className = "community-status";
  status.textContent = `CI-valid bundle · ${submission.verification} execution · @${submission.contact_github}`;

  const table = document.createElement("table");
  table.className = "community-table";
  const thead = document.createElement("thead");
  const headingRow = document.createElement("tr");
  ["Task", "Moderate model / random", "Strict model / random"].forEach(
    (label) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      headingRow.append(cell);
    },
  );
  thead.append(headingRow);
  const tbody = document.createElement("tbody");
  const byKey = new Map(
    submission.results.map((result) => [
      `${result.task}/${result.protocol}`,
      result,
    ]),
  );
  Object.entries(taskLabels).forEach(([task, label]) => {
    const moderate = byKey.get(`${task}/moderate`);
    const strict = byKey.get(`${task}/strict`);
    if (!moderate && !strict) return;
    const row = document.createElement("tr");
    const taskCell = document.createElement("th");
    taskCell.scope = "row";
    taskCell.textContent = label;
    row.append(taskCell, resultCell(moderate), resultCell(strict));
    tbody.append(row);
  });
  table.append(thead, tbody);

  card.append(head, status, table);
  if (submission.missing_results.length) {
    const missing = document.createElement("p");
    missing.className = "community-missing";
    const tasks = [
      ...new Set(
        submission.missing_results.map(
          (item) => taskLabels[item.split("/")[0]],
        ),
      ),
    ];
    missing.textContent = `Not submitted: ${tasks.join(", ")}.`;
    card.append(missing);
  }
  return card;
}

async function loadCommunityResults() {
  if (!communityResults) return;
  try {
    const response = await fetch("submissions/leaderboard.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const registry = await response.json();
    communityResults.replaceChildren(
      ...registry.submissions.map(communityCard),
    );
  } catch {
    const message = document.createElement("p");
    message.className = "community-error";
    message.textContent =
      "Community results could not be loaded. Open the submission registry on GitHub for the complete records.";
    communityResults.replaceChildren(message);
  }
}

loadCommunityResults();
