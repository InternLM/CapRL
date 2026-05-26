const state = {
  cases: [],
  activeIndex: 0,
  query: "",
};

const els = {
  caseCount: document.querySelector("#case-count"),
  caseList: document.querySelector("#case-list"),
  video: document.querySelector("#video"),
  title: document.querySelector("#case-title"),
  subtitle: document.querySelector("#case-subtitle"),
  caption: document.querySelector("#caption"),
  search: document.querySelector("#caption-search"),
  copy: document.querySelector("#copy-caption"),
};

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function appendHighlightedText(parent, text, query) {
  if (!query) {
    parent.append(document.createTextNode(text));
    return;
  }

  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  let cursor = 0;
  let matchIndex = lowerText.indexOf(lowerQuery, cursor);

  while (matchIndex !== -1) {
    if (matchIndex > cursor) {
      parent.append(document.createTextNode(text.slice(cursor, matchIndex)));
    }

    const mark = document.createElement("mark");
    mark.textContent = text.slice(matchIndex, matchIndex + query.length);
    parent.append(mark);

    cursor = matchIndex + query.length;
    matchIndex = lowerText.indexOf(lowerQuery, cursor);
  }

  if (cursor < text.length) {
    parent.append(document.createTextNode(text.slice(cursor)));
  }
}

function appendInline(parent, text) {
  const inlinePattern = /(\*\*.+?\*\*|\[\d{2}:\d{2}(?:\s*-\s*\d{2}:\d{2})?\])/g;
  let cursor = 0;
  let match;

  while ((match = inlinePattern.exec(text)) !== null) {
    if (match.index > cursor) {
      appendHighlightedText(parent, text.slice(cursor, match.index), state.query);
    }

    const token = match[0];
    if (token.startsWith("**")) {
      const strong = document.createElement("strong");
      appendHighlightedText(strong, token.slice(2, -2), state.query);
      parent.append(strong);
    } else {
      const timestamp = document.createElement("span");
      timestamp.className = "timestamp";
      timestamp.textContent = token;
      parent.append(timestamp);
    }

    cursor = inlinePattern.lastIndex;
  }

  if (cursor < text.length) {
    appendHighlightedText(parent, text.slice(cursor), state.query);
  }
}

function splitCaption(caption) {
  const normalized = caption.replace(/\r\n/g, "\n").trim();
  const sections = [];
  const headingPattern = /\*\*(Setting|Key Visuals|Sequence of Events):\*\*/g;
  const matches = [...normalized.matchAll(headingPattern)];

  if (!matches.length) {
    return [{ heading: "Caption", body: normalized }];
  }

  const intro = normalized.slice(0, matches[0].index).trim();
  if (intro) sections.push({ heading: "Overview", body: intro });

  matches.forEach((match, index) => {
    const start = match.index + match[0].length;
    const end = matches[index + 1]?.index ?? normalized.length;
    sections.push({
      heading: match[1],
      body: normalized.slice(start, end).trim(),
    });
  });

  return sections;
}

function splitKeyVisuals(body) {
  const matches = [...body.matchAll(/(?:^|\n)\s*-\s+\*\*(.+?):\*\*/g)];
  if (!matches.length) return body.split(/\n\s*\n/);

  const intro = body.slice(0, matches[0].index).trim();
  const blocks = intro ? [intro] : [];

  matches.forEach((match, index) => {
    const start = match.index;
    const end = matches[index + 1]?.index ?? body.length;
    blocks.push(body.slice(start, end).replace(/^\s*-\s+/, "").trim());
  });

  return blocks;
}

function splitSequence(body) {
  const compact = body.replace(/\n+/g, " ").replace(/\s+/g, " ").trim();
  const matches = [...compact.matchAll(/\[\d{2}:\d{2}(?:\s*-\s*\d{2}:\d{2})?\]/g)];
  if (!matches.length) return body.split(/\n\s*\n/);

  const blocks = [];
  const intro = compact.slice(0, matches[0].index).trim();
  if (intro) blocks.push(intro);

  matches.forEach((match, index) => {
    const start = match.index;
    const end = matches[index + 1]?.index ?? compact.length;
    blocks.push(compact.slice(start, end).trim());
  });

  return blocks;
}

function splitBlocks(section) {
  if (section.heading === "Key Visuals") return splitKeyVisuals(section.body);
  if (section.heading === "Sequence of Events") return splitSequence(section.body);
  return section.body.split(/\n\s*\n/);
}

function renderCaption(caption) {
  const sections = splitCaption(caption);
  els.caption.replaceChildren();

  sections.forEach((section) => {
    const sectionEl = document.createElement("section");
    sectionEl.className = "caption-section";

    const heading = document.createElement("h3");
    heading.className = "caption-heading";
    heading.textContent = section.heading;
    sectionEl.append(heading);

    const blocks = splitBlocks(section)
      .map((block) => block.trim())
      .filter(Boolean);

    if (!blocks.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "No caption text.";
      sectionEl.append(empty);
    }

    blocks.forEach((block) => {
      const paragraph = document.createElement("p");
      appendInline(paragraph, block);
      sectionEl.append(paragraph);
    });

    els.caption.append(sectionEl);
  });
}

function renderCaseList() {
  els.caseCount.textContent = String(state.cases.length);
  els.caseList.innerHTML = state.cases
    .map((item, index) => {
      const selected = index === state.activeIndex ? "true" : "false";
      return `
        <button class="case-button" type="button" aria-selected="${selected}" data-index="${index}">
          <strong>${escapeHtml(item.title || `Case ${index + 1}`)}</strong>
          <span>${escapeHtml(item.subtitle || "")}</span>
        </button>
      `;
    })
    .join("");
}

function selectCase(index) {
  state.activeIndex = index;
  const item = state.cases[index];
  els.video.src = `${item.video}?v=${encodeURIComponent(item.id || index)}`;
  els.video.load();
  els.video.poster = "";
  els.title.textContent = item.title || `Case ${index + 1}`;
  els.subtitle.textContent = item.subtitle || "";
  renderCaseList();
  renderCaption(item.caption || "");
}

async function loadCases() {
  const response = await fetch("cases.json");
  if (!response.ok) {
    throw new Error(`Failed to load cases.json: ${response.status}`);
  }
  state.cases = await response.json();
  if (!Array.isArray(state.cases) || !state.cases.length) {
    throw new Error("cases.json does not contain any demo cases.");
  }
  selectCase(0);
}

els.caseList.addEventListener("click", (event) => {
  const button = event.target.closest(".case-button");
  if (!button) return;
  selectCase(Number(button.dataset.index));
});

els.search.addEventListener("input", (event) => {
  state.query = event.target.value.trim();
  const item = state.cases[state.activeIndex];
  if (item) renderCaption(item.caption || "");
});

els.copy.addEventListener("click", async () => {
  const item = state.cases[state.activeIndex];
  if (!item) return;

  await navigator.clipboard.writeText(item.caption || "");
  els.copy.textContent = "Copied";
  els.copy.classList.add("is-copied");

  window.setTimeout(() => {
    els.copy.textContent = "Copy caption";
    els.copy.classList.remove("is-copied");
  }, 1400);
});

loadCases().catch((error) => {
  els.title.textContent = "Demo failed to load";
  els.subtitle.textContent = error.message;
  els.caption.innerHTML = `<p class="empty">${escapeHtml(error.message)}</p>`;
});
