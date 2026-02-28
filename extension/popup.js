const askBtn = document.getElementById("askBtn");
const queryInput = document.getElementById("query");
const modelInput = document.getElementById("model");
const useLiveWebInput = document.getElementById("useLiveWeb");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const toolsEl = document.getElementById("tools");
const statusEl = document.getElementById("status");

const BACKEND_CHAT_URL = "http://localhost:8080/chat";
const BACKEND_HEALTH_URL = "http://localhost:8080/health";

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

function setEmptyList(container, label) {
  container.innerHTML = "";
  const li = document.createElement("li");
  li.textContent = label;
  container.appendChild(li);
}

async function checkBackendHealth() {
  try {
    const resp = await fetch(BACKEND_HEALTH_URL);
    if (!resp.ok) {
      throw new Error(`health status ${resp.status}`);
    }
    setStatus("Backend connected.", "ok");
  } catch (_err) {
    setStatus("Backend not reachable on localhost:8080", "warn");
  }
}

async function getActiveTabContext() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  if (!tab || !tab.id) {
    return { title: "", url: "", content: "" };
  }

  try {
    const response = await chrome.tabs.sendMessage(tab.id, { type: "GET_PAGE_CONTEXT" });
    if (!response) {
      return { title: tab.title || "", url: tab.url || "", content: "" };
    }
    return response;
  } catch (_err) {
    return { title: tab.title || "", url: tab.url || "", content: "" };
  }
}

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  if (!sources || sources.length === 0) {
    setEmptyList(sourcesEl, "No sources returned");
    return;
  }

  for (const source of sources) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = source.url;
    a.target = "_blank";
    a.rel = "noreferrer";
    a.textContent = source.title || source.url;
    li.appendChild(a);
    sourcesEl.appendChild(li);
  }
}

function renderToolCalls(toolCalls) {
  toolsEl.innerHTML = "";
  if (!toolCalls || toolCalls.length === 0) {
    setEmptyList(toolsEl, "No tool calls");
    return;
  }

  for (const tool of toolCalls) {
    const li = document.createElement("li");
    li.textContent = `${tool.name}: ${tool.status}`;
    toolsEl.appendChild(li);
  }
}

async function askQuestion() {
  const query = queryInput.value.trim();
  if (!query) {
    setStatus("Enter a question first.", "warn");
    answerEl.textContent = "Please enter a question.";
    return;
  }

  askBtn.disabled = true;
  setStatus("Collecting page context...");
  answerEl.textContent = "Thinking...";
  setEmptyList(sourcesEl, "Searching...");
  setEmptyList(toolsEl, "Running tools...");

  try {
    const context = await getActiveTabContext();
    const payload = {
      query,
      model: modelInput.value.trim() || "llama3.2",
      use_live_web: useLiveWebInput.checked,
      page_title: context.title || "",
      page_url: context.url || "",
      page_content: context.content || ""
    };

    setStatus("Calling local backend...");
    const resp = await fetch(BACKEND_CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      const errText = await resp.text();
      throw new Error(`Backend error ${resp.status}: ${errText}`);
    }

    const data = await resp.json();
    answerEl.textContent = data.answer || "No answer generated.";
    renderSources(data.citations || []);
    renderToolCalls(data.tool_calls || []);
    setStatus("Completed.", "ok");
  } catch (err) {
    answerEl.textContent = `Request failed: ${err.message}`;
    setStatus("Request failed. Check backend logs.", "warn");
    setEmptyList(sourcesEl, "No sources returned");
    setEmptyList(toolsEl, "No tool calls");
  } finally {
    askBtn.disabled = false;
  }
}

askBtn.addEventListener("click", askQuestion);
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    askQuestion();
  }
});

checkBackendHealth();
setEmptyList(sourcesEl, "No sources yet");
setEmptyList(toolsEl, "No tool calls yet");
