const askBtn = document.getElementById("askBtn");
const queryInput = document.getElementById("query");
const conversationInput = document.getElementById("conversation");
const newChatBtn = document.getElementById("newChatBtn");
const deleteChatBtn = document.getElementById("deleteChatBtn");
const providerInput = document.getElementById("provider");
const modelInput = document.getElementById("model");
const refreshModelsBtn = document.getElementById("refreshModelsBtn");
const useLiveWebInput = document.getElementById("useLiveWeb");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const toolsEl = document.getElementById("tools");
const historyEl = document.getElementById("history");
const statusEl = document.getElementById("status");
const signalEl = document.querySelector(".signal");
const chipEls = document.querySelectorAll(".chip");

const BACKEND_CHAT_URL = "http://localhost:8080/chat";
const BACKEND_HEALTH_URL = "http://localhost:8080/health";
const BACKEND_MODELS_URL = "http://localhost:8080/models";
const BACKEND_REFRESH_MODELS_URL = "http://localhost:8080/models/refresh";
const BACKEND_CONVERSATIONS_URL = "http://localhost:8080/conversations";

const DEFAULT_MODELS = {
  ollama: "llama3.2",
  openai: "gpt-4.1-mini",
  gemini: "gemini-2.0-flash"
};

let modelCatalog = {
  ollama: { ready: false, models: [], error: "not loaded" },
  openai: { ready: false, models: [], error: "not loaded" },
  gemini: { ready: false, models: [], error: "not loaded" }
};

let conversations = [];
let activeConversationId = "";

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
  signalEl.className = `signal ${kind}`.trim();
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
    return true;
  } catch (_err) {
    setStatus("Backend not reachable on localhost:8080", "warn");
    return false;
  }
}

function providerDisplayName(provider) {
  if (provider === "ollama") return "Ollama";
  if (provider === "openai") return "OpenAI";
  if (provider === "gemini") return "Gemini";
  return provider;
}

function getProviderState(provider) {
  return modelCatalog[provider] || { ready: false, models: [], error: "not loaded" };
}

function populateModelOptions(provider, preferredModel = "") {
  const state = getProviderState(provider);
  const discovered = Array.isArray(state.models) ? state.models : [];
  const models = discovered.length > 0 ? discovered : [DEFAULT_MODELS[provider] || DEFAULT_MODELS.ollama];

  modelInput.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    modelInput.appendChild(option);
  }

  const pick = preferredModel && models.includes(preferredModel) ? preferredModel : models[0];
  modelInput.value = pick;

  if (!state.ready) {
    const reason = state.error ? ` (${state.error})` : "";
    setStatus(`${providerDisplayName(provider)} models not ready${reason}`, "warn");
  }
}

function applyModelsResponse(data, preserveCurrent = true) {
  const providers = (data && data.providers) || {};
  modelCatalog = {
    ollama: providers.ollama || { ready: false, models: [], error: "not loaded" },
    openai: providers.openai || { ready: false, models: [], error: "not loaded" },
    gemini: providers.gemini || { ready: false, models: [], error: "not loaded" }
  };

  const provider = providerInput.value || "ollama";
  const preferred = preserveCurrent ? modelInput.value : "";
  populateModelOptions(provider, preferred);
}

async function fetchModels(refresh = false) {
  const endpoint = refresh ? BACKEND_REFRESH_MODELS_URL : BACKEND_MODELS_URL;
  const method = refresh ? "POST" : "GET";
  const resp = await fetch(endpoint, { method });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`models endpoint ${resp.status}: ${body}`);
  }
  const data = await resp.json();
  applyModelsResponse(data, true);
}

async function refreshModels() {
  refreshModelsBtn.disabled = true;
  setStatus("Refreshing models...");
  try {
    await fetchModels(true);
    setStatus("Models refreshed.", "ok");
  } catch (err) {
    setStatus(`Model refresh failed: ${err.message}`, "warn");
  } finally {
    refreshModelsBtn.disabled = false;
  }
}

function formatConversationLabel(conv) {
  const title = conv.title || "Untitled";
  const suffix = `${conv.provider || ""}/${conv.model || ""}`;
  return suffix.trim() ? `${title} (${suffix})` : title;
}

function renderConversations(preferredId = "") {
  conversationInput.innerHTML = "";

  if (!conversations || conversations.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No conversations";
    conversationInput.appendChild(option);
    activeConversationId = "";
    return;
  }

  for (const conv of conversations) {
    const option = document.createElement("option");
    option.value = conv.id;
    option.textContent = formatConversationLabel(conv);
    conversationInput.appendChild(option);
  }

  const existing = conversations.some((c) => c.id === preferredId) ? preferredId : conversations[0].id;
  activeConversationId = existing;
  conversationInput.value = existing;
}

async function fetchConversations(preferredId = "") {
  const resp = await fetch(BACKEND_CONVERSATIONS_URL);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`conversations ${resp.status}: ${body}`);
  }
  const data = await resp.json();
  conversations = data.conversations || [];
  renderConversations(preferredId || activeConversationId);
}

function renderHistory(messages) {
  historyEl.innerHTML = "";
  if (!messages || messages.length === 0) {
    setEmptyList(historyEl, "No messages yet");
    answerEl.textContent = "No response yet.";
    return;
  }

  let lastAssistant = "";
  for (const msg of messages) {
    const li = document.createElement("li");
    const role = (msg.role || "unknown").toUpperCase();
    const content = msg.content || "";
    li.textContent = `${role}: ${content}`;
    historyEl.appendChild(li);

    if (msg.role === "assistant") {
      lastAssistant = content;
    }
  }
  answerEl.textContent = lastAssistant || "No assistant response yet.";
}

async function loadConversation(id) {
  if (!id) {
    renderHistory([]);
    setEmptyList(sourcesEl, "No sources yet");
    setEmptyList(toolsEl, "No tool calls yet");
    return;
  }

  const resp = await fetch(`${BACKEND_CONVERSATIONS_URL}/${encodeURIComponent(id)}`);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`conversation ${resp.status}: ${body}`);
  }

  const data = await resp.json();
  const conv = data.conversation || {};
  const messages = data.messages || [];
  if (conv.provider) {
    providerInput.value = conv.provider;
    populateModelOptions(conv.provider, conv.model || "");
  }
  renderHistory(messages);

  let lastAssistant = null;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i].role === "assistant") {
      lastAssistant = messages[i];
      break;
    }
  }
  renderSources((lastAssistant && lastAssistant.citations) || []);
  renderToolCalls((lastAssistant && lastAssistant.tool_calls) || []);
}

async function createConversation() {
  const provider = providerInput.value || "ollama";
  const model = modelInput.value || DEFAULT_MODELS[provider] || DEFAULT_MODELS.ollama;
  const payload = {
    title: "New Chat",
    provider,
    model
  };

  const resp = await fetch(BACKEND_CONVERSATIONS_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`create conversation ${resp.status}: ${body}`);
  }

  const data = await resp.json();
  const id = data.conversation?.id || "";
  await fetchConversations(id);
  await loadConversation(id);
  setStatus("New conversation created.", "ok");
}

async function deleteConversation(id) {
  if (!id) {
    return;
  }
  const resp = await fetch(`${BACKEND_CONVERSATIONS_URL}/${encodeURIComponent(id)}`, {
    method: "DELETE"
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`delete conversation ${resp.status}: ${body}`);
  }
  await fetchConversations("");
  if (activeConversationId) {
    await loadConversation(activeConversationId);
  } else {
    renderHistory([]);
  }
  setStatus("Conversation deleted.", "ok");
}

async function bootstrapBackendState() {
  const healthy = await checkBackendHealth();
  if (!healthy) {
    populateModelOptions(providerInput.value || "ollama");
    renderConversations("");
    renderHistory([]);
    return;
  }

  try {
    await fetchModels(false);
  } catch (err) {
    setStatus(`Model list unavailable: ${err.message}`, "warn");
    populateModelOptions(providerInput.value || "ollama");
  }

  try {
    await fetchConversations("");
    if (activeConversationId) {
      await loadConversation(activeConversationId);
    } else {
      renderHistory([]);
    }
    setStatus("Backend connected.", "ok");
  } catch (err) {
    setStatus(`Conversation load failed: ${err.message}`, "warn");
    renderHistory([]);
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
    const provider = providerInput.value || "ollama";
    const payload = {
      query,
      provider,
      model: modelInput.value || DEFAULT_MODELS[provider] || DEFAULT_MODELS.ollama,
      conversation_id: activeConversationId || "",
      use_live_web: useLiveWebInput.checked,
      page_title: context.title || "",
      page_url: context.url || "",
      page_content: context.content || ""
    };

    setStatus("Calling backend...");
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
    activeConversationId = data.conversation_id || activeConversationId;
    answerEl.textContent = data.answer || "No answer generated.";
    renderSources(data.citations || []);
    renderToolCalls(data.tool_calls || []);

    await fetchConversations(activeConversationId);
    if (activeConversationId) {
      await loadConversation(activeConversationId);
    }

    queryInput.value = "";
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
refreshModelsBtn.addEventListener("click", refreshModels);
newChatBtn.addEventListener("click", async () => {
  try {
    await createConversation();
  } catch (err) {
    setStatus(`Create chat failed: ${err.message}`, "warn");
  }
});
deleteChatBtn.addEventListener("click", async () => {
  try {
    await deleteConversation(activeConversationId);
  } catch (err) {
    setStatus(`Delete chat failed: ${err.message}`, "warn");
  }
});

conversationInput.addEventListener("change", async () => {
  activeConversationId = conversationInput.value || "";
  try {
    await loadConversation(activeConversationId);
  } catch (err) {
    setStatus(`Load chat failed: ${err.message}`, "warn");
  }
});

queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    askQuestion();
  }
});

providerInput.addEventListener("change", () => {
  const provider = providerInput.value || "ollama";
  populateModelOptions(provider);
});

for (const chip of chipEls) {
  chip.addEventListener("click", () => {
    queryInput.value = chip.dataset.template || "";
    queryInput.focus();
  });
}

setEmptyList(sourcesEl, "No sources yet");
setEmptyList(toolsEl, "No tool calls yet");
setEmptyList(historyEl, "No messages yet");
populateModelOptions("ollama");
bootstrapBackendState();
