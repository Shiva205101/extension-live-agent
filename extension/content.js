function getPageText(maxChars = 6000) {
  const bodyText = document.body ? document.body.innerText || "" : "";
  return bodyText.trim().slice(0, maxChars);
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "GET_PAGE_CONTEXT") {
    return;
  }

  sendResponse({
    title: document.title || "",
    url: window.location.href || "",
    content: getPageText()
  });
});
