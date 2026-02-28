package main

import (
	"bytes"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	defaultProvider = "ollama"
	defaultModel    = "llama3.2"
	ollamaURL       = "http://localhost:11434/api/generate"
	ollamaTagsURL   = "http://localhost:11434/api/tags"
	openaiURL       = "https://api.openai.com/v1/chat/completions"
	openaiModelsURL = "https://api.openai.com/v1/models"
	geminiBaseURL   = "https://generativelanguage.googleapis.com"
	maxPageChars    = 6000
	httpUA          = "LocalLLMAssistant/0.1 (+https://localhost)"
)

var requestSeq uint64
var modelCatalog = newModelCatalogStore()
var chats = newConversationStore(defaultStorePath())

type ChatRequest struct {
	Query          string `json:"query"`
	Provider       string `json:"provider"`
	Model          string `json:"model"`
	ConversationID string `json:"conversation_id"`
	UseLiveWeb     bool   `json:"use_live_web"`
	PageTitle      string `json:"page_title"`
	PageURL        string `json:"page_url"`
	PageContent    string `json:"page_content"`
}

type Citation struct {
	Title string `json:"title"`
	URL   string `json:"url"`
}

type ToolCall struct {
	Name   string `json:"name"`
	Input  string `json:"input"`
	Status string `json:"status"`
}

type ChatResponse struct {
	ConversationID string     `json:"conversation_id"`
	Answer         string     `json:"answer"`
	Citations      []Citation `json:"citations"`
	ToolCalls      []ToolCall `json:"tool_calls"`
}

type Conversation struct {
	ID           string `json:"id"`
	Title        string `json:"title"`
	Provider     string `json:"provider"`
	Model        string `json:"model"`
	CreatedAt    int64  `json:"created_at"`
	UpdatedAt    int64  `json:"updated_at"`
	MessageCount int    `json:"message_count"`
}

type StoredMessage struct {
	ID        string     `json:"id"`
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Provider  string     `json:"provider,omitempty"`
	Model     string     `json:"model,omitempty"`
	CreatedAt int64      `json:"created_at"`
	Citations []Citation `json:"citations,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ConversationRecord struct {
	Conversation Conversation    `json:"conversation"`
	Messages     []StoredMessage `json:"messages"`
}

type ConversationsListResponse struct {
	Conversations []Conversation `json:"conversations"`
}

type ConversationDetailResponse struct {
	Conversation Conversation    `json:"conversation"`
	Messages     []StoredMessage `json:"messages"`
}

type createConversationRequest struct {
	Title    string `json:"title"`
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

type conversationStoreData struct {
	Records map[string]*ConversationRecord `json:"records"`
}

type conversationStore struct {
	mu   sync.RWMutex
	path string
	data conversationStoreData
}

type ModelsResponse struct {
	Providers map[string]ProviderModels `json:"providers"`
}

type ProviderModels struct {
	Ready           bool     `json:"ready"`
	Models          []string `json:"models"`
	LastRefreshUnix int64    `json:"last_refresh_unix"`
	Error           string   `json:"error"`
}

type modelCatalogStore struct {
	mu        sync.RWMutex
	providers map[string]ProviderModels
}

type ollamaGenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type ollamaGenerateResponse struct {
	Response string `json:"response"`
}

type ollamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type openAIChatRequest struct {
	Model       string              `json:"model"`
	Messages    []openAIChatMessage `json:"messages"`
	Temperature float64             `json:"temperature"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message openAIChatMessage `json:"message"`
	} `json:"choices"`
}

type openAIModelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

type geminiGenerateRequest struct {
	Contents []geminiContent `json:"contents"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenerateResponse struct {
	Candidates []struct {
		Content geminiContent `json:"content"`
	} `json:"candidates"`
}

type geminiListModelsResponse struct {
	Models []struct {
		Name                       string   `json:"name"`
		SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
	} `json:"models"`
}

type duckDuckGoTopic struct {
	Text     string            `json:"Text"`
	FirstURL string            `json:"FirstURL"`
	Topics   []duckDuckGoTopic `json:"Topics"`
}

type duckDuckGoResponse struct {
	AbstractText  string            `json:"AbstractText"`
	AbstractURL   string            `json:"AbstractURL"`
	RelatedTopics []duckDuckGoTopic `json:"RelatedTopics"`
}

type wikipediaSearchResponse struct {
	Query struct {
		Search []struct {
			Title   string `json:"title"`
			Snippet string `json:"snippet"`
		} `json:"search"`
	} `json:"query"`
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	loadEnvFile()
	if err := chats.load(); err != nil {
		log.Printf("chat.store.load.fail error=%q", err.Error())
	} else {
		log.Printf("chat.store.load.ok path=%q", chats.path)
	}
	refreshModelCatalog()
	go startModelCatalogRefresher(10 * time.Minute)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/models", modelsHandler)
	mux.HandleFunc("/models/refresh", refreshModelsHandler)
	mux.HandleFunc("/conversations", conversationsHandler)
	mux.HandleFunc("/conversations/", conversationByIDHandler)
	mux.HandleFunc("/chat", chatHandler)

	h := loggingMiddleware(withCORS(mux))
	server := &http.Server{
		Addr:              ":8080",
		Handler:           h,
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Println("backend listening on http://localhost:8080")
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	writeJSON(w, http.StatusOK, ModelsResponse{Providers: modelCatalog.snapshot()})
}

func refreshModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}
	refreshModelCatalog()
	writeJSON(w, http.StatusOK, ModelsResponse{Providers: modelCatalog.snapshot()})
}

func conversationsHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, ConversationsListResponse{Conversations: chats.listConversations()})
	case http.MethodPost:
		var req createConversationRequest
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&req)
		}
		provider := normalizeProvider(req.Provider)
		if provider == "" {
			provider = defaultProvider
		}
		model := strings.TrimSpace(req.Model)
		if model == "" {
			model = resolveModel(provider, "")
		}
		title := strings.TrimSpace(req.Title)
		if title == "" {
			title = "New Chat"
		}
		id, err := chats.createConversation(title, provider, model)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to create conversation"})
			return
		}
		record, ok := chats.getConversation(id)
		if !ok {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to load created conversation"})
			return
		}
		writeJSON(w, http.StatusCreated, ConversationDetailResponse{
			Conversation: record.Conversation,
			Messages:     record.Messages,
		})
	case http.MethodDelete:
		if err := chats.clearAll(); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to clear conversations"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "cleared"})
	default:
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
	}
}

func conversationByIDHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimSpace(strings.TrimPrefix(r.URL.Path, "/conversations/"))
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "conversation id is required"})
		return
	}

	switch r.Method {
	case http.MethodGet:
		record, ok := chats.getConversation(id)
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "conversation not found"})
			return
		}
		writeJSON(w, http.StatusOK, ConversationDetailResponse{
			Conversation: record.Conversation,
			Messages:     record.Messages,
		})
	case http.MethodDelete:
		if err := chats.deleteConversation(id); err != nil {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
	default:
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
	}
}

func chatHandler(w http.ResponseWriter, r *http.Request) {
	reqID := nextRequestID()
	chatStart := time.Now()

	if r.Method != http.MethodPost {
		log.Printf("chat.reject request_id=%s reason=method_not_allowed method=%s", reqID, r.Method)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		log.Printf("chat.reject request_id=%s reason=invalid_json error=%q", reqID, err.Error())
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid json body"})
		return
	}

	req.Query = strings.TrimSpace(req.Query)
	if req.Query == "" {
		log.Printf("chat.reject request_id=%s reason=missing_query", reqID)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query is required"})
		return
	}

	requestedModel := strings.TrimSpace(req.Model)
	provider := normalizeProvider(req.Provider)
	if provider == "" {
		provider = defaultProvider
	}
	model := resolveModel(provider, requestedModel)
	conversationID := strings.TrimSpace(req.ConversationID)
	if conversationID == "" || !chats.exists(conversationID) {
		var createErr error
		conversationID, createErr = chats.createConversation(conversationTitleFromQuery(req.Query), provider, model)
		if createErr != nil {
			log.Printf("chat.store.create.fail request_id=%s error=%q", reqID, createErr.Error())
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to create conversation"})
			return
		}
		log.Printf("chat.store.create request_id=%s conversation_id=%s", reqID, conversationID)
	}
	log.Printf(
		"chat.start request_id=%s conversation_id=%s provider=%q requested_model=%q model=%q live_web=%t query_chars=%d page_chars=%d page_url=%q",
		reqID,
		conversationID,
		provider,
		requestedModel,
		model,
		req.UseLiveWeb,
		len(req.Query),
		len(req.PageContent),
		req.PageURL,
	)
	if err := chats.appendMessage(conversationID, StoredMessage{
		ID:        newID("msg"),
		Role:      "user",
		Content:   req.Query,
		Provider:  provider,
		Model:     model,
		CreatedAt: time.Now().Unix(),
	}); err != nil {
		log.Printf("chat.store.append_user.fail request_id=%s conversation_id=%s error=%q", reqID, conversationID, err.Error())
	}

	toolCalls := make([]ToolCall, 0, 3)
	citations := make([]Citation, 0, 5)
	knowledge := make([]string, 0, 8)

	if req.UseLiveWeb {
		toolCalls = append(toolCalls, ToolCall{Name: "web_search", Input: req.Query, Status: "running"})
		searchStart := time.Now()
		log.Printf("tool.start request_id=%s tool=web_search query=%q", reqID, truncateForLog(req.Query, 120))
		results, provider, err := webSearch(req.Query)
		if err != nil {
			toolCalls[len(toolCalls)-1].Status = "failed: " + shortError(err, 80)
			knowledge = append(knowledge, "Live web search failed. Continue with best effort using current context.")
			log.Printf("tool.fail request_id=%s tool=web_search duration_ms=%d error=%q", reqID, time.Since(searchStart).Milliseconds(), err.Error())
		} else {
			toolCalls[len(toolCalls)-1].Status = fmt.Sprintf("ok (%d results via %s)", len(results), provider)
			for _, r := range results {
				citations = append(citations, Citation{Title: r.Title, URL: r.URL})
				knowledge = append(knowledge, fmt.Sprintf("- %s\n  %s", r.Title, r.Snippet))
			}
			log.Printf("tool.done request_id=%s tool=web_search duration_ms=%d results=%d provider=%q", reqID, time.Since(searchStart).Milliseconds(), len(results), provider)
		}
	}

	prompt := buildPrompt(req, knowledge)
	log.Printf("model.start request_id=%s provider=%q model=%q prompt_chars=%d", reqID, provider, model, len(prompt))
	modelStart := time.Now()
	answer, effectiveModel, err := callModel(provider, model, prompt, reqID)
	if err != nil {
		log.Printf("model.fail request_id=%s provider=%q duration_ms=%d error=%q", reqID, provider, time.Since(modelStart).Milliseconds(), err.Error())
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "failed to call provider model: " + err.Error()})
		return
	}
	log.Printf("model.done request_id=%s provider=%q model=%q duration_ms=%d answer_chars=%d", reqID, provider, effectiveModel, time.Since(modelStart).Milliseconds(), len(strings.TrimSpace(answer)))

	resp := ChatResponse{
		ConversationID: conversationID,
		Answer:         strings.TrimSpace(answer),
		Citations:      dedupeCitations(citations),
		ToolCalls:      toolCalls,
	}
	if err := chats.appendMessage(conversationID, StoredMessage{
		ID:        newID("msg"),
		Role:      "assistant",
		Content:   resp.Answer,
		Provider:  provider,
		Model:     effectiveModel,
		CreatedAt: time.Now().Unix(),
		Citations: resp.Citations,
		ToolCalls: resp.ToolCalls,
	}); err != nil {
		log.Printf("chat.store.append_assistant.fail request_id=%s conversation_id=%s error=%q", reqID, conversationID, err.Error())
	}
	log.Printf(
		"chat.done request_id=%s duration_ms=%d citations=%d tool_calls=%d",
		reqID,
		time.Since(chatStart).Milliseconds(),
		len(resp.Citations),
		len(resp.ToolCalls),
	)
	writeJSON(w, http.StatusOK, resp)
}

type searchResult struct {
	Title   string
	URL     string
	Snippet string
}

func webSearch(query string) ([]searchResult, string, error) {
	normalized := normalizeSearchQuery(query)
	endpoint := "https://api.duckduckgo.com/?format=json&no_html=1&skip_disambig=1&q=" + urlQueryEscape(normalized)
	client := &http.Client{Timeout: 12 * time.Second}

	resp, err := doGET(client, endpoint)
	if err != nil {
		wikiResults, wikiErr := wikipediaSearch(normalized)
		if wikiErr != nil {
			return nil, "", fmt.Errorf("duckduckgo request failed (%v) and wikipedia fallback failed (%v)", err, wikiErr)
		}
		return wikiResults, "wikipedia", nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		wikiResults, wikiErr := wikipediaSearch(normalized)
		if wikiErr != nil {
			return nil, "", fmt.Errorf("duckduckgo status %d and wikipedia fallback failed (%v)", resp.StatusCode, wikiErr)
		}
		return wikiResults, "wikipedia", nil
	}

	var ddg duckDuckGoResponse
	if err := json.NewDecoder(resp.Body).Decode(&ddg); err != nil {
		wikiResults, wikiErr := wikipediaSearch(normalized)
		if wikiErr != nil {
			return nil, "", fmt.Errorf("duckduckgo decode failed (%v) and wikipedia fallback failed (%v)", err, wikiErr)
		}
		return wikiResults, "wikipedia", nil
	}

	results := make([]searchResult, 0, 5)
	if ddg.AbstractText != "" && ddg.AbstractURL != "" {
		results = append(results, searchResult{
			Title:   "Instant Answer",
			URL:     ddg.AbstractURL,
			Snippet: ddg.AbstractText,
		})
	}

	flattened := flattenTopics(ddg.RelatedTopics)
	for _, t := range flattened {
		if t.FirstURL == "" || t.Text == "" {
			continue
		}
		title := t.Text
		if idx := strings.Index(title, " - "); idx > 0 {
			title = title[:idx]
		}
		results = append(results, searchResult{
			Title:   title,
			URL:     t.FirstURL,
			Snippet: t.Text,
		})
		if len(results) >= 5 {
			break
		}
	}

	if len(results) >= 2 {
		return results[:min(5, len(results))], "duckduckgo", nil
	}

	wikiResults, wikiErr := wikipediaSearch(normalized)
	if wikiErr == nil {
		combined := mergeSearchResults(results, wikiResults, 5)
		if len(combined) > 0 {
			return combined, "duckduckgo+wikipedia", nil
		}
	}
	if len(results) > 0 {
		return results[:min(5, len(results))], "duckduckgo", nil
	}

	if wikiErr != nil {
		return nil, "", fmt.Errorf("no search results from duckduckgo and wikipedia (%v)", wikiErr)
	}
	return nil, "", fmt.Errorf("no search results")
}

func wikipediaSearch(query string) ([]searchResult, error) {
	endpoint := "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&srlimit=5&srsearch=" + urlQueryEscape(query)
	client := &http.Client{Timeout: 12 * time.Second}
	resp, err := doGET(client, endpoint)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("wikipedia status %d", resp.StatusCode)
	}

	var wiki wikipediaSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&wiki); err != nil {
		return nil, err
	}

	out := make([]searchResult, 0, 5)
	for _, item := range wiki.Query.Search {
		title := strings.TrimSpace(item.Title)
		if title == "" {
			continue
		}
		snippet := stripHTML(item.Snippet)
		out = append(out, searchResult{
			Title:   title,
			URL:     "https://en.wikipedia.org/wiki/" + strings.ReplaceAll(title, " ", "_"),
			Snippet: snippet,
		})
		if len(out) >= 5 {
			break
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("wikipedia returned no search results")
	}
	return out, nil
}

func flattenTopics(topics []duckDuckGoTopic) []duckDuckGoTopic {
	out := make([]duckDuckGoTopic, 0, len(topics))
	var walk func(items []duckDuckGoTopic)
	walk = func(items []duckDuckGoTopic) {
		for _, t := range items {
			if len(t.Topics) > 0 {
				walk(t.Topics)
				continue
			}
			out = append(out, t)
		}
	}
	walk(topics)
	return out
}

func buildPrompt(req ChatRequest, knowledge []string) string {
	pageText := strings.TrimSpace(req.PageContent)
	if len(pageText) > maxPageChars {
		pageText = pageText[:maxPageChars]
	}

	var b strings.Builder
	b.WriteString("You are a browser assistant. Answer clearly and factually. If web knowledge is provided, prioritize it and mention uncertainty when data may be incomplete.\\n\\n")
	b.WriteString("User question:\\n")
	b.WriteString(req.Query)
	b.WriteString("\\n\\n")

	if strings.TrimSpace(req.PageURL) != "" || strings.TrimSpace(req.PageTitle) != "" || pageText != "" {
		b.WriteString("Current page context:\\n")
		if req.PageTitle != "" {
			b.WriteString("- Title: ")
			b.WriteString(req.PageTitle)
			b.WriteString("\\n")
		}
		if req.PageURL != "" {
			b.WriteString("- URL: ")
			b.WriteString(req.PageURL)
			b.WriteString("\\n")
		}
		if pageText != "" {
			b.WriteString("- Page text excerpt:\\n")
			b.WriteString(pageText)
			b.WriteString("\\n")
		}
		b.WriteString("\\n")
	}

	if len(knowledge) > 0 {
		b.WriteString("Live web knowledge:\\n")
		for _, k := range knowledge {
			b.WriteString(k)
			b.WriteString("\\n")
		}
		b.WriteString("\\n")
	}

	b.WriteString("Return only the final answer text. Keep it concise.")
	return b.String()
}

func callOllama(model, prompt string) (string, error) {
	body, err := json.Marshal(ollamaGenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: false,
	})
	if err != nil {
		return "", err
	}

	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Post(ollamaURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return "", fmt.Errorf("ollama status %d: %s", resp.StatusCode, string(raw))
	}

	var out ollamaGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if strings.TrimSpace(out.Response) == "" {
		return "", fmt.Errorf("empty response from ollama")
	}
	return out.Response, nil
}

func callModel(provider, model, prompt, reqID string) (string, string, error) {
	switch provider {
	case "ollama":
		out, err := callOllama(model, prompt)
		return out, model, err
	case "openai":
		out, err := callOpenAI(model, prompt)
		return out, model, err
	case "gemini":
		return callGemini(model, prompt, reqID)
	default:
		return "", "", fmt.Errorf("unsupported provider %q; expected ollama, openai, or gemini", provider)
	}
}

func callOpenAI(model, prompt string) (string, error) {
	key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if key == "" {
		return "", errors.New("OPENAI_API_KEY is not set")
	}

	answer, err := callOpenAIChatOnce(model, prompt, key)
	if err == nil {
		return answer, nil
	}
	if !isOpenAINotChatModelError(err) {
		return "", err
	}

	// If a model was discovered but is not compatible with chat/completions,
	// retry once with a known chat-safe fallback.
	for _, candidate := range preferredOpenAIChatFallbacks(modelCatalog.getModels("openai")) {
		if candidate == model {
			continue
		}
		log.Printf("model.retry provider=openai from=%q to=%q reason=not_chat_model", model, candidate)
		retryAnswer, retryErr := callOpenAIChatOnce(candidate, prompt, key)
		if retryErr == nil {
			return retryAnswer, nil
		}
	}

	return "", err
}

func callOpenAIChatOnce(model, prompt, key string) (string, error) {
	body, err := json.Marshal(openAIChatRequest{
		Model: model,
		Messages: []openAIChatMessage{
			{Role: "user", Content: prompt},
		},
		Temperature: 0.2,
	})
	if err != nil {
		return "", err
	}

	client := &http.Client{Timeout: 90 * time.Second}
	req, err := http.NewRequest(http.MethodPost, openaiURL, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", httpUA)

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("openai status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out openAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 || strings.TrimSpace(out.Choices[0].Message.Content) == "" {
		return "", errors.New("empty response from openai")
	}
	return out.Choices[0].Message.Content, nil
}

func isOpenAINotChatModelError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "not a chat model") ||
		strings.Contains(msg, "not supported in the v1/chat/completions endpoint")
}

func callGemini(model, prompt, reqID string) (string, string, error) {
	key := strings.TrimSpace(os.Getenv("GEMINI_API_KEY"))
	if key == "" {
		return "", "", errors.New("GEMINI_API_KEY is not set")
	}
	if strings.TrimSpace(model) == "" {
		model = defaultModelForProvider("gemini")
	}
	primary := normalizeGeminiModelName(model)
	answer, err := callGeminiOnce(primary, prompt, key)
	if err == nil {
		return answer, primary, nil
	}
	if !isGeminiModelNotFound(err) {
		return "", primary, err
	}

	candidates := modelCatalog.getModels("gemini")
	for _, candidate := range candidates {
		nextModel := normalizeGeminiModelName(candidate)
		if nextModel == "" || nextModel == primary {
			continue
		}
		log.Printf("model.retry request_id=%s provider=gemini from=%q to=%q reason=model_not_found", reqID, primary, nextModel)
		retryAnswer, retryErr := callGeminiOnce(nextModel, prompt, key)
		if retryErr == nil {
			return retryAnswer, nextModel, nil
		}
	}

	suggestions := strings.Join(limitStrings(candidates, 5), ", ")
	if suggestions == "" {
		suggestions = "none discovered; call /models/refresh"
	}
	return "", primary, fmt.Errorf("gemini model %q unavailable; discovered models: %s", primary, suggestions)
}

func callGeminiOnce(model, prompt, key string) (string, error) {
	body, err := json.Marshal(geminiGenerateRequest{
		Contents: []geminiContent{
			{Parts: []geminiPart{{Text: prompt}}},
		},
	})
	if err != nil {
		return "", err
	}

	apiVersion := geminiAPIVersion()
	modelPath := url.PathEscape(normalizeGeminiModelName(model))
	endpoint := fmt.Sprintf("%s/%s/models/%s:generateContent?key=%s", geminiBaseURL, apiVersion, modelPath, url.QueryEscape(key))
	client := &http.Client{Timeout: 90 * time.Second}
	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", httpUA)

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("gemini status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out geminiGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Candidates) == 0 || len(out.Candidates[0].Content.Parts) == 0 || strings.TrimSpace(out.Candidates[0].Content.Parts[0].Text) == "" {
		return "", errors.New("empty response from gemini")
	}
	return out.Candidates[0].Content.Parts[0].Text, nil
}

func isGeminiModelNotFound(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "gemini status 404") || (strings.Contains(msg, "not_found") && strings.Contains(msg, "model"))
}

func dedupeCitations(in []Citation) []Citation {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(in))
	out := make([]Citation, 0, len(in))
	for _, c := range in {
		key := c.URL
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Title < out[j].Title })
	return out
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, GET, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		log.Printf(
			"http.request method=%s path=%s status=%d bytes=%d duration_ms=%d remote_addr=%q ua=%q",
			r.Method,
			r.URL.Path,
			rec.status,
			rec.bytes,
			time.Since(start).Milliseconds(),
			r.RemoteAddr,
			truncateForLog(r.UserAgent(), 120),
		)
	})
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

func urlQueryEscape(s string) string {
	re := regexp.MustCompile(`\s+`)
	return strings.ReplaceAll(re.ReplaceAllString(strings.TrimSpace(s), " "), " ", "+")
}

func doGET(client *http.Client, url string) (*http.Response, error) {
	return doGETWithHeaders(client, url, nil)
}

func doGETWithHeaders(client *http.Client, url string, headers map[string]string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", httpUA)
	req.Header.Set("Accept", "application/json, text/plain, */*")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	return client.Do(req)
}

func newModelCatalogStore() *modelCatalogStore {
	initial := map[string]ProviderModels{
		"ollama": emptyProviderModels(),
		"openai": emptyProviderModels(),
		"gemini": emptyProviderModels(),
	}
	return &modelCatalogStore{providers: initial}
}

func emptyProviderModels() ProviderModels {
	return ProviderModels{
		Ready:           false,
		Models:          nil,
		LastRefreshUnix: 0,
		Error:           "not loaded",
	}
}

func (m *modelCatalogStore) set(provider string, state ProviderModels) {
	m.mu.Lock()
	defer m.mu.Unlock()
	models := dedupeAndSortStrings(state.Models)
	state.Models = models
	m.providers[provider] = state
}

func (m *modelCatalogStore) snapshot() map[string]ProviderModels {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make(map[string]ProviderModels, len(m.providers))
	for provider, state := range m.providers {
		out[provider] = ProviderModels{
			Ready:           state.Ready,
			Models:          append([]string(nil), state.Models...),
			LastRefreshUnix: state.LastRefreshUnix,
			Error:           state.Error,
		}
	}
	return out
}

func (m *modelCatalogStore) getModels(provider string) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	state, ok := m.providers[provider]
	if !ok {
		return nil
	}
	return append([]string(nil), state.Models...)
}

func refreshModelCatalog() {
	refreshOneProvider("ollama")
	refreshOneProvider("openai")
	refreshOneProvider("gemini")
}

func startModelCatalogRefresher(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for range ticker.C {
		refreshModelCatalog()
	}
}

func refreshOneProvider(provider string) {
	log.Printf("models.refresh.start provider=%q", provider)
	start := time.Now()

	models, err := discoverProviderModels(provider)
	state := ProviderModels{
		Ready:           err == nil,
		Models:          models,
		LastRefreshUnix: time.Now().Unix(),
		Error:           "",
	}
	if err != nil {
		state.Error = shortError(err, 280)
	}
	modelCatalog.set(provider, state)
	if err != nil {
		log.Printf("models.refresh.fail provider=%q duration_ms=%d error=%q", provider, time.Since(start).Milliseconds(), err.Error())
		return
	}
	log.Printf("models.refresh.done provider=%q duration_ms=%d count=%d", provider, time.Since(start).Milliseconds(), len(models))
}

func discoverProviderModels(provider string) ([]string, error) {
	switch provider {
	case "ollama":
		return discoverOllamaModels()
	case "openai":
		return discoverOpenAIModels()
	case "gemini":
		return discoverGeminiModels()
	default:
		return nil, fmt.Errorf("unsupported provider %q", provider)
	}
}

func discoverOllamaModels() ([]string, error) {
	client := &http.Client{Timeout: 12 * time.Second}
	resp, err := doGET(client, ollamaTagsURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("ollama tags status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out ollamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	models := make([]string, 0, len(out.Models))
	for _, model := range out.Models {
		name := strings.TrimSpace(model.Name)
		if name != "" {
			models = append(models, name)
		}
	}
	models = dedupeAndSortStrings(models)
	if len(models) == 0 {
		return nil, errors.New("no ollama models installed")
	}
	return models, nil
}

func discoverOpenAIModels() ([]string, error) {
	key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))
	if key == "" {
		return nil, errors.New("OPENAI_API_KEY is not set")
	}

	client := &http.Client{Timeout: 12 * time.Second}
	resp, err := doGETWithHeaders(client, openaiModelsURL, map[string]string{
		"Authorization": "Bearer " + key,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("openai models status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out openAIModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	models := make([]string, 0, len(out.Data))
	for _, item := range out.Data {
		id := strings.TrimSpace(item.ID)
		if isLikelyOpenAIChatModel(id) {
			models = append(models, id)
		}
	}
	models = dedupeAndSortStrings(models)
	if len(models) == 0 {
		return nil, errors.New("no openai chat models discovered")
	}
	return models, nil
}

func discoverGeminiModels() ([]string, error) {
	key := strings.TrimSpace(os.Getenv("GEMINI_API_KEY"))
	if key == "" {
		return nil, errors.New("GEMINI_API_KEY is not set")
	}
	apiVersion := geminiAPIVersion()
	endpoint := fmt.Sprintf("%s/%s/models?key=%s", geminiBaseURL, apiVersion, url.QueryEscape(key))
	client := &http.Client{Timeout: 12 * time.Second}
	resp, err := doGET(client, endpoint)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("gemini list models status %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	var out geminiListModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	models := make([]string, 0, len(out.Models))
	for _, item := range out.Models {
		if !containsString(item.SupportedGenerationMethods, "generateContent") {
			continue
		}
		name := normalizeGeminiModelName(item.Name)
		if name != "" {
			models = append(models, name)
		}
	}
	models = dedupeAndSortStrings(models)
	if len(models) == 0 {
		return nil, errors.New("no gemini models supporting generateContent discovered")
	}
	return models, nil
}

func loadEnvFile() {
	candidates := []string{
		".env",
		"../.env",
	}

	for _, path := range candidates {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		lines := strings.Split(string(data), "\n")
		count := 0
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			kv := strings.SplitN(line, "=", 2)
			if len(kv) != 2 {
				continue
			}
			key := strings.TrimSpace(kv[0])
			val := strings.TrimSpace(kv[1])
			val = strings.Trim(val, `"'`)
			if key == "" {
				continue
			}
			if os.Getenv(key) != "" {
				continue
			}
			if err := os.Setenv(key, val); err == nil {
				count++
			}
		}
		log.Printf("env.load path=%q vars=%d", path, count)
		return
	}
	log.Printf("env.load path=none vars=0")
}

func defaultStorePath() string {
	if p := strings.TrimSpace(os.Getenv("CHAT_STORE_PATH")); p != "" {
		return p
	}
	return filepath.Join("..", "data", "chat_store.json")
}

func newConversationStore(path string) *conversationStore {
	return &conversationStore{
		path: path,
		data: conversationStoreData{
			Records: make(map[string]*ConversationRecord),
		},
	}
}

func (s *conversationStore) load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	raw, err := os.ReadFile(s.path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return s.saveLocked()
		}
		return err
	}
	if len(raw) == 0 {
		s.data.Records = make(map[string]*ConversationRecord)
		return nil
	}
	if err := json.Unmarshal(raw, &s.data); err != nil {
		return err
	}
	if s.data.Records == nil {
		s.data.Records = make(map[string]*ConversationRecord)
	}
	return nil
}

func (s *conversationStore) saveLocked() error {
	raw, err := json.MarshalIndent(s.data, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

func (s *conversationStore) createConversation(title, provider, model string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	id := newID("conv")
	now := time.Now().Unix()
	record := &ConversationRecord{
		Conversation: Conversation{
			ID:           id,
			Title:        strings.TrimSpace(title),
			Provider:     provider,
			Model:        model,
			CreatedAt:    now,
			UpdatedAt:    now,
			MessageCount: 0,
		},
		Messages: []StoredMessage{},
	}
	if record.Conversation.Title == "" {
		record.Conversation.Title = "New Chat"
	}
	s.data.Records[id] = record
	if err := s.saveLocked(); err != nil {
		delete(s.data.Records, id)
		return "", err
	}
	return id, nil
}

func (s *conversationStore) exists(id string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.data.Records[id]
	return ok
}

func (s *conversationStore) appendMessage(conversationID string, msg StoredMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	record, ok := s.data.Records[conversationID]
	if !ok {
		return fmt.Errorf("conversation %q not found", conversationID)
	}
	msg.Content = strings.TrimSpace(msg.Content)
	if msg.Content == "" {
		return nil
	}
	if msg.CreatedAt == 0 {
		msg.CreatedAt = time.Now().Unix()
	}
	record.Messages = append(record.Messages, msg)
	record.Conversation.UpdatedAt = msg.CreatedAt
	record.Conversation.MessageCount = len(record.Messages)
	if msg.Provider != "" {
		record.Conversation.Provider = msg.Provider
	}
	if msg.Model != "" {
		record.Conversation.Model = msg.Model
	}
	return s.saveLocked()
}

func (s *conversationStore) listConversations() []Conversation {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Conversation, 0, len(s.data.Records))
	for _, record := range s.data.Records {
		if record == nil {
			continue
		}
		out = append(out, record.Conversation)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].UpdatedAt > out[j].UpdatedAt
	})
	return out
}

func (s *conversationStore) getConversation(id string) (ConversationRecord, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	record, ok := s.data.Records[id]
	if !ok || record == nil {
		return ConversationRecord{}, false
	}
	clone := ConversationRecord{
		Conversation: record.Conversation,
		Messages:     append([]StoredMessage(nil), record.Messages...),
	}
	return clone, true
}

func (s *conversationStore) deleteConversation(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.data.Records[id]; !ok {
		return fmt.Errorf("conversation not found")
	}
	delete(s.data.Records, id)
	return s.saveLocked()
}

func (s *conversationStore) clearAll() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data.Records = make(map[string]*ConversationRecord)
	return s.saveLocked()
}

func conversationTitleFromQuery(query string) string {
	q := strings.TrimSpace(query)
	if q == "" {
		return "New Chat"
	}
	if len(q) > 60 {
		return q[:60] + "..."
	}
	return q
}

func newID(prefix string) string {
	var buf [6]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return fmt.Sprintf("%s_%x", prefix, buf[:])
}

func normalizeSearchQuery(q string) string {
	clean := strings.TrimSpace(strings.ToLower(q))
	clean = strings.NewReplacer("scren", "screen", "pls", "please", "u ", "you ").Replace(clean)
	if clean == "" {
		return q
	}
	return clean
}

func resolveModel(provider, requestedModel string) string {
	requestedModel = strings.TrimSpace(requestedModel)
	if requestedModel != "" {
		return requestedModel
	}
	discovered := modelCatalog.getModels(provider)
	if len(discovered) > 0 {
		return discovered[0]
	}
	return defaultModelForProvider(provider)
}

func normalizeProvider(in string) string {
	p := strings.ToLower(strings.TrimSpace(in))
	switch p {
	case "", "local", "ollama":
		return "ollama"
	case "openai":
		return "openai"
	case "gemini", "google":
		return "gemini"
	default:
		return p
	}
}

func normalizeGeminiModelName(model string) string {
	name := strings.TrimSpace(model)
	name = strings.TrimPrefix(name, "models/")
	return name
}

func geminiAPIVersion() string {
	version := strings.TrimSpace(os.Getenv("GEMINI_API_VERSION"))
	if version == "" {
		return "v1beta"
	}
	return strings.TrimPrefix(version, "/")
}

func containsString(items []string, target string) bool {
	for _, item := range items {
		if strings.EqualFold(strings.TrimSpace(item), target) {
			return true
		}
	}
	return false
}

func dedupeAndSortStrings(values []string) []string {
	seen := make(map[string]bool, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		v := strings.TrimSpace(value)
		if v == "" || seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

func limitStrings(values []string, limit int) []string {
	if limit <= 0 || len(values) == 0 {
		return nil
	}
	if len(values) <= limit {
		return values
	}
	return values[:limit]
}

func isLikelyOpenAIChatModel(id string) bool {
	lower := strings.ToLower(strings.TrimSpace(id))
	if lower == "" {
		return false
	}
	rejectContains := []string{
		"embedding",
		"transcribe",
		"whisper",
		"moderation",
		"tts",
		"speech",
		"realtime",
		"audio",
		"image",
		"vision-preview",
		"instruct",
	}
	for _, token := range rejectContains {
		if strings.Contains(lower, token) {
			return false
		}
	}
	return strings.HasPrefix(lower, "gpt-") || strings.HasPrefix(lower, "o1-") || strings.HasPrefix(lower, "o3-") || strings.HasPrefix(lower, "chatgpt-")
}

func preferredOpenAIChatFallbacks(discovered []string) []string {
	priority := []string{
		"gpt-4.1-mini",
		"gpt-4.1",
		"gpt-4o-mini",
		"gpt-4o",
	}
	out := make([]string, 0, len(priority)+len(discovered))
	seen := map[string]bool{}

	add := func(id string) {
		id = strings.TrimSpace(id)
		if id == "" || seen[id] {
			return
		}
		seen[id] = true
		out = append(out, id)
	}

	for _, id := range priority {
		add(id)
	}
	for _, id := range discovered {
		if isLikelyOpenAIChatModel(id) {
			add(id)
		}
	}
	return out
}

func defaultModelForProvider(provider string) string {
	switch provider {
	case "openai":
		return "gpt-4.1-mini"
	case "gemini":
		return "gemini-2.0-flash"
	default:
		return defaultModel
	}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (r *statusRecorder) WriteHeader(statusCode int) {
	r.status = statusCode
	r.ResponseWriter.WriteHeader(statusCode)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	n, err := r.ResponseWriter.Write(b)
	r.bytes += n
	return n, err
}

func nextRequestID() string {
	id := atomic.AddUint64(&requestSeq, 1)
	return "req-" + strconv.FormatUint(id, 10)
}

func truncateForLog(s string, max int) string {
	if max <= 0 {
		return ""
	}
	clean := strings.ReplaceAll(strings.ReplaceAll(strings.TrimSpace(s), "\n", " "), "\r", " ")
	if len(clean) <= max {
		return clean
	}
	return clean[:max] + "..."
}

func shortError(err error, max int) string {
	if err == nil {
		return ""
	}
	return truncateForLog(err.Error(), max)
}

func stripHTML(in string) string {
	tagRE := regexp.MustCompile(`<[^>]*>`)
	s := tagRE.ReplaceAllString(in, "")
	s = strings.ReplaceAll(s, "&quot;", "\"")
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&#39;", "'")
	return strings.TrimSpace(s)
}

func mergeSearchResults(primary []searchResult, secondary []searchResult, limit int) []searchResult {
	if limit <= 0 {
		return nil
	}
	out := make([]searchResult, 0, limit)
	seen := make(map[string]bool, limit)
	add := func(items []searchResult) {
		for _, item := range items {
			if len(out) >= limit {
				return
			}
			key := item.URL
			if key == "" || seen[key] {
				continue
			}
			seen[key] = true
			out = append(out, item)
		}
	}
	add(primary)
	add(secondary)
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
