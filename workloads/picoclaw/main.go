// PicoClaw — Ultra-lightweight AI agent
//
// Simulates a minimal-footprint Go agent workload on AGOS:
// - Low memory (<10MB target)
// - Fast request processing
// - Periodic health checks
// - Token budget tracking
// - File I/O for persistent state

package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"
)

type AgentEvent struct {
	Agent       string  `json:"agent"`
	Event       string  `json:"event"`
	RequestID   int     `json:"request_id,omitempty"`
	Task        string  `json:"task,omitempty"`
	TokensUsed  int     `json:"tokens_used,omitempty"`
	TokensTotal int     `json:"tokens_total"`
	MemoryMB    float64 `json:"memory_mb"`
	Goroutines  int     `json:"goroutines"`
	UptimeS     int     `json:"uptime_s"`
	CacheHits   int     `json:"cache_hits,omitempty"`
	CacheMisses int     `json:"cache_misses,omitempty"`
}

var (
	totalTokens  int
	requestCount int
	cacheHits    int
	cacheMisses  int
	startTime    = time.Now()
	cache        = make(map[string]string)
)

// Tasks this lightweight agent handles
var tasks = []string{
	"classify_intent:user_query",
	"extract_entities:document",
	"sentiment_analysis:feedback",
	"summarize:paragraph",
	"translate:phrase",
	"spell_check:text",
	"keyword_extract:article",
	"topic_detect:message",
}

func getMemoryMB() float64 {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.Alloc) / 1024 / 1024
}

func emit(event AgentEvent) {
	event.MemoryMB = float64(int(getMemoryMB()*10)) / 10
	event.Goroutines = runtime.NumGoroutine()
	event.UptimeS = int(time.Since(startTime).Seconds())
	event.TokensTotal = totalTokens
	data, _ := json.Marshal(event)
	fmt.Println(string(data))
}

func processTask() {
	requestCount++
	task := tasks[requestCount%len(tasks)]

	// Simulate token usage (lightweight: 10-50 tokens per request)
	tokens := 10 + rand.Intn(40)
	totalTokens += tokens

	// Check cache
	key := fmt.Sprintf("%x", sha256.Sum256([]byte(task)))[:16]
	if _, ok := cache[key]; ok {
		cacheHits++
	} else {
		cacheMisses++
		// Simulate processing result
		result := fmt.Sprintf("result_%d_%s", requestCount, task[:10])
		cache[key] = result

		// Evict old cache entries (keep it small)
		if len(cache) > 100 {
			for k := range cache {
				delete(cache, k)
				break
			}
		}
	}

	// Write state to disk periodically
	if requestCount%10 == 0 {
		state := map[string]interface{}{
			"requests":    requestCount,
			"tokens":      totalTokens,
			"cache_size":  len(cache),
			"cache_hits":  cacheHits,
			"cache_ratio": float64(cacheHits) / float64(cacheHits+cacheMisses+1),
		}
		data, _ := json.Marshal(state)
		os.WriteFile("/tmp/picoclaw_state.json", data, 0644)
	}

	emit(AgentEvent{
		Agent:       "picoclaw",
		Event:       "task_completed",
		RequestID:   requestCount,
		Task:        task,
		TokensUsed:  tokens,
		CacheHits:   cacheHits,
		CacheMisses: cacheMisses,
	})
}

func main() {
	emit(AgentEvent{
		Agent: "picoclaw",
		Event: "started",
	})

	// Process tasks every 8 seconds
	ticker := time.NewTicker(8 * time.Second)
	defer ticker.Stop()

	// Graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)

	for {
		select {
		case <-ticker.C:
			processTask()
		case <-sigChan:
			emit(AgentEvent{
				Agent: "picoclaw",
				Event: "shutting_down",
			})
			return
		}
	}
}
