/**
 * OpenClaw — AI Personal Assistant Agent
 *
 * Simulates a real AI assistant workload running on AGOS:
 * - Processes user queries in a loop
 * - Consumes memory for conversation context
 * - Makes (simulated) API calls with token tracking
 * - Writes temp files for caching
 * - Reports resource usage to stdout for OS monitoring
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// ── Configuration ──────────────────────────────────────────────
const CYCLE_INTERVAL_MS = 10000;  // Process a "request" every 10s
const CONTEXT_WINDOW = 4096;
const MAX_CONVERSATIONS = 50;

// ── State ──────────────────────────────────────────────────────
let totalTokens = 0;
let requestCount = 0;
let conversations = new Map();
let cacheDir = '/tmp/openclaw_cache';

// Ensure cache dir
try { fs.mkdirSync(cacheDir, { recursive: true }); } catch(e) {}

// ── Simulated query topics ─────────────────────────────────────
const QUERIES = [
    "Summarize the latest AI research papers on multi-agent coordination",
    "Write a Python function to implement exponential backoff retry logic",
    "Explain the differences between RAG and fine-tuning approaches",
    "Debug this error: TypeError: Cannot read property 'map' of undefined",
    "Create a Kubernetes deployment manifest for a microservice",
    "Review this PR: changes to the authentication middleware",
    "Analyze system logs for anomalous memory consumption patterns",
    "Generate test cases for the payment processing module",
    "Optimize this SQL query that's causing a full table scan",
    "Design an event-driven architecture for real-time notifications",
];

// ── Simulated AI processing ────────────────────────────────────
function simulateTokenUsage(query) {
    // Rough token estimation: ~1.3 tokens per word
    const inputTokens = Math.ceil(query.split(' ').length * 1.3);
    // Response is typically 2-5x the input
    const outputTokens = inputTokens * (2 + Math.random() * 3);
    return { input: Math.ceil(inputTokens), output: Math.ceil(outputTokens) };
}

function generateResponse(query) {
    // Simulate processing time variability
    const words = 50 + Math.floor(Math.random() * 200);
    const chars = 'abcdefghijklmnopqrstuvwxyz ';
    let response = '';
    for (let i = 0; i < words; i++) {
        const wordLen = 3 + Math.floor(Math.random() * 8);
        let word = '';
        for (let j = 0; j < wordLen; j++) {
            word += chars[Math.floor(Math.random() * chars.length)];
        }
        response += word + ' ';
    }
    return response.trim();
}

// ── Conversation management ────────────────────────────────────
function getOrCreateConversation(id) {
    if (!conversations.has(id)) {
        conversations.set(id, {
            id: id,
            messages: [],
            tokenCount: 0,
            createdAt: Date.now(),
        });
    }
    // Evict old conversations if over limit
    if (conversations.size > MAX_CONVERSATIONS) {
        const oldest = [...conversations.entries()]
            .sort((a, b) => a[1].createdAt - b[1].createdAt)[0];
        conversations.delete(oldest[0]);
    }
    return conversations.get(id);
}

// ── File caching ───────────────────────────────────────────────
function cacheResponse(query, response) {
    const hash = crypto.createHash('md5').update(query).digest('hex');
    const cachePath = path.join(cacheDir, `${hash}.json`);
    try {
        fs.writeFileSync(cachePath, JSON.stringify({
            query: query,
            response: response.substring(0, 500),
            timestamp: Date.now(),
        }));
    } catch(e) {
        console.error(`[openclaw] cache write error: ${e.message}`);
    }
}

// ── Main processing loop ───────────────────────────────────────
function processRequest() {
    requestCount++;
    const query = QUERIES[requestCount % QUERIES.length];
    const convId = `conv-${Math.floor(requestCount / 3)}`;  // Group 3 requests per conversation

    const conv = getOrCreateConversation(convId);
    const tokens = simulateTokenUsage(query);
    totalTokens += tokens.input + tokens.output;
    conv.tokenCount += tokens.input + tokens.output;

    // Add to conversation context
    conv.messages.push({ role: 'user', content: query });
    const response = generateResponse(query);
    conv.messages.push({ role: 'assistant', content: response });

    // Trim conversation if too long
    while (conv.messages.length > 20) {
        conv.messages.shift();
    }

    // Cache the response
    cacheResponse(query, response);

    // Report to stdout (AGOS monitors this)
    const memUsage = process.memoryUsage();
    console.log(JSON.stringify({
        agent: 'openclaw',
        event: 'request_processed',
        request_id: requestCount,
        query: query.substring(0, 80),
        tokens_input: tokens.input,
        tokens_output: tokens.output,
        tokens_total: totalTokens,
        conversations_active: conversations.size,
        memory_rss_mb: Math.round(memUsage.rss / 1024 / 1024 * 10) / 10,
        memory_heap_mb: Math.round(memUsage.heapUsed / 1024 / 1024 * 10) / 10,
        cache_files: fs.readdirSync(cacheDir).length,
        uptime_s: Math.round(process.uptime()),
    }));
}

// ── Startup ────────────────────────────────────────────────────
console.log(JSON.stringify({
    agent: 'openclaw',
    event: 'started',
    pid: process.pid,
    node_version: process.version,
    description: 'AI Personal Assistant Agent',
}));

// Process requests on interval
const timer = setInterval(processRequest, CYCLE_INTERVAL_MS);

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log(JSON.stringify({
        agent: 'openclaw',
        event: 'shutting_down',
        total_requests: requestCount,
        total_tokens: totalTokens,
    }));
    clearInterval(timer);
    process.exit(0);
});

process.on('SIGINT', () => {
    clearInterval(timer);
    process.exit(0);
});
