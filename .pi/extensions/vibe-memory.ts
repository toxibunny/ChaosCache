import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const MEMORY_MARKER = "vibe-memory:chaoscache memories";
const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = resolve(EXTENSION_DIR, "..");

// Use process.env for Node.js (pi runs in Node.js) or Deno.env for Deno
const env = (typeof process !== "undefined" ? process.env : (typeof Deno !== "undefined" ? Deno.env : {})) as Record<string, string | undefined>;

// Configuration — override with environment variables
const NEO4J_URL = env["CHAOSCACHE_NEO4J_URL"] ?? "bolt://localhost:7687";
const NEO4J_USER = env["CHAOSCACHE_NEO4J_USER"] ?? "neo4j";
const NEO4J_PASSWORD = env["CHAOSCACHE_NEO4J_PASSWORD"] ?? "password";
// Can be a local model path OR a llama.cpp server URL (http://...)
const MODEL_PATH = env["CHAOSCACHE_MODEL_PATH"] ?? "";
const SERENDIPITY = parseFloat(env["CHAOSCACHE_SERENDIPITY"] ?? "0.15");
const MAX_MEMORIES = parseInt(env["CHAOSCACHE_MAX_MEMORIES"] ?? "5");

let cachedMemories: string | null | undefined;
let lastQueryTime = 0;
const QUERY_CACHE_TTL = 60_000; // 1 minute cache

export default function vibeMemoryExtension(pi: ExtensionAPI) {
	let sessionActive = false;

	pi.on("resources_discover", async () => ({
		skillPaths: [],
	}));

	pi.on("session_start", async () => {
		sessionActive = true;
		cachedMemories = undefined;
	});

	pi.on("session_compact", async () => {
		cachedMemories = undefined;
	});

	pi.on("agent_end", async () => {
		sessionActive = false;
	});

	pi.on("context", async (event) => {
		if (!sessionActive) return;
		if (event.messages.some(messageContainsMarker)) return;

		// Extract context from recent messages
		const recentMessages = extractRecentMessages(event.messages);
		if (recentMessages.length === 0) return;

		// Query for relevant memories
		const memories = await queryMemories(recentMessages);
		if (!memories || memories.length === 0) return;

		// Format memories into a message
		const memoryText = formatMemories(memories);
		const memoryMessage = {
			role: "user" as const,
			content: [{ type: "text" as const, text: memoryText }],
			timestamp: Date.now(),
		};

		const insertAt = firstNonCompactionSummaryIndex(event.messages);
		return {
			messages: [
				...event.messages.slice(0, insertAt),
				memoryMessage,
				...event.messages.slice(insertAt),
			],
		};
	});
}

function extractRecentMessages(messages: unknown[]): Array<{ role: string; content: string }> {
	const recent: Array<{ role: string; content: string }> = [];
	const maxMessages = 10;

	for (const message of messages.slice(-maxMessages)) {
		const msg = message as { role?: string; content?: unknown };
		if (!msg.role || msg.role === "compactionSummary") continue;

		let content = "";
		if (typeof msg.content === "string") {
			content = msg.content;
		} else if (Array.isArray(msg.content)) {
			content = msg.content
				.filter((part: unknown) => part && typeof part === "object" && (part as { type?: string }).type === "text")
				.map((part: unknown) => (part as { text?: string }).text ?? "")
				.join("\n");
		}

		if (content.trim()) {
			recent.push({ role: msg.role, content: content.trim() });
		}
	}

	return recent;
}

async function queryMemories(recentMessages: Array<{ role: string; content: string }>): Promise<MemoryResult[] | null> {
	// Check cache
	const now = Date.now();
	if (cachedMemories !== undefined && now - lastQueryTime < QUERY_CACHE_TTL) {
		return cachedMemories ? JSON.parse(cachedMemories) : null;
	}

	try {
		// Try to use the Python script for querying
		const context = recentMessages.map(m => `${m.role}: ${m.content.slice(0, 200)}`).join("\n");
		const result = await queryNeo4j(context);
		lastQueryTime = now;
		cachedMemories = result ? JSON.stringify(result) : null;
		return result;
	} catch (error) {
		// Silently fail — memory system might not be available
		return null;
	}
}

async function queryNeo4j(context: string): Promise<MemoryResult[] | null> {
	// Use the vibe_memory Python module to query
	const scriptPath = resolve(PACKAGE_ROOT, "tools", "query_memory.py");

	try {
		// Ensure PYTHONPATH includes the package root for vibe_memory module
		const existingPath = env["PYTHONPATH"] ?? "";
		const childEnv: Record<string, string> = {
			...(typeof process !== "undefined" ? process.env : {}) as Record<string, string>,
			PYTHONPATH: `${PACKAGE_ROOT}${existingPath ? `:${existingPath}` : ""}`,
		};

		const { exec } = await import("node:child_process");
		const { promisify } = await import("node:util");
		const execAsync = promisify(exec);

		const escapedContext = context.replace(/'/g, "'\"'\"'");
		const cmd = `echo '${escapedContext}' | python3 ${scriptPath} ${NEO4J_URL} ${MODEL_PATH} ${SERENDIPITY} ${MAX_MEMORIES}`;
		const { stdout } = await execAsync(cmd, { env: childEnv });
		const result = JSON.parse(stdout.trim());
		return result as MemoryResult[];
	} catch {
		return null;
	}
}

function formatMemories(memories: MemoryResult[]): string {
	const lines = [
		`<EXTREMELY_IMPORTANT>
${MEMORY_MARKER}

Relevant memories from ChaosCache:`,
	];

	for (const mem of memories) {
		const emotions = mem.emotion_tags.length > 0 ? ` [${mem.emotion_tags.join(", ")}]` : "";
		const entities = mem.entities.length > 0 ? ` | entities: ${mem.entities.join(", ")}` : "";
		lines.push(`- ${mem.summary}${emotions}${entities}`);

		if (mem.notable_quotes && mem.notable_quotes.length > 0) {
			lines.push(`  > "${mem.notable_quotes[0]}"`);
		}
	}

	lines.push(`
Score: ${memories[0]?.relevance_score ?? "N/A"} | Serendipity: ${SERENDIPITY}
</EXTREMELY_IMPORTANT>`);

	return lines.join("\n");
}

function messageContainsMarker(message: unknown): boolean {
	const content = (message as { content?: unknown }).content;
	if (typeof content === "string") return content.includes(MEMORY_MARKER);
	if (!Array.isArray(content)) return false;
	return content.some((part: unknown) =>
		part && typeof part === "object" &&
		(part as { type?: unknown }).type === "text" &&
		typeof (part as { text?: unknown }).text === "string" &&
		(part as { text: string }).text.includes(MEMORY_MARKER)
	);
}

function firstNonCompactionSummaryIndex(messages: unknown[]): number {
	let index = 0;
	while ((messages[index] as { role?: unknown } | undefined)?.role === "compactionSummary") {
		index += 1;
	}
	return index;
}

interface MemoryResult {
	summary: string;
	emotion_tags: string[];
	entities: string[];
	notable_quotes: string[];
	relevance_score: number;
}
