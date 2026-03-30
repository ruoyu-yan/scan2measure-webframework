// Manual test: run with ts-node or node after tsc compilation
// This verifies the progress parsing regex works correctly.

function parseProgressLine(line: string) {
  const match = line.match(/^\[PROGRESS\]\s+(\d+)\s+(\d+)\s+(.*)$/);
  if (!match) return null;
  return {
    current: parseInt(match[1], 10),
    total: parseInt(match[2], 10),
    message: match[3].trim(),
  };
}

// Test cases
const tests = [
  {
    input: "[PROGRESS] 45 100 Processing tile 3 of 8",
    expected: { current: 45, total: 100, message: "Processing tile 3 of 8" },
  },
  {
    input: "[PROGRESS] 0 10 Starting...",
    expected: { current: 0, total: 10, message: "Starting..." },
  },
  {
    input: "Some regular log output",
    expected: null,
  },
  {
    input: "[INFO] Not a progress line",
    expected: null,
  },
];

let passed = 0;
for (const t of tests) {
  const result = parseProgressLine(t.input);
  const ok = JSON.stringify(result) === JSON.stringify(t.expected);
  console.log(ok ? "PASS" : "FAIL", t.input);
  if (ok) passed++;
}
console.log(`${passed}/${tests.length} tests passed`);
