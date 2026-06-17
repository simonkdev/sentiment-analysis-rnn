INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Sentiment RNN Demo</title>
    <style>
        :root {
            --bg: #f6f7fb;
            --card: #ffffff;
            --text: #172033;
            --muted: #667085;
            --border: #d9deea;
            --accent: #3758f9;
            --accent-soft: #eef2ff;
            --positive: #16885f;
            --negative: #c2415d;
            --code: #f1f5f9;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            color: var(--text);
            background: var(--bg);
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            line-height: 1.5;
        }

        main {
            width: min(980px, calc(100% - 32px));
            margin: 0 auto;
            padding: 36px 0;
        }

        header {
            margin-bottom: 24px;
        }

        .eyebrow {
            display: inline-block;
            margin-bottom: 10px;
            padding: 5px 10px;
            border-radius: 999px;
            color: #3445a0;
            background: var(--accent-soft);
            font-size: 13px;
            font-weight: 700;
        }

        h1 {
            margin: 0 0 10px;
            font-size: clamp(32px, 5vw, 52px);
            line-height: 1.05;
            letter-spacing: -0.04em;
        }

        .intro {
            max-width: 720px;
            margin: 0;
            color: var(--muted);
            font-size: 17px;
        }

        .grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 340px;
            gap: 18px;
            align-items: start;
        }

        .card {
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--card);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }

        .panel {
            padding: 22px;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 12px;
        }

        h2 {
            margin: 0;
            font-size: 18px;
        }

        .hint {
            color: var(--muted);
            font-size: 13px;
        }

        textarea {
            width: 100%;
            min-height: 230px;
            resize: vertical;
            padding: 14px;
            border: 1px solid var(--border);
            border-radius: 12px;
            outline: none;
            color: var(--text);
            background: #fbfcff;
            font: inherit;
        }

        textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(55, 88, 249, 0.12);
        }

        .actions,
        .samples {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }

        .actions {
            justify-content: space-between;
            margin-top: 14px;
        }

        .samples {
            margin-top: 14px;
        }

        button {
            border: 0;
            border-radius: 10px;
            cursor: pointer;
            font: inherit;
            font-weight: 700;
        }

        .primary {
            padding: 11px 14px;
            color: white;
            background: var(--accent);
        }

        .secondary,
        .sample {
            border: 1px solid var(--border);
            color: var(--text);
            background: white;
        }

        .secondary {
            padding: 10px 13px;
        }

        .sample {
            padding: 8px 10px;
            font-size: 13px;
        }

        .result-label {
            display: inline-block;
            margin: 8px 0 18px;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .result-label.positive {
            color: var(--positive);
            background: #eaf8f2;
        }

        .result-label.negative {
            color: var(--negative);
            background: #fff0f3;
        }

        .score-row {
            margin-top: 14px;
        }

        .score-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            color: var(--muted);
            font-size: 13px;
        }

        .track {
            height: 9px;
            overflow: hidden;
            border-radius: 999px;
            background: #eef1f6;
        }

        .fill {
            height: 100%;
            width: 0;
            transition: width 0.2s ease;
        }

        .fill.positive {
            background: var(--positive);
        }

        .fill.negative {
            background: var(--negative);
        }

        .explain {
            margin-top: 18px;
            padding: 14px;
            border-radius: 12px;
            background: var(--code);
            color: #475467;
            font-size: 13px;
        }

        .notes {
            margin-top: 18px;
        }

        .notes ul {
            margin: 10px 0 0;
            padding-left: 18px;
            color: var(--muted);
        }

        .notes li {
            margin: 7px 0;
        }

        footer {
            margin-top: 22px;
            color: var(--muted);
            font-size: 13px;
        }

        .hidden {
            display: none;
        }

        @media (max-width: 820px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <main>
        <header>
            <span class="eyebrow">Educational RNN demo</span>
            <h1>Sentiment analysis using a recurrent neural network</h1>
            <p class="intro">
                Paste a movie-review style sentence or paragraph. The model returns a positive or negative label and the two raw class probabilities.
            </p>
        </header>

        <section class="grid">
            <div class="card panel">
                <div class="panel-header">
                    <h2>Input text</h2>
                    <span id="charCount" class="hint">0 characters</span>
                </div>
                <textarea id="inputText" placeholder="Example: The acting was thoughtful and the ending was surprisingly moving."></textarea>
                <div class="actions">
                    <div>
                        <button class="primary" onclick="processText()">Analyze</button>
                        <button class="secondary" onclick="clearText()">Clear</button>
                    </div>
                    <span id="status" class="hint">Ready</span>
                </div>
                <div class="samples">
                    <button class="sample" onclick="useSample('positive')">Positive example</button>
                    <button class="sample" onclick="useSample('negative')">Negative example</button>
                    <button class="sample" onclick="useSample('mixed')">Mixed example</button>
                </div>
            </div>

            <aside class="card panel">
                <div class="panel-header">
                    <h2>Output</h2>
                    <span id="latency" class="hint"></span>
                </div>
                <div id="emptyState" class="hint">
                    Run an example to see the model output here.
                </div>
                <div id="resultState" class="hidden">
                    <div id="sentimentBadge" class="result-label">—</div>
                    <div class="score-row">
                        <div class="score-label"><span>Positive</span><span id="positiveScore">0%</span></div>
                        <div class="track"><div id="positiveFill" class="fill positive"></div></div>
                    </div>
                    <div class="score-row">
                        <div class="score-label"><span>Negative</span><span id="negativeScore">0%</span></div>
                        <div class="track"><div id="negativeFill" class="fill negative"></div></div>
                    </div>
                    <div class="explain">
                        The larger probability becomes the label. If both bars are close, the model is not very decisive.
                    </div>
                </div>
                <div class="notes">
                    <h2>Notes</h2>
                    <ul>
                        <li>This is a from-scratch NumPy RNN, not a transformer model.</li>
                        <li>It works best with review-like English text.</li>
                        <li>Short prompts can be noisy because the model has limited context.</li>
                    </ul>
                </div>
            </aside>
        </section>

        <footer>Built for learning: Flask API, NumPy model, IMDB sentiment labels. Source code: <a https://github.com/simonkdev/sentiment-analysis-rnn></a> </footer>
    </main>

    <script>
        const samples = {
            positive: "The acting was thoughtful, the story was engaging, and the ending was genuinely moving.",
            negative: "The plot felt empty, the dialogue was awkward, and the movie became boring very quickly.",
            mixed: "The visuals were strong and the cast was good, but the story felt uneven and too long."
        };

        const input = document.getElementById("inputText");
        const charCount = document.getElementById("charCount");
        const statusText = document.getElementById("status");

        input.addEventListener("input", updateCount);
        updateCount();

        function updateCount() {
            charCount.textContent = `${input.value.length} characters`;
        }

        function useSample(type) {
            input.value = samples[type];
            updateCount();
            processText();
        }

        function clearText() {
            input.value = "";
            updateCount();
            statusText.textContent = "Ready";
            document.getElementById("latency").textContent = "";
            document.getElementById("emptyState").classList.remove("hidden");
            document.getElementById("resultState").classList.add("hidden");
        }

        async function processText() {
            const text = input.value.trim();
            if (!text) {
                statusText.textContent = "Enter text first";
                return;
            }

            const startedAt = performance.now();
            statusText.textContent = "Analyzing...";

            try {
                const response = await fetch("/api/process", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text })
                });
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || "Request failed");
                }
                renderResult(data, Math.round(performance.now() - startedAt));
                statusText.textContent = "Done";
            } catch (error) {
                statusText.textContent = error.message;
            }
        }

        function renderResult(data, latencyMs) {
            const result = data.result;
            const positive = Math.round(((data.scores || {}).positive || 0) * 100);
            const negative = Math.round(((data.scores || {}).negative || 0) * 100);
            const badge = document.getElementById("sentimentBadge");

            badge.textContent = result;
            badge.className = `result-label ${result}`;
            document.getElementById("positiveScore").textContent = `${positive}%`;
            document.getElementById("negativeScore").textContent = `${negative}%`;
            document.getElementById("positiveFill").style.width = `${positive}%`;
            document.getElementById("negativeFill").style.width = `${negative}%`;
            document.getElementById("latency").textContent = `${latencyMs} ms`;
            document.getElementById("emptyState").classList.add("hidden");
            document.getElementById("resultState").classList.remove("hidden");
        }
    </script>
</body>
</html>
"""
