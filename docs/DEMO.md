# 90-second demo script

Dataset: `shopping_trends.csv` (any file works; these questions match its columns).

| Time | Do | Say |
|---|---|---|
| 0:00 | Open the app, point at the key status, the model and the privacy notice | "The model is reached through OpenRouter, so only a data summary leaves my machine, and the app makes me confirm that." |
| 0:10 | Upload the CSV; show the schema table | "I only see the schema, never data rows." |
| 0:20 | Type: **Give me the average review rating for each season and draw a bar chart of it.** Press Run | "A router plans two tasks; the agents call tools on an MCP server." |
| 0:30 | Watch the live trace (ROUTER, TOOL lines, OBSERVE) | "Every figure comes from a tool call. The model never does arithmetic." |
| 0:50 | Show the report preview: Key figures table, chart, Caveats | "The writer only used placeholders; code filled in the numbers, and a number check verified them." |
| 1:05 | Try to download: buttons are disabled; tick "I reviewed this report"; download the DOCX | "A human review gate before anything leaves the app." |
| 1:15 | Ask: **What was the profit margin on each item?** | "There is no such column, so it asks for clarification instead of inventing a number." |
| 1:25 | Show the Evaluation section of the README | "I measure this: TODO(results) tool_correct / report_correct from the eval, with retries on vs off." |

Before recording: set `OPENROUTER_API_KEY`, press "Check key and model" (it should say ready), tick the privacy box, and use a non-confidential file.
