# Browser-first Google Cloud Roadmap

## E0–E1

GPU is not required.

Recommended browser-first environments:
- Cloud Shell Editor
- Cloud Workstations without GPU
- VS Code + Gemini Code Assist agent mode if available

## E2+

When real CV inference begins, move the worker execution to a GPU-capable environment. Keep the API/contracts/repository unchanged.

## Separation of concerns

Gemini is the coding agent.
Google Cloud is the execution environment.
The Video Blueprint contracts remain portable and should not depend on a single cloud vendor.
