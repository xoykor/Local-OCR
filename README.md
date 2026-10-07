# Local OCR

A local, privacy-focused desktop application that converts PDFs and images to
structured Markdown using Vision-Language models served by
[Ollama](https://ollama.com). Nothing leaves your machine or network: the app
talks only to the Ollama server URL you configure.

## Requirements

- Python 3.10 or newer with Tkinter support (macOS, Linux, or Windows).
  Both Tk 8.6 and Tk 9.0 work with the pinned customtkinter version
  (customtkinter 6.x; older 5.2.x renders blank windows under Tk 9.0 on
  macOS).
- A running Ollama server — locally or reachable on your network. The app
  never starts, installs, or pulls anything itself.
- A vision-capable model installed on that server.

## Installation

### Linux AppImage

Download the latest `Local-OCR-*-x86_64.AppImage` from the
[GitHub Releases](https://github.com/xoykor/Local-OCR/releases/latest), make it
executable, and run it:

```bash
chmod +x Local-OCR-*-x86_64.AppImage
./Local-OCR-*-x86_64.AppImage
```

The AppImage bundles the Python application and its Python dependencies. It
still requires a running Ollama server and a vision-capable model; neither is
included. AppImage mounting requires FUSE 2 on some distributions. If it is not
available, install the distribution's `libfuse2` package.

### Run from source

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Setting up Ollama

The app requires an Ollama server that is already running. To see which
models a server has installed:

```bash
ollama list
```

To install a vision-capable model:

```bash
ollama pull <model-tag>
```

The model suggestions shown in the app (`gemma4:12b`, `qwen3.6:27b`) are
examples only — they are not guaranteed to exist on your server and are never
pulled automatically. Use `Refresh Models` to list what your server actually
has, or type any model tag manually.

### Server URL

- Local Ollama: keep the default `http://localhost:11434`.
- Ollama on another machine: use its address, e.g.
  `http://192.168.1.50:11434`.

**Network safety:** exposing Ollama beyond localhost makes it reachable by
anyone who can connect to that port. Only bind it to a trusted network and
protect it with your firewall; Ollama has no built-in authentication.

## Usage

1. `Select File` — choose one PDF or image (`.pdf`, `.png`, `.jpg`, `.jpeg`,
   `.webp`).
2. Confirm the server URL, pick or type a model tag, and choose a PDF DPI
   (100/150/200/300; higher is sharper but slower — DPI only affects PDFs).
3. `Start OCR`. Each PDF page is rendered and sent to the model in order.

While a job runs you can follow it in several places:

- A **progress bar** with a page counter (`Page 3 / 12`) shows real progress —
  a short render phase followed by recognition.
- A **preview panel** displays a thumbnail of the page currently being read.
- The **Result** tab fills with the recognized Markdown live, token by token,
  as the model streams it. A `Copy` button copies the full text to the
  clipboard.
- The **Log** tab keeps the status messages.
- The **Review** tab pairs each finished page's image with its text
  side by side, with `◀` / `▶` navigation for spot-checking quality.

On success the app switches to the Result tab and shows a dialog with `Open`
(open the `.md` in your default app), `Show in Finder` (or `Open Folder` off
macOS), and `OK`. On error it switches to the Log tab.

### Output

The result is saved as UTF-8 Markdown **next to the input file** with the
`_extracted.md` suffix — `/docs/report.pdf` becomes
`/docs/report_extracted.md`. If that file already exists you are asked before
it is overwritten; declining leaves it untouched. The file is written
atomically, so a failed run never leaves a partial result.

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| `connection refused` | Ollama is not running, or the URL/port is wrong. Start Ollama (`ollama serve` or the desktop app) and verify the URL. |
| Timeout | Server unreachable (wrong LAN address, firewall) or the model is too slow for the page. Try a smaller model or lower DPI. |
| `model not found` | The tag is not installed on that server. Check `ollama list` and `ollama pull <tag>` on the server. Models are never pulled automatically. |
| Empty or garbage output / "returned no text" | The selected model has no vision support. Choose a vision-capable model. |
