# SEVA Multiview Demo

## Gradio UI

Launch the interactive demo with:

```
python gradio_app.py --host 0.0.0.0 --port 7860
```

### Options
- `--share`: request a temporary Gradio link if outbound internet is allowed.
- `--concurrency`: maximum simultaneous runs (leave at 1 unless the GPU is oversized).

The UI enumerates examples from `images/objects` and `images/references`; drop new PNG/JPG files there to make them selectable without restarting the app.
