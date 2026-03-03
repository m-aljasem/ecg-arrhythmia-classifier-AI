# Website

Static project website for the ECG Arrhythmia Classifier.

## Pages
- `index.html`: project overview, dataset summary, and training workflow
- `methodology.html`: technical pipeline and reproducibility notes
- `demo.html`: embedded Streamlit live demo

## Streamlit Embed URL
The demo iframe URL is currently set to:

`https://ecg-classifier.aljasem.eu.org?embedded=true`

If needed, update this URL in `demo.html` (iframe `src`) and the "Open Demo in New Tab" links.

## Local Preview
From `project/website`:

```bash
python -m http.server 8080
```

Then open: `http://localhost:8080`
