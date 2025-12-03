# AI Sales Forecasting Dashboard

This project provides an end-to-end AI sales forecasting pipeline and a simple dashboard (Streamlit) to visualize results.

## Files generated
- `sales_sample.csv` — synthetic sample dataset (Date, Sales)
- `streamlit_app.py` — Streamlit dashboard app
- `report.pdf` — concise project report
- `presentation.pptx` or `presentation_fallback.txt` — slide deck (if pptx not available)

## How to run the dashboard
1. Install requirements:
```bash
pip install pandas prophet matplotlib streamlit altair
```
2. Run the app:
```bash
streamlit run streamlit_app.py
```

## Notes
- Upload your own CSV with **Date** and **Sales** columns (Date in YYYY-MM-DD).
- The app uses Prophet for forecasting. Adjust parameters in the sidebar.
