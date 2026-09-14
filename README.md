# Urban Growth Intelligence — Streamlit Prototype

Single-file SIH 2026 prototype for the "Managing Urban Growth" concept.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

Push `app.py` and `requirements.txt` to a GitHub repository, then select `app.py` as the app entrypoint.

## Important data honesty

The included Nagpur map and case records are clearly labeled DEMO/REFERENCE visualizations. They are not authoritative cadastral records, satellite imagery, government land ownership data, or legal-violation determinations.

The snapshot analyzer is a lightweight prototype CV signal using image preprocessing/edge and temporal-difference analysis. It is **not a trained building-segmentation model**, and its confidence value is not a real-world accuracy metric.

For a production/SIH-ready version, replace the prototype CV block with a validated building-segmentation/change-detection model and ingest legitimately licensed geospatial layers.
