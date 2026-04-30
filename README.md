# ATS Resume Tailor

Streamlit app that takes a resume plus a job description and generates a tailored recruiter-facing resume.

## Run locally

```bash
cd /Users/shrumani/Downloads/uttarkar-job-assistant-jd
venv/bin/streamlit run app.py
```

Open the local URL printed by Streamlit.

## Run on another machine

Start the app on a machine that is reachable on your network. The app is configured to bind to `0.0.0.0`, so Streamlit will print a network URL you can open from another device on the same network.

## Optional AI support

Set `GEMINI_API_KEY` in the environment to enable Gemini-backed generation. Without it, the app uses the local resume engine.
