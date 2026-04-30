# ATS Resume Tailor

Streamlit app that takes a resume plus a job description and generates a tailored recruiter-facing resume.

## Requirements

- Python 3.8 or newer
- `pip`
- macOS, Linux, or Windows

## Local Setup

### 1. Create a virtual environment

```bash
cd /Users/shrumani/Downloads/ATS-Resume-Tailor
python3 -m venv venv
```

### 2. Activate the virtual environment

```bash
source venv/bin/activate
```

On Windows PowerShell:

```bash
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the app locally

```bash
streamlit run app.py
```

If you want the app to be reachable from another device on the same Wi-Fi, expose the address and port:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then open the `Network URL` Streamlit prints, or use your laptop IP:

```text
http://<your-laptop-ip>:8501
```

## Running On Another Machine On The Same Wi-Fi

1. Start the app on your laptop with `--server.address 0.0.0.0`.
2. Find your laptop IP:

```bash
ipconfig getifaddr en0
```

If `en0` is empty, try:

```bash
ipconfig getifaddr en1
```

3. On the other device, open:

```text
http://<your-laptop-ip>:8501
```

You can also try the Bonjour hostname:

```text
http://Manikantas-MacBook-Pro.local:8501
```

## Optional AI Support

Set `GEMINI_API_KEY` in your environment to enable Gemini-backed generation.
Without it, the app uses the local resume engine.

## Deploy To Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in with GitHub.
4. Click `Create app`.
5. Select this repository, branch, and set the entrypoint file to `app.py`.
6. Add `GEMINI_API_KEY` as a secret only if you want Gemini-backed generation.
7. Deploy the app.

Official docs:

- [Deploy your app on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

## Footer

Developed by Manikanta Uttarkar  
Contact: [ukantesh@icloud.com](mailto:ukantesh@icloud.com)
