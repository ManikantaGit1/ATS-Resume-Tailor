import datetime
import base64
import html
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from streamlit import runtime as streamlit_runtime

from backend.ats import generate_resume_package
from backend.document import resume_markdown_to_docx
from backend.parser import extract_text_from_resume


if __name__ == "__main__" and not streamlit_runtime.exists():
    print("This app must be started with Streamlit.")
    print("")
    print("Run:")
    print("  venv/bin/streamlit run app.py")
    sys.exit(0)


st.set_page_config(page_title="ATS Resume Tailor", layout="wide")
OUTPUT_DIR = Path(__file__).resolve().parent / "generated_resumes"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def render_score_scanner(score: int, original_score: int) -> None:
    safe_score = max(0, min(100, int(score)))
    safe_original = max(0, min(100, int(original_score)))
    delta = safe_score - safe_original
    delta_text = f"+{delta}" if delta >= 0 else str(delta)

    components.html(
        f"""
        <div class="scan-wrap">
          <style>
            .scan-wrap {{
              font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              color: #162033;
              display: grid;
              grid-template-columns: minmax(220px, 0.9fr) minmax(260px, 1.1fr);
              gap: 18px;
              align-items: stretch;
              width: 100%;
            }}
            .scanner {{
              position: relative;
              min-height: 260px;
              border: 1px solid #d9dee8;
              border-radius: 8px;
              background: #f8fafc;
              overflow: hidden;
              padding: 18px;
            }}
            .sheet {{
              height: 224px;
              border-radius: 6px;
              background: #ffffff;
              border: 1px solid #e3e8f0;
              box-shadow: 0 10px 28px rgba(22, 32, 51, 0.08);
              padding: 18px;
              position: relative;
              overflow: hidden;
            }}
            .line {{
              height: 9px;
              background: #d7dfeb;
              border-radius: 999px;
              margin-bottom: 12px;
            }}
            .line.short {{ width: 48%; }}
            .line.medium {{ width: 72%; }}
            .line.long {{ width: 92%; }}
            .scan-line {{
              position: absolute;
              left: 0;
              right: 0;
              height: 3px;
              background: #16a34a;
              box-shadow: 0 0 18px rgba(22, 163, 74, 0.8);
              animation: scan 2.1s ease-in-out infinite;
            }}
            @keyframes scan {{
              0% {{ top: 14px; opacity: 0.35; }}
              50% {{ top: 205px; opacity: 1; }}
              100% {{ top: 14px; opacity: 0.35; }}
            }}
            .score-panel {{
              border: 1px solid #d9dee8;
              border-radius: 8px;
              background: #ffffff;
              padding: 22px;
              display: grid;
              grid-template-columns: 156px 1fr;
              gap: 22px;
              align-items: center;
            }}
            .score-label {{
              font-size: 12px;
              color: #667085;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              margin-bottom: 6px;
            }}
            .gauge {{
              --score: {safe_score};
              width: 150px;
              height: 150px;
              border-radius: 50%;
              background:
                radial-gradient(closest-side, #ffffff 70%, transparent 71%),
                conic-gradient(#16a34a calc(var(--score) * 1%), #e6ebf2 0);
              display: grid;
              place-items: center;
              animation: pulse 1.8s ease-in-out infinite;
            }}
            .gauge strong {{
              font-size: 34px;
              letter-spacing: 0;
            }}
            @keyframes pulse {{
              0%, 100% {{ transform: scale(1); }}
              50% {{ transform: scale(1.025); }}
            }}
            .score-copy h3 {{
              margin: 0 0 8px;
              font-size: 18px;
            }}
            .score-copy p {{
              margin: 6px 0;
              color: #4b5563;
              font-size: 14px;
              line-height: 1.45;
            }}
            .delta {{
              display: inline-flex;
              align-items: center;
              gap: 8px;
              color: #166534;
              font-weight: 700;
              margin-top: 8px;
            }}
            @media (max-width: 720px) {{
              .scan-wrap {{
                grid-template-columns: 1fr;
              }}
              .scanner {{
                display: none;
              }}
              .score-panel {{
                width: 100%;
                box-sizing: border-box;
              }}
              .score-copy h3 {{
                font-size: 16px;
                line-height: 1.2;
              }}
              .score-copy p {{
                font-size: 13px;
              }}
              .score-panel {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 14px;
                padding: 16px;
              }}
              .gauge {{
                margin: 0 auto;
              }}
            }}
          </style>
          <div class="scanner" aria-label="Animated resume scan">
            <div class="sheet">
              <div class="scan-line"></div>
              <div class="line short"></div>
              <div class="line long"></div>
              <div class="line medium"></div>
              <br />
              <div class="line long"></div>
              <div class="line medium"></div>
              <div class="line long"></div>
              <br />
              <div class="line medium"></div>
              <div class="line long"></div>
              <div class="line short"></div>
            </div>
          </div>
          <div class="score-panel">
            <div class="gauge"><strong>{safe_score}%</strong></div>
            <div class="score-copy">
              <div class="score-label">Internal match heuristic</div>
              <h3>Source vs generated resume</h3>
              <p>Source resume score against this JD: <strong>{safe_original}%</strong></p>
              <p>Generated resume score against this JD: <strong>{safe_score}%</strong></p>
              <div class="delta">Score movement: {delta_text} points</div>
              <p>This is a local heuristic, not a real ATS score from a hiring system.</p>
            </div>
          </div>
        </div>
        """,
        height=460,
    )


def render_change_tile(change: dict) -> None:
    section = html.escape(str(change.get("section", "Resume")))
    update = html.escape(str(change.get("change", "")))
    reason = html.escape(str(change.get("reason", "")))
    st.markdown(
        f"""
        <div class="change-tile">
          <strong>{section}</strong>
          <p>{update}</p>
          <span>{reason}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_generated_resume(docx_bytes: bytes, uploaded_file) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_stem = Path(getattr(uploaded_file, "name", "resume")).stem
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in source_stem)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{safe_stem}_ats_tailored_{timestamp}.docx"
    output_path.write_bytes(docx_bytes)
    return output_path


def render_browser_download_button(docx_bytes: bytes, filename: str) -> None:
    encoded = base64.b64encode(docx_bytes).decode("ascii")
    button_id = f"download-{abs(hash(filename))}"
    safe_filename = html.escape(filename, quote=True)
    components.html(
        f"""
        <div style="margin: 0 0 10px;">
          <button id="{button_id}" style="
            width: 100%;
            min-height: 42px;
            border: none;
            border-radius: 8px;
            background: #166534;
            color: #ffffff;
            font-weight: 700;
            font-size: 14px;
            cursor: pointer;
          ">Download Generated Resume</button>
        </div>
        <script>
          const button = document.getElementById("{button_id}");
          if (button) {{
            button.addEventListener("click", () => {{
              const binary = atob("{encoded}");
              const bytes = new Uint8Array(binary.length);
              for (let i = 0; i < binary.length; i += 1) {{
                bytes[i] = binary.charCodeAt(i);
              }}
              const blob = new Blob(
                [bytes],
                {{ type: "{DOCX_MIME}" }}
              );
              const url = URL.createObjectURL(blob);
              const link = document.createElement("a");
              link.href = url;
              link.download = "{safe_filename}";
              document.body.appendChild(link);
              link.click();
              link.remove();
              window.setTimeout(() => URL.revokeObjectURL(url), 1200);
            }});
          }}
        </script>
        """,
        height=62,
    )


def inject_page_styles() -> None:
    st.markdown(
        """
        <style>
          .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
          }
          .app-title {
            margin-bottom: 0.25rem;
          }
          .subtle {
            color: #5b6472;
            margin-top: 0;
          }
          .change-tile {
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            margin-bottom: 10px;
          }
          .change-tile strong {
            display: block;
            color: #162033;
            margin-bottom: 6px;
          }
          .change-tile p {
            margin: 0 0 5px;
            color: #293241;
            line-height: 1.45;
          }
          .change-tile span {
            color: #667085;
            font-size: 0.92rem;
          }
          .direct-download {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            min-height: 42px;
            border: 1px solid #93c5fd;
            border-radius: 8px;
            color: #dbeafe !important;
            background: #1e3a8a;
            text-decoration: none !important;
            font-weight: 700;
            margin: 8px 0 12px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_template_bytes(uploaded_file) -> Optional[bytes]:
    if uploaded_file and uploaded_file.name.lower().endswith(".docx"):
        return uploaded_file.getvalue()
    return None


inject_page_styles()

st.markdown("<h1 class='app-title'>ATS Resume Tailor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtle'>Upload a resume, paste a job description, generate a recruiter-aligned resume.</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.subheader("Inputs")
    uploaded_resume = st.file_uploader("Resume", type=["pdf", "docx", "txt"])
    job_description = st.text_area("Job Description", height=360, placeholder="Paste the full job description here.")
    generate = st.button("Generate ATS Resume", type="primary", use_container_width=True)

with right:
    st.subheader("Result")
    if "resume_package" not in st.session_state:
        st.info("Generated resume, score, and change details will appear here. The score is an internal match heuristic.")

if generate:
    if not uploaded_resume:
        st.error("Upload a resume first.")
    elif not job_description.strip():
        st.error("Paste the job description first.")
    else:
        try:
            progress = st.progress(0)
            for percent in (12, 28, 44, 61, 78):
                progress.progress(percent)
                time.sleep(0.08)

            resume_text = extract_text_from_resume(uploaded_resume)
            package = generate_resume_package(resume_text, job_description)
            package["docx_bytes"] = resume_markdown_to_docx(
                package["resume_markdown"],
                template_bytes=get_template_bytes(uploaded_resume),
            )
            local_path = save_generated_resume(package["docx_bytes"], uploaded_resume)
            package["local_path"] = str(local_path)

            progress.progress(100)
            time.sleep(0.08)
            progress.empty()
            st.session_state.resume_package = package
            st.session_state.resume_filename = local_path.name
            st.success("Resume generated.")
        except Exception as exc:
            st.error(f"Could not generate resume: {exc}")

package = st.session_state.get("resume_package")
if package:
    score = package.get("score", 0)
    original_score = package.get("original_score", 0)
    if package.get("tailoring_decision"):
        st.warning(package["tailoring_decision"])
    st.markdown(
        f"""
        <style>
          @media (max-width: 720px) {{
            .score-mobile-summary {{
              display: grid !important;
              gap: 10px;
              margin-bottom: 12px;
            }}
            .score-mobile-tile {{
              background: #111827;
              border: 1px solid #374151;
              border-radius: 12px;
              padding: 12px 14px;
            }}
            .score-mobile-tile .label {{
              font-size: 11px;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              color: #9ca3af;
              margin-bottom: 4px;
            }}
            .score-mobile-tile .value {{
              font-size: 24px;
              font-weight: 700;
              color: #f9fafb;
              line-height: 1.1;
            }}
            .score-mobile-tile .sub {{
              margin-top: 6px;
              color: #d1d5db;
              font-size: 13px;
              line-height: 1.35;
            }}
          }}
          @media (min-width: 721px) {{
            .score-mobile-summary {{
              display: none !important;
            }}
          }}
        </style>
        <div class="score-mobile-summary">
          <div class="score-mobile-tile">
            <div class="label">Source score</div>
            <div class="value">{original_score}%</div>
            <div class="sub">Score of the uploaded resume against the job description.</div>
          </div>
          <div class="score-mobile-tile">
            <div class="label">Final score</div>
            <div class="value">{score}%</div>
            <div class="sub">Score of the kept version after tailoring decisions.</div>
          </div>
          <div class="score-mobile-tile">
            <div class="label">Change</div>
            <div class="value">{score - original_score:+d}</div>
            <div class="sub">Positive means tailoring helped. Negative means the original was better.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_score_scanner(score, original_score)

    render_browser_download_button(
        package["docx_bytes"],
        st.session_state.get("resume_filename", "ats_tailored_resume.docx"),
    )

    if package.get("local_path"):
        st.success("A copy was saved locally.")
        st.text(package["local_path"])

    preview_tab, changes_tab, scan_tab = st.tabs(["Generated Resume", "Modified Details", "ATS Verification"])

    with preview_tab:
        st.text_area("Generated resume text", package["resume_markdown"], height=520)

    with changes_tab:
        st.caption(f"Engine: {package.get('engine', 'Unknown')}")
        for change in package.get("modifications", []):
            render_change_tile(change)

        keywords_added = package.get("keywords_added", [])
        keywords_missing = package.get("keywords_missing", [])
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Keywords incorporated**")
            st.write(", ".join(keywords_added) if keywords_added else "None detected")
        with col_b:
            st.markdown("**Remaining gaps**")
            st.write(", ".join(keywords_missing) if keywords_missing else "No major gaps detected")

    with scan_tab:
        scan = package.get("scan", {})
        st.caption("This panel shows an internal heuristic for keyword and structure match. It is not an external ATS score.")
        metric_cols = st.columns(5)
        metric_cols[0].metric("Keyword", f"{scan.get('keyword_score', 0)}%")
        metric_cols[1].metric("Similarity", f"{scan.get('similarity_score', 0)}%")
        metric_cols[2].metric("Sections", f"{scan.get('section_score', 0)}%")
        metric_cols[3].metric("Impact", f"{scan.get('impact_score', 0)}%")
        metric_cols[4].metric("Action Verbs", f"{scan.get('action_verb_score', 0)}%")

        st.markdown("**Recruiter validation checklist**")
        st.table(package.get("recruiter_checks", []))

        st.markdown("**16-point resume checker**")
        st.dataframe(scan.get("quality_checks", []), use_container_width=True, hide_index=True)

        st.markdown("**Recommendations**")
        for recommendation in scan.get("recommendations", []):
            st.write(f"- {recommendation}")

        if package.get("tailoring_decision"):
            st.markdown("**Tailoring decision**")
            st.write(package["tailoring_decision"])

        st.caption(package.get("disclaimer", ""))

st.markdown(
    """
    <div style="margin-top: 2rem; padding: 1rem 0 0.5rem; border-top: 1px solid rgba(148, 163, 184, 0.25); color: #667085; font-size: 0.9rem; text-align: center;">
      Developed by Manikanta Uttarkar | Contact: <a href="mailto:ukantesh@icloud.com">ukantesh@icloud.com</a>
    </div>
    """,
    unsafe_allow_html=True,
)
