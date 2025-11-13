
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
streamlit run n11_ai_satici_os_streamlit_app_v2.py --server.port "${PORT:-8620}"
