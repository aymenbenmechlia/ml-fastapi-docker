import os

from dotenv import load_dotenv

load_dotenv()


PROJECT_ID = os.getenv(
    "GOOGLE_CLOUD_PROJECT",
    "ddp-dtm-perf-prd-frlm",
)