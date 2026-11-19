# Dataset Setup: CMU CERT Insider Threat Test Dataset (r4.2)

## Official Source
Download the **CERT Insider Threat Test Dataset (r4.2)** from the Carnegie Mellon University Software Engineering Institute (SEI) digital repository:
- **DOI:** https://doi.org/10.1184/R1/12841247.v1

## Directory Placement (Optional for Raw Log Aggregation)
If running `01_preprocess_and_feature_engineering.py` directly on raw CERT r4.2 logs, place the extracted files inside `data/r4.2/`:
- `data/r4.2/logon.csv` (Kerberos/RADIUS/LDAP authentication logs)
- `data/r4.2/device.csv` (USB host controller connect/disconnect events)
- `data/r4.2/file.csv` (Endpoint SCSI/USB & SMB/NTFS file access telemetry)
- `data/r4.2/email.csv` (SMTP/MIME mail transfer agent logs)
- `data/r4.2/http.csv` (Perimeter proxy HTTP/HTTPS outbound request logs)

## Reproducible Benchmark Mode
If raw CERT r4.2 files are not placed in `data/r4.2/`, `src/01_preprocess_and_feature_engineering.py` automatically constructs the standardized **10,000-sample daily user-session telemetry benchmark** (9,187 benign records, 813 exfiltration instances, and 3.5% stochastic telemetry noise) used in the manuscript experiments.
