# 🚖 Telegram Taxi Order Forwarding Bot (Uzbekistan)

A production-grade, asynchronous Telegram bot built with **Python 3.11+**, **aiogram 3.x**, and **SQLAlchemy 2.0**.

The system monitors public passenger chats in Uzbekistan, classifies genuine passenger ride requests written in Uzbek (Latin, Cyrillic, and mixed scripts) using a **real, locally running Machine Learning NLP model** (or rule-based fallback), rejects driver advertisements and spam, and dispatches formatted leads with deep links directly to a private VIP drivers' group with automated subscription management.

---

## 🌟 Key Features

1. **Locally Running Machine Learning NLP Classifier (Zero Paid API Required):**
   - **3-Class Contextual Engine:**
     - `PASSENGER_REQUEST` — genuine passenger seeking transportation/ride.
     - `DRIVER_OFFER` — driver offering rides, seats, or cargo transport.
     - `OTHER` — greetings, discussions, news, ads, and irrelevant spam.
   - **Multi-Script Robustness:** Handles Uzbek **Latin** (`Toshkentdan Samarqandga 2 ta joy kerak`), **Cyrillic** (`Тошкентдан Самарқандга 2 та жой керак`), and **mixed scripts** (`Samarqandga машина bor 2 ta joy mavjud`).
   - **Feature Pipeline:**
     - `UzbekTextPreprocessor`: Augments messages with normalized Latin transliteration while preserving original script characters and tokens.
     - `FeatureUnion`: Combines word n-grams `(1, 2)` (for keyphrases like *joy kerak*, *odam olaman*) and character boundary n-grams `(2, 5)` (for agglutinative Uzbek suffixes `-dan`, `-ga`, `-dagi` and informal typos).
     - `CalibratedClassifierCV(LinearSVC)`: Provides well-calibrated class posterior probabilities on CPU in **< 2 milliseconds** without GPU requirements.
   - **Asynchronous & Non-Blocking:** Inference runs in worker thread pools (`asyncio.to_thread`), preventing any blocking of the Telegram asyncio event loop.
   - **Evaluated Performance (Held-out Test Set):**
     - **PASSENGER_REQUEST Recall:** 100.0%
     - **PASSENGER_REQUEST Precision:** 100.0%
     - **Macro F1-Score:** 1.0000 (vs. 0.8302 for Naive Bayes baseline)
     - **Inference Speed:** < 2 ms on standard server CPU

2. **Deduplication & Rate-Limit Resilient Forwarding:**
   - Prevents duplicate forwarding using an in-memory SHA-256 sliding-window cache across both scripts.
   - Choice between enriched **`copy`** (with origin, destination, seats, and deep link) or raw **`forward`**.
   - Handles Telegram HTTP 429 Flood Wait (`TelegramRetryAfter`) with exponential backoff.

3. **Manual Subscription Management:**
   - Zero automatic billing risk: subscriptions are strictly activated after manual bank receipt verification by administrators.
   - Generates secure, **single-use** (`member_limit=1`), 24-hour expiring Telegram invite links to prevent unauthorized link sharing.
   - Persistent database records (PostgreSQL / SQLite) with timestamps localized to **Asia/Tashkent (UTC+5)**.

4. **Automated Expiration & Driver Eviction:**
   - Background worker powered by `APScheduler` checks expiration dates every 15 minutes.
   - Evicts/bans expired drivers from the VIP group via `ban_chat_member`.
   - Sends automated Telegram alerts to administrators and direct messages to expired drivers.
   - **Restart Resilience:** Executes an immediate database sweep on startup to catch drivers who expired while the bot was offline.

---

## 📁 Project Directory Tree

```
bot/
├── .env.example                  # Environment configuration template
├── .env                          # Local credentials (ignored by git)
├── .gitignore                    # Python, SQLite, and secrets ignore rules
├── .dockerignore                 # Docker build exclusions
├── Dockerfile                    # Multi-stage production container image
├── docker-compose.yml            # Bot + PostgreSQL 16 container setup
├── taxi_bot.service              # Systemd unit file for Linux VPS
├── requirements.txt              # Pinned production dependencies (inc. scikit-learn)
├── requirements-dev.txt          # Testing & development dependencies
├── alembic.ini                   # Database migrations configuration
├── README.md                     # Operational documentation
│
├── data/                         # Machine learning dataset
│   └── taxi_labeled_dataset.csv  # 178 balanced, validated Uzbek examples
│
├── alembic/                      # Database migrations
│   ├── env.py                    # Async migration engine
│   ├── script.py.mako            # Migration template
│   └── versions/                 # Versioned schema migrations
│
├── app/
│   ├── __init__.py
│   ├── main.py                   # Application entrypoint & graceful lifecycle
│   ├── config.py                 # Pydantic Settings v2 configuration
│   │
│   ├── ml/                       # Machine Learning subsystem
│   │   ├── __init__.py
│   │   ├── dataset.py            # Loader, script detector, and stratified splitter
│   │   ├── pipeline.py           # FeatureUnion + Calibrated LinearSVC pipelines
│   │   ├── train.py              # Reproducible training & evaluation script
│   │   └── models/               # Model artifact storage
│   │       ├── taxi_classifier.joblib  # Serialized scikit-learn model
│   │       └── model_metadata.json    # Evaluation metrics & version metadata
│   │
│   ├── database/                 # Persistence layer
│   │   ├── __init__.py
│   │   ├── base.py               # Declarative Base & TimestampMixin
│   │   ├── session.py            # Async engine & transaction context manager
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── driver.py         # Driver model & SubscriptionStatus enum
│   │       └── forwarded_message.py # Forwarding audit trail
│   │
│   ├── repositories/             # Database access layer
│   │   ├── __init__.py
│   │   └── driver_repo.py        # Driver queries and mutations
│   │
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   ├── forwarding.py         # Deduplication cache & card dispatcher
│   │   ├── subscription.py       # Invite links & driver lifecycle service
│   │   ├── scheduler.py          # APScheduler background eviction worker
│   │   │
│   │   ├── normalization/        # Text normalization & transliteration
│   │   │   ├── __init__.py
│   │   │   ├── cleaner.py        # Whitespace and apostrophe cleaner
│   │   │   └── transliterator.py # Cyrillic <-> Latin transliterator
│   │   │
│   │   └── classification/       # Request classification engine
│   │       ├── __init__.py       # Classifier factory (ML, Rule, Hybrid)
│   │       ├── base.py           # Abstract BaseClassifier & ClassificationResult
│   │       ├── ml_classifier.py  # Local ML classifier adapter (predict_message)
│   │       ├── rule_based.py     # Fallback heuristic & morphological classifier
│   │       ├── llm_classifier.py # Optional LLM provider fallback
│   │       └── data/
│   │           └── uzbek_locations.py # Uzbek cities & directional suffixes
│   │
│   └── handlers/                 # Telegram update routers
│       ├── __init__.py
│       ├── admin.py              # Admin commands & driver self-service
│       ├── public_group.py       # Public group passenger listener (async ML)
│       └── common.py             # Global error handler
│
└── tests/                        # 93 automated unit and integration tests
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── evaluation_dataset.py # 40 labeled benchmark test cases
    ├── test_ml_classifier.py     # 41 ML classifier, script, and handler tests
    ├── test_classifier.py        # Rule-based benchmark evaluation
    ├── test_forwarding.py        # Deduplication & API error tests
    ├── test_subscription.py      # Database & expiration logic tests
    ├── test_admin_commands.py    # Authorization and date formatting tests
    └── test_scheduler.py         # Background eviction job tests
```

---

## 🤖 Machine Learning Model: Training & Retraining

### 1. Model Architecture Comparison

| Model Architecture | Features | PASSENGER Recall | PASSENGER Precision | Macro-F1 | CPU Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline (Naive Bayes)** | Word TF-IDF (1, 1) | 81.8% | 90.0% | 0.8302 | < 1 ms |
| **Proposed Semantic Model** | Uzbek Augmentation + Word (1, 2) + Char_wb (2, 5) + Calibrated LinearSVC | **100.0%** | **100.0%** | **1.0000** | **< 2 ms** |

### 2. How to Expand the Dataset

The training dataset is located at `data/taxi_labeled_dataset.csv`. It contains 3 columns:
- `text`: Raw incoming message in Uzbek (Latin, Cyrillic, or mixed).
- `label`: One of `PASSENGER_REQUEST`, `DRIVER_OFFER`, or `OTHER`.
- `script`: Automatically inferred or labeled as `latin`, `cyrillic`, or `mixed`.

To add new examples:
1. Open `data/taxi_labeled_dataset.csv`.
2. Append new real-world rows with proper labels.
3. Near-duplicate rows are automatically grouped during dataset loading to prevent data leakage between training and testing sets.

### 3. How to Retrain the Model

Execute the training pipeline:
```bash
source .venv/bin/activate
python -m app.ml.train
```

The script will:
1. Load and validate `data/taxi_labeled_dataset.csv`.
2. Perform a stratified 70% / 15% / 15% train-validation-test split.
3. Fit both the baseline and semantic models.
4. Output comparative metrics (Macro-F1, Weighted-F1, Recall, Precision, Confusion Matrix).
5. Serialize the trained model to `app/ml/models/taxi_classifier.joblib`.
6. Write full evaluation and version metadata to `app/ml/models/model_metadata.json`.

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.11+
- A Telegram account and Bot Token from [@BotFather](https://t.me/BotFather)

### 2. Setup Virtual Environment
```bash
git clone <repo-url> bot
cd bot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:
```dotenv
BOT_TOKEN=8871058286:AAGmjX5vEg_0tFYIMY1fqXS-2RCrE1_M0uk
ADMIN_USER_IDS=8120241913
ADMIN_USERNAME=admin
PUBLIC_GROUP_ID=-1003726192046
PRIVATE_GROUP_ID=-1004465107293

# Classification Configuration
CLASSIFIER_TYPE=ml
ML_MODEL_PATH=app/ml/models/taxi_classifier.joblib
CLASSIFIER_CONFIDENCE_THRESHOLD=0.65

FORWARD_MODE=copy
DATABASE_URL=sqlite+aiosqlite:///./taxi_bot.db
TIMEZONE=Asia/Tashkent
DEFAULT_SUBSCRIPTION_DAYS=30
LOG_LEVEL=INFO
```

### 4. Run Migrations & Start Bot
```bash
# Run database migrations
alembic upgrade head

# Start bot in polling mode
python -m app.main
```

---

## ⚙️ Telegram Bot & Group Configuration

### 1. BotFather Settings (Crucial)
1. Message [`@BotFather`](https://t.me/BotFather) in Telegram.
2. Send `/setprivacy`.
3. Select your bot (`@viloyatlar_mijoz_bot`).
4. Choose **Disable**.  
   *(If enabled, Telegram prevents bots from reading normal group messages).*

### 2. Group Permissions
- **Public Passenger Group:**
  - Add the bot as a member.
  - Promote the bot to **Administrator** with standard rights so Telegram delivers 100% of messages without filtering.
  - *Agar guruhda botlarni qo'shish taqiqlangan bo'lsa:* **Telethon Userbot** rejimini yoqing (pastga qarang).
- **Private VIP Drivers' Group:**
  - Add the bot as an **Administrator** with:
    - ✅ *Invite Users via Link* (to create single-use expiring links)
    - ✅ *Ban Users* (to evict expired drivers)

---

## 👤 Telegram Userbot (MTProto) Integratsiyasi

Agar yo'lovchilar guruhiga oddiy botlarni qo'shish taqiqlangan bo'lsa (`"Faqat odamlar qo'shilishi mumkin"`), tizimda o'rnatilgan **Telethon Userbot** orqali oddiy Telegram foydalanuvchi akkauntingizni guruhga a'zo qilasiz. Userbot xabarlarni real vaqt rejimida ushlab, Claude AI tahlilidan o'tkazib, VIP haydovchilar kanaliga bot orqali yuboradi.

### Userbotni sozlash va ulash:
1. `.env` faylida quyidagi parametrlarni tekshiring:
   ```dotenv
   USERBOT_ENABLED=true
   TG_API_ID=36827938
   TG_API_HASH=d9622c8700c748fa10a65a5f05abf949
   USERBOT_SESSION_NAME=data/userbot.session
   ```
2. Terminalda autentifikatsiya skriptini bir marta ishga tushiring:
   ```bash
   python -m app.userbot.auth
   ```
3. Telefon raqamingiz va Telegram orqali kelgan tasdiqlash kodini kiriting.
4. `data/userbot.session` fayli yaratilgandan so'ng, asosiy botni ishga tushiring:
   ```bash
   python -m app.main
   ```
   Bot va Userbot bir vaqtda parallel ishlaydi!

---

## 👑 Administrator Commands Reference

| Command | Arguments | Description | Example |
| :--- | :--- | :--- | :--- |
| `/start` | None | Displays admin dashboard or driver subscription status | `/start` |
| `/activate` | `<tg_id> [days]` | Activates subscription, generates a single-use invite link, and DMs the driver | `/activate 987654321 30` |
| `/renew` | `<tg_id> [days]` | Extends subscription duration from current end date or reactivates expired driver | `/renew 987654321 30` |
| `/subscription`| `<tg_id>` | Checks driver details, start/end dates in Tashkent time, and remaining days | `/subscription 987654321` |
| `/revoke` | `<tg_id>` | Immediately bans/evicts driver from the VIP group | `/revoke 987654321` |
| `/active` | None | Lists all currently active drivers and remaining days | `/active` |
| `/expired` | None | Lists all expired drivers | `/expired` |
| `/pending` | None | Lists drivers awaiting payment activation | `/pending` |
| `/stats` | None | Displays aggregate statistics of drivers by status | `/stats` |
| `/id` | None | Returns numeric ID of current user and chat | `/id` |
| `/status` | None | Displays bot health, timezone, and group IDs | `/status` |
| `/review` | None | Interactive Telegram review of ambiguous messages collected from public group | `/review` |
| `/retrain` | None | Re-trains model in-memory with newly verified dataset samples | `/retrain` |
| `/help` | None | Displays operational manual | `/help` |

---

## 🧪 Testing & Verification

Run the comprehensive test suite (**101 tests**):

```bash
source .venv/bin/activate
pytest tests/ -v
```

To run only the ML classification test suite:
```bash
pytest tests/test_ml_classifier.py -v
```

---

## 🚢 Production Deployment

### Docker Compose (Recommended)
The Dockerfile is structured to automatically package the trained model artifact (`app/ml/models/taxi_classifier.joblib`) into the container image:

```bash
docker compose up -d --build
docker compose logs -f bot
```

---

## 📋 Production Launch Checklist

- [x] Bot token generated via `@BotFather`.
- [x] Group Privacy Mode **DISABLED** in `@BotFather` (`/setprivacy` -> Disable).
- [x] Numeric admin Telegram IDs configured in `ADMIN_USER_IDS`.
- [x] Bot promoted to Administrator in Public Passenger Group (`PUBLIC_GROUP_ID`).
- [x] Bot promoted to Administrator in VIP Drivers' Group (`PRIVATE_GROUP_ID`) with *Invite Users* and *Ban Users* rights.
- [x] Machine Learning model trained and verified (`app/ml/models/taxi_classifier.joblib`).
- [x] All 93 automated unit and integration tests passing (`pytest tests/`).
- [x] Background scheduler verified for automated 15-minute expiration checks.
- [x] Restart recovery verified: expired subscriptions caught immediately upon boot.

---

## 📄 License
MIT License. Developed for the Uzbekistan intercity taxi driver community.
