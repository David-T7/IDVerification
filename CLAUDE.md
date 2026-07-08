# EthioGig KYC / ID Verification Service — Developer Reference

Django REST microservice for **document verification** (ID/passport MRZ) and **liveness detection** (face match, smile, head rotation).

**Path:** `c:\toptal\ID Verification\`  
**Port:** `8005`  
**Frontend env:** `REACT_APP_KYC_URL`  
**API prefix:** `/api/`

---

## Running

```bash
cd "c:\toptal\ID Verification"
docker compose up -d    # migrate on startup — required for core_userimage table
```

Postgres + media volume. ML models (dlib, etc.) load on first request — can take 30–60s on CPU.

Warmup endpoint preloads models: `GET /api/warmup/`

---

## Authentication

`core.authentication.CustomJWTAuthentication` — Bearer JWT with shared **`SECRET_KEY`**.

Candidate tokens use `user_id` = resume UUID. `freelancer_id` is extracted server-side from the verified JWT payload — do **not** include it in the multipart form body.

---

## Endpoints (`core/urls.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/warmup/` | Preload face/MRZ models |
| POST | `/api/verify-id/` | National ID verification |
| POST | `/api/verify-passport/` | Passport + MRZ OCR |
| POST | `/api/verify/face-match/` | Liveness step 1 — frontal face vs ID photo |
| POST | `/api/verify/smile/` | Liveness step 2 |
| POST | `/api/verify/head-rotation-right/` | Liveness step 3 |
| POST | `/api/verify/head-rotation-left/` | Liveness step 4 |
| POST | `/api/verify/update-user-image/` | Update stored ID face image |

All verify endpoints expect `multipart/form-data` with `user_image` (and `full_name` where noted). `freelancer_id` is read from the JWT — do not include it in the form body.

---

## Frontend Integration

### Document KYC (`VerifyAccountPage.js`)

1. Load candidate name from JWT or `GET /api/resumes/{id}/candidate-info/` (main backend).
2. POST passport/ID to KYC service.
3. Navigate to liveness on success.

### Liveness (`LivelinessTest.js`)

Four-step flow with countdown + camera preview:

1. Face match → on success uploads surveillance reference via `REACT_APP_SURVEILLANCE_URL/api/fetch-and-store-profile-picture/`
2. Smile
3. Head right
4. Head left

On candidate completion:

```http
POST /api/resumes/{candidateId}/pipeline-stage/
{ "stage": "kyc", "passed": true, "score": 100 }
```

Then redirect to `/application/status`.

### Performance notes (implemented)

- Model warmup on page load (`/api/warmup/`)
- HOG detector before CNN where applicable
- Cached encodings on backend
- Client timeout 300s for rotation steps
- Camera: attach stream after `<video>` mount, `muted` for autoplay

---

## Database

**`core_userimage`** — stores ID document images linked to `freelancer_id` (resume UUID for candidates).

If 500 errors mention missing table → run `docker compose exec app python manage.py migrate`.

---

## Project Layout

```
app/
├── idverification/    # settings, urls
└── core/
    ├── views.py         # VerifyID, passport, liveness views
    ├── models.py        # UserImage, etc.
    ├── authentication.py
    └── management/
```

---

## Common Gotchas

- **400 on passport** — often empty `full_name` from frontend or unreadable MRZ; backend returns explicit error message.
- **500 on verify** — usually unmigrated DB.
- **`freelancer_id` is always the JWT `user_id`** — enforced server-side; sending it in the form body has no effect.
- **DeepFace/model paths** — `DEEPFACE_HOME=/vol/web/deepface` in docker-compose.
- **Never commit API keys** in docker-compose — use `app/.env`.

---

## Pipeline Position

Stage **`kyc`** in main backend pipeline — after `ai_screening`, before `theoretical_test`.

---

## Release readiness

- [x] KYC stage in pipeline (`kyc` before theoretical tests)
- [x] Liveness step 1 uploads surveillance reference (`LivelinessTest.js`)
- [ ] Smoke test: passport + liveness → main backend `pipeline-stage` passed

**2026-06-02 vetting batch:** No KYC code changes — see main `RELEASE_READINESS.md` for stack hub / hold / taxonomy flow after KYC.

**Checklist:** `c:\toptal\Django Project\RELEASE_READINESS.md`

---

## Related

| Service | Role |
|---|---|
| Main backend (8000) | Pipeline status, candidate-info, stage reporting |
| Surveillance (8003) | Reference face for proctoring (uploaded after liveness step 1) |
| Frontend | `VerifyAccountPage.js`, `LivelinessTest.js` |
