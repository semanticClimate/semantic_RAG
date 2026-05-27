# 404 Troubleshooting README (Gunicorn + Flask + Optional Nginx)

Use this checklist when your deployed app returns `404` for `/` or `/index.html`.

---

## 1. Check Gunicorn Service Status

```bash
sudo systemctl status gunicorn-climate --no-pager
```

Expected: `active (running)`

---

## 2. Verify Service Uses Correct Project Path

```bash
sudo systemctl cat gunicorn-climate
```

Confirm:
- `WorkingDirectory=/.../semantic_RAG`
- `ExecStart=... gunicorn "app:create_app()" ...`

---

## 3. Confirm Latest Code Is Deployed

```bash
cd /path/to/semantic_RAG
git log -1 --oneline
```

Expected: latest commit that includes the index route.

---

## 4. Confirm Index Route Exists in Deployed Code

```bash
rg -n "@bp.route\\(\"/\"\\)|@bp.route\\(\"/index.html\"\\)|def index\\(" app/routes.py
```

Expected: 3 matching lines.

---

## 5. Confirm Frontend File Exists

```bash
ls -l static/index.html
```

Expected: file exists and is readable.

---

## 6. Restart Gunicorn After Code Changes

```bash
sudo systemctl restart gunicorn-climate
sleep 2
sudo systemctl status gunicorn-climate --no-pager
```

---

## 7. Test Flask Directly on Server (Bypass DNS/Proxy)

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/
curl -i http://127.0.0.1:8000/index.html
```

Expected:
- `/health` => `200`
- `/` => `200`
- `/index.html` => `200`

---

## 8. If Using Nginx, Detect Proxy-Level 404

```bash
curl -i http://127.0.0.1/
```

If Gunicorn tests in Step 7 pass but this returns 404, Nginx config is likely wrong (often still pointing to old `/app` route or wrong upstream).

---

## 9. Check Gunicorn Logs for Import/Path Errors

```bash
sudo journalctl -u gunicorn-climate -n 150 --no-pager
```

Look for:
- import errors
- wrong module path
- missing file path

---

## 10. Final External Test

```bash
curl -i http://your-domain-or-ip/
curl -i http://your-domain-or-ip/health
```

---

## Fast Diagnosis Rule

- If Step 7 fails: issue is Flask/Gunicorn app/service config.
- If Step 7 passes but external URL fails: issue is Nginx/proxy/firewall/DNS.
- If only `/` fails but `/health` works: index route or static file path issue.

