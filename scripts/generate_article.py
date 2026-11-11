#!/usr/bin/env python3
"""
RapidPress.INFO - สร้างบทความภาษาไทยอัตโนมัติด้วย Gemini API
ใช้แค่ Python standard library (ไม่ต้อง pip install)

ใช้งาน:
    GEMINI_API_KEY=xxx python scripts/generate_article.py --topic "หัวข้อ" --requirements "ความต้องการเพิ่มเติม"

ทดสอบโดยไม่เรียก Gemini จริง:
    python scripts/generate_article.py --topic "ทดสอบ" --mock

ตัวแปรแวดล้อม (ไม่บังคับ):
    GEMINI_API_KEY   คีย์ Gemini (จำเป็น ถ้าไม่ใช้ --mock)
    GEMINI_MODEL     รุ่นที่อยากใช้เป็นอันดับแรก
    GEMINI_API_BASE  เปลี่ยนปลายทาง API (ใช้ตอนทดสอบ)
    TEMPLATE_FILE    ไฟล์บทความที่ใช้เป็นแม่แบบ
"""

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# ตั้งค่า
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_FILE = "index.html"
DEFAULT_TEMPLATE = "ai-speaker-training-thailand-ajarn-champ.html"
INSERT_MARKER = "<!-- NEW_ARTICLE_INSERTION_POINT -->"
SITE_NAME = "RapidPress.INFO"
# ใช้รูปแบบลิงก์รูปภาพเดียวกับบทความล่าสุดในเว็บ
IMAGE_URL_PREFIX = "https://www.famast.com/genimgg.php?g "

API_BASE = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
# ลองตามลำดับนี้ก่อน ถ้าใช้ไม่ได้ทั้งหมดจะค้นหารุ่นที่ยังเปิดอยู่จาก API เอง
PREFERRED_MODELS = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]
BKK = dt.timezone(dt.timedelta(hours=7))

# ให้พิมพ์ภาษาไทยได้ทุกเครื่อง (รวมถึง Windows)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
class ModelUnavailable(Exception):
    """รุ่นนี้ถูกปิด / ไม่มีสิทธิ์ใช้ / โควตาหมด ให้ลองรุ่นถัดไป"""


def _request(method, url, api_key, body=None, timeout=300):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("x-goog-api-key", api_key)  # ส่งคีย์ทาง header จะได้ไม่โผล่ใน log
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_available_models(api_key):
    """ถามจาก API ว่าตอนนี้มีรุ่นไหนที่ยังเรียก generateContent ได้ เรียงจากใหม่ไปเก่า"""
    try:
        res = _request("GET", f"{API_BASE}/models?pageSize=1000", api_key, timeout=60)
    except Exception as e:  # noqa: BLE001
        log(f"  ! ดึงรายชื่อรุ่นไม่ได้: {e}")
        return []
    skip = ("tts", "image", "embed", "live", "audio", "vision", "robotics",
            "computer-use", "aqa", "learnlm", "gemma", "nano", "veo", "imagen", "exp")
    found = []
    for m in res.get("models", []):
        name = m.get("name", "").replace("models/", "")
        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue
        if not name.startswith("gemini") or any(s in name for s in skip):
            continue
        ver = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        version = float(ver.group(1)) if ver else 0.0
        # flash ใช้ฟรีได้คุ้มสุด > flash-lite > pro ; รุ่นเสถียรมาก่อน preview
        if "flash-lite" in name:
            family = 1
        elif "flash" in name:
            family = 2
        else:
            family = 0
        stable = 0 if "preview" in name else 1
        found.append(((family, stable, version), name))
    found.sort(reverse=True)
    return [n for _, n in found]


def call_model(model, api_key, prompt):
    url = f"{API_BASE}/models/{urllib.parse.quote(model)}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 32768,
            "responseMimeType": "application/json",
        },
    }
    for attempt in range(3):
        try:
            res = _request("POST", url, api_key, body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            low = detail.lower()
            if e.code in (404, 410) or (e.code in (400, 403) and (
                    "model" in low or "not found" in low or "no longer" in low
                    or "deprecat" in low or "not supported" in low)):
                raise ModelUnavailable(f"{model}: HTTP {e.code} {detail}")
            if e.code == 429:
                raise ModelUnavailable(f"{model}: โควตาหมด (429)")
            if e.code in (500, 502, 503, 504) and attempt < 2:
                wait = 10 * (attempt + 1)
                log(f"  ! เซิร์ฟเวอร์ไม่ว่าง (HTTP {e.code}) รอ {wait} วินาทีแล้วลองใหม่")
                time.sleep(wait)
                continue
            raise RuntimeError(f"{model}: HTTP {e.code} {detail}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 2:
                log(f"  ! เชื่อมต่อไม่ได้ ({e}) ลองใหม่")
                time.sleep(5)
                continue
            raise
        cands = res.get("candidates") or []
        if not cands:
            raise RuntimeError(f"{model}: ไม่มีคำตอบ {json.dumps(res.get('promptFeedback', {}), ensure_ascii=False)}")
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise RuntimeError(f"{model}: คำตอบว่าง (finishReason={cands[0].get('finishReason')})")
        return text
    raise RuntimeError(f"{model}: เรียกไม่สำเร็จ")


def generate_with_fallback(prompt, api_key):
    """ลองรุ่นที่ตั้งไว้ก่อน ถ้าถูกปิดไปแล้วจะหารุ่นที่ยังใช้ได้มาแทนเอง คืนค่า (ข้อความ, ชื่อรุ่น)"""
    first = os.environ.get("GEMINI_MODEL", "").strip()
    tried = set()
    queue = ([first] if first else []) + PREFERRED_MODELS
    discovered = False
    errors = []
    while True:
        while queue:
            model = queue.pop(0)
            if model in tried:
                continue
            tried.add(model)
            log(f"- ลองใช้รุ่น {model}")
            try:
                return call_model(model, api_key, prompt), model
            except ModelUnavailable as e:
                log(f"  ! ใช้ไม่ได้: {str(e)[:200]}")
                errors.append(str(e)[:200])
        if discovered:
            break
        discovered = True
        log("- ค้นหารุ่นที่ยังเปิดให้ใช้อยู่จาก Gemini API ...")
        queue = [m for m in list_available_models(api_key) if m not in tried][:8]
        log(f"  พบ: {', '.join(queue) or '(ไม่มี)'}")
    raise RuntimeError("ไม่มีรุ่น Gemini ที่ใช้ได้เลย\n" + "\n".join(errors))


def mock_response(topic, requirements):
    """คำตอบจำลองสำหรับทดสอบ (ไม่เรียก Gemini)"""
    para = ("นี่คือย่อหน้าทดสอบเกี่ยวกับ " + topic + " ซึ่งสร้างขึ้นเพื่อทดสอบระบบอัตโนมัติ "
            "เนื้อหาจริงจะถูกเขียนโดย Gemini ตามหัวข้อและความต้องการที่ระบุไว้ ") * 3
    return json.dumps({
        "title": f"{topic} คู่มือฉบับเข้าใจง่าย",
        "slug": "mock-test-article",
        "meta_description": f"สรุปทุกเรื่องที่ควรรู้เกี่ยวกับ {topic} พร้อมตัวอย่างและแนวทางลงมือทำจริงสำหรับธุรกิจไทย",
        "keywords": [topic, "การตลาดออนไลน์", "AI", "ธุรกิจไทย", "อ.แชมป์"],
        "subtitle": f"เจาะลึก {topic} แบบเข้าใจง่ายและนำไปใช้ได้จริง",
        "excerpt": para[:220],
        "image_prompts": [
            "Thai business owner reading about marketing on laptop modern Bangkok office photorealistic",
            "Thai marketing team brainstorming with AI dashboard on screen photorealistic",
            "Happy Thai entrepreneur celebrating business growth chart photorealistic",
        ],
        "intro_html": f"<p>{para}</p>",
        "sections": [
            {"heading": "ทำไมเรื่องนี้จึงสำคัญ", "html": f"<p>{para}</p><ul><li>ข้อดีข้อแรก</li><li>ข้อดีข้อสอง</li></ul>"},
            {"heading": "ขั้นตอนเริ่มต้น", "html": f"<p>{para} <a href=\"top-ai-speakers-thailand-ajarn-champ.html\">วิทยากร AI</a></p>"},
            {"heading": "ข้อผิดพลาดที่พบบ่อย", "html": f"<p>{para}<script>alert(1)</script> <a href=\"javascript:alert(1)\">x</a></p>"},
            {"heading": "สรุปแนวทางลงมือทำ", "html": f"{para}\n\n{para}"},
        ],
        "faq": [
            {"question": f"{topic} เหมาะกับใคร", "answer": "เหมาะกับเจ้าของธุรกิจและนักการตลาดทุกระดับ"},
            {"question": "ต้องใช้งบประมาณเท่าไร", "answer": "เริ่มต้นได้จากเครื่องมือฟรีก่อน แล้วค่อยขยายตามผลลัพธ์"},
        ],
        "conclusion_html": f"<p>{para} ความต้องการเพิ่มเติม: {html.escape(requirements or '-')}</p>",
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def build_prompt(topic, requirements, existing_articles):
    links = "\n".join(f"- {href} : {title}" for href, title in existing_articles[:15])
    return f"""คุณคือนักเขียนบทความ SEO ภาษาไทยมืออาชีพของเว็บไซต์ {SITE_NAME}
เขียนบทความภาษาไทยคุณภาพสูง อ่านง่าย มีข้อมูลเชิงลึก ตัวอย่างจริง และนำไปใช้ได้จริง

หัวข้อ: {topic}
ความต้องการเพิ่มเติม: {requirements or "ไม่มี"}

ข้อกำหนด:
- ความยาวเนื้อหารวมประมาณ 1,500-2,500 คำ
- มี 4-6 หัวข้อย่อย (sections) แต่ละหัวข้อมีหลายย่อหน้า
- ใส่คีย์เวิร์ดหลักอย่างเป็นธรรมชาติ ไม่ยัดคีย์เวิร์ด
- ถ้าเหมาะสม ให้ลิงก์ภายในไปยังบทความเดิมของเว็บ (ใช้ href ตามรายการนี้เท่านั้น ห้ามแต่งลิงก์ขึ้นเอง):
{links}
- HTML ที่ใช้ได้มีแค่ <p> <strong> <em> <ul> <ol> <li> <h3> <a href> <br> ห้ามใช้ <h1> <h2> <img> <script> <style>
- ห้ามใส่ markdown

ตอบกลับเป็น JSON อย่างเดียว ตามโครงสร้างนี้:
{{
  "title": "ชื่อบทความ ไม่เกิน 65 ตัวอักษร มีคีย์เวิร์ดหลัก",
  "slug": "ชื่อไฟล์ภาษาอังกฤษ ตัวพิมพ์เล็ก คั่นด้วย - เช่น ai-marketing-guide",
  "meta_description": "คำอธิบาย 120-160 ตัวอักษร ชวนคลิก",
  "keywords": ["คีย์เวิร์ด 5-8 คำ"],
  "subtitle": "ประโยคโปรยใต้ชื่อเรื่อง",
  "excerpt": "สรุปสั้น 180-220 ตัวอักษร สำหรับการ์ดหน้าแรก",
  "image_prompts": ["คำอธิบายภาพภาษาอังกฤษ 3 ภาพ แบบ photorealistic เกี่ยวกับเนื้อหา"],
  "intro_html": "<p>บทนำ</p>",
  "sections": [{{"heading": "หัวข้อย่อย", "html": "<p>เนื้อหา</p>"}}],
  "faq": [{{"question": "คำถามที่พบบ่อย", "answer": "คำตอบ"}}],
  "conclusion_html": "<p>บทสรุป</p>"
}}
faq มี 3-5 ข้อ"""


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def validate(data):
    for key in ("title", "meta_description", "intro_html", "sections"):
        if not data.get(key):
            raise ValueError(f"คำตอบไม่มี '{key}'")
    if not isinstance(data["sections"], list) or len(data["sections"]) < 2:
        raise ValueError("หัวข้อย่อยน้อยเกินไป")


# ---------------------------------------------------------------------------
# ทำความสะอาด HTML จาก AI (กันโค้ดแปลกปลอม)
# ---------------------------------------------------------------------------
class Sanitizer(HTMLParser):
    ALLOWED = {"p", "strong", "b", "em", "i", "ul", "ol", "li", "h3", "a", "br"}
    DROP_CONTENT = {"script", "style", "iframe", "object", "noscript", "svg", "head", "title"}

    def __init__(self, valid_local):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.stack = []
        self.drop = 0
        self.valid_local = valid_local

    def _href_ok(self, href):
        href = (href or "").strip()
        if re.match(r"^https?://", href, re.I):
            return href
        local = href.lstrip("./")
        if re.fullmatch(r"[A-Za-z0-9_\-]+\.html", local) and local in self.valid_local:
            return local
        return None

    def handle_starttag(self, tag, attrs):
        if tag in self.DROP_CONTENT:
            self.drop += 1
            return
        if self.drop or tag not in self.ALLOWED:
            return
        if tag == "br":
            self.out.append("<br>")
            return
        if tag == "a":
            href = self._href_ok(dict(attrs).get("href"))
            if not href:
                self.stack.append(None)  # เก็บข้อความ ทิ้งลิงก์
                return
            ext = re.match(r"^https?://", href) and "rapidpress.info" not in href.lower()
            extra = ' target="_blank" rel="noopener"' if ext else ""
            self.out.append(f'<a href="{html.escape(href, quote=True)}"{extra}>')
            self.stack.append("a")
            return
        tag = {"b": "strong", "i": "em"}.get(tag, tag)
        self.out.append(f"<{tag}>")
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.DROP_CONTENT:
            self.drop = max(0, self.drop - 1)
            return
        if self.drop or tag not in self.ALLOWED or tag == "br":
            return
        tag = {"b": "strong", "i": "em"}.get(tag, tag)
        targets = {"a", None} if tag == "a" else {tag}
        if not any(t in targets for t in self.stack):
            return
        # ปิดแท็กที่ค้างอยู่จนถึงแท็กที่ตรงกัน
        while self.stack:
            top = self.stack.pop()
            if top is not None:
                self.out.append(f"</{top}>")
            if top in targets:
                return

    def handle_data(self, data):
        if not self.drop:
            self.out.append(html.escape(data, quote=False))

    def result(self):
        while self.stack:
            top = self.stack.pop()
            if top:
                self.out.append(f"</{top}>")
        return "".join(self.out).strip()


def clean_html(fragment, valid_local):
    fragment = str(fragment or "")
    if not re.search(r"<\s*(p|ul|ol|h3)\b", fragment, re.I):
        # เป็นข้อความธรรมดา แบ่งย่อหน้าตามบรรทัดว่าง
        paras = [p.strip() for p in re.split(r"\n\s*\n", fragment) if p.strip()]
        fragment = "".join(f"<p>{p}</p>" for p in paras)
    s = Sanitizer(valid_local)
    s.feed(fragment)
    s.close()
    return s.result()


def plain(text):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(str(text or "")))).strip()


# ---------------------------------------------------------------------------
# อ่านเว็บเดิม
# ---------------------------------------------------------------------------
def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def write(path, content):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def site_url():
    try:
        host = read(os.path.join(REPO_ROOT, "CNAME")).strip()
    except OSError:
        host = "rapidpress.info"
    return f"https://{host}"


def existing_articles(index_html):
    seen, items = set(), []
    for href, title in re.findall(r'<h3><a href="([^"]+\.html)">(.*?)</a></h3>', index_html):
        if href not in seen:
            seen.add(href)
            items.append((href, plain(title)))
    return items


def load_template():
    name = os.environ.get("TEMPLATE_FILE", DEFAULT_TEMPLATE)
    path = os.path.join(REPO_ROOT, name)
    if not os.path.exists(path):
        # หาไฟล์บทความที่มีโครงสร้าง article-full ไฟล์แรกที่เจอแทน
        for href, _ in existing_articles(read(os.path.join(REPO_ROOT, INDEX_FILE))):
            p = os.path.join(REPO_ROOT, href)
            if os.path.exists(p) and '<article class="article-full">' in read(p):
                path = p
                break
    tpl = read(path)
    if '<article class="article-full">' not in tpl or "</article>" not in tpl:
        raise RuntimeError(f"แม่แบบ {path} ไม่มี <article class=\"article-full\">")
    log(f"- ใช้แม่แบบ: {os.path.basename(path)}")
    return tpl


def make_slug(raw, title):
    slug = re.sub(r"[^a-z0-9]+", "-", str(raw or "").lower()).strip("-")[:70].strip("-")
    if len(slug) < 3:
        slug = "article-" + dt.datetime.now(BKK).strftime("%Y%m%d-%H%M")
    base, n = slug, 2
    while os.path.exists(os.path.join(REPO_ROOT, slug + ".html")):
        slug = f"{base}-{n}"
        n += 1
    return slug


def image_url(prompt):
    words = re.sub(r"[^A-Za-z0-9 ,\-]", " ", str(prompt)).split()
    return IMAGE_URL_PREFIX + "+".join(words[:25])


# ---------------------------------------------------------------------------
# สร้างหน้า HTML
# ---------------------------------------------------------------------------
def build_page(tpl, data, slug, base_url, model):
    e = lambda s: html.escape(plain(s), quote=True)  # noqa: E731
    now = dt.datetime.now(BKK).replace(microsecond=0).isoformat()
    url = f"{base_url}/{slug}.html"
    title = plain(data["title"])
    desc = plain(data["meta_description"])
    keywords = [plain(k) for k in (data.get("keywords") or []) if plain(k)]
    subtitle = plain(data.get("subtitle")) or desc
    images = [image_url(p) for p in (data.get("image_prompts") or []) if plain(p)]
    if not images:
        images = [image_url("Thai business marketing modern Bangkok office photorealistic")]
    valid_local = {href for href, _ in existing_articles(read(os.path.join(REPO_ROOT, INDEX_FILE)))}

    faq = [(plain(f.get("question")), plain(f.get("answer")))
           for f in (data.get("faq") or []) if isinstance(f, dict) and plain(f.get("question"))]

    schema = [
        {
            "@context": "https://schema.org",
            "@type": "BlogPosting",
            "headline": title,
            "description": desc,
            "keywords": ", ".join(keywords),
            "image": images,
            "datePublished": now,
            "dateModified": now,
            "inLanguage": "th-TH",
            "mainEntityOfPage": {"@type": "WebPage", "@id": url},
            "author": {"@type": "Organization", "name": SITE_NAME, "url": base_url + "/"},
            "publisher": {
                "@type": "Organization",
                "name": SITE_NAME,
                "logo": {"@type": "ImageObject", "url": f"{base_url}/images/RapidPress_logo.png"},
            },
        },
        {
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": [
                {"@type": "ListItem", "position": 1, "name": "หน้าแรก", "item": base_url + "/"},
                {"@type": "ListItem", "position": 2, "name": title, "item": url},
            ],
        },
    ]
    if faq:
        schema.append({
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [{"@type": "Question", "name": q,
                            "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in faq],
        })
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2).replace("</", "<\\/")

    head_meta = f"""    <meta name="description" content="{e(desc)}">
    <meta name="keywords" content="{e(', '.join(keywords))}">
    <meta name="robots" content="index, follow, max-image-preview:large">
    <link rel="canonical" href="{e(url)}">
    <meta property="og:type" content="article">
    <meta property="og:site_name" content="{SITE_NAME}">
    <meta property="og:title" content="{e(title)}">
    <meta property="og:description" content="{e(desc)}">
    <meta property="og:url" content="{e(url)}">
    <meta property="og:image" content="{e(images[0])}">
    <meta property="og:locale" content="th_TH">
    <meta property="article:published_time" content="{now}">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{e(title)}">
    <meta name="twitter:description" content="{e(desc)}">
    <meta name="twitter:image" content="{e(images[0])}">
    <meta name="generator" content="RapidPress auto-writer ({e(model)})">
    <script type="application/ld+json">
{schema_json}
    </script>"""

    # ---- head: ล้างค่าของบทความเดิมออก แล้วใส่ของใหม่ ----
    page = tpl
    page = re.sub(r"<title>.*?</title>", lambda _: f"<title>{e(title)} : {SITE_NAME}</title>", page, count=1, flags=re.S)
    page = re.sub(r'\s*<meta\s+(?:name|property)="(?:description|keywords|robots|generator|og:[^"]*|twitter:[^"]*|article:[^"]*)"[^>]*>', "", page)
    page = re.sub(r'\s*<link\s+rel="canonical"[^>]*>', "", page)
    page = re.sub(r'\s*<script type="application/ld\+json">.*?</script>', "", page, flags=re.S)
    page = re.sub(r'(<meta name="viewport"[^>]*>)', lambda m: m.group(1) + "\n" + head_meta, page, count=1)
    page = re.sub(r'<html lang="[^"]*">', '<html lang="th">', page, count=1)

    # ---- เนื้อหาบทความ ----
    img_tag = lambda src: (f'            <img src="{html.escape(src, quote=True)}" alt="{e(title)}" '  # noqa: E731
                           f'title="{e(subtitle)}" loading="lazy">\n\n')
    body = [
        '<article class="article-full">\n',
        f"            <h1>{e(title)}</h1>\n",
        f'            <p class="subtitle">{e(subtitle)}</p>\n\n',
        img_tag(images[0]).replace(' loading="lazy"', ""),
        f"            {clean_html(data['intro_html'], valid_local)}\n\n",
    ]
    sections = [s for s in data["sections"] if isinstance(s, dict) and plain(s.get("heading"))]
    extra_imgs = images[1:]
    for i, sec in enumerate(sections):
        body.append(f"            <h2>{e(sec['heading'])}</h2>{clean_html(sec.get('html'), valid_local)}\n\n")
        # แทรกรูปเป็นระยะ ๆ เหมือนบทความเดิม
        if extra_imgs and i % 2 == 1:
            body.append(img_tag(extra_imgs.pop(0)))
    if faq:
        body.append("            <h2>คำถามที่พบบ่อย (FAQ)</h2>")
        for q, a in faq:
            body.append(f"<h3>{html.escape(q)}</h3><p>{html.escape(a)}</p>")
        body.append("\n\n")
    if data.get("conclusion_html"):
        body.append(f"            {clean_html(data['conclusion_html'], valid_local)}\n")
    body.append("        </article>")
    article = "".join(body)

    page = re.sub(r'<article class="article-full">.*?</article>', lambda _: article, page, count=1, flags=re.S)
    return page, title, images[0]


def insert_card(index_html, slug, title, excerpt, img):
    t = html.escape(title, quote=True)
    excerpt = plain(excerpt)
    if len(excerpt) > 200:
        excerpt = excerpt[:200].rstrip() + "..."
    card = f"""
            <div class="list-h">
                <div class="list-detail">
                    <h3><a href="{slug}.html">{t}</a></h3>
                    <p>{html.escape(excerpt)}</p>
                </div>
                <figure>
                    <a href="{slug}.html">
                        <img src="{html.escape(img, quote=True)}" alt="{t}">
                    </a>
                </figure>
            </div>"""
    if INSERT_MARKER in index_html:
        return index_html.replace(INSERT_MARKER, INSERT_MARKER + card, 1)
    m = re.search(r'(<section class="article-col-highlights-side">\s*<h2>[^<]*</h2>)', index_html)
    if not m:
        raise RuntimeError("หาส่วน 'บทความล่าสุด' ใน index.html ไม่เจอ")
    return index_html[:m.end()] + card + index_html[m.end():]


def set_output(**kw):
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            for k, v in kw.items():
                f.write(f"{k}={str(v).replace(chr(10), ' ')}\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="สร้างบทความภาษาไทยด้วย Gemini")
    ap.add_argument("--topic", default=os.environ.get("ARTICLE_TOPIC", ""))
    ap.add_argument("--requirements", default=os.environ.get("ARTICLE_REQUIREMENTS", ""))
    ap.add_argument("--mock", action="store_true", help="ใช้คำตอบจำลอง ไม่เรียก Gemini")
    args = ap.parse_args()

    topic = args.topic.strip()
    if not topic:
        sys.exit("กรุณาระบุหัวข้อ (--topic)")
    requirements = args.requirements.strip()
    mock = args.mock or os.environ.get("GEMINI_MOCK") == "1"
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not mock and not api_key:
        sys.exit("ไม่พบ GEMINI_API_KEY (ใส่ไว้ใน GitHub Secrets)")

    index_path = os.path.join(REPO_ROOT, INDEX_FILE)
    index_html = read(index_path)
    tpl = load_template()
    prompt = build_prompt(topic, requirements, existing_articles(index_html))

    log(f"- หัวข้อ: {topic}")
    data, model = None, "mock"
    for attempt in range(1, 3):
        try:
            if mock:
                text = mock_response(topic, requirements)
            else:
                text, model = generate_with_fallback(prompt, api_key)
            data = parse_json(text)
            validate(data)
            break
        except (ValueError, json.JSONDecodeError) as err:
            log(f"  ! คำตอบไม่สมบูรณ์ ({err}) ครั้งที่ {attempt}")
            if attempt == 2:
                raise
    log(f"- เขียนเสร็จด้วยรุ่น {model}")

    slug = make_slug(data.get("slug"), data["title"])
    page, title, first_img = build_page(tpl, data, slug, site_url(), model)
    write(os.path.join(REPO_ROOT, slug + ".html"), page)
    write(index_path, insert_card(index_html, slug, title, data.get("excerpt") or data["intro_html"], first_img))

    url = f"{site_url()}/{slug}.html"
    log(f"- สร้างไฟล์ {slug}.html แล้ว")
    log(f"- ลิงก์: {url}")
    set_output(slug=slug, url=url, title=title, model=model)


if __name__ == "__main__":
    main()
