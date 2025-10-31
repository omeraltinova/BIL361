#!/usr/bin/env python3
"""
CLI aracı: OpenRouter üzerinden seçtiğiniz modele istek atar ve örnek SMS veri seti üretir.

Özellikler:
- OpenRouter Chat Completions API'ını kullanır (base: https://openrouter.ai/api/v1)
- Kimlik doğrulama: .env (OPENROUTER_API_KEY) veya ortam değişkeni
- Varsayılan model: .env (OPENROUTER_DEFAULT_MODEL) ile ayarlanabilir
- Varsayılan satır ve çalışma sayısı: .env (OPENROUTER_DEFAULT_ROWS, OPENROUTER_DEFAULT_RUNS)
- Seed destekli çoğaltma: --seed veya .env (OPENROUTER_DEFAULT_SEED) ile seed CSV'yi referans alır
- Çıktı: Test klasörü altında yeni, benzersiz isimli bir CSV dosyası (mevcut dosyalara dokunmaz)

Kullanım örneği (.env ile):
    # Proje köküne .env oluşturun
    # içerik:
    # OPENROUTER_API_KEY="<api-key>"
    # OPENROUTER_DEFAULT_MODEL="meta-llama/llama-3.1-8b-instruct"
    # OPENROUTER_DEFAULT_ROWS=20
    # OPENROUTER_DEFAULT_RUNS=5
    # OPENROUTER_DEFAULT_SEED="Test/seed.csv"
    python Test/generate_ai_dataset.py \
        --language tr
    # Yukarıdaki örnek toplam 20*5=100 satır üretir (5 ayrı istekte), tek CSV'de birleştirilir.
    # Ayrıca seed varsa model, üreteceği veriyi seed'e benzer şekilde çoğaltır.

Not: Model adlarını görmek için: https://openrouter.ai/models
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict
import csv
import io


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def parse_int_env(key: str, default: int) -> int:
    """Ortam değişkeninden pozitif int okur; yoksa default döner."""
    val = os.getenv(key)
    if val is None or val == "":
        return default
    try:
        parsed = int(val)
        return parsed
    except Exception:
        return default


def load_env_from_file(env_path: Path, override: bool = False) -> None:
    """Basit .env yükleyici: KEY=VALUE satırlarını okur ve ortam değişkeni olarak ayarlar.

    - # ile başlayan veya boş satırları yok sayar
    - Çift veya tek tırnaklı değerleri soyup ayarlar
    - Varsayılan olarak mevcut ortam değişkenlerini EZMEZ
    """
    try:
        if not env_path.exists() or not env_path.is_file():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            if override or key not in os.environ:
                os.environ[key] = value
    except Exception:
        # .env yükleme opsiyoneldir; hata olsa bile akışı bozmayalım
        pass


def load_env() -> None:
    """.env dosyasını olası konumlardan yükler.

    Öncelik sırası:
    1) Çalışma dizini: .env
    2) Proje kökü (script'in bir üstü): .env
    3) Script dizini: .env
    """
    candidates = []
    try:
        candidates.append(Path.cwd() / ".env")
    except Exception:
        pass
    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir.parent / ".env")
    candidates.append(script_dir / ".env")

    seen = set()
    for p in candidates:
        if p and str(p) not in seen:
            load_env_from_file(p)
            seen.add(str(p))


def ensure_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY ortam değişkeni bulunamadı. Lütfen anahtarınızı ayarlayın."
        )
    return api_key


def http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]):
    """requests varsa onu, yoksa urllib ile POST yapar."""
    try:
        import requests  # type: ignore

        return requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception:
        # Minimal urllib fallback
        import urllib.request  # type: ignore

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        for k, v in headers.items():
            req.add_header(k, v)

        class _Resp:
            def __init__(self, code: int, body: str):
                self.status_code = code
                self.text = body

            def json(self):
                return json.loads(self.text)

        with urllib.request.urlopen(req, timeout=60) as resp:  # nosec - CLI amaçlı
            body = resp.read().decode("utf-8")
            return _Resp(resp.getcode(), body)


def strip_code_fences(text: str) -> str:
    """Geri dönen içerikte varsa ```...``` bloklarını temizler."""
    fence_pattern = re.compile(r"```(?:[a-zA-Z]+)?\n([\s\S]*?)\n```", re.MULTILINE)
    match = fence_pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def ensure_csv_header(content: str) -> str:
    """İlk satırda label,text başlığı yoksa ekler."""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if not lines:
        return "label,text\n"  # boşsa en az başlık dön

    first = lines[0].strip().lower().replace(" ", "")
    if first.startswith("label,") and ",text" in first:
        return "\n".join(lines) + "\n"
    # değilse başlık ekle
    return "label,text\n" + "\n".join(lines) + "\n"


def aggregate_csv(chunks: list[str]) -> str:
    """Birden çok CSV metnini tek bir CSV'de toplar; başlığı tekilleştirir."""
    if not chunks:
        return "label,text\n"

    output_rows: list[str] = []
    header_written = False

    for raw in chunks:
        if not raw:
            continue
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            continue

        first_norm = lines[0].strip().lower().replace(" ", "")
        has_header = first_norm.startswith("label,") and ",text" in first_norm
        start_index = 1 if has_header else 0

        if not header_written:
            output_rows.append("label,text")
            header_written = True

        output_rows.extend(lines[start_index:])

    if not header_written:
        return "label,text\n"
    return "\n".join(output_rows) + "\n"


def mask_sensitive_text(text: str) -> str:
    """Linkleri ve muhtemel şirket/kurum adlarını maskeleyerek döndürür."""
    if not text:
        return text

    out = text

    # 1) URL ve alan adlarını maskele
    url_patterns = [
        re.compile(r"(?i)\b(?:https?://|http://|www\.)\S+\b"),
        re.compile(r"(?i)\b[\w.-]+\.(?:com|net|org|io|ai|co|uk|tr|de|ru|cn|xyz|info|biz|me|ly|app|dev|top|live|shop|club|site|link|click|gg|tv|in)(?:/\S*)?\b"),
    ]
    for pat in url_patterns:
        out = pat.sub("[LINK_MASKED]", out)

    # 2) Şirket/marka/kategori anahtarlarını maskele (basit sezgisel)
    company_keywords = [
        "yurtici",
        "yurtiçi",
        "casino",
        "otel",
        "bankası",
        "bankasi",
        "bank",
        "kargo",
        "telekom",
    ]
    # Anahtar sözcüğü içeren tüm kelime parçalarını tek parça halinde maskele
    comp_pat = re.compile(
        r"(?i)(?<!\[)\b[\w\-]*?(" + "|".join(re.escape(k) for k in company_keywords) + r")[\w\-]*\b"
    )
    out = comp_pat.sub("[FIRMA_MASKED]", out)

    return out


def sanitize_csv(content: str) -> str:
    """CSV'yi okuyup metin alanına maskeleme uygular ve tekrar CSV olarak yazar."""
    if not content:
        return "label,text\n"

    input_io = io.StringIO(content)
    reader = csv.reader(input_io)

    rows = list(reader)
    if not rows:
        return "label,text\n"

    output_io = io.StringIO()
    writer = csv.writer(output_io)
    # Başlık
    writer.writerow(["label", "text"])

    # Veriler
    for row in rows[1:] if rows[0] and len(rows[0]) >= 2 else rows:
        if not row:
            continue
        label = (row[0] if len(row) >= 1 else "").strip()
        text_field = (
            row[1] if len(row) >= 2 else ("" if len(row) == 1 else ",".join(row[1:]))
        )
        if len(row) > 2:
            # Beklenmedik fazla kolonları metne birleştir
            text_field = ",".join(row[1:])

        masked_text = mask_sensitive_text(text_field)
        writer.writerow([label, masked_text])

    return output_io.getvalue()

def read_seed_excerpt(seed_path: Path, max_lines: int = 200, max_bytes: int = 4096) -> str:
    """Seed CSV'den makul büyüklükte bir alıntı döner (başlık + ilk satırlar)."""
    content = seed_path.read_text(encoding="utf-8")
    # Byte sınırlaması uygula
    if len(content.encode("utf-8")) > max_bytes:
        # Kabaca karakter bazında kes; sonra satır sonuna kadar kısalt
        truncated = content.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        # Son tam satıra indir
        last_nl = truncated.rfind("\n")
        if last_nl > 0:
            content = truncated[:last_nl]
        else:
            content = truncated
    # Satır limiti uygula
    lines = content.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)


def build_prompt(num_rows: int, language: str, seed_csv_excerpt: str | None = None) -> str:
    """Model için açık ve katı çıktı talimatı oluşturur; seed varsa referans verir."""
    base = (
        "Aşağıdaki gereksinimlerle sentetik SMS veri seti üret:\n"
        "- ÇIKTI FORMAT: Sadece CSV metni üret. Kod bloğu, açıklama, ön/son metin ekleme.\n"
        "- BAŞLIK: label,text\n"
        f"- SATIR SAYISI: Tam olarak {num_rows} satır.\n"
        "- label: 'spam' veya 'ham' değerlerinden biri olmalı.\n"
        f"- text: {language} dilinde gerçekçi SMS içeriği yaz.\n"
        "- CSV kurallarına uygun şekilde gerekiyorsa metni tırnakla.\n"
        "- Virgül, yeni satır gibi karakterlerde CSV uyumunu koru.\n"
        "- ZORUNLU MASKELEME: Tüm URL/alan adlarını [LINK_MASKED] olarak yaz.\n"
        "- ZORUNLU MASKELEME: Şirket/marka/kurum adlarını [FIRMA_MASKED] olarak yaz.\n"
        "- ASLA gerçek link veya gerçek isim verme.\n"
        "- Her satır seed ile TUTARLI ama birebir KOPYA OLMAYAN yeni örnek olsun.\n"
        "- Varyasyon üret; tekrar etmeyi en aza indir.\n"
        "- Yalnızca ham CSV döndür.\n"
    )
    if seed_csv_excerpt:
        base += (
            "\nSeed CSV (referans, kopyalama yapma):\n"
            "```csv\n" + seed_csv_excerpt.strip() + "\n```\n"
        )
    return base


def request_openrouter_csv(model: str, num_rows: int, language: str, seed_csv_excerpt: str | None = None) -> str:
    api_key = ensure_api_key()
    prompt = build_prompt(num_rows=num_rows, language=language, seed_csv_excerpt=seed_csv_excerpt)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter iyi uygulama başlıkları (isteğe bağlı ama faydalı)
        "X-Title": "BIL361 Data Generator",
        "HTTP-Referer": "https://github.com/omeraltinova/BIL361",  # kendi kaynağınızı yazabilirsiniz
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise data generator. Output exactly and only the requested format.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 5000,
    }

    resp = http_post_json(OPENROUTER_API_URL, headers, payload)
    if getattr(resp, "status_code", 0) != 200:
        msg = getattr(resp, "text", "<no body>")
        raise RuntimeError(f"OpenRouter isteği başarısız oldu ({resp.status_code}): {msg}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 - JSON şekli beklenmeyebilir
        raise RuntimeError(f"Beklenmeyen yanıt: {json.dumps(data)[:500]}") from exc

    content = strip_code_fences(content)
    content = ensure_csv_header(content)
    content = sanitize_csv(content)
    return content


def default_output_path() -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).resolve().parent
    return script_dir / f"ai_generated_sms_{ts}.csv"


def write_unique(path: Path, content: str) -> Path:
    """Var olanı ezmeden benzersiz bir dosyaya yazar."""
    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    candidate.write_text(content, encoding="utf-8")
    return candidate


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenRouter ile sentetik SMS CSV veri seti üretir.",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=None,
        help=(
            "Kullanılacak model adı (örn: meta-llama/llama-3.1-8b-instruct, openai/gpt-4o-mini). "
            "Boş bırakılırsa OPENROUTER_DEFAULT_MODEL (.env) kullanılır. "
            "Tüm modeller: https://openrouter.ai/models"
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help=(
            "Her istekte üretilecek satır sayısı. Boşsa OPENROUTER_DEFAULT_ROWS (.env) veya 20 kullanılır."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=(
            "Çalışma sayısı (ayrı istek sayısı). Boşsa OPENROUTER_DEFAULT_RUNS (.env) veya 1 kullanılır."
        ),
    )
    parser.add_argument(
        "--language",
        choices=["tr", "en"],
        default="tr",
        help="SMS metin dili (tr veya en). Varsayılan: tr",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Çıktı CSV yolu (varsayılan: Test klasöründe zaman damgalı dosya)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help=(
            "Seed CSV yolu. Verilen örnekleri referans alarak benzer yeni veriler üretir. "
            "Boşsa OPENROUTER_DEFAULT_SEED (.env) kullanılır."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    # .env (varsa) yükle
    load_env()
    args = parse_args(argv)

    try:
        model = args.model or os.getenv("OPENROUTER_DEFAULT_MODEL")
        if not model:
            raise RuntimeError(
                "Model belirtilmedi. '--model' parametresi verin veya .env içine OPENROUTER_DEFAULT_MODEL ekleyin."
            )

        rows = args.rows if args.rows is not None else parse_int_env("OPENROUTER_DEFAULT_ROWS", 20)
        runs = args.runs if args.runs is not None else parse_int_env("OPENROUTER_DEFAULT_RUNS", 1)

        if rows <= 0:
            raise RuntimeError("'--rows' 1 veya daha büyük olmalı (varsayılan 20).")
        if runs <= 0:
            raise RuntimeError("'--runs' 1 veya daha büyük olmalı (varsayılan 1).")

        # Seed'i hazırla (varsa)
        seed_path_str = args.seed or os.getenv("OPENROUTER_DEFAULT_SEED")
        seed_excerpt: str | None = None
        if seed_path_str:
            sp = Path(seed_path_str)
            if not sp.exists() or not sp.is_file():
                raise RuntimeError(f"Seed bulunamadı: {sp}")
            seed_excerpt = read_seed_excerpt(sp)

        parts: list[str] = []
        for _ in range(runs):
            part = request_openrouter_csv(
                model=model,
                num_rows=rows,
                language=args.language,
                seed_csv_excerpt=seed_excerpt,
            )
            parts.append(part)

        csv_content = aggregate_csv(parts)
        csv_content = sanitize_csv(csv_content)
    except Exception as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1

    out_path = (
        Path(args.output).resolve()
        if args.output
        else default_output_path()
    )

    # Yalnızca yeni dosya oluştur; var olanları asla ezme
    try:
        final_path = write_unique(out_path, csv_content)
    except Exception as exc:
        print(f"Çıktı yazılırken hata: {exc}", file=sys.stderr)
        return 1

    print(f"Oluşturuldu: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


