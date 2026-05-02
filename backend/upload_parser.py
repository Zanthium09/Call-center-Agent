# ==============================================================================
#  TETRANETICS — Conversation dataset parser
#  Parses uploaded files (jsonl / json / csv / txt / zip) into a uniform
#  internal shape so the worker can process all formats identically.
# ==============================================================================

import io
import json
import csv
import re
import zipfile
import uuid
from typing import Iterator, Tuple, List, Optional

# Each parser yields tuples of (conversation_dict, source_label)
# conversation_dict shape:
#   {
#     "id":      str,        # auto if missing
#     "date":    str | None,
#     "persona": str,        # default "Unknown"
#     "issue":   str,        # default "General Inquiry"
#     "outcome": str | None, # 'win' | 'loss' | None
#     "turns":   [ { "agent": str, "customer": str, "score": int? }, ... ]
#   }

MAX_CONVERSATIONS = 500
MAX_TURNS_PER_CONV = 60
MAX_REPLY_CHARS = 5000

DEFAULT_PERSONA = "Unknown"
DEFAULT_ISSUE = "General Inquiry"


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _coerce_int_score(v) -> Optional[int]:
    if v is None:
        return None
    try:
        n = int(round(float(v)))
        return max(0, min(10, n))
    except (TypeError, ValueError):
        return None


def _truncate(text: str) -> str:
    text = (text or "").strip()
    if len(text) > MAX_REPLY_CHARS:
        return text[:MAX_REPLY_CHARS] + "…"
    return text


def _normalise_conv(raw: dict, default_persona: str, default_issue: str,
                    fallback_id: str = None) -> Optional[dict]:
    """Take a raw parsed conversation dict and return a clean one,
    or None if it's empty / malformed."""
    if not isinstance(raw, dict):
        return None

    turns_in = raw.get("turns") or raw.get("messages") or []
    if not isinstance(turns_in, list):
        return None

    cleaned_turns: List[dict] = []
    for t in turns_in:
        if not isinstance(t, dict):
            continue
        agent = _truncate(t.get("agent") or t.get("agent_text")
                          or t.get("agent_reply") or "")
        customer = _truncate(t.get("customer") or t.get("customer_text")
                             or t.get("customer_reply") or t.get("user") or "")
        if not agent and not customer:
            continue
        out = {"agent": agent, "customer": customer}
        sc = _coerce_int_score(t.get("score"))
        if sc is not None:
            out["score"] = sc
        if t.get("tip"):
            out["tip"] = _truncate(t["tip"])
        cleaned_turns.append(out)

    if len(cleaned_turns) < 1:
        return None
    if len(cleaned_turns) > MAX_TURNS_PER_CONV:
        cleaned_turns = cleaned_turns[:MAX_TURNS_PER_CONV]

    return {
        "id":       str(raw.get("id") or fallback_id or uuid.uuid4()),
        "date":     raw.get("date") or raw.get("started_at"),
        "persona":  (raw.get("persona") or raw.get("customer_persona")
                     or default_persona).strip()[:80],
        "issue":    (raw.get("issue") or raw.get("issue_type")
                     or default_issue).strip()[:80],
        "outcome":  raw.get("outcome"),
        "turns":    cleaned_turns,
    }


# ── Format-specific parsers ───────────────────────────────────────────────────

def _parse_jsonl_text(text: str, default_persona: str,
                      default_issue: str) -> Iterator[Tuple[dict, str]]:
    """One JSON object per line."""
    for n, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except Exception:
            continue
        conv = _normalise_conv(
            raw, default_persona, default_issue,
            fallback_id=f"line_{n}",
        )
        if conv:
            yield conv, f"line {n}"


def _parse_json_text(text: str, default_persona: str,
                     default_issue: str) -> Iterator[Tuple[dict, str]]:
    """JSON object that is either a single conversation or an array of them.
    Also tolerates {"conversations": [...]}."""
    data = json.loads(text)
    if isinstance(data, dict):
        if isinstance(data.get("conversations"), list):
            data = data["conversations"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON root must be an object or a list")
    for n, raw in enumerate(data, start=1):
        conv = _normalise_conv(
            raw, default_persona, default_issue,
            fallback_id=f"item_{n}",
        )
        if conv:
            yield conv, f"item {n}"


def _parse_csv_text(text: str, default_persona: str,
                    default_issue: str) -> Iterator[Tuple[dict, str]]:
    """CSV with columns: conversation_id, agent, customer, [persona, issue, score, date].
    Rows are grouped by conversation_id (or 'cid' / 'id'). If no
    conversation_id column, the entire file is treated as ONE conversation."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return
    cid_field = None
    for cand in ("conversation_id", "cid", "call_id", "id"):
        if cand in reader.fieldnames:
            cid_field = cand
            break

    grouped: dict = {}  # cid -> conversation dict in raw form
    order: List[str] = []  # preserve insertion order

    for row in reader:
        agent = row.get("agent") or ""
        customer = row.get("customer") or row.get("user") or ""
        if not agent and not customer:
            continue

        cid = row.get(cid_field) if cid_field else "single"
        cid = (cid or "anon").strip() or "anon"
        if cid not in grouped:
            order.append(cid)
            grouped[cid] = {
                "id":      cid,
                "persona": row.get("persona") or row.get("customer_persona")
                             or default_persona,
                "issue":   row.get("issue")   or row.get("issue_type")
                             or default_issue,
                "date":    row.get("date") or row.get("started_at"),
                "outcome": row.get("outcome"),
                "turns":   [],
            }
        t = {"agent": agent, "customer": customer}
        if "score" in row and row["score"]:
            sc = _coerce_int_score(row["score"])
            if sc is not None:
                t["score"] = sc
        grouped[cid]["turns"].append(t)

    for cid in order:
        conv = _normalise_conv(grouped[cid], default_persona, default_issue,
                               fallback_id=cid)
        if conv:
            yield conv, f"cid {cid}"


_TXT_AGENT = re.compile(r"^\s*(agent|rep|support|me)\s*[:\-]\s*",
                        re.IGNORECASE)
_TXT_CUSTOMER = re.compile(r"^\s*(customer|caller|client|user|them)\s*[:\-]\s*",
                           re.IGNORECASE)
_TXT_SEPARATOR = re.compile(
    r"^\s*(?:={3,}|-{3,}|\*{3,}|conversation\s*\d+|call\s*\d+|---+)\s*$",
    re.IGNORECASE,
)


def _parse_txt_text(text: str, default_persona: str,
                    default_issue: str) -> Iterator[Tuple[dict, str]]:
    """Plain-text transcript with 'Agent:' / 'Customer:' prefixes.
    Conversations are separated by blank-block lines like '===' or 'Call 2'."""
    blocks: List[List[str]] = [[]]
    for line in text.splitlines():
        if _TXT_SEPARATOR.match(line):
            if blocks[-1]:
                blocks.append([])
            continue
        blocks[-1].append(line)

    for n, block_lines in enumerate(blocks, start=1):
        if not any(line.strip() for line in block_lines):
            continue
        turns = []
        cur_agent = cur_customer = ""
        for raw in block_lines:
            line = raw.rstrip()
            if not line.strip():
                continue
            if _TXT_AGENT.match(line):
                # If both halves filled, push the previous turn first
                if cur_agent and cur_customer:
                    turns.append({"agent": cur_agent, "customer": cur_customer})
                    cur_agent = cur_customer = ""
                cur_agent = (cur_agent + " " + _TXT_AGENT.sub("", line, 1)).strip()
            elif _TXT_CUSTOMER.match(line):
                cur_customer = (cur_customer + " "
                                + _TXT_CUSTOMER.sub("", line, 1)).strip()
                # When customer's reply lands after an agent line, complete a turn
                if cur_agent:
                    turns.append({"agent": cur_agent, "customer": cur_customer})
                    cur_agent = cur_customer = ""
            else:
                # Continuation of the most recent role
                if cur_customer and not cur_agent:
                    cur_customer += " " + line.strip()
                elif cur_agent:
                    cur_agent += " " + line.strip()
        if cur_agent or cur_customer:
            turns.append({"agent": cur_agent, "customer": cur_customer})

        if not turns:
            continue
        raw_conv = {
            "id": f"block_{n}",
            "persona": default_persona,
            "issue":   default_issue,
            "turns":   turns,
        }
        conv = _normalise_conv(raw_conv, default_persona, default_issue,
                               fallback_id=f"block_{n}")
        if conv:
            yield conv, f"block {n}"


# ── ZIP — recurse into each entry, dispatch by extension ─────────────────────

def _parse_zip_bytes(data: bytes, default_persona: str,
                     default_issue: str) -> Iterator[Tuple[dict, str]]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            ext = member.rsplit(".", 1)[-1].lower() if "." in member else ""
            try:
                with zf.open(member) as f:
                    content = f.read()
            except Exception:
                continue
            try:
                content_text = content.decode("utf-8", errors="replace")
            except Exception:
                continue

            if ext == "jsonl":
                yield from _parse_jsonl_text(content_text, default_persona,
                                             default_issue)
            elif ext == "json":
                try:
                    yield from _parse_json_text(content_text, default_persona,
                                                default_issue)
                except Exception:
                    continue
            elif ext == "csv":
                yield from _parse_csv_text(content_text, default_persona,
                                           default_issue)
            elif ext in ("txt", "log"):
                yield from _parse_txt_text(content_text, default_persona,
                                           default_issue)
            # silently skip other extensions inside the zip


# ── Public dispatch ──────────────────────────────────────────────────────────

def parse_dataset(
    raw_bytes: bytes,
    filename: str,
    default_persona: str = DEFAULT_PERSONA,
    default_issue: str = DEFAULT_ISSUE,
) -> Tuple[List[dict], str, List[str]]:
    """Detect format from filename + content, parse, return:
        (conversations, format_label, warnings)
    Caps the result at MAX_CONVERSATIONS."""
    name = (filename or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    warnings: List[str] = []

    if ext == "zip" or raw_bytes[:4] == b"PK\x03\x04":
        fmt = "zip"
        gen = _parse_zip_bytes(raw_bytes, default_persona, default_issue)
    else:
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_bytes.decode("utf-8", errors="replace")
            warnings.append("File contained non-UTF-8 bytes; replaced.")

        if ext == "jsonl" or "\n{" in text[:2048]:
            fmt = "jsonl"
            gen = _parse_jsonl_text(text, default_persona, default_issue)
        elif ext == "json" or text.lstrip().startswith(("{", "[")):
            fmt = "json"
            try:
                gen = list(_parse_json_text(text, default_persona, default_issue))
            except Exception:
                # Fall through to JSONL (some users mislabel multi-line JSON)
                fmt = "jsonl"
                gen = _parse_jsonl_text(text, default_persona, default_issue)
        elif ext == "csv" or "," in text.splitlines()[0:1][0:1] and (
                "agent" in text.lower()[:200]):
            fmt = "csv"
            gen = _parse_csv_text(text, default_persona, default_issue)
        elif ext in ("txt", "log") or _TXT_AGENT.search(text[:2000] or ""):
            fmt = "txt"
            gen = _parse_txt_text(text, default_persona, default_issue)
        else:
            # Last resort: try JSONL, then plain text
            fmt = "jsonl"
            gen = _parse_jsonl_text(text, default_persona, default_issue)

    convos: List[dict] = []
    for conv, _label in gen:
        convos.append(conv)
        if len(convos) >= MAX_CONVERSATIONS:
            warnings.append(
                f"File contained more than {MAX_CONVERSATIONS} conversations; "
                f"only the first {MAX_CONVERSATIONS} were imported."
            )
            break

    return convos, fmt, warnings
