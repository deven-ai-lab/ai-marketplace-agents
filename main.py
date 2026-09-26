import os
import re
import uuid
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Callable
from types import SimpleNamespace
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel, field_validator
from anthropic import AsyncAnthropic

# ============ CONFIGURATION ============
DATABASE_URL = os.getenv("DATABASE_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

EMAIL_MODEL = "claude-sonnet-5"   # Model used for all pitch emails
BATCH_SIZE = 5                    # Items per Claude call (quality sweet spot)
MAX_CONCURRENT = 5                # Batches running at the same time (protects rate limits)
MAX_TOKENS = 16000                # Room for adaptive thinking + emails

# ============ APP SETUP ============
app = FastAPI(
    title="AI Marketplace Agents",
    description="Ananya (Brand Partnerships, internally 'steve'), Fred (Matcher), Aditya (Creator Manager)",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ DATABASE ============
engine = create_engine(
    DATABASE_URL,
    echo=DEBUG,
    pool_pre_ping=True,      # test each pooled connection before use; replace it if Supabase closed it
    pool_recycle=280,        # never reuse a connection older than ~5 minutes
    connect_args={
        "connect_timeout": 10,       # fail fast instead of hanging when the database can't be reached
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Brand(Base):
    __tablename__ = "brands"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    brand_id = Column(String, unique=True, index=True)
    brand_name = Column(String)
    industry = Column(String)
    email = Column(String)
    phone = Column(String, nullable=True)
    website = Column(String, nullable=True)
    basic_info = Column(Text)
    budget = Column(Integer, nullable=True)
    platform = Column(String, nullable=True)
    timeline_days = Column(Integer, nullable=True)
    niche = Column(String, nullable=True)
    requirements = Column(Text, nullable=True)
    email_sent_date = Column(DateTime, nullable=True)
    response_received = Column(String, default="false")
    approval_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)


class Creator(Base):
    __tablename__ = "creators"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    creator_id = Column(String, unique=True, index=True)
    name = Column(String)
    platform = Column(String)
    handle = Column(String)
    followers = Column(Integer, nullable=True)
    engagement_rate = Column(String, nullable=True)
    niche = Column(String, nullable=True)
    email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    restrictions = Column(Text, nullable=True)
    min_budget = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ============ PYDANTIC MODELS ============
class SheetRow(BaseModel):
    """
    Base for rows coming from Google Sheets:
    - blank cells ("") become None
    - numbers with commas or % ("4,50,000", "4.2%") are cleaned
    - phone numbers stored as numbers become text
    """
    @field_validator("*", mode="before")
    @classmethod
    def blank_to_none(cls, value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class PitchBrandInput(SheetRow):
    brand_id: str
    brand_name: str
    industry: Optional[str] = None
    email: str
    phone: Optional[str] = None
    website: Optional[str] = None
    basic_info: Optional[str] = None

    @field_validator("phone", mode="before")
    @classmethod
    def phone_to_text(cls, value):
        return None if value is None else str(value).strip()


class BatchPitchRequest(BaseModel):
    action: str
    brands: List[PitchBrandInput]


class BrandResponse(BaseModel):
    brand_id: str
    brand_name: str
    email: str
    pitch_email: str
    email_sent_date: str
    status: str


class PitchCreatorInput(SheetRow):
    creator_id: str
    creator_name: str
    platform: Optional[str] = None
    handle: Optional[str] = None
    email: str
    phone: Optional[str] = None
    basic_info: Optional[str] = None
    follower_count: Optional[int] = None
    engagement_rate: Optional[float] = None
    comments: Optional[str] = None

    @field_validator("phone", mode="before")
    @classmethod
    def phone_to_text(cls, value):
        return None if value is None else str(value).strip()

    @field_validator("follower_count", "engagement_rate", mode="before")
    @classmethod
    def clean_number(cls, value):
        if isinstance(value, str):
            cleaned = value.replace(",", "").replace("%", "").strip()
            return cleaned or None
        return value


class BatchCreatorPitchRequest(BaseModel):
    action: str
    creators: List[PitchCreatorInput]


class CreatorResponse(BaseModel):
    creator_id: str
    creator_name: str
    email: str
    pitch_email: str
    email_sent_date: str
    status: str


class ParseReplyRequest(BaseModel):
    contact_id: str                        # brand_id or creator_id
    contact_name: Optional[str] = None
    from_email: Optional[str] = None
    reply_subject: Optional[str] = None
    reply_body: str
    original_pitch: Optional[str] = None   # our pitch email, gives Claude context


class CampaignAnalysisRequest(BaseModel):
    brand_id: str
    brand_name: str
    budget: int
    timeline_days: int
    niche: str
    requirements: str
    target_audience: Optional[str] = None


# ============ CLAUDE HELPERS ============
async_client = AsyncAnthropic()


async def call_claude(system_prompt: str, user_message: str, model: Optional[str] = None) -> str:
    """
    Call Claude and return only the text output.
    - Skips thinking blocks (adaptive thinking is on by default)
    - Fails loudly if output was cut off by max_tokens
    """
    response = await async_client.messages.create(
        model=model or EMAIL_MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    output = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()

    if response.stop_reason == "max_tokens":
        raise Exception(f"Output cut off at max_tokens={MAX_TOKENS}")
    if not output:
        raise Exception(f"No text returned (stop_reason={response.stop_reason})")

    return output


def clean_json_text(response_text: str) -> str:
    """Remove markdown code fences if present"""
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text.strip()


def parse_pitches(response_text: str, id_key: str, text_key: str = "pitch_email") -> list:
    """Parse Claude's JSON array, with repair fallbacks"""
    response_text = clean_json_text(response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Fallback 1: fix unescaped quotes inside pitch_email
    repaired = response_text.replace('\\"', '__ESCAPED_QUOTE__')
    repaired = re.sub(
        r'"' + text_key + r'":\s*"([^"]*)"',
        lambda m: f'"{text_key}": "{m.group(1).replace(chr(34), chr(92) + chr(34))}"',
        repaired
    )
    repaired = repaired.replace('__ESCAPED_QUOTE__', '\\"')
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Fallback 2: extract individual objects
    pitches = []
    pattern = r'\{[^}]*?"' + id_key + r'"[^}]*?"' + text_key + r'"[^}]*?\}'
    for match in re.findall(pattern, response_text, re.DOTALL):
        try:
            pitches.append(json.loads(match))
        except json.JSONDecodeError:
            pass
    return pitches


async def generate_pitches_parallel(
    items: list,
    id_key: str,
    format_item: Callable,
    system_prompt: str,
    label: str,
    text_key: str = "pitch_email",
    task: str = "pitch emails"
) -> dict:
    """
    Shared engine for Steve and Aditya:
    1. Split items into batches of BATCH_SIZE
    2. Run batches in parallel (max MAX_CONCURRENT at once)
    3. Retry any missing item individually (also in parallel)
    Returns {item_id: pitch_email}
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def run_batch(batch: list) -> list:
        items_text = "\n\n".join(format_item(i, item) for i, item in enumerate(batch))
        user_message = f"""Write {task} for ALL {len(batch)} {label} provided below. Return exactly {len(batch)} entries in the JSON array - one for each.

{items_text}

Do not skip anyone. Return ONLY the JSON array, no other text."""
        async with semaphore:
            try:
                return parse_pitches(await call_claude(system_prompt, user_message), id_key, text_key)
            except Exception as e:
                print(f"[{label}] batch failed ({[getattr(x, id_key) for x in batch]}): {e}")
                return []

    valid_ids = {getattr(item, id_key) for item in items}
    pitch_map = {}

    def collect(results: list):
        for batch_pitches in results:
            for pitch in batch_pitches:
                pid = pitch.get(id_key)
                # Keep only IDs we actually sent, first valid pitch wins
                if pid in valid_ids and pid not in pitch_map and pitch.get(text_key):
                    pitch_map[pid] = pitch[text_key]

    # Pass 1: all batches in parallel
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    collect(await asyncio.gather(*(run_batch(b) for b in batches)))

    # Pass 2: retry missing items one by one, in parallel
    missing = [item for item in items if getattr(item, id_key) not in pitch_map]
    if missing:
        print(f"[{label}] retrying {len(missing)} missing: {[getattr(x, id_key) for x in missing]}")
        collect(await asyncio.gather(*(run_batch([m]) for m in missing)))

    return pitch_map


# ============ HEALTH CHECK ============
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if DATABASE_URL:
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}
    return {"status": "healthy"}


# ============ STEVE: BRAND MANAGER AGENT ============
STEVE_SYSTEM_PROMPT = """You are Ananya, the Brand Partnerships Manager at an influencer marketing agency in India.

Your role: Generate compelling pitch emails to brands interested in creator partnerships.

You have access to 200+ micro and macro influencers across various niches (sports, fitness, beauty, tech, lifestyle, gaming, etc).

For each brand provided, generate a professional pitch email that:
1. Opens with the brand's context
2. Explains what you do (connect brands with influencers)
3. Highlights relevant creators in their niche
4. Asks for their campaign details (budget, platform, timeline)
5. Calls them to action

Email must be:
- Professional but friendly
- Concise (under 200 words)
- Personalized to their industry
- Include a clear call-to-action
- Sign off exactly as:
  Ananya
  Brand Partnerships

CRITICAL REQUIREMENT: Return exactly as many pitch emails as brands provided. Do NOT skip anyone.

IMPORTANT: Return ONLY a valid JSON array. No preamble, no explanation.

Format:
[
  {
    "brand_id": "NIKE-001",
    "brand_name": "Nike India",
    "pitch_email": "Subject: Creator Partnership Opportunity - Nike India\\n\\nDear Nike Team,..."
  }
]
"""


def format_brand(i: int, brand: PitchBrandInput) -> str:
    return f"""Brand #{i+1}:
- ID: {brand.brand_id}
- Name: {brand.brand_name}
- Industry: {brand.industry}
- Email: {brand.email}
- Website: {brand.website}
- Basic Info: {brand.basic_info}"""


@app.post("/agent/steve/generate-pitches-batch")
async def generate_pitches_batch(request: BatchPitchRequest):
    """
    STEVE: Generate pitch emails for brands.
    Batches of 5, run in parallel, missing brands retried individually.
    """
    if request.action != "send_pitch_emails_batch":
        raise HTTPException(status_code=400, detail="Invalid action")

    try:
        pitch_map = await generate_pitches_parallel(
            request.brands, "brand_id", format_brand, STEVE_SYSTEM_PROMPT, "brands"
        )

        email_sent_date = datetime.utcnow().isoformat()
        results = [
            BrandResponse(
                brand_id=b.brand_id,
                brand_name=b.brand_name,
                email=b.email or "",
                pitch_email=pitch_map[b.brand_id],
                email_sent_date=email_sent_date,
                status="pitch_generated"
            ).dict()
            for b in request.brands if b.brand_id in pitch_map
        ]
        failed_ids = [b.brand_id for b in request.brands if b.brand_id not in pitch_map]

        return {
            "status": "success" if not failed_ids else "partial_success",
            "action": "send_pitch_emails_batch",
            "total_brands": len(request.brands),
            "total_generated": len(results),
            "failed_ids": failed_ids,
            "pitches": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating brand pitches: {str(e)}")


# ============ ADITYA: CREATOR MANAGER AGENT ============
ADITYA_SYSTEM_PROMPT = """You are Aditya, the Creator Manager Agent for an AI-powered influencer marketing agency.

Your role: Generate compelling pitch emails to creators interested in brand collaborations.

You represent premium brands looking for authentic creator partnerships across multiple niches.

For each creator provided, generate a professional pitch email that:
1. Opens with the creator's context
2. Explains what you do (connect creators with premium brands)
3. Highlights the opportunity (paid collaborations, exposure, products)
4. Asks for their details (min budget, restrictions, availability, best format)
5. Makes clear that joining is completely free: we never charge creators anything,
   and they keep their full payout because the brand pays us
6. Calls them to action

Email must be:
- Professional but friendly
- Concise (under 200 words)
- Personalized to their platform and niche
- Include a clear call-to-action

CRITICAL REQUIREMENT: Return exactly as many pitch emails as creators provided. Do NOT skip anyone.

IMPORTANT: Return ONLY a valid JSON array. No preamble, no explanation.

Format:
[
  {
    "creator_id": "CREATOR-001",
    "creator_name": "Ali Khan",
    "pitch_email": "Subject: Brand Collaboration Opportunity for @alikhan\\n\\nHi Ali,..."
  }
]
"""


def format_creator(i: int, creator: PitchCreatorInput) -> str:
    return f"""Creator #{i+1}:
- ID: {creator.creator_id}
- Name: {creator.creator_name}
- Platform: {creator.platform}
- Handle: {creator.handle}
- Email: {creator.email}
- Followers: {creator.follower_count}
- Engagement Rate: {creator.engagement_rate}%
- Basic Info: {creator.basic_info}
- Notes: {creator.comments}"""


@app.post("/agent/aditya/generate-pitches-batch")
async def aditya_generate_creator_pitches(request: BatchCreatorPitchRequest):
    """
    ADITYA: Generate pitch emails for creators.
    Batches of 5, run in parallel, missing creators retried individually.
    """
    if request.action != "send_creator_pitches_batch":
        raise HTTPException(status_code=400, detail="Invalid action")

    try:
        pitch_map = await generate_pitches_parallel(
            request.creators, "creator_id", format_creator, ADITYA_SYSTEM_PROMPT, "creators"
        )

        email_sent_date = datetime.utcnow().isoformat()
        results = [
            CreatorResponse(
                creator_id=c.creator_id,
                creator_name=c.creator_name,
                email=c.email or "",
                pitch_email=pitch_map[c.creator_id],
                email_sent_date=email_sent_date,
                status="pitch_generated"
            ).dict()
            for c in request.creators if c.creator_id in pitch_map
        ]
        failed_ids = [c.creator_id for c in request.creators if c.creator_id not in pitch_map]

        return {
            "status": "success" if not failed_ids else "partial_success",
            "action": "send_creator_pitches_batch",
            "total_creators": len(request.creators),
            "total_generated": len(results),
            "failed_ids": failed_ids,
            "pitches": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating creator pitches: {str(e)}")


# ============ REPLY PARSING (STEVE + ADITYA) ============
VALID_INTENTS = {"interested", "not_interested", "needs_info", "auto_reply", "unsubscribe", "other"}


def strip_quoted_reply(body: str) -> str:
    """Remove the quoted original email so Claude only reads the new reply"""
    lines = []
    for line in body.splitlines():
        stripped = line.strip()
        # Gmail / Outlook style markers that start the quoted section
        if re.match(r"^On .+wrote:$", stripped) or stripped.startswith("-----Original Message-----"):
            break
        if stripped.startswith(">"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned if cleaned else body.strip()


def parse_json_object(response_text: str) -> dict:
    """Parse a single JSON object from Claude's output"""
    response_text = clean_json_text(response_text)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start, end = response_text.find("{"), response_text.rfind("}")
        if start != -1 and end > start:
            return json.loads(response_text[start:end + 1])
        raise


def to_days(value):
    """Turn a timeline into days: 14, "14", "14 days", "2 weeks", "1 month", "a week" -> int, else None"""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    t = str(value).strip().lower()
    per_unit = {"day": 1, "week": 7, "month": 30}
    # Digits, with an optional unit: "14", "14 days", "2 weeks", "3.5 weeks"
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(day|week|month)?", t)
    if match:
        days = float(match.group(1)) * per_unit[match.group(2) or "day"]
        return int(round(days)) if days > 0 else None
    # Words only count WITH a unit: "a week", "two weeks" (so "asap" stays empty)
    words = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
    match = re.search(r"\b(a|an|one|two|three|four|five|six)\s+(day|week|month)s?\b", t)
    if match:
        return int(words[match.group(1)] * per_unit[match.group(2)])
    return None


def to_int_or_none(value):
    try:
        return int(float(value)) if value not in (None, "") else None
    except (ValueError, TypeError):
        return None


REPLY_RULES = """
INTENT (pick exactly one):
- "interested": wants to proceed or shares details
- "not_interested": declines
- "needs_info": asks questions before deciding
- "auto_reply": out-of-office, automated or bounce message (NOT a real response)
- "unsubscribe": asks to stop receiving emails
- "other": anything else

BUDGET RULES:
- Convert to an integer in INR. "2 lakh" = 200000, "50k" = 50000, "1.5L" = 150000, "1 crore" = 10000000
- If a foreign currency is used, keep the number and mention the currency in "notes"
- If not mentioned, use null. Never guess a number.

Only extract what the reply actually says. Use null for anything not stated.
Return ONLY a valid JSON object. No preamble, no markdown.
"""

STEVE_PARSE_PROMPT = """You are Ananya, Brand Partnerships Manager at an influencer marketing agency.
You sent a pitch email to a brand and they replied. Read the reply and extract structured data.
""" + REPLY_RULES + """
PRICING MODEL:
- "campaign_pool": the brand gives a TOTAL budget for the campaign (e.g. "2.5 lakh for the campaign")
- "per_collab": the brand gives a rate PER creator or PER collaboration (e.g. "20k per collab", "15k per reel")
- null: no budget mentioned
Put a total budget in total_budget. Put a per-creator rate in rate_per_collab, and the number of creators wanted in slots.

TIMELINE:
- timeline_days is the number of days until the content should be live, as an integer.
- Convert: "2 weeks" = 14, "10 days" = 10, "1 month" = 30, "next week" = 7.
- If they give a specific date instead, put null here and mention the date in "notes".

Format:
{
  "intent": "interested",
  "summary": "One sentence summary of the reply for the founder",
  "questions": ["Any questions the brand asked"],
  "pricing_model": "campaign_pool",
  "total_budget": 250000,
  "rate_per_collab": null,
  "slots": null,
  "budget_text": "exact budget wording from the email, or null",
  "platform": "Instagram, YouTube, etc. or null",
  "niche": "creator niche they want, or null",
  "target_audience": "age, gender, city tier or other audience they want, or null",
  "location_requirement": "cities or regions creators must be in, or null",
  "deliverables": "e.g. 1 Reel + 2 Stories per creator, or null",
  "timeline_days": 21,
  "requirements": "any other requirements, or null",
  "notes": "anything else important, or null"
}
"""

ADITYA_PARSE_PROMPT = """You are Aditya, Creator Manager Agent at an influencer marketing agency.
You sent a pitch email to a creator and they replied. Read the reply and extract structured data.
""" + REPLY_RULES + """
For min_budget, use the lowest amount the creator says they accept per collaboration.
For niche, use the reply, or our original pitch if it clearly states their niche.
For location, audience_type and languages, only use what the reply says.

Format:
{
  "intent": "interested",
  "summary": "One sentence summary of the reply for the founder",
  "questions": ["Any questions the creator asked"],
  "min_budget": 25000,
  "budget_text": "exact budget wording from the email, or null",
  "restrictions": "categories or brands they will not promote, or null",
  "availability": "when they are available, or null",
  "best_format": "Reels, Stories, YouTube videos, etc. or null",
  "niche": "e.g. fitness, beauty & skincare, tech reviews, or null",
  "location": "city or region, or null",
  "audience_type": "e.g. Gen Z, tier-2 cities, mostly female, or null",
  "languages": "e.g. Hindi, Marathi, English, or null",
  "notes": "anything else important, or null"
}
"""


def build_reply_message(request: ParseReplyRequest, contact_label: str) -> str:
    original = f"\nOUR ORIGINAL PITCH (for context):\n{request.original_pitch}\n" if request.original_pitch else ""
    return f"""{contact_label}: {request.contact_name or request.contact_id}
From: {request.from_email or 'unknown'}
Subject: {request.reply_subject or ''}
{original}
THEIR REPLY:
{strip_quoted_reply(request.reply_body)}

Extract the data and return ONLY the JSON object."""


def normalize_intent(parsed: dict) -> str:
    intent = str(parsed.get("intent", "other")).strip().lower()
    return intent if intent in VALID_INTENTS else "other"


def clean_text(value):
    """Return None for empty or 'null'-like strings"""
    if value is None:
        return None
    value = str(value).strip()
    return None if value == "" or value.lower() in ("null", "none", "n/a") else value


@app.post("/agent/steve/parse-reply")
async def steve_parse_reply(request: ParseReplyRequest):
    """
    STEVE: Read a brand's reply and extract intent + campaign brief.
    campaign_brief can be sent straight to /campaigns/upsert-from-brief.
    """
    try:
        parsed = parse_json_object(
            await call_claude(STEVE_PARSE_PROMPT, build_reply_message(request, "Brand"))
        )
        intent = normalize_intent(parsed)

        pricing_model = clean_text(parsed.get("pricing_model"))
        if pricing_model not in ("campaign_pool", "per_collab"):
            pricing_model = None

        campaign_brief = {
            "pricing_model": pricing_model,
            "total_budget": to_int_or_none(parsed.get("total_budget")),
            "rate_per_collab": to_int_or_none(parsed.get("rate_per_collab")),
            "slots": to_int_or_none(parsed.get("slots")),
            "budget_text": clean_text(parsed.get("budget_text")),
            "platform": clean_text(parsed.get("platform")),
            "niche": clean_text(parsed.get("niche")),
            "target_audience": clean_text(parsed.get("target_audience")),
            "location_requirement": clean_text(parsed.get("location_requirement")),
            "deliverables": clean_text(parsed.get("deliverables")),
            "timeline_days": to_days(parsed.get("timeline_days")),
            "requirements": clean_text(parsed.get("requirements")),
        }

        # A campaign is created only when the brand shares real brief details
        detail_keys = ["total_budget", "rate_per_collab", "platform", "niche", "deliverables", "timeline_days"]
        has_campaign_details = (
            intent not in ("not_interested", "auto_reply", "unsubscribe")
            and any(campaign_brief[k] is not None for k in detail_keys)
        )

        return {
            "status": "success",
            "contact_type": "brand",
            "brand_id": request.contact_id,
            "intent": intent,
            "is_real_response": intent != "auto_reply",
            "summary": parsed.get("summary"),
            "questions": parsed.get("questions") or [],
            "has_campaign_details": has_campaign_details,
            "campaign_brief": campaign_brief,
            "notes": clean_text(parsed.get("notes")),
            "extracted_data": parsed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing brand reply: {str(e)}")


@app.post("/agent/aditya/parse-reply")
async def aditya_parse_reply(request: ParseReplyRequest):
    """
    ADITYA: Read a creator's reply and extract intent, terms and profile details.
    Output fields map directly to the creators table.
    """
    try:
        parsed = parse_json_object(
            await call_claude(ADITYA_PARSE_PROMPT, build_reply_message(request, "Creator"))
        )
        intent = normalize_intent(parsed)

        return {
            "status": "success",
            "contact_type": "creator",
            "creator_id": request.contact_id,
            "intent": intent,
            "is_real_response": intent != "auto_reply",
            "summary": parsed.get("summary"),
            "questions": parsed.get("questions") or [],
            "min_budget": to_int_or_none(parsed.get("min_budget")),
            "budget_text": clean_text(parsed.get("budget_text")),
            "restrictions": clean_text(parsed.get("restrictions")),
            "availability": clean_text(parsed.get("availability")),
            "best_format": clean_text(parsed.get("best_format")),
            "niche": clean_text(parsed.get("niche")),
            "location": clean_text(parsed.get("location")),
            "audience_type": clean_text(parsed.get("audience_type")),
            "languages": clean_text(parsed.get("languages")),
            "notes": clean_text(parsed.get("notes")),
            "extracted_data": parsed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing creator reply: {str(e)}")


# ============ CAMPAIGNS ============
BRIEF_FIELDS = [
    "pricing_model", "total_budget", "rate_per_collab", "slots", "budget_text",
    "platform", "niche", "target_audience", "location_requirement",
    "deliverables", "requirements", "timeline_days"
]


class CampaignFromBriefRequest(BaseModel):
    brand_id: str
    gmail_thread_id: Optional[str] = None
    source: str = "email_reply"
    pricing_model: Optional[str] = None
    total_budget: Optional[int] = None
    rate_per_collab: Optional[int] = None
    slots: Optional[int] = None
    budget_text: Optional[str] = None
    platform: Optional[str] = None
    niche: Optional[str] = None
    target_audience: Optional[str] = None
    location_requirement: Optional[str] = None
    deliverables: Optional[str] = None
    requirements: Optional[str] = None
    timeline_days: Optional[int] = None


def load_settings() -> dict:
    """Read business rules from the settings table"""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT key, value FROM settings")).fetchall()
    return {key: float(value) for key, value in rows}


@app.post("/campaigns/upsert-from-brief")
async def upsert_campaign_from_brief(req: CampaignFromBriefRequest):
    """
    Create a campaign from a brand's brief, or update the open campaign
    in the same email thread. Snapshots the current business rules.
    """
    try:
        s = load_settings()
        brief = {f: getattr(req, f) for f in BRIEF_FIELDS}

        # Infer pricing model only when budget info exists
        if brief["pricing_model"] not in ("campaign_pool", "per_collab"):
            if req.rate_per_collab and not req.total_budget:
                brief["pricing_model"] = "per_collab"
            elif req.total_budget:
                brief["pricing_model"] = "campaign_pool"
            else:
                brief["pricing_model"] = None

        # Deal value decides exclusivity and revision rules
        deal_value = req.total_budget or (
            req.rate_per_collab * (req.slots or 1) if req.rate_per_collab else None
        )
        terms = {"exclusivity_days": None, "revisions_allowed": None}
        if deal_value:
            is_small = deal_value < s.get("small_deal_threshold_inr", 25000)
            terms["exclusivity_days"] = 0 if is_small else int(s.get("exclusivity_days", 30))
            terms["revisions_allowed"] = int(
                s.get("revisions_small", 1) if is_small else s.get("revisions_campaign", 2)
            )
        below_minimum = bool(deal_value and deal_value < s.get("min_deal_inr", 5000))

        with engine.begin() as conn:
            existing = None
            if req.gmail_thread_id:
                existing = conn.execute(text("""
                    SELECT campaign_id FROM campaigns
                    WHERE gmail_thread_id = :thread
                      AND status NOT IN ('completed', 'cancelled', 'lost')
                    ORDER BY id DESC LIMIT 1
                """), {"thread": req.gmail_thread_id}).first()

            update_fields = BRIEF_FIELDS + ["exclusivity_days", "revisions_allowed"]

            if existing:
                campaign_id = existing[0]
                # Only fill in what the new reply mentions, never erase existing details
                set_clause = ", ".join(f"{f} = COALESCE(:{f}, {f})" for f in update_fields)
                conn.execute(
                    text(f"UPDATE campaigns SET {set_clause} WHERE campaign_id = :campaign_id"),
                    {**brief, **terms, "campaign_id": campaign_id}
                )
                action = "updated"
            else:
                campaign_id = f"CMP_{req.brand_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                if brief["pricing_model"] is None:
                    brief["pricing_model"] = "campaign_pool"
                params = {
                    "campaign_id": campaign_id,
                    "brand_id": req.brand_id,
                    "source": req.source,
                    "gmail_thread_id": req.gmail_thread_id,
                    "margin_target": s.get("margin_target"),
                    "margin_floor": s.get("margin_floor"),
                    "advance_pct": s.get("advance_pct"),
                    **brief,
                    **terms
                }
                columns = list(params.keys())
                conn.execute(
                    text(f"INSERT INTO campaigns ({', '.join(columns)}) "
                         f"VALUES ({', '.join(':' + c for c in columns)})"),
                    params
                )
                action = "created"

            row = conn.execute(
                text("SELECT * FROM campaigns WHERE campaign_id = :campaign_id"),
                {"campaign_id": campaign_id}
            ).mappings().first()

        return {
            "status": "success",
            "action": action,
            "deal_value": deal_value,
            "below_minimum": below_minimum,
            "campaign": jsonable_encoder(dict(row))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving campaign: {str(e)}")


# ============ FRED: MATCHING AGENT ============
FRED_MODEL = "claude-opus-5-5"
FRED_VERSION = "fred-v1"


class FredMatchRequest(BaseModel):
    campaign_id: str


FRED_SYSTEM_PROMPT = """You are Fred, the Matching Agent at an influencer marketing agency in India.
You score how well each candidate creator fits a brand's campaign.

Score each creator 0-100 as the sum of four parts, each 0-25:
- niche_fit: does their content naturally fit this product and category?
- audience_fit: does their audience (location, city tier, language, age, gender) match the target?
- engagement_quality: judge engagement RELATIVE to follower tier. Smaller accounts naturally have
  higher engagement rates, so 4% at 5M followers can be stronger than 9% at 20K.
- value_for_money: expected impact for the price. Use their rate if known, otherwise your estimate.

Also return:
- restriction_conflict: true if the creator's stated restrictions rule out this brand or product
- availability_conflict: true if their stated availability clearly misses the campaign timeline
- red_flags: suspicious numbers (very few followers, engagement implausible for the size), or null
- estimated_rate: ONLY when their rate is unknown, a fair INR fee for this work in the Indian market
  given their tier, niche and platform. Otherwise null.
- anon_summary: one positive sentence on why they fit, written for the brand. Describe strengths only.
  Never mention price, budget, fees, concerns or weaknesses, and never include a name, handle,
  or anything that identifies the creator.

Return ONLY a JSON array with one object per creator, no other text:
[
  {
    "creator_id": "CREATOR_001",
    "match_score": 82,
    "score_breakdown": {"niche_fit": 22, "audience_fit": 20, "engagement_quality": 20, "value_for_money": 20},
    "reasoning": "Why this creator fits or doesn't, 1-2 sentences",
    "concerns": null,
    "red_flags": null,
    "restriction_conflict": false,
    "availability_conflict": false,
    "estimated_rate": null,
    "anon_summary": "Pune-based fitness educator with a highly engaged local audience"
  }
]
"""


def _words(value) -> set:
    return {w for w in re.findall(r"[a-z]+", str(value or "").lower()) if len(w) > 2}


def _platform_ok(creator_platform, campaign_platform) -> bool:
    if not campaign_platform or not creator_platform:
        return True
    return str(creator_platform).strip().lower() in str(campaign_platform).lower()


def _location_ok(creator_location, requirement) -> bool:
    if not requirement or not creator_location:
        return True   # unknown location: let Fred judge it
    return bool(_words(creator_location) & _words(requirement))


def _format_followers(n) -> str:
    if not n:
        return "unknown"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{round(n / 1_000)}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _anonymize(text_value, creator) -> str:
    """Strip a creator's name and handle from brand-facing text"""
    result = str(text_value or "")
    for token in (creator.get("creator_name"), creator.get("handle"),
                  str(creator.get("handle") or "").lstrip("@")):
        if token and len(str(token)) > 1:
            result = re.sub(re.escape(str(token)), "this creator", result, flags=re.IGNORECASE)
    return result


def parse_json_array(response_text: str) -> list:
    response_text = clean_json_text(response_text)
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        start, end = response_text.find("["), response_text.rfind("]")
        data = json.loads(response_text[start:end + 1])
    return data if isinstance(data, list) else []


@app.post("/agent/fred/match")
async def fred_match(request: FredMatchRequest):
    """
    FRED: Match creators to a campaign.
    1. Budget math   2. Hard filters (Python)   3. Claude scores up to N survivors
    4. Shortlist (per collab: top N; campaign pool: best mix within budget)
    Saves the shortlist to matches and returns internal + anonymized views.
    """
    try:
        s = load_settings()
        shortlist_size = int(s.get("shortlist_size", 5))
        max_candidates = int(s.get("fred_max_candidates", 15))
        include_unconfirmed = s.get("fred_include_unconfirmed", 1) >= 1

        # ---------- Load campaign, creators, exclusivity blocks ----------
        with engine.connect() as conn:
            campaign = conn.execute(text("""
                SELECT c.*, b.brand_name, b.industry, b.basic_info AS brand_info
                FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.campaign_id = :cid
            """), {"cid": request.campaign_id}).mappings().first()
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            campaign = dict(campaign)

            creators = [dict(r) for r in conn.execute(text("""
                SELECT creator_id, creator_name, platform, handle, basic_info, follower_count,
                       engagement_rate, comments, min_budget, restrictions, availability,
                       best_format, niche, location, audience_type, languages
                FROM creators
                WHERE COALESCE(unsubscribed, FALSE) = FALSE
                  AND COALESCE(response_intent, '') NOT IN ('not_interested', 'unsubscribe')
                  AND creator_id NOT IN (
                      SELECT creator_id FROM matches
                      WHERE campaign_id = :cid AND approval_status <> 'pending'
                  )
            """), {"cid": request.campaign_id}).mappings().all()]

            # Creators who accepted a competing brand's campaign inside its exclusivity window
            blocked = {row[0] for row in conn.execute(text("""
                SELECT DISTINCT o.creator_id
                FROM offers o
                JOIN campaigns c ON c.campaign_id = o.campaign_id
                JOIN brands b ON b.brand_id = c.brand_id
                WHERE o.status = 'accepted'
                  AND c.brand_id <> :brand_id
                  AND COALESCE(c.exclusivity_days, 0) > 0
                  AND c.updated_at > NOW() - make_interval(days => c.exclusivity_days)
                  AND :industry <> ''
                  AND LOWER(COALESCE(b.industry, '')) = LOWER(:industry)
            """), {"brand_id": campaign["brand_id"],
                   "industry": campaign.get("industry") or ""}).fetchall()}

        # ---------- Step 1: budget math ----------
        margin_target = float(campaign["margin_target"] or s.get("margin_target", 0.30))
        margin_floor = float(campaign["margin_floor"] or s.get("margin_floor", 0.20))
        pricing_model = campaign["pricing_model"] or "campaign_pool"

        if pricing_model == "per_collab":
            rate = campaign["rate_per_collab"]
            if not rate:
                raise HTTPException(status_code=400, detail="Campaign has no rate_per_collab yet")
            slots = campaign["slots"] or 1
            brand_price = rate * slots
            cap_target = int(rate * (1 - margin_target))    # max payout per creator at target margin
            cap_stretch = int(rate * (1 - margin_floor))    # max payout per creator at margin floor
        else:
            total = campaign["total_budget"]
            if not total:
                raise HTTPException(status_code=400, detail="Campaign has no total_budget yet")
            rate, slots = None, None
            brand_price = total
            cap_target = int(total * (1 - margin_target))   # creator pool at target margin
            cap_stretch = int(total * (1 - margin_floor))   # creator pool at margin floor

        # ---------- Step 2: hard filters ----------
        excluded = {"exclusivity": 0, "platform": 0, "location": 0, "over_budget": 0, "rate_unknown": 0}
        survivors = []
        for c in creators:
            confirmed = c["min_budget"] is not None
            if c["creator_id"] in blocked:
                excluded["exclusivity"] += 1
            elif not _platform_ok(c["platform"], campaign["platform"]):
                excluded["platform"] += 1
            elif not _location_ok(c["location"], campaign["location_requirement"]):
                excluded["location"] += 1
            elif confirmed and c["min_budget"] > cap_stretch:
                excluded["over_budget"] += 1
            elif not confirmed and not include_unconfirmed:
                excluded["rate_unknown"] += 1
            else:
                survivors.append(c)

        # Cheap pre-ranking so Claude only sees the most promising candidates
        target_words = _words(campaign["niche"]) | _words(campaign["target_audience"]) | _words(campaign.get("industry"))

        def prescore(c):
            overlap = len(target_words & (_words(c["niche"]) | _words(c["basic_info"]) | _words(c["comments"])))
            return overlap * 3 + (2 if c["min_budget"] is not None else 0) + min(float(c["engagement_rate"] or 0), 10) / 2

        candidates = sorted(survivors, key=prescore, reverse=True)[:max_candidates]

        budget_math = {
            "pricing_model": pricing_model,
            "brand_price": brand_price,
            "rate_per_collab": rate,
            "slots": slots,
            "margin_target": margin_target,
            "margin_floor": margin_floor,
            "creator_budget_at_target": cap_target,
            "creator_budget_at_floor": cap_stretch,
        }

        if not candidates:
            return {
                "status": "no_candidates",
                "campaign_id": request.campaign_id,
                "needs_recruiting": True,
                "message": "No creators passed the filters. Recruit in this niche or relax the brief.",
                "filters_excluded": excluded,
                "budget_math": budget_math,
                "internal_view": [],
                "anonymized_view": []
            }

        # ---------- Step 3: Claude scores the candidates ----------
        def val(v):
            return v if v not in (None, "") else "unknown"

        pricing_line = (f"Per collab: INR {rate} per creator, {slots} creator(s) wanted"
                        if pricing_model == "per_collab" else f"Total campaign budget: INR {brand_price}")
        campaign_text = f"""CAMPAIGN
Brand: {campaign['brand_name']} (industry: {val(campaign.get('industry'))})
About the brand: {val(campaign.get('brand_info'))}
{pricing_line}
Platform: {val(campaign['platform'])}
Niche wanted: {val(campaign['niche'])}
Target audience: {val(campaign['target_audience'])}
Location requirement: {val(campaign['location_requirement'])}
Deliverables: {val(campaign['deliverables'])}
Timeline: {val(campaign['timeline_days'])} days
Other requirements: {val(campaign['requirements'])}"""

        candidates_text = "\n\n".join(
            f"""Creator {c['creator_id']}:
- Platform: {val(c['platform'])}
- Followers: {val(c['follower_count'])}
- Engagement rate: {val(c['engagement_rate'])}%
- Niche: {val(c['niche'])}
- About: {val(c['basic_info'])}
- Notes: {val(c['comments'])}
- Location: {val(c['location'])}
- Audience: {val(c['audience_type'])}
- Languages: {val(c['languages'])}
- Rate: {('INR ' + str(c['min_budget']) + ' minimum') if c['min_budget'] is not None else 'unknown'}
- Restrictions: {val(c['restrictions'])}
- Availability: {val(c['availability'])}
- Best format: {val(c['best_format'])}"""
            for c in candidates
        )

        user_message = f"""{campaign_text}

CANDIDATES ({len(candidates)}):

{candidates_text}

Score ALL {len(candidates)} candidates. Return ONLY the JSON array."""

        scores = parse_json_array(await call_claude(FRED_SYSTEM_PROMPT, user_message, model=FRED_MODEL))

        # ---------- Merge scores, drop conflicts, price each creator ----------
        by_id = {c["creator_id"]: c for c in candidates}
        conflicts, scored, estimated_over_budget = [], [], []
        for sc in scores:
            c = by_id.get(sc.get("creator_id"))
            if not c:
                continue
            if sc.get("restriction_conflict") or sc.get("availability_conflict"):
                conflicts.append({"creator_id": c["creator_id"], "reason": sc.get("reasoning")})
                continue

            confirmed = c["min_budget"] is not None
            estimated = to_int_or_none(sc.get("estimated_rate"))
            cost = c["min_budget"] if confirmed else estimated
            concerns = sc.get("concerns")

            # An estimate above the stretch budget rules the creator out, like a confirmed high price
            if not confirmed and estimated is not None and estimated > cap_stretch:
                estimated_over_budget.append({"creator_id": c["creator_id"], "estimated_rate": estimated})
                continue

            if pricing_model == "per_collab":
                effective = cost if cost is not None else cap_target
                if effective <= cap_target:
                    is_stretch, payout = False, effective
                else:
                    is_stretch, payout = True, effective
            else:
                is_stretch = False
                payout = cost if cost is not None else cap_target // max(shortlist_size, 1)

            scored.append({
                **c,
                "match_score": to_int_or_none(sc.get("match_score")) or 0,
                "score_breakdown": sc.get("score_breakdown") or {},
                "reasoning": sc.get("reasoning"),
                "concerns": concerns,
                "red_flags": sc.get("red_flags"),
                "anon_summary": sc.get("anon_summary"),
                "rate_confirmed": confirmed,
                "estimated_rate": None if confirmed else estimated,
                "is_stretch": is_stretch,
                "suggested_payout": int(payout),
            })

        # ---------- Step 4: build the shortlist ----------
        if pricing_model == "per_collab":
            ranked = (sorted([x for x in scored if not x["is_stretch"]], key=lambda x: -x["match_score"])
                      + sorted([x for x in scored if x["is_stretch"]], key=lambda x: -x["match_score"]))
            shortlist = ranked[:max(shortlist_size, slots)]
            recommended = shortlist[:slots]
            creator_cost = sum(x["suggested_payout"] for x in recommended)
            projected_price = rate * len(recommended)
        else:
            # Best-fit mix within the pool at target margin, then stretch picks up to the floor
            shortlist, spent = [], 0
            remaining = sorted(scored, key=lambda x: -x["match_score"])
            for x in list(remaining):
                if len(shortlist) >= shortlist_size:
                    break
                if spent + x["suggested_payout"] <= cap_target:
                    shortlist.append(x)
                    spent += x["suggested_payout"]
                    remaining.remove(x)
            for x in list(remaining):
                if len(shortlist) >= shortlist_size:
                    break
                if spent + x["suggested_payout"] <= cap_stretch:
                    x["is_stretch"] = True
                    shortlist.append(x)
                    spent += x["suggested_payout"]
                    remaining.remove(x)
            creator_cost = spent
            projected_price = brand_price

        projected_margin = projected_price - creator_cost
        budget_math.update({
            "projected_brand_price": projected_price,
            "projected_creator_cost": creator_cost,
            "projected_margin": projected_margin,
            "projected_margin_pct": round(projected_margin / projected_price, 3) if projected_price else None,
            "below_margin_floor": bool(projected_price and projected_margin / projected_price < margin_floor),
        })

        # ---------- Save shortlist to matches ----------
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM matches WHERE campaign_id = :cid AND approval_status = 'pending'"),
                         {"cid": request.campaign_id})
            for rank, x in enumerate(shortlist, start=1):
                x["rank"] = rank
                row = conn.execute(text("""
                    INSERT INTO matches (campaign_id, creator_id, rank, match_score, score_breakdown,
                        reasoning, concerns, red_flags, is_stretch, suggested_payout,
                        creator_min_budget_at_match, fred_model, fred_version, anon_summary)
                    VALUES (:campaign_id, :creator_id, :rank, :match_score, CAST(:score_breakdown AS JSONB),
                        :reasoning, :concerns, :red_flags, :is_stretch, :suggested_payout,
                        :min_budget, :fred_model, :fred_version, :anon_summary)
                    ON CONFLICT (campaign_id, creator_id) DO NOTHING
                    RETURNING id
                """), {
                    "campaign_id": request.campaign_id,
                    "creator_id": x["creator_id"],
                    "rank": rank,
                    "match_score": x["match_score"],
                    "score_breakdown": json.dumps(x["score_breakdown"]),
                    "reasoning": x["reasoning"],
                    "concerns": x["concerns"],
                    "red_flags": x["red_flags"],
                    "is_stretch": x["is_stretch"],
                    "suggested_payout": x["suggested_payout"],
                    "min_budget": x["min_budget"],
                    "fred_model": FRED_MODEL,
                    "fred_version": FRED_VERSION,
                    "anon_summary": _anonymize(x["anon_summary"], x),
                }).first()
                x["match_id"] = row[0] if row else None

            if shortlist:
                conn.execute(text("""
                    UPDATE campaigns SET status = 'awaiting_approval'
                    WHERE campaign_id = :cid AND status IN ('brief_received', 'matching')
                """), {"cid": request.campaign_id})

        # ---------- Two views of the same shortlist ----------
        internal_view = [{
            "rank": x["rank"],
            "match_id": x["match_id"],
            "creator_id": x["creator_id"],
            "creator_name": x["creator_name"],
            "handle": x["handle"],
            "platform": x["platform"],
            "followers": x["follower_count"],
            "engagement_rate": x["engagement_rate"],
            "niche": x["niche"],
            "location": x["location"],
            "match_score": x["match_score"],
            "score_breakdown": x["score_breakdown"],
            "reasoning": x["reasoning"],
            "concerns": x["concerns"],
            "red_flags": x["red_flags"],
            "rate_confirmed": x["rate_confirmed"],
            "creator_min_budget": x["min_budget"],
            "estimated_rate": x["estimated_rate"],
            "suggested_payout": x["suggested_payout"],
            "is_stretch": x["is_stretch"],
        } for x in shortlist]

        anonymized_view = [{
            "label": f"Creator {chr(65 + i)}",
            "platform": x["platform"],
            "followers": _format_followers(x["follower_count"]),
            "engagement_rate": x["engagement_rate"],
            "niche": x["niche"],
            "location": x["location"],
            "audience": x["audience_type"],
            "why_this_creator": _anonymize(x["anon_summary"], x),
        } for i, x in enumerate(shortlist)]

        slots_unfilled = max((slots or 0) - len(shortlist), 0)

        return {
            "status": "success",
            "campaign_id": request.campaign_id,
            "brand_name": campaign["brand_name"],
            "needs_recruiting": slots_unfilled > 0 or len(shortlist) < min(3, max(shortlist_size, slots or 1)),
            "slots_requested": slots,
            "slots_unfilled": slots_unfilled,
            "candidates_considered": len(creators),
            "candidates_scored": len(candidates),
            "filters_excluded": excluded,
            "excluded_by_fred": conflicts,
            "excluded_estimated_over_budget": estimated_over_budget,
            "budget_math": budget_math,
            "internal_view": internal_view,
            "anonymized_view": anonymized_view,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fred matching error: {str(e)}")


# ============ FOLLOW-UPS (STEVE + ADITYA) ============
FOLLOWUP_RULES = """
Rules:
- 2-4 short sentences, under 70 words. No subject line (it is sent as a reply in the same thread).
- Follow-up 1: a friendly nudge with ONE new, specific reason to reply that fits them.
  Make replying effortless, e.g. "Just reply with ..." and the one or two details you need.
- Follow-up 2 (final): brief and polite. Say this is your last note and leave the door open.
- Never guilt-trip. Never write "just following up", "circling back" or "bumping this".
- Do not repeat the original pitch.
- Never invent facts, client names, past results or numbers.

Return ONLY a JSON array, no other text:
[{"contact_id": "ID_001", "followup_email": "Hi ...,\\n\\n...\\n\\n<sign-off>"}]
"""

STEVE_FOLLOWUP_PROMPT = """You are Ananya, Brand Partnerships Manager at an influencer marketing agency in India.
These brands have not replied to your partnership pitch. Write a short follow-up email for each.
Sign off exactly as:
Ananya
Brand Partnerships
""" + FOLLOWUP_RULES

ADITYA_FOLLOWUP_PROMPT = """You are Aditya, Creator Manager at an influencer marketing agency in India.
These creators have not replied to your collaboration pitch. Write a short follow-up email for each.
The easiest reply to ask for is their rate per collaboration and the kind of brands they like.
If it fits naturally, remind them that working with us is free: we never charge creators anything.
Sign off exactly as:
Aditya
Creator Manager
""" + FOLLOWUP_RULES

FOLLOWUP_SOURCES = {
    "brand": {
        "table": "brands", "id_col": "brand_id", "name_col": "brand_name",
        "extra": "industry, basic_info, pitch_email AS pitch_text",
        "prompt": STEVE_FOLLOWUP_PROMPT, "label": "brands",
    },
    "creator": {
        "table": "creators", "id_col": "creator_id", "name_col": "creator_name",
        "extra": "platform, niche, basic_info, CONCAT_WS(E'\\n\\n', pitch_subject, pitch_body) AS pitch_text",
        "prompt": ADITYA_FOLLOWUP_PROMPT, "label": "creators",
    },
}

OPEN_CONTACT_FILTER = """
    COALESCE(response_received, FALSE) = FALSE
    AND COALESCE(unsubscribed, FALSE) = FALSE
    AND COALESCE(status, '') NOT IN ('unsubscribed', 'no_response', 'responded')
"""


def _format_followup_item(i: int, item) -> str:
    details = "\n".join(f"- {k}: {v}" for k, v in item.details.items() if v not in (None, ""))
    excerpt = (item.pitch_text or "")[:400]
    return f"""Contact #{i + 1}:
- ID: {item.contact_id}
- Name: {item.contact_name}
{details}
- Follow-up number: {item.followup_number} of {item.max_followups}
- Our original pitch (excerpt): {excerpt}"""


@app.post("/followups/prepare")
async def prepare_followups():
    """
    Daily follow-up run:
    1. Close out contacts that got every follow-up and still never replied (status = no_response)
    2. Find contacts due for follow-up 1 or 2
    3. Steve / Aditya write the follow-ups (parallel batches)
    Returns a list ready for N8N to send as replies in the original Gmail threads.
    """
    try:
        s = load_settings()
        f1 = int(s.get("followup_1_days", 3))
        f2 = int(s.get("followup_2_days", 7))
        max_f = int(s.get("max_followups", 2))
        close_days = int(s.get("followup_close_days", 4))
        limit = int(s.get("followup_daily_limit", 100))

        closed_out, due = {}, {}
        with engine.begin() as conn:
            for ctype, src in FOLLOWUP_SOURCES.items():
                # 1. Close out contacts who ignored every follow-up
                closed_out[ctype] = conn.execute(text(f"""
                    UPDATE {src['table']} SET status = 'no_response'
                    WHERE {OPEN_CONTACT_FILTER}
                      AND COALESCE(followup_count, 0) >= :max_f
                      AND last_followup_at <= NOW() - make_interval(days => :close_days)
                """), {"max_f": max_f, "close_days": close_days}).rowcount

                # 2. Contacts due for their next follow-up
                rows = conn.execute(text(f"""
                    SELECT {src['id_col']} AS contact_id, {src['name_col']} AS contact_name,
                           email, gmail_thread_id, gmail_message_id,
                           COALESCE(followup_count, 0) AS followup_count, {src['extra']}
                    FROM {src['table']}
                    WHERE {OPEN_CONTACT_FILTER}
                      AND gmail_message_id IS NOT NULL
                      AND COALESCE(email, '') <> ''
                      AND COALESCE(followup_count, 0) < :max_f
                      AND (
                        (COALESCE(followup_count, 0) = 0 AND email_sent_date <= NOW() - make_interval(days => :f1))
                        OR (COALESCE(followup_count, 0) >= 1 AND email_sent_date <= NOW() - make_interval(days => :f2))
                      )
                    ORDER BY email_sent_date ASC
                    LIMIT :lim
                """), {"max_f": max_f, "f1": f1, "f2": f2, "lim": limit}).mappings().all()
                due[ctype] = [dict(r) for r in rows]

        # 3. Write follow-ups: Steve for brands, Aditya for creators, both in parallel
        async def write(ctype: str) -> dict:
            rows = due[ctype]
            if not rows:
                return {}
            items = []
            for r in rows:
                details = ({"Industry": r.get("industry"), "About": r.get("basic_info")} if ctype == "brand"
                           else {"Platform": r.get("platform"), "Niche": r.get("niche"), "About": r.get("basic_info")})
                items.append(SimpleNamespace(
                    contact_id=r["contact_id"], contact_name=r["contact_name"],
                    details=details, pitch_text=r.get("pitch_text"),
                    followup_number=r["followup_count"] + 1, max_followups=max_f,
                ))
            src = FOLLOWUP_SOURCES[ctype]
            return await generate_pitches_parallel(
                items, "contact_id", _format_followup_item, src["prompt"], src["label"],
                text_key="followup_email", task="follow-up emails"
            )

        written_brand, written_creator = await asyncio.gather(write("brand"), write("creator"))
        written = {"brand": written_brand, "creator": written_creator}

        followups, failed = [], []
        for ctype, rows in due.items():
            for r in rows:
                body = written[ctype].get(r["contact_id"])
                if not body:
                    failed.append({"contact_type": ctype, "contact_id": r["contact_id"]})
                    continue
                followups.append({
                    "contact_type": ctype,
                    "contact_id": r["contact_id"],
                    "contact_name": r["contact_name"],
                    "email": r["email"],
                    "gmail_thread_id": r["gmail_thread_id"],
                    "gmail_message_id": r["gmail_message_id"],   # original pitch: reply to this
                    "followup_number": r["followup_count"] + 1,
                    "body": body,
                })

        return {
            "status": "success",
            "closed_out": closed_out,
            "count": len(followups),
            "failed": failed,
            "followups": followups,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing follow-ups: {str(e)}")


class FollowupMarkRequest(BaseModel):
    contact_type: str
    contact_id: str
    followup_number: int
    gmail_message_id: Optional[str] = None   # ID of the follow-up we just sent
    gmail_thread_id: Optional[str] = None
    body: Optional[str] = None


@app.post("/followups/mark")
async def mark_followup(req: FollowupMarkRequest):
    """Record a sent follow-up on the contact and log it in conversations"""
    try:
        src = FOLLOWUP_SOURCES.get(req.contact_type)
        if not src:
            raise HTTPException(status_code=400, detail="contact_type must be 'brand' or 'creator'")

        with engine.begin() as conn:
            updated = conn.execute(text(f"""
                UPDATE {src['table']}
                SET followup_count = :n, last_followup_at = NOW()
                WHERE {src['id_col']} = :cid
            """), {"n": req.followup_number, "cid": req.contact_id}).rowcount

            conn.execute(text("""
                INSERT INTO conversations (contact_type, contact_id, gmail_thread_id, gmail_message_id,
                                           direction, subject, body, intent)
                VALUES (:ctype, :cid, :thread, :msg, 'outbound', :subject, :body, 'followup')
                ON CONFLICT (gmail_message_id) DO NOTHING
            """), {
                "ctype": req.contact_type, "cid": req.contact_id,
                "thread": req.gmail_thread_id, "msg": req.gmail_message_id,
                "subject": f"Follow-up {req.followup_number}", "body": req.body,
            })

        return {"status": "success", "updated": updated}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marking follow-up: {str(e)}")


# ============ DAILY DIGEST ============
@app.post("/digest/daily")
async def daily_digest():
    """Last 24h activity, 7-day reply rates and anything waiting on Deven, as a Slack message"""
    try:
        s = load_settings()
        approval_days = int(s.get("approval_stale_days", 1))
        brief_days = int(s.get("brief_stale_days", 3))

        with engine.connect() as conn:
            def count(sql: str) -> int:
                return conn.execute(text(sql)).scalar() or 0

            pitched_b = count("SELECT COUNT(*) FROM brands WHERE email_sent_date >= NOW() - INTERVAL '1 day'")
            pitched_c = count("SELECT COUNT(*) FROM creators WHERE email_sent_date >= NOW() - INTERVAL '1 day'")
            replies_b = count("""SELECT COUNT(*) FROM conversations WHERE direction = 'inbound' AND contact_type = 'brand'
                                 AND COALESCE(intent, '') <> 'auto_reply' AND created_at >= NOW() - INTERVAL '1 day'""")
            replies_c = count("""SELECT COUNT(*) FROM conversations WHERE direction = 'inbound' AND contact_type = 'creator'
                                 AND COALESCE(intent, '') <> 'auto_reply' AND created_at >= NOW() - INTERVAL '1 day'""")
            followups = count("""SELECT (SELECT COUNT(*) FROM brands WHERE last_followup_at >= NOW() - INTERVAL '1 day')
                                      + (SELECT COUNT(*) FROM creators WHERE last_followup_at >= NOW() - INTERVAL '1 day')""")

            # 7-day reply rate: contacts pitched 1-8 days ago, so each had at least a day to reply
            def rate(table: str):
                replied, total = conn.execute(text(f"""
                    SELECT COUNT(*) FILTER (WHERE response_received), COUNT(*) FROM {table}
                    WHERE email_sent_date BETWEEN NOW() - INTERVAL '8 days' AND NOW() - INTERVAL '1 day'
                """)).first()
                return replied or 0, total or 0

            rate_b, rate_c = rate("brands"), rate("creators")

            waiting_approval = conn.execute(text("""
                SELECT c.campaign_id, b.brand_name,
                       EXTRACT(DAY FROM NOW() - c.updated_at)::INT AS days
                FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.status = 'awaiting_approval'
                  AND c.updated_at <= NOW() - make_interval(days => :d)
                ORDER BY c.updated_at
            """), {"d": approval_days}).mappings().all()

            stalled_briefs = conn.execute(text("""
                SELECT c.campaign_id, b.brand_name,
                       EXTRACT(DAY FROM NOW() - c.created_at)::INT AS days
                FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.status = 'brief_received'
                  AND c.created_at <= NOW() - make_interval(days => :d)
                ORDER BY c.created_at
            """), {"d": brief_days}).mappings().all()

        def pct(replied, total):
            return f"{(replied / total * 100):.1f}% ({replied}/{total})" if total else "no data yet"

        lines = [
            "📊 *Daily digest*",
            f"*Last 24h:* {pitched_b} brands and {pitched_c} creators pitched · "
            f"{replies_b + replies_c} replies ({replies_b} brands, {replies_c} creators) · {followups} follow-ups sent",
            f"*7-day reply rate:* brands {pct(*rate_b)} · creators {pct(*rate_c)}",
        ]
        if waiting_approval:
            lines.append(f"\n⏳ *Waiting on you ({len(waiting_approval)}):* shortlists pending approval")
            lines += [f"• {r['brand_name']} · `{r['campaign_id']}` · {r['days']} day(s)" for r in waiting_approval[:10]]
        if stalled_briefs:
            lines.append(f"\n🟡 *Stalled briefs ({len(stalled_briefs)}):* no budget yet, or below minimum and undecided")
            lines += [f"• {r['brand_name']} · `{r['campaign_id']}` · {r['days']} day(s)" for r in stalled_briefs[:10]]
        if not waiting_approval and not stalled_briefs:
            lines.append("\n✅ Nothing waiting on you.")

        return {
            "status": "success",
            "slack_text": "\n".join(lines),
            "stats": {
                "pitched_brands": pitched_b, "pitched_creators": pitched_c,
                "replies_brands": replies_b, "replies_creators": replies_c,
                "followups_sent": followups,
                "reply_rate_brands": rate_b, "reply_rate_creators": rate_c,
                "waiting_approval": len(waiting_approval), "stalled_briefs": len(stalled_briefs),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building digest: {str(e)}")


# ============ OFFERS (STAGE 5, PHASE A) ============
IST = timezone(timedelta(hours=5, minutes=30))


def format_inr(amount) -> str:
    """Indian number format: 450000 -> ₹4,50,000"""
    s = str(int(round(float(amount))))
    if len(s) <= 3:
        return "₹" + s
    last3, rest, parts = s[-3:], s[:-3], []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return "₹" + ",".join(parts) + "," + last3


def _strip_brand(text_value: str, brand_name: str) -> str:
    """Safety net: remove the brand's name from creator-facing text"""
    if not brand_name or len(brand_name.strip()) < 2:
        return text_value
    return re.sub(re.escape(brand_name.strip()), "the brand", text_value, flags=re.IGNORECASE)


def creator_payment_terms(amount, s: dict) -> str:
    """How and when a creator gets paid. The single source for every creator-facing email."""
    first = int(round(float(s.get("creator_first_payment_pct", 0.3)) * 100))
    hours_min = int(s.get("creator_final_payment_hours_min", 48))
    hours_max = int(s.get("creator_final_payment_hours_max", 58))
    return (f"{first}% once the brand approves your content, and the remaining {100 - first}% "
            f"within {hours_min}-{hours_max} hours after your post goes live")


def build_offer_terms(amount, deliverables, platform, deadline, revisions, exclusivity_days,
                      industry, expires_at, s: dict) -> str:
    """The binding part of every offer: always generated from data, never by the model"""
    kill_fee = int(float(s.get("kill_fee_pct", 0.5)) * 100)

    lines = [
        "Here are the details:",
        f"• Deliverables: {deliverables or ('1 post on ' + platform if platform else 'to be confirmed')}",
    ]
    if platform:
        lines.append(f"• Platform: {platform}")
    lines.append(f"• Content deadline: {deadline.strftime('%d %b %Y') if deadline else 'to be confirmed with the brief'}")
    lines.append(f"• Your payout: {format_inr(amount)}")
    lines.append(f"• Payment: {creator_payment_terms(amount, s)}")
    if revisions:
        lines.append(f"• Revisions: up to {revisions} round{'s' if revisions > 1 else ''} of changes")
    if exclusivity_days:
        category = (industry or "competing").strip()
        lines.append(f"• Exclusivity: no posts for competing {category} brands for {exclusivity_days} days after publishing")
    lines.append(f"• Cancellation: if the brand cancels for its own reasons after you've created content that meets the brief, you receive {kill_fee}% of the payout in total. If the content doesn't meet the brief after the agreed revisions, no payment is due")
    lines.append(f"• This offer is open until {expires_at.astimezone(IST).strftime('%d %b, %I:%M %p')} IST")
    lines += [
        "",
        'To confirm, just reply "Accept". If you\'d like to discuss the payout or anything else, reply with what works for you.',
        "We'll share the brand's name and the full brief as soon as you accept.",
        "",
        "Aditya",
        "Creator Manager",
    ]
    return "\n".join(lines)


ADITYA_OFFER_PROMPT_TEMPLATE = """You are Aditya, Creator Manager at an influencer marketing agency in India.
You are sending paid collaboration offers to creators for ONE campaign.

CAMPAIGN
- Brand name (CONFIDENTIAL): {brand_name}
  Never mention this name, or any product, store or detail that would reveal it.
- Industry: {industry}
- About the brand: {brand_info}
- Campaign niche: {niche}
- Target audience: {target_audience}
- Location: {location}
- Requirements: {requirements}
- Platform(s): {platform}
- Deliverables from the brief: {deliverables}

For each creator, write ONLY the opening of the offer email:
- Greet them by first name
- 2-3 sentences: why they were picked (their content, niche or audience) and a vivid but
  anonymous description of the brand and campaign, e.g. "a Pune-based sportswear brand opening a new store"
- Under 80 words. Warm and professional.
- Describe the brand factually. Never call it leading, top, prominent, famous, premium,
  or say anything about its size or status that the details above don't state.
- Do NOT mention fees, dates, payment or terms. Those are added separately.
- Do NOT sign off.
- On the LAST line, write the deliverable for THIS creator on THEIR platform, as:
  DELIVERABLE: <for example: 1 Instagram Reel>
  Base it on the brief. If the brief is unclear for their platform, use 1 post on their platform.

Return ONLY a JSON array, no other text:
[{{"offer_id": "OFR_...", "offer_intro": "Hi Ali,\\n\\n..."}}]
"""


PLATFORM_NAMES = {"instagram": "Instagram", "youtube": "YouTube", "facebook": "Facebook",
                  "snapchat": "Snapchat", "linkedin": "LinkedIn", "twitter": "Twitter", "x": "X"}


def _creator_platform(creator_platform, campaign_platform):
    """The platform this creator will post on, nicely capitalized"""
    if creator_platform and (not campaign_platform or creator_platform.strip().lower() in str(campaign_platform).lower()):
        return PLATFORM_NAMES.get(creator_platform.strip().lower(), creator_platform.strip().title())
    return campaign_platform


def _split_deliverable(intro: str):
    """Pull the DELIVERABLE line out of the model's opening"""
    match = re.search(r"^\s*DELIVERABLE:\s*(.+?)\s*$", intro, flags=re.MULTILINE | re.IGNORECASE)
    deliverable = match.group(1).strip() if match else None
    cleaned = re.sub(r"^\s*DELIVERABLE:.*$", "", intro, flags=re.MULTILINE | re.IGNORECASE).strip()
    return cleaned, deliverable


def _first_name(name) -> str:
    parts = str(name or "").strip().split()
    return parts[0].capitalize() if parts else "there"


def _format_offer_item(i: int, item) -> str:
    return f"""Creator #{i + 1}:
- ID: {item.offer_id}
- Name: {item.creator_name}
- Platform: {item.platform or 'unknown'}
- Niche: {item.niche or 'unknown'}
- About: {item.basic_info or 'unknown'}"""


class OfferPrepareRequest(BaseModel):
    campaign_id: str


@app.post("/offers/prepare")
async def prepare_offers(req: OfferPrepareRequest):
    """
    Create offers for approved creators who don't have one yet, up to the open slots.
    Aditya writes a personal opening; fees and terms come from data.
    Returns emails ready for N8N to send.
    """
    try:
        s = load_settings()

        with engine.begin() as conn:
            campaign = conn.execute(text("""
                SELECT c.*, b.brand_name, b.industry, b.basic_info AS brand_info
                FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.campaign_id = :cid
            """), {"cid": req.campaign_id}).mappings().first()
            if not campaign:
                raise HTTPException(status_code=404, detail="Campaign not found")
            campaign = dict(campaign)

            # Unsent drafts from an earlier failed run are rebuilt from scratch
            conn.execute(text("DELETE FROM offers WHERE campaign_id = :cid AND status = 'draft'"),
                         {"cid": req.campaign_id})

            approved_count = conn.execute(text("""
                SELECT COUNT(*) FROM matches WHERE campaign_id = :cid AND approval_status = 'approved'
            """), {"cid": req.campaign_id}).scalar() or 0

            # How many creators this campaign needs
            if campaign["pricing_model"] == "per_collab":
                target = campaign["slots"] or 1
            else:
                target = approved_count

            in_play = conn.execute(text("""
                SELECT COUNT(DISTINCT creator_id) FROM offers
                WHERE campaign_id = :cid AND status IN ('sent', 'accepted', 'countered')
            """), {"cid": req.campaign_id}).scalar() or 0
            open_slots = max(target - in_play, 0)

            candidates = [dict(r) for r in conn.execute(text("""
                SELECT m.id AS match_id, m.rank, m.suggested_payout,
                       cr.creator_id, cr.creator_name, cr.email, cr.platform, cr.niche, cr.basic_info
                FROM matches m JOIN creators cr ON cr.creator_id = m.creator_id
                WHERE m.campaign_id = :cid
                  AND m.approval_status = 'approved'
                  AND NOT EXISTS (SELECT 1 FROM offers o WHERE o.campaign_id = m.campaign_id
                                  AND o.creator_id = m.creator_id)
                  AND COALESCE(cr.email, '') <> ''
                  AND COALESCE(cr.unsubscribed, FALSE) = FALSE
                ORDER BY m.rank
                LIMIT :lim
            """), {"cid": req.campaign_id, "lim": open_slots}).mappings().all()]

        base = {
            "campaign_id": req.campaign_id,
            "target_creators": target,
            "offers_in_play": in_play,
            "open_slots": open_slots,
        }
        if not candidates:
            spare_needed = open_slots > 0
            return {**base, "status": "success", "count": 0, "offers": [],
                    "needs_more_creators": spare_needed,
                    "message": ("No approved creators left to offer. Approve more or rerun Fred."
                                if spare_needed else "All slots already have offers.")}

        # Offer terms shared by every creator in this campaign
        timeline = campaign["timeline_days"]
        short = timeline is not None and timeline <= int(s.get("short_timeline_days", 7))
        expiry_hours = int(s.get("offer_expiry_hours_short", 24) if short else s.get("offer_expiry_hours", 48))
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        expires_at = now + timedelta(hours=expiry_hours)

        if campaign["deadline"]:
            deadline = campaign["deadline"]
        elif timeline:
            deadline = (campaign["created_at"] + timedelta(days=timeline)).date()
        else:
            deadline = None

        # Aditya writes the personal openings (parallel batches, retries, brand name kept out)
        def v(x):
            return x if x not in (None, "") else "not specified"

        prompt = ADITYA_OFFER_PROMPT_TEMPLATE.format(
            brand_name=campaign["brand_name"], industry=v(campaign["industry"]),
            brand_info=v(campaign["brand_info"]), niche=v(campaign["niche"]),
            target_audience=v(campaign["target_audience"]),
            location=v(campaign["location_requirement"]), requirements=v(campaign["requirements"]),
            platform=v(campaign["platform"]), deliverables=v(campaign["deliverables"]),
        )
        items = []
        for c in candidates:
            c["offer_id"] = f"OFR_{req.campaign_id}_{c['creator_id']}_R1"
            items.append(SimpleNamespace(**c))

        intros = await generate_pitches_parallel(
            items, "offer_id", _format_offer_item, prompt, "creators",
            text_key="offer_intro", task="offer email openings"
        )

        offers, failed = [], []
        with engine.begin() as conn:
            for c in candidates:
                intro = intros.get(c["offer_id"])
                if not intro:
                    failed.append(c["creator_id"])
                    continue
                amount = int(c["suggested_payout"] or 0)
                intro, deliverable = _split_deliverable(intro)
                deliverable = deliverable or campaign["deliverables"]
                creator_platform = _creator_platform(c["platform"], campaign["platform"])
                terms = build_offer_terms(
                    amount, deliverable, creator_platform, deadline,
                    campaign["revisions_allowed"], campaign["exclusivity_days"],
                    campaign["industry"], expires_at, s
                )
                body = _strip_brand(intro.strip(), campaign["brand_name"]) + "\n\n" + terms
                subject = f"Paid collaboration offer for {_first_name(c['creator_name'])}: {format_inr(amount)}"

                rate = campaign["rate_per_collab"] if campaign["pricing_model"] == "per_collab" else None
                conn.execute(text("""
                    INSERT INTO offers (offer_id, campaign_id, creator_id, match_id, round,
                        offered_amount, brand_price_share, margin_pct, deliverables, deadline,
                        exclusivity_days, status, expires_at, subject, body)
                    VALUES (:offer_id, :campaign_id, :creator_id, :match_id, 1,
                        :amount, :share, :margin, :deliverables, :deadline,
                        :exclusivity, 'draft', :expires_at, :subject, :body)
                """), {
                    "offer_id": c["offer_id"], "campaign_id": req.campaign_id,
                    "creator_id": c["creator_id"], "match_id": c["match_id"],
                    "amount": amount, "share": rate,
                    "margin": round(1 - amount / rate, 3) if rate else None,
                    "deliverables": deliverable, "deadline": deadline,
                    "exclusivity": campaign["exclusivity_days"],
                    "expires_at": expires_at.replace(tzinfo=None),
                    "subject": subject, "body": body,
                })
                offers.append({
                    "offer_id": c["offer_id"], "creator_id": c["creator_id"],
                    "creator_name": c["creator_name"], "email": c["email"],
                    "amount": amount, "subject": subject, "body": body,
                })

        short_by = open_slots - len(offers)
        return {**base, "status": "success", "count": len(offers), "failed": failed,
                "needs_more_creators": short_by > 0,
                "message": (f"{short_by} slot(s) still open with no approved creator. Approve more or rerun Fred."
                            if short_by > 0 else "All open slots have offers."),
                "offers": offers}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing offers: {str(e)}")


class OfferSentRequest(BaseModel):
    offer_id: str
    gmail_message_id: Optional[str] = None
    gmail_thread_id: Optional[str] = None


@app.post("/offers/mark-sent")
async def mark_offer_sent(req: OfferSentRequest):
    """Record that an offer email went out, and move the campaign to offers_sent"""
    try:
        with engine.begin() as conn:
            offer = conn.execute(text("""
                UPDATE offers SET status = 'sent', sent_at = NOW(),
                       gmail_message_id = :msg, gmail_thread_id = :thread
                WHERE offer_id = :oid
                RETURNING campaign_id, creator_id, subject, body
            """), {"oid": req.offer_id, "msg": req.gmail_message_id,
                   "thread": req.gmail_thread_id}).mappings().first()
            if not offer:
                raise HTTPException(status_code=404, detail="Offer not found")

            conn.execute(text("""
                UPDATE campaigns SET status = 'offers_sent'
                WHERE campaign_id = :cid AND status IN ('brief_received', 'matching', 'awaiting_approval')
            """), {"cid": offer["campaign_id"]})

            conn.execute(text("""
                INSERT INTO conversations (contact_type, contact_id, gmail_thread_id, gmail_message_id,
                                           direction, subject, body, intent)
                VALUES ('creator', :cid, :thread, :msg, 'outbound', :subject, :body, 'offer')
                ON CONFLICT (gmail_message_id) DO NOTHING
            """), {"cid": offer["creator_id"], "thread": req.gmail_thread_id, "msg": req.gmail_message_id,
                   "subject": offer["subject"], "body": offer["body"]})

        return {"status": "success", "offer_id": req.offer_id, "campaign_id": offer["campaign_id"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marking offer sent: {str(e)}")


# ============ OFFER REPLIES (STAGE 5, PHASE B) ============
OFFER_REPLY_PROMPT = """You are Aditya, Creator Manager at an influencer marketing agency in India.
A creator replied to a paid collaboration offer. Classify the reply.

INTENT (pick exactly one):
- "accept": clearly agrees, with NO conditions and NO questions.
  e.g. "Yes, I accept", "I accept your offer", "Deal", "Sounds good, count me in"
- "decline": says no, not interested, or not available
- "counter": asks for a different payout, or accepts only if the payout changes
- "question": asks something, or adds conditions (dates, deliverables, advance, brand name)
  without clearly accepting or declining
- "auto_reply": out-of-office or automated message
- "unsubscribe": asks not to be contacted again

counter_amount: the payout they ask for, as an integer in INR ("12k" = 12000, "1.5L" = 150000), else null.
Never guess an amount.

Return ONLY a JSON object, no other text:
{"intent": "accept", "counter_amount": null, "reason": "short reason if they declined, else null",
 "questions": ["any questions they asked"], "summary": "one sentence for the founder"}
"""

OFFER_INTENTS = {"accept", "decline", "counter", "question", "auto_reply", "unsubscribe"}


def build_acceptance_email(first_name, brand_name, deliverable, deadline, amount, script_step, s,
                           opening: str = "Thank you, and welcome aboard! 🎉") -> str:
    """Confirmation after a creator accepts. Honest: the brand still gives final confirmation."""
    steps = [f"Once {brand_name} gives final confirmation, we'll send you the full brief."]
    if script_step:
        steps.append("Before shooting, you'll share a short script or concept so the brand can approve the direction.")
    steps.append("After the brand approves your content, you post it, and we take care of the rest.")
    numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(steps, start=1))

    return f"""Hi {first_name},

{opening}

The brand is {brand_name}. You're now on the final creator lineup we're presenting to them, and we'll confirm with you shortly.

Here's a recap of what you accepted:
• Deliverables: {deliverable or 'to be confirmed with the brief'}
• Content deadline: {deadline.strftime('%d %b %Y') if deadline else 'to be confirmed with the brief'}
• Your payout: {format_inr(amount)}
• Payment: {creator_payment_terms(amount, s)}

What happens next:
{numbered}

Reply here anytime if you have questions.

Aditya
Creator Manager"""


def build_decline_email(first_name, unsubscribed: bool) -> str:
    if unsubscribed:
        body = "Understood. We've removed you from our list and won't contact you again."
    else:
        body = ("No problem at all, thanks for letting us know. "
                "We'll keep you in mind for future campaigns that suit you better.")
    return f"Hi {first_name},\n\n{body}\n\nAditya\nCreator Manager"


class OfferReplyRequest(BaseModel):
    offer_id: str
    gmail_message_id: str                   # the creator's reply
    gmail_thread_id: Optional[str] = None
    from_email: Optional[str] = None
    reply_subject: Optional[str] = None
    reply_body: str


@app.post("/offers/parse-reply")
async def parse_offer_reply(req: OfferReplyRequest):
    """
    Understand a creator's reply to an offer and apply the rules:
    accept -> confirm + reveal brand, decline -> thank + next creator,
    counter / question -> Deven in Slack. Returns what N8N should send.
    """
    try:
        s = load_settings()

        with engine.connect() as conn:
            # The same email is never processed twice
            if conn.execute(text("SELECT 1 FROM conversations WHERE gmail_message_id = :m"),
                            {"m": req.gmail_message_id}).first():
                return {"status": "success", "action": "duplicate", "offer_id": req.offer_id,
                        "reply_email": None, "start_next_offers": False,
                        "slack_replies_text": None, "slack_approvals_text": None}

            # Always use the LATEST round of this negotiation (a revised offer shares the thread)
            offer = conn.execute(text("""
                WITH requested AS (SELECT campaign_id, creator_id FROM offers WHERE offer_id = :oid)
                SELECT o.*, cr.creator_name, c.campaign_id AS cid, c.pricing_model, c.rate_per_collab,
                       c.total_budget, c.slots, c.margin_target, c.margin_floor, b.brand_name
                FROM offers o
                JOIN requested r ON r.campaign_id = o.campaign_id AND r.creator_id = o.creator_id
                JOIN creators cr ON cr.creator_id = o.creator_id
                JOIN campaigns c ON c.campaign_id = o.campaign_id
                JOIN brands b ON b.brand_id = c.brand_id
                WHERE o.status <> 'draft'
                ORDER BY o.round DESC
                LIMIT 1
            """), {"oid": req.offer_id}).mappings().first()
        if not offer:
            raise HTTPException(status_code=404, detail="Offer not found")
        offer = dict(offer)
        oid = offer["offer_id"]

        # ---------- Classify the reply ----------
        message = f"""OFFER
Payout offered: INR {offer['offered_amount']}
Deliverables: {offer['deliverables'] or 'not specified'}

CREATOR'S REPLY:
{strip_quoted_reply(req.reply_body)}

Return ONLY the JSON object."""
        parsed = parse_json_object(await call_claude(OFFER_REPLY_PROMPT, message))
        intent = str(parsed.get("intent", "question")).strip().lower()
        if intent not in OFFER_INTENTS:
            intent = "question"
        counter_amount = to_int_or_none(parsed.get("counter_amount"))
        if intent == "counter" and not counter_amount:
            intent = "question"          # a counter without a number needs a human
        questions = parsed.get("questions") or []
        reason = clean_text(parsed.get("reason"))
        summary = parsed.get("summary") or ""

        name = offer["creator_name"]
        first = _first_name(name)
        brand = offer["brand_name"]
        offered = offer["offered_amount"]
        margin_target = float(offer["margin_target"] or s.get("margin_target", 0.30))
        margin_floor = float(offer["margin_floor"] or s.get("margin_floor", 0.20))

        result = {"status": "success", "offer_id": oid, "campaign_id": offer["cid"],
                  "intent": intent, "summary": summary, "reply_email": None,
                  "start_next_offers": False, "slack_approvals_text": None,
                  "slack_replies_text": None, "all_slots_filled": False}

        # Replies after the offer is already settled: just keep Deven informed
        if offer["status"] not in ("sent", "countered"):
            if intent != "auto_reply":
                result["slack_replies_text"] = f"💬 *{name}* wrote again about the {brand} offer ({offer['status']}): {summary}"
                if questions:
                    result["slack_approvals_text"] = (f"❓ *{name}* has a question about the {brand} offer:\n"
                                                      + "\n".join(f"• {q}" for q in questions)
                                                      + "\nReply to them in Gmail in the same thread.")
            action = "note"
        else:
            action = intent

        # ---------- Margin if a counter were accepted ----------
        def margin_at(amount: int):
            if offer["pricing_model"] == "per_collab" and offer["rate_per_collab"]:
                return 1 - amount / offer["rate_per_collab"]
            if offer["total_budget"]:
                with engine.connect() as c2:
                    committed = c2.execute(text("""
                        SELECT COALESCE(SUM(COALESCE(final_amount, offered_amount)), 0) FROM offers
                        WHERE campaign_id = :cid AND offer_id <> :oid AND status IN ('sent', 'accepted', 'countered')
                    """), {"cid": offer["cid"], "oid": oid}).scalar() or 0
                return 1 - (committed + amount) / offer["total_budget"]
            return None

        auto_accept_margin = float(s.get("counter_auto_accept_margin", 0))
        accepted_amount = None

        if action == "counter":
            m = margin_at(counter_amount)
            if auto_accept_margin > 0 and m is not None and m >= auto_accept_margin:
                action, accepted_amount = "accept", counter_amount       # auto-accept is switched on and margin holds
            else:
                pct = lambda x: f"{round(x * 100)}%" if x is not None else "unknown"
                warn = " 🔴 *below your floor*" if (m is not None and m < margin_floor) else ""
                result["slack_approvals_text"] = (
                    f"💬 *{name}* countered on the {brand} offer: asks *{format_inr(counter_amount)}* "
                    f"(offered {format_inr(offered)}).\n"
                    f"Your margin on this slot would be *{pct(m)}* (target {pct(margin_target)}, floor {pct(margin_floor)}){warn}.\n"
                    f"Use the form below to accept, decline, or offer a different amount."
                )
                result["slack_replies_text"] = f"💬 *{name}* countered on {brand}: {format_inr(counter_amount)} vs {format_inr(offered)} offered."

        if action == "question":
            result["slack_approvals_text"] = (f"❓ *{name}* has a question about the {brand} offer:\n"
                                              + ("\n".join(f"• {q}" for q in questions) if questions else f"• {summary}")
                                              + "\nReply to them in Gmail in the same thread.")
            result["slack_replies_text"] = f"❓ *{name}* asked about the {brand} offer: {summary}"

        # ---------- Apply state changes ----------
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO conversations (contact_type, contact_id, gmail_thread_id, gmail_message_id,
                                           direction, from_email, subject, body, intent, extracted_data)
                VALUES ('creator', :cid, :thread, :msg, 'inbound', :from_email, :subject, :body,
                        :intent, CAST(:data AS JSONB))
                ON CONFLICT (gmail_message_id) DO NOTHING
            """), {"cid": offer["creator_id"], "thread": req.gmail_thread_id, "msg": req.gmail_message_id,
                   "from_email": req.from_email, "subject": req.reply_subject, "body": req.reply_body,
                   "intent": f"offer_{intent}", "data": json.dumps(parsed)})

            if action == "accept":
                final = accepted_amount or offered
                conn.execute(text("""
                    UPDATE offers SET status = 'accepted', final_amount = :final,
                           counter_amount = COALESCE(:counter, counter_amount), responded_at = NOW()
                    WHERE offer_id = :oid
                """), {"final": final, "counter": accepted_amount, "oid": oid})

                deal_value = offer["total_budget"] or ((offer["rate_per_collab"] or 0) * (offer["slots"] or 1))
                script_step = deal_value >= float(s.get("small_deal_threshold_inr", 25000))
                result["reply_email"] = build_acceptance_email(
                    first, brand, offer["deliverables"], offer["deadline"], final, script_step, s)

                # Slots filled?
                accepted = conn.execute(text("""
                    SELECT COUNT(*) FROM offers WHERE campaign_id = :cid AND status = 'accepted'
                """), {"cid": offer["cid"]}).scalar() or 0
                if offer["pricing_model"] == "per_collab":
                    target = offer["slots"] or 1
                else:
                    target = conn.execute(text("""
                        SELECT COUNT(*) FROM matches WHERE campaign_id = :cid AND approval_status = 'approved'
                    """), {"cid": offer["cid"]}).scalar() or 0
                note = " (accepted your counter)" if accepted_amount else ""
                result["slack_replies_text"] = (f"✅ *{name}* accepted the {brand} offer{note} at {format_inr(final)}. "
                                                f"{accepted}/{target} slot(s) filled.")
                if target and accepted >= target:
                    result["all_slots_filled"] = True
                    result["slack_approvals_text"] = (f"🎯 *All {target} slot(s) filled for {brand}* "
                                                      f"(`{offer['cid']}`). Preparing the lineup for the brand.")

            elif action in ("decline", "unsubscribe"):
                conn.execute(text("""
                    UPDATE offers SET status = 'declined', decline_reason = :reason, responded_at = NOW()
                    WHERE offer_id = :oid
                """), {"reason": reason or ("unsubscribed" if action == "unsubscribe" else None), "oid": oid})
                if action == "unsubscribe":
                    conn.execute(text("""
                        UPDATE creators SET unsubscribed = TRUE, status = 'unsubscribed' WHERE creator_id = :cid
                    """), {"cid": offer["creator_id"]})
                result["reply_email"] = build_decline_email(first, action == "unsubscribe")
                result["start_next_offers"] = True
                why = f" ({reason})" if reason else ""
                result["slack_replies_text"] = (f"❌ *{name}* declined the {brand} offer{why}. "
                                                f"Sending the next offer if an approved creator is available.")

            elif action == "counter":
                conn.execute(text("""
                    UPDATE offers SET status = 'countered', counter_amount = :amt, responded_at = NOW()
                    WHERE offer_id = :oid
                """), {"amt": counter_amount, "oid": oid})

        result["action"] = action
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling offer reply: {str(e)}")


# ============ COUNTERS + EXPIRY (STAGE 5, PHASE C) ============
def build_revised_offer_email(first_name, amount, deliverable, deadline, expires_at, note, s) -> str:
    note_block = f"\n\n{note.strip()}" if note and note.strip() else ""
    return f"""Hi {first_name},

Thanks for coming back to us. We can offer {format_inr(amount)} for this collaboration.{note_block}

Everything else stays the same:
• Deliverables: {deliverable or 'to be confirmed with the brief'}
• Content deadline: {deadline.strftime('%d %b %Y') if deadline else 'to be confirmed with the brief'}
• Your payout: {format_inr(amount)}
• Payment: {creator_payment_terms(amount, s)}
• This offer is open until {expires_at.astimezone(IST).strftime('%d %b, %I:%M %p')} IST

Just reply "Accept" to confirm.

Aditya
Creator Manager"""


def build_counter_decline_email(first_name, counter_amount) -> str:
    return f"""Hi {first_name},

Thanks for sharing your rate. Unfortunately we can't go up to {format_inr(counter_amount)} for this campaign, so we'll have to pass this time.

We'd love to work with you on a future campaign that fits your rate better.

Aditya
Creator Manager"""


def build_expiry_email(first_name) -> str:
    return f"""Hi {first_name},

Just a quick note: the collaboration offer we sent has now closed, as we needed to confirm creators for the campaign.

No worries at all. We'll reach out again when a campaign that suits you comes up.

Aditya
Creator Manager"""


def _slot_status(conn, campaign_id, pricing_model, slots):
    """(accepted, target) for a campaign"""
    accepted = conn.execute(text("""
        SELECT COUNT(*) FROM offers WHERE campaign_id = :cid AND status = 'accepted'
    """), {"cid": campaign_id}).scalar() or 0
    if pricing_model == "per_collab":
        target = slots or 1
    else:
        target = conn.execute(text("""
            SELECT COUNT(*) FROM matches WHERE campaign_id = :cid AND approval_status = 'approved'
        """), {"cid": campaign_id}).scalar() or 0
    return accepted, target


class ResolveCounterRequest(BaseModel):
    offer_id: str
    decision: str                 # "Accept their number" / "Decline" / "Offer a different amount"
    amount: Optional[int] = None  # only for a different amount
    note: Optional[str] = None    # optional line added to the creator email


@app.post("/offers/resolve-counter")
async def resolve_counter(req: ResolveCounterRequest):
    """Apply Deven's decision on a counter-offer and return the email Aditya should send"""
    try:
        s = load_settings()
        d = (req.decision or "").strip().lower()
        if d.startswith("accept"):
            decision = "accept"
        elif d.startswith("decline"):
            decision = "decline"
        elif d.startswith("offer") or d.startswith("propose") or "different" in d:
            decision = "propose"
        else:
            raise HTTPException(status_code=400, detail="decision must be accept, decline or a different amount")
        if decision == "propose" and (not req.amount or req.amount <= 0):
            raise HTTPException(status_code=400, detail="A different amount needs a number")

        with engine.begin() as conn:
            offer = conn.execute(text("""
                SELECT o.*, cr.creator_name, c.pricing_model, c.rate_per_collab, c.total_budget, c.slots,
                       c.timeline_days, c.margin_floor, b.brand_name
                FROM offers o
                JOIN creators cr ON cr.creator_id = o.creator_id
                JOIN campaigns c ON c.campaign_id = o.campaign_id
                JOIN brands b ON b.brand_id = c.brand_id
                WHERE o.offer_id = :oid
            """), {"oid": req.offer_id}).mappings().first()
            if not offer:
                raise HTTPException(status_code=404, detail="Offer not found")
            offer = dict(offer)

            base = {"status": "success", "offer_id": req.offer_id, "campaign_id": offer["campaign_id"],
                    "decision": decision, "reply_email": None, "reply_to_message_id": None,
                    "new_offer_id": None, "start_next_offers": False, "slack_text": None,
                    "all_slots_filled": False}

            if offer["status"] != "countered":
                return {**base, "status": "skipped",
                        "slack_text": f"ℹ️ The offer to {offer['creator_name']} is already `{offer['status']}`, so nothing was changed."}

            # Reply to the creator's latest message in this thread
            reply_to = conn.execute(text("""
                SELECT gmail_message_id FROM conversations
                WHERE gmail_thread_id = :t AND direction = 'inbound'
                ORDER BY created_at DESC LIMIT 1
            """), {"t": offer["gmail_thread_id"]}).scalar() or offer["gmail_message_id"]

            first = _first_name(offer["creator_name"])
            name, brand = offer["creator_name"], offer["brand_name"]
            counter = offer["counter_amount"]
            base["reply_to_message_id"] = reply_to

            if decision == "accept":
                conn.execute(text("""
                    UPDATE offers SET status = 'accepted', final_amount = :final WHERE offer_id = :oid
                """), {"final": counter, "oid": req.offer_id})
                deal_value = offer["total_budget"] or ((offer["rate_per_collab"] or 0) * (offer["slots"] or 1))
                script_step = deal_value >= float(s.get("small_deal_threshold_inr", 25000))
                base["reply_email"] = build_acceptance_email(
                    first, brand, offer["deliverables"], offer["deadline"], counter, script_step, s,
                    opening=f"Good news: we can do {format_inr(counter)}. Welcome aboard! 🎉")
                accepted, target = _slot_status(conn, offer["campaign_id"], offer["pricing_model"], offer["slots"])
                text_out = f"✅ Counter accepted: *{name}* joins {brand} at {format_inr(counter)}. {accepted}/{target} slot(s) filled."
                if target and accepted >= target:
                    text_out += f"\n🎯 *All {target} slot(s) filled for {brand}*. Preparing the lineup for the brand."
                    base["all_slots_filled"] = True
                base["slack_text"] = text_out

            elif decision == "decline":
                conn.execute(text("""
                    UPDATE offers SET status = 'declined', decline_reason = :reason WHERE offer_id = :oid
                """), {"reason": f"counter of {counter} declined by agency", "oid": req.offer_id})
                base["reply_email"] = build_counter_decline_email(first, counter)
                base["start_next_offers"] = True
                base["slack_text"] = (f"❌ Counter declined: *{name}* ({format_inr(counter)}) is out of {brand}. "
                                      f"Sending the next offer if an approved creator is available.")

            else:  # propose a different amount: new negotiation round
                amount = int(req.amount)
                short = offer["timeline_days"] is not None and offer["timeline_days"] <= int(s.get("short_timeline_days", 7))
                hours = int(s.get("offer_expiry_hours_short", 24) if short else s.get("offer_expiry_hours", 48))
                expires_at = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=hours)
                new_round = (offer["round"] or 1) + 1
                new_id = f"OFR_{offer['campaign_id']}_{offer['creator_id']}_R{new_round}"
                rate = offer["rate_per_collab"] if offer["pricing_model"] == "per_collab" else None
                body = build_revised_offer_email(first, amount, offer["deliverables"], offer["deadline"],
                                                 expires_at, req.note, s)

                conn.execute(text("UPDATE offers SET status = 'withdrawn' WHERE offer_id = :oid"),
                             {"oid": req.offer_id})
                conn.execute(text("DELETE FROM offers WHERE offer_id = :nid AND status = 'draft'"), {"nid": new_id})
                conn.execute(text("""
                    INSERT INTO offers (offer_id, campaign_id, creator_id, match_id, round, offered_amount,
                        brand_price_share, margin_pct, deliverables, deadline, exclusivity_days, status,
                        expires_at, subject, body, gmail_thread_id)
                    VALUES (:nid, :cid, :creator, :match, :round, :amount, :share, :margin, :deliverables,
                        :deadline, :excl, 'draft', :expires, :subject, :body, :thread)
                """), {
                    "nid": new_id, "cid": offer["campaign_id"], "creator": offer["creator_id"],
                    "match": offer["match_id"], "round": new_round, "amount": amount, "share": rate,
                    "margin": round(1 - amount / rate, 3) if rate else None,
                    "deliverables": offer["deliverables"], "deadline": offer["deadline"],
                    "excl": offer["exclusivity_days"], "expires": expires_at.replace(tzinfo=None),
                    "subject": f"Revised offer: {format_inr(amount)}", "body": body,
                    "thread": offer["gmail_thread_id"],
                })
                base["new_offer_id"] = new_id
                base["reply_email"] = body
                margin = f" Your margin on this slot: {round((1 - amount / rate) * 100)}%." if rate else ""
                base["slack_text"] = (f"💬 Revised offer sent to *{name}*: {format_inr(amount)} "
                                      f"(they asked {format_inr(counter)}).{margin}")

        return base
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resolving counter: {str(e)}")


@app.post("/offers/expire")
async def expire_offers():
    """Close offers nobody answered in time. Counters waiting on Deven never expire."""
    try:
        with engine.begin() as conn:
            rows = [dict(r) for r in conn.execute(text("""
                UPDATE offers o SET status = 'expired'
                FROM creators cr, campaigns c, brands b
                WHERE o.status = 'sent'
                  AND o.expires_at IS NOT NULL AND o.expires_at < NOW()
                  AND cr.creator_id = o.creator_id
                  AND c.campaign_id = o.campaign_id
                  AND b.brand_id = c.brand_id
                RETURNING o.offer_id, o.campaign_id, o.gmail_message_id, o.gmail_thread_id,
                          cr.creator_name, b.brand_name
            """)).mappings().all()]

        expired = [{
            "offer_id": r["offer_id"],
            "campaign_id": r["campaign_id"],
            "creator_name": r["creator_name"],
            "brand_name": r["brand_name"],
            "reply_to_message_id": r["gmail_message_id"],   # our offer email: the note goes in the same thread
            "note_email": build_expiry_email(_first_name(r["creator_name"])),
        } for r in rows]
        campaigns = [{"campaign_id": cid} for cid in sorted({r["campaign_id"] for r in rows})]

        slack_text = None
        if expired:
            lines = [f"⌛ *{len(expired)} offer(s) expired* with no reply:"]
            lines += [f"• {e['creator_name']} ({e['brand_name']})" for e in expired]
            lines.append("Next offers are going out where approved creators are available.")
            slack_text = "\n".join(lines)

        return {"status": "success", "count": len(expired), "expired": expired,
                "campaigns": campaigns, "slack_text": slack_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error expiring offers: {str(e)}")


# ============ BRAND LINEUP (STAGE 6, PHASE A) ============
def _brand_price_lines(price: int, s: dict) -> dict:
    """Price, GST and the 70/30 split. GST appears only once registered."""
    registered = float(s.get("gst_registered", 0)) >= 1
    rate = float(s.get("gst_rate", 0.18))
    advance_pct = float(s.get("advance_pct", 0.7))
    advance = int(round(price * advance_pct))
    balance = price - advance
    gst = int(round(price * rate)) if registered else 0
    suffix = " + GST" if registered else ""
    total_line = (f"{format_inr(price)} + {int(rate * 100)}% GST ({format_inr(gst)}) = {format_inr(price + gst)} total"
                  if registered else f"{format_inr(price)} total")
    return {
        "price": price, "gst": gst, "advance": advance, "balance": balance,
        "total_line": total_line,
        "advance_line": f"{int(advance_pct * 100)}% advance ({format_inr(advance)}{suffix}) to confirm the lineup",
        "balance_line": f"Remaining {100 - int(advance_pct * 100)}% ({format_inr(balance)}{suffix}) before the content goes live",
    }


class LineupRequest(BaseModel):
    campaign_id: str


@app.post("/lineup/prepare")
async def prepare_lineup(req: LineupRequest):
    """
    Build the anonymized lineup email for the brand from accepted offers.
    Prices, terms and dates come from data; creator names never appear.
    """
    try:
        s = load_settings()
        with engine.begin() as conn:
            c = conn.execute(text("""
                SELECT c.*, b.brand_name, b.industry
                FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.campaign_id = :cid
            """), {"cid": req.campaign_id}).mappings().first()
            if not c:
                raise HTTPException(status_code=404, detail="Campaign not found")
            c = dict(c)

            creators = [dict(r) for r in conn.execute(text("""
                SELECT o.offer_id, o.deliverables, cr.creator_name, cr.handle, cr.platform,
                       cr.follower_count, cr.engagement_rate, cr.location, cr.niche,
                       m.anon_summary
                FROM offers o
                JOIN creators cr ON cr.creator_id = o.creator_id
                LEFT JOIN matches m ON m.id = o.match_id
                WHERE o.campaign_id = :cid AND o.status = 'accepted'
                ORDER BY m.rank NULLS LAST, o.offer_id
            """), {"cid": req.campaign_id}).mappings().all()]
            if not creators:
                raise HTTPException(status_code=400, detail="No accepted creators for this campaign yet")

            # The brand's latest message in the campaign thread: the lineup goes out as a reply to it
            reply_to = conn.execute(text("""
                SELECT gmail_message_id FROM conversations
                WHERE gmail_thread_id = :t AND direction = 'inbound' AND contact_type = 'brand'
                ORDER BY created_at DESC LIMIT 1
            """), {"t": c["gmail_thread_id"]}).scalar()

            # Brand price: per collab = rate x confirmed creators; campaign pool = the agreed budget
            if c["pricing_model"] == "per_collab" and c["rate_per_collab"]:
                price = c["rate_per_collab"] * len(creators)
                per_creator = f" ({format_inr(c['rate_per_collab'])} per creator)"
            else:
                price = c["total_budget"] or 0
                per_creator = ""
            p = _brand_price_lines(price, s)

            # Confirmation window
            short = c["timeline_days"] is not None and c["timeline_days"] <= int(s.get("short_timeline_days", 7))
            hours = int(s.get("lineup_window_hours_short", 24) if short else s.get("lineup_window_hours", 72))
            expires_at = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=hours)

            if c["deadline"]:
                deadline = c["deadline"]
            elif c["timeline_days"]:
                deadline = (c["created_at"] + timedelta(days=c["timeline_days"])).date()
            else:
                deadline = None

            # Anonymized creator blocks; the label -> offer mapping is saved so replies hit the right creator
            blocks, labels = [], {}
            for i, cr in enumerate(creators):
                labels[f"Creator {chr(65 + i)}"] = cr["offer_id"]
                facts = " · ".join(x for x in [
                    PLATFORM_NAMES.get(str(cr["platform"] or "").lower(), cr["platform"]),
                    f"{_format_followers(cr['follower_count'])} followers" if cr["follower_count"] else None,
                    f"{cr['engagement_rate']}% engagement" if cr["engagement_rate"] else None,
                    cr["location"],
                ] if x)
                why = cr["anon_summary"] or f"{(cr['niche'] or 'Content').capitalize()} creator with an engaged audience."
                why = _anonymize(why, cr)
                blocks.append(f"Creator {chr(65 + i)} · {facts}\n{why}\nDeliverable: {cr['deliverables'] or 'as per the brief'}")

            terms = [
                f"• {p['advance_line']}",
                f"• {p['balance_line']}",
                "• The advance becomes non-refundable once creators have started producing content",
            ]
            if c["revisions_allowed"]:
                terms.append(f"• Up to {c['revisions_allowed']} round{'s' if c['revisions_allowed'] > 1 else ''} of revisions per creator")
            if deadline:
                terms.append(f"• Content live by {deadline.strftime('%d %b %Y')}")
            if c["exclusivity_days"]:
                terms.append(f"• Creators won't post for competing {(c['industry'] or '').strip() or 'category'} brands "
                             f"for {c['exclusivity_days']} days after publishing")

            months = int(s.get("non_circumvention_months", 12))
            n = len(creators)
            body = f"""Hi {c['brand_name']} team,

{"Here's your updated creator lineup." if c["lineup_sent_at"] else "Your creator lineup is ready."} All {n} creator{'s have' if n > 1 else ' has'} already accepted, so the campaign can start as soon as you confirm.

YOUR LINEUP

""" + "\n\n".join(blocks) + f"""

PRICE
{p['total_line']} for {n} creator{'s' if n > 1 else ''}{per_creator}
""" + "\n".join(terms) + f"""

Creator names and profiles are shared as soon as the advance is received. For the next {months} months, collaborations with creators introduced by us are arranged through us.

To confirm, just reply "Confirm". If you'd like to swap anyone or have questions, reply and let us know.
We're holding this lineup for you until {expires_at.astimezone(IST).strftime('%d %b, %I:%M %p')} IST.

Ananya
Brand Partnerships"""

            conn.execute(text("""
                UPDATE campaigns SET lineup_email = :body, final_brand_price = :price,
                       gst_amount = :gst, advance_amount = :advance, lineup_expires_at = :expires,
                       lineup_labels = CAST(:labels AS JSONB),
                       lineup_reminder_sent_at = NULL, lineup_deadline_alerted_at = NULL
                WHERE campaign_id = :cid
            """), {"body": body, "price": price, "gst": p["gst"], "advance": p["advance"],
                   "expires": expires_at.replace(tzinfo=None), "labels": json.dumps(labels),
                   "cid": req.campaign_id})

        return {
            "status": "success",
            "campaign_id": req.campaign_id,
            "brand_name": c["brand_name"],
            "creators": n,
            "price": price, "gst": p["gst"], "advance": p["advance"],
            "reply_to_message_id": reply_to,
            "body": body,
            "slack_preview": (f"📬 *Lineup ready for {c['brand_name']}* (`{req.campaign_id}`): {n} creator(s), "
                              f"{p['total_line']}.\nApprove to send it as a reply in their email thread.\n\n"
                              f"```{body}```"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing lineup: {str(e)}")


class LineupSentRequest(BaseModel):
    campaign_id: str
    gmail_message_id: Optional[str] = None
    gmail_thread_id: Optional[str] = None


@app.post("/lineup/mark-sent")
async def mark_lineup_sent(req: LineupSentRequest):
    """Record the lineup email and start the brand's confirmation window"""
    try:
        with engine.begin() as conn:
            row = conn.execute(text("""
                UPDATE campaigns SET status = 'shortlist_sent', lineup_sent_at = NOW(),
                       lineup_message_id = :msg
                WHERE campaign_id = :cid
                RETURNING brand_id, lineup_email, gmail_thread_id
            """), {"cid": req.campaign_id, "msg": req.gmail_message_id}).mappings().first()
            if not row:
                raise HTTPException(status_code=404, detail="Campaign not found")
            conn.execute(text("""
                INSERT INTO conversations (contact_type, contact_id, gmail_thread_id, gmail_message_id,
                                           direction, subject, body, intent)
                VALUES ('brand', :bid, :thread, :msg, 'outbound', 'Creator lineup', :body, 'lineup')
                ON CONFLICT (gmail_message_id) DO NOTHING
            """), {"bid": row["brand_id"], "thread": req.gmail_thread_id or row["gmail_thread_id"],
                   "msg": req.gmail_message_id, "body": row["lineup_email"]})
        return {"status": "success", "campaign_id": req.campaign_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marking lineup sent: {str(e)}")


# ============ BRAND CONFIRMATION + PAYMENT (STAGE 6, PHASE B) ============
PAYMENT_DETAILS = os.getenv("PAYMENT_DETAILS", "").replace("\\n", "\n").strip()

LINEUP_REPLY_PROMPT = """You are Ananya, Brand Partnerships Manager at an influencer marketing agency in India.
You sent a brand an anonymized creator lineup (Creator A, Creator B, ...) with a price. Classify their reply.

INTENT (pick exactly one):
- "confirm": clearly confirms or approves the lineup, with no conditions
- "payment_sent": says they have paid or transferred the advance
- "reject_creators": wants to remove or swap one or more specific creators
- "decline": does not want to go ahead with the campaign at all
- "question": asks something, or adds conditions, without clearly confirming
- "auto_reply": out-of-office or automated message

rejected_labels: for reject_creators, the creator labels they want removed, e.g. ["Creator B"]. Else [].

Return ONLY a JSON object, no other text:
{"intent": "confirm", "rejected_labels": [], "reason": "their reason if they reject or decline, else null",
 "questions": ["any questions they asked"], "summary": "one sentence for the founder"}
"""

LINEUP_INTENTS = {"confirm", "payment_sent", "reject_creators", "decline", "question", "auto_reply"}


def _profile_link(platform, handle):
    h = str(handle or "").strip().lstrip("@")
    if not h:
        return None
    p = str(platform or "").strip().lower()
    if p == "instagram":
        return f"https://www.instagram.com/{h}"
    if p == "youtube":
        return f"https://www.youtube.com/@{h}"
    return None


REJECTION_DRAFT_PROMPT = """You are Ananya, Brand Partnerships Manager at an influencer marketing agency in India.
A brand reviewed an anonymized creator lineup and wants to remove one or more creators. Write a short reply email.

Tone: curious and helpful, never defensive or pushy.
- Thank them for reviewing the lineup and say you're happy to adjust.
- For each creator they removed, give one or two sentences on why we picked them, using ONLY the facts provided.
- If they gave a reason, address it directly. Do NOT ask them why again.
- If they gave no reason, ask what they're looking for instead (audience, content style, follower range).
- Refer to creators only by their label (Creator A, Creator B). Never use names or handles.
- Never invent numbers, results or facts.
- Under 150 words. Sign off exactly as:
Ananya
Brand Partnerships

Return ONLY the email text, starting with "Hi"."""


def _match_labels(requested, labels: dict) -> list:
    """Map what the brand wrote ("Creator B", "creator b", "B") to offer IDs from the saved lineup"""
    by_letter = {k.split()[-1].upper(): v for k, v in (labels or {}).items()}
    found = []
    for r in requested or []:
        token = re.sub(r"(?i)creator", "", str(r)).strip().strip(".:,;").upper()
        if len(token) == 1 and token in by_letter and by_letter[token] not in found:
            found.append(by_letter[token])
    return found


async def draft_rejection_reply(brand: str, reason, rejected: list) -> str:
    """Ananya's draft. Falls back to a simple template if the AI call fails."""
    blocks = []
    for r in rejected:
        facts = ", ".join(x for x in [
            PLATFORM_NAMES.get(str(r.get("platform") or "").lower(), r.get("platform")),
            f"{_format_followers(r.get('follower_count'))} followers" if r.get("follower_count") else None,
            f"{r.get('engagement_rate')}% engagement" if r.get("engagement_rate") else None,
            r.get("location"), r.get("niche"),
        ] if x)
        why = _anonymize(r.get("anon_summary") or "", r)
        blocks.append(f"{r.get('label')}: {facts}. Why we picked them: {why or 'not recorded'}")
    try:
        draft = (await call_claude(REJECTION_DRAFT_PROMPT, f"""BRAND: {brand}
THEIR REASON: {reason or 'none given'}

CREATORS THEY WANT TO REMOVE:
""" + "\n".join(blocks))).strip()
    except Exception as e:
        print(f"Rejection draft failed for {brand}: {e}")
        labels = ", ".join(r.get("label") or "the creator" for r in rejected)
        ask = ("" if reason else " So we can find the right fit, could you share what you're looking for, "
               "e.g. audience, content style or follower range?")
        draft = (f"Hi {brand} team,\n\nThanks for reviewing the lineup. We're happy to adjust {labels}.{ask}"
                 f"\n\nAnanya\nBrand Partnerships")
    for r in rejected:
        draft = _anonymize(draft, r)
    return draft


class LineupReplyRequest(BaseModel):
    campaign_id: str
    gmail_message_id: str
    gmail_thread_id: Optional[str] = None
    from_email: Optional[str] = None
    reply_subject: Optional[str] = None
    reply_body: str


@app.post("/lineup/parse-reply")
async def parse_lineup_reply(req: LineupReplyRequest):
    """Understand a brand's reply to the lineup and decide the next step"""
    try:
        s = load_settings()
        with engine.connect() as conn:
            if conn.execute(text("SELECT 1 FROM conversations WHERE gmail_message_id = :m"),
                            {"m": req.gmail_message_id}).first():
                return {"status": "success", "action": "duplicate", "campaign_id": req.campaign_id,
                        "reply_email": None, "start_payment_wait": False,
                        "slack_replies_text": None, "slack_approvals_text": None}
            c = conn.execute(text("""
                SELECT c.*, b.brand_name FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.campaign_id = :cid
            """), {"cid": req.campaign_id}).mappings().first()
        if not c:
            raise HTTPException(status_code=404, detail="Campaign not found")
        c = dict(c)
        brand = c["brand_name"]

        parsed = parse_json_object(await call_claude(
            LINEUP_REPLY_PROMPT,
            f"BRAND: {brand}\n\nTHEIR REPLY:\n{strip_quoted_reply(req.reply_body)}\n\nReturn ONLY the JSON object."
        ))
        intent = str(parsed.get("intent", "question")).strip().lower()
        if intent not in LINEUP_INTENTS:
            intent = "question"
        summary = parsed.get("summary") or ""
        questions = parsed.get("questions") or []
        reason = clean_text(parsed.get("reason"))

        result = {"status": "success", "campaign_id": req.campaign_id, "intent": intent, "summary": summary,
                  "reply_email": None, "start_payment_wait": False, "payment_prompt": None,
                  "start_rejection_review": False, "rejection_prompt": None, "rejection_draft": None,
                  "start_close_review": False, "close_prompt": None, "close_reason": None,
                  "slack_replies_text": None, "slack_approvals_text": None}
        action = intent

        # Confirming only counts while the lineup is open
        if intent == "confirm" and c["status"] != "shortlist_sent":
            action = "question" if c["status"] != "confirmed" else "note"

        # Rejections and declines only apply while the lineup is open
        if intent in ("reject_creators", "decline") and c["status"] != "shortlist_sent":
            action = "question"

        # Work out exactly which creators, and draft Ananya's reply, BEFORE any database transaction
        rejected, draft = [], None
        if action == "reject_creators":
            saved_labels = c.get("lineup_labels") or {}
            rejected_ids = _match_labels(parsed.get("rejected_labels"), saved_labels)
            if not rejected_ids:
                action = "question"          # can't tell which creator they mean: Deven reads it
            else:
                with engine.connect() as conn:
                    rows = conn.execute(text("""
                        SELECT o.offer_id, cr.creator_name, cr.handle, cr.platform, cr.follower_count,
                               cr.engagement_rate, cr.location, cr.niche, m.anon_summary
                        FROM offers o
                        JOIN creators cr ON cr.creator_id = o.creator_id
                        LEFT JOIN matches m ON m.id = o.match_id
                        WHERE o.offer_id = ANY(:ids)
                    """), {"ids": rejected_ids}).mappings().all()
                label_of = {v: k for k, v in saved_labels.items()}
                rejected = [{**dict(r), "label": label_of.get(r["offer_id"])} for r in rows]
                draft = await draft_rejection_reply(brand, reason, rejected)

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO conversations (contact_type, contact_id, gmail_thread_id, gmail_message_id,
                                           direction, from_email, subject, body, intent, extracted_data)
                VALUES ('brand', :bid, :thread, :msg, 'inbound', :from_email, :subject, :body,
                        :intent, CAST(:data AS JSONB))
                ON CONFLICT (gmail_message_id) DO NOTHING
            """), {"bid": c["brand_id"], "thread": req.gmail_thread_id, "msg": req.gmail_message_id,
                   "from_email": req.from_email, "subject": req.reply_subject, "body": req.reply_body,
                   "intent": f"lineup_{intent}", "data": json.dumps(parsed)})

            if action == "confirm":
                conn.execute(text("UPDATE campaigns SET status = 'confirmed' WHERE campaign_id = :cid"),
                             {"cid": req.campaign_id})
                conn.execute(text("""
                    UPDATE offers SET brand_rejected_at = NULL, release_after = NULL
                    WHERE campaign_id = :cid AND status = 'accepted'
                """), {"cid": req.campaign_id})
                advance = c["advance_amount"] or 0
                registered = float(s.get("gst_registered", 0)) >= 1
                gst_on_advance = int(round(advance * float(s.get("gst_rate", 0.18)))) if registered else 0
                amount_line = (f"{format_inr(advance)} + GST ({format_inr(gst_on_advance)}) = {format_inr(advance + gst_on_advance)}"
                               if registered else format_inr(advance))
                balance = (c["final_brand_price"] or 0) - advance
                balance_line = format_inr(balance) + (" + GST" if registered else "")
                pct = int(float(s.get("advance_pct", 0.7)) * 100)
                result["payment_prompt"] = (f"💰 *{brand} confirmed the lineup!* Waiting for the {pct}% advance of "
                                            f"*{amount_line}*.\nTap the button once it's in your account.")
                if PAYMENT_DETAILS:
                    result["reply_email"] = f"""Hi {brand} team,

Thank you for confirming! 🎉

To lock in the lineup, please transfer the {pct}% advance:
Amount: {amount_line}

{PAYMENT_DETAILS}

Please reply to this email once the transfer is done. As soon as we receive it, we'll share each creator's name and profile with you and send them the full brief.

The remaining {100 - pct}% ({balance_line}) is due before the content goes live.

Ananya
Brand Partnerships"""
                    result["start_payment_wait"] = True
                    result["slack_replies_text"] = f"🎉 *{brand}* confirmed the lineup. Advance request sent ({amount_line})."
                else:
                    result["slack_approvals_text"] = (f"⚠️ *{brand}* confirmed the lineup, but PAYMENT_DETAILS isn't set on "
                                                      f"Railway, so no advance request was sent. Add it, then send the request manually.")

            elif action == "payment_sent":
                result["slack_approvals_text"] = (f"💸 *{brand}* says the advance has been sent. Check your account, "
                                                  f"then tap *Payment received* on the waiting message.")
                result["slack_replies_text"] = f"💸 *{brand}*: {summary}"

            elif action == "reject_creators":
                hold = int(s.get("creator_hold_hours", 48))
                conn.execute(text("""
                    UPDATE offers SET brand_rejected_at = NOW(), release_after = NOW() + make_interval(hours => :h)
                    WHERE offer_id = ANY(:ids)
                """), {"h": hold, "ids": [r["offer_id"] for r in rejected]})
                labels_txt = ", ".join(r["label"] for r in rejected if r["label"])
                names = ", ".join(r["creator_name"] for r in rejected)
                why = f"\nTheir reason: _{reason}_" if reason else "\nNo reason given."
                result["start_rejection_review"] = True
                result["rejection_draft"] = draft
                result["rejection_prompt"] = (
                    f"🔁 *{brand}* wants to remove {labels_txt} ({names}).{why}\n"
                    f"Held for {hold} hours.\n\n*Ananya's draft reply:*\n```{draft}```\n\n"
                    f"Send the draft, edit it, replace the creator, or proceed with fewer creators.")
                result["slack_replies_text"] = f"🔁 *{brand}* wants to remove {labels_txt}."

            elif action == "decline":
                why = f" Reason: _{reason}_" if reason else ""
                result["start_close_review"] = True
                result["close_reason"] = reason
                result["close_prompt"] = (f"🛑 *{brand}* doesn't want to go ahead with the campaign.{why}\n"
                                          f"Close it and release the creators, or keep it open to follow up yourself?")
                result["slack_replies_text"] = f"🛑 *{brand}* declined the lineup."

            elif action == "question":
                result["slack_approvals_text"] = (f"❓ *{brand}* has a question about the lineup:\n"
                                                  + ("\n".join(f"• {q}" for q in questions) if questions else f"• {summary}")
                                                  + "\nReply to them in Gmail in the same thread.")
                result["slack_replies_text"] = f"❓ *{brand}* asked about the lineup: {summary}"

            elif action == "note":
                result["slack_replies_text"] = f"💬 *{brand}* wrote again about the confirmed lineup: {summary}"

        result["action"] = action
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling lineup reply: {str(e)}")


CREATOR_POINTERS_PROMPT = """You are Aditya, Creator Manager at an influencer marketing agency in India.
A brand has just confirmed a campaign. Write 2 short, practical pointers that help the creators
start planning their content before the full brief arrives.

Rules:
- Base them ONLY on the campaign details given. Never invent products, offers, discounts, prices,
  store details or claims about the brand.
- Each pointer under 25 words, specific to this campaign (audience, location, content angle).
- Do not mention ad disclosure, payment or deadlines. Those are covered separately.

Return ONLY a JSON object, no other text:
{"pointers": ["...", "..."]}
"""

AD_DISCLOSURE_POINTER = ("Mark the post clearly as a paid partnership: use the platform's paid-partnership "
                         "label, or #ad / #collab in the caption, as required by ASCI guidelines.")


async def write_content_pointers(campaign: dict) -> list:
    """2 campaign-specific content pointers. Returns [] if anything goes wrong, so the email still sends."""
    def v(x):
        return x if x not in (None, "") else "not specified"
    try:
        parsed = parse_json_object(await call_claude(CREATOR_POINTERS_PROMPT, f"""CAMPAIGN
Brand: {campaign.get('brand_name')} (industry: {v(campaign.get('industry'))})
About the brand: {v(campaign.get('brand_info'))}
Platform: {v(campaign.get('platform'))}
Niche: {v(campaign.get('niche'))}
Target audience: {v(campaign.get('target_audience'))}
Location: {v(campaign.get('location_requirement'))}
Deliverables: {v(campaign.get('deliverables'))}
Requirements: {v(campaign.get('requirements'))}

Return ONLY the JSON object."""))
        pointers = [str(p).strip() for p in (parsed.get("pointers") or []) if str(p).strip()]
        return pointers[:2]
    except Exception as e:
        print(f"Content pointers failed for {campaign.get('brand_name')}: {e}")
        return []


class PaymentReceivedRequest(BaseModel):
    campaign_id: str


@app.post("/lineup/payment-received")
async def payment_received(req: PaymentReceivedRequest):
    """Advance received: reveal creators to the brand and tell each creator the campaign is on"""
    try:
        s = load_settings()

        # Content pointers are written first, so no database transaction waits on the AI call
        with engine.connect() as conn:
            pre = conn.execute(text("""
                SELECT c.status, c.niche, c.target_audience, c.location_requirement, c.deliverables,
                       c.requirements, c.platform, b.brand_name, b.industry, b.basic_info AS brand_info
                FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.campaign_id = :cid
            """), {"cid": req.campaign_id}).mappings().first()
        pointers = await write_content_pointers(dict(pre)) if pre and pre["status"] == "confirmed" else []
        pointer_block = "\n".join(f"• {p}" for p in pointers + [AD_DISCLOSURE_POINTER])

        with engine.begin() as conn:
            c = conn.execute(text("""
                SELECT c.*, b.brand_name FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.campaign_id = :cid
            """), {"cid": req.campaign_id}).mappings().first()
            if not c:
                raise HTTPException(status_code=404, detail="Campaign not found")
            c = dict(c)
            if c["status"] != "confirmed":
                return {"status": "skipped", "campaign_id": req.campaign_id, "brand_email": None,
                        "creator_emails": [], "slack_text": f"ℹ️ `{req.campaign_id}` is `{c['status']}`, so nothing was changed."}

            conn.execute(text("""
                UPDATE campaigns SET status = 'in_production', advance_received_at = NOW()
                WHERE campaign_id = :cid
            """), {"cid": req.campaign_id})

            creators = [dict(r) for r in conn.execute(text("""
                SELECT o.offer_id, o.creator_id, o.deliverables, o.deadline, o.gmail_thread_id,
                       o.gmail_message_id AS offer_msg,
                       COALESCE(o.final_amount, o.offered_amount) AS payout,
                       cr.creator_name, cr.handle, cr.platform
                FROM offers o
                JOIN creators cr ON cr.creator_id = o.creator_id
                LEFT JOIN matches m ON m.id = o.match_id
                WHERE o.campaign_id = :cid AND o.status = 'accepted'
                ORDER BY m.rank NULLS LAST, o.offer_id
            """), {"cid": req.campaign_id}).mappings().all()]

            brand_reply_to = conn.execute(text("""
                SELECT gmail_message_id FROM conversations
                WHERE gmail_thread_id = :t AND direction = 'inbound' AND contact_type = 'brand'
                ORDER BY created_at DESC LIMIT 1
            """), {"t": c["gmail_thread_id"]}).scalar()

            creator_emails = []
            brand = c["brand_name"]
            deal_value = c["final_brand_price"] or 0
            script_step = deal_value >= float(s.get("small_deal_threshold_inr", 25000))
            if c["deadline"]:
                campaign_deadline = c["deadline"]
            elif c["timeline_days"]:
                campaign_deadline = (c["created_at"] + timedelta(days=c["timeline_days"])).date()
            else:
                campaign_deadline = None

            for cr in creators:
                # Find something in this negotiation to reply to, from most to least specific:
                # 1. the creator's latest reply in the offer thread   2. this round's offer email
                # 3. any earlier round's offer email                  4. the creator's latest message anywhere
                reply_to = conn.execute(text("""
                    SELECT gmail_message_id FROM conversations
                    WHERE gmail_thread_id = :t AND direction = 'inbound' AND gmail_message_id IS NOT NULL
                    ORDER BY created_at DESC LIMIT 1
                """), {"t": cr["gmail_thread_id"]}).scalar() if cr["gmail_thread_id"] else None
                reply_to = reply_to or cr["offer_msg"]
                if not reply_to:
                    reply_to = conn.execute(text("""
                        SELECT gmail_message_id FROM offers
                        WHERE campaign_id = :cid AND creator_id = :crid AND gmail_message_id IS NOT NULL
                        ORDER BY round DESC LIMIT 1
                    """), {"cid": req.campaign_id, "crid": cr["creator_id"]}).scalar()
                if not reply_to:
                    reply_to = conn.execute(text("""
                        SELECT gmail_message_id FROM conversations
                        WHERE contact_type = 'creator' AND contact_id = :crid
                          AND direction = 'inbound' AND gmail_message_id IS NOT NULL
                        ORDER BY created_at DESC LIMIT 1
                    """), {"crid": cr["creator_id"]}).scalar()
                deadline = cr["deadline"] or campaign_deadline
                steps = "Next, we'll send you the full brief."
                if script_step:
                    steps += " Before shooting, you'll share a short script or concept so the brand can approve the direction."
                creator_emails.append({
                    "offer_id": cr["offer_id"],
                    "creator_name": cr["creator_name"],
                    "reply_to_message_id": reply_to,
                    "body": f"""Hi {_first_name(cr['creator_name'])},

Great news: {brand} has confirmed the campaign, so you're officially on! 🎉

Quick recap:
• Deliverables: {cr['deliverables'] or 'as per the brief'}
• Content deadline: {deadline.strftime('%d %b %Y') if deadline else 'to be confirmed with the brief'}
• Your payout: {format_inr(cr['payout'])}
• Payment: {creator_payment_terms(cr['payout'], s)}

A few pointers as you start planning:
{pointer_block}

{steps}

Aditya
Creator Manager""",
                })

        # Name reveal for the brand
        lines = []
        for i, cr in enumerate(creators):
            link = _profile_link(cr["platform"], cr["handle"])
            platform = PLATFORM_NAMES.get(str(cr["platform"] or "").lower(), cr["platform"])
            lines.append(f"Creator {chr(65 + i)}: {cr['creator_name']} ({cr['handle']}) · {platform}"
                         + (f"\n{link}" if link else "")
                         + f"\nDeliverable: {cr['deliverables'] or 'as per the brief'}")
        next_steps = ["We're sending each creator the full brief next."]
        if script_step:
            next_steps.append("Each creator will share a script or concept for your approval before shooting.")
        rev = c["revisions_allowed"]
        next_steps.append("You approve the final content before it goes live"
                          + (f" (up to {rev} round{'s' if rev and rev > 1 else ''} of revisions)." if rev else "."))
        numbered = "\n".join(f"{i}. {x}" for i, x in enumerate(next_steps, start=1))

        brand_email = f"""Hi {brand} team,

Payment received, thank you! Here's your confirmed creator lineup:

""" + "\n\n".join(lines) + f"""

What happens next:
{numbered}

Ananya
Brand Partnerships"""

        return {
            "status": "success",
            "campaign_id": req.campaign_id,
            "brand_reply_to_message_id": brand_reply_to,
            "brand_email": brand_email,
            "creator_emails": creator_emails,
            "slack_text": (f"✅ Advance received for *{brand}*. Creator names sent to the brand, and "
                           f"{len(creator_emails)} creator(s) told they're confirmed. Campaign is now in production."),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording payment: {str(e)}")


# ============ REJECTIONS, DECLINES, DEADLINES (STAGE 6, PHASE C) ============
def _creator_reply_to(conn, campaign_id, creator_id, thread_id, offer_msg):
    """Something in this creator's negotiation to reply to, from most to least specific"""
    reply_to = conn.execute(text("""
        SELECT gmail_message_id FROM conversations
        WHERE gmail_thread_id = :t AND direction = 'inbound' AND gmail_message_id IS NOT NULL
        ORDER BY created_at DESC LIMIT 1
    """), {"t": thread_id}).scalar() if thread_id else None
    reply_to = reply_to or offer_msg
    if not reply_to:
        reply_to = conn.execute(text("""
            SELECT gmail_message_id FROM offers
            WHERE campaign_id = :cid AND creator_id = :crid AND gmail_message_id IS NOT NULL
            ORDER BY round DESC LIMIT 1
        """), {"cid": campaign_id, "crid": creator_id}).scalar()
    if not reply_to:
        reply_to = conn.execute(text("""
            SELECT gmail_message_id FROM conversations
            WHERE contact_type = 'creator' AND contact_id = :crid AND direction = 'inbound'
              AND gmail_message_id IS NOT NULL
            ORDER BY created_at DESC LIMIT 1
        """), {"crid": creator_id}).scalar()
    return reply_to


def _brand_reply_to(conn, thread_id, fallback=None):
    return conn.execute(text("""
        SELECT gmail_message_id FROM conversations
        WHERE gmail_thread_id = :t AND direction = 'inbound' AND contact_type = 'brand'
        ORDER BY created_at DESC LIMIT 1
    """), {"t": thread_id}).scalar() or fallback


def build_release_email(first_name, brand_name) -> str:
    return f"""Hi {first_name},

An update on the {brand_name} campaign: the brand has decided to go in a different direction with the lineup, so we won't be moving ahead with this collaboration.

This isn't a reflection on your content, and there's nothing you need to do. We'll reach out again for a campaign that suits you.

Aditya
Creator Manager"""


def _release_offers(conn, campaign_id, offer_ids, brand_name, reason):
    """Withdraw accepted offers and prepare a polite release email for each creator"""
    if not offer_ids:
        return []
    rows = [dict(r) for r in conn.execute(text("""
        UPDATE offers o SET status = 'withdrawn', decline_reason = :reason,
               brand_rejected_at = NULL, release_after = NULL
        FROM creators cr
        WHERE o.offer_id = ANY(:ids) AND cr.creator_id = o.creator_id
        RETURNING o.offer_id, o.creator_id, o.gmail_thread_id, o.gmail_message_id, cr.creator_name
    """), {"ids": offer_ids, "reason": reason}).mappings().all()]
    return [{
        "offer_id": r["offer_id"],
        "creator_name": r["creator_name"],
        "reply_to_message_id": _creator_reply_to(conn, campaign_id, r["creator_id"],
                                                 r["gmail_thread_id"], r["gmail_message_id"]),
        "body": build_release_email(_first_name(r["creator_name"]), brand_name),
    } for r in rows]


class ResolveRejectionRequest(BaseModel):
    campaign_id: str
    decision: str                        # send draft / edit and send / replace / proceed with fewer
    edited_email: Optional[str] = None
    draft: Optional[str] = None


@app.post("/lineup/resolve-rejection")
async def resolve_rejection(req: ResolveRejectionRequest):
    """Apply Deven's decision after a brand rejected creators from the lineup"""
    try:
        d = (req.decision or "").strip().lower()
        if "edit" in d:
            decision = "edit"
        elif "send" in d:
            decision = "send"
        elif "replace" in d:
            decision = "replace"
        elif "fewer" in d or "proceed" in d:
            decision = "fewer"
        else:
            raise HTTPException(status_code=400, detail="decision must be send draft, edit and send, replace, or proceed with fewer")
        if decision == "edit" and not (req.edited_email or "").strip():
            raise HTTPException(status_code=400, detail="Edit and send needs the edited email text")

        with engine.begin() as conn:
            c = conn.execute(text("""
                SELECT c.*, b.brand_name FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.campaign_id = :cid
            """), {"cid": req.campaign_id}).mappings().first()
            if not c:
                raise HTTPException(status_code=404, detail="Campaign not found")
            c = dict(c)
            brand = c["brand_name"]
            held = [r[0] for r in conn.execute(text("""
                SELECT offer_id FROM offers
                WHERE campaign_id = :cid AND status = 'accepted' AND brand_rejected_at IS NOT NULL
            """), {"cid": req.campaign_id}).fetchall()]

            result = {"status": "success", "campaign_id": req.campaign_id, "decision": decision,
                      "reply_email": None, "brand_reply_to_message_id": _brand_reply_to(conn, c["gmail_thread_id"],
                                                                                        c["lineup_message_id"]),
                      "creator_release_emails": [], "start_next_offers": False, "start_send_lineup": False,
                      "slack_text": None}

            if not held:
                result["status"] = "skipped"
                result["slack_text"] = f"ℹ️ No held creators for *{brand}*, so nothing was changed."
                return result

            labels_of = {v: k for k, v in (c.get("lineup_labels") or {}).items()}
            held_labels = ", ".join(labels_of.get(o, "a creator") for o in held)

            if decision in ("send", "edit"):
                body = req.edited_email.strip() if decision == "edit" else (req.draft or "").strip()
                if not body:
                    raise HTTPException(status_code=400, detail="No draft text to send")
                result["reply_email"] = body
                result["slack_text"] = (f"📨 Reply sent to *{brand}* about {held_labels}. "
                                        f"The creator(s) stay held until they answer or the hold runs out.")

            elif decision == "replace":
                result["creator_release_emails"] = _release_offers(conn, req.campaign_id, held, brand, "rejected by brand")
                conn.execute(text("UPDATE campaigns SET status = 'offers_sent' WHERE campaign_id = :cid"),
                             {"cid": req.campaign_id})
                result["reply_email"] = (f"Hi {brand} team,\n\nThanks for the feedback. We're lining up a replacement "
                                         f"for {held_labels} and will send you an updated lineup shortly.\n\n"
                                         f"Ananya\nBrand Partnerships")
                result["start_next_offers"] = True
                result["slack_text"] = (f"🔄 Replacing {held_labels} for *{brand}*. The creator(s) were released, "
                                        f"and the next approved creator gets an offer.")

            else:  # proceed with fewer creators
                result["creator_release_emails"] = _release_offers(conn, req.campaign_id, held, brand, "rejected by brand")
                result["start_send_lineup"] = True
                note = (" ⚠️ This is a campaign-pool budget, so the price stays the same with fewer creators. "
                        "Check it in the preview before sending." if c["pricing_model"] != "per_collab" else "")
                result["slack_text"] = (f"➖ Proceeding without {held_labels} for *{brand}*. "
                                        f"An updated lineup is coming for your approval.{note}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resolving rejection: {str(e)}")


class CloseCampaignRequest(BaseModel):
    campaign_id: str
    reason: Optional[str] = None


@app.post("/lineup/close")
async def close_campaign(req: CloseCampaignRequest):
    """Brand declined: mark the campaign lost, release every accepted creator, thank the brand"""
    try:
        with engine.begin() as conn:
            c = conn.execute(text("""
                SELECT c.*, b.brand_name FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.campaign_id = :cid
            """), {"cid": req.campaign_id}).mappings().first()
            if not c:
                raise HTTPException(status_code=404, detail="Campaign not found")
            c = dict(c)
            brand = c["brand_name"]
            if c["status"] in ("lost", "cancelled", "completed", "in_production"):
                return {"status": "skipped", "campaign_id": req.campaign_id, "reply_email": None,
                        "brand_reply_to_message_id": None, "creator_release_emails": [],
                        "slack_text": f"ℹ️ `{req.campaign_id}` is `{c['status']}`, so nothing was changed."}

            conn.execute(text("""
                UPDATE campaigns SET status = 'lost', lost_reason = :reason WHERE campaign_id = :cid
            """), {"cid": req.campaign_id, "reason": req.reason or "brand declined the lineup"})
            active = [r[0] for r in conn.execute(text("""
                SELECT offer_id FROM offers WHERE campaign_id = :cid AND status IN ('accepted', 'sent', 'countered')
            """), {"cid": req.campaign_id}).fetchall()]
            releases = _release_offers(conn, req.campaign_id, active, brand, "campaign closed")

            return {
                "status": "success",
                "campaign_id": req.campaign_id,
                "brand_reply_to_message_id": _brand_reply_to(conn, c["gmail_thread_id"], c["lineup_message_id"]),
                "reply_email": (f"Hi {brand} team,\n\nThanks for letting us know, and for considering the lineup. "
                                f"If your plans change or you have another campaign coming up, just reply here "
                                f"and we'll put together a fresh set of creators.\n\nAnanya\nBrand Partnerships"),
                "creator_release_emails": releases,
                "slack_text": f"🛑 *{brand}* campaign closed. {len(releases)} creator(s) released politely.",
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error closing campaign: {str(e)}")


@app.post("/lineup/check-deadlines")
async def check_lineup_deadlines():
    """
    Hourly: remind brands, alert Deven at the deadline, and release creators whose hold ran out.
    Nothing is ever cancelled automatically.
    """
    try:
        s = load_settings()
        short_days = int(s.get("short_timeline_days", 7))
        remind_h = int(s.get("lineup_reminder_hours", 48))
        remind_h_short = int(s.get("lineup_reminder_hours_short", 12))
        reminders, alerts, releases, refill = [], [], [], []

        with engine.begin() as conn:
            open_lineups = [dict(r) for r in conn.execute(text("""
                SELECT c.*, b.brand_name FROM campaigns c JOIN brands b ON b.brand_id = c.brand_id
                WHERE c.status = 'shortlist_sent' AND c.lineup_sent_at IS NOT NULL
            """)).mappings().all()]
            now = datetime.utcnow()

            for c in open_lineups:
                short = c["timeline_days"] is not None and c["timeline_days"] <= short_days
                hours = remind_h_short if short else remind_h

                # 1. Friendly reminder to the brand
                if not c["lineup_reminder_sent_at"] and c["lineup_sent_at"] + timedelta(hours=hours) <= now:
                    expires = (c["lineup_expires_at"].replace(tzinfo=timezone.utc).astimezone(IST).strftime('%d %b, %I:%M %p')
                               if c["lineup_expires_at"] else "soon")
                    reminders.append({
                        "campaign_id": c["campaign_id"],
                        "brand_name": c["brand_name"],
                        "reply_to_message_id": _brand_reply_to(conn, c["gmail_thread_id"], c["lineup_message_id"]),
                        "body": f"""Hi {c['brand_name']} team,

A quick reminder: we're holding your creator lineup until {expires} IST. The creators have set aside time for this campaign.

Just reply "Confirm" to lock it in, or let us know if you'd like any changes.

Ananya
Brand Partnerships""",
                    })
                    conn.execute(text("UPDATE campaigns SET lineup_reminder_sent_at = NOW() WHERE campaign_id = :cid"),
                                 {"cid": c["campaign_id"]})

                # 2. Deadline passed: alert Deven once, never cancel automatically
                if not c["lineup_deadline_alerted_at"] and c["lineup_expires_at"] and c["lineup_expires_at"] <= now:
                    alerts.append(f"⏰ *{c['brand_name']}* hasn't confirmed the lineup (`{c['campaign_id']}`), "
                                  f"and the window has passed. Chase them personally, or close the campaign.")
                    conn.execute(text("UPDATE campaigns SET lineup_deadline_alerted_at = NOW() WHERE campaign_id = :cid"),
                                 {"cid": c["campaign_id"]})

            # 3. Held creators whose 48 hours ran out while the brand hadn't come round
            expired_holds = [dict(r) for r in conn.execute(text("""
                SELECT o.offer_id, o.campaign_id, b.brand_name
                FROM offers o
                JOIN campaigns c ON c.campaign_id = o.campaign_id
                JOIN brands b ON b.brand_id = c.brand_id
                WHERE o.status = 'accepted' AND o.release_after IS NOT NULL AND o.release_after <= NOW()
                  AND c.status = 'shortlist_sent'
            """)).mappings().all()]
            by_campaign = {}
            for h in expired_holds:
                by_campaign.setdefault((h["campaign_id"], h["brand_name"]), []).append(h["offer_id"])
            for (cid, brand), ids in by_campaign.items():
                releases += _release_offers(conn, cid, ids, brand, "brand hold expired")
                conn.execute(text("UPDATE campaigns SET status = 'offers_sent' WHERE campaign_id = :cid"), {"cid": cid})
                refill.append({"campaign_id": cid})
                alerts.append(f"⌛ Hold ran out for {len(ids)} creator(s) on *{brand}* (`{cid}`). "
                              f"They were released, and the next approved creator gets an offer.")

        return {
            "status": "success",
            "reminders": reminders,
            "creator_release_emails": releases,
            "refill_campaigns": refill,
            "slack_text": "\n".join(alerts) if alerts else None,
            "counts": {"reminders": len(reminders), "releases": len(releases), "alerts": len(alerts)},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking lineup deadlines: {str(e)}")


# ============ BRANDS CRUD ============
@app.post("/brands")
async def create_brand(brand: dict):
    """Create a new brand"""
    db = SessionLocal()
    try:
        db_brand = Brand(**brand, id=str(uuid.uuid4()))
        db.add(db_brand)
        db.commit()
        db.refresh(db_brand)
        return db_brand
    finally:
        db.close()


@app.get("/brands")
async def get_brands():
    """Get all brands"""
    db = SessionLocal()
    try:
        return db.query(Brand).all()
    finally:
        db.close()


@app.get("/brands/{brand_id}")
async def get_brand(brand_id: str):
    """Get specific brand"""
    db = SessionLocal()
    try:
        brand = db.query(Brand).filter(Brand.brand_id == brand_id).first()
        if not brand:
            raise HTTPException(status_code=404, detail="Brand not found")
        return brand
    finally:
        db.close()


# ============ CREATORS CRUD ============
@app.post("/creators")
async def create_creator(creator: dict):
    """Create a new creator"""
    db = SessionLocal()
    try:
        db_creator = Creator(**creator, id=str(uuid.uuid4()))
        db.add(db_creator)
        db.commit()
        db.refresh(db_creator)
        return db_creator
    finally:
        db.close()


@app.get("/creators")
async def get_creators():
    """Get all creators"""
    db = SessionLocal()
    try:
        return db.query(Creator).all()
    finally:
        db.close()


# ============ ANALYTICS ============
@app.get("/analytics/revenue")
async def get_revenue_analytics():
    """Get revenue analytics"""
    return {"status": "analytics_endpoint", "message": "Analytics implementation pending"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
