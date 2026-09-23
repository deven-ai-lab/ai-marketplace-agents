import os
import re
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Callable
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
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
    description="Steve (Brand Manager), Fred (Matcher), Aditya (Creator Manager)",
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
engine = create_engine(DATABASE_URL, echo=DEBUG)
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


class Match(Base):
    __tablename__ = "matches"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    brand_id = Column(String)
    creator_id = Column(String)
    creator_name = Column(String)
    match_score = Column(Integer)
    niche_match = Column(String)
    audience_match = Column(String)
    budget_fit = Column(String)
    platform_match = Column(String)
    engagement_metric = Column(String)
    key_strengths = Column(Text)
    concerns = Column(Text)
    overall_reasoning = Column(Text)
    approval_status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ============ PYDANTIC MODELS ============
class PitchBrandInput(BaseModel):
    brand_id: str
    brand_name: str
    industry: str
    email: str
    phone: Optional[str] = None
    website: Optional[str] = None
    basic_info: str


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


class PitchCreatorInput(BaseModel):
    creator_id: str
    creator_name: str
    platform: str
    handle: str
    email: str
    phone: Optional[str] = None
    basic_info: str
    follower_count: Optional[int] = None
    engagement_rate: Optional[float] = None
    comments: Optional[str] = None


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


async def call_claude(system_prompt: str, user_message: str) -> str:
    """
    Call Claude and return only the text output.
    - Skips thinking blocks (adaptive thinking is on by default)
    - Fails loudly if output was cut off by max_tokens
    """
    response = await async_client.messages.create(
        model=EMAIL_MODEL,
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


def parse_pitches(response_text: str, id_key: str) -> list:
    """Parse Claude's JSON array, with repair fallbacks"""
    response_text = clean_json_text(response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Fallback 1: fix unescaped quotes inside pitch_email
    repaired = response_text.replace('\\"', '__ESCAPED_QUOTE__')
    repaired = re.sub(
        r'"pitch_email":\s*"([^"]*)"',
        lambda m: f'"pitch_email": "{m.group(1).replace(chr(34), chr(92) + chr(34))}"',
        repaired
    )
    repaired = repaired.replace('__ESCAPED_QUOTE__', '\\"')
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Fallback 2: extract individual objects
    pitches = []
    pattern = r'\{[^}]*?"' + id_key + r'"[^}]*?"pitch_email"[^}]*?\}'
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
    label: str
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
        user_message = f"""Generate pitch emails for ALL {len(batch)} {label} provided below. Return exactly {len(batch)} entries in the JSON array - one for each.

{items_text}

Do not skip anyone. Return ONLY the JSON array, no other text."""
        async with semaphore:
            try:
                return parse_pitches(await call_claude(system_prompt, user_message), id_key)
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
                if pid in valid_ids and pid not in pitch_map and pitch.get("pitch_email"):
                    pitch_map[pid] = pitch["pitch_email"]

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
STEVE_SYSTEM_PROMPT = """You are Steve, the Brand Manager Agent for an AI-powered influencer marketing agency.

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
5. Calls them to action

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

STEVE_PARSE_PROMPT = """You are Steve, Brand Manager Agent at an influencer marketing agency.
You sent a pitch email to a brand and they replied. Read the reply and extract structured data.
""" + REPLY_RULES + """
PRICING MODEL:
- "campaign_pool": the brand gives a TOTAL budget for the campaign (e.g. "2.5 lakh for the campaign")
- "per_collab": the brand gives a rate PER creator or PER collaboration (e.g. "20k per collab", "15k per reel")
- null: no budget mentioned
Put a total budget in total_budget. Put a per-creator rate in rate_per_collab, and the number of creators wanted in slots.

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
            "timeline_days": to_int_or_none(parsed.get("timeline_days")),
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


# ============ MATCHES CRUD ============
@app.post("/matches")
async def create_match(match: dict):
    """Create a new match"""
    db = SessionLocal()
    try:
        db_match = Match(**match, id=str(uuid.uuid4()))
        db.add(db_match)
        db.commit()
        db.refresh(db_match)
        return db_match
    finally:
        db.close()


@app.get("/matches")
async def get_matches():
    """Get all matches"""
    db = SessionLocal()
    try:
        return db.query(Match).all()
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
