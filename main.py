import os
import re
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Callable
from types import SimpleNamespace
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


# ============ FRED: MATCHING AGENT ============
FRED_MODEL = "claude-sonnet-5"
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
                        creator_min_budget_at_match, fred_model, fred_version)
                    VALUES (:campaign_id, :creator_id, :rank, :match_score, CAST(:score_breakdown AS JSONB),
                        :reasoning, :concerns, :red_flags, :is_stretch, :suggested_payout,
                        :min_budget, :fred_model, :fred_version)
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

STEVE_FOLLOWUP_PROMPT = """You are Steve, Brand Manager at an influencer marketing agency in India.
These brands have not replied to your partnership pitch. Write a short follow-up email for each.
Sign off exactly as:
Steve
Brand Partnerships
""" + FOLLOWUP_RULES

ADITYA_FOLLOWUP_PROMPT = """You are Aditya, Creator Manager at an influencer marketing agency in India.
These creators have not replied to your collaboration pitch. Write a short follow-up email for each.
The easiest reply to ask for is their rate per collaboration and the kind of brands they like.
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
