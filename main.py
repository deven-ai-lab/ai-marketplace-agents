import os
import re
import uuid
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Callable
from fastapi import FastAPI, HTTPException
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
