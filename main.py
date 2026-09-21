import os
import uuid
import json
import re
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from anthropic import Anthropic

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="AI Marketplace Agents",
    description="Steve (Brand Manager), Fred (Matcher), Aditya (Creator Manager)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
engine = create_engine(DATABASE_URL, echo=DEBUG)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
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

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
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

# Initialize Anthropic client
client = Anthropic()

# ============ HEALTH CHECK ============
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if DATABASE_URL:
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
            return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}
    return {"status": "healthy"}

# ============ STEVE: BRAND MANAGER AGENT ============

@app.post("/agent/steve/generate-pitches-batch")
async def generate_pitches_batch(request: BatchPitchRequest):
    """
    STEVE: Batch generate pitch emails for multiple brands
    Input: Array of brands with basic info
    Output: Array of pitch emails ready to send
    Model: Claude Opus-5
    """
    try:
        if request.action != "send_pitch_emails_batch":
            raise HTTPException(status_code=400, detail="Invalid action")

        # Prepare brands data for Claude
        brands_text = "\n\n".join([
            f"""Brand #{i+1}:
- ID: {brand.brand_id}
- Name: {brand.brand_name}
- Industry: {brand.industry}
- Email: {brand.email}
- Basic Info: {brand.basic_info}"""
            for i, brand in enumerate(request.brands)
        ])

        # System prompt for batch pitch generation
        system_prompt = """You are Steve, the Brand Manager Agent for an AI-powered influencer marketing agency.

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

IMPORTANT: Return ONLY valid JSON array. No preamble, no explanation.

Format:
[
  {
    "brand_id": "NIKE-001",
    "brand_name": "Nike India",
    "pitch_email": "Subject: Creator Partnership Opportunity - Nike India\\n\\nDear Nike Team,..."
  },
  {
    "brand_id": "ADIDAS-001",
    "brand_name": "Adidas India",
    "pitch_email": "Subject: Creator Partnership Opportunity - Adidas India\\n\\nDear Adidas Team,..."
  }
]
"""

        user_message = f"""Generate pitch emails for these brands:

{brands_text}

For each brand, create a personalized pitch email. Return ONLY the JSON array, no other text."""

        # Call Claude API
        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=4000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        # Extract and parse response
        if not response.content or not response.content[0].text:
            raise Exception("Empty response from Claude")
        
        response_text = response.content[0].text.strip()
        
        # Clean response (remove markdown code blocks if present)
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            pitches = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to fix common escaping issues
            response_text = response_text.replace('\\"', '__ESCAPED_QUOTE__')
            response_text = re.sub(r'"pitch_email":\s*"([^"]*)"', lambda m: f'"pitch_email": "{m.group(1).replace(chr(34), chr(92) + chr(34))}"', response_text)
            response_text = response_text.replace('__ESCAPED_QUOTE__', '\\"')
            
            try:
                pitches = json.loads(response_text)
            except json.JSONDecodeError:
                raise json.JSONDecodeError("Could not parse Claude response", response_text, 0)

        # Format response with metadata
        db = SessionLocal()
        results = []
        email_sent_date = datetime.utcnow().isoformat()

        for pitch in pitches:
            result = {
                "brand_id": pitch.get("brand_id"),
                "brand_name": pitch.get("brand_name"),
                "email": next((b.email for b in request.brands if b.brand_id == pitch.get("brand_id")), ""),
                "pitch_email": pitch.get("pitch_email"),
                "email_sent_date": email_sent_date,
                "status": "pitch_generated"
            }
            results.append(BrandResponse(**result))

        db.close()

        return {
            "status": "success",
            "action": "send_pitch_emails_batch",
            "total_brands": len(request.brands),
            "pitches": [r.dict() for r in results]
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Claude response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating brand pitches: {str(e)}")


# ============ ADITYA: CREATOR MANAGER AGENT ============

@app.post("/agent/aditya/generate-pitches-batch")
async def aditya_generate_creator_pitches(request: BatchCreatorPitchRequest):
    """
    ADITYA: Batch generate pitch emails for multiple creators
    Input: Array of creators with basic info
    Output: Array of pitch emails ready to send (explaining brand collaboration opportunity)
    
    Uses batch processing: splits large batches into smaller chunks (max 6 per batch)
    Model: Claude Sonnet-5
    """
    try:
        if request.action != "send_creator_pitches_batch":
            raise HTTPException(status_code=400, detail="Invalid action")

        # System prompt for batch pitch generation
        system_prompt = """You are Aditya, the Creator Manager Agent for an AI-powered influencer marketing agency.

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

CRITICAL REQUIREMENT: You MUST return exactly as many pitch emails as creators provided. Count the creators and ensure every single one is included in the JSON array. Do NOT skip anyone.

IMPORTANT: Return ONLY valid JSON array. No preamble, no explanation.

Format:
[
  {
    "creator_id": "CREATOR-001",
    "creator_name": "Ali Khan",
    "pitch_email": "Subject: Brand Collaboration Opportunity for @alikhan\\n\\nHi Ali,..."
  },
  {
    "creator_id": "CREATOR-002",
    "creator_name": "Priya Singh",
    "pitch_email": "Subject: Creator Partnership Opportunity - @priyasingh\\n\\nHi Priya,..."
  }
]
"""

        # Split creators into batches of 6 (max) to avoid Claude truncation
        batch_size = 6
        all_pitches = []
        
        for batch_start in range(0, len(request.creators), batch_size):
            batch_end = min(batch_start + batch_size, len(request.creators))
            batch_creators = request.creators[batch_start:batch_end]
            
            # Prepare creators data for this batch
            creators_text = "\n\n".join([
                f"""Creator #{i+1}:
- ID: {creator.creator_id}
- Name: {creator.creator_name}
- Platform: {creator.platform}
- Handle: {creator.handle}
- Email: {creator.email}
- Followers: {creator.follower_count}
- Engagement Rate: {creator.engagement_rate}%
- Basic Info: {creator.basic_info}"""
                for i, creator in enumerate(batch_creators)
            ])

            user_message = f"""Generate pitch emails for ALL {len(batch_creators)} creators provided below. You must return exactly {len(batch_creators)} entries in the JSON array - one for each creator.

{creators_text}

MANDATORY: Do not skip anyone. Generate a pitch for every single creator listed. Return ONLY the JSON array with ALL {len(batch_creators)} pitches, no other text."""

            # Call Claude API for this batch with Sonnet-5
            response = client.messages.create(
                model="claude-sonnet-5",
                max_tokens=4000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            # Extract and parse response
            if not response.content or not response.content[0].text:
                raise Exception(f"Empty response from Claude for batch {batch_start}-{batch_end}")
            
            response_text = response.content[0].text.strip()
            
            # Clean response (remove markdown code blocks if present)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Repair common JSON issues in Claude's response
            try:
                batch_pitches = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to fix common escaping issues
                response_text = response_text.replace('\\"', '__ESCAPED_QUOTE__')
                response_text = re.sub(r'"pitch_email":\s*"([^"]*)"', lambda m: f'"pitch_email": "{m.group(1).replace(chr(34), chr(92) + chr(34))}"', response_text)
                response_text = response_text.replace('__ESCAPED_QUOTE__', '\\"')
                
                try:
                    batch_pitches = json.loads(response_text)
                except json.JSONDecodeError:
                    # Last resort: try to extract and repair individual pitch objects
                    pitch_matches = re.findall(r'\{[^}]*?"creator_id"[^}]*?"pitch_email"[^}]*?\}', response_text, re.DOTALL)
                    batch_pitches = []
                    for match in pitch_matches:
                        try:
                            batch_pitches.append(json.loads(match))
                        except:
                            pass
                    
                    if not batch_pitches:
                        raise json.JSONDecodeError("Could not repair or parse Claude response", response_text, 0)
            
            all_pitches.extend(batch_pitches)

        # Format response with metadata
        db = SessionLocal()
        results = []
        email_sent_date = datetime.utcnow().isoformat()

        for pitch in all_pitches:
            result = {
                "creator_id": pitch.get("creator_id"),
                "creator_name": pitch.get("creator_name"),
                "email": next((c.email for c in request.creators if c.creator_id == pitch.get("creator_id")), ""),
                "pitch_email": pitch.get("pitch_email"),
                "email_sent_date": email_sent_date,
                "status": "pitch_generated"
            }
            results.append(CreatorResponse(**result))

        db.close()

        return {
            "status": "success",
            "action": "send_creator_pitches_batch",
            "total_creators": len(request.creators),
            "pitches": [r.dict() for r in results]
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Claude response: {str(e)}")
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
        brands = db.query(Brand).all()
        return brands
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
        creators = db.query(Creator).all()
        return creators
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
        matches = db.query(Match).all()
        return matches
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
