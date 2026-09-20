import os
import uuid
import json
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, text
from pydantic import BaseModel
import anthropic

from config import engine, SessionLocal, get_db, HOST, PORT, DEBUG
from models import Base, Brand, Creator, Campaign

app = FastAPI(title="AI-to-AI Marketplace API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== PYDANTIC SCHEMAS ==========

class BrandCreate(BaseModel):
    name: str
    industry: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[float] = None
    target_audience_age: Optional[str] = None
    target_audience_interests: Optional[str] = None
    target_audience_gender: Optional[str] = None
    platforms_needed: Optional[List[str]] = None
    content_format_wanted: Optional[List[str]] = None
    content_type: Optional[List[str]] = None
    location_geography_focus: Optional[str] = None
    timeline: Optional[str] = None
    brand_brief: Optional[str] = None
    past_campaigns_links: Optional[List[str]] = None
    deven_notes: Optional[str] = None

class BrandResponse(BrandCreate):
    id: uuid.UUID
    status: str
    stage: str
    matched_creators: Optional[List[uuid.UUID]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CreatorCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    instagram_handle: Optional[str] = None
    youtube_channel: Optional[str] = None
    tiktok_handle: Optional[str] = None
    other_platforms: Optional[dict] = None
    follower_count_instagram: Optional[float] = None
    follower_count_youtube: Optional[float] = None
    follower_count_tiktok: Optional[float] = None
    engagement_rate: Optional[float] = None
    niche: Optional[str] = None
    content_specialties: Optional[List[str]] = None
    audience_age: Optional[str] = None
    audience_gender: Optional[str] = None
    location: Optional[str] = None
    rates_per_post: Optional[float] = None
    rates_per_story: Optional[float] = None
    rates_per_reel: Optional[float] = None
    rates_longform_video: Optional[float] = None
    barter_willing: Optional[bool] = False
    past_brand_collaborations: Optional[List[str]] = None
    bio: Optional[str] = None
    profile_picture_url: Optional[str] = None
    deven_notes: Optional[str] = None

class CreatorResponse(CreatorCreate):
    id: uuid.UUID
    availability_status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CampaignCreate(BaseModel):
    brand_id: uuid.UUID
    creator_id: uuid.UUID
    campaign_name: Optional[str] = None
    collab_type: str
    campaign_value: Optional[float] = None
    commission_percentage: Optional[float] = None
    flat_fee_amount: Optional[float] = None
    deliverables: Optional[List[str]] = None
    content_requirements: Optional[str] = None
    timeline_start: Optional[str] = None
    timeline_end: Optional[str] = None
    notes: Optional[str] = None

class CampaignResponse(CampaignCreate):
    id: uuid.UUID
    commission_amount: Optional[float] = None
    status: str
    payment_status: str
    amount_paid: float
    content_submission_status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class SteveRequest(BaseModel):
    brand_id: str
    brand_name: str
    budget: int
    timeline_days: int
    niche: str
    requirements: str
    target_audience: Optional[str] = None

# ========== HEALTH CHECK ==========

@app.get("/health")
async def health_check():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "degraded", "database": "disconnected", "error": str(e)}

# ========== STEVE - BRAND MANAGER AGENT ==========

@app.post("/agent/steve")
async def steve_agent(request: SteveRequest):
    """Steve - Brand Manager Agent"""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {"status": "error", "detail": "ANTHROPIC_API_KEY not set"}
        
        client = anthropic.Anthropic(api_key=api_key)
        
        system_prompt = """You are Steve, Brand Manager Agent. Analyze brand briefs and return ONLY a JSON object with these fields:
{
  "feasibility_score": (1-10),
  "strategy": "brief strategy description",
  "recommended_creators_count": (number),
  "email_template": "draft email to send",
  "next_steps": ["step1", "step2"],
  "flags_for_deven": ["flag1", "flag2"]
}

No other text, just JSON."""

        user_message = f"""Analyze this brand brief:

Brand: {request.brand_name}
Budget: ₹{request.budget:,}
Timeline: {request.timeline_days} days
Niche: {request.niche}
Requirements: {request.requirements}
Target Audience: {request.target_audience or 'Not specified'}

Return ONLY JSON object."""

        message = client.messages.create(
            model="claude-opus-5",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        response_text = message.content[0].text
        
        try:
            strategy = json.loads(response_text)
        except json.JSONDecodeError:
            strategy = {"raw_response": response_text}
        
        return {
            "status": "success",
            "brand_id": request.brand_id,
            "strategy": strategy
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ========== BRANDS ENDPOINTS ==========

@app.post("/brands", response_model=BrandResponse)
async def create_brand(brand: BrandCreate, db: Session = Depends(get_db)):
    db_brand = Brand(**brand.dict())
    db.add(db_brand)
    db.commit()
    db.refresh(db_brand)
    return db_brand

@app.get("/brands", response_model=List[BrandResponse])
async def get_brands(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Brand)
    if status:
        query = query.filter(Brand.status == status)
    return query.offset(skip).limit(limit).all()

@app.get("/brands/{brand_id}", response_model=BrandResponse)
async def get_brand(brand_id: uuid.UUID, db: Session = Depends(get_db)):
    db_brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not db_brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    return db_brand

@app.put("/brands/{brand_id}", response_model=BrandResponse)
async def update_brand(brand_id: uuid.UUID, brand: BrandCreate, db: Session = Depends(get_db)):
    db_brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not db_brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    
    for key, value in brand.dict(exclude_unset=True).items():
        setattr(db_brand, key, value)
    db_brand.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_brand)
    return db_brand

@app.delete("/brands/{brand_id}")
async def delete_brand(brand_id: uuid.UUID, db: Session = Depends(get_db)):
    db_brand = db.query(Brand).filter(Brand.id == brand_id).first()
    if not db_brand:
        raise HTTPException(status_code=404, detail="Brand not found")
    
    db.delete(db_brand)
    db.commit()
    return {"message": "Brand deleted successfully"}

# ========== CREATORS ENDPOINTS ==========

@app.post("/creators", response_model=CreatorResponse)
async def create_creator(creator: CreatorCreate, db: Session = Depends(get_db)):
    db_creator = Creator(**creator.dict())
    db.add(db_creator)
    db.commit()
    db.refresh(db_creator)
    return db_creator

@app.get("/creators", response_model=List[CreatorResponse])
async def get_creators(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    niche: Optional[str] = None,
    availability: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Creator)
    if niche:
        query = query.filter(Creator.niche == niche)
    if availability:
        query = query.filter(Creator.availability_status == availability)
    return query.offset(skip).limit(limit).all()

@app.get("/creators/{creator_id}", response_model=CreatorResponse)
async def get_creator(creator_id: uuid.UUID, db: Session = Depends(get_db)):
    db_creator = db.query(Creator).filter(Creator.id == creator_id).first()
    if not db_creator:
        raise HTTPException(status_code=404, detail="Creator not found")
    return db_creator

@app.put("/creators/{creator_id}", response_model=CreatorResponse)
async def update_creator(creator_id: uuid.UUID, creator: CreatorCreate, db: Session = Depends(get_db)):
    db_creator = db.query(Creator).filter(Creator.id == creator_id).first()
    if not db_creator:
        raise HTTPException(status_code=404, detail="Creator not found")
    
    for key, value in creator.dict(exclude_unset=True).items():
        setattr(db_creator, key, value)
    db_creator.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_creator)
    return db_creator

@app.delete("/creators/{creator_id}")
async def delete_creator(creator_id: uuid.UUID, db: Session = Depends(get_db)):
    db_creator = db.query(Creator).filter(Creator.id == creator_id).first()
    if not db_creator:
        raise HTTPException(status_code=404, detail="Creator not found")
    
    db.delete(db_creator)
    db.commit()
    return {"message": "Creator deleted successfully"}

# ========== CAMPAIGNS ENDPOINTS ==========

@app.post("/campaigns", response_model=CampaignResponse)
async def create_campaign(campaign: CampaignCreate, db: Session = Depends(get_db)):
    commission_amount = None
    if campaign.collab_type == "paid" and campaign.campaign_value and campaign.commission_percentage:
        commission_amount = campaign.campaign_value * (campaign.commission_percentage / 100)
    
    db_campaign = Campaign(**campaign.dict(), commission_amount=commission_amount)
    db.add(db_campaign)
    db.commit()
    db.refresh(db_campaign)
    return db_campaign

@app.get("/campaigns", response_model=List[CampaignResponse])
async def get_campaigns(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[str] = None,
    brand_id: Optional[str] = None,
    creator_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Campaign)
    if status:
        query = query.filter(Campaign.status == status)
    if brand_id:
        query = query.filter(Campaign.brand_id == uuid.UUID(brand_id))
    if creator_id:
        query = query.filter(Campaign.creator_id == uuid.UUID(creator_id))
    return query.offset(skip).limit(limit).all()

@app.get("/campaigns/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: uuid.UUID, db: Session = Depends(get_db)):
    db_campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not db_campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return db_campaign

@app.put("/campaigns/{campaign_id}", response_model=CampaignResponse)
async def update_campaign(campaign_id: uuid.UUID, campaign: CampaignCreate, db: Session = Depends(get_db)):
    db_campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not db_campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    for key, value in campaign.dict(exclude_unset=True).items():
        setattr(db_campaign, key, value)
    db_campaign.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_campaign)
    return db_campaign

@app.delete("/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: uuid.UUID, db: Session = Depends(get_db)):
    db_campaign = db.query(Campaign).filter(Campaign.id == campaign_id).first()
    if not db_campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    
    db.delete(db_campaign)
    db.commit()
    return {"message": "Campaign deleted successfully"}

# ========== ANALYTICS ==========

@app.get("/analytics/revenue")
async def revenue_analytics(db: Session = Depends(get_db)):
    paid_campaigns = db.query(Campaign).filter(
        and_(Campaign.collab_type == "paid", Campaign.status == "completed")
    ).all()
    
    barter_campaigns = db.query(Campaign).filter(
        and_(Campaign.collab_type == "barter", Campaign.status == "completed")
    ).all()
    
    total_commission = sum(float(c.commission_amount or 0) for c in paid_campaigns)
    total_flat_fee = sum(float(c.flat_fee_amount or 0) for c in barter_campaigns)
    
    return {
        "total_commission_revenue": total_commission,
        "total_flat_fee_revenue": total_flat_fee,
        "total_revenue": total_commission + total_flat_fee,
        "paid_campaigns_completed": len(paid_campaigns),
        "barter_campaigns_completed": len(barter_campaigns)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
