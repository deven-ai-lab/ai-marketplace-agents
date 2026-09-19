from sqlalchemy import Column, String, Numeric, DateTime, Boolean, ARRAY, JSON, ForeignKey, Text, Date
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()


class Brand(Base):
    __tablename__ = "brands"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    industry = Column(String)
    email = Column(String)
    phone = Column(String)
    location = Column(String)
    budget = Column(Numeric)
    target_audience_age = Column(String)
    target_audience_interests = Column(String)
    target_audience_gender = Column(String)
    platforms_needed = Column(ARRAY(String))
    content_format_wanted = Column(ARRAY(String))
    content_type = Column(ARRAY(String))
    location_geography_focus = Column(String)
    timeline = Column(String)
    brand_brief = Column(Text)
    past_campaigns_links = Column(ARRAY(String))
    status = Column(String, default="new")
    stage = Column(String, default="initial_pitch")
    matched_creators = Column(ARRAY(UUID(as_uuid=True)))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deven_notes = Column(Text)

    # Relationships
    campaigns = relationship("Campaign", back_populates="brand", cascade="all, delete-orphan")


class Creator(Base):
    __tablename__ = "creators"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    email = Column(String)
    phone = Column(String)
    instagram_handle = Column(String)
    youtube_channel = Column(String)
    tiktok_handle = Column(String)
    other_platforms = Column(JSON)
    follower_count_instagram = Column(Numeric)
    follower_count_youtube = Column(Numeric)
    follower_count_tiktok = Column(Numeric)
    engagement_rate = Column(Numeric)
    niche = Column(String)
    content_specialties = Column(ARRAY(String))
    audience_age = Column(String)
    audience_gender = Column(String)
    location = Column(String)
    availability_status = Column(String, default="available")
    rates_per_post = Column(Numeric)
    rates_per_story = Column(Numeric)
    rates_per_reel = Column(Numeric)
    rates_longform_video = Column(Numeric)
    barter_willing = Column(Boolean, default=False)
    past_brand_collaborations = Column(ARRAY(String))
    bio = Column(Text)
    profile_picture_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deven_notes = Column(Text)

    # Relationships
    campaigns = relationship("Campaign", back_populates="creator", cascade="all, delete-orphan")


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand_id = Column(UUID(as_uuid=True), ForeignKey("brands.id"), nullable=True)
    creator_id = Column(UUID(as_uuid=True), ForeignKey("creators.id"), nullable=True)
    campaign_name = Column(String)
    collab_type = Column(String)  # "paid" or "barter"
    campaign_value = Column(Numeric)
    commission_percentage = Column(Numeric)
    commission_amount = Column(Numeric)
    flat_fee_amount = Column(Numeric)
    deliverables = Column(ARRAY(String))
    content_requirements = Column(Text)
    timeline_start = Column(Date)
    timeline_end = Column(Date)
    status = Column(String, default="negotiation")
    payment_status = Column(String, default="pending")
    amount_paid = Column(Numeric, default=0)
    content_submission_status = Column(String, default="pending")
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    brand = relationship("Brand", back_populates="campaigns")
    creator = relationship("Creator", back_populates="campaigns")
