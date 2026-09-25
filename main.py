import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./pocketsmart.db"
)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing from .env file")


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="PocketSmart AI",
    description="Smart Personal Budget Planning Assistant",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")


# =========================================================
# GEMINI
# =========================================================

client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# =========================================================
# DATABASE
# =========================================================

# PostgreSQL example:
#
# DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/pocketsmart
#
# For testing without PostgreSQL, the default above uses SQLite.

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# =========================================================
# DATABASE TABLES
# =========================================================

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


class PlanningHistory(Base):
    __tablename__ = "planning_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)

    planner_type = Column(String(50), nullable=False)

    input_data = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


Base.metadata.create_all(bind=engine)


# =========================================================
# REQUEST MODELS
# =========================================================

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str


class HomePlannerRequest(BaseModel):
    budget: float
    style: str
    room: str
    priority: str
    items: Optional[str] = ""


class PartyPlannerRequest(BaseModel):
    budget: float
    guests: int
    event_type: str
    venue: str
    preferences: Optional[str] = ""


class JewelryPlannerRequest(BaseModel):
    budget: float
    occasion: str
    outfit: str
    material: str
    description: Optional[str] = ""


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_db():
    db = SessionLocal()

    try:
        return db
    except Exception:
        db.close()
        raise


def save_history(
    user_id: int,
    planner_type: str,
    input_data: str,
    recommendation: str
):
    db = get_db()

    try:
        history = PlanningHistory(
            user_id=user_id,
            planner_type=planner_type,
            input_data=input_data,
            recommendation=recommendation
        )

        db.add(history)
        db.commit()

    finally:
        db.close()


def generate_ai_response(prompt: str) -> str:

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text

    except Exception as e:

        print("GEMINI ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail="Unable to generate AI recommendation."
        )


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():

    return {
        "success": True,
        "message": "PocketSmart AI backend is running"
    }


# =========================================================
# REGISTER
# =========================================================

@app.post("/api/auth/register")
def register(data: RegisterRequest):

    db = get_db()

    try:

        existing_email = db.query(User).filter(
            User.email == data.email
        ).first()

        if existing_email:

            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )

        existing_username = db.query(User).filter(
            User.username == data.username
        ).first()

        if existing_username:

            raise HTTPException(
                status_code=400,
                detail="Username already exists"
            )

        user = User(
            username=data.username,
            email=data.email,
            password=data.password
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return {
            "success": True,
            "message": "Registration successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        }

    finally:
        db.close()


# =========================================================
# LOGIN
# =========================================================

@app.post("/api/auth/login")
def login(data: LoginRequest):

    db = get_db()

    try:

        user = db.query(User).filter(
            User.email == data.email
        ).first()

        if not user:

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        if user.password != data.password:

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )

        return {
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        }

    finally:
        db.close()


# =========================================================
# LOGOUT
# =========================================================

@app.post("/api/auth/logout")
def logout():

    return {
        "success": True,
        "message": "Logout successful"
    }


# =========================================================
# CURRENT USER
# =========================================================

@app.get("/api/auth/me")
def current_user(user_id: int):

    db = get_db()

    try:

        user = db.query(User).filter(
            User.id == user_id
        ).first()

        if not user:

            raise HTTPException(
                status_code=404,
                detail="User not found"
            )

        return {
            "success": True,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        }

    finally:
        db.close()


# =========================================================
# GENERAL AI CHAT
# =========================================================

@app.post("/api/chat")
def chat(data: ChatRequest):

    prompt = f"""
You are PocketSmart AI.

You are a friendly personal budgeting and planning assistant.

User message:
{data.message}

Give practical, simple and easy-to-understand guidance.

Do not claim to be a professional financial advisor.

Keep the response concise and useful.
"""

    reply = generate_ai_response(prompt)

    return {
        "success": True,
        "reply": reply
    }


# =========================================================
# HOME PLANNER
# =========================================================

@app.post("/api/planner/home")
def home_planner(
    data: HomePlannerRequest,
    user_id: int
):

    prompt = f"""
You are PocketSmart AI Home Planner.

Create a practical home planning recommendation.

Budget: ₹{data.budget}
Interior Style: {data.style}
Room: {data.room}
Priority: {data.priority}
Required Items: {data.items}

Provide:

1. Recommended items
2. Estimated budget allocation
3. Priority items
4. Money-saving suggestions

Keep everything simple and realistic.
Use Indian Rupees.
"""

    recommendation = generate_ai_response(prompt)

    save_history(
        user_id=user_id,
        planner_type="Home",
        input_data=data.model_dump_json(),
        recommendation=recommendation
    )

    return {
        "success": True,
        "planner": "home",
        "recommendation": recommendation
    }


# =========================================================
# PARTY PLANNER
# =========================================================

@app.post("/api/planner/party")
def party_planner(
    data: PartyPlannerRequest,
    user_id: int
):

    prompt = f"""
You are PocketSmart AI Party Planner.

Create a practical party plan.

Budget: ₹{data.budget}
Number of Guests: {data.guests}
Event Type: {data.event_type}
Venue: {data.venue}
Preferences: {data.preferences}

Provide:

1. Food budget
2. Decoration budget
3. Venue budget
4. Entertainment suggestions
5. Other important expenses
6. Money-saving suggestions

Use Indian Rupees.
Keep the plan simple and realistic.
"""

    recommendation = generate_ai_response(prompt)

    save_history(
        user_id=user_id,
        planner_type="Party",
        input_data=data.model_dump_json(),
        recommendation=recommendation
    )

    return {
        "success": True,
        "planner": "party",
        "recommendation": recommendation
    }


# =========================================================
# JEWELRY PLANNER
# =========================================================

@app.post("/api/planner/jewelry")
def jewelry_planner(
    data: JewelryPlannerRequest,
    user_id: int
):

    prompt = f"""
You are PocketSmart AI Jewelry Planner.

Recommend suitable jewelry based on the user's requirements.

Budget: ₹{data.budget}
Occasion: {data.occasion}
Outfit: {data.outfit}
Preferred Material: {data.material}
Description: {data.description}

Provide:

1. Recommended jewelry type
2. Suitable material
3. Suitable design
4. Estimated budget
5. Styling suggestion
6. Budget-saving suggestion

Keep the recommendation simple and practical.
Use Indian Rupees.
"""

    recommendation = generate_ai_response(prompt)

    save_history(
        user_id=user_id,
        planner_type="Jewelry",
        input_data=data.model_dump_json(),
        recommendation=recommendation
    )

    return {
        "success": True,
        "planner": "jewelry",
        "recommendation": recommendation
    }


# =========================================================
# HISTORY
# =========================================================

@app.get("/api/history")
def get_history(user_id: int):

    db = get_db()

    try:

        records = (
            db.query(PlanningHistory)
            .filter(
                PlanningHistory.user_id == user_id
            )
            .order_by(
                PlanningHistory.created_at.desc()
            )
            .all()
        )

        history = []

        for record in records:

            history.append({
                "id": record.id,
                "planner_type": record.planner_type,
                "input_data": record.input_data,
                "recommendation": record.recommendation,
                "created_at": record.created_at.isoformat()
            })

        return {
            "success": True,
            "history": history
        }

    finally:
        db.close()


# =========================================================
# DELETE HISTORY
# =========================================================

@app.delete("/api/history/{history_id}")
def delete_history(
    history_id: int,
    user_id: int
):

    db = get_db()

    try:

        record = db.query(PlanningHistory).filter(
            PlanningHistory.id == history_id,
            PlanningHistory.user_id == user_id
        ).first()

        if not record:

            raise HTTPException(
                status_code=404,
                detail="History not found"
            )

        db.delete(record)
        db.commit()

        return {
            "success": True,
            "message": "History deleted"
        }

    finally:
        db.close()