from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base


DATABASE_URL = "sqlite:///./tickets.db"
MODEL_PATH = "models/model_pipeline.joblib"


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class TicketDB(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, index=True)

    text = Column(String)
    label = Column(String)
    confidence = Column(Float)
    status = Column(String, default="open")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    triage_required = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)

Base.metadata.create_all(bind=engine)


try:
    pipeline = joblib.load(MODEL_PATH)
    ml_models_loaded = True
except Exception as e:
    print(f"Warning: ML models not loaded. {e}")
    ml_models_loaded = False


class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float

class IngestRequest(BaseModel):
    sender: str = Field(alias="from")
    text: str
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "from": "+1234567890",
                "text": "Hello"
            }
        }

class IngestResponse(BaseModel):
    id: int
    sender: str = Field(alias="from")
    text: str
    label: str
    confidence: float
    status: str
    created_at: datetime.datetime
    triage_required: bool
    
    class Config:
        populate_by_name = True

class TicketResponse(BaseModel):
    id: int
    sender: str
    label: str
    status: str
    
    class Config:
        from_attributes = True

class ResolveRequest(BaseModel):
    status: str

class ResolveResponse(BaseModel):
    id: int
    status: str
    resolved_at: datetime.datetime


app = FastAPI(title="AI Message Triage System")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ml/predict", response_model=PredictResponse)
def predict_category(request: PredictRequest):
    if not ml_models_loaded:
        raise HTTPException(status_code=503, detail="ML models not available")
    
    prediction = pipeline.predict([request.text])[0]
    probabilities = pipeline.predict_proba([request.text])[0]
    confidence = max(probabilities)
    
    return {"label": prediction, "confidence": float(confidence)}

@app.post("/messages/ingest")
def ingest_message(request: IngestRequest):
    if not ml_models_loaded:
        raise HTTPException(status_code=503, detail="ML models not available")


    label = pipeline.predict([request.text])[0]
    probabilities = pipeline.predict_proba([request.text])[0]
    confidence = float(max(probabilities))
    
    triage_required = confidence < 0.7
    

    db = SessionLocal()
    db_ticket = TicketDB(
        sender=request.sender,
        text=request.text,
        label=label,
        confidence=confidence,
        status="open",
        created_at=datetime.datetime.utcnow(),
        triage_required=triage_required
    )
    db.add(db_ticket)
    db.commit()
    db.refresh(db_ticket)
    db.close()
    
    return {
        "id": db_ticket.id,
        "from": db_ticket.sender,
        "sender": db_ticket.sender,

        "text": db_ticket.text,
        "label": db_ticket.label,
        "confidence": db_ticket.confidence,
        "status": db_ticket.status,
        "created_at": db_ticket.created_at,
        "triage_required": db_ticket.triage_required
    }

@app.get("/tickets")
def list_tickets(label: Optional[str] = None, status: Optional[str] = None):
    db = SessionLocal()
    query = db.query(TicketDB)
    if label:
        query = query.filter(TicketDB.label == label)
    if status:
        query = query.filter(TicketDB.status == status)
    
    tickets = query.all()
    db.close()
    
    return [
        {
            "id": t.id,
            "from": t.sender,
            "label": t.label,
            "status": t.status
        }
        for t in tickets
    ]

@app.patch("/tickets/{id}")
def resolve_ticket(id: int, request: ResolveRequest):
    if request.status != "resolved":
        raise HTTPException(status_code=400, detail="Only status='resolved' is supported")
        
    db = SessionLocal()
    ticket = db.query(TicketDB).filter(TicketDB.id == id).first()
    if not ticket:
        db.close()
        raise HTTPException(status_code=404, detail="Ticket not found")
        
    ticket.status = "resolved"
    ticket.resolved_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(ticket)
    db.close()
    
    return {
        "id": ticket.id,
        "status": ticket.status,
        "resolved_at": ticket.resolved_at
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
