from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime

from auth import get_password_hash, verify_password

from my_database import SessionLocal, engine
from my_models import Base, User, Car
from plate_processor import PlateDetector
import shutil
import os
import cv2
import config
from sqlalchemy import text
from sqlalchemy import update

Base.metadata.create_all(bind=engine)
app = FastAPI()

## إعداد قاعدة البيانات
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
config.validate_paths()

# إعداد كاشف اللوحات
model_path = config.MODEL_PATH
tesseract_path = config.TESSERACT_PATH
detector = PlateDetector(model_path=model_path, tesseract_path=tesseract_path)


import re
from utils import create_access_token


def is_valid_email(email: str) -> bool:
    # Regular expression for email validation
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None


@app.post("/register")
def register(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if not is_valid_email(email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    user = db.query(User).filter_by(email=email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = get_password_hash(password)
    new_user = User(email=email, password_hash=hashed)
    db.add(new_user)
    db.commit()

    # Permanent token (no expiry)
    token = create_access_token({"sub": email}, expires_delta=None)

    return {"message": "User created", "access_token": token}


@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered. Please create an account.")

    if not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid password")

    # Permanent token (no expiry)
    token = create_access_token({"sub": email}, expires_delta=None)

    return {"access_token": token}


@app.post("/detect-plate")
def detect_plate(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    image = cv2.imread(temp_path)
    plate = detector.detect_plate_number(image)
    os.remove(temp_path)
    return {"detected_plate": plate}

@app.post("/confirm-plate")
def confirm_plate(email: str = Form(...), car_plat: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.car_plat = car_plat

    ##car = Car(car_plat=car_plat, email=email, enter_time=datetime.utcnow())
    
   ## db.add(car)
    db.commit()
    return {"message": "Plate linked and car entry registered"}

@app.post("/choose-segment")
def choose_segment(car_plat: str = Form(...), segment_id: str = Form(...), db: Session = Depends(get_db)):
    car = db.query(Car).filter_by(car_plat=car_plat).first()
    if not car:
        raise HTTPException(status_code=404, detail="Car not found")
    car.segment_id = segment_id
    db.commit()
    return {"message": "Segment updated"}

from sqlalchemy import text

@app.get("/current-fees")
def current_fees(car_plat: str, db: Session = Depends(get_db)):
    query = text("SELECT fees FROM car_live_fees WHERE car_plat = :car_plat")
    result = db.execute(query, {"car_plat": car_plat}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Plate not found in car_live_fees view")

    return {"car_plat": car_plat, "fees": result[0]}


@app.get("/current-duration")
def current_duration(car_plat: str, db: Session = Depends(get_db)):
    query = text("SELECT time_inside FROM car_duration_inside WHERE car_plat = :car_plat")
    result = db.execute(query, {"car_plat": car_plat}).fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="Plate not found in car_duration_inside view")

    return {"car_plat": car_plat, "time_inside": result[0]}

from sqlalchemy import text

@app.post("/exit")
def exit_car(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # حفظ الصورة مؤقتًا
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    image = cv2.imread(temp_path)
    os.remove(temp_path)

    # استخلاص الرقم
    car_plat = detector.detect_plate_number(image)

    if not car_plat:
        raise HTTPException(status_code=400, detail="Could not detect plate number")

    # إدراج رقم العربية فقط في جدول car
    new_car = Car(car_plat=car_plat)
    db.add(new_car)
    db.commit()

    # جلب fees من جدول car
    query = text("SELECT fees FROM car WHERE car_plat = :car_plat")
    result = db.execute(query, {"car_plat": car_plat}).fetchone()
    fees = result[0] if result else None

    return {
        "message": "Car plate recorded at exit",
        "car_plat": car_plat,
        "fees": fees
    }



@app.post("/confirm-payment")
def confirm_payment(car_plat: str = Form(...), db: Session = Depends(get_db)):
    stmt = (
        update(Car)
        .where(Car.car_plat == car_plat)
        # .where(Car.exit_time.isnot(None))  # مهم: لازم يكون خرج بالفعل
        .values(payment_status="yes")
        .execution_options(synchronize_session="fetch")  # يمنع StaleDataError
    )

    result = db.execute(stmt)
    db.commit()

    if result.rowcount == 0:
        raise HTTPException(
            status_code=404,
            detail="Payment confirmed successfully."
        )

    return {
        "message": "Payment confirmed successfully",
        "car_plat": car_plat
    }
