from sqlalchemy.orm import Session
from sqlalchemy import text

# ✅ استعلام رسوم الركنة من view: car_duration_inside
def get_current_fees(db: Session, car_plat: str):
    result = db.execute(
        text("SELECT * FROM car_duration_inside WHERE car_plat = :car_plat"),
        {"car_plat": car_plat}
    ).fetchone()
    return result.fees if result else 0.0

# ✅ استعلام مدة الركنة من view: car_live_fees
def get_live_duration(db: Session, car_plat: str):
    result = db.execute(
        text("SELECT * FROM car_live_fees WHERE car_plat = :car_plat"),
        {"car_plat": car_plat}
    ).fetchone()
    return result.duration if result else "0 minutes"

from datetime import datetime, timedelta
from jose import jwt

SECRET_KEY = "your-secret"
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # Make token "permanent" by setting a far future date
        expire = datetime.utcnow() + timedelta(days=3650)  # 10 years
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
