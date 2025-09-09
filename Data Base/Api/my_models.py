from sqlalchemy import Column, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from my_database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"

    email = Column(String, primary_key=True, index=True)
    password_hash = Column(String, nullable=False)
    car_plat = Column(String, unique=True)

    cars = relationship("Car", back_populates="owner")

# جدول السيارات
class Car(Base):
    __tablename__ = "car"

    car_plat = Column(String, primary_key=True, index=True)
    email = Column(String, ForeignKey("users.email"), nullable=False)
    fees = Column(Float, default=0.0)
    enter_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    segment_id = Column(String, nullable=True)  # A, B, C, D
    payment_status = Column(String, default="no")  # yes / no

    owner = relationship("User", back_populates="cars")
