import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import User
from app.db.session import get_db

_bearer = HTTPBearer()


class AuthService:
    def hash_password(self, plain: str) -> str:
        # Convert string to bytes before hashing
        pwd_bytes = plain.encode("utf-8")
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

    def verify_password(self, plain: str, hashed: str) -> bool:
        pwd_bytes = plain.encode("utf-8")
        hashed_bytes = hashed.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hashed_bytes)

    def create_access_token(self, user_id: int, username: str) -> str:
        payload = {"sub": str(user_id), "username": username}
        return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    def decode_token(self, token: str) -> dict:
        try:
            return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(_bearer),
        db: AsyncSession = Depends(get_db),
    ) -> User:
        payload = self.decode_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        result = await db.execute(select(User).where(User.id == int(user_id)))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        return user


auth_service = AuthService()