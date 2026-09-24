"""Asynchronous SQLAlchemy database session configuration."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from autopassai.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
)

session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession]:
    """Provide an asynchronous database session."""
    async with session_factory() as session:
        yield session
