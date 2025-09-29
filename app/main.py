from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as api_router
from app.api.composable_routes import router as composable_router
from app.core.config import settings

# Import composable components to register them
from app.features.composable_generators import *

app = FastAPI(
    title="MLOps Composable Platform",
    description="Composable ML Components with Visual Pipeline Builder",
    version="2.0.0"
)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router)
app.include_router(composable_router)

# Serve static files for UI
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass  # Directory might not exist yet

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)