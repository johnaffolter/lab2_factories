"""
Composable System API Routes
REST endpoints for managing composable components and pipelines
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import asyncio
from datetime import datetime

from app.core.composable import global_registry, ComposablePipeline
from app.services.mlops_neo4j_integration import get_knowledge_graph

router = APIRouter(prefix="/api/composable", tags=["composable"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ComponentSearchRequest(BaseModel):
    query: str
    type_filter: Optional[str] = None


class PipelineCreateRequest(BaseModel):
    name: str
    components: List[Dict[str, Any]]


class PipelineExecuteRequest(BaseModel):
    email_subject: str
    email_body: str


class ComponentResponse(BaseModel):
    name: str
    version: str
    type: str
    description: str
    author: str
    tags: List[str]
    icon: str
    color: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]
    config_schema: Dict[str, Any]


# ============================================================================
# COMPONENT ENDPOINTS
# ============================================================================

@router.get("/components", response_model=Dict[str, Any])
async def list_components(type_filter: Optional[str] = None):
    """
    List all registered composable components

    Query Parameters:
    - type_filter: Filter by component type (feature_generator, model, etc.)

    Returns component metadata for UI display
    """
    try:
        from app.core.composable import ComponentType

        filter_enum = None
        if type_filter:
            try:
                filter_enum = ComponentType(type_filter)
            except ValueError:
                pass

        components = global_registry.list_all(type_filter=filter_enum)

        return {
            "total": len(components),
            "components": [
                {
                    "name": c.name,
                    "version": c.version,
                    "type": c.type.value,
                    "description": c.description,
                    "author": c.author,
                    "tags": c.tags,
                    "icon": c.icon,
                    "color": c.color,
                    "inputs": c.inputs,
                    "outputs": c.outputs,
                    "config_schema": c.config_schema
                }
                for c in components
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/components/{component_name}")
async def get_component_details(component_name: str):
    """Get detailed information about a specific component"""
    try:
        ComponentClass = global_registry.get(component_name)

        if not ComponentClass:
            raise HTTPException(status_code=404, detail=f"Component '{component_name}' not found")

        temp = ComponentClass()
        metadata = temp.get_metadata()

        return {
            "name": metadata.name,
            "version": metadata.version,
            "type": metadata.type.value,
            "description": metadata.description,
            "author": metadata.author,
            "tags": metadata.tags,
            "icon": metadata.icon,
            "color": metadata.color,
            "inputs": metadata.inputs,
            "outputs": metadata.outputs,
            "config_schema": metadata.config_schema,
            "default_config": temp.config
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/components/search")
async def search_components(request: ComponentSearchRequest):
    """Search components by name, description, or tags"""
    try:
        results = global_registry.search(request.query)

        return {
            "query": request.query,
            "total_results": len(results),
            "results": [
                {
                    "name": r.name,
                    "version": r.version,
                    "description": r.description,
                    "tags": r.tags,
                    "icon": r.icon,
                    "color": r.color
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================

# In-memory pipeline storage (replace with database in production)
pipelines_db: Dict[str, Dict[str, Any]] = {}


@router.post("/pipelines")
async def create_pipeline(request: PipelineCreateRequest):
    """
    Create a new composable pipeline

    Body:
    {
        "name": "My Pipeline",
        "components": [
            {"name": "Spam Detector", "config": {"threshold": 0.5}},
            {"name": "Sentiment Analyzer", "config": {}}
        ]
    }
    """
    try:
        # Create pipeline from request
        components = []
        for comp_data in request.components:
            ComponentClass = global_registry.get(comp_data["name"])
            if not ComponentClass:
                raise HTTPException(
                    status_code=400,
                    detail=f"Component '{comp_data['name']}' not found"
                )

            component = ComponentClass(config=comp_data.get("config", {}))
            components.append(component)

        pipeline = ComposablePipeline(components)
        pipeline.name = request.name

        # Generate pipeline ID
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Store pipeline
        pipelines_db[pipeline_id] = {
            "id": pipeline_id,
            "name": pipeline.name,
            "components": request.components,
            "created_at": datetime.now().isoformat(),
            "executions": 0
        }

        return {
            "id": pipeline_id,
            "name": pipeline.name,
            "components": len(components),
            "status": "created"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipelines")
async def list_pipelines():
    """List all saved pipelines"""
    return {
        "total": len(pipelines_db),
        "pipelines": list(pipelines_db.values())
    }


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str):
    """Get details of a specific pipeline"""
    if pipeline_id not in pipelines_db:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    return pipelines_db[pipeline_id]


@router.post("/pipelines/{pipeline_id}/execute")
async def execute_pipeline(pipeline_id: str, request: PipelineExecuteRequest):
    """
    Execute a saved pipeline on an email

    Body:
    {
        "email_subject": "Test Email",
        "email_body": "Email content..."
    }
    """
    try:
        if pipeline_id not in pipelines_db:
            raise HTTPException(status_code=404, detail="Pipeline not found")

        pipeline_data = pipelines_db[pipeline_id]

        # Reconstruct pipeline
        pipeline = ComposablePipeline.from_dict(
            {"name": pipeline_data["name"], "components": pipeline_data["components"]},
            global_registry
        )

        # Create email object
        from dataclasses import dataclass

        @dataclass
        class Email:
            subject: str
            body: str

        email = Email(subject=request.email_subject, body=request.email_body)

        # Execute each component and collect results
        results = []
        current_input = email

        for component in pipeline.components:
            result = component.execute(current_input)
            results.append({
                "component": component.metadata.name,
                "output": result
            })

        # Update execution count
        pipelines_db[pipeline_id]["executions"] += 1

        return {
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline_data["name"],
            "email_subject": request.email_subject,
            "results": results,
            "execution_count": pipelines_db[pipeline_id]["executions"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a saved pipeline"""
    if pipeline_id not in pipelines_db:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    del pipelines_db[pipeline_id]

    return {
        "status": "deleted",
        "pipeline_id": pipeline_id
    }


# ============================================================================
# SYSTEM STATUS & MONITORING
# ============================================================================

@router.get("/status")
async def get_system_status():
    """Get overall system status and statistics"""
    try:
        # Get Neo4j status
        kg = get_knowledge_graph()
        neo4j_overview = kg.get_mlops_system_overview()

        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "total": len(global_registry.list_all()),
                "types": {
                    "feature_generators": len([
                        c for c in global_registry.list_all()
                        if c.type.value == "feature_generator"
                    ])
                }
            },
            "pipelines": {
                "total": len(pipelines_db),
                "total_executions": sum(p["executions"] for p in pipelines_db.values())
            },
            "neo4j": {
                "emails": neo4j_overview.get("emails", {}).get("total", 0),
                "labeled": neo4j_overview.get("emails", {}).get("labeled", 0),
                "models": neo4j_overview.get("models", {}).get("total", 0)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/marketplace")
async def get_marketplace_data():
    """Get data for component marketplace view"""
    try:
        components = global_registry.list_all()

        # Group by type
        by_type = {}
        for comp in components:
            comp_type = comp.type.value
            if comp_type not in by_type:
                by_type[comp_type] = []

            by_type[comp_type].append({
                "name": comp.name,
                "version": comp.version,
                "description": comp.description,
                "author": comp.author,
                "tags": comp.tags,
                "icon": comp.icon,
                "color": comp.color,
                "downloads": 0,  # Placeholder
                "rating": 4.5  # Placeholder
            })

        return {
            "total_components": len(components),
            "categories": by_type,
            "featured": components[:3] if len(components) >= 3 else components,
            "recently_added": []  # Placeholder
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET - REAL-TIME UPDATES
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@router.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    """
    WebSocket endpoint for real-time system status updates

    Connect to: ws://localhost:8000/api/composable/ws/status
    """
    await manager.connect(websocket)

    try:
        while True:
            # Get current status
            status = await get_system_status()

            # Send to client
            await websocket.send_json({
                "type": "status_update",
                "data": status,
                "timestamp": datetime.now().isoformat()
            })

            # Wait 2 seconds before next update
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)