"""Read-only HTTP viewer over the existing, validated reconstruction gallery."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'scripts'))
from build_viewer import prepare_payload


def create_app(run: Path | None = None):
    run = Path(run or os.environ.get('SPEN_RUN', PROJECT / 'runs/all_raw_260915')).resolve()

    @asynccontextmanager
    async def lifespan(app):
        payload, assets, _ = prepare_payload(run)
        app.state.payload = payload
        yield

    app = FastAPI(title='SPEN Reconstruction Viewer', lifespan=lifespan)

    @app.get('/api/catalog')
    def catalog():
        payload = app.state.payload
        return {**payload, 'cases': [
            {**{k: v for k, v in c.items() if k != 'frames'}, 'frame_count': len(c['frames'])}
            for c in payload['cases']]}

    @app.get('/api/experiments/{experiment}')
    def experiment_data(experiment: str):
        cases = [c for c in app.state.payload['cases'] if c['experiment'] == experiment]
        if not cases:
            raise HTTPException(404, 'Experiment not found')
        return {'cases': cases}

    # Starlette resolves paths within this root and rejects directory traversal.
    app.mount('/data', StaticFiles(directory=run), name='data')
    dist = PROJECT / 'web/frontend/dist'
    if dist.is_dir():
        app.mount('/assets', StaticFiles(directory=dist / 'assets'), name='assets')

    @app.get('/')
    def frontend():
        if not (dist / 'index.html').is_file():
            raise HTTPException(503, 'Build frontend first: cd web/frontend && npm ci && npm run build')
        return FileResponse(dist / 'index.html')

    return app


app = create_app()
