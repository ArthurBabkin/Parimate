import uvicorn
from fastapi import FastAPI
from src.routes import router

def create_app() -> FastAPI:
    app = FastAPI()
    set_routers(app)
    return app


def set_routers(app: FastAPI) -> None:
    app.include_router(router.router)

if __name__=="__main__":
    app = create_app()
    uvicorn.run(app, port=5005)

