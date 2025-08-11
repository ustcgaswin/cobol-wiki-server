import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8003,
        reload=True,
        reload_dirs=["./app"],
        reload_excludes=[".venv/*"],
        log_config=None
    )

# mlflow server --backend-store-uri sqlite:///mydb.sqlite