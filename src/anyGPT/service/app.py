import argparse
import uvicorn
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from anyGPT.inference.runner import AnyGPTRunner


# Parsing for cli
def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anyGPT inference service",
        description="Loads an anyGPT model and hosts it on a simple microservice that can run inference over the network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model", action="store", help="Path t0 the trained model checkpoint to load"
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        default=5000,
        help="Port to start the microservice on",
    )
    parser.add_argument(
        "--host",
        action="store",
        type=str,
        default="127.0.0.1",
        help="Host to bind microservice to",
    )
    parser.add_argument(
        "--log-level", action="store", default="info", help="uvicorn log level"
    )

    return parser


# Pydantic DataModel to define request data
class InferenceRequest(BaseModel):
    data: str
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None


class Config:
    model: str


# Main app
app = FastAPI()
config = Config()


# Runner dependency
def get_runner():
    return AnyGPTRunner(config.model)


# Runs inference on the model
@app.post("/infer")
async def infer(request: InferenceRequest, runner: AnyGPTRunner = Depends(get_runner)):
    print(f"Running inference on {request.data}")
    kwargs = {
        "max_new_tokens": request.max_new_tokens,
        "temperature": request.temperature,
        "top_k": request.top_k,
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return runner.sample(request.data, **filtered_kwargs)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    config.model = args.model
    uvicorn.run(
        "anyGPT.service.app:app",
        port=args.port,
        host=args.host,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
