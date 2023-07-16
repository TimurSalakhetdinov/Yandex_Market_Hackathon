from fastapi import FastAPI, Request
from fastapi import HTTPException
import uvicorn
import argparse
from model import predict
from pydantic import BaseModel
from typing import List

class Item(BaseModel):
    sku: str
    count: int
    size1: str
    size2: str
    size3: str
    weight: str
    type: List[str]

class Order(BaseModel):
    orderId: str
    items: List[Item]

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/pack")
def get_prediction(request: Order):
    try:
        items = [el.dict() for el in request.items]
        y = predict({'items': items})
        return {"orderId": request.orderId, "package": y, "status": "ok"}


    except Exception as e:

        print(f"Exception: {e}")

        raise HTTPException(status_code=400, detail={"orderId": request.orderId, "status": "fail"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    parser.add_argument("--debug", action="store_true", dest="debug")
    args = vars(parser.parse_args())

    if args['debug']:
        uvicorn.run(app, host=args['host'], port=args['port'], log_level="debug")
    else:
        uvicorn.run(app, host=args['host'], port=args['port'])