import requests
import json

data = {"orderId": "unique_order_id",
 "items": [
    {"sku": "unique_sku_1", "count": 1, "size1": "5.1", "size2": "2.2", "size3": "5.3",
     "weight": "7.34", "type": ["2"]},
    {"sku": "unique_sku_2", "count": 3, "size1": "4", "size2": "5.23", "size3": "6.2",
     "weight": "7.45", "type": ["8", "9", "10"]},
    {"sku": "unique_sku_3", "count": 2, "size1": "11", "size2": "12.5", "size3": "13.3",
     "weight": "14.2", "type": ["15", "16"]}
   ]
}

r = requests.post("http://localhost:8000/pack", json=data)

try:
    print(r.json())
except json.JSONDecodeError:
    print("Invalid or empty JSON response.")
