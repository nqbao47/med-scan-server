import requests
import time
import random
import pandas as pd

cookies = {}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.pharmacity.vn/',
}

params = {
    'slug': 'duoc-pham',
    'page': '1',
    'page_size': '20',
}

product_id = []
for i in range(1, 179):
    params['page'] = i
    response = requests.get('https://api-gateway.pharmacity.vn/api/category',
                            headers=headers, params=params)
    if response.status_code == 200:
        print('request success!!!')
        data = response.json().get('data')
        if data:
            products = data.get('products')
            if products:
                edges = products.get('edges')
                if edges:
                    for record in edges:
                        node = record.get('node')
                        if node:
                            product_id.append({
                                'slug': node.get('slug')
                            })
    time.sleep(random.randrange(3, 10))

df = pd.DataFrame(product_id)
df.to_csv('product_id_ncds.csv', index=False)
