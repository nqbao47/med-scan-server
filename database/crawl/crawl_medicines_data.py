import pandas as pd
import requests
import time
import random
from tqdm import tqdm

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.pharmacity.vn/',
}


def parser_product(json):
    d = dict()

    # Crawl dữ liệu từ 'data/product' trong JSON
    product_data = json.get('data', {}).get('product', {})

    d['id'] = product_data.get('id')
    d['name'] = product_data.get('name')
    d['longDescription'] = product_data.get('longDescription')

    # Crawl thông tin về hình ảnh
    images = product_data.get('images', [])
    image_info = []

    for image in images:
        image_data = {
            'id': image.get('id'),
            'url': image.get('url')
        }
        image_info.append(image_data)

    d['images'] = image_info

    return d


# Đọc danh sách slug từ file CSV
df_slug = pd.read_csv('product_id_ncds.csv')
slugs = df_slug.slug.to_list()  # Lấy danh sách slug
print(slugs)

result = []
for slug in tqdm(slugs, total=len(slugs)):
    response = requests.get(
        'https://api-gateway.pharmacity.vn/api/product?slug={}'.format(slug), headers=headers)
    if response.status_code == 200:
        print('Crawl data {} success !!!'.format(slug))
        result.append(parser_product(response.json()))

df_product = pd.DataFrame(result)
df_product.to_csv('crawled_data_ncds.csv', index=False)
