"""Modal credentials never sent to storage hosts."""
import requests

def predict(url, key, secret, image):
    if len(image)>10*1024*1024: raise ValueError('Frame exceeds Modal byte limit')
    response=requests.post(url,data=image,headers={'Content-Type':'image/jpeg','Modal-Key':key,'Modal-Secret':secret},
                           timeout=(10,120),allow_redirects=False)
    if response.status_code != 200: raise RuntimeError('Modal inference request failed')
    return response.json()
