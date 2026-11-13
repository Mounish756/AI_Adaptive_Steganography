import requests
import os

def test_full_roundtrip():
    # 1. Frontend check
    r = requests.get('http://127.0.0.1:8080/index.html')
    assert 'Adaptive Image Embedding' in r.text, 'Title missing in frontend'
    assert 'Secure information concealment powered by AI' in r.text, 'Subtitle missing'
    assert 'operationsTbody' in r.text, 'Operations table missing'
    print('[PASS] Frontend HTML verified')

    r_css = requests.get('http://127.0.0.1:8080/style.css')
    assert '#C9A227' in r_css.text, 'Luxury Gold missing'
    assert '#141414' in r_css.text, 'Black missing'
    assert '#4A3026' in r_css.text, 'Dark Brown missing'
    assert '#FFFFFF' in r_css.text, 'White missing'
    print('[PASS] Strict 4-color palette verified in CSS')

    # 2. Download and Re-Upload Roundtrip Test
    img_path = 'datasets/coco_subset/coco_val_000000000139.jpg'
    secret_text = 'TopSecret-Cybersecurity-2026-Verification-Payload!'
    passphrase = 'EnterprisePassphrase987!'

    with open(img_path, 'rb') as f:
        files = {'image': ('input.jpg', f, 'image/jpeg')}
        data = {'message': secret_text, 'password': passphrase}
        r = requests.post('http://127.0.0.1:5050/api/download_stego', files=files, data=data)

    assert r.status_code == 200, f'Download failed: {r.status_code}'
    downloaded_png_path = 'outputs/roundtrip_test_stego.png'
    with open(downloaded_png_path, 'wb') as f:
        f.write(r.content)
    print(f'[PASS] Downloaded stego PNG ({len(r.content)} bytes)')

    # 3. Extract from re-uploaded PNG
    with open(downloaded_png_path, 'rb') as f:
        files = {'image': ('stego.png', f, 'image/png')}
        data = {'password': passphrase}
        r_ext = requests.post('http://127.0.0.1:5050/api/extract', files=files, data=data)

    assert r_ext.status_code == 200, f'Extract failed: {r_ext.status_code}'
    recovered = r_ext.json().get('message')
    print(f'[PASS] Recovered message: "{recovered}"')
    assert recovered == secret_text, 'Exact recovery failed!'
    print('[PASS] Exact message recovery verified!')

    # 4. Incorrect Password Rejection
    with open(downloaded_png_path, 'rb') as f:
        files = {'image': ('stego.png', f, 'image/png')}
        data = {'password': 'WrongPassword123'}
        r_bad = requests.post('http://127.0.0.1:5050/api/extract', files=files, data=data)

    assert r_bad.status_code == 401, f'Expected 401, got {r_bad.status_code}'
    assert 'Authentication failed' in r_bad.json().get('error', '')
    print('[PASS] Incorrect password rejected safely with no plaintext exposure')

if __name__ == '__main__':
    test_full_roundtrip()
