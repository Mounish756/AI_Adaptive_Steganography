import sys, os, torch
sys.path.insert(0, ".")
from ai.models.full_model import AdaptiveStegoModel
from ai.data.dataset import SteganographyDataset
from security.encryption import encrypt_message, decrypt_message
from security.payload import serialize_payload, deserialize_payload, bytes_to_bits, bits_to_bytes
import reedsolo

device = torch.device("cpu")
model = AdaptiveStegoModel(base_dim=32, scaling_factor=0.15).to(device)
ckpt = torch.load("checkpoints/phase2a/best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Test message
secret_msg = "AI-Adaptive-Steganography: Final Major Academic Project 2026. Top secret payload verified!"
password = "StrongMasterPassword2026!"

# 1. Encrypt
salt, nonce, ct, tag = encrypt_message(secret_msg, password)
framed = serialize_payload(salt, nonce, ct, tag)
print(f"Plaintext len: {len(secret_msg)} B | Framed payload len: {len(framed)} B")

# 2. Reed-Solomon encode
rsc = reedsolo.RSCodec(nsym=32)
ecc_encoded = bytes(rsc.encode(framed))
print(f"ECC encoded len: {len(ecc_encoded)} B")

# 3. Pack into 65536 bits (8192 bytes)
full_payload = len(ecc_encoded).to_bytes(4, "big") + ecc_encoded
pad_len = 8192 - len(full_payload)
assert pad_len >= 0, "Exceeds carrier capacity"
full_payload += b"\x00" * pad_len

bits = bytes_to_bits(full_payload)
payload_tensor = torch.tensor(bits, dtype=torch.float32).view(1, 1, 256, 256).to(device)

# 4. Neural embed & extract on test image
test_ds = SteganographyDataset("datasets/coco_subset", split="test", seed=42)
batch = test_ds[0]
cover = batch["image"].unsqueeze(0).to(device)

with torch.no_grad():
    stego = model.encoder(cover, payload_tensor, model.adaptive_analyzer(cover), scaling_factor=0.15)
    extracted = model.decoder(stego)

pred_bits = (extracted[0, 0].view(-1) > 0.5).int().tolist()
bit_errors = sum(b1 != b2 for b1, b2 in zip(bits, pred_bits))
print(f"Raw channel bit errors: {bit_errors} / {len(bits)} (BER: {bit_errors/len(bits)*100:.3f}%)")

# 5. Extract bytes
rec_bytes = bits_to_bytes(pred_bits)
rec_len = int.from_bytes(rec_bytes[:4], "big")
rec_ecc = rec_bytes[4:4 + rec_len]

# 6. Reed-Solomon decode
try:
    decoded_framed = bytes(rsc.decode(rec_ecc)[0])
    print("RS Decode SUCCESS!")
    d_salt, d_nonce, d_ct, d_tag, d_ver = deserialize_payload(decoded_framed)
    recovered_text = decrypt_message(d_salt, d_nonce, d_ct, d_tag, password)
    print("AES-GCM Decrypt SUCCESS!")
    print(f"Recovered message: \"{recovered_text}\"")
    assert recovered_text == secret_msg
    print("100% EXACT RECOVERY MATCH VERIFIED!")
except Exception as e:
    print("Decode/decrypt failed:", e)
