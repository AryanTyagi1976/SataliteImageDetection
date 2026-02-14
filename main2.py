import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# 1️⃣ Load Vision Transformer
# -------------------------------------------------------
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.to(device)
model.eval()

# -------------------------------------------------------
# 2️⃣ Image Transform
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((224,224))
    tensor = transform(img_resized).unsqueeze(0).to(device)
    return img_resized, tensor

def load_mask(path):
    mask = Image.open(path).convert("L")
    mask = mask.resize((224,224))
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)
    return mask

# -------------------------------------------------------
# 3️⃣ Load Your Images
# -------------------------------------------------------
before_raw, before_tensor = load_image("beforeimage.jpg")
after_raw, after_tensor = load_image("afterimage.jpg")

before_np = np.array(before_raw)

# -------------------------------------------------------
# 4️⃣ Predict Change
# -------------------------------------------------------
with torch.no_grad():
    f_before = model.forward_features(before_tensor)[:,1:,:]
    f_after = model.forward_features(after_tensor)[:,1:,:]

diff = torch.abs(f_before - f_after).mean(dim=2)
diff = diff.reshape(14,14).cpu().numpy()

norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

heatmap_up = np.kron(norm, np.ones((16,16)))

threshold = 0.4
pred_mask = (heatmap_up > threshold).astype(np.uint8)

# -------------------------------------------------------
# 5️⃣ Percentage Change
# -------------------------------------------------------
percentage_change = (pred_mask.sum() / pred_mask.size) * 100
print(f"\n🔎 Percentage Area Changed: {percentage_change:.2f}%")

# -------------------------------------------------------
# 6️⃣ If Ground Truth Exists → Evaluate
# -------------------------------------------------------
if os.path.exists("gtmask.png"):

    gt_mask = load_mask("gtmask.png")

    y_pred = pred_mask.flatten()
    y_true = gt_mask.flatten()

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / (union + 1e-8)

    accuracy = (y_pred == y_true).sum() / len(y_true)

    print("\n📊 Evaluation Results")
    print("-------------------------------------------")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"IoU       : {iou:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

else:
    print("\n⚠ No ground truth mask found (gtmask.png). Evaluation skipped.")

# -------------------------------------------------------
# 7️⃣ Overlay Visualization
# -------------------------------------------------------
red_mask = np.zeros_like(before_np)
red_mask[:,:,0] = pred_mask * 255

overlay = (0.7 * before_np + 0.6 * red_mask).astype(np.uint8)

plt.figure(figsize=(14,5))

plt.subplot(1,3,1)
plt.imshow(before_raw)
plt.title("Before Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(heatmap_up, cmap="Reds")
plt.title("Change Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(overlay)
plt.title("Detected Change Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()

Image.fromarray(overlay).save("detected_change_overlay.png")
print("\n✅ Overlay saved as detected_change_overlay.png")
