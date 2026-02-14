import torch
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load ViT model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)

before_img = load_image("beforeimage.jpg")
after_img  = load_image("afterimage.jpg")

# Extract features
with torch.no_grad():
    before_f = model.forward_features(before_img)   # shape: [1,197,768]
    after_f  = model.forward_features(after_img)

# Remove CLS token
before_f = before_f[:, 1:, :]   # [1,196,768]
after_f  = after_f[:, 1:, :]

# Diff per patch
diff = torch.abs(before_f - after_f).mean(dim=2)   # [1,196]

# Reshape to grid (14x14)
diff_map = diff.reshape(14, 14).detach().numpy()

# Normalize
diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())

# Plot heatmap
plt.imshow(diff_map, cmap='Reds')
plt.title("Detected Change Heatmap")
plt.axis("off")
plt.show()
