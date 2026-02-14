import torch
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_img(path):
    img = Image.open(path).convert("RGB")
    resized = img.resize((224, 224))
    return resized, transform(resized).unsqueeze(0)


before_raw, before_tensor = load_img("beforeimage3.jpg")
after_raw, after_tensor = load_img("afterimage3.jpg")

before_np = np.array(before_raw)
after_np = np.array(after_raw)


with torch.no_grad():
    f_before = model.forward_features(before_tensor)[:, 1:, :]  
    f_after = model.forward_features(after_tensor)[:, 1:, :]


diff = torch.abs(f_before - f_after).mean(dim=2).reshape(14, 14).detach().numpy()
norm_map = (diff - diff.min()) / (diff.max() - diff.min())


heatmap_up = np.kron(norm_map, np.ones((16,16)))  


heat_intensity = (heatmap_up * 255).astype(np.uint8)
red_mask = np.zeros_like(before_np)
red_mask[:,:,0] = heat_intensity

overlay = (0.7 * before_np + 0.6 * red_mask).astype(np.uint8)


delta = after_np.astype(np.int16) - before_np.astype(np.int16)


flood_mask = (after_np[:,:,2] > after_np[:,:,1]) & (after_np[:,:,2] > after_np[:,:,0]) & (delta[:,:,2] > 20)


veg_loss_mask = (before_np[:,:,1] > 120) & (after_np[:,:,1] < 80)


building_mask = (delta.mean(axis=2) > 25)


fire_mask = ((before_np.mean(axis=2) - after_np.mean(axis=2)) > 40)


snow_mask = ((after_np.mean(axis=2) > 200) & (before_np.mean(axis=2) < 180)) | \
            ((before_np.mean(axis=2) > 200) & (after_np.mean(axis=2) < 180))


multi_class = np.zeros((224,224,3), dtype=np.uint8)
multi_class[flood_mask] = [0, 0, 255]       
multi_class[veg_loss_mask] = [0,255,0]      
multi_class[building_mask] = [255,255,0]    
multi_class[fire_mask] = [255,0,0]          
multi_class[snow_mask] = [255,255,255]      

multi_overlay = (0.6 * before_np + 0.4 * multi_class).astype(np.uint8)


threshold = 0.4
changed_pixels = np.sum(norm_map > threshold)
total_pixels = norm_map.size
percentage_change = (changed_pixels / total_pixels) * 100

print(f"\nPercentage Area Changed: {percentage_change:.2f}%\n")



frames = [
    before_raw,
    after_raw,
    Image.fromarray(overlay),
    Image.fromarray(heat_intensity),
    Image.fromarray(multi_overlay)
]

frames[0].save(
    "change_animation.gif",
    save_all=True,
    append_images=frames[1:],
    duration=1000,
    loop=0
)

print("GIF saved as change_animation.gif")


plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(before_raw)
plt.title("Before Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(after_raw)
plt.title("After Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(norm_map, cmap="Reds")
plt.title("Heatmap (14×14)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(overlay)
plt.title("Red Overlay Change Map")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(multi_overlay)
plt.title("Multi-Class Change Map")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.text(0.2, 0.5, f"Percentage Change:\n{percentage_change:.2f}%", fontsize=18)
plt.axis("off")

plt.tight_layout()
plt.show()
