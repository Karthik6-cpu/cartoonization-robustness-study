import lpips
import torch
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()

t_lpips = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def compute_lpips(img1_pil, img2_pil):
    t1 = t_lpips(img1_pil.convert('RGB')).unsqueeze(0).to(device)
    t2 = t_lpips(img2_pil.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = loss_fn_vgg(t1, t2).item()
    return dist
