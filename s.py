import torch
from resnet import ResNet18

# مسیر مدل اصلی
model_path = "C:/Users/Basabr/edl-prospect-certainty/results/model_uncertainty_mse.pt"
# ساخت مدل با همان ساختار
model = ResNet18(num_classes=10, dropout=False)
state_dict = torch.load(model_path)

# ساخت دیکشنری با کلید موردنیاز
checkpoint = {"model_state_dict": state_dict}

# ذخیره با نام موردنظر
torch.save(checkpoint, "C:/Users/Basabr/edl-prospect-certainty/model_uncertainty_mse.pt")

print("✅ مدل با کلید 'model_state_dict' ذخیره شد.")
