"""Quick shape/gradient smoke test for CAFR-Net."""
import torch
from models.cafr_net import CAFRNet, paired_consistency_loss

def main():
    model = CAFRNet(num_classes=10, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    output = model(x, return_features=True)
    assert output["logits"].shape == (2, 10)
    assert output["embedding"].shape == (2, 256)
    assert output["gates"].shape == (2, 4)
    assert torch.allclose(output["gates"].sum(1), torch.ones(2), atol=1e-5)
    loss = output["logits"].square().mean() + paired_consistency_loss(
        output["logits"], model(x)
    )
    loss.backward()
    print("CAFR-Net smoke test passed.")

if __name__ == "__main__":
    main()
