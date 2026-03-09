import torch
import time
import torch.nn as nn
import accelerate


if __name__ == "__main__":
    model = nn.Linear(512, 512)
    for p in model.parameters():
        p.requires_grad = False
    accelerator = accelerate.Accelerator()
    model = accelerator.prepare_model(model)
    model.to(accelerator.device)
    data = torch.randn(1024, 512).to(accelerator.device)
    while True:
        time.sleep(0.01)
        accelerator.wait_for_everyone()
        if torch.cuda.utilization() < 1.5:
            with torch.no_grad():
                model(data)
        else:
            time.sleep(1)
        # print(f"rank:{accelerator.process_index}->usage:{torch.cuda.utilization()}")