import torch
import torchvision.transforms as transforms

def get_transforms(res):
    return transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])

def encode_pixels(vae, x):  # raw pixels => raw latents
    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    return x

def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z

def get_f(model, vae, image, res, device, label_id, layersout, time, baseline=False, legacy=True):
    """
    image:
      - single PIL.Image
      - OR list[PIL.Image]
    label_id:
      - int
      - OR list[int]  (len must match image list)
    return:
      f = model(...)[1]
    """
    transform = get_transforms(res)

    # ---- make batch ----
    if isinstance(image, (list, tuple)):
        assert isinstance(label_id, (list, tuple)), "If image is list, label_id must be list"
        assert len(image) == len(label_id), "image list and label_id list must have same length"
        # stack -> (B,3,res,res)
        image = torch.stack([transform(im) for im in image], dim=0).to(device)
        labels = torch.tensor(label_id, device=device, dtype=torch.long)  # (B,)
        B = image.shape[0]
        time_vec = torch.full((B,), float(time), device=device, dtype=image.dtype)  # (B,)
    else:
        image = transform(image).unsqueeze(0).to(device)  # (1,3,res,res)
        labels = torch.tensor([int(label_id)], device=device, dtype=torch.long)     # (1,)
        time_vec = torch.tensor([float(time)], device=device, dtype=image.dtype)   # (1,)

    # ---- constants (broadcastable) ----
    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215],
                                 device=device, dtype=image.dtype).view(1, 4, 1, 1)
    latents_bias = torch.tensor([0., 0., 0., 0.],
                                device=device, dtype=image.dtype).view(1, 4, 1, 1)

    with torch.no_grad():
        if baseline:
            f = encode_pixels(vae, image)  # (B,4,h,w)
        else:
            # diffusers 推荐用 .latent_dist.mean/std 访问；你这里保持原写法
            moments = torch.cat(
                [vae.encode(image)["latent_dist"].mean, vae.encode(image)["latent_dist"].std],
                dim=1
            )  # (B,8,h,w)
            f = sample_posterior(moments, latents_scale, latents_bias)  # (B,4,h,w)

        noise = torch.randn_like(f)
        # (B,1,1,1) broadcast
        t_img = time_vec.view(-1, 1, 1, 1).to(f.dtype)
        f = (1 - t_img) * f + t_img * noise

        # model forward (batch)
        if baseline:
            if legacy:
                out = model(f, time_vec, labels, layersout)
            else:
                out = model(f, (1 - time_vec), labels, layersout)
        else:
            out = model(f, time_vec, labels, layersout)

        f = out[1]
    # f = f.mean(dim=1)  # (B, C)
    return f
