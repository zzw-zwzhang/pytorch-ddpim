import torch


def DDIM(img, t, t_next, model, alphas_cump):
    et = model(img, t)

    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * img
                                - 1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)
    img_next = img + x_delta

    return img_next


def DDPM(img, t, t_next, model, alphas_cump):
    et = model(img, t)

    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x0_from_e = (1.0 / at).sqrt() * img - (1.0 / at - 1).sqrt() * et
    x0_from_e = torch.clamp(x0_from_e, -1, 1)

    beta_t = 1 - at / at_next
    mean_eps = ((at_next.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - at_next)) * img) / (1.0 - at)

    mean = mean_eps
    noise = torch.randn_like(img)
    mask = 1 - (t == 0).float()
    mask = mask.view(-1, 1, 1, 1)
    logvar = beta_t.log()
    img_next = mean + mask * torch.exp(0.5 * logvar) * noise

    return img_next