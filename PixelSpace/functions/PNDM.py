import torch

def sample_image(model,  x, seq, alphas_cump):
    schedule = Schedule(alphas_cump)
    with torch.no_grad():
        imgs = [x]
        seq_next = [-1] + list(seq[:-1])
        start = True
        n = x.shape[0]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            t_next = (torch.ones(n) * j).to(x.device)
            img_t = imgs[-1].to(x.device)
            img_next = schedule.denoising_step(img_t, t_next, t, model, start)
            start = False
            imgs.append(img_next.to('cpu'))

        img = imgs[-1]
        return img



class Schedule(object):
    def __init__(self, alphas_cump):
    
        # betas, alphas_cump = get_schedule(args, config)

        # self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        # self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        # self.total_step = config['diffusion_step']

        self.method = gen_order_4 # f-pdnm
        self.alphas_cump = alphas_cump
        self.ets = None

    def denoising_step(self, img_n, t_end, t_start, model, first_step=False):
        if first_step:
            self.ets = []
        img_next = self.method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)
        return img_next
    

def gen_order_4(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t+t_next)/2, t_next]
    if len(ets) > 2:
        noise_ = model(img, t,sample= True)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = runge_kutta(img, t_list, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def runge_kutta(x, t_list, model, alphas_cump, ets):
    e_1 = model(x, t_list[0],sample=True)
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(x_2, t_list[1],sample=True)
    x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(x_3, t_list[1],sample=True)
    x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(x_4, t_list[2],sample=True)
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et

def transfer(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

    x_next = x + x_delta
    return x_next
