import torch

from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *

from typing import Callable


def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt


import logging
logger = logging.getLogger(__name__)

class EulerSampler(BaseSampler):
    def __init__(
            self,
            w_scheduler: BaseScheduler = None,
            timeshift=1.0,
            guidance_interval_min: float = 0.0,
            guidance_interval_max: float = 1.0,
            state_refresh_rate=1,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler
        self.timeshift = timeshift
        self.state_refresh_rate = state_refresh_rate
        self.guidance_interval_min = guidance_interval_min
        self.guidance_interval_max = guidance_interval_max

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps

        timesteps = torch.linspace(0.0, 1 - self.last_step, self.num_steps)
        timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
        self.timesteps = shift_respace_fn(timesteps, self.timeshift)

        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

        # init recompute

        self.recompute_timesteps = list(range(self.num_steps))

    def sharing_dp(self, net, noise, condition, uncondition):
        _, C, H, W = noise.shape
        B = 8
        template_noise = torch.randn((B, C, H, W), generator=torch.Generator("cuda").manual_seed(0), device=noise.device)
        template_condition = torch.randint(0, 1000, (B,), generator=torch.Generator("cuda").manual_seed(0), device=condition.device)
        template_uncondition = torch.full((B, ), 1000, device=condition.device)
        _, state_list = self._impl_sampling(net, template_noise, template_condition, template_uncondition)
        states = torch.stack(state_list)
        N, B, L, C = states.shape
        states = states.view(N, B*L, C )
        states = states.permute(1, 0, 2)
        states = torch.nn.functional.normalize(states, dim=-1)
        with torch.autocast(device_type="cuda", dtype=torch.float64):
            sim = torch.bmm(states, states.transpose(1, 2))
        sim = torch.mean(sim, dim=0).cpu()
        error_map = (1-sim).tolist()

        # init cum-error
        for i in range(1, self.num_steps):
            for j in range(0, i):
                error_map[i][j] = error_map[i-1][j] + error_map[i][j]

        # init dp and force 0 start
        C = [[0.0, ] * (self.num_steps + 1) for _ in range(self.num_recompute_timesteps+1)]
        P = [[-1, ] * (self.num_steps + 1) for _ in range(self.num_recompute_timesteps+1)]
        for i in range(1, self.num_steps+1):
            C[1][i] = error_map[i - 1][0]
            P[1][i] = 0

        # dp state
        for step in range(2, self.num_recompute_timesteps+1):
            for i in range(step, self.num_steps+1):
                min_value = 99999
                min_index = -1
                for j in range(step-1, i):
                    value = C[step-1][j] + error_map[i-1][j]
                    if value < min_value:
                        min_value = value
                        min_index = j
                C[step][i] = min_value
                P[step][i] = min_index

        # trace back
        timesteps = [self.num_steps,]
        for i in range(self.num_recompute_timesteps, 0, -1):
            idx = timesteps[-1]
            timesteps.append(P[i][idx])
        timesteps.reverse()

        print("recompute timesteps solved by DP: ", timesteps)
        return timesteps[:-1]

    def _impl_sampling(self, net, noise, condition, uncondition):
        """
        sampling process of Euler sampler
        -
        """
        batch_size = noise.shape[0]
        steps = self.timesteps.to(noise.device)
        cfg_condition = torch.cat([uncondition, condition], dim=0)
        x = noise
        state = None
        pooled_state_list = []
        for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
            dt = t_next - t_cur
            t_cur = t_cur.repeat(batch_size)
            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            if i in self.recompute_timesteps:
                state = None
            out, state = net(cfg_x, cfg_t, cfg_condition, state)
            if t_cur[0] > self.guidance_interval_min and t_cur[0] < self.guidance_interval_max:
                out = self.guidance_fn(out, self.guidance)
            else:
                out = self.guidance_fn(out, 1.0)
            v = out
            if i < self.num_steps -1 :
                x = self.step_fn(x, v, dt, s=0.0, w=0.0)
            else:
                x = self.last_step_fn(x, v, dt, s=0.0, w=0.0)
            pooled_state_list.append(state)
        return x, pooled_state_list

    def __call__(self, net, noise, condition, uncondition):
        self.num_recompute_timesteps = int(self.num_steps / self.state_refresh_rate)
        if len(self.recompute_timesteps) != self.num_recompute_timesteps:
            self.recompute_timesteps = self.sharing_dp(net, noise, condition, uncondition)
        denoised, _ = self._impl_sampling(net, noise, condition, uncondition)
        return denoised