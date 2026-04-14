import torch

states = torch.load("./output/state.pt", map_location="cpu").to(dtype=torch.float32)
states = states.permute(1, 2, 0, 3)
print(states.shape)
states = states.view(-1, 49, 1152)
states = torch.nn.functional.normalize(states, dim=-1)
sim = torch.bmm(states, states.transpose(1, 2))
mean_sim = torch.mean(sim, dim=0, keepdim=False)

mean_sim = mean_sim.numpy()
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
timesteps = np.linspace(0, 1, 5)
# plt.rc('axes.spines', **{'bottom':False, 'left':False, 'right':False, 'top':False})
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#7400b8","#5e60ce","#4ea8de", "#64dfdf", "#80ffdb"])
plt.imshow(mean_sim, cmap="inferno")
plt.xticks([])
plt.yticks([])
# plt.show()
plt.colorbar()
plt.savefig("./output/mean_sim.png", pad_inches=0, bbox_inches="tight")
# cos_sim = torch.nn.functional.cosine_similarity(states, states)


# for i in range(49):
#     cos_sim = torch.nn.functional.cosine_similarity(states[i], states[i + 1])
#     cos_sim = cos_sim.min()
#     print(cos_sim)
# state = torch.max(states, dim=-1)[1]
# # state = torch.softmax(state, dim=-1)
# state = state.view(-1, 16, 16)
#
# state = state.numpy()
#
# import numpy as np
# import matplotlib.pyplot as plt
# for i in range(0, 49):
#     print(i)
#     plt.imshow(state[i])
#     plt.savefig("./output2/{}.png".format(i))