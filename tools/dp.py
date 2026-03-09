import matplotlib.pyplot as plt
print(len([0, 3, 6, 9, 12, 16, 20, 24, 28, 33, 38, 43, 48, 53, 57, 62, 67, 72, 78, 83, 87, 91, 95, 98, 102, 106, 110, 115, 120, 125, 130, 135, 141, 146, 152, 158, 164, 171, 179, 185, 191, 197, 203, 209, 216, 223, 229, 234, 240, 245, 250]))
print(len(list(range(0, 251, 5))))
exit()
plt.plot()
plt.plot()
plt.show()
exit()




import torch

num_steps = 10
num_recompute_timesteps = 4
sim = torch.randint(0, 100, (num_steps, num_steps))
sim[:5, :5] = 100
for i in range(num_steps):
    sim[i, i] = 100

error_map = (100-sim).tolist()


# init
for i in range(1, num_steps):
    for j in range(0, i):
        error_map[i][j] = error_map[i-1][j] + error_map[i][j]

C = [[0, ] * (num_steps + 1) for _ in range(num_recompute_timesteps+1)]
P = [[-1, ] * (num_steps + 1) for _ in range(num_recompute_timesteps+1)]

for i in range(1, num_steps+1):
    C[1][i] = error_map[i-1][0]
    P[1][i] = 0


# dp
for step in range(2, num_recompute_timesteps+1):
    for i in range(step, num_steps+1):
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
tracback_end_index = num_steps
# min_value = 99999
# for i in range(num_recompute_timesteps-1, num_steps):
#     if C[-1][i] < min_value:
#         min_value = C[-1][i]
#         tracback_end_index = i

timesteps = [tracback_end_index, ]
for i in range(num_recompute_timesteps, 0, -1):
    idx = timesteps[-1]
    timesteps.append(P[i][idx])
timesteps.reverse()
print(timesteps)