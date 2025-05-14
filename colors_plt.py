import matplotlib.pyplot as plt

for i in range(7):
    plt.plot([i, i + 1], label=f'Line {i}')

plt.legend()
plt.show()