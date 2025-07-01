import matplotlib.pyplot as plt

for i in range(7):
    plt.plot([i, i + 1], label=f'Line {i+1}')

plt.legend()
plt.show()