import matplotlib.pyplot as plt

epoch = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# uas = [7.92, 9.44, 22.4, 38.14, 43.47, 53.4, 59.27, 60.34, 62.59, 63.33, 64.45, 63.81, 65.62, 64.89, 65.62, 66.65, 66.06, 67.04, 64.16, 66.06]
# las = [3.08, 8.31, 18.19, 29.98, 35.06, 44.45, 48.7, 50.22, 50.86, 53.55, 54.28, 52.76, 54.96, 54.18, 54.47, 56.58, 55.65, 56.58, 54.57, 55.35]

uas = [13.01, 43.52, 63.18, 70.32, 73.59, 75.65, 74.57, 76.38, 76.87, 78.04, 77.6, 78.34, 77.75, 76.43, 77.36, 75.89, 77.75, 77.56, 78.14, 77.11]
las = [9.58, 39.61, 58.73, 65.43, 68.85, 71.15, 69.54, 72.27, 71.78, 73.55, 72.37, 73.5, 72.81, 71.49, 73.11, 70.32, 72.52, 72.81, 73.55, 72.27]

plt.plot(epoch, uas, label="UAS")
plt.plot(epoch, las, label="LAS")

plt.xlabel('epoch')
plt.xticks(range(0, 21, 5))
plt.ylabel('UAS / LAS')
plt.yticks(range(0, 101, 20))
plt.legend()
plt.title('UAS and LAS with POS embedding')
plt.grid(True)
plt.savefig("pos_accuracy.png")
plt.show()
