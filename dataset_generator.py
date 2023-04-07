from random import randint, gauss, randrange
from sys import argv
#import matplotlib.pyplot as plt

if __name__ == "__main__":
    NUM_CENTERS = int(argv[1])
    NUM_POINTS = int(argv[2])
    x = []
    y = []
    for c in range(NUM_CENTERS):
        cx = randint(-1000, 1000)
        cy = randint(-1000, 1000)
        s = randrange(1, 10)
        for p in range(NUM_POINTS):
            coord = []
            #for d in range(2):
                #coord.append(gauss(cx,s))
            x.append(gauss(cx, s))
            y.append(gauss(cy, s))
    #plt.scatter(x, y)
    for px, py in zip(x, y):
        print(f"{px},{py}")

