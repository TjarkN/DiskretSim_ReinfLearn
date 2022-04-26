import numpy as np
import random
from time import sleep

def simulation(qCars = 20, streetLength= 40, prob= 0.5, maxV= 40):
    position = random.sample(range(0, streetLength), qCars)
    print(np.sort(position))
    cars = [[p, 0] for p in np.sort(position)]
    t = 0

    while t < maxV:

        # print out the cars & enumerate while condition
        print(cars)
        sleep(1)
        t += 1

        #
        for i in range(len(cars)):
            cars[i][1] = np.min([cars[i][1]+1, 4])

            if cars[(i+1) % qCars][0] < cars [i][0]:
              gap = streetLength - cars[i][0] + cars[(i+1) % qCars][0] -1
            else:
                gap = cars [(i+1) % qCars][0] - cars[i][0] -1
            cars[i][1] = np.min([cars[i][1], gap])

            if cars[i][1] >  0 and random.uniform(0,1) < prob:
                cars[i][0] -= 1

        for i in range(len(cars)):
            cars[i][0] = (cars[i][0] + cars[i][1]) % streetLength


if __name__ == "__main__":
        simulation()