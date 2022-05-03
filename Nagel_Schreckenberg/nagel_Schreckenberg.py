import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

from time import sleep

class car:
    def __init__(self, number, velocity, position):
        self.number = number
        self.velocity = velocity
        self.position = position

def simulation(quan_cars = 20, street_len= 40, prob= 0.3, rounds= 40, max_vel=5):

    cars = []
    positions = np.sort(random.sample(range(0, street_len-1), quan_cars))
    history = [0]*rounds

    for i in range(quan_cars):
        cars.append(car(i,0,positions[i]))
        print(cars[i].position,cars[i].velocity)

    print(cars)

    for i in range(0,rounds):
        output = ""
        for e,act_car in enumerate(cars):
            if act_car.velocity != max_vel:
                act_car.velocity += 1

            if act_car.velocity+act_car.position > cars[(e+1) % (quan_cars-1)].position:
                dist = street_len-1 - act_car.position + cars[(e + 1) % (quan_cars-1)].position
            else:
                dist = cars[(e+1) % (quan_cars-1)].position - act_car.position

            if act_car.velocity >= dist and act_car.velocity > 0:
                act_car.velocity = dist-1

            if act_car.velocity > 0 and random.random() < prob:
                act_car.velocity -= 1

            act_car.position = (act_car.position+act_car.velocity)%street_len
            output = output + "[ " + str(act_car.position) + " , " + str(act_car.velocity) + " ]"
        print(output)

        #for c in cars:
            #c.position = (c.position + c.velocity) % street_len
            #output = output + "[ " + str(c.position) + " , " + str(c.velocity) + " ]"
            #+ str(c.number) + " , "
        #print(output)
        history[i] = copy.deepcopy(cars)
    return history



if __name__ == "__main__":
        rounds = 50
        street_len = 30
        quan_cars = 10
        history = simulation(quan_cars, street_len, 0.5, rounds, 5)

        cells_c = np.ones((rounds,street_len))*-1
        cells = np.ones((rounds,street_len))
        for i in range(rounds):
            for e in range(quan_cars):
                cells_c[i,history[i][e].position]=history[i][e].velocity
                cells[i, history[i][e].position] = 0

        plt.figure(1)
        im = plt.imshow(cells_c, cmap='Greys')
        plt.xlabel('Road')
        plt.ylabel('Time')

        plt.show()

"""for i in range(0, rounds):
    street = []
    for n,c in enumerate(cars):
        if cars[n].position==qCars-1:
            if cars[n+1].position > c.position+c.velocity:
                c.position=c.position+c.velocity
            else:
                c.position = cars[n+1].position-1
        else:
            if cars[1].position > c.position+c.velocity-streetLength:
                c.position=c.position+c.velocity-streetLength
                #street.insert([c.position]) = c
            else:
                c.position = cars[n+1].position-1

        #street[c.position] = c.posit"""

# print(np.sort(position))
# cars = [[p, 0] for p in np.sort(position)]
# t = 0

"""while t < maxV:

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
        cars[i][0] = (cars[i][0] + cars[i][1]) % streetLength"""