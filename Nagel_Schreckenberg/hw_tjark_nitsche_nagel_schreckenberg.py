import numpy as np
import random
import copy
import matplotlib.pyplot as plt

"""Das ist mein Versuch am Nagel-Schreckenberg-Modell. 
Es ist mir leider nicht gelungen, die Crashes zu verhindern, auch, nachdem ich mich mit Kommolitonen drangesetzt habe"""


class car:
    """Klasse Auto: Enthält die Nummer, Geschwindigkeit und Position des Autos"""

    def __init__(self, number, velocity, position):
        self.number = number
        self.velocity = velocity
        self.position = position


def simulation(quan_cars=20, street_len=40, prob=0.3, rounds=40, max_vel=5):
    """Simulationsmethode für das Nagel-Schreckenberg Modell"""
    cars = []
    positions = np.sort(random.sample(range(0, street_len - 1), quan_cars))
    c_history = [0] * rounds

    # Erstellung des 'cars' Array, der die Autos enthält
    for i in range(quan_cars):
        cars.append(car(i, 0, positions[i]))
        print(cars[i].position, cars[i].velocity)

    print(cars)

    # Erste for-Schleife. Richtet sich nach der Anzahl der Simulations-"Runden"
    for i in range(0, rounds):
        output = ""
        # Durchlaufen des 'cars'-Array
        for e, act_car in enumerate(cars):
            # Erhöhung der Geschwindigkeit
            if act_car.velocity != max_vel:
                act_car.velocity += 1

            # Ermittlung der Distanz zum vorherfahrenden Auto
            if act_car.velocity + act_car.position > cars[(e + 1) % (quan_cars - 1)].position:
                dist = street_len - 1 - act_car.position + cars[(e + 1) % (quan_cars - 1)].position
            else:
                dist = cars[(e + 1) % (quan_cars - 1)].position - act_car.position

            # Abbremsen auf die Distanz-1 zum Vordermann
            if act_car.velocity >= dist and act_car.velocity > 0:
                act_car.velocity = dist - 1

            # Trödeln
            if act_car.velocity > 0 and random.random() < prob:
                act_car.velocity -= 1

            # Position durch das Aufaddieren der Geschwindigkeit ändern
            act_car.position = (act_car.position + act_car.velocity) % street_len
            # Ausdruck des aktuellen Autos
            output = output + "[ " + str(act_car.position) + " , " + str(act_car.velocity) + " ]"
        print(output)
        c_history[i] = copy.deepcopy(cars)
    return c_history


if __name__ == "__main__":
    # Aufruf der Simulationsmethode
    rounds = 100
    street_len = 50
    quan_cars = 20
    c_history = simulation(quan_cars, street_len, 0.5, rounds, 5)

    # Erstellung der Werte für den Graphen
    cells_c = np.ones((rounds, street_len)) * -1
    cells = np.ones((rounds, street_len))
    for i in range(rounds):
        for e in range(quan_cars):
            cells_c[i, c_history[i][e].position] = c_history[i][e].velocity
            cells[i, c_history[i][e].position] = 0

    # Erstellung des Matplotlib-Graphen
    plt.figure(1)
    im = plt.imshow(cells_c, cmap='Greys')
    plt.xlabel('Road')
    plt.ylabel('Time')

    plt.show()
