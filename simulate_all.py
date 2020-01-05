#!/usr/bin/env python
import time

import numpy as np
from tqdm import tqdm

from config import City, Rain
from simulate import simulate, FramesMismatchError

sims = []


def num_cars(city: City):
    if city is City.Town01:
        return 50
    elif city is City.Town02:
        return 20
    elif city is City.Town03:
        return 20
    elif city is City.Town04:
        return 20
    elif city is City.Town05:
        return 20
    else:
        raise ValueError


for city in list(City):
    for rain in list(Rain):

        if rain == Rain.Clear:
            for i in range(5):
                sims.append((city, rain, False, i))

            sims.append((city, rain, True, 0))
            sims.append((city, rain, True, 1))
        else:
            sims.append((city, rain, False, 0))
            sims.append((city, rain, False, 1))

            sims.append((city, rain, True, 0))

for sim in tqdm(sims, desc="Simulations", unit='sim'):
    while True:
        try:
            simulate(sim[0], num_cars(sim[0]), 200, sim[1], sim[2], seed=np.random.randint(0, 100000), overwrite=False, index=sim[3])
            time.sleep(10)
            break
        except FileExistsError:
            break
            pass
        except FramesMismatchError as fme:
            raise fme
        except Exception as e:
            continue
