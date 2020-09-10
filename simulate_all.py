#!/usr/bin/env python
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from config import City, Rain
from multiprocess_helper import do_sim
from simulate import SimulateArgs

sims = []

NON_CLEAR = False

def num_cars(city: City):
    if city is City.Town01:
        return 30
    elif city is City.Town02:
        return 10
    elif city is City.Town03:
        return 40
    elif city is City.Town04:
        return 30
    elif city is City.Town05:
        return 40
    else:
        raise ValueError


"""
per City (5) (25 per, 125 total):
    Clear (7 per, 7 total):
        noon: 5
        sunset: 2
    Others (6) (3 per, 18 total):
        noon: 2
        sunset: 1
"""
port = 2000
for city in list(City):
    for rain in list(Rain):

        if rain == Rain.Clear:
            for i in range(5):
                sims.append((city, rain, False, i, port))
                port += 2

            # sims.append((city, rain, True, 0, port))
            # port += 2
            # sims.append((city, rain, True, 1, port))
            # port += 2
        elif NON_CLEAR:
            sims.append((city, rain, False, 0, port))
            port += 2
            sims.append((city, rain, False, 1, port))
            port += 2

            sims.append((city, rain, True, 0, port))
            port += 2

sims = [SimulateArgs(sim[0], num_cars(sim[0]), 200, sim[1], sim[2], seed=np.random.randint(0, 100000), overwrite=False,
                     index=sim[3], port=sim[4]) for sim in sims]

# pool = Pool(processes=2)

#pool.imap
for _ in tqdm(map(do_sim, sims), total=len(sims), desc="Simulations"):
    pass
