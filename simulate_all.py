#!/usr/bin/env python
import numpy as np
from tqdm import tqdm

from config import City, Rain
from simulate import simulate

np.random.seed(645)

sims = []

for city in list(City):
    for rain in list(Rain):

        if rain == Rain.Clear:
            for i in range(5):
                sims.append((city, rain, False, np.random.randint(0, 100000), i))

            sims.append((city, rain, True, np.random.randint(0, 100000), 0))
            sims.append((city, rain, True, np.random.randint(0, 100000), 1))
        else:
            sims.append((city, rain, False, np.random.randint(0, 100000), 0))
            sims.append((city, rain, False, np.random.randint(0, 100000), 1))

            sims.append((city, rain, True, np.random.randint(0, 100000), 0))

for sim in tqdm(sims, desc="Simulations", unit='sim'):
    try:
        simulate(sim[0], 50, 200, sim[1], sim[2], seed=sim[3], overwrite=False, index=sim[4])
    except FileExistsError:
        pass
