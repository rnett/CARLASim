import random
import time
import traceback

from simulate import SimulateArgs, simulate, FramesMismatchError


def do_sim(args: SimulateArgs):
    time.sleep(random.randint(1, 5))
    while True:
        try:
            simulate(args)
            time.sleep(10)
            break
        except FileExistsError:
            break
            pass
        except FramesMismatchError as fme:
            # raise fme
            break
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue