import time
import traceback

from simulate import SimulateArgs, simulate, FramesMismatchError


def do_sim(args: SimulateArgs):
    while True:
        try:
            simulate(args)
            time.sleep(10)
            break
        except FileExistsError:
            break
            pass
        except FramesMismatchError as fme:
            print(fme)
            traceback.print_exc()
            break
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue