from Environment import ONOSEnv
from utils import setup_exp, setup_run
import time

if __name__ == "__main__":
    setup_exp()
    folder = setup_run()
    env = ONOSEnv(folder)
    time.sleep(10)
    while True:
        time.sleep(5)
        env.update_intent_load()
