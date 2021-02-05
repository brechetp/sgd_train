import time
import sys
import random

if __name__ == '__main__':

    time.sleep(5)

    r = random.randint(0,1)

    if r:
        print('fail!')
        sys.exit(1)
    else:
        print('success!')
        sys.exit(0)
