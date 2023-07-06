import time
import datetime
import subprocess


def main():
    count = 0
    max_count = 5

    checked_in = False
    while True:

        now = datetime.datetime.now()
        if now.weekday() < 5:

            if 8 <= now.hour <= 10:
                if not checked_in:
                    subprocess.call("check-in", shell=True)

                    checked_in = True
                    print(f"{now}: logged in")

            if 17 <= now.hour <= 19:
                if checked_in:
                    subprocess.call("check-out", shell=True)
                    checked_in = False
                    count += 1
                    print(f"{now}: logged out")

            if count >= max_count:
                break

        time.sleep(3600)


if __name__ == "__main__":
    main()
