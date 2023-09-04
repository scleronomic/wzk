import time
import datetime
import subprocess

PxWxD = "SnapeSev44"  # TODO set in environment variable


def check(in_or_out):
    assert in_or_out in ["in", "out"]
    cmd = f"/home/seth_da/usr/bin/zerf --check-{in_or_out}"
    cmd = f"yes {PxWxD} | {cmd}"
    subprocess.call(cmd, shell=True)

    print(f"{datetime.datetime.now()}: checked {in_or_out}")


def main():
    count = 0
    max_count = 5

    checked_in = False
    while True:

        now = datetime.datetime.now()
        if now.weekday() < 5:

            if 8 <= now.hour <= 12:
                if not checked_in:
                    check("in")
                    checked_in = True

            if 18 <= now.hour <= 19:
                if checked_in:
                    check("out")
                    checked_in = False
                    count += 1
                    print(f"{now}: logged ")

            if count >= max_count:
                break

        print("sleep")
        time.sleep(600)


if __name__ == "__main__":
    main()
