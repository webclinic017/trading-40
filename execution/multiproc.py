import itertools
import multiprocessing


def foo(x):
    return x*x


def bar(x, y):
    return x*y


def main_v1():
    pool = multiprocessing.Pool(processes=4)

    out1 = pool.map(foo, range(10))
    print(out1)

    out2 = pool.starmap(bar, (itertools.product(range(2), range(2))))
    print(out2)


def baz(x, y):
    return (x, y), x*y


def main():
    with multiprocessing.Pool(processes=4) as pool:
        res = pool.starmap(baz, (itertools.product(range(2), range(2))))
    print(f"res = {res}")

    out = dict(res)
    print(f"out = {out}")


if __name__ == "__main__":
    main()
