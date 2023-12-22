def FOR1():
    return range(3)


def FOR2():
    return ((i, j) for i in FOR1() for j in FOR1())


def FOR3():
    return ((i, j, k) for i in FOR1() for j in FOR1() for k in FOR1())


def FOR4():
    return (
        (i, j, k, l) for i in FOR1() for j in FOR1() for k in FOR1() for l in FOR1()
    )
