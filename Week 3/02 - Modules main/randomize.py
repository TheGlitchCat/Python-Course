import random as rn


def rand():
    return rn.random()


def rand10():
    return rn.randint(0,10)


def rand100():
    return rn.randrange(0,100)


def choice(data):
    return rn.choice(data)