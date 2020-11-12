#!/usr/bin/env python

import itertools

items = {0: [[19, 53, 12]], 2: [[16, 12, 17]], 4: [12], 12: [[2, 43, 21], 1], 20: [11, 10]}

def combinate(items, size=5):
    if size > len(items):
        raise Exception("Lower the `size` or add more products, dude!")

    for cats in itertools.combinations(items.keys(), size):
        cat_items = [[products for products in items[cat]] for cat in cats]
        for x in itertools.product(*cat_items):
            yield zip(cats, x)

if __name__ == '__main__':
    for x in combinate(items):
        for y in x:
            a, b = y
            print(a)
            print(b)
