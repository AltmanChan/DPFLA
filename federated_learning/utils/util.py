def contains_class(dataset, source_class):
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == source_class:
            return True
    return False
