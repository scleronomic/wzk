import msgpack


def read_msgpack(file):
    with open(file, 'rb') as f:
        b = f.read()
    return msgpack.unpackb(b)


def write_msgpack(file, nested_list):
    arr_bin = msgpack.packb(nested_list, use_bin_type=True)
    with open(file, 'wb') as f:
        f.write(arr_bin)
