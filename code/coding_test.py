import time

import numpy as np

from coding import ArithmeticCoder, write_bin_code, read_bin_code


num_symbols = 2**4
message_length = 200000
test_file_path = "scratch_compression_test.miracle"

P = np.ones(num_symbols + 1, dtype=np.int32)
P[1:] = np.random.choice(1000, size=num_symbols) + 1

message = np.zeros(message_length, dtype=np.int32)

message[:-1] = np.random.choice(num_symbols, size=message_length - 1) + 1

ac = ArithmeticCoder(P, precision=32)

start = time.time()

print("Coding..")
code = ac.encode(message)

print("Coded in {:.4f}s".format(time.time() - start))

write_bin_code(''.join(code), test_file_path, [152, 42069, 1231, 6272])

code, extras, _ = read_bin_code(test_file_path, 4)

print(extras)
# start = time.time()

# print("Decoding...")
# decompressed = ac.decode(code)
# print("Decoded in {:.4f}s".format(time.time() - start))


# print(np.all(decompressed == message))

start = time.time()

print("Fast Decoding...")
decompressed = ac.decode_fast(code)
print("Decoded in {:.4f}s".format(time.time() - start))


print(np.all(decompressed == message))
