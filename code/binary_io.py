import numpy as np

# ==============================================================================
# Helper Functions
# ==============================================================================

def to_bit_string(num, num_bits):
    
    if num >= 2**num_bits:
        raise Exception("The number {} (>= {}) is bigger than what we can encode!".format(num, 2**num_bits))
        
    bitcode = []
    
    for i in range(num_bits):
        bitcode.append(str(num % 2))
        
        num //= 2
        
    return ''.join(bitcode)

def from_bit_string(bitcode):
    
    num = 0
    
    if isinstance(bitcode, bytes):
        bitcode = bitcode.decode("utf-8") 
    
    for i in range(len(bitcode)):
        
        if bitcode[i] == "1":
            num += 2**i
        
    return num

def write_bin_code(code, path, extras=None, extra_var_bits=None, var_length_extras=None, var_length_bits=None):
    
    if var_length_extras is not None:
        if var_length_bits is None or len(var_length_extras) != len(var_length_bits):
            raise Exception("Each var length extra needs to have a bitlength associated!")
            
    # Pad the code
    code += "0" * (8 - len(code) % 8) if len(code) % 8 != 0 else ""

    message_bytes = [int('0b' + code[s:s + 8], 2) for s in range(0, len(code), 8)]

    with open(path, "wb") as compressed_file:
        
        if extras is not None:
            for extra in extras:
                compressed_file.write(bytes([extra // 256, extra % 256]))
        
        if extra_var_bits is not None:
            for bits in extra_var_bits:
                # First code the length of the extra on 16 bits
                compressed_file.write(bytes([len(bits) // 256, len(bits) % 256]))
                
                # Pad the code
                bits += "0" * (8 - len(bits) % 8) if len(bits) % 8 != 0 else ""

                bits_message_bytes = [int('0b' + ''.join(bits[s:s + 8]), 2) for s in range(0, len(bits), 8)]
                
                compressed_file.write(bytes(bits_message_bytes))
        
        if var_length_extras is not None:
            for extra, extra_bit_size in zip(var_length_extras, var_length_bits):
                
                # First code the length of the extra on 16 bits
                compressed_file.write(bytes([len(extra) // 256, len(extra) % 256]))
                
                # Code the number of bits to code per item in 8 bits
                compressed_file.write(bytes([extra_bit_size]))
                
                extra_code = []
                
                # Write the rest on var_length_bytes bytes
                for item in extra:
                    
                    extra_code.append(to_bit_string(item, extra_bit_size))
                    
                extra_code = ''.join(extra_code)
                
                # Pad the code
                extra_code += "0" * (8 - len(extra_code) % 8) if len(extra_code) % 8 != 0 else ""

                extra_message_bytes = [int('0b' + extra_code[s:s + 8], 2) for s in range(0, len(extra_code), 8)]
                
                compressed_file.write(bytes(extra_message_bytes))
                
        
        compressed_file.write(bytes(message_bytes))


def read_bin_code(path, num_extras=0, num_extra_var_bits=0, num_var_length_extras=0, extra_bytes=2):

    with open(path, "rb") as compressed_file:
        compressed = ''.join(["{:08b}".format(x) for x in compressed_file.read()])

    extra_bits = compressed[:num_extras * extra_bytes * 8]
    compressed = compressed[num_extras * extra_bytes * 8:]
    
    extras = [int('0b' + extra_bits[s:s + extra_bytes * 8], 2) for s in range(0, 
                                                                              num_extras * extra_bytes * 8, 
                                                                              extra_bytes * 8)]
    extra_var_bits = []
    
    for i in range(num_extra_var_bits):
        
        # Read length of current extra
        bits_length = int('0b' + compressed[:extra_bytes * 8], 2)
        
        # Chop off the length information
        compressed = compressed[extra_bytes * 8:]
        
        # Calculate how many bytes to read
        bytes_to_read = bits_length // 8
        
        # Check if there was any padding
        if bits_length % 8 != 0:
            bytes_to_read += 1
        
        bits = compressed[:bits_length]
        
        compressed = compressed[bytes_to_read * 8:]
        
        extra_var_bits.append(bits)
        
        
    var_length_extras = []
    
    for i in range(num_var_length_extras):
        
        # Read length of current extra
        extra_length = int('0b' + compressed[:extra_bytes * 8], 2)
        
        # Read the number of bits used to code each item in the extra
        extra_bit_size = int('0b' + compressed[extra_bytes * 8:(extra_bytes + 1) * 8], 2)
        
        # Chop off the length information
        compressed = compressed[(extra_bytes + 1) * 8:]
        
        # Calculate how many bytes to read
        bytes_to_read = extra_bit_size * extra_length // 8
        
        # Check if there was any padding
        if extra_bit_size * extra_length % 8 != 0:
            bytes_to_read += 1
        
        extra = [from_bit_string(compressed[s:s + extra_bit_size])
                 for s in range(0, bytes_to_read * 8, extra_bit_size)]
        
        compressed = compressed[bytes_to_read * 8:]
        
        var_length_extras.append(extra[:extra_length])

    return compressed, extras, extra_var_bits, var_length_extras